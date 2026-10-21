// Exercise reporting math on real model dimensions; this does not run the S1 engine.
#include "common.h"
#include "json.h"
#include "llama-ext.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: test-s1-reporting TINY_FIXTURE_DIRECTORY\n");
        return 2;
    }
    try {
        int checks = 0;
        auto check = [&](bool ok) {
            if (!ok) { throw std::runtime_error("S1 reporting check failed"); }
            ++checks;
        };
        auto refused = [&](auto operation, const std::string & message) {
            bool caught = false;
            try { operation(); }
            catch (const std::exception & error) { caught = std::string(error.what()).find(message) != std::string::npos; }
            check(caught);
        };
        common_params p;
        p.ridgefill_mode = "quality+qattn";
        p.ridgefill_qattn_share = 0.2f;
        p.ridgefill_s1_window = 64;
        p.ridgefill_vtail = true;
        p.ridgefill_vtail_rows = 32;
        p.ridgefill_alpha = 0;
        common_ridgefill_validate_sweep(p, 700, 500, 636, 128);
        ++checks;
        refused([&]() { common_ridgefill_validate_sweep(p, 700, 0, 636, 128); }, "nonempty filled prefix");
        refused([&]() { common_ridgefill_validate_sweep(p, 700, 500, 563, 128); }, "overlap the prefix");
        refused([&]() { common_ridgefill_validate_sweep(p, 700, 500, 701, 128); }, "within LOWER rows");
        refused([&]() { common_ridgefill_validate_sweep(p, 700, 500, 636, 31); }, "vtail first tail ubatch");
        p.ridgefill_header_rows = 1;
        refused([&]() { common_ridgefill_validate_options(p); }, "header-rows");
        p.ridgefill_header_rows = 0;
        p.ridgefill_qattn_share = 0;
        refused([&]() { common_ridgefill_validate_options(p); }, "positive --ridgefill-qattn-share");
        p.ridgefill_qattn_share = 0.2f;
        p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        refused([&]() { common_ridgefill_validate_options(p); }, "n_rs_seq > 0");
        p.speculative.types.clear();
        p.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        refused([&]() { common_ridgefill_validate_options(p); }, "non-causal contexts");
        p.attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED;
        p.ridgefill_qattn_blind = true;
        refused([&]() { common_ridgefill_validate_options(p); }, "S1 does not support");
        p.ridgefill_qattn_blind = false;
        p.ridgefill_mode = "speed";
        refused([&]() { common_ridgefill_validate_options(p); }, "two-sweep flow");
        p.ridgefill_mode = "quality+qattn";
#if defined(LLAMA_RIDGEFILL_API_VERSION) && LLAMA_RIDGEFILL_API_VERSION >= 2
        common_ridgefill_require_api_v2(p);
        ++checks;
#else
        refused([&]() { common_ridgefill_require_api_v2(p); }, "C API v2");
#endif
        p.ridgefill_mode = "off";
        common_ridgefill_require_api_v2(p);
        ++checks;

        const std::string fixture = argv[1];
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file((fixture + "/tiny-qwen35-f32.gguf").c_str(), mp), llama_model_free);
        if (!model) { throw std::runtime_error("cannot load tiny model"); }
        std::ifstream input(fixture + "/tokens.bin", std::ios::binary);
        std::vector<llama_token> tokens(700);
        if (!input.read(reinterpret_cast<char *>(tokens.data()), tokens.size() * sizeof(llama_token))) { throw std::runtime_error("need 700 fixture tokens"); }
        tokens[0] = 1; tokens[100] = 2;
        const std::vector<llama_pos> rows = {0, 7, 100, 127, 255, 499};
        common_json records = common_json::array();
        for (int split : {3, 4}) {
            std::unique_ptr<llama_ridgefill, decltype(&llama_ridgefill_free)> rf(llama_ridgefill_init(model.get(), (fixture + "/projector-s" + std::to_string(split) + "-f32.gguf").c_str()), llama_ridgefill_free);
            if (!rf) { throw std::runtime_error("cannot load projector"); }
            p = common_params{};
            p.ridgefill_mode = "quality+qattn";
            p.ridgefill_share = 0; p.ridgefill_digits = 0; p.ridgefill_alpha = 0;
            p.ridgefill_special_exact = true;
            auto base = common_ridgefill_get_report(model.get(), rf.get(), p, tokens.data(), 700, 500, rows, p.ridgefill_mode, 636, 700);
            common_ridgefill_sweep_report empty;
            auto defaults = common_ridgefill_get_report(model.get(), rf.get(), p, tokens.data(), 700, 500, rows, p.ridgefill_mode, 636, 700, 0, 500, &empty);
            check(base.to_json().dump() == defaults.to_json().dump());
            for (const auto & config : std::vector<std::vector<int>>{{16,700,0},{64,700,0},{128,700,0},{64,636,0},{0,700,32},{0,700,512},{64,700,32},{128,636,512}}) {
                p.ridgefill_s1_window = config[0];
                p.ridgefill_vtail = config[2] != 0;
                p.ridgefill_vtail_rows = config[2] ? config[2] : 512;
                common_ridgefill_sweep_report work;
                if (config[0]) {
                    work.s1_begin = config[1] - config[0]; work.s1_end = config[1];
                    work.s1_probe_kv_pairs = uint64_t(config[0]) * (config[1] == 700 ? 768 : 1024);
                }
                work.vtail_rows = p.ridgefill_vtail ? std::min(config[2], 200) : 0;
                auto report = common_ridgefill_get_report(model.get(), rf.get(), p, tokens.data(), 700, 500, rows,
                        p.ridgefill_mode, 636, 700, 0, 500, &work);
                check(report.n_exact_prefix == int(rows.size()));
                check(report.exact_by_source[2] == 2 && report.exact_by_source[config[0] ? 5 : 4] == 4);
                check(report.macs_ridgefill == base.macs_ridgefill + report.macs_s1_fill + report.macs_s1_probe + report.macs_vtail);
                refused([&]() { common_ridgefill_get_report(model.get(), rf.get(), p, tokens.data(), 700, 500, rows, p.ridgefill_mode, 636, 700); }, "completed sweep metadata");
                if (config[0]) {
                    auto incomplete = work;
                    incomplete.s1_probe_kv_pairs = 0;
                    refused([&]() { common_ridgefill_get_report(model.get(), rf.get(), p, tokens.data(), 700, 500, rows,
                            p.ridgefill_mode, 636, 700, 0, 500, &incomplete); }, "dense query/key pair count");
                }
                records.push_back(common_json {{"split", split}, {"window", config[0]}, {"end", config[1]},
                    {"vtail_rows", work.vtail_rows}, {"kv_pairs", work.s1_probe_kv_pairs}, {"rows", rows}, {"base", base.to_json()}, {"report", report.to_json()}});
            }
        }
        std::cout << common_json {{"result", "PASS"}, {"scope", "reporting math only; S1 engine not exercised"}, {"checks", checks}, {"records", records}}.dump() << '\n';
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
