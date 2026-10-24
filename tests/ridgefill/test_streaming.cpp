#include "llama.h"
#include "llama-ext.h"
#include "llama-context.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "server-ridgefill-sweep.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static bool same_logits(const std::vector<float> & a, const std::vector<float> & b) {
    return !a.empty() && a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

struct snapshot {
    std::vector<uint8_t> prefill, final;
    std::vector<float> logits, scales;
    std::vector<double> mass;
    std::vector<llama_pos> rows;
    size_t map_bytes = 0;
};

static std::vector<uint8_t> state(llama_context * ctx, llama_seq_id seq = 0) {
    std::vector<uint8_t> bytes(llama_state_seq_get_size(ctx, seq));
    require(!bytes.empty(), "empty sequence state");
    require(llama_state_seq_get_data(ctx, bytes.data(), bytes.size(), seq) == bytes.size(), "incomplete sequence state");
    return bytes;
}

static int check_replay_lookup(const llama_model & model) {
    auto hp = model.hparams;
    hp.rope_type = LLAMA_ROPE_TYPE_NORM;
    llama_kv_cache cache(model, hp, GGML_TYPE_F32, GGML_TYPE_F32, false, false, true,
            64, 2, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, [](uint32_t) { return false; }, nullptr, nullptr);
    std::vector<llama_pos> positions(32);
    std::vector<int32_t> counts(32, 1);
    llama_seq_id owner = 0, wrong = 1;
    std::vector<llama_seq_id *> owners(32, &owner);
    for (int i = 0; i < 32; ++i) { positions[i] = i; }
    llama_ubatch batch = {};
    batch.n_tokens = batch.n_seq_tokens = 32; batch.n_seqs = batch.n_seqs_unq = batch.n_pos = 1;
    batch.pos = positions.data(); batch.n_seq_id = counts.data(); batch.seq_id = owners.data(); batch.seq_id_unq = &owner;
    int checks = 0;
    auto compare = [&]() {
        std::vector<uint32_t> expected;
        const auto & cells = cache.get_cells(owner);
        for (uint32_t q = 0; q < batch.n_tokens; ++q) {
            int64_t found = -1;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.is_empty(i) && cells.seq_has(i, owner) && cells.pos_get(i) == positions[q]) { found = i; }
            }
            if (counts[q] != 1 || *owners[q] != owner || found < 0 || cells.seq_count(found) != 1) { expected.clear(); break; }
            expected.push_back(found);
        }
        const auto actual = cache.find_slot_replay(batch);
        require(expected.empty() ? actual.empty() : !actual.empty() && actual.strm == std::vector<llama_seq_id>{0} && actual.idxs[0] == expected,
                "replay lookup differs from exhaustive cell lookup");
        ++checks;
    };
    compare();
    const auto slot = cache.find_slot(batch, false);
    require(!slot.empty(), "cannot seed replay lookup fixture");
    cache.apply_ubatch(slot, batch);
    compare();
    batch.n_tokens = batch.n_seq_tokens = 5;
    const auto queries = {0, 8, 4, 8, 31};
    std::copy(queries.begin(), queries.end(), positions.begin());
    compare();
    positions[0] = 40; compare();
    positions[0] = -1; compare();
    positions[0] = 0; owners[1] = &wrong; compare(); owners[1] = &owner;
    cache.seq_cp(0, 1, 8, 9); compare();
    cache.seq_rm(1, -1, -1);
    cache.seq_rm(0, 3, 5); compare();
    cache.seq_div(0, 0, 32, 2);
    const auto repeated = {0, 4, 2, 4, 15};
    std::copy(repeated.begin(), repeated.end(), positions.begin());
    compare();
    require(cache.find_slot_replay(batch).idxs[0] == std::vector<uint32_t>({1, 9, 5, 9, 31}), "duplicate positions did not select the highest cell");
    ++checks;
    cache.seq_add(0, 4, 6, 7);
    positions[1] = 11; positions[3] = 12;
    compare();
    return checks;
}

static snapshot run(llama_model * model, const char * projector, const std::vector<llama_token> & ids,
        bool streamed, bool stock, int window, bool vtail, bool cancel_first = false, ggml_type hidden_type = GGML_TYPE_F16, bool other_sequence = false,
        bool unified = true, uint32_t parallel = 2, llama_seq_id injector = 0) {
    std::unique_ptr<llama_ridgefill, decltype(&llama_ridgefill_free)> rf(streamed ?
            llama_ridgefill_init_streamed(model, projector) : llama_ridgefill_init(model, projector), llama_ridgefill_free);
    require(bool(rf), "projector load failed");
    if (!std::getenv("LLAMA_RIDGEFILL_TRIM_KV")) { require(rf->trim_kv, "K/V reuse must default on"); }
    auto cp = llama_context_default_params();
    cp.n_seq_max = other_sequence ? parallel : 1;
    cp.n_ctx = (ids.size() + 256)*cp.n_seq_max; cp.n_batch = cp.n_ubatch = 128;
    cp.kv_unified = unified;
    cp.n_threads = cp.n_threads_batch = 4; cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, cp), llama_free);
    require(bool(ctx), "context creation failed");
    require(llama_set_ridgefill(ctx.get(), rf.get()) == 0, "projector attach failed");
    if (streamed) { require(llama_ridgefill_set_alpha(ctx.get(), 1) < 0, "stream accepted nonzero alpha"); }
    require(llama_ridgefill_set_alpha(ctx.get(), 0) == 0, "alpha refused");
    llama_ridgefill_seq_set(ctx.get(), injector, 1);
    auto single_pass = llama_batch_init(1, 0, 1);
    single_pass.n_tokens = 1; single_pass.token[0] = ids[0]; single_pass.pos[0] = 0;
    single_pass.n_seq_id[0] = 1; single_pass.seq_id[0][0] = injector; single_pass.logits[0] = false;
    const int single_rc = llama_decode(ctx.get(), single_pass);
    llama_batch_free(single_pass);
    require(streamed ? single_rc < 0 : single_rc == 0, "single-pass storage guard or resident control failed");
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    llama_ridgefill_seq_set(ctx.get(), injector, 0);
    auto decode = [&](int begin, int count, bool output, const llama_token * tokens, llama_seq_id seq) {
        auto batch = llama_batch_init(count, 0, 1);
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = tokens[i]; batch.pos[i] = begin + i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = seq; batch.logits[i] = output && i + 1 == count;
        }
        const int rc = llama_decode(ctx.get(), batch);
        llama_batch_free(batch);
        require(rc == 0, "decode failed");
    };
    const llama_seq_id background = injector == 0 ? 1 : 0;
    if (other_sequence) { decode(0, 16, false, ids.data(), background); }
    const auto other_before = other_sequence ? state(ctx.get(), background) : std::vector<uint8_t>();
    snapshot result;
    if (streamed) {
        result.map_bytes = ctx->get_ridgefill_stream()->buffer_bytes();
        size_t largest = 0;
        for (size_t il = 0; il < rf->w.size(); ++il) {
            if (rf->w[il]) { largest = std::max(largest, ggml_nbytes(rf->w[il]) + ggml_nbytes(rf->b[il])); }
        }
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
        require(result.map_bytes <= 2*(largest + 3*alignment), "more than two map payloads staged");
    }
    if (stock) {
        llama_ridgefill_seq_set(ctx.get(), injector, 0);
        for (size_t pos = 0; pos < ids.size(); pos += 128) {
            const int count = std::min<size_t>(128, ids.size() - pos);
            decode(pos, count, pos + count == ids.size(), ids.data() + pos, injector);
        }
    } else {
        server_ridgefill_sweep_params params;
        params.n_embd = llama_model_n_embd(model); params.n_tokens = ids.size(); params.n_prefix = ids.size() - 400;
        params.n_batch = 128; params.split = llama_ridgefill_split(rf.get()); params.qattn_share = 0.2f;
        params.type = hidden_type;
        params.s1 = window > 0; params.s1_window = window; params.projector = rf.get();
        params.vtail = vtail; params.vtail_rows = 32;
        llama_ridgefill_seq_set(ctx.get(), injector, params.n_prefix);
        if (cancel_first) {
            server_ridgefill_sweep partial;
            auto status = server_ridgefill_sweep_result::pending;
            int steps = 0;
            while (status == server_ridgefill_sweep_result::pending && ++steps < 10000) {
                status = server_ridgefill_sweep_step(partial, ctx.get(), injector, ids.data(), {}, params);
                if (std::strcmp(partial.phase_name(), "stream_tail") == 0 && partial.step_tokens() > 0) { break; }
            }
            require(status == server_ridgefill_sweep_result::pending && partial.step_tokens() > 0 && steps < 10000, "did not reach an active streamed prefix");
            auto narrowed = params;
            narrowed.n_batch = 127;
            require(partial.next_tokens(ctx.get(), narrowed) == 128, "atomic first-tail span shrank with the prompt budget");
            require(partial.next_work(ctx.get(), injector, params, 128) <= 128, "atomic span refused a full prompt budget");
            bool refused = false;
            try { partial.next_work(ctx.get(), injector, narrowed, 127); }
            catch (const std::runtime_error & error) {
                refused = std::strstr(error.what(), "exceeds the stock prompt chunk budget") != nullptr;
            }
            require(refused, "server budget check admitted an oversized atomic span");
            partial.cancel(ctx.get(), injector);
            require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), injector) < 0, "cancel retained sequence memory");
            llama_ridgefill_seq_set(ctx.get(), injector, params.n_prefix);
        }
        server_ridgefill_sweep sweep;
        auto status = server_ridgefill_sweep_result::pending;
        int steps = 0;
        while (status == server_ridgefill_sweep_result::pending && ++steps < 10000) {
            status = server_ridgefill_sweep_step(sweep, ctx.get(), injector, ids.data(), {}, params);
        }
        require(status == server_ridgefill_sweep_result::complete, "sweep did not complete");
        result.mass = sweep.get_s1_mass(); result.rows = sweep.get_rows(); result.scales = sweep.get_vtail_scales();
    }
    result.prefill = state(ctx.get(), injector);
    if (other_sequence) { require(other_before == state(ctx.get(), background), "prefill changed the other sequence state"); }
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int i = 0; i < 5; ++i) {
        const auto * logits = llama_get_logits(ctx.get());
        require(logits && vocab > 0, "missing full-vocabulary logits");
        result.logits.insert(result.logits.end(), logits, logits + vocab);
        if (i < 4) { decode(ids.size() + i, 1, true, ids.data() + i, injector); }
    }
    result.final = state(ctx.get(), injector);
    return result;
}

int main(int argc, char ** argv) {
    try {
        const bool separate = argc == 6 && std::strcmp(argv[5], "--separate") == 0;
        require(argc == 5 || separate, "usage: test-streaming MODEL PROJECTOR INPUT_JSON OUTPUT_JSON [--separate]");
        std::ifstream input(argv[3]); json document; input >> document;
        const auto ids = document.at("ids").get<std::vector<llama_token>>();
        require(ids.size() > 400 && ids.size() < 4096, "use a short CPU fixture with an exact tail");
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "model load failed");
        if (separate) {
            json cases = json::array();
            int checks = 0;
            auto check = [&](bool ok, const char * message) { require(ok, message); ++checks; };
            for (const uint32_t parallel : {2, 8}) {
                for (const llama_seq_id injector : {0, int32_t(parallel - 1)}) {
                    const auto a = run(model.get(), argv[2], ids, false, false, 256, true, false, GGML_TYPE_F16, true, false, parallel, injector);
                    const auto b = run(model.get(), argv[2], ids, true, false, 256, true, false, GGML_TYPE_F16, true, false, parallel, injector);
                    check(!a.rows.empty() && a.rows.size() < ids.size() - 400, "missing separate-KV approximate prefix");
                    check(a.mass.size() == ids.size() - 400 && *std::max_element(a.mass.begin(), a.mass.end()) > 0, "separate-KV S1 mass was not measured");
                    check(a.prefill == b.prefill && a.final == b.final, "separate-KV resident/streamed state bytes differ");
                    check(same_logits(a.logits, b.logits), "separate-KV resident/streamed logit bytes differ");
                    check(a.rows == b.rows && a.mass == b.mass, "separate-KV resident/streamed selection differs");
                    check(!a.scales.empty() && a.scales == b.scales, "separate-KV vtail scales differ");
                    cases.push_back({{"parallel", parallel}, {"injector", injector}, {"state_bytes", b.prefill.size()}, {"exact_rows", b.rows.size()}});
                }
            }
            std::ofstream output(argv[4]);
            require(bool(output), "cannot open result file");
            output << json({{"result", "PASS"}, {"checks", checks}, {"cases", cases}}).dump(2) << '\n';
            require(bool(output), "cannot write result file");
            std::printf("separate KV: %d checks passed\n", checks);
            return 0;
        }
        const auto stock = run(model.get(), argv[2], ids, false, true, 0, false);
        const auto stock_stream = run(model.get(), argv[2], ids, true, true, 0, false);
        int checks = check_replay_lookup(*model);
        auto check = [&](bool ok, const char * message) { require(ok, message); ++checks; };
        check(stock.prefill == stock_stream.prefill, "off prefill-state bytes differ");
        check(stock.final == stock_stream.final, "off final-state bytes differ");
        check(same_logits(stock.logits, stock_stream.logits), "off full-vocabulary logits differ");
        json cases = json::array();
        for (int index = 0; index < 4; ++index) {
            const int window = index == 0 ? 0 : index == 3 ? 256 : 64;
            const bool vt = index == 2;
            std::printf("streaming case %d: window %d vtail %d\n", index, window, vt);
            std::fflush(stdout);
            const auto a = run(model.get(), argv[2], ids, false, false, window, vt);
            const auto b = run(model.get(), argv[2], ids, true, false, window, vt);
            check(!a.rows.empty() && a.rows.size() < ids.size() - 400, "missing approximate-prefix control");
            check(a.prefill != stock.prefill, "RidgeFill positive control is stock");
            check(a.prefill == b.prefill, "streamed prefill KV/conv/recurrent-state bytes differ");
            check(a.final == b.final, "streamed final KV/conv/recurrent-state bytes differ");
            check(same_logits(a.logits, b.logits), "streamed full-vocabulary logits differ");
            check(a.rows == b.rows && a.mass == b.mass, "S1 rows or mass differ");
            check(a.scales == b.scales, "vtail scales differ");
            check(b.map_bytes > 0 && b.map_bytes == stock_stream.map_bytes, "staging buffer grows with prompt work");
            if (window) { check(a.mass.size() == ids.size() - 400 && *std::max_element(a.mass.begin(), a.mass.end()) > 0, "S1 mass was not measured"); }
            if (vt) { check(!a.scales.empty(), "vtail scales were not measured"); }
            cases.push_back({{"window", window}, {"vtail", vt}, {"map_bytes", b.map_bytes},
                    {"state_bytes", b.prefill.size()}, {"exact_rows", b.rows}});
        }
        const auto reference = run(model.get(), argv[2], ids, true, false, 64, true);
        const auto reused = run(model.get(), argv[2], ids, true, false, 64, true, true);
        check(reference.prefill == reused.prefill && reference.final == reused.final, "cancel/reuse state bytes differ");
        check(same_logits(reference.logits, reused.logits), "cancel/reuse logit bytes differ");
        check(reference.rows == reused.rows && reference.mass == reused.mass && reference.scales == reused.scales,
                "cancel/reuse selection or calibration leaked");
        for (const auto type : {GGML_TYPE_BF16, GGML_TYPE_F32}) {
            const auto a = run(model.get(), argv[2], ids, false, false, 64, true, false, type);
            const auto b = run(model.get(), argv[2], ids, true, false, 64, true, false, type);
            check(a.prefill == b.prefill, "compact alternate-type prefill bytes differ");
            check(a.final == b.final, "compact alternate-type final bytes differ");
            check(same_logits(a.logits, b.logits), "compact alternate-type logits differ");
            check(a.rows == b.rows && a.mass == b.mass, "compact alternate-type selection differs");
            check(a.scales == b.scales, "compact alternate-type vtail differs");
        }
        const auto background_a = run(model.get(), argv[2], ids, false, false, 64, true, false, GGML_TYPE_F16, true);
        const auto background_b = run(model.get(), argv[2], ids, true, false, 64, true, false, GGML_TYPE_F16, true);
        check(background_a.prefill == background_b.prefill, "other-layout prefill bytes differ");
        check(background_a.final == background_b.final, "other-layout final bytes differ");
        check(same_logits(background_a.logits, background_b.logits), "other-layout logits differ");
        check(background_a.rows == background_b.rows && background_a.mass == background_b.mass, "other-layout selection differs");
        check(background_a.scales == background_b.scales, "other-layout vtail differs");
        std::ofstream output(argv[4]);
        require(bool(output), "cannot open result file");
        output << json({{"result", "PASS"}, {"checks", checks}, {"atomic_budget_regression", true}, {"cases", cases}}).dump(2) << '\n';
        require(bool(output), "cannot write result file");
        std::printf("streaming: %d checks passed\n", checks);
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
