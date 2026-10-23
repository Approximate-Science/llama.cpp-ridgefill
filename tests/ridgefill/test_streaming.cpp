#include "llama.h"
#include "llama-ext.h"
#include "llama-context.h"
#include "server-ridgefill-sweep.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdio>
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
    std::vector<std::vector<float>> beta;
    size_t map_bytes = 0;
};

static std::vector<uint8_t> state(llama_context * ctx) {
    std::vector<uint8_t> bytes(llama_state_seq_get_size(ctx, 0));
    require(!bytes.empty(), "empty sequence state");
    require(llama_state_seq_get_data(ctx, bytes.data(), bytes.size(), 0) == bytes.size(), "incomplete sequence state");
    return bytes;
}

static snapshot run(llama_model * model, const char * projector, const std::vector<llama_token> & ids,
        bool streamed, bool stock, int window, bool vtail, int tau, bool cancel_first = false) {
    std::unique_ptr<llama_ridgefill, decltype(&llama_ridgefill_free)> rf(streamed ?
            llama_ridgefill_init_streamed(model, projector) : llama_ridgefill_init(model, projector), llama_ridgefill_free);
    require(bool(rf), "projector load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = ids.size() + 256; cp.n_batch = cp.n_ubatch = 128; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 4; cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, cp), llama_free);
    require(bool(ctx), "context creation failed");
    require(llama_set_ridgefill(ctx.get(), rf.get()) == 0, "projector attach failed");
    if (streamed) { require(llama_ridgefill_set_alpha(ctx.get(), 1) < 0, "stream accepted nonzero alpha"); }
    require(llama_ridgefill_set_alpha(ctx.get(), 0) == 0, "alpha refused");
    llama_ridgefill_seq_set(ctx.get(), 0, 1);
    auto single_pass = llama_batch_init(1, 0, 1);
    single_pass.n_tokens = 1; single_pass.token[0] = ids[0]; single_pass.pos[0] = 0;
    single_pass.n_seq_id[0] = 1; single_pass.seq_id[0][0] = 0; single_pass.logits[0] = false;
    const int single_rc = llama_decode(ctx.get(), single_pass);
    llama_batch_free(single_pass);
    require(streamed ? single_rc < 0 : single_rc == 0, "single-pass storage guard or resident control failed");
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    llama_ridgefill_seq_set(ctx.get(), 0, 0);
    auto decode = [&](int begin, int count, bool output, const llama_token * tokens) {
        auto batch = llama_batch_init(count, 0, 1);
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = tokens[i]; batch.pos[i] = begin + i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = output && i + 1 == count;
        }
        const int rc = llama_decode(ctx.get(), batch);
        llama_batch_free(batch);
        require(rc == 0, "decode failed");
    };
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
        llama_ridgefill_seq_set(ctx.get(), 0, 0);
        for (size_t pos = 0; pos < ids.size(); pos += 128) {
            const int count = std::min<size_t>(128, ids.size() - pos);
            decode(pos, count, pos + count == ids.size(), ids.data() + pos);
        }
    } else {
        server_ridgefill_sweep_params params;
        params.n_embd = llama_model_n_embd(model); params.n_tokens = ids.size(); params.n_prefix = ids.size() - 400;
        params.n_batch = 128; params.split = llama_ridgefill_split(rf.get()); params.qattn_share = 0.2f;
        params.s1 = window > 0; params.s1_window = window; params.projector = rf.get();
        params.vtail = vtail; params.vtail_rows = 32; params.certificate = tau >= 0; params.certificate_tau = tau;
        llama_ridgefill_seq_set(ctx.get(), 0, params.n_prefix);
        if (cancel_first) {
            server_ridgefill_sweep partial;
            auto status = server_ridgefill_sweep_result::pending;
            int steps = 0;
            while (status == server_ridgefill_sweep_result::pending && ++steps < 10000) {
                status = server_ridgefill_sweep_step(partial, ctx.get(), 0, ids.data(), {}, params);
                if (std::strcmp(partial.phase_name(), "stream_tail") == 0 && partial.step_tokens() > 0) { break; }
            }
            require(status == server_ridgefill_sweep_result::pending && partial.step_tokens() > 0 && steps < 10000, "did not reach an active streamed prefix");
            auto narrowed = params;
            narrowed.n_batch = 127;
            require(partial.next_tokens(ctx.get(), narrowed) == 128, "atomic first-tail span shrank with the prompt budget");
            require(partial.next_work(ctx.get(), 0, params, 128) <= 128, "atomic span refused a full prompt budget");
            bool refused = false;
            try { partial.next_work(ctx.get(), 0, narrowed, 127); }
            catch (const std::runtime_error & error) {
                refused = std::strstr(error.what(), "exceeds the stock prompt chunk budget") != nullptr;
            }
            require(refused, "server budget check admitted an oversized atomic span");
            partial.cancel(ctx.get(), 0);
            require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) < 0, "cancel retained sequence memory");
            llama_ridgefill_seq_set(ctx.get(), 0, params.n_prefix);
        }
        server_ridgefill_sweep sweep;
        auto status = server_ridgefill_sweep_result::pending;
        int steps = 0;
        while (status == server_ridgefill_sweep_result::pending && ++steps < 10000) {
            status = server_ridgefill_sweep_step(sweep, ctx.get(), 0, ids.data(), {}, params);
        }
        require(status == server_ridgefill_sweep_result::complete, "sweep did not complete");
        result.mass = sweep.get_s1_mass(); result.rows = sweep.get_rows(); result.scales = sweep.get_vtail_scales();
        result.beta = sweep.get_certificate().beta;
    }
    result.prefill = state(ctx.get());
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int i = 0; i < 5; ++i) {
        const auto * logits = llama_get_logits(ctx.get());
        require(logits && vocab > 0, "missing full-vocabulary logits");
        result.logits.insert(result.logits.end(), logits, logits + vocab);
        if (i < 4) { decode(ids.size() + i, 1, true, ids.data() + i); }
    }
    result.final = state(ctx.get());
    return result;
}

int main(int argc, char ** argv) {
    try {
        require(argc == 5, "usage: test-streaming MODEL PROJECTOR INPUT_JSON OUTPUT_JSON");
        std::ifstream input(argv[3]); json document; input >> document;
        const auto ids = document.at("ids").get<std::vector<llama_token>>();
        require(ids.size() > 400 && ids.size() < 4096, "use a short CPU fixture with an exact tail");
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "model load failed");
        const auto stock = run(model.get(), argv[2], ids, false, true, 0, false, -1);
        const auto stock_stream = run(model.get(), argv[2], ids, true, true, 0, false, -1);
        int checks = 0;
        auto check = [&](bool ok, const char * message) { require(ok, message); ++checks; };
        check(stock.prefill == stock_stream.prefill, "off prefill-state bytes differ");
        check(stock.final == stock_stream.final, "off final-state bytes differ");
        check(same_logits(stock.logits, stock_stream.logits), "off full-vocabulary logits differ");
        json cases = json::array();
        size_t s1_rows = 0;
        for (int index = 0; index < 6; ++index) {
            const int window = index == 0 ? 0 : index == 5 ? 256 : 64;
            const bool vt = index == 2;
            const int tau = index == 3 ? 1 : index >= 4 ? 0 : -1;
            std::printf("streaming case %d: window %d vtail %d tau %d\n", index, window, vt, tau);
            std::fflush(stdout);
            const auto a = run(model.get(), argv[2], ids, false, false, window, vt, tau);
            const auto b = run(model.get(), argv[2], ids, true, false, window, vt, tau);
            check(!a.rows.empty() && a.rows.size() < ids.size() - 400, "missing approximate-prefix control");
            check(a.prefill != stock.prefill, "RidgeFill positive control is stock");
            check(a.prefill == b.prefill, "streamed prefill KV/conv/recurrent-state bytes differ");
            check(a.final == b.final, "streamed final KV/conv/recurrent-state bytes differ");
            check(same_logits(a.logits, b.logits), "streamed full-vocabulary logits differ");
            check(a.rows == b.rows && a.mass == b.mass, "S1 rows or mass differ");
            check(a.beta == b.beta, "per-head probe beta differs");
            check(a.scales == b.scales, "vtail scales differ");
            check(b.map_bytes > 0 && b.map_bytes == stock_stream.map_bytes, "staging buffer grows with prompt work");
            if (window) { check(a.mass.size() == ids.size() - 400 && *std::max_element(a.mass.begin(), a.mass.end()) > 0, "S1 mass was not measured"); }
            if (vt) { check(!a.scales.empty(), "vtail scales were not measured"); }
            if (tau >= 0) { check(!a.beta.empty() && !a.beta[0].empty() && a.beta[0][0] > 0, "beta was not measured"); }
            if (index == 1) { s1_rows = a.rows.size(); }
            if (index == 3) { check(a.rows.size() == s1_rows, "tau1 negative control added rows"); }
            if (index == 4) { check(a.rows.size() > s1_rows && a.rows.size() <= size_t(0.23*(ids.size() - 400)), "tau0 positive control did not repair within cap"); }
            cases.push_back({{"window", window}, {"vtail", vt}, {"tau", tau}, {"map_bytes", b.map_bytes},
                    {"state_bytes", b.prefill.size()}, {"exact_rows", b.rows}, {"beta", b.beta}});
        }
        const auto reference = run(model.get(), argv[2], ids, true, false, 64, true, 0);
        const auto reused = run(model.get(), argv[2], ids, true, false, 64, true, 0, true);
        check(reference.prefill == reused.prefill && reference.final == reused.final, "cancel/reuse state bytes differ");
        check(same_logits(reference.logits, reused.logits), "cancel/reuse logit bytes differ");
        check(reference.rows == reused.rows && reference.mass == reused.mass && reference.beta == reused.beta && reference.scales == reused.scales,
                "cancel/reuse selection or calibration leaked");
        std::ofstream output(argv[4]);
        require(bool(output), "cannot open result file");
        output << json({{"result", "PASS"}, {"checks", checks}, {"atomic_budget_regression", true}, {"cases", cases}}).dump(2) << '\n';
        require(bool(output), "cannot write result file");
        std::printf("streaming: %d checks passed\n", checks);
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
