#include "llama.h"
#include "llama-ext.h"
#include "llama-context.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-model.h"
#include "server-ridgefill-sweep.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static std::vector<uint8_t> state(llama_context * ctx, llama_seq_id seq) {
    std::vector<uint8_t> bytes(llama_state_seq_get_size(ctx, seq));
    require(!bytes.empty() && llama_state_seq_get_data(ctx, bytes.data(), bytes.size(), seq) == bytes.size(), "incomplete sequence state");
    return bytes;
}

struct result {
    std::vector<double> mass;
    std::vector<llama_pos> rows;
    std::vector<uint8_t> cache;
    std::vector<float> logits;
};

static result run(llama_model * model, const char * projector, const std::vector<llama_token> & ids,
        int occupied, ggml_type type, bool gpu, bool streamed) {
    std::unique_ptr<llama_ridgefill, decltype(&llama_ridgefill_free)> rf(streamed ?
            llama_ridgefill_init_streamed(model, projector) : llama_ridgefill_init(model, projector), llama_ridgefill_free);
    require(bool(rf), "projector load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 49152; cp.n_seq_max = 8; cp.kv_unified = true;
    cp.n_batch = cp.n_ubatch = 512; cp.n_threads = cp.n_threads_batch = 4;
    cp.type_k = cp.type_v = type;
    cp.flash_attn_type = gpu ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, cp), llama_free);
    require(bool(ctx) && llama_set_ridgefill(ctx.get(), rf.get()) == 0, "context/projector initialization failed");

    // Occupied zero-valued cells isolate physical placement from background model arithmetic.
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(ctx->get_memory());
    require(hybrid != nullptr, "fixture needs a hybrid model");
    auto & cache = *hybrid->get_mem_attn();
    std::vector<llama_pos> positions(occupied);
    std::iota(positions.begin(), positions.end(), 0);
    std::vector<int32_t> counts(occupied, 1);
    llama_seq_id background = 0;
    std::vector<llama_seq_id *> owners(occupied, &background);
    llama_ubatch batch = {};
    batch.n_tokens = batch.n_seq_tokens = occupied;
    batch.n_seqs = batch.n_seqs_unq = batch.n_pos = 1;
    batch.pos = positions.data(); batch.n_seq_id = counts.data();
    batch.seq_id = owners.data(); batch.seq_id_unq = &background;
    const auto slot = cache.find_slot(batch, false);
    require(!slot.empty(), "cannot seed the shared KV cache");
    cache.apply_ubatch(slot, batch);
    const auto before = state(ctx.get(), background);

    constexpr llama_seq_id seq = 7;
    server_ridgefill_sweep_params params;
    params.n_embd = llama_model_n_embd(model); params.n_tokens = ids.size(); params.n_prefix = ids.size() - 400;
    params.n_batch = 128; params.split = llama_ridgefill_split(rf.get()); params.qattn_share = 0.2f;
    params.s1 = true; params.s1_window = 256; params.projector = rf.get();
    llama_ridgefill_seq_set(ctx.get(), seq, params.n_prefix);
    server_ridgefill_sweep sweep;
    auto status = server_ridgefill_sweep_result::pending;
    int steps = 0;
    while (status == server_ridgefill_sweep_result::pending && ++steps < 10000) {
        status = server_ridgefill_sweep_step(sweep, ctx.get(), seq, ids.data(), {}, params);
    }
    require(status == server_ridgefill_sweep_result::complete, "S1 sweep did not complete");
    if (occupied >= 30000) {
        require(sweep.get_cost().s1_probe_kv_pairs == uint64_t(params.s1_window)*params.n_tokens, "probe pair count includes other sequences");
    }
    result out;
    out.mass = sweep.get_s1_mass(); out.rows = sweep.get_rows();
    require(out.mass.size() == size_t(params.n_prefix) && !out.rows.empty(), "missing S1 mass/rows");
    require(std::all_of(out.mass.begin(), out.mass.end(), [](double x) { return std::isfinite(x) && x >= 0; }) &&
            *std::max_element(out.mass.begin(), out.mass.end()) > 0, "non-finite, negative or empty S1 mass");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto * logits = llama_get_logits(ctx.get());
    require(logits && vocab > 0, "missing logits");
    out.logits.assign(logits, logits + vocab);
    require(std::all_of(out.logits.begin(), out.logits.end(), [](float x) { return std::isfinite(x); }), "non-finite logits");
    out.cache = state(ctx.get(), seq);
    require(before == state(ctx.get(), background), "S1 changed the background sequence");
    return out;
}

int main(int argc, char ** argv) {
    try {
        require(argc == 5 || (argc == 6 && std::strcmp(argv[5], "--gpu") == 0), "usage: test-s1-unified MODEL PROJECTOR INPUT_JSON OUTPUT_JSON [--gpu]");
        std::ifstream input(argv[3]); json doc; input >> doc;
        const auto ids = doc.at("ids").get<std::vector<llama_token>>();
        require(ids.size() > 656 && ids.size() < 4096, "use a tiny fixture with a 400-row exact tail");
        const bool gpu = argc == 6;
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = gpu ? 99 : 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "model load failed");
        if (gpu) {
            const auto dev = ggml_backend_buft_get_device(model->select_buft(0));
            require(dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU, "GPU fixture was not offloaded");
        }
        json cases = json::array();
        for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            for (const int occupied : {16, 30000, 33000}) {
                const auto a = run(model.get(), argv[2], ids, occupied, type, gpu, false);
                const auto b = run(model.get(), argv[2], ids, occupied, type, gpu, true);
                require(a.mass == b.mass && a.rows == b.rows, "resident/streamed S1 selection differs");
                require(a.cache == b.cache, "resident/streamed cache bytes differ");
                require(a.logits.size() == b.logits.size() && std::memcmp(a.logits.data(), b.logits.data(), a.logits.size()*sizeof(float)) == 0,
                        "resident/streamed logit bytes differ");
                cases.push_back({{"occupied", occupied}, {"cache_type", ggml_type_name(type)}, {"mass", a.mass}, {"rows", a.rows}});
                std::printf("PASS: occupied=%d cache=%s resident/streamed\n", occupied, ggml_type_name(type));
                std::fflush(stdout);
            }
        }
        std::ofstream(argv[4]) << json{{"passed", true}, {"gpu", gpu}, {"cases", cases}}.dump(2) << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
