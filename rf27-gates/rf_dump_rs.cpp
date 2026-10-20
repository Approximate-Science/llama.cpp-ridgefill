// Derived from L-B's tests/ridgefill/rf_dump.cpp (rf27-tools e0e1fa9) with --n-rs-seq (recurrent rollback snapshots)
// and --cross (batches cross the fill end, as the server submits them; no prefix dump then), --fa 0|1,
// --n-seq N (the same prompt on N sequences of a unified KV cache, in the same batches; seq s dumps to <output>.s<s>)
// and --rows1 (exact rows of sequences >= 1; empty: speed).
// CPU gate harness. Dump prefix and final sequence state plus tail logits; never score a filled row.
#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void write_bytes(const std::string & path, const void * data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<const char *>(data), size);
    if (!file) {
        throw std::runtime_error("cannot write " + path);
    }
}

static void dump_state(llama_context * ctx, const std::string & path, llama_seq_id seq = 0) {
    const size_t size = llama_state_seq_get_size(ctx, seq);
    if (size == 0) {
        throw std::runtime_error("sequence state is empty");
    }
    std::vector<uint8_t> data(size);
    if (llama_state_seq_get_data(ctx, data.data(), data.size(), seq) != size) {
        throw std::runtime_error("sequence state size changed");
    }
    write_bytes(path, data.data(), data.size());
}

static int run(int argc, char ** argv) {
    std::string model_path, tokens_path, output, projector, mode = "off", rows_arg;
    int n_rs_seq = 0;
    bool cross = false;
    int fa = 0;
    int n_seq = 1;
    std::string rows1_arg;
    int tail = 200, keep = 200, n_batch = 700, n_ubatch = 512;
    float alpha = 0;
    bool filled_logit = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() {
            if (++i >= argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            return std::string(argv[i]);
        };
        if      (arg == "--model")     { model_path = next(); }
        else if (arg == "--tokens")    { tokens_path = next(); }
        else if (arg == "--output")    { output = next(); }
        else if (arg == "--projector") { projector = next(); }
        else if (arg == "--mode")      { mode = next(); }
        else if (arg == "--tail")      { tail = std::stoi(next()); }
        else if (arg == "--keep")      { keep = std::stoi(next()); }
        else if (arg == "--batch")     { n_batch = std::stoi(next()); }
        else if (arg == "--ubatch")    { n_ubatch = std::stoi(next()); }
        else if (arg == "--alpha")     { alpha = std::stof(next()); }
        else if (arg == "--rows")      { rows_arg = next(); }
        else if (arg == "--filled-logit") { filled_logit = true; }
        else if (arg == "--n-rs-seq")  { n_rs_seq = std::stoi(next()); }
        else if (arg == "--cross")     { cross = true; }
        else if (arg == "--fa")        { fa = std::stoi(next()); }
        else if (arg == "--n-seq")     { n_seq = std::stoi(next()); }
        else if (arg == "--rows1")     { rows1_arg = next(); }
        else { throw std::invalid_argument("unknown argument " + arg); }
    }
    if (model_path.empty() || tokens_path.empty() || output.empty() || n_batch < 1 || n_ubatch < 1 || tail < 1 || keep < 1 || !(alpha >= 0 && alpha <= 1)) {
        throw std::invalid_argument("model, tokens, output and positive batch/tail/keep are required; alpha must be in [0, 1]");
    }
    if (mode != "off" && mode != "speed" && mode != "quality") {
        throw std::invalid_argument("unsupported --mode " + mode + ": quality+qattn requires the sweep harness");
    }
#ifndef RF27_API
    if (alpha != 0 || mode == "quality") {
        throw std::invalid_argument("--alpha/--mode quality require the RF27 engine API");
    }
#endif
    std::ifstream input(tokens_path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0 || input.tellg() % sizeof(int32_t) != 0) {
        throw std::invalid_argument("--tokens must contain int32 token IDs");
    }
    std::vector<int32_t> tokens(static_cast<size_t>(input.tellg()) / sizeof(int32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(tokens.data()), tokens.size() * sizeof(int32_t));
    const int n = tokens.size();
    const int prefix = mode == "off" || tail >= n ? 0 : n - tail;
    if (keep > n - prefix || (mode != "off" && projector.empty())) {
        throw std::invalid_argument("keep exceeds exact tail or mode requires --projector");
    }
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
    if (!model) { throw std::runtime_error("model load failed"); }
    std::unique_ptr<llama_ridgefill, decltype(&llama_ridgefill_free)> rf(nullptr, llama_ridgefill_free);
    if (!projector.empty()) {
        rf.reset(llama_ridgefill_init(model.get(), projector.c_str()));
        if (!rf) { throw std::runtime_error("projector load failed"); }
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = ((n + 255) / 256) * 256 * n_seq;
    cp.n_batch = n_batch * n_seq;
    cp.n_ubatch = n_ubatch * n_seq;
    cp.n_seq_max = n_seq;
    cp.kv_unified = true;
    cp.n_rs_seq = n_rs_seq;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) { throw std::runtime_error("context creation failed"); }
    if (rf) {
        llama_set_ridgefill(ctx.get(), rf.get());
#ifdef RF27_API
        llama_ridgefill_set_alpha(ctx.get(), alpha);
#endif
        for (int seq = 0; seq < n_seq; ++seq) {
        llama_ridgefill_seq_set(ctx.get(), seq, prefix);
#ifdef RF27_API
        const std::string & arg_rows = seq == 0 ? rows_arg : rows1_arg;
        if (mode == "quality" && !arg_rows.empty()) {
            std::vector<llama_pos> rows;
            size_t begin = 0;
            while (begin < arg_rows.size()) {
                size_t end = arg_rows.find(',', begin);
                const int row = std::stoi(arg_rows.substr(begin, end - begin));
                if (row < 0 || row >= prefix || (!rows.empty() && row <= rows.back())) {
                    throw std::invalid_argument("--rows must be sorted unique prefix positions");
                }
                rows.push_back(row);
                if (end == std::string::npos) { break; }
                begin = end + 1;
            }
            if (llama_ridgefill_seq_set_rows(ctx.get(), seq, rows.data(), rows.size()) != 0) {
                throw std::invalid_argument("llama_ridgefill_seq_set_rows refused the rows");
            }
        }
#endif
        }
    }
    llama_batch batch = llama_batch_init(n_batch * n_seq, 0, 1);
    std::vector<std::vector<float>> logits(n_seq);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    auto name = [&](int seq) { return n_seq == 1 ? output : output + ".s" + std::to_string(seq); };
    for (int start = 0; start < n;) {
        // Also dump stock at the same boundary, to compare prefix states before the tail overwrites them.
        const int boundary = !cross && start < n - std::min(tail, n) ? n - std::min(tail, n) : n;
        const int count = std::min(n_batch, boundary - start);
        batch.n_tokens = count * n_seq;
        for (int seq = 0; seq < n_seq; ++seq) {
            for (int i = 0; i < count; ++i) {
                const int pos = start + i;
                const int j = seq * count + i;
                batch.token[j] = tokens[pos];
                batch.pos[j] = pos;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = seq;
                batch.logits[j] = pos >= n - keep || (filled_logit && pos == 0);
            }
        }
        const int result = llama_decode(ctx.get(), batch);
        if (result != 0) {
            llama_batch_free(batch);
            fprintf(stderr, "llama_decode refused batch: %d\n", result);
            return 2;
        }
        for (int seq = 0; seq < n_seq; ++seq) {
            for (int i = 0; i < count; ++i) {
                if (start + i >= n - keep) {
                    const float * row = llama_get_logits_ith(ctx.get(), seq * count + i);
                    if (!row) { throw std::runtime_error("missing tail logits"); }
                    logits[seq].insert(logits[seq].end(), row, row + vocab);
                }
            }
        }
        start += count;
        if (!cross && start == n - std::min(tail, n) && start > 0) {
            for (int seq = 0; seq < n_seq; ++seq) {
                dump_state(ctx.get(), name(seq) + ".prefix.kv", seq);
            }
        }
    }
    llama_batch_free(batch);
    for (int seq = 0; seq < n_seq; ++seq) {
        if (logits[seq].size() != static_cast<size_t>(keep) * vocab) {
            throw std::runtime_error("incomplete tail logits");
        }
        write_bytes(name(seq) + ".logits", logits[seq].data(), logits[seq].size() * sizeof(float));
        dump_state(ctx.get(), name(seq) + ".kv", seq);
    }
    fprintf(stderr, "dumped n=%d prefix=%d keep=%d vocab=%d mode=%s alpha=%g\n", n, prefix, keep, vocab, mode.c_str(), alpha);
    return 0;
}

int main(int argc, char ** argv) {
    try {
        llama_backend_init();
        const int result = run(argc, argv);
        llama_backend_free();
        return result;
    } catch (const std::exception & error) {
        fprintf(stderr, "rf_dump: %s\n", error.what());
        return 1;
    }
}
