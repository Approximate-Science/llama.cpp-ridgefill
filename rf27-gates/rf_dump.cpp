// Gate harness: decode token files (int32), dump the logits of the last `keep` rows, the NLL of the last `score`
// tokens and the KV cache of seq 0 (one prompt at a time, the memory is cleared between prompts).
// Built against the RidgeFill branch, or with -DSTOCK against stock llama.cpp (no RidgeFill calls).
#include "llama.h"
#ifndef STOCK
#include "llama-ext.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int32_t> read_tokens(const std::string & path) {
    std::vector<int32_t> toks;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return toks;
    }
    int32_t t;
    while (fread(&t, 4, 1, f) == 1) {
        toks.push_back(t);
    }
    fclose(f);
    return toks;
}

int main(int argc, char ** argv) {
    std::string model_path, tok_list, out_prefix, proj;
    int tail = 0, keep = 1, n_batch = 512, n_ubatch = 512, fa = 0, threads = 4, ngl = 0, score = 0, no_kv = 0;
    ggml_type ctk = GGML_TYPE_F32, ctv = GGML_TYPE_F32;
    auto type_of = [](const std::string & t) {
        if (t == "f16")  return GGML_TYPE_F16;
        if (t == "bf16") return GGML_TYPE_BF16;
        if (t == "q8_0") return GGML_TYPE_Q8_0;
        return GGML_TYPE_F32;
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if      (a == "-m")        model_path = next();
        else if (a == "-t")        tok_list   = next(); // comma-separated token files
        else if (a == "-o")        out_prefix = next();
        else if (a == "--proj")    proj       = next();
        else if (a == "--tail")    tail       = std::stoi(next());
        else if (a == "--keep")    keep       = std::stoi(next());
        else if (a == "-b")        n_batch    = std::stoi(next());
        else if (a == "-ub")       n_ubatch   = std::stoi(next());
        else if (a == "--fa")      fa         = std::stoi(next());
        else if (a == "--threads") threads    = std::stoi(next());
        else if (a == "-ngl")      ngl        = std::stoi(next());
        else if (a == "-ctk")      ctk        = type_of(next());
        else if (a == "-ctv")      ctv        = type_of(next());
        else if (a == "--score")   score      = std::stoi(next()); // NLL of the last `score` tokens
        else if (a == "--no-kv")   no_kv      = 1;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    std::vector<std::string> files;
    {
        std::stringstream ss(tok_list);
        std::string f;
        while (std::getline(ss, f, ',')) {
            files.push_back(f);
        }
    }
    std::vector<std::vector<int32_t>> prompts;
    int n_max = 0;
    for (const auto & f : files) {
        prompts.push_back(read_tokens(f));
        if (prompts.back().empty()) { fprintf(stderr, "no tokens in %s\n", f.c_str()); return 1; }
        n_max = std::max(n_max, (int) prompts.back().size());
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) return 1;

    auto cp = llama_context_default_params();
    cp.n_ctx           = (n_max + 255) / 256 * 256;
    cp.n_batch         = n_batch;
    cp.n_ubatch        = n_ubatch;
    cp.n_seq_max       = 1;
    cp.n_threads       = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k          = ctk;
    cp.type_v          = ctv;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

#ifndef STOCK
    llama_ridgefill * rf = nullptr;
    if (!proj.empty()) {
        rf = llama_ridgefill_init(model, proj.c_str());
        if (!rf) return 1;
        llama_set_ridgefill(ctx, rf);
    }
#endif

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    double nll_sum = 0.0;
    long   nll_n = 0, top1_n = 0;

    for (size_t ip = 0; ip < prompts.size(); ++ip) {
        const auto & toks = prompts[ip];
        const int n = toks.size();
        const std::string pfx = prompts.size() == 1 ? out_prefix : out_prefix + "." + std::to_string(ip);

        llama_memory_clear(llama_get_memory(ctx), true);
#ifndef STOCK
        if (rf) {
            llama_ridgefill_seq_set(ctx, 0, tail > 0 && tail < n ? n - tail : 0);
        }
#else
        (void) tail;
#endif

        std::vector<float> logits;
        std::vector<float> nll;
        int top1 = 0;
        const int64_t t0 = ggml_time_us();

        for (int p0 = 0; p0 < n; p0 += n_batch) {
            const int nb = std::min(n_batch, n - p0);
            batch.n_tokens = nb;
            for (int k = 0; k < nb; ++k) {
                const int p = p0 + k;
                batch.token[k]     = toks[p];
                batch.pos[k]       = p;
                batch.n_seq_id[k]  = 1;
                batch.seq_id[k][0] = 0;
                batch.logits[k]    = p >= n - keep || (p >= n - 1 - score && p < n - 1);
            }
            const int ret = llama_decode(ctx, batch);
            if (ret != 0) {
                fprintf(stderr, "decode failed: %d\n", ret);
                return 2;
            }
            for (int k = 0; k < nb; ++k) {
                if (!batch.logits[k]) {
                    continue;
                }
                const float * l = llama_get_logits_ith(ctx, k);
                const int p = p0 + k;
                if (p >= n - keep) {
                    logits.insert(logits.end(), l, l + n_vocab);
                }
                if (score > 0 && p >= n - 1 - score && p < n - 1) {
                    double mx = l[0];
                    int am = 0;
                    for (int v = 1; v < n_vocab; ++v) { if (l[v] > mx) { mx = l[v]; am = v; } }
                    double se = 0.0;
                    for (int v = 0; v < n_vocab; ++v) { se += std::exp(l[v] - mx); }
                    nll.push_back((float) (mx + std::log(se) - l[toks[p + 1]]));
                    top1 += am == toks[p + 1];
                }
            }
        }
        llama_synchronize(ctx);
        const double t_ms = (ggml_time_us() - t0) / 1000.0;

        if (keep > 0) {
            FILE * f = fopen((pfx + ".logits").c_str(), "wb");
            fwrite(logits.data(), sizeof(float), logits.size(), f);
            fclose(f);
        }
        if (score > 0) {
            FILE * f = fopen((pfx + ".nll").c_str(), "wb");
            fwrite(nll.data(), sizeof(float), nll.size(), f);
            fclose(f);
            double s = 0;
            for (float x : nll) { s += x; }
            nll_sum += s; nll_n += nll.size(); top1_n += top1;
            fprintf(stderr, "score %s: n = %zu, mean nll = %.6f, top1 = %.4f\n", pfx.c_str(), nll.size(), s / nll.size(), (double) top1 / nll.size());
        }
        if (!no_kv) {
            const size_t sz = llama_state_seq_get_size(ctx, 0);
            std::vector<uint8_t> buf(sz);
            llama_state_seq_get_data(ctx, buf.data(), sz, 0);
            FILE * f = fopen((pfx + ".kv").c_str(), "wb");
            fwrite(buf.data(), 1, buf.size(), f);
            fclose(f);
        }
        fprintf(stderr, "ok: %s n = %d, keep = %d, vocab = %d, decode %.1f ms\n", pfx.c_str(), n, keep, n_vocab, t_ms);
    }
    if (nll_n > 0) {
        fprintf(stderr, "score total: n = %ld, mean nll = %.6f, ppl = %.4f, top1 = %.4f\n", nll_n, nll_sum / nll_n, std::exp(nll_sum / nll_n), (double) top1_n / nll_n);
    }

    llama_batch_free(batch);
    llama_free(ctx);
#ifndef STOCK
    llama_ridgefill_free(rf);
#endif
    llama_model_free(model);
    return 0;
}
