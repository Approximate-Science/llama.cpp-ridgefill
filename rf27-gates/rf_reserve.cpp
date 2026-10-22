// Load-time compute-buffer reserve, no decode: creates a context (n_ctx, n_batch, n_ubatch, F16 KV, flash attention
// on or off), prints the stock reserve, attaches the projector (llama_set_ridgefill reserves every RidgeFill graph)
// and prints the reserve again. Usage:
//   rf_reserve --model M --projector P --ctx N [--batch 2048] [--ubatch 512] [--fa 1] [--gpu-layers 0]
// Output: one JSON line {"ctx":N, "stock_mib":{buft:MiB}, "ridgefill_mib":{buft:MiB}, "refused":bool}
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

static std::map<std::string, double> * g_sizes = nullptr;

static void log_cb(ggml_log_level level, const char * text, void *) {
    // "sched_reserve:  <buft> compute buffer size =  X MiB": the last reserve wins
    const char * p = strstr(text, "sched_reserve:");
    if (p && g_sizes) {
        char buft[128]; double mib;
        if (sscanf(p, "sched_reserve: %127s compute buffer size = %lf MiB", buft, &mib) == 2) {
            (*g_sizes)[buft] = mib;
        }
    }
    if (level >= GGML_LOG_LEVEL_WARN || strstr(text, "RidgeFill graph")) {
        fputs(text, stderr);
    }
}

static std::string json(const std::map<std::string, double> & m) {
    std::string s = "{";
    for (const auto & [k, v] : m) {
        s += (s.size() > 1 ? "," : "") + std::string("\"") + k + "\":" + std::to_string(v);
    }
    return s + "}";
}

int main(int argc, char ** argv) {
    std::string model_path, projector;
    int ctx = 0, batch = 2048, ubatch = 512, fa = 1, ngl = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if      (a == "--model")      { model_path = v; }
        else if (a == "--projector")  { projector = v; }
        else if (a == "--ctx")        { ctx = std::stoi(v); }
        else if (a == "--batch")      { batch = std::stoi(v); }
        else if (a == "--ubatch")     { ubatch = std::stoi(v); }
        else if (a == "--fa")         { fa = std::stoi(v); }
        else if (a == "--gpu-layers") { ngl = std::stoi(v); }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || projector.empty() || ctx <= 0) {
        fprintf(stderr, "usage: %s --model M --projector P --ctx N [--batch B] [--ubatch U] [--fa 0|1] [--gpu-layers L]\n", argv[0]);
        return 2;
    }
    std::map<std::string, double> sizes;
    g_sizes = &sizes;
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    std::unique_ptr<llama_ridgefill, decltype(&llama_ridgefill_free)> rf(llama_ridgefill_init(model.get(), projector.c_str()), llama_ridgefill_free);
    if (!rf) { fprintf(stderr, "projector load failed\n"); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = ctx; cp.n_batch = batch; cp.n_ubatch = ubatch; cp.n_seq_max = 1;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    std::unique_ptr<llama_context, decltype(&llama_free)> lctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!lctx) { fprintf(stderr, "context creation failed\n"); return 1; }
    const auto stock = sizes;
    const bool refused = llama_set_ridgefill(lctx.get(), rf.get()) < 0;
    printf("{\"ctx\":%d,\"batch\":%d,\"ubatch\":%d,\"fa\":%d,\"stock_mib\":%s,\"ridgefill_mib\":%s,\"refused\":%s}\n",
            ctx, batch, ubatch, fa, json(stock).c_str(), json(sizes).c_str(), refused ? "true" : "false");
    lctx.reset();
    llama_backend_free();
    return refused ? 1 : 0;
}
