// Self-test of llama_ridgefill_graph_peaks, the check behind LLAMA_RIDGEFILL_GRAPH_CHECK (tests/ridgefill/
// check_compute_buffer.py): at Qwen3.8-27B's attention shape (24 query heads, 4 key heads, head dim 256) and a
// 512-row ubatch, over 8K and 32K cells, the S1 mass blocks as bc3a6af85 built them (independent blocks summed at the
// end: a reordering backend keeps them all alive, 1.8 GiB at 32K on Vulkan) must be flagged by the gate's rule (the
// any-order peak may grow by at most 16 MiB + 8 B x ubatch per added cell), and the chained blocks of
// llama_ridgefill_attn_mass must not be. No compute: graphs are built in a no_alloc context.
// Build: c++ -std=c++17 -O2 -Iinclude -Isrc -Iggml/include tests/ridgefill/test_graph_peaks.cpp -Lbuild/bin -lllama -lggml -lggml-base
#include "llama-ridgefill.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>

static const int64_t D = 256, N_HEAD = 24, N_HEAD_KV = 4, N_Q = 512;
static int64_t N_KV = 32768;
static const int64_t BLOCK = int64_t(1) << 24; // bc3a6af85's LLAMA_RIDGEFILL_MASS_BLOCK

struct inputs {
    ggml_tensor * q_base; // [d, n_head, n_q]
    ggml_tensor * k;      // [d, n_kv, n_head_kv] (a permuted cache view)
    ggml_tensor * mask;   // [n_kv, n_q]
    ggml_tensor * ones;   // [1, n_q * n_head]
};

static inputs make_inputs(ggml_context * ctx) {
    inputs in;
    in.q_base = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N_HEAD, N_Q);
    ggml_tensor * cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, N_HEAD_KV, N_KV);
    in.k = ggml_permute(ctx, cache, 0, 2, 1, 3);
    in.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N_KV, N_Q);
    in.ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, N_Q*N_HEAD);
    return in;
}

// bc3a6af85's llama_ridgefill_attn_mass (ones reduction): the blocks depend on the inputs only
static ggml_tensor * unchained_mass(ggml_context * ctx, const inputs & in) {
    ggml_tensor * q = ggml_permute(ctx, in.q_base, 0, 2, 1, 3);
    const int64_t rep = N_HEAD/N_HEAD_KV;
    int64_t hb, qb;
    llama_ridgefill_attn_mass_split(N_KV, N_Q, rep, BLOCK, hb, qb);
    ggml_tensor * total = nullptr;
    for (int64_t g = 0; g < N_HEAD_KV; ++g) {
        ggml_tensor * kg = ggml_view_3d(ctx, in.k, D, N_KV, 1, in.k->nb[1], in.k->nb[2], g*in.k->nb[2]);
        ggml_tensor * acc = nullptr;
        for (int64_t h0 = g*rep; h0 < (g + 1)*rep; h0 += hb) {
            for (int64_t q0 = 0; q0 < N_Q; q0 += qb) {
                const int64_t nq = std::min(qb, N_Q - q0);
                ggml_tensor * qv = ggml_view_3d(ctx, q, D, nq, hb, q->nb[1], q->nb[2], q0*q->nb[1] + h0*q->nb[2]);
                ggml_tensor * kq = ggml_mul_mat(ctx, kg, qv);
                ggml_tensor * mv = ggml_view_2d(ctx, in.mask, N_KV, nq, in.mask->nb[1], q0*in.mask->nb[1]);
                ggml_tensor * p = ggml_soft_max_ext(ctx, kq, mv, 0.0625f, 0.0f);
                ggml_tensor * m = ggml_out_prod(ctx, ggml_reshape_2d(ctx, p, N_KV, nq*hb), ggml_view_2d(ctx, in.ones, 1, nq*hb, in.ones->nb[1], 0));
                m = ggml_reshape_1d(ctx, m, N_KV);
                acc = acc ? ggml_add(ctx, acc, m) : m;
            }
        }
        total = total ? ggml_add(ctx, total, acc) : acc;
    }
    return total;
}

static double any_order_mib(bool chained, int64_t n_kv, double & in_order_mib) {
    N_KV = n_kv;
    ggml_init_params ip = { 64*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    inputs in = make_inputs(ctx);
    ggml_tensor * out;
    if (chained) {
        ggml_tensor * dep = nullptr;
        out = llama_ridgefill_attn_mass(ctx, in.q_base, 0, N_HEAD, in.k, in.mask, 0, 0.0625f, in.ones, BLOCK, &dep);
    } else {
        out = unchained_mass(ctx, in);
    }
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, out);
    size_t in_order = 0, any_order = 0;
    llama_ridgefill_graph_peaks(gf, in_order, any_order);
    ggml_free(ctx);
    in_order_mib = in_order/1048576.0;
    return any_order/1048576.0;
}

static bool check(const char * name, bool chained, bool expect_flagged) {
    double io8, io32;
    const double a8 = any_order_mib(chained, 8192, io8), a32 = any_order_mib(chained, 32768, io32);
    const double limit = 16.0 + 8.0*N_Q*(32768 - 8192)/1048576.0;
    const bool flagged = a32 - a8 > limit;
    const bool ok = flagged == expect_flagged;
    printf("{\"check\": \"%s\", \"result\": \"%s\", \"any_order_mib\": [%.2f, %.2f], \"in_order_mib\": [%.2f, %.2f], "
           "\"growth_mib\": %.2f, \"limit_mib\": %.2f, \"flagged\": %s}\n", name, ok ? "PASS" : "FAIL", a8, a32, io8, io32,
           a32 - a8, limit, flagged ? "true" : "false");
    return ok;
}

int main() {
    bool ok = true;
    ok &= check("unchained-bc3a6af85-blocks-flagged", false, true);
    ok &= check("chained-blocks-not-flagged", true, false);
    printf("{\"test\": \"graph-peaks\", \"result\": \"%s\"}\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
