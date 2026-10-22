// Self-test of llama_ridgefill_graph_peaks, the check behind LLAMA_RIDGEFILL_GRAPH_CHECK (tests/ridgefill/
// check_compute_buffer.py): at Qwen3.8-27B's attention shape (24 query heads, 4 key heads, head dim 256) and a
// 512-row ubatch, over 8K and 32K cells, the S1 mass blocks as bc3a6af85 built them (independent blocks summed at the
// end: a reordering backend keeps them all alive, 1.8 GiB at 32K on Vulkan) must be flagged by the gate's rule (the
// any-order peak may grow by at most 16 MiB + 8 B x ubatch per added cell); chained blocks that still view the mask
// and ones inputs per block as 848c757ad did (one input copy per view on a non-CPU backend: 2.7 GiB at 9K on Vulkan)
// must be flagged by the same rule on their input-view bytes; chained blocks without per-block input views must not
// be flagged. No compute: graphs are built in a no_alloc context.
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
    ggml_set_input(in.mask);
    ggml_set_input(in.ones);
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

// the S1 probe's mass structure: 12 late attention layers x 24 query heads, one chained block per head with all of the
// ubatch's queries; the mask and ones either viewed per block (848c757ad) or shared by every block (the fix)
static ggml_tensor * probe_mass(ggml_context * ctx, const inputs & in, bool per_block_views) {
    const int64_t rep = N_HEAD/N_HEAD_KV;
    ggml_tensor * dep = nullptr;
    ggml_tensor * total = nullptr;
    for (int il = 0; il < 12; ++il) {
        for (int64_t h = 0; h < N_HEAD; ++h) {
            ggml_tensor * kg = ggml_view_3d(ctx, in.k, D, N_KV, 1, in.k->nb[1], in.k->nb[2], (h/rep)*in.k->nb[2]);
            ggml_tensor * qp = ggml_permute(ctx, llama_ridgefill_follow(ctx, in.q_base, dep), 0, 2, 1, 3);
            ggml_tensor * qv = ggml_view_3d(ctx, qp, D, N_Q, 1, qp->nb[1], qp->nb[2], h*qp->nb[2]);
            ggml_tensor * mask = per_block_views ? ggml_view_2d(ctx, in.mask, N_KV, N_Q, in.mask->nb[1], 0) : in.mask;
            ggml_tensor * ones = ggml_view_2d(ctx, in.ones, 1, N_Q, in.ones->nb[1], 0);
            ggml_tensor * p = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, kg, qv), mask, 0.0625f, 0.0f);
            ggml_tensor * m = ggml_reshape_1d(ctx, ggml_out_prod(ctx, p, ones), N_KV);
            dep = m;
            total = total ? ggml_add(ctx, total, m) : m;
        }
    }
    return total;
}

enum variant { UNCHAINED, PROBE_INPUT_VIEWS, PROBE_SHARED_INPUTS };

// any-order peak and input-view bytes (MiB) of a variant's graph over n_kv cells
static void peaks(variant v, int64_t n_kv, double & in_order_mib, double & any_order_mib, double & input_views_mib) {
    N_KV = n_kv;
    ggml_init_params ip = { 64*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    inputs in = make_inputs(ctx);
    ggml_tensor * out;
    ggml_tensor * dep = nullptr;
    if (v == UNCHAINED) {
        out = unchained_mass(ctx, in);
    } else {
        out = probe_mass(ctx, in, v == PROBE_INPUT_VIEWS);
    }
    GGML_UNUSED(dep);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, out);
    size_t in_order = 0, any_order = 0, input_views = 0;
    llama_ridgefill_graph_peaks(gf, in_order, any_order, input_views);
    ggml_free(ctx);
    in_order_mib = in_order/1048576.0;
    any_order_mib = any_order/1048576.0;
    input_views_mib = input_views/1048576.0;
}

static bool check(const char * name, variant v, bool expect_flagged) {
    double io8, ao8, iv8, io32, ao32, iv32;
    peaks(v, 8192, io8, ao8, iv8);
    peaks(v, 32768, io32, ao32, iv32);
    const double limit = 16.0 + 8.0*N_Q*(32768 - 8192)/1048576.0;
    const bool flagged = ao32 - ao8 > limit || iv32 - iv8 > limit;
    const bool ok = flagged == expect_flagged;
    printf("{\"check\": \"%s\", \"result\": \"%s\", \"any_order_mib\": [%.2f, %.2f], \"in_order_mib\": [%.2f, %.2f], "
           "\"input_views_mib\": [%.2f, %.2f], \"limit_mib\": %.2f, \"flagged\": %s}\n", name, ok ? "PASS" : "FAIL",
           ao8, ao32, io8, io32, iv8, iv32, limit, flagged ? "true" : "false");
    return ok;
}

int main() {
    bool ok = true;
    ok &= check("unchained-bc3a6af85-blocks-flagged", UNCHAINED, true);
    ok &= check("probe-per-block-mask-views-848c757ad-flagged", PROBE_INPUT_VIEWS, true);
    ok &= check("probe-shared-inputs-chained-not-flagged", PROBE_SHARED_INPUTS, false);
    printf("{\"test\": \"graph-peaks\", \"result\": \"%s\"}\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
