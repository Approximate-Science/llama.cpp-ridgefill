#include "llama-ridgefill-sweep.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-model.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

size_t llama_ridgefill_sweep_buffer::required_bytes(int32_t n_embd, int32_t n_tokens, ggml_type type, size_t max_bytes) {
    if (n_embd <= 0 || n_tokens <= 0) {
        throw std::runtime_error("RidgeFill quality+qattn: buffer dimensions must be positive");
    }
    if (type != GGML_TYPE_F16 && type != GGML_TYPE_BF16 && type != GGML_TYPE_F32) {
        throw std::runtime_error("RidgeFill quality+qattn: buffer type must be f16, bf16, or f32");
    }

    const size_t element_size = ggml_type_size(type);
    const size_t limit = std::min(max_bytes, size_t(std::numeric_limits<int64_t>::max()));
    if (size_t(n_embd) > limit / element_size) {
        throw std::runtime_error("RidgeFill quality+qattn: embedding row exceeds the sweep buffer byte limit");
    }
    const size_t row_size = size_t(n_embd) * element_size;
    if (size_t(n_tokens) > limit / row_size) {
        throw std::runtime_error("RidgeFill quality+qattn: prompt exceeds the sweep buffer byte limit (maximum " +
                std::to_string(limit / row_size) + " tokens)");
    }
    return row_size * size_t(n_tokens);
}

llama_ridgefill_sweep_buffer::llama_ridgefill_sweep_buffer(
        ggml_backend_buffer_type_t buft,
        int32_t n_embd,
        int32_t n_tokens,
        ggml_type type,
        size_t max_bytes) {
    required_bytes(n_embd, n_tokens, type, max_bytes);
    if (!buft) {
        throw std::runtime_error("RidgeFill quality+qattn: missing layer buffer type");
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx.reset(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("RidgeFill quality+qattn: failed to create sweep buffer context");
    }
    data = ggml_new_tensor_2d(ctx.get(), type, n_embd, n_tokens);
    ggml_set_name(data, "ridgefill_sweep_buffer");
    buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
    if (!buf) {
        throw std::runtime_error("RidgeFill quality+qattn: failed to allocate sweep buffer on the layer device");
    }
}

size_t llama_ridgefill_sweep_buffer::size_bytes() const {
    return ggml_nbytes(data);
}

ggml_tensor * llama_ridgefill_sweep_buffer::slice(ggml_context * graph_ctx, int32_t first, int32_t count) const {
    if (!graph_ctx || first < 0 || count <= 0 || first > data->ne[1] || count > data->ne[1] - first) {
        throw std::runtime_error("RidgeFill quality+qattn: sweep buffer slice is outside the prompt");
    }
    return ggml_view_2d(graph_ctx, data, data->ne[0], count, data->nb[1], size_t(first) * data->nb[1]);
}

ggml_tensor * llama_ridgefill_sweep_buffer::capture(
        ggml_context * graph_ctx, ggml_cgraph * gf, ggml_tensor * src, int32_t first) const {
    if (!gf || !src || src->ne[0] != data->ne[0] || src->ne[1] > std::numeric_limits<int32_t>::max() ||
            src->ne[2] != 1 || src->ne[3] != 1 ||
            (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16)) {
        throw std::runtime_error("RidgeFill quality+qattn: capture requires floating-point [n_embd, n_tokens] rows");
    }
    ggml_tensor * dst = slice(graph_ctx, first, int32_t(src->ne[1]));
    ggml_tensor * copy = ggml_cpy(graph_ctx, src, dst);
    ggml_build_forward_expand(gf, copy);
    return copy;
}

ggml_tensor * llama_ridgefill_sweep_mass(
        ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, int32_t window_end, float scale) {
    if (!ctx || !q || !k || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 ||
            q->ne[0] <= 0 || q->ne[1] <= 0 || q->ne[2] <= 0 || k->ne[1] <= 0 || k->ne[2] <= 0 ||
            q->ne[0] != k->ne[0] || q->ne[3] != 1 || k->ne[3] != 1 ||
            q->ne[2] % k->ne[2] != 0 || window_end < q->ne[1] || window_end > k->ne[1] ||
            !std::isfinite(scale) || scale <= 0.0f) {
        throw std::runtime_error("RidgeFill quality+qattn: mass requires FP32 queries and keys, grouped heads, and a valid causal window");
    }

    const int64_t n_keys = k->ne[1];
    const int64_t n_queries = q->ne[1];
    const int64_t n_kv_heads = k->ne[2];
    const int64_t rep = q->ne[2] / n_kv_heads;

    // Flatten each KV-head group's query rows before reducing, as window_mass does.
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    ggml_prec_set_acc(scores, GGML_PREC_F32);
    scores = ggml_diag_mask_inf(ctx, scores, window_end - int32_t(n_queries));
    ggml_tensor * probabilities = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
    probabilities = ggml_reshape_3d(ctx, probabilities, n_keys, n_queries * rep, n_kv_heads);
    probabilities = ggml_cont(ctx, ggml_permute(ctx, probabilities, 1, 0, 2, 3));
    ggml_tensor * mass = ggml_sum_rows(ctx, probabilities);
    return ggml_reshape_2d(ctx, mass, n_keys, n_kv_heads);
}

ggml_tensor * build_ridgefill_question_mass(
        llm_graph_context & g, const llama_layer & layer, ggml_tensor * window, ggml_tensor * positions,
        ggml_tensor * keys, int32_t window_end, int32_t il, ggml_tensor * key_rotation) {
    if (!window || window->type != GGML_TYPE_F32 || window->ne[0] != g.n_embd || window->ne[1] != g.n_tokens ||
            window->ne[2] != 1 || window->ne[3] != 1 || !positions || positions->type != GGML_TYPE_I32 ||
            ggml_nelements(positions) != 4*g.n_tokens || !keys || keys->ne[0] != g.n_embd_head_k ||
            keys->ne[2] != g.n_head_kv || g.arch != LLM_ARCH_QWEN35 ||
            il < 0 || il >= g.n_layer || g.hparams.is_recr(il)) {
        throw std::runtime_error("RidgeFill quality+qattn: question graph requires a qwen35 attention layer and its window inputs");
    }
    if (key_rotation && (key_rotation->type != GGML_TYPE_F32 || key_rotation->ne[0] <= 0 ||
            key_rotation->ne[0] != key_rotation->ne[1] || key_rotation->ne[2] != 1 || key_rotation->ne[3] != 1 ||
            g.n_embd_head_k % key_rotation->ne[0] != 0)) {
        throw std::runtime_error("RidgeFill quality+qattn: invalid key-cache rotation");
    }

    ggml_tensor * normalized = g.build_norm(window, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    auto qkv = g.build_qkv(layer, normalized, 2*g.n_embd_head_k, g.n_head,
            g.n_embd_head_k, g.n_head_kv, g.n_embd_head_v, g.n_head_kv, il, false);
    ggml_tensor * q = ggml_view_3d(g.ctx0, qkv.q, g.n_embd_head_k, g.n_head, g.n_tokens,
            2*g.n_embd_head_k*ggml_element_size(qkv.q), qkv.q->nb[1], 0);
    q = g.build_norm(q, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    int sections[4];
    std::copy(std::begin(g.hparams.rope_sections), std::end(g.hparams.rope_sections), sections);
    q = ggml_rope_multi(g.ctx0, q, positions, nullptr, g.n_rot, sections, g.rope_type, g.n_ctx_orig,
            g.freq_base, g.freq_scale, g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);
    if (key_rotation) {
        q = llama_mul_mat_hadamard(g.ctx0, q, key_rotation);
    }
    q = ggml_permute(g.ctx0, q, 0, 2, 1, 3);
    const float scale = g.hparams.f_attention_scale == 0.0f ? 1.0f / std::sqrt(float(g.n_embd_head_k)) : g.hparams.f_attention_scale;
    return llama_ridgefill_sweep_mass(g.ctx0, q, keys, window_end, scale);
}

static void ridgefill_sweep_check_sequence(const llama_kv_cache & cache, int32_t seq_id) {
    if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ ||
            (cache.get_n_stream() > 1 && uint32_t(seq_id) >= cache.get_n_stream())) {
        throw std::runtime_error("RidgeFill quality+qattn: sequence is outside the attention cache");
    }
}

std::vector<int32_t> llama_ridgefill_sweep_key_rows(const llama_kv_cache & cache, int32_t seq_id, int32_t n_tokens) {
    ridgefill_sweep_check_sequence(cache, seq_id);
    if (n_tokens <= 0 || uint32_t(n_tokens) > cache.get_size() || cache.get_has_shift()) {
        throw std::runtime_error("RidgeFill quality+qattn: invalid prompt length or pending attention-cache shift");
    }
    const auto & positions = cache.get_cells(seq_id).seq_pos_get(seq_id);
    if (positions.size() != size_t(n_tokens)) {
        throw std::runtime_error("RidgeFill quality+qattn: attention cache must contain exactly the prompt rows");
    }
    std::vector<int32_t> rows;
    rows.reserve(n_tokens);
    for (const auto & entry : positions) {
        if (entry.first != int32_t(rows.size()) || entry.second >= cache.get_size() ||
                entry.second > uint32_t(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("RidgeFill quality+qattn: attention cache has missing, repeated, or invalid prompt positions");
        }
        rows.push_back(int32_t(entry.second));
    }
    return rows;
}

ggml_tensor * build_ridgefill_sweep_keys(
        ggml_context * ctx, const llama_kv_cache & cache, int32_t seq_id, int32_t il, int32_t head_dim, ggml_tensor * rows) {
    ridgefill_sweep_check_sequence(cache, seq_id);
    if (!ctx || !rows || rows->type != GGML_TYPE_I32 || rows->ne[0] <= 0 || rows->ne[0] > cache.get_size() ||
            rows->ne[1] != 1 || rows->ne[2] != 1 || rows->ne[3] != 1 || head_dim <= 0) {
        throw std::runtime_error("RidgeFill quality+qattn: invalid key-gather shape or head dimension");
    }
    const auto layers = cache.get_layer_ids();
    if (std::find(layers.begin(), layers.end(), uint32_t(il)) == layers.end()) {
        throw std::runtime_error("RidgeFill quality+qattn: selected layer has no attention key cache");
    }
    ggml_tensor * storage = cache.get_k_storage(il);
    const uint32_t stream = cache.get_stream(seq_id);
    if (storage->ne[0] % head_dim != 0 || storage->ne[1] != cache.get_size() || stream >= storage->ne[2] || storage->ne[3] != 1) {
        throw std::runtime_error("RidgeFill quality+qattn: unexpected attention key-cache layout");
    }
    ggml_tensor * keys = ggml_view_2d(ctx, storage, storage->ne[0], storage->ne[1], storage->nb[1], stream*storage->nb[2]);
    keys = ggml_get_rows(ctx, keys, rows);
    keys = ggml_reshape_3d(ctx, keys, head_dim, storage->ne[0] / head_dim, rows->ne[0]);
    return ggml_permute(ctx, keys, 0, 2, 1, 3);
}

std::vector<int32_t> llama_ridgefill_sweep_top_rows(const std::vector<double> & mass, int32_t prefix, double share) {
    if (prefix < 0 || size_t(prefix) > mass.size() || !std::isfinite(share) || share < 0.0 || share > 1.0) {
        throw std::runtime_error("RidgeFill quality+qattn: invalid prefix or qattn share");
    }
    for (int32_t i = 0; i < prefix; ++i) {
        if (!std::isfinite(mass[i]) || mass[i] < 0.0) {
            throw std::runtime_error("RidgeFill quality+qattn: non-finite or negative question mass");
        }
    }

    // Python round uses ties to even; do not depend on the process rounding mode.
    const double budget = share * prefix;
    int32_t count = int32_t(std::floor(budget));
    const double fraction = budget - count;
    if (fraction > 0.5 || (fraction == 0.5 && (count & 1))) {
        ++count;
    }
    std::vector<int32_t> rows(prefix);
    std::iota(rows.begin(), rows.end(), 0);
    std::partial_sort(rows.begin(), rows.begin() + count, rows.end(), [&](int32_t a, int32_t b) {
        return mass[a] > mass[b] || (mass[a] == mass[b] && a < b);
    });
    rows.resize(count);
    std::sort(rows.begin(), rows.end());
    return rows;
}
