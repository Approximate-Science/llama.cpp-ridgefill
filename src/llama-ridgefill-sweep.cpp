#include "llama-ridgefill-sweep.h"

#include "llama-graph.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama-memory-hybrid.h"
#include "llama-ridgefill.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

std::vector<double> llama_ridgefill_sweep_question_mass(
        llama_context & ctx, int32_t seq_id, int32_t n_tokens, int32_t window_end,
        const std::vector<int32_t> & layers, const std::vector<ggml_tensor *> & windows) {
    const auto & model = ctx.get_model();
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(ctx.get_memory());
    if (!hybrid || layers.empty() || layers.size() != windows.size() || n_tokens <= 0 || window_end <= 0 || window_end > n_tokens) {
        throw std::runtime_error("RidgeFill quality+qattn: invalid question window or hybrid cache");
    }
    const size_t width = model.hparams.n_embd;
    const size_t count = windows[0] ? windows[0]->ne[1] : 0;
    if (count == 0 || count > size_t(window_end)) {
        throw std::runtime_error("RidgeFill quality+qattn: empty or oversized question window");
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i] < 0 || layers[i] >= (int32_t) model.hparams.n_layer() || model.hparams.is_recr(layers[i]) ||
                !windows[i] || !windows[i]->buffer || windows[i]->type != GGML_TYPE_F32 ||
                windows[i]->ne[0] != int64_t(width) || windows[i]->ne[1] != int64_t(count) ||
                windows[i]->ne[2] != 1 || windows[i]->ne[3] != 1 || !ggml_is_contiguous(windows[i])) {
            throw std::runtime_error("RidgeFill quality+qattn: question layer/window shape mismatch");
        }
    }
    ctx.synchronize();
    auto & cache = *hybrid->get_mem_attn();
    const auto rows = llama_ridgefill_sweep_key_rows(cache, seq_id, n_tokens);
    std::vector<ggml_backend_t> backends;
    ggml_backend_t cpu = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(ctx.get_sched()); ++i) {
        auto * backend = ggml_backend_sched_get_backend(ctx.get_sched(), i);
        backends.push_back(backend);
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu = backend;
        }
    }
    // the window's Q per layer, then per KV head its key gather and score blocks
    int64_t hb = 1, qb = 1;
    llama_ridgefill_attn_mass_split(n_tokens, count, model.hparams.n_head(layers[0]) / model.hparams.n_head_kv(layers[0]),
            LLAMA_RIDGEFILL_QMASS_BLOCK, hb, qb);
    const int64_t blocks = (model.hparams.n_head(layers[0]) / model.hparams.n_head_kv(layers[0]) / hb) * ((int64_t(count) + qb - 1) / qb);
    const size_t graph_size = layers.size() * size_t(512 + model.hparams.n_head_kv(layers[0]) * (16 + 12 * blocks));
    llm_graph_result result(graph_size);
    llama_adapter_loras loras;
    llm_graph_params params = {};
    params.arch = model.arch;
    params.hparams = model.hparams;
    params.cparams = ctx.get_cparams();
    params.ubatch.n_tokens = params.ubatch.n_seq_tokens = count;
    params.ubatch.n_seqs = params.ubatch.n_seqs_unq = 1;
    params.loras = &loras;
    params.backend_cpu = cpu;
    params.res = &result;
    llm_graph_context g(params);
    auto * row_input = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, n_tokens);
    auto * positions = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, count * model.hparams.n_pos_per_embd());
    ggml_set_input(row_input);
    ggml_set_input(positions);
    auto * rotation = cache.build_input_k_rot(g.ctx0);
    std::vector<ggml_tensor *> outputs;
    for (size_t i = 0; i < layers.size(); ++i) {
        const int32_t il = layers[i];
        auto * input = windows[i];
        // the keys of one KV head at a time (F32, prompt order): the gather is never all heads x all prompt rows
        auto * mass = build_ridgefill_question_mass_by_head(g, model.layers[il], input, positions, cache, seq_id, row_input,
                window_end, il, rotation);
        ggml_set_output(mass);
        ggml_build_forward_expand(g.gf, mass);
        outputs.push_back(mass);
    }
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(), nullptr, backends.size(), graph_size, false, true));
    if (!sched || !ggml_backend_sched_alloc_graph(sched.get(), g.gf)) {
        throw std::runtime_error("RidgeFill quality+qattn: question graph allocation failed");
    }
    for (auto * backend : backends) {
        const size_t size = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        if (size > 0) {
            LLAMA_LOG_INFO("%s: RidgeFill question-mass graph: %s compute buffer %.4f MiB (%d keys)\n", __func__,
                    ggml_backend_name(backend), size / (1024.0*1024.0), n_tokens);
        }
    }
    ggml_backend_tensor_set(row_input, rows.data(), 0, rows.size() * sizeof(int32_t));
    std::vector<int32_t> pos(ggml_nelements(positions));
    for (size_t i = 0; i < pos.size(); ++i) {
        pos[i] = window_end - count + i % count;
    }
    ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    if (rotation) {
        cache.set_input_k_rot(rotation);
    }
    if (ggml_backend_sched_graph_compute(sched.get(), g.gf) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("RidgeFill quality+qattn: question graph execution failed");
    }
    std::vector<double> mass(n_tokens, 0.0);
    for (auto * output : outputs) {
        std::vector<float> groups(ggml_nelements(output));
        ggml_backend_tensor_get(output, groups.data(), 0, groups.size() * sizeof(float));
        for (size_t i = 0; i < groups.size(); ++i) {
            if (!std::isfinite(groups[i])) {
                throw std::runtime_error("RidgeFill quality+qattn: question mass is non-finite");
            }
            mass[i % n_tokens] += groups[i];
        }
    }
    return mass;
}

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

    const int64_t n_queries = q->ne[1];

    // Flatten each KV-head group's query rows before reducing, as window_mass does. The scores are cut per KV head
    // (and, beyond LLAMA_RIDGEFILL_QMASS_BLOCK scores, into query ranges), so memory does not grow as keys x queries x
    // heads; when a KV head's block fits, the sums are those of the unsliced softmax-then-sum_rows, bit for bit.
    return llama_ridgefill_attn_mass(ctx, q, k, nullptr, window_end - int32_t(n_queries), scale, nullptr,
            LLAMA_RIDGEFILL_QMASS_BLOCK);
}

// the window's queries of layer il: [head_dim, n_window, n_head], post-RoPE (and rotated as the cached keys)
static ggml_tensor * ridgefill_question_queries(llm_graph_context & g, const llama_layer & layer, ggml_tensor * window,
        ggml_tensor * positions, int32_t il, ggml_tensor * key_rotation) {
    if (!window || window->type != GGML_TYPE_F32 || window->ne[0] != g.n_embd || window->ne[1] != g.n_tokens ||
            window->ne[2] != 1 || window->ne[3] != 1 || !positions || positions->type != GGML_TYPE_I32 ||
            ggml_nelements(positions) != 4*g.n_tokens || g.arch != LLM_ARCH_QWEN35 ||
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
    return q;
}

ggml_tensor * build_ridgefill_question_mass(
        llm_graph_context & g, const llama_layer & layer, ggml_tensor * window, ggml_tensor * positions,
        ggml_tensor * keys, int32_t window_end, int32_t il, ggml_tensor * key_rotation) {
    if (!keys || keys->ne[0] != g.n_embd_head_k || keys->ne[2] != g.n_head_kv) {
        throw std::runtime_error("RidgeFill quality+qattn: question graph requires a qwen35 attention layer and its window inputs");
    }
    ggml_tensor * q = ridgefill_question_queries(g, layer, window, positions, il, key_rotation);
    const float scale = g.hparams.f_attention_scale == 0.0f ? 1.0f / std::sqrt(float(g.n_embd_head_k)) : g.hparams.f_attention_scale;
    return llama_ridgefill_sweep_mass(g.ctx0, q, keys, window_end, scale);
}

static ggml_tensor * ridgefill_sweep_keys_head(ggml_context * ctx, const llama_kv_cache & cache, int32_t seq_id, int32_t il,
        int32_t head_dim, ggml_tensor * rows, int64_t head);

ggml_tensor * build_ridgefill_question_mass_by_head(
        llm_graph_context & g, const llama_layer & layer, ggml_tensor * window, ggml_tensor * positions,
        const llama_kv_cache & cache, int32_t seq_id, ggml_tensor * rows, int32_t window_end, int32_t il, ggml_tensor * key_rotation) {
    // one KV head at a time: the head's keys gathered alone and its query heads' mass (llama_ridgefill_sweep_mass's
    // reduction), concatenated as [n_tokens, n_kv_heads] (the inputs are checked as by build_ridgefill_question_mass)
    ggml_tensor * q = ridgefill_question_queries(g, layer, window, positions, il, key_rotation);
    const float scale = g.hparams.f_attention_scale == 0.0f ? 1.0f / std::sqrt(float(g.n_embd_head_k)) : g.hparams.f_attention_scale;
    const int64_t rep = g.n_head / g.n_head_kv;
    ggml_tensor * mass = nullptr;
    for (int64_t h = 0; h < g.n_head_kv; ++h) {
        ggml_tensor * k = ridgefill_sweep_keys_head(g.ctx0, cache, seq_id, il, g.n_embd_head_k, rows, h);
        ggml_tensor * qh = ggml_view_3d(g.ctx0, q, q->ne[0], q->ne[1], rep, q->nb[1], q->nb[2], h*rep*q->nb[2]);
        ggml_tensor * m = llama_ridgefill_sweep_mass(g.ctx0, qh, k, window_end, scale);
        mass = mass ? ggml_concat(g.ctx0, mass, m, 1) : m;
    }
    return mass;
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

// the validated key storage of the sequence's stream at layer il: [n_embd_k_gqa, kv_size]
static ggml_tensor * ridgefill_sweep_key_storage(
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
    return ggml_view_2d(ctx, storage, storage->ne[0], storage->ne[1], storage->nb[1], stream*storage->nb[2]);
}

ggml_tensor * build_ridgefill_sweep_keys(
        ggml_context * ctx, const llama_kv_cache & cache, int32_t seq_id, int32_t il, int32_t head_dim, ggml_tensor * rows) {
    ggml_tensor * keys = ridgefill_sweep_key_storage(ctx, cache, seq_id, il, head_dim, rows);
    keys = ggml_get_rows(ctx, keys, rows);
    keys = ggml_reshape_3d(ctx, keys, head_dim, keys->ne[0] / head_dim, rows->ne[0]);
    return ggml_permute(ctx, keys, 0, 2, 1, 3);
}

// the keys of one KV head, F32 [head_dim, n_rows, 1] in prompt order (the same values as that head of
// build_ridgefill_sweep_keys)
static ggml_tensor * ridgefill_sweep_keys_head(ggml_context * ctx, const llama_kv_cache & cache, int32_t seq_id, int32_t il,
        int32_t head_dim, ggml_tensor * rows, int64_t head) {
    ggml_tensor * all = ridgefill_sweep_key_storage(ctx, cache, seq_id, il, head_dim, rows);
    // head `head` of every cell: a [head_dim, kv_size] view at its column offset (rows keep the cell stride)
    ggml_tensor * keys = ggml_view_2d(ctx, all, head_dim, all->ne[1], all->nb[1], head*ggml_row_size(all->type, head_dim));
    keys = ggml_get_rows(ctx, keys, rows);
    return ggml_reshape_3d(ctx, keys, head_dim, rows->ne[0], 1);
}

std::vector<int32_t> llama_ridgefill_sweep_top_rows(const std::vector<double> & mass, int32_t prefix, double share) {
    if (prefix < 0 || size_t(prefix) > mass.size() || !std::isfinite(share) || share < 0.0 || share > 1.0) {
        throw std::runtime_error("RidgeFill quality+qattn: invalid prefix or qattn share");
    }
    // Python round uses ties to even; do not depend on the process rounding mode.
    const double budget = share * prefix;
    int32_t count = int32_t(std::floor(budget));
    const double fraction = budget - count;
    if (fraction > 0.5 || (fraction == 0.5 && (count & 1))) {
        ++count;
    }
    return llama_ridgefill_sweep_top_k(mass, prefix, count);
}

std::vector<int32_t> llama_ridgefill_sweep_top_k(const std::vector<double> & mass, int32_t prefix, int32_t count) {
    if (prefix < 0 || size_t(prefix) > mass.size() || count < 0 || count > prefix) {
        throw std::runtime_error("RidgeFill: invalid prefix or exact-row count");
    }
    for (int32_t i = 0; i < prefix; ++i) {
        if (!std::isfinite(mass[i]) || mass[i] < 0.0) {
            throw std::runtime_error("RidgeFill quality+qattn: non-finite or negative question mass");
        }
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
