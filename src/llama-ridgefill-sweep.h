#pragma once

#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct llm_graph_context;
struct llama_context;
struct llama_layer;
class llama_kv_cache;

constexpr size_t LLAMA_RIDGEFILL_SWEEP_MAX_BYTES = size_t(1) << 30;

// The slot owns this buffer until all graphs that read or write it have finished. F32 is for query windows and oracle checks.
struct llama_ridgefill_sweep_buffer {
    llama_ridgefill_sweep_buffer(
            ggml_backend_buffer_type_t buft,
            int32_t n_embd,
            int32_t n_tokens,
            ggml_type type = GGML_TYPE_F16,
            size_t max_bytes = LLAMA_RIDGEFILL_SWEEP_MAX_BYTES);

    static size_t required_bytes(int32_t n_embd, int32_t n_tokens, ggml_type type, size_t max_bytes);

    // Row offsets are fixed in the graph; rebuild it when first changes.
    ggml_tensor * capture(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * src, int32_t first) const;
    ggml_tensor * slice(ggml_context * ctx, int32_t first, int32_t count) const;

    ggml_tensor * tensor() const { return data; }
    size_t size_bytes() const;

private:
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor * data = nullptr;
};

// Q and K are post-RoPE [head_dim, rows, heads], with keys in prompt order.
// The result is FP32 [n_tokens, n_kv_heads]; sum groups/layers in FP64 on the host.
ggml_tensor * llama_ridgefill_sweep_mass(
        ggml_context * ctx,
        ggml_tensor * q,
        ggml_tensor * k,
        int32_t window_end,
        float scale);

// Pass the cache's key rotation when its stored keys are rotated.
ggml_tensor * build_ridgefill_question_mass(
        llm_graph_context & g,
        const llama_layer & layer,
        ggml_tensor * window,
        ggml_tensor * positions,
        ggml_tensor * keys,
        int32_t window_end,
        int32_t il,
        ggml_tensor * key_rotation = nullptr);

std::vector<int32_t> llama_ridgefill_sweep_key_rows(const llama_kv_cache & cache, int32_t seq_id, int32_t n_tokens);

// rows contains the indices returned by llama_ridgefill_sweep_key_rows for this sequence.
ggml_tensor * build_ridgefill_sweep_keys(
        ggml_context * ctx,
        const llama_kv_cache & cache,
        int32_t seq_id,
        int32_t il,
        int32_t head_dim,
        ggml_tensor * rows);

std::vector<int32_t> llama_ridgefill_sweep_top_rows(
        const std::vector<double> & mass,
        int32_t prefix,
        double share);

std::vector<int32_t> llama_ridgefill_sweep_top_k(const std::vector<double> & mass, int32_t prefix, int32_t count);

std::vector<double> llama_ridgefill_sweep_question_mass(
        llama_context & ctx, int32_t seq_id, int32_t n_tokens, int32_t window_end,
        const std::vector<int32_t> & layers, const std::vector<ggml_tensor *> & windows);

