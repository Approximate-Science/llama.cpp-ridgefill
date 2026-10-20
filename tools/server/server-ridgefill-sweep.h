#pragma once

#include "llama.h"
#include "../../src/llama-ridgefill-sweep.h"

#include <cstddef>
#include <memory>
#include <vector>

struct server_ridgefill_sweep_params {
    int32_t n_embd = 0;
    int32_t n_tokens = 0;
    int32_t n_prefix = 0;
    int32_t n_batch = 0;
    float qattn_share = -1.0f;
    int32_t qattn_window = 64;
    bool qattn_blind = false;
    ggml_type type = GGML_TYPE_F16;
    size_t max_bytes = LLAMA_RIDGEFILL_SWEEP_MAX_BYTES;
};

enum class server_ridgefill_sweep_result {
    quality,
    pending,
    complete,
};

struct server_ridgefill_sweep {
    server_ridgefill_sweep();
    ~server_ridgefill_sweep();

private:
    enum class phase { lower, mass, rows, prefix, tail, done };
    phase next = phase::lower;
    int32_t position = 0;
    std::unique_ptr<llama_ridgefill_sweep_buffer> hidden;
    std::vector<double> mass;
    std::vector<llama_pos> rows;

    void sweep_lower(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, const server_ridgefill_sweep_params & params);
    void question_mass(llama_context * ctx, llama_seq_id seq, const server_ridgefill_sweep_params & params);
    void select_rows(const std::vector<llama_pos> & base_rows, const server_ridgefill_sweep_params & params);
    void sweep_prefix(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, const server_ridgefill_sweep_params & params);
    void sweep_tail(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, const server_ridgefill_sweep_params & params);

    friend server_ridgefill_sweep_result server_ridgefill_sweep_step(
            server_ridgefill_sweep &, llama_context *, llama_seq_id, const llama_token *,
            const std::vector<llama_pos> &, const server_ridgefill_sweep_params &);
};

// Call only for quality+qattn. Zero share returns quality before allocation or context access.
// A step runs at most one ubatch or one mass/selection stage, so the server can service other slots.
server_ridgefill_sweep_result server_ridgefill_sweep_step(
        server_ridgefill_sweep & sweep,
        llama_context * ctx,
        llama_seq_id seq,
        const llama_token * tokens,
        const std::vector<llama_pos> & base_rows,
        const server_ridgefill_sweep_params & params);
