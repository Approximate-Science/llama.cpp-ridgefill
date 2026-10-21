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
    int32_t split = 0;
    float qattn_share = -1.0f;
    int32_t qattn_window = 64;
    bool qattn_blind = false;
    bool all_tail_logits = false;
    // Leave the last token for the server's normal output/sampling batch.
    bool defer_last_token = false;
    ggml_type type = GGML_TYPE_F16;
    size_t max_bytes = LLAMA_RIDGEFILL_SWEEP_MAX_BYTES;
};

enum class server_ridgefill_sweep_result {
    quality,
    pending,
    ready,
    complete,
};

struct server_ridgefill_sweep {
    server_ridgefill_sweep();
    ~server_ridgefill_sweep();

    const std::vector<double> & get_mass() const { return mass; }
    const std::vector<llama_pos> & get_rows() const { return rows; }
    int32_t lower_progress() const { return n_lower; }
    bool prefix_done() const { return next == phase::tail || next == phase::external || next == phase::done; }
    int32_t last_outputs() const { return n_outputs; }
    // Finish after the deferred token is decoded. Cancel before releasing an incomplete slot.
    void finish_external_tail(llama_context * ctx, llama_seq_id seq);
    void cancel(llama_context * ctx, llama_seq_id seq);

private:
    enum class phase { lower, mass, rows, prefix, tail, external, done };
    phase next = phase::lower;
    int32_t position = 0;
    int32_t n_lower = 0;
    int32_t n_outputs = 0;
    bool active = false;
    std::unique_ptr<llama_ridgefill_sweep_buffer> hidden;
    std::vector<double> mass;
    std::vector<llama_pos> rows;
    std::vector<int32_t> layers;
    std::vector<std::unique_ptr<llama_ridgefill_sweep_buffer>> windows;

    void sweep_lower(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, const server_ridgefill_sweep_params & params);
    void question_mass(llama_context * ctx, llama_seq_id seq, const server_ridgefill_sweep_params & params);
    void select_rows(const std::vector<llama_pos> & base_rows, const server_ridgefill_sweep_params & params);
    void sweep_prefix(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, const server_ridgefill_sweep_params & params);
    void sweep_tail(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, const server_ridgefill_sweep_params & params);
    void decode(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, int32_t count, bool output_all, bool output_last);

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
