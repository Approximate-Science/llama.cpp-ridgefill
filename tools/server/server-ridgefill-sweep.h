#pragma once

#include "llama.h"
#include "common.h"
#include "../../src/llama-ext.h"
#include "../../src/llama-ridgefill-sweep.h"

#include <cstddef>
#include <future>
#include <memory>
#include <string>
#include <vector>

struct server_ridgefill_sweep_params {
    int32_t n_embd = 0;
    int32_t n_tokens = 0;
    int32_t n_prefix = 0;
    int32_t n_batch = 0;
    int32_t split = 0;
    float qattn_share = -1.0f;
    int32_t qattn_window = 64;
    std::string qattn_window_mode = "last";
    int32_t qattn_turn_cap = 256;
    // Exclusive selector boundary; -1 uses the prompt end (or prefix end when blind).
    int32_t qattn_end = -1;
    bool qattn_blind = false;
    bool s1 = false;
    int32_t s1_window = 256;
    double s1_factor = 1.0;
    const llama_ridgefill * projector = nullptr;
    bool special_exact = false;
    bool lexical = false;
    common_ridgefill_lexical_params lexical_options;
    bool vtail = false;
    int32_t vtail_rows = 512;
    bool all_tail_logits = false;
    // Optional half-open range of exact tail rows whose logits are needed for scoring.
    int32_t logits_begin = -1;
    int32_t logits_end = -1;
    // Leave the last token for the server's normal output/sampling batch.
    bool defer_last_token = false;
    ggml_type type = GGML_TYPE_F16;
    size_t max_bytes = LLAMA_RIDGEFILL_SWEEP_MAX_BYTES;
};

struct server_ridgefill_sweep_cost {
    int32_t s1_fill_rows = 0;
    int32_t s1_probe_rows = 0;
    int32_t vtail_rows = 0;
    double s1_fill_macs = 0;
    double s1_probe_macs = 0;
    double vtail_macs = 0;
    uint64_t s1_probe_kv_pairs = 0;
    uint32_t vtail_cache_rows = 0;
    uint64_t vtail_cache_elements = 0;
    uint64_t vtail_cache_read_bytes = 0;
    uint64_t vtail_cache_write_bytes = 0;
};

int32_t server_ridgefill_sweep_api_version();

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
    int32_t first_output_position() const { return output_position; }
    int32_t window_begin() const { return win_begin; }
    int32_t window_end() const { return win_end; }
    int32_t base_row_count() const { return n_base; }
    int32_t step_tokens() const { return n_step_tokens; }
    int32_t next_tokens(llama_context * ctx, const server_ridgefill_sweep_params & params) const;
    // Estimated work of the next unchanged step, in full-model prompt row equivalents.
    int32_t next_work(llama_context * ctx, llama_seq_id seq, const server_ridgefill_sweep_params & params, int32_t capacity) const;
    const char * phase_name() const;
    int32_t cursor() const { return stream_layer >= 0 ? (stream_layer - stream_split)*stream_stride + position : position; }
    const std::vector<double> & get_s1_mass() const { return s1_mass; }
    const std::vector<float> & get_vtail_scales() const { return vtail_scales; }
    const server_ridgefill_sweep_cost & get_cost() const { return cost; }
    const common_ridgefill_lexical_result & get_lexical() const { return lexical; }
    // Finish after the deferred token is decoded. Cancel before releasing an incomplete slot.
    void finish_external_tail(llama_context * ctx, llama_seq_id seq);
    void cancel(llama_context * ctx, llama_seq_id seq);

private:
    enum class phase { lower, mass, rows, s1_fill, s1_probe, s1_rows, prefix, stream_tail, tail, external, done };
    phase next = phase::lower;
    int32_t position = 0;
    int32_t n_lower = 0;
    int32_t n_outputs = 0;
    int32_t output_position = -1;
    int32_t win_begin = -1;
    int32_t win_end = -1;
    int32_t n_base = 0;
    int32_t n_step_tokens = 0;
    bool active = false;
    bool vtail_enabled = false;
    std::unique_ptr<llama_ridgefill_sweep_buffer> hidden;
    // Destroy the future before its source buffer.
    std::future<std::unique_ptr<llama_ridgefill_sweep_buffer>> hidden_readback;
    std::unique_ptr<llama_ridgefill_sweep_buffer> stream_carry;
    std::unique_ptr<llama_ridgefill_sweep_buffer> compact_hidden, tail_hidden;
    std::vector<int32_t> hidden_indices;
    int32_t hidden_keep_end = 0;
    uint64_t hidden_upload_bytes = 0, hidden_download_bytes = 0;
    int32_t stream_layer = -1, stream_calib_rows = 0, stream_stride = 0, stream_split = 0;
    std::vector<double> mass;
    std::vector<double> s1_mass;
    std::vector<float> vtail_scales;
    server_ridgefill_sweep_cost cost;
    common_ridgefill_lexical_result lexical;
    std::vector<llama_pos> rows;
    std::vector<int32_t> layers;
    std::vector<std::unique_ptr<llama_ridgefill_sweep_buffer>> windows;

    void sweep_lower(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, const server_ridgefill_sweep_params & params);
    void question_mass(llama_context * ctx, llama_seq_id seq, const server_ridgefill_sweep_params & params);
    void select_rows(llama_context * ctx, const llama_token * tokens, const std::vector<llama_pos> & base_rows, const server_ridgefill_sweep_params & params);
    void prepare_cost(const llama_model & model, const server_ridgefill_sweep_params & params);
    void sweep_s1_fill(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, const server_ridgefill_sweep_params & params);
    void sweep_s1_probe(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, const server_ridgefill_sweep_params & params);
    void select_s1_rows(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, const server_ridgefill_sweep_params & params);
    void sweep_prefix(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, const server_ridgefill_sweep_params & params);
    void sweep_tail(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, const server_ridgefill_sweep_params & params);
    bool advance_stream(llama_context * ctx, llama_seq_id seq, const server_ridgefill_sweep_params & params, bool attn_only);
    void compact_hidden_rows(llama_context * ctx, llama_seq_id seq, const server_ridgefill_sweep_params & params);
    void prefetch_hidden(llama_context * ctx, const server_ridgefill_sweep_params & params);
    void decode(llama_context * ctx, llama_seq_id seq, const llama_token * tokens, int32_t count, bool output_all, bool output_last,
            int32_t logits_begin = -1, int32_t logits_end = -1);

    friend server_ridgefill_sweep_result server_ridgefill_sweep_step(
            server_ridgefill_sweep &, llama_context *, llama_seq_id, const llama_token *,
            const std::vector<llama_pos> &, const server_ridgefill_sweep_params &);
};

// Zero qattn share returns quality before allocation unless S1 or vtail needs the two-sweep flow.
// A step runs at most one ubatch or one mass/selection stage, so the server can service other slots.
// base_rows includes the caller's class, digit, special and header selection from select_rows_ext.
server_ridgefill_sweep_result server_ridgefill_sweep_step(
        server_ridgefill_sweep & sweep,
        llama_context * ctx,
        llama_seq_id seq,
        const llama_token * tokens,
        const std::vector<llama_pos> & base_rows,
        const server_ridgefill_sweep_params & params);
