#pragma once

#include "llama.h"
#include "llama-graph.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <string>
#include <vector>

// RidgeFill: the prompt tokens before the exact tail run layers [0, split) only. For each layer il >= split,
// the input of layer il is predicted from the input of layer split with a linear map, and only the cache of
// layer il is written from it: K/V for attention layers, the conv and recurrent state for Gated DeltaNet layers.
// The projector GGUF format is described in RF27-INTERFACE.md §1.
struct llama_ridgefill {
    int32_t split  = -1;
    int32_t format = 1; // 1: the Oct-8 dense files (llama, qwen3), 2: RF27 (qwen35)

    // per layer, nullptr for il < split; w and b are nullptr for a GDN layer filled with the identity map
    std::vector<ggml_tensor *> w;  // [n_embd, n_embd]
    std::vector<ggml_tensor *> b;  // [n_embd], F32
    std::vector<ggml_tensor *> st; // F32 [S_k, S_v, H_v] for the late GDN layers (llama.cpp's state layout)

    // index of the late GDN layer il among the late GDN layers (-1 otherwise): the row of the correction scale
    std::vector<int32_t> gdn_idx;
    int32_t n_gdn_late = 0;

    // defaults from the GGUF (format 2), used by the tools
    float   alpha_default  = 0.0f;
    int32_t tail_default   = 4096;
    int32_t tail_min       = 1;
    float   share_default  = 0.5f;
    bool    digits_default = false;
    float   qattn_default  = 0.0f;

    std::string desc; // "name (authors, license, doi)"

    // host tables for the exact-row selection (format 2)
    std::vector<float>   score;    // [n_vocab], non-finite = the token is in no class
    std::vector<uint8_t> is_digit; // [n_vocab], token text contains an ASCII digit (added tokens excluded)
    std::vector<uint8_t> is_special; // [n_vocab], an added token: vocab attribute control, user-defined or unknown

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
};

// the two-sweep prefill state of one sequence (llama_ridgefill_seq_set_part), owned by llama_context
struct llama_ridgefill_part_state {
    llm_ridgefill_part part = LLM_RIDGEFILL_PART_FULL;
    uint64_t           gen  = 0; // bumped on every set, part of the graph-reuse key (the buffers are baked in)

    ggml_tensor * h_buf = nullptr; // [n_embd, n_rows]: row p = h_S of position p (LOWER writes, LATE reads)

    // LOWER only: the inputs of these layers (< split) at positions [win_begin, win_begin + win_n) are written to
    // row (pos - win_begin) of win_bufs[i] ([n_embd, win_n])
    std::vector<int32_t>       win_layers;
    std::vector<ggml_tensor *> win_bufs;
    llama_pos                  win_begin = 0;
    int32_t                    win_n     = 0;

    // cursors (adopted from L-C's draft 8de161386): the next position LOWER / LATE accepts; rows arrive in order,
    // LATE replays exactly the rows LOWER processed
    llama_pos lower_end = 0;
    llama_pos late_end  = 0;

    // S1_PROBE: the window [s1_begin, s1_end); S1_FILL / S1_PROBE / LATE replay rows with the late_end cursor
    llama_pos s1_begin = 0;
    llama_pos s1_end   = 0;
};

class llama_kv_cache_context;

// graph input of a LOWER / LATE ubatch (one sequence)
class llm_graph_input_ridgefill_part : public llm_graph_input_i {
public:
    llm_graph_input_ridgefill_part(const llm_graph_params_ridgefill & params) : params(params) {}
    virtual ~llm_graph_input_ridgefill_part() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * h_rows   = nullptr; // I32 [n_tokens]: the h_buf row (position) of every ubatch row
    ggml_tensor * win_sel  = nullptr; // I32 [n_win]: LOWER, the ubatch rows inside the capture window
    ggml_tensor * win_rows = nullptr; // I32 [n_win]: their rows in the window buffers

    // S1_PROBE: F32 [n_kv]: 0, or -inf for the cells of positions [p_end, s1_begin) (tail rows the probe does not run)
    ggml_tensor * s1_bias  = nullptr;
    // vtail: F32 [n_kv] 1 for the cells of the bulk prefix rows (positions < p_end, not exact), else 0; and 1 - that
    ggml_tensor * vt_bulk  = nullptr;
    ggml_tensor * vt_keep  = nullptr;

    const llama_kv_cache_context * kv = nullptr; // the ubatch's attention cache (cell positions for the masks)
    llama_seq_id seq = -1;

    const llm_graph_params_ridgefill params;
};

// S1_PROBE, one late attention layer of the window rows (as qwen35's build_layer_attn, with an explicit softmax over
// the visible cells: the KQ mask plus s1_bias); adds the layer's attention mass per cell (summed over heads and the
// ubatch's queries) to *mass ([n_kv], created on the first call). Returns the layer's attention output (after wo).
ggml_tensor * build_ridgefill_s1_attn(llm_graph_context & g, const llama_model & model, const llm_graph_input_ridgefill_part * inp,
        ggml_tensor * cur, ggml_tensor * inp_pos, llm_graph_input_attn_kv * inp_attn, int il, ggml_tensor ** mass);

// vtail on the first tail ubatch of a LATE replay, one late attention layer: s_h = mean over the ubatch's first
// vtail_rows rows of ||V_true|| / ||V_fill|| per KV head (V_true from cur = attn_norm of the rows' exact input, V_fill
// from the projector on their h_S); the cached V of the bulk prefix cells times s_h. Inserted before the layer's own
// attention. Appends s ([n_head_kv]) to *scales ([n_head_kv, n_late_attn]).
void build_ridgefill_vtail(llm_graph_context & g, const llama_model & model, const llm_graph_input_ridgefill_part * inp,
        ggml_tensor * cur, ggml_tensor * h_S, llm_graph_input_attn_kv * inp_attn, int il, ggml_tensor ** scales);

// LOWER: write h_S (the input of layer split) of every ubatch row to h_buf; LOWER, il < split: the window rows of
// the input of layer il to its window buffer (no-op for layers without one)
void build_ridgefill_capture(llm_graph_context & g, const llm_graph_input_ridgefill_part * inp, ggml_tensor * cur, int il);

// LATE: h_S of the ubatch rows from h_buf (F32 [n_embd, n_tokens])
ggml_tensor * build_ridgefill_h_input(llm_graph_context & g, const llm_graph_input_ridgefill_part * inp);

// what a filled ubatch does at the late layers
enum llm_ridgefill_fill : int32_t {
    LLM_RIDGEFILL_FILL_NONE   = 0, // not filled: stock layers (never passed to build_ridgefill_late)
    LLM_RIDGEFILL_FILL_BULK   = 1, // speed mode, or a quality ubatch without exact rows: every row bulk
    LLM_RIDGEFILL_FILL_MASKED = 2, // quality: the rows in `rows` are exact (oracle layer_rows), the others bulk
};

// graph input of a filled ubatch: the exact rows and the recurrent-state correction scale
class llm_graph_input_ridgefill : public llm_graph_input_i {
public:
    llm_graph_input_ridgefill(const llm_graph_params_ridgefill & params) : params(params) {}
    virtual ~llm_graph_input_ridgefill() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * rows  = nullptr; // I32 [n_R]: ubatch row indices of the exact rows R, ascending
    ggml_tensor * pos   = nullptr; // I32 [n_R * n_pos]: their positions, section-major as the model's inp_pos
    ggml_tensor * scale = nullptr; // F32 [H_v, n_seqs, n_gdn_late]: correction scale per head, sequence, late GDN layer

    const llm_graph_params_ridgefill params;
};

// the ubatch row indices of the exact rows (positions in rows_seq[seq], sorted), ascending; returns their count
int32_t llama_ridgefill_ubatch_rows(const llama_ubatch & ubatch, const std::vector<llama_pos> * rows_seq, std::vector<int32_t> * idx);

struct llm_build_delta_net_base;

// Builds, for every layer il in [split, n_layer), the cache-writing ops of il from h_S (qwen35; see
// RF27-INTERFACE.md §2). Bulk rows never get an attention output, an MLP or a residual. Returns nullptr (BULK)
// or the exact rows' residual after the last layer (MASKED). Adds no output head: the caller returns before it.
ggml_tensor * build_ridgefill_late(
        llm_build_delta_net_base        & g,
        const llama_model               & model,
        ggml_tensor                     * h_S,
        const llm_graph_input_ridgefill * rows_mask,
        ggml_tensor                     * inp_pos,
        llm_graph_input_attn_kv         * inp_attn,
        llm_graph_input_rs              * inp_rs,
        llm_ridgefill_fill                mode);
