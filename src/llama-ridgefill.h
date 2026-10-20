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

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
};

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
