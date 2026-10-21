#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>
#include <vector>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// RidgeFill: approximate prefill of long prompts
// the filled tokens run the layers [0, split) only, the K/V of the later layers are predicted by a linear map
//

struct llama_ridgefill;

// load a projector (GGUF, see convert_ridgefill_to_gguf.py) for the model, returns nullptr on error
LLAMA_API struct llama_ridgefill * llama_ridgefill_init(const struct llama_model * model, const char * path);
LLAMA_API void                     llama_ridgefill_free(struct llama_ridgefill * rf);
LLAMA_API int32_t                  llama_ridgefill_split(const struct llama_ridgefill * rf);

// attach a projector to the context (nullptr to detach), it must outlive the context
LLAMA_API void llama_set_ridgefill(struct llama_context * ctx, const struct llama_ridgefill * rf);

// from now on, the tokens of seq_id at positions < p_end are filled (p_end <= 0: none)
// a filled token cannot be an output, llama_decode() returns -1 for such a batch
// the two-sweep prefill (quality+qattn, RF27-INTERFACE.md §3): the next decodes of seq_id run one part of the model.
//   LOWER: layers [0, split) of every row (stock lower caches, no output head, no row may be an output); h_S of the
//          row at position p is written to row p of h_buf; optionally the inputs of win_layers (< split) at positions
//          [win_begin, win_begin + win_n) to row (pos - win_begin) of win_bufs[i].
//   LATE:  a replay over positions already decoded (in LOWER): h_S from row p of h_buf, then for rows < p_end the fill
//          of the late layers (bulk or exact rows R, the correction on the ubatch holding p_end - 1), for rows >= p_end
//          the stock late layers and the output head. The lower layers' caches and the cache cells are untouched; the
//          late layers' recurrent states start from zero when the replay starts at the sequence's first position.
//   FULL:  the single pass (default).
// Buffers: 2-D ggml tensors [n_embd, n] (F32, F16 or BF16) in a backend buffer, owned by the caller until the
// decodes that use them have finished. One sweep sequence per batch; a LATE batch holds LATE rows of one sequence only.
// Returns 0, or < 0 refused (logged by name). Qwen3.5 (qwen35) only.
enum llama_ridgefill_part {
    LLAMA_RIDGEFILL_PART_FULL  = 0,
    LLAMA_RIDGEFILL_PART_LOWER = 1,
    LLAMA_RIDGEFILL_PART_LATE  = 2,
};

struct llama_ridgefill_part_params {
    enum llama_ridgefill_part part;
    struct ggml_tensor * h_buf;

    int32_t                     n_win_layers;
    const int32_t             * win_layers;
    struct ggml_tensor * const * win_bufs;
    llama_pos                   win_begin;
    int32_t                     win_n;
};

LLAMA_API struct llama_ridgefill_part_params llama_ridgefill_part_default_params(void);

LLAMA_API int32_t llama_ridgefill_seq_set_part(struct llama_context * ctx, llama_seq_id seq_id,
        const struct llama_ridgefill_part_params * params);

// recurrent rollback (n_rs_seq > 0): a filled ubatch never holds exact tokens, so the snapshots of a decode whose exact
// run is shorter than n_rs_seq + 1 cover only that run; roll back only into the tokens of the last exact ubatch (as
// stock, where older slots are caller-owned) - never into filled prompt rows
LLAMA_API void llama_ridgefill_seq_set(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p_end);

// RF27 (qwen35): defaults from the projector GGUF (format 2; format-1 files report the built-in defaults)
LLAMA_API float        llama_ridgefill_alpha_default (const struct llama_ridgefill * rf);
LLAMA_API int32_t      llama_ridgefill_tail_default  (const struct llama_ridgefill * rf);
LLAMA_API int32_t      llama_ridgefill_tail_min      (const struct llama_ridgefill * rf);
LLAMA_API float        llama_ridgefill_share_default (const struct llama_ridgefill * rf);
LLAMA_API bool         llama_ridgefill_digits_default(const struct llama_ridgefill * rf);
LLAMA_API float        llama_ridgefill_qattn_default (const struct llama_ridgefill * rf);
LLAMA_API const char * llama_ridgefill_desc          (const struct llama_ridgefill * rf); // "name (authors; license; doi ...)"

// the recurrent-state correction scale in [0, 1] (default: the projector's alpha_default); 0 ok, < 0 refused
LLAMA_API int32_t llama_ridgefill_set_alpha(struct llama_context * ctx, float alpha);

// the exact rows R of seq_id (sorted, unique, < p_end; n = 0: speed mode); call after llama_ridgefill_seq_set,
// which clears them. 0 ok, < 0 refused. llama_decode() refuses filled rows of a sequence with exact rows when alpha
// != 0 (the rho-scaled correction is not implemented) or with parallel sequences on a non-unified KV cache.
LLAMA_API int32_t llama_ridgefill_seq_set_rows(struct llama_context * ctx, llama_seq_id seq_id,
        const llama_pos * rows, int32_t n);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);
