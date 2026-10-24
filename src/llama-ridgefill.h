#pragma once

#include "llama.h"
#include "llama-graph.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <future>
#include <array>
#include <utility>
#include <string>
#include <vector>

// RidgeFill: the prompt tokens before the exact tail run layers [0, split) only. For each layer il >= split,
// the input of layer il is predicted from the input of layer split with a linear map, and only the cache of
// layer il is written from it: K/V for attention layers, the conv and recurrent state for Gated DeltaNet layers.
// The projector GGUF format is described in RF27-INTERFACE.md §1.
struct llama_ridgefill {
    int32_t split  = -1;
    int32_t format = 1; // 1: the Oct-8 dense files (llama, qwen3), 2: RF27 (qwen35)
    bool streamed = false;
    bool profiling = false;
    bool trim_kv = true;

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

void llama_ridgefill_profile(const llama_ridgefill * rf, const char * phase, int32_t layer, int32_t pos, int64_t begin, int64_t end);

// One context owns the staging buffers; asynchronous copies read the immutable host projector.
class llama_ridgefill_stream {
public:
    llama_ridgefill_stream(const llama_model & model, const llama_ridgefill & rf);
    void stage(llama_context & ctx, int32_t layer, int32_t next_layer);
    ggml_tensor * weight(int32_t layer) const;
    ggml_tensor * bias(int32_t layer) const;
    ggml_tensor * reserve_weight(int32_t layer) const { return rf.w.at(layer) ? w.at(layer)[0] : nullptr; }
    ggml_tensor * reserve_bias(int32_t layer) const { return rf.b.at(layer) ? b.at(layer)[0] : nullptr; }
    size_t buffer_bytes() const { return n_bytes; }
    uint64_t uploaded_bytes() const { return n_uploaded; }

private:
    struct device_maps {
        ggml_backend_buffer_type_t buft = nullptr;
        std::array<ggml_backend_buffer_ptr, 2> buffers;
        std::array<int32_t, 2> layers = {{-1, -1}};
        std::array<std::future<void>, 2> pending;
    };
    const llama_ridgefill & rf;
    ggml_context_ptr metadata;
    std::vector<device_maps> devices;
    std::vector<int32_t> device;
    std::vector<int32_t> selected;
    std::vector<std::array<ggml_tensor *, 2>> w, b;
    size_t n_bytes = 0;
    uint64_t n_uploaded = 0;

    void copy(llama_context & ctx, int32_t layer, int32_t slot, bool prefetch);
};

// the two-sweep prefill state of one sequence (llama_ridgefill_seq_set_part), owned by llama_context
struct llama_ridgefill_mask_cache;
struct llama_ridgefill_part_state {
    llm_ridgefill_part part = LLM_RIDGEFILL_PART_FULL;
    uint64_t           gen  = 0; // bumped on every set, part of the graph-reuse key (the buffers are baked in)

    ggml_tensor * h_buf = nullptr; // [n_embd, n_rows]: row p = h_S of position p (LOWER writes, LATE reads)
    ggml_tensor * h_compact = nullptr;
    ggml_tensor * h_tail = nullptr;
    const std::vector<int32_t> * h_indices = nullptr;
    llama_pos h_keep_end = 0, h_tail_begin = 0;
    mutable std::shared_ptr<llama_ridgefill_mask_cache> mask_cache;
    int32_t stream_layer = -1;
    llama_pos stream_end = 0;
    ggml_tensor * stream_carry = nullptr; // F32 [n_embd, |R| + first tail chunk]
    const std::vector<llama_pos> * stream_rows = nullptr;

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
    bool prefix_kv_filled = false;
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
    ggml_tensor * carry_rows = nullptr; // I32: compact exact prefix or first-tail residual rows
    ggml_tensor * attention_mask = nullptr;
    ggml_tensor * mask_rows = nullptr;
    llama_pos mask_past = -1;
    ggml_tensor * win_sel  = nullptr; // I32 [n_win]: LOWER, the ubatch rows inside the capture window
    ggml_tensor * win_rows = nullptr; // I32 [n_win]: their rows in the window buffers

    // S1_PROBE: the KQ mask of the probe graph (the stock mask's type and shape [n_kv, n_tokens, 1, 1]): causal over
    // the sequence's cells, and -inf for the cells of positions [p_end, s1_begin) (tail rows the probe does not run);
    // the probe's attention layers use it instead of the stock mask. ones: F32 [1, n_tokens], the mass reduction.
    // Per query block (s1_qb rows; one block: the tensors themselves) their views, made once per graph: every view of
    // an input is a separate split input, which a non-CPU backend copies into the compute buffer for the whole split
    // (one view per head and layer was 288 copies of the mask, 2.7 GiB at 9K cells on Vulkan)
    ggml_tensor * s1_mask  = nullptr;
    ggml_tensor * s1_rows  = nullptr; // I32: this sequence's cells in prompt order, for a shared KV stream
    ggml_tensor * ones     = nullptr;
    int64_t                    s1_qb = 0;
    std::vector<ggml_tensor *> s1_mask_blk;
    std::vector<ggml_tensor *> s1_ones_blk;
    // vtail: F32 [n_kv] 1 for the cells of the bulk prefix rows (positions < p_end, not exact), else 0; and 1 - that
    ggml_tensor * vt_bulk  = nullptr;
    ggml_tensor * vt_keep  = nullptr;

    const llama_kv_cache_context * kv = nullptr; // the ubatch's attention cache (cell positions for the masks)
    llama_seq_id seq = -1;

    const llm_graph_params_ridgefill params;
};

// x, computed after dep in every topological order (x + 0 * dep[0]; dep finite): keeps a backend that reorders
// independent nodes (the Vulkan graph optimizer) from hoisting the next block of scores while earlier ones are alive
ggml_tensor * llama_ridgefill_follow(ggml_context * ctx, ggml_tensor * x, ggml_tensor * dep);

// queries per block so that n_kv x block <= max_elems (all n_q when they fit; else equal blocks)
int64_t llama_ridgefill_query_block(int64_t n_kv, int64_t n_q, int64_t max_elems);

// the attention mass per key of query heads [h0, h0 + n_heads) of q_base ([d, n_head, n_q], contiguous, roped and
// rotated as the cached keys): sum over those heads and the queries of softmax_k(scale * k . q + mask) (mask:
// [n_kv, >= n_q], or nullptr with a causal diagonal at diag_past, as ggml_diag_mask_inf). k: [d, n_kv, n_head_kv]
// (views allowed), query head h0 + i uses key head i / (n_heads / n_head_kv). Cut into blocks of at most max_elems
// scores (one key head's query heads or fewer x a query range); each block follows *dep (in/out: the last block's
// result, nullptr at the start), so one block is alive at a time on any backend.
//   ones != nullptr: F32 [1, >= max block], the reduction is out_prod(probs, ones) (no transposed copy); returns [n_kv].
//   ones == nullptr: per key head, the reduction is sum_rows of the transposed block, the summation order of an
//                    unsliced softmax-then-sum_rows whenever a key head's block fits max_elems; returns [n_kv, n_head_kv].
ggml_tensor * llama_ridgefill_attn_mass(ggml_context * ctx, ggml_tensor * q_base, int64_t h0, int64_t n_heads,
        ggml_tensor * k, ggml_tensor * mask, int32_t diag_past, float scale, ggml_tensor * ones, int64_t max_elems,
        ggml_tensor ** dep);

// the block size of llama_ridgefill_attn_mass: query heads per block (a divisor of rep) and queries per block
void llama_ridgefill_attn_mass_split(int64_t n_kv, int64_t n_q, int64_t rep, int64_t max_elems, int64_t & hb, int64_t & qb);

// scores per block (F32 elements), independent of the prompt length: the S1 probe 32 MiB (in the context's reserve;
// one query head with all of a 256-row probe ubatch, the default window, up to 32K cells: the unsliced matrix shapes);
// the Tier-3 question mass 4 MiB (its own per-prompt scheduler; a 700-token prompt still fits one block per key head,
// the unsliced summation order)
constexpr int64_t LLAMA_RIDGEFILL_PROBE_BLOCK = int64_t(1) << 23;
constexpr int64_t LLAMA_RIDGEFILL_QMASS_BLOCK = int64_t(1) << 20;

// peak bytes of the live tensors a graph's nodes own (views and leaves excluded, no in-place reuse), executed in graph
// order, and in the order of a backend that pulls every node as early as its inputs and the memory hazards allow (reads
// of a storage after its earlier writers, writes after earlier readers / writers: what a reordering backend such as
// the Vulkan graph optimizer keeps). A large gap means independent large tensors: blocks that would all be alive at once.
// input_views: the bytes of the distinct graph inputs and views of them that compute nodes read; a non-CPU backend's
// scheduler copies each into the compute buffer for the whole split, which a CPU run never shows.
void llama_ridgefill_graph_peaks(const ggml_cgraph * gf, size_t & in_order, size_t & any_order, size_t & input_views);

// graph nodes the RidgeFill graphs may add over a stock graph of n_ubatch tokens over n_ctx cells (the S1 probe's
// per-head blocks, vtail), for the graph and scheduler sizes
int64_t llama_ridgefill_extra_nodes(const llama_ridgefill * rf, const llama_hparams & hparams, int64_t n_ctx, int64_t n_ubatch);

// S1_PROBE, one late attention layer of the window rows: qwen35's build_layer_attn with the attention written out
// (the math of 22ef3df4a: kq, softmax over the probe mask, kqv), one query head (and, beyond
// LLAMA_RIDGEFILL_PROBE_BLOCK scores, one query block) at a time, the blocks chained; adds the layer's attention mass
// per cell (summed over heads and the ubatch's queries) to *mass ([n_kv], created on the first call). Returns the
// layer's attention output (after wo).
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
    ggml_tensor * k_idxs = nullptr; // I64: exact-row destinations when reusing the bulk K/V
    ggml_tensor * v_idxs = nullptr;
    const llm_graph_input_attn_kv * attn = nullptr;

    const llm_graph_params_ridgefill params;
};

// the ubatch row indices of the exact rows (positions in rows_seq[seq], sorted), ascending; returns their count
int32_t llama_ridgefill_ubatch_rows(const llama_ubatch & ubatch, const std::vector<llama_pos> * rows_seq, std::vector<int32_t> * idx);
bool llama_ridgefill_ubatch_dense(const llama_ubatch & ubatch);

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
        llm_ridgefill_fill                mode,
        const llm_graph_input_ridgefill_part * part);
