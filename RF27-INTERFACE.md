# RF27 interface note (L-A engine → L-B tools, L-C sweep)

Written 2026-10-20 ~18:00 (T+0:15) by L-A on `rf27-engine`. Binding between the three lanes; a change is a new dated
section at the bottom, never a silent edit. Where this note and the oracle (`kva`) disagree, the oracle wins and the
note gets a dated correction.

## 1. Projector GGUF (L-B's converter writes, L-A's loader `src/llama-ridgefill.cpp` reads)

### Metadata

| key | GGUF type | value / rule | loader |
|---|---|---|---|
| `general.type` | string | `adapter` | refuse otherwise |
| `adapter.type` | string | `ridgefill` | refuse otherwise |
| `general.architecture` | string | `qwen35` (dense branch: `llama` / `qwen3`) | refuse if != model arch |
| `adapter.ridgefill.format` | u32 | `2` (new; the Oct-8 dense files have no key = format 1) | qwen35 requires 2 |
| `adapter.ridgefill.split` | u32 | S (27B: 16, tiny fixture: 3) | refuse outside `[1, n_layer)` |
| `adapter.ridgefill.n_layer` | u32 | trunk layers without MTP (27B: 64) | refuse if != `hparams.n_layer()` |
| `adapter.ridgefill.n_embd` | u32 | 5120 | refuse if != `n_embd` |
| `adapter.ridgefill.alpha_default` | f32 | manifest `correction.alpha_default` (27B: 0.0) | default of `--ridgefill-alpha` |
| `adapter.ridgefill.tail_default` | u32 | manifest `tail.default` (4096) | default tail |
| `adapter.ridgefill.tail_min` | u32 | manifest `tail.min` (512) | tail below it refused by name |
| `adapter.ridgefill.share_default` | f32 | manifest `rowsel.share_default` (0.5) | default class share |
| `adapter.ridgefill.digits_default` | bool | manifest `rowsel.digits_default` (true) | default digit rows |
| `adapter.ridgefill.qattn_default` | f32 | manifest `rowsel.qattn_default` (0.2) | default qattn share (L-C) |
| `adapter.ridgefill.name` | string | manifest `name` | startup log line |
| `adapter.ridgefill.authors` | string array | manifest `authors` | startup log line |
| `adapter.ridgefill.license` | string | manifest `license` | startup log line |
| `adapter.ridgefill.doi` | string | manifest `doi` | startup log line |
| `adapter.ridgefill.source_manifest_sha256` | string | sha256 of `ridgefill.json` | logged |
| `adapter.ridgefill.source.<file>` | string | sha256 of every source file (`proj8.L*.safetensors`, `correction.safetensors`, `rowsel.safetensors`) | provenance only |
| `adapter.ridgefill.tokenizer_sha256` | string | manifest `model.tokenizer_json_sha256` | provenance only (an engine cannot hash `tokenizer.json`) |
| `adapter.ridgefill.vocab_sha256` | string | **new, checked**: sha256 over the base model GGUF's `tokenizer.ggml.tokens` in id order, each token's UTF-8 bytes followed by one `0x00` byte. Converter flag `--model-gguf PATH` (gguf-py reads only the metadata). | recomputed from `model.vocab` at load; refuse on mismatch by name |
| `adapter.ridgefill.model_meta.<key>` | u32 (ints), f32 (floats), string, bool — the JSON type | one per `model.meta` entry of the manifest | each key compared with the model (map below); an unknown key is refused by name (fail closed) |

`model_meta` → llama.cpp hparams the loader compares: `num_hidden_layers` = `n_layer()`, `hidden_size` = `n_embd`,
`intermediate_size` = `n_ff(il)` for every trunk layer, `vocab_size` = `n_vocab`, `num_attention_heads` = `n_head(il)`
(attention layers), `num_key_value_heads` = `n_head_kv(il)` (attention layers), `head_dim` = `n_embd_head_k`,
`full_attention_interval` = the period of `!is_recr(il)`, `linear_num_key_heads` = `ssm_n_group`,
`linear_num_value_heads` = `ssm_dt_rank`, `linear_key_head_dim` = `ssm_d_state`, `linear_value_head_dim` =
`ssm_d_inner / ssm_dt_rank`.

### Tensors

| name | type | ggml `ne` (numpy shape is the reverse) | for |
|---|---|---|---|
| `blk.<l>.ridgefill.weight` | Q8_0 (default; from the dequantised `code × scale` in f32) or `--outtype bf16/f16/f32` | `[n_embd_in, n_embd_out]` = numpy `[out, in]` (HF `Linear` orientation; `x_l = W·h_S + c`) | every `l` in `[S, n_layer)` |
| `blk.<l>.ridgefill.bias` | F32 | `[n_embd]` | same layers |
| `blk.<l>.ridgefill.st` | F32 | `[S_k, S_v, H_v]` = `[128, 128, 48]`, i.e. numpy `[H_v, V, K]`: element `st[h, j, i]` adds to llama.cpp's recurrent state `S[i][j]` (i = key index, j = value index) of value head `h` | every late GDN layer (`is_recr(l)`, `l ≥ S`) |
| `ridgefill.rowsel.score` | F32 | `[n_vocab]`; finite = token is in a class (higher = rarer), non-finite (`-inf`/NaN) = not | once |

Rules for `st`, read before writing the converter:
- llama.cpp's state memory is `s[h][j][i]` with the key index `i` fastest (`ggml-cpu/ops.cpp`
  `gated_delta_net`: "state is stored transposed: s_out[j*S_v + i] = S[i][j]"). The oracle's recurrent state is
  `[H, dk, dv]` (`kva/ops.py delta_rule`, `model.py fill`: `cache["rec"] + st`), so the oracle's `st[h, i, j]` must be
  written as numpy `[h', j, i]` (swap the last two axes) **if** `correction.safetensors` holds the oracle's
  orientation. The manifest says `layout "[heads, V, K]"`; check which one `correction.safetensors` actually is
  (how it is added in `kva`), and say in your report which transform you applied. Since dk = dv the shape cannot tell;
  only the alpha-1 G1 run with a random, non-symmetric `st` can.
- Heads: `h'` is llama.cpp's value-head order: apply the converter's grouped→tiled reorder (`_reorder_v_heads`; the
  inverse of `kva/gguf_ckpt.py _untile`) along the head axis.
- Missing `weight`+`bias` for a **GDN** layer = identity fill on that layer (`x_l = h_S`), logged once (for the future
  attention-only projector). Missing for an attention layer, only one of the pair present, a missing `st` for a late
  GDN layer, any tensor not listed here, or a wrong shape/type: refused by name.
- The digit set is not in the file: the loader builds it from `model.vocab` (token text contains an ASCII `0-9`;
  tokens with attribute control / user-defined / unknown / unused excluded).

## 2. The fill graph (L-A implements, L-C calls)

Declared in `src/llama-ridgefill.h`, defined in `src/llama-ridgefill.cpp`. A free function, because it needs the
delta-net helpers (`build_rs`, `build_conv_state`, `build_delta_net`) that live in `llm_build_delta_net_base`, and
`src/models/models.h` is not touched. Every member it uses of the builder is public.

```cpp
struct llm_build_delta_net_base;          // src/models/models.h
struct llm_graph_input_attn_kv;
struct llm_graph_input_rs;

// what a filled ubatch does at the late layers
enum llm_ridgefill_fill : int32_t {
    LLM_RIDGEFILL_FILL_NONE   = 0, // not filled: stock layers (never passed to build_ridgefill_late)
    LLM_RIDGEFILL_FILL_BULK   = 1, // speed mode, or a quality ubatch without exact rows: every row bulk
    LLM_RIDGEFILL_FILL_MASKED = 2, // quality: the rows in `rows` are exact (oracle layer_rows), the others bulk
};

// graph input of a MASKED ubatch (owned by the graph result like every llm_graph_input_i)
struct llm_graph_input_ridgefill : public llm_graph_input_i {
    ggml_tensor * rows   = nullptr; // I32 [n_R]: ubatch row indices of the exact rows R, ascending
    ggml_tensor * scale  = nullptr; // F32 [H_v, n_seqs, n_late_gdn] correction scale, only on the ubatch that holds
                                    //     row p_end-1 of a sequence and alpha != 0 (else nullptr, no nodes)
    // set_input() fills both from llama_context's per-ubatch data
};

// Builds, for every layer l in [split, n_layer), the cache-writing ops of l from h_S:
//   x_l = W_l h_S + c_l (or x_mix = set_rows(x_pred, rows, h_R) in MASKED), attn_norm, then
//   attention layer: K,V via build_qkv (Q never expanded), attn_k_norm, ggml_rope_multi(sections) on K, cpy_k/cpy_v
//   GDN layer:       wqkv only (no wqkv_gate), beta, alpha -> gate, build_conv_state, ssm_conv + silu, q/k/v views,
//                    l2 norms, the delta rule continuing the state, the state written back (+ scale * st when the
//                    `scale` input exists); the rule's output dropped (BULK) or gathered at R (MASKED)
//   MASKED only, for rows R: attention Q/gate/attention over the cache with the KQ-mask rows of R, wo; GDN: z,
//                    gated norm, ssm_out; residual, attn_post_norm, MLP; carried as h_R to the next late layer.
// Bulk rows never get an attention output, an MLP or a residual.
// Returns nullptr (BULK) or h_R after layer n_layer-1, F32 [n_embd, n_R] (MASKED; for dumps, never logits).
// Adds no output head: the caller returns before it (as build_ridgefill does for llama/qwen3).
ggml_tensor * build_ridgefill_late(
        llm_build_delta_net_base        & g,        // the graph being built (qwen35 graph, or L-C's sweep graph)
        const llama_model               & model,
        ggml_tensor                     * h_S,      // F32 [n_embd, n_tokens]: residual entering layer `split`
        const llm_graph_input_ridgefill * rows_mask,// nullptr for BULK without correction
        ggml_tensor                     * inp_pos,  // I32 [n_tokens * 4]: build_inp_pos() (M-RoPE, 4 per token)
        llm_graph_input_attn_kv         * inp_attn, // build_inp_mem_hybrid()->get_attn()
        llm_graph_input_rs              * inp_rs,   // build_inp_mem_hybrid()->get_recr()
        llm_ridgefill_fill                mode);
```

Preconditions (asserted, the context guarantees them): `ubatch.equal_seqs()` (hybrid memory always splits equal),
every row of the ubatch is filled, no row is an output. The llama/qwen3 path keeps its existing
`llm_graph_context::build_ridgefill(...)` member unchanged.

The hook in `src/models/qwen35.cpp` `graph::graph` (L-A):

```cpp
for (int il = 0; il < n_layer; ++il) {
    if (ridgefill_late_at(il)) {                       // filled ubatch (or L-C's LOWER part) and il == split
        build_ridgefill_late(*this, model, inpL, build_inp_ridgefill() /* rows + scale, nullptr if neither */,
                             inp_pos, inp->get_attn(), inp->get_recr(), ridgefill.fill);
        return;                                        // no output head
    }
    ...stock layer...
```

## 3. Graph params, context fields, API (L-A adds; L-C uses)

### `llm_graph_params` (`src/llama-graph.h`), all part of `allow_reuse()`

```cpp
// which part of the model this ubatch runs (the two-sweep prefill's graph types)
enum llm_ridgefill_part : int32_t {
    LLM_RIDGEFILL_PART_FULL  = 0, // single pass: layers [0, split), then fill or stock late layers
    LLM_RIDGEFILL_PART_LOWER = 1, // sweep 1: layers [0, split) only; h_S of every row is a graph output
    LLM_RIDGEFILL_PART_LATE  = 2, // sweeps 2-3: no embedding, no lower layers; h_S is a graph input
};

struct llm_graph_params_ridgefill {          // value member `ridgefill` of llm_graph_params (replaces the pointer)
    const llama_ridgefill * rf   = nullptr;  // non-null iff the ubatch is filled or part != FULL
    llm_ridgefill_fill      fill = LLM_RIDGEFILL_FILL_NONE;
    llm_ridgefill_part      part = LLM_RIDGEFILL_PART_FULL;
    int32_t                 n_rows  = 0;     // |R| in this ubatch (shape of the rows input)
    bool                    correct = false; // the `scale` input exists (correction ubatch, alpha != 0)
    ggml_tensor *           h_buf   = nullptr; // PART_LOWER: written; PART_LATE: read (see below)
};
```
In PART_LATE with `fill == NONE` the qwen35 graph runs the **stock** late layers from the h_S input and the output
head (sweep 3: the tail, logits for its output rows); with `fill == BULK/MASKED` it calls `build_ridgefill_late` on the
h_S input (sweep 2). Stock graphs have `rf == nullptr` and are built and reused exactly as upstream.

### h_S buffer (L-C allocates and owns it)

`h_buf`: a 2-D ggml tensor `[n_embd, n_rows_max]` in any backend buffer, type F32, F16 or BF16; **row `p` holds h_S of
the sequence's position `p`**. PART_LOWER: after the compute, the context copies the graph's h_S output of each row
into row `pos` of `h_buf` (`ggml_backend_tensor_copy` on a row view; converts to the buffer's type in-graph).
PART_LATE: before the compute, the context copies rows `[pos_0, pos_0 + n)` of `h_buf` into the graph's h_S input
(rows of the ubatch must be consecutive positions of one sequence; the context refuses otherwise). One sequence per
PART_LOWER/LATE ubatch; other sequences' tokens in the same `llama_decode` batch go to other ubatches (the split is
homogeneous in the marker below).

### Per-sequence state in `llama_context` (`src/llama-context.{h,cpp}`)

| field | meaning |
|---|---|
| `ridgefill_p_end[seq]` (exists) | rows at pos < p_end are filled (bulk or exact R) |
| `ridgefill_rows[seq]` | `std::vector<llama_pos>`, sorted, ⊂ [0, p_end): the exact rows R. Empty = speed |
| `ridgefill_part[seq]`, `ridgefill_h_buf[seq]` | the sweep part and buffer for the next decodes of seq |
| `ridgefill_alpha` | context-wide, default `alpha_default` from the GGUF |
| `ridgefill_rho` | device tensors D, Nn `[H_v, n_late_gdn, n_seq_max]` F32, the quality ρ accumulators, reset when p_end is set |

The batch marker (`llama_batch_allocr::set_fill`, `std::vector<int8_t>`) becomes a class code per token, and
`split_simple`/`split_equal` keep a ubatch to one code (they already compare codes for equality): 0 stock,
1 filled single pass (bulk ∪ R), 2 PART_LOWER, 3 PART_LATE filled (bulk ∪ R), 4 PART_LATE exact (tail). The empty
vector stays the off path (no allocation, no extra split). The rows input (indices of R inside the ubatch) is built per
ubatch from `ridgefill_rows[seq]` (binary search over the ubatch's positions; per-ubatch host work only).

### API (`src/llama-ext.h`; L-A adds these, L-C adds its own sweep entries in its own files)

```c
// existing: llama_ridgefill_init / _free / _split, llama_set_ridgefill, llama_ridgefill_seq_set(ctx, seq, p_end)

// defaults from the GGUF
LLAMA_API float   llama_ridgefill_alpha_default (const struct llama_ridgefill * rf);
LLAMA_API int32_t llama_ridgefill_tail_default  (const struct llama_ridgefill * rf);
LLAMA_API int32_t llama_ridgefill_tail_min      (const struct llama_ridgefill * rf);
LLAMA_API float   llama_ridgefill_share_default (const struct llama_ridgefill * rf);
LLAMA_API bool    llama_ridgefill_digits_default(const struct llama_ridgefill * rf);
LLAMA_API float   llama_ridgefill_qattn_default (const struct llama_ridgefill * rf);
LLAMA_API const char * llama_ridgefill_desc     (const struct llama_ridgefill * rf); // "name (authors, license, doi)"

LLAMA_API void llama_ridgefill_set_alpha(struct llama_context * ctx, float alpha);

// class + digit rows of a prompt (SPEC §4, per prompt, not per chunk): writes the sorted positions < n_prefix,
// returns their count (<= n_prefix). tokens[i] is the token at position i.
LLAMA_API int32_t llama_ridgefill_select_rows(const struct llama_ridgefill * rf, const llama_token * tokens,
        int32_t n_prefix, float share, bool digits, llama_pos * out);

// the exact rows R of seq (sorted, < p_end; n = 0: speed mode). Set after llama_ridgefill_seq_set, before decoding.
LLAMA_API void llama_ridgefill_seq_set_rows(struct llama_context * ctx, llama_seq_id seq_id,
        const llama_pos * rows, int32_t n);

// sweep support: the next decodes of seq run only one part (see §3); h_buf as above. part FULL resets.
enum llama_ridgefill_part { LLAMA_RIDGEFILL_PART_FULL = 0, LLAMA_RIDGEFILL_PART_LOWER = 1, LLAMA_RIDGEFILL_PART_LATE = 2 };
LLAMA_API int32_t llama_ridgefill_seq_set_part(struct llama_context * ctx, llama_seq_id seq_id,
        enum llama_ridgefill_part part, struct ggml_tensor * h_buf); // 0 ok, < 0 refused (logged by name)
```

For L-C's question-mass graph: upstream already exposes layer inputs on the host
(`llama_set_embeddings_layer_inp(ctx, lid, true)` / `llama_get_embeddings_layer_inp`, `src/llama-ext.h`), for output
rows; in PART_LOWER the last 64 rows can be outputs (they are not filled), so the inputs of layers 3/7/11/15 for those
rows can come from there. The mass graph itself is L-C's (its own `llm_graph_context` subclass and files); it reads
the attention layers' K cache through `llama_kv_cache` views.

## 4. Ownership reminders and open items

- `tools/server/*` (per-request `ridgefill_mode`, calling `llama_ridgefill_select_rows`) and `llama-perplexity` /
  `llama-bench` are not assigned in HANDOVER §4. L-A will do the libllama side above; the coordinator should name an
  owner for the tool-side calls (proposal: L-B, next to the `common/` args).
- Rollback (`n_rs_seq > 0`): decision pending (T+2:30); either way it is internal to `build_ridgefill_late` and the
  split, nothing above changes.
- The rows / scale input shapes may get a dated correction once the masked path is built; the function signature will not.

## Correction 2026-10-20 18:25 (L-A, with the qwen35 speed fill)

- `llama_ridgefill_set_alpha` and `llama_ridgefill_seq_set_rows` return `int32_t` (0 ok, < 0 refused and logged by
  name) instead of `void`, so a refusal is visible to the caller. Both are in `src/llama-ext.h` now, with the
  default getters and `llama_ridgefill_desc`.
- Until the masked path lands, `llama_decode` returns -1 for a batch with filled rows of a sequence that has exact rows
  ("quality mode ... not implemented yet"); `llama_ridgefill_select_rows` and `llama_ridgefill_seq_set_part` are not
  in the build yet (they will not link, rather than return success).
- `llm_graph_params_ridgefill` also carries `alpha` (the speed-mode correction scale) and `p_end` (the context's
  per-sequence fill-end array) — both part of the reuse key; `h_buf` is not added until the parts are implemented.
- The correction-scale input is exactly as written in §2: F32 `[H_v, n_seqs, n_gdn_late]`; in speed mode every entry
  of a sequence whose last filled row is in the ubatch is `alpha`, every other entry 0.

## Correction 2026-10-20 18:45 (L-A, with the masked path)

- `llm_graph_params_ridgefill` also carries `rows_seq` (the context's per-sequence exact-row vectors, stable
  pointer; part of the reuse key with `n_rows`). `llm_graph_input_ridgefill` has a third tensor `pos` (I32
  `[n_R * n_pos]`, the rows' positions section-major, for Q's M-RoPE).
- `build_ridgefill_late(..., MASKED)` returns h_R after the last late layer but does not expand it (the graph prunes
  the last layer's R-only work); a caller that wants it (a dump) must `ggml_build_forward_expand` it.
- Quality mode refusals at decode: alpha != 0 (rho), parallel sequences on a non-unified KV cache.

## Correction 2026-10-20 19:15 (L-A, the two-sweep dispatch; supersedes §3 "h_S buffer" and the seq_set_part line)

```c
enum llama_ridgefill_part { LLAMA_RIDGEFILL_PART_FULL = 0, LLAMA_RIDGEFILL_PART_LOWER = 1, LLAMA_RIDGEFILL_PART_LATE = 2 };
struct llama_ridgefill_part_params {
    enum llama_ridgefill_part part;
    struct ggml_tensor * h_buf;                       // [n_embd, >= N], F32/F16/BF16, row p = position p
    int32_t n_win_layers; const int32_t * win_layers; // LOWER only: capture these layers' inputs (< split) ...
    struct ggml_tensor * const * win_bufs;            // ... into [n_embd, >= win_n] buffers, row = pos - win_begin
    llama_pos win_begin; int32_t win_n;
};
struct llama_ridgefill_part_params llama_ridgefill_part_default_params(void);
int32_t llama_ridgefill_seq_set_part(struct llama_context *, llama_seq_id, const struct llama_ridgefill_part_params *); // nullptr = FULL
```
- The copies happen inside the graph (`ggml_set_rows` / `ggml_get_rows` with a positions input), not on the host.
- Order (refused otherwise, by name): `llama_ridgefill_seq_set(seq, P)` -> `seq_set_part(LOWER)` on an empty sequence ->
  decode all N rows in order, no outputs -> (mass, then `llama_ridgefill_seq_set_rows(seq, R)`) ->
  `seq_set_part(LATE)` with the same h_buf -> decode all N rows again in order (outputs allowed for rows >= P) ->
  `seq_set_part(nullptr)` (FULL), which is refused until every LOWER row was replayed or the sequence is cleared.
- A LATE batch holds LATE rows of one sequence only; a batch holds LOWER rows of at most one sequence. A failed sweep
  ubatch clears the sequence and resets it to FULL. Refused: n_rs_seq > 0, LoRA, control vectors, embeddings.
- The window capture replaces the context-wide layer-input extraction for the question rows (item 2).

## Addition 2026-10-20 22:50 (L-A): chat-template row rules in the selector (CHAT-PLAN H1-H3), engine c6c485629

Libllama + C API only (`src/llama-ext.h`); the common / server / tools plumbing is L-B's, the window in the sweep
driver L-C's. Defaults reproduce today's selection exactly.

```c
struct llama_ridgefill_select_params {
    float   share;         // class rows: round-half-even(share x matches) rarest, [0, 1]
    bool    digits;        // digit rows
    bool    special_exact; // added-token rows                         (CHAT-PLAN H1)
    int32_t header_rows;   // rows after each added token, >= 0        (CHAT-PLAN H2)
};

// share = the projector's share_default, digits = its digits_default, special_exact = false, header_rows = 0
struct llama_ridgefill_select_params llama_ridgefill_select_default_params(const struct llama_ridgefill * rf);

// sorted, unique prefix positions into out (room for n_prefix); returns the count, < 0 refused (format-1 projector,
// null params, n_prefix < 0, share outside [0, 1], header_rows < 0, a token outside the vocabulary)
int32_t llama_ridgefill_select_rows_ext(const struct llama_ridgefill * rf, const llama_token * tokens,
        int32_t n_prefix, const struct llama_ridgefill_select_params * params, llama_pos * out);

// unchanged: = _ext with llama_ridgefill_select_default_params(rf) and the given share / digits
int32_t llama_ridgefill_select_rows(const struct llama_ridgefill * rf, const llama_token * tokens,
        int32_t n_prefix, float share, bool digits, llama_pos * out);

bool llama_ridgefill_token_is_special(const struct llama_ridgefill * rf, llama_token id);

// start of the last occurrence of the token sequence marker[0, n_marker) in tokens[0, n); -1 if none or bad args
int32_t llama_ridgefill_last_marker(const llama_token * tokens, int32_t n, const llama_token * marker, int32_t n_marker);
```

Rules (identical in lane T's kva driver):
- special(id) <=> the vocab attribute of id is control, user-defined or unknown <=> id is in the HF tokenizer's
  `added_tokens_decoder` (llama.cpp's converter: added special tokens -> control, added non-special such as <think> ->
  user-defined, <unk> -> unknown; vocabulary padding -> unused, not special).
- special_exact: every prefix row i < P with special(tokens[i]).
- header_rows K: rows i+1 .. min(i+K, P-1) after every prefix row i with special(tokens[i]); independent of
  special_exact. Both are united with the class and digit rows.
- Question window, last user turn (H3): start = llama_ridgefill_last_marker(tokens, N, marker) with marker = the tokens
  of "<|im_start|>user" (llama_tokenize with parse_special = true, no BOS); window = [max(start, end - cap), end),
  end = the prompt end (non-blind). No marker: the driver's documented fallback (last-W rows) or a named refusal —
  L-C's choice, mirrored in lane T's oracle.

For L-B (proposal; names are L-B's to settle): `--ridgefill-special-exact on|off` (default off),
`--ridgefill-header-rows K` (default 0), request fields `ridgefill_special_exact` / `ridgefill_header_rows`;
`common_ridgefill_rows` fills llama_ridgefill_select_params and calls _ext; llama-perplexity / llama-bench call _ext
with them or refuse them by name (never ignore them); refused with mode speed (no exact rows). For L-C: a question
window mode `last` (today) / `turn` (above) with a cap, window buffers sized per prompt.
Gate: rf27-gates/g1_select_chat.py, 42/42.

## Correction 2026-10-20 23:10 (L-A): the last-user-turn window as implemented (L-C 51d57b1fd)
Supersedes the H3 bullet of the 22:50 addition (the driver's rule, which lane T's oracle mirrors):
- `server_ridgefill_sweep_params.qattn_window_mode` = "last" (default: [max(0, end - qattn_window), end), as before) or
  "turn"; `qattn_turn_cap` (int32, default 256, > 0). end = qattn_end if set, else the prompt end (blind: prefix end).
- turn: marker = llama_tokenize("<|im_start|>user\n", add_special = false, parse_special = true) (n tokens, 1..64);
  start = llama_ridgefill_last_marker(tokens, end, marker, n) (search over [0, end) only);
  window = [max(start + n, end - qattn_turn_cap), end) — the header tokens themselves are not in the window.
- Named refusals: no marker before end; empty window (start + n >= end); turn with blind selection; mode other than
  last / turn; cap <= 0. No fallback to "last".
- Caveat for prompts: the marker must tokenize identically inside the prompt; a user turn whose content begins with a
  newline merges "\n\n" in the Qwen pre-tokenizer and is then refused (not silently misplaced).
- llama-ridgefill-score: --qattn-window-mode / --qattn-turn-cap, --special-exact / --header-rows (select_rows_ext);
  the chat options are refused with mode speed, turn needs quality+qattn.
