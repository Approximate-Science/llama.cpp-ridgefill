# NOTES (lane L-A engine, branch rf27-engine)

Measurements and rejected options, newest last. Times are local (CDT), 2026-10-20.

## 17:45-18:00 interface
- `~/.claude/skills/build-anything/SKILL.md` is not readable from this sandbox; the four rules restated in the brief
  (nothing unimplemented returns success, a test that checks nothing is SKIPPED, fail closed and refuse by name, read a
  file's header before changing it) are followed as written.
- Rejected: `build_ridgefill_late` as a member of `llm_graph_context` (it needs `build_conv_state` / `build_delta_net`
  of `llm_build_delta_net_base`; a cast from the base is UB for other archs) and as a member of
  `llm_build_delta_net_base` (needs `src/models/models.h`, not owned). Chosen: a free function taking the builder.
- Rejected: checking the manifest's `tokenizer_json_sha256` in the engine (a GGUF has no `tokenizer.json`). Chosen: a
  new `adapter.ridgefill.vocab_sha256` over `tokenizer.ggml.tokens`, recomputed from `model.vocab`.
- GDN state layout (ggml-cpu `gated_delta_net`): memory `s[h][j][i]`, key index i fastest, i.e. ggml ne
  [K, V, H]; the oracle's `rec` is [H, dk, dv]. The `st` orientation is pinned in RF27-INTERFACE.md §1.
- Upstream moved: `origin/master` is `1242be66f` (266 commits past `de7fa0a3c`), 12 past the lane's `1ba568c5a`.

## 18:00-18:20 rebase, build, G0
- Rebased onto `1242be66f` (origin/master at 17:47): 4 conflicts in llama-context.{h,cpp}, llama-graph.{h,cpp}, all
  adjacent insertions (upstream's `hdmd` Hadamard field next to `ridgefill`), kept both. One semantic change: qwen3
  moved into `build_backbone()` (shared with `vllm-decision2`); the hook returns nullptr there (72d79baa7).
  Pre-rebase branch kept as `rf27-engine-prerebase`.
- CPU build (`build-cpu`, Release, GGML_NATIVE, -j4): rc 0.
- G0 + dense regression (`rf27-gates/g0_dense.sh`, upstream built from `git archive 1242be66f`): 62 PASS / 4 FAIL on
  the first run; the 4 FAILs were all perplexity rows and were the harness: llama.cpp's async logger interleaves the
  timestamped "Final estimate" line with the chunk list, so a line-anchored grep dropped `[33]` on one side. Fixed with
  `grep -o` and a required `[33]` on the stock side; perplexity rows rerun (`ONLY=ppl`): 4 PASS / 0 FAIL. Run directly,
  both binaries print the same 33 chunks and PPL 301.4395 (llama) / 305.1462 (qwen3).
- Dense fill vs kva after the rebase, FA off: llama logits rel 4.3e-7-level as on Oct 8; qwen3 4.32e-07 / KV 2.80e-06.

## 18:15-18:25 qwen35 speed fill, G1 speed
- Implemented: format-2 loader (model_meta compared key by key, unknown key refused; vocab_sha256 recomputed from
  `model.vocab`; `model_layer_types` checked when present; `st` F32 [S_k, S_v, H_v] required per late GDN layer;
  score table and digit set on the host; a GDN layer without a map = identity fill, warned), `build_ridgefill_late`
  speed fill (attention: K/V via build_qkv, k-norm, `ggml_rope_multi(sections)`, cpy_k/cpy_v; GDN: wqkv only, beta,
  gate, build_conv_state, ssm_conv + silu, l2 norms, the delta rule, state written back; output dropped), the alpha
  correction on the ubatch holding row p_end-1, the qwen35 hook (+11 lines).
- Correction path: on the correction ubatch the fill calls `build_delta_net` itself and stores
  `new_state + scale * st` into the current slot, instead of `build_recurrent_attn` (whose store is internal; a
  second add on the cache view after its cpy would be an ordering hazard across backends). On every other filled
  ubatch it calls `build_recurrent_attn` unchanged, so rollback snapshots are written as stock. Rejected: undo/redo
  of the correction per chunk (radiance) — the homogeneous split makes "the ubatch holding p_end-1" unique.
- Upstream trap: `llm_graph_input_mem_hybrid::set_input` wrote the KQ mask unconditionally; a fill graph never
  attends, so the mask is not allocated -> `GGML_ASSERT(buffer)`. Added the same `->buffer` guard upstream already
  has in `llm_graph_input_attn_kv::set_input` (src/llama-graph.cpp, 2 lines).
- G1 speed (L-B's run_gate1_qwen35.py from rf27-tools e0e1fa9, run here against build-cpu with `--dump`): 27/27
  PASS. Fill vs oracle max rel 6.31e-07 (alpha 0 and 1, 3 chunk configs); prefix GDN states at alpha 1 2.0e-08..4.6e-08;
  wrong map 2.88 / 1.53; wrong st orientation 1.29-1.37 on the prefix states (its final states match because the
  200-token tail decays the difference; its logits miss by 0.34). Results: rf27-gates/results-g1-qwen35-speed.json.
- G0 rerun with the qwen35 fixture (byte-identical to upstream 1242be66f, no projector and T >= N, 3 chunk configs):
  72/0 in total. A first rerun showed 2 FAIL refuse-output rows: my `ONLY=ppl` edit made `$?` read a test's status;
  fixed (rc captured), both rows PASS with rc 2.
- Quality mode refuses by name at decode (exact rows set): rc 2, "exact rows (quality mode), which this build does not
  implement yet".

## 18:25-18:35 rollback decision (n_rs_seq > 0)
- Decision: keep rollback on (no forced n_rs_seq = 0); filled ubatches write the stock snapshots through
  `build_recurrent_attn`; the correction ubatch writes the current slot only.
- Bug found and fixed (src/llama-batch.cpp `split_equal`): with n_rs_seq > 0 the keep-tail rule keeps a sequence's
  last n_rs_seq + 1 tokens of a batch in one ubatch; when a batch crossed P and held fewer exact tokens than that,
  its "extend the sequence to completion" branch appended tail tokens to the filled ubatch. The mixed ubatch had
  outputs, so it ran stock: the filled rows were silently computed exactly (rs4 vs rs0: 1.73 / 1.64 max rel at
  batch 694/695, T = 8). Fix: the rule counts only the remaining tokens of the current fill class (a filled run never
  shares a ubatch with the next run). No effect when nothing is filled (the marker vector is empty).
- Test (`rf27-gates/g1_rollback.sh`, L-B's fixture): L-B's G1 speed harness with n_rs_seq = 4 (27/27 PASS);
  batches crossing P (`--cross`, as the server submits them) at 700/512, 694/512, 695/128, 128/128, 50/50 with T = 8:
  n_rs_seq 4 == n_rs_seq 0 (max rel 0.0, same fused GDN op), control fill != stock 1.42 everywhere. 11/11.
  Negative check: with only the split fix reverted, exactly the two crossing rows (694, 695) FAIL.
- Rejected: refusing outputs within n_rs_seq + 1 exact tokens of the fill end. Too strict (it refused every harness
  run scoring all tail rows and would refuse a server batch that happens to end a few tokens after P), and not
  needed: stock does not shift snapshot slots either, a rollback is valid only into the tokens of the last ubatch
  ("older slots are caller-owned"), and callers roll back drafts, never prompt rows. Documented in llama-ext.h.
- rf27-gates/rf_dump_rs.cpp = L-B's rf_dump.cpp + `--n-rs-seq`, `--cross` (L-B's file untouched).

## 18:35-18:45 masked path (quality mode), G1 quality at alpha 0
- Implemented in `build_ridgefill_late` (MASKED): per late layer `x_mix = set_rows(W h_S + c, h_R, rows)`; norm; K/V
  (attention) or conv + scan (GDN) for all rows from x_mix; for rows R only: attention Q + output gate via build_qkv
  on `get_rows(norm(x_mix), rows)`, q-norm, `ggml_rope_multi` at R's positions (an input gathered from the ubatch's
  section-major pos), attention over the cache with the KQ-mask rows of R (`ggml_get_rows` on the ubatch mask; F16
  cast with FA), gate, wo; GDN: the scan's output gathered at R, z from wqkv_gate, gated RMS norm, ssm_out; then
  residual, attn_post_norm, MLP, residual -> h_R. h_R after the last late layer is not expanded, so that layer's
  R-only attention output / ssm_out / MLP are pruned (they write no cache and a filled ubatch has no logits).
- What a masked chunk skips (SPEC §4 asks every lane to say): for bulk rows, Q, attention, the output gate, wo, z, the
  gated norm, ssm_out, the residual and the MLP of every late layer. Computed for all rows: the map, attn_norm, K/V
  (+ k-norm, RoPE) or the GDN in-projection (qkv, beta, alpha), conv and scan. For rows R: everything except the
  last late layer's post-mixer part. Not measured for speed yet (lane Q, G4).
- Context: a ubatch with any exact row gets fill MASKED and n_rows (part of the reuse key); rows are found per
  ubatch by binary search in the per-sequence sorted R (per-ubatch host work). Refused by name at decode: quality
  with alpha != 0 (rho-scaled correction not implemented), and quality with parallel sequences on a non-unified KV
  cache (the KQ-mask rows of R are gathered from one stream).
- G1 quality alpha 0 (`rf27-gates/g1_quality_a0.py`, L-B's reference/compare unchanged): 27/27. Oracle agreement
  at 700/512, 128/128, 50/50 for R1 = {0,7,31,127,255,499} (4.05e-07) and R2 = {3,64,65,66,200,383,384,450}
  (5.12e-07); quality != speed (1.73); wrong map (1.48); R1 run vs the R2 oracle fails (1.52); T >= N byte-identical
  to stock; FA on agrees at 1e-2 (1.7e-03, 7.2e-04); two sequences in the same ubatches (R1 + R2; R1 + speed) agree
  per sequence (<= 7.1e-07); alpha 1 refused by name (rc 2).
- L-B's harness in quality mode stops at its own T >= N control (it passes --rows with no prefix; rf_dump refuses)
  and would then run alpha 1, which the engine refuses; asked L-B (via the coordinator) for both.
- G0 rerun: 71/1 — FAIL `ppl-tgen qwen3`: chunks 1-32 identical, [33] and the final line missing on the branch side.
  Not reproducible (6/6 direct runs complete and identical, ONLY=ppl rerun 4/4); it ran while my quality tests
  loaded the CPU. Suspected the async log losing its tail at exit; unconfirmed, the script did not keep the exit
  code. It now records each run's rc and shows it in a FAIL line. Rollback 11/11, G1 speed 27/27 after the change.

## 18:43-18:50 G0 rerun with exit codes (closing the ppl-tgen qwen3 FAIL)
- Full G0 rerun, nothing else of mine running: 71/1, this time FAIL `ppl-off llama` (rc 0 on both sides, 32/34
  lines): the branch log ends after `[32]` with rc 0. Same signature as the earlier `ppl-tgen qwen3` FAIL.
- Measured: 40 runs each of upstream (1242be66f) and branch llama-perplexity on the tiny Llama, 4 in parallel:
  "Final estimate" missing in 6/40 (upstream) and 4/40 (branch), rc 0 in all 80. Upstream's async logger drops its
  last lines at exit under load; the computed values are identical whenever printed. Not a fork defect.
- Harness: the trap is named in g0_dense.sh's header; a log without "Final estimate" is rerun (<= 3 attempts, the
  count printed); comparison unchanged (all 33 chunks + the final line on both sides). Rerun: 72/0, attempts 1/1 on
  all four perplexity rows.

## 18:50-19:15 two-sweep dispatch (L-C's items 1-3), compared with L-C's draft 8de161386
- Implemented: `llama_ridgefill_seq_set_part(ctx, seq, &params)` with LOWER (layers [0, split), h_S of row p ->
  h_buf row p via `ggml_set_rows`; optional window capture of chosen lower layers' inputs at [win_begin, +win_n)
  into caller buffers) and LATE (a replay over positions already decoded: h_S via `ggml_get_rows`, then the fill for
  rows < p_end — bulk, exact rows R, the correction on the ubatch holding p_end-1 — and the stock late layers + head
  for rows >= p_end). Batch marker classes 2 / 3 / 4. Replay in the memory classes: `llama_kv_cache::find_slot_replay`
  (the cells that already hold the positions) + a replay `llama_kv_cache_context` whose apply() skips apply_ubatch
  (it would purge the sequence's earlier positions) and refreshes n_kv; `llama_memory_recurrent::find_slot_replay`
  (the sequence's own cell, position unchanged, rs_z = the cell when the replay starts at the KV cache's first
  position); `llama_memory_hybrid::init_replay`. A replay batch is validated without the memory position check.
- Upstream traps: the hybrid input wrote the KV row indices unconditionally (a LOWER graph below split 3 has no
  attention layer): `->buffer` guards as in `llm_graph_input_attn_kv`; a LOWER graph has no RoPE, so `inp_pos` is
  expanded in its return path. The hybrid memory's `seq_pos_min` is the recurrent cell's position (699, not 0): the
  replay cursor uses the attention cache's.
- Compared with L-C's draft (same core: existing cells, no apply_ubatch, rs_z at pos 0). Kept mine: replay inside the
  memory classes (L-C subclasses the hybrid context and writes recurrent internals), graphs reusable across
  ubatches (L-C bakes the buffer offset into the graph and its reuse key -> a rebuild per ubatch), window capture
  into caller buffers (L-C changes upstream's layer-input extraction). Adopted from L-C: per-sequence cursors (rows
  in order; LOWER on an empty sequence; LATE needs a complete LOWER pass with the same h_buf; FULL refused in the
  middle of a sweep), refusals for n_rs_seq > 0 / LoRA / control vectors / embeddings, clearing the sequence when a
  sweep ubatch fails, and the split rule "a part ubatch holds one sequence". Not adopted yet: LATE rows mixed with
  other sequences' rows in one batch (refused by name; the driver issues a replay ubatch as its own llama_decode),
  several LOWER sequences in one batch (refused by name until tested).
- G1 sweep dispatch (`rf27-gates/g1_sweep.py`, L-B's compare unchanged): 44/44. Sweep vs kva at 700/512, 128/128,
  50/50 for speed alpha 0 / 1 and quality R1: 6.31e-07 / 6.31e-07 / 4.05e-07; sweep == single pass (max rel 0.0);
  lower caches byte-equal to the single pass; KV cells stay 700; h_S and window captures byte-equal to upstream's
  `llama_get_embeddings_layer_inp` (which copies every token of the last decode in batch order, not the output
  rows: my first comparison read it wrong and failed 3 rows); wrong map 1.15; T >= N sweep == stock (0.0); after a
  stale prompt in the same sequence: oracle agreement, and the late states after a 2-row first replay batch
  byte-equal to a cold context. Negative check: with the zeroing disabled that row differs by 0.59 (with a 128-row
  first batch it did not: the tiny model's gates decay 1e-3-scale stale states below float resolution, recorded so
  nobody reads the 128-row variant as a test). Refusals by name: LOWER output, LATE without LOWER, LOWER on a
  non-empty sequence, FULL mid-sweep, a part on a dense model.
- Regression after the change: G0 72/0, rollback 11/11, G1 speed 27/27 (L-B's harness), G1 quality a0 27/27.
