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

## 19:15-19:25 dispatch decision, llama_ridgefill_select_rows
- Decision on L-C's 8de161386 (+ driver 68fe8a45d): keep 78d4a2551 (mine), with 8de161386's sequencing and
  refusals already adopted; the driver ports to `llama_ridgefill_seq_set_part(ctx, seq, &params)` and takes the
  question rows' layer inputs from the window capture (upstream's extraction is unchanged and still works, but it
  copies every row of the last decode).
- Implemented `llama_ridgefill_select_rows(rf, tokens, n_prefix, share, digits, out)`: the oracle's class_rows
  (lane T's q27/rf27_quality.py: round-half-even(share x matches) rarest finite-score rows, ties by lower position)
  united with select.digits (digit set built at load from token text, control / user-defined / unknown / unused
  tokens excluded). G1 (`rf27-gates/g1_select_rows.py`, oracle functions imported, the tiny fixture's rowsel table
  and HF tokenizer): 23/23 — shares 0, 0.1, 0.25, 0.5, 1 x digits off/on x P 500 and 699 identical row sets
  (254 class matches, 20 digit rows in [0, 500)); share and digit controls change the set; a format-1 projector is
  refused. A first run failed the P = 700 half: the tool refuses --tail 0, so nothing ran there (now P = 699).
- Not checked: the digit set on the real Qwen3.8 vocabulary against kva's digit_ids (needs the HF tokenizer.json,
  sha 0997f410..., which is not on this box) — a follow-up for whoever holds the tokenizer (lane T has kva).

## 19:25-19:45 integration branch rf27-integ
- `rf27-integ` = rf27-engine (ff598e03d) + merge of rf27-tools (L-B, tip 0c1a20276): no conflicts; L-B's diff
  touches no src/ file. CPU build (-j4) clean.
- L-B's plumbing still refuses quality mode in three places written before the selector existed
  (`common_ridgefill_rows` in common/common.cpp, llama-perplexity, llama-bench: "requires
  llama_ridgefill_select_rows, which is unavailable in this build"). Fail-closed and correct for now; left to L-B
  (their files) to call `llama_ridgefill_select_rows`.
- Gates on the merged tree: L-B's canonical G1 (tests/ridgefill/run_gate1_qwen35.py, rf_dump built from the merged
  tests/ridgefill/rf_dump.cpp against build-cpu, fixtures byte-identical to L-B's out/): speed 27/27 (fill worst
  6.31e-07), quality 27/27 (fill worst 4.05e-07, incl. alpha-1 refusal and wrong-rows / speed-instead-of-quality
  controls); G0 72/0; rollback 11/11; G1 quality a0 27/27; dispatch 44/44; select_rows 23/23.

## 19:45-19:55 rf27-integ: rf27-tools be962e03a merged forward
- Merge 4638252bc (no conflicts): `common_ridgefill_rows` now calls `llama_ridgefill_select_rows` (validated inputs,
  checked output). Build clean. Canonical G1 speed 27/27, quality 27/27; G0 72/0; select_rows 23/23; L-B's
  tests/ridgefill/test_row_selection.cpp (built here against build-cpu + libllama-common): "PASS: 21 common
  class/digit selector checks". Lane T (via the coordinator): the digit rule equals kva's digit_ids on the real
  Qwen3.8 vocabulary (10 ids, 248,077 pieces, no difference). llama-bench / llama-perplexity quality mode still
  refused in L-B's files until their next commit.

## 19:28-19:40 rf27-integ: L-C's Tier-3 driver (c9d3bbdbf) and L-B's 23da65965 merged
- Review of c9d3bbdbf as core owner: libllama changes are only L-C's own `src/llama-ridgefill-sweep.{h,cpp}` + one
  `src/CMakeLists.txt` line; no L-A file changed. The mass graph builds its own scheduler from the context's backends
  once per prompt (per-prompt host work, not per token). The server driver: `seq_set_part` with the params struct,
  window capture into per-slot device buffers, one `llama_decode` per sweep step, slot restart from n_past 0 (LOWER
  needs an empty sequence), `seq_rm` on cancel, refusals for speculative / LoRA / child tasks. Merged (44d4828b8);
  3 conflicts in tools/server/server-context.cpp, all L-B's interim "needs the sweep driver" refusals vs L-C's
  driver: took L-C's side; the merged common/ tools/ tests/ trees equal c9d3bbdbf exactly. Then 23da65965 (L-B:
  class+digit quality in llama-bench / llama-perplexity) merged cleanly.
- L-C's harnesses (local in its build-sweep/, copied unchanged into this repo's ignored build-sweep/, bin -> build-cpu)
  on the merged tree: native Tier-3 G1 split 4 at 50/50, 128/128, 700/512: mass 2.66e-07, base rows 147/147, union
  217/217, 25 arrays worst 6.63e-07; defer / repeat pass; wrong projector 1.34; share 0 == quality byte-identical.
  HTTP: check-http-sweep 3/3 (share 0.2 x1, share 0 x1 with zero == quality, share 0.2 x2 slots); shutdown-recheck
  3/3; check-http-cancel 3 of 4 runs: run 1 (load average 15) failed its shutdown step — after SIGTERM during an
  active 3500-token sweep the server logged "stop: cancel task" 1 s later and then nothing for 10 s (the test's
  limit). Server lifecycle code (L-C's); reported, not fixed here. Runs 2-4 pass (~0.3 s each).
- Full gate set on the merged tree: canonical G1 speed 27/27, quality 27/27; G0 72/0; rollback 11/11; quality a0
  27/27; dispatch 44/44; select_rows 23/23. The server log shows "graphs reused = 114" for a qattn prompt (the
  dispatch's reusable part graphs).

## 19:40 rf27-integ: rf27-tools tip 6622ae3ad merged (NOTES-tools.md only on top of 23da65965)
- Code tree identical to 962f8c3f0. Rerun: canonical G1 speed 27/27, quality 27/27; G0 72/0; L-C native Tier-3 G1
  12/12 runs + 3/3 control rows; dispatch 44/44; select_rows 23/23.

## 19:45-20:15 G5 CPU, independent reproduction of L-B's claims on rf27-integ dbc6afea6 (tree 713119574096)
Second instrument, written from L-B's claims (NOTES-tools.md G5), not from its scripts.
- Clean archive build: `git archive dbc6afea6` into an empty directory, built under `env -i` (PATH = the cmake binary
  dir + /usr/bin:/bin, HOME a fresh directory, GIT_CEILING_DIRECTORIES set), Release, GGML_NATIVE=OFF, curl/OpenSSL/UI/
  tests off, -j4: llama-server, llama-bench, llama-perplexity built, 0 errors. The archive finds no git repository
  ("commit unknown", build 0: it cannot see this checkout) and `ldd` shows it links only its own build's libraries.
  Two false starts, both instrument errors: cmake from ~/.local/bin was outside the clean PATH (rc 127), then that
  cmake is a pip wrapper that needs the user site-packages under HOME (used its real binary directory instead).
- Refusals (`rf27-gates/g5_refusals.py`, own HTTP client): L-B's 15 claims + 2 of mine, 17/17 on the archive build
  and 17/17 on the dev build: startup refusals named for LoRA, control vector, embeddings, --cache-reuse, -sm row;
  bench refusals for -sm row and embeddings; stock missing-LoRA control fails in the normal loader without a RidgeFill
  message; request n_cache_reuse refused (400); fill active 500/700; filled-slot save refused (501) with no file;
  off request exact (0 filled); exact-slot save works (300876-byte file); mine: a dense qwen3 projector on the qwen35
  model refused by architecture, quality with alpha 1 refused naming rho. Cosmetic: the alpha-1 message prints
  "RidgeFill: RidgeFill:" (common/, L-B's).
- Portability grep (own patterns in `rf27-gates/g5-portability-patterns.txt`: home / mount / Windows paths, user and
  host names, PCI ids, *_VISIBLE_DEVICES, GPU model names, device indices, GPU UUIDs, TP sizes, the storage and
  container names of this box), over every code file this fork changes vs upstream 1242be66f (66 files: L-B's 36
  plus llama-batch, the KV cache and recurrent/hybrid memory, llama-context.h, the llama/qwen3 hooks, rf27-gates/,
  tests/ridgefill/), on the archive; positive control (a planted file, 2 hits) and negative control (upstream
  llama-vocab.cpp, 0) first. Engine, common, tools, converter, tests/ridgefill code: 0 hits — agrees with L-B and
  covers the files its list misses. Hits: my own rf27-gates scripts defaulted to this machine's fixture / kva / venv
  paths (12 lines) — fixed: they now refuse by name unless the paths are given (env or argv); all five rerun with
  explicit inputs: G0 72/0, rollback 11/11, quality a0 27/27, dispatch 44/44, select_rows 23/23. Benign remaining:
  .gitignore "AGENTS.local.md" (pattern false positive), run_g5.py's own pattern list and planted controls,
  g0_dense.sh clearing *_VISIBLE_DEVICES to force the CPU.
  First grep attempt reported 0 hits for everything: `grep -P -f` takes one pattern only and the error was hidden by
  2>/dev/null; the planted-file positive control is why that would not pass again.
- Functional parity (`rf27-gates/g5_parity.py`): archive vs dev server, greedy 8 tokens with top-5 logprobs on the
  700-token prompt, modes off / speed / quality: same text, same chosen and top-5 ids, max |delta logprob| 4.77e-07,
  fill counts 0 / 500 / 500, 4/4. Not byte-identical, and mode off (stock code) differs by the same 4.77e-07: the
  archive is GGML_NATIVE=OFF, the dev build native (AVX-512), so the SIMD rounding differs, not RidgeFill. Two
  instrument bugs on the way: probabilities merge partial-UTF-8 byte tokens (fewer than n_predict entries), and the
  response's `tokens` field is empty without return_tokens (an equality of two empty lists checked nothing).
- Agreement with L-B: archive build, the 15 controls and the empty grep over its 36 files reproduce. Not covered by
  either instrument here: container isolation, Vulkan / HIP / CUDA execution, multi-GPU layer split (lane Q).

## 20:20 rf27-integ: rf27-tools 20261fbe3 merged (afdaa3e26)
- One conflict, NOTES-tools.md (a rebased copy of L-B's already-merged notes commit): took L-B's version; L-B's files
  equal 20261fbe3. Code change: 5 lines in llama-bench (a RidgeFill summary line). Build clean.
- Gates: canonical G1 speed 27/27, quality 27/27; G0 72/0; L-C native Tier-3 G1 12/12 runs + 3/3 controls; dispatch
  44/44; select_rows 23/23.
- My portability grep (72 changed code files) disagrees with L-B's "empty over 44 sources" on the new GPU smoke
  scripts, which L-B's patterns do not cover (no GPU model / target names): tests/ridgefill/check_gpu_smoke.py
  `--two-card` accepts only devices named R9700 / Radeon AI PRO 9700; tests/ridgefill/run_g5_gpu.sh fixes
  `-DAMDGPU_TARGETS=gfx1201` and requires ROCm 7.10. Product code is unaffected (engine/common/tools: 0 hits); these
  are this box's validation harness, but shipped in tests/ they should take the device-name pattern, the AMDGPU
  target and the SDK version as arguments. L-B's files: reported, not edited.

## 20:25-20:55 rf27-integ: L-C's shutdown fix (rf27-sweep 276da13af) reviewed as an upstream server change, merged
- (1) Upstream bug, independent of RidgeFill: before this fix the fork had not changed tools/server/server.cpp or
  server-queue.* at all (diff vs 1242be66f empty). Upstream's signal handler calls `shutdown_handler` synchronously;
  for a model server that is `ctx_server.terminate()` -> `server_queue::terminate()`, which locks `mutex_tasks`
  (std::mutex). The main thread holds `mutex_tasks` at 17 sites in server-queue.cpp, including `yield_to_queue()`,
  which upstream itself calls 5x in normal prompt/decode processing. A signal delivered there self-deadlocks; the
  handler also calls fprintf/exit (not async-signal-safe). L-C's sweep stages yield once per ubatch (often), which
  made it frequent; L-C's deterministic negative control compiled against the pre-fix server.cpp (= upstream's)
  times out. Measured here on upstream 1242be66f without RidgeFill, double SIGINT during a generation (tiny qwen35,
  10 runs): exit 1 x8, SIGABRT x1, SIGSEGV x1; one further run in the G0 harness hung 20 s. The fork: exit 1, 10/10.
- (2) Behaviour with RidgeFill off DOES change (upstream llama-server signal handling, every mode):
  * SIGTERM/SIGINT dispatch: the shutdown callback runs on a dispatcher thread that polls every 100 ms, not in the
    handler; measured idle exit 15-31 ms (fork) vs 3 ms (upstream);
  * one extra thread for the server's lifetime, waking every 100 ms;
  * a second interrupt: `std::_Exit(1)` without the "Received second interrupt, terminating immediately." message
    (upstream: message + exit(1), which crashed or hung in 2-3 of 11 runs above); exit code 1 either way;
  * embedded CLI use (is_run_by_cli) unchanged. Windows not run (its console handler calls the same signal_handler).
  Request/response behaviour unchanged: new `rf27-gates/g0_server.py` (upstream vs fork llama-server, no
  --ridgefill; 3 prompts + a cache hit, greedy 16 tokens with return_tokens and top-5 probabilities, every field but
  timings; /props; signals) PASS 8/8 on tiny Llama, tiny qwen35, tiny qwen35 with 2 slots: responses and /props
  identical; SIGTERM idle exit 0 both; SIGTERM during a generation: both wait for the in-flight request (same);
  second SIGINT: fork exit 1 in 3 ms. Instrument fixes on the way, none counted as passes: the random per-process
  media marker (pinned via LLAMA_MEDIA_MARKER), partial-UTF-8 tokens without probabilities (guard on return_tokens),
  a busy case that was idle (probe /slots), a stray server holding a fixed port (free ports, fail on bind error).
- L-C's HTTP checks on the merged tree: check-http-sweep 3/3, shutdown-recheck 3/3, check-http-cancel 5/5 plus
  10/10 with four busy-loop processes running beside the server (the run that stalled before was under load).
- Quick gates: canonical G1 speed 27/27, quality 27/27; G0 72/0; L-C native Tier-3 G1 12/12 + 3/3; dispatch 44/44;
  select_rows 23/23.

## 20:55 rf27-integ: rf27-tools dd48f86ff merged (GPU smoke scripts without hardware pins)
- Clean merge; tests/ridgefill and docs only (no C++). My extended grep (own patterns, planted 2/2 and upstream 0/0
  controls first) over the 74 changed code files of the committed tree (archive): the R9700 / gfx1201 / ROCm 7.10
  hits are gone (now --device-name-pattern / --hip-targets / --sdk-min-version, defaults: any GPU, CMake-detected
  targets, SDK minimum only); the remaining 4 hits are the known benign ones (.gitignore AGENTS.local.md,
  run_g5.py's pattern list and planted controls, g0_dense.sh clearing *_VISIBLE_DEVICES). Agrees with L-B's empty
  result over the same 72+ files (its patterns do not flag those 4).
- L-B's tests/ridgefill/test_gpu_smoke.py here (run from its directory, with <repo>/out/ created): 6 of 7 tests
  pass; test_extended_audit_positive_and_neutral_controls errors because run_g5.py's audit runs `rg` (ripgrep) as a
  subprocess and this sandbox has no rg executable (only a shell function). An environment dependency, not a logic
  failure: run_g5.py needs ripgrep installed (worth stating in tests/ridgefill/README.md or a grep fallback; L-B's).
  An earlier version of this note said the tests ran; it was written before reading the result.

## 21:00 rf27-integ: rf27-tools c578a8f6c merged (portability audit falls back to Python without ripgrep)
- Clean merge, tests/ridgefill + notes only (no C++). L-B's GPU smoke unit suite here: 8/8 OK; this sandbox has no rg
  executable (shutil.which('rg') is None), so the run exercised the Python fallback, including its positive control.

## 21:05 rf27-integ: rf27-sweep 36bf1218f merged (L-C's single-pass NLL scorer llama-ridgefill-score)
- Review: new tools/ridgefill/ (score.cpp, run_nll.py, check_score.py, README, CMake) + one tools/CMakeLists.txt line
  + tools/server/server-ridgefill-sweep.{h,cpp}. The scorer uses only the C API (llama-ext.h) and L-C's sweep
  driver. Server hunks: new fields qattn_end / logits_begin / logits_end default to -1, which the server keeps
  (`sweep_params = {}`; server-context.cpp unchanged); with -1 every changed expression equals the old one (the
  new logits term is false, n_outputs counted from the flags equals the old count, the window end falls back to the
  old blind / prompt-end choice, the tail's output_last is unchanged) and the new validations accept the defaults:
  no behaviour change for existing modes. Clean merge; CPU build clean (llama-ridgefill-score built).
- check_score.py on the merged tree: "PASS native scorer: 24/24 four-mode oracle comparisons, two K values, three
  chunk sizes", worst logprob/LSE error 1.81e-07, plus its window / input / named refusals.
- Quick gates: canonical G1 speed 27/27, quality 27/27; G0 72/0; G0 server vs upstream 8/8 (qwen35); L-C native
  Tier-3 G1 12/12 + 3/3 (harness recompiled against the changed sweep source); L-C HTTP sweep 3/3; dispatch 44/44;
  select_rows 23/23.
## 2026-10-20 22:20-22:45 chat-template row rules (CHAT-PLAN H1-H3), libllama side
- `llama_ridgefill_select_rows_ext(rf, tokens, n_prefix, &params, out)` with `llama_ridgefill_select_params`
  {share, digits, special_exact, header_rows}; `llama_ridgefill_select_default_params(rf)` (share / digits from the
  projector, special off, header 0); the old `llama_ridgefill_select_rows` is unchanged (it calls _ext with those
  defaults). Defaults stay as today until lane T confirms.
- The rule, exactly (for lane T's kva driver): special(id) <=> the vocab attribute of id is control, user-defined or
  unknown <=> id is in the HF tokenizer's `added_tokens_decoder` (llama.cpp's converter writes added special tokens as
  control, added non-special ones such as <think> as user-defined, <unk> as unknown; vocabulary padding is unused
  and not special). special_exact: every prefix row i < P with special(tokens[i]). header_rows K: rows i+1 .. i+K,
  capped below P, after every prefix row i with special(tokens[i]) — independent of special_exact. Union with the
  class and digit rows; ascending. Built at load as a host table next to the digit set.
- Question window (H3): `llama_ridgefill_last_marker(tokens, n, marker, n_marker)` = the start of the last
  occurrence of a token sequence (the caller tokenizes e.g. "<|im_start|>user" with special parsing), -1 if none.
  The window [start, end) and its cap are the sweep driver's (L-C's file).
- G1 (`rf27-gates/g1_select_chat.py`, expectations from the HF tokenizer's added_tokens_decoder + lane T's
  class_rows + kva digits, never from the GGUF): 42/42 — a crafted 700-token prompt (8 added tokens in the prefix:
  adjacent pairs, <unk>, the last prefix row, one in the tail) over share {0, 0.5} x digits x special x header
  {0..3} = 32 identical row sets; controls: special and header change the set on the crafted prompt, change nothing
  on the fixture's own prompt (no added tokens); header rows stop at the prefix end; the masked fill with the 158
  selected rows vs kva finish(exact_rows=R) 6.96e-07; last_marker vs a Python search on 5 markers (incl. none and the
  last row). Regression: select_rows 23/23, G0 72/0.
- rf27-integ after merging c6c485629 (a40a4ad17): build clean; g1_select_chat 42/42; canonical G1 speed 27/27,
  quality 27/27; G0 72/0; G0 server 8/8; L-C native Tier-3 G1 12/12 + 3/3; check_score 24/24; dispatch 44/44;
  select_rows 23/23. Plumbing (flags / per-request fields) not done here: specified for L-B, window for L-C.

## 23:10-23:30 rf27-integ: L-C's last-user-turn window + chat rows in the scorer (rf27-sweep 51d57b1fd) merged
- Review: libllama untouched (tools/ridgefill, tools/server/server-ridgefill-sweep.{h,cpp}, notes). Defaults
  (qattn_window_mode "last", qattn_turn_cap 256) give win_end = the old end expression and win_begin =
  max(0, end - qattn_window) as before, same window-buffer size, mass end unchanged, new validations accept them:
  a no-op for existing modes. Turn rule vs my 22:50 note: the marker is "<|im_start|>user\n" (newline included) and
  the window starts after it, [max(start + n, end - cap), end); missing marker / empty turn / turn+blind are named
  refusals (no fallback). L-C's choice for the driver; RF27-INTERFACE.md corrected (23:10) so lane T mirrors it.
  The scorer now calls select_rows_ext directly (defaults select the same rows; chat knobs refused with speed).
- Gates on the merged tree: canonical G1 speed 27/27, quality 27/27; G0 72/0; L-C native Tier-3 G1 12/12 + 3/3;
  check_score 24/24; g1_select_chat 42/42; select_rows 23/23; dispatch 44/44; L-C HTTP sweep (defaults) 3/3;
  L-C check_chat_score.py against a baseline scorer built from a clean archive of d619c13d6: all PASS (defaults
  byte-identical old == new == explicit, raw and templated, four modes, F32/F16; 8 chat configurations incl. caps 32
  / 256 / 1024 across 3 chunk sizes; controls; paired wrapper; named refusals).

## 2026-10-21 rf27-s1: the S1 window selector and vtail in libllama (C API v2)
- Interface first: RF27-INTERFACE.md "Addition 2026-10-21" (2e06536c2). Implementation: parts S1_FILL (a replay of
  [0, P) writing only the late attention layers' K/V from the projector; late GDN layers skipped; graph flag
  s1_attn_only) and S1_PROBE (the window [s1_begin, s1_end) through the late layers from h_S; the late attention
  layers written out non-FA (kq, KQ mask + an s1_bias that hides the unrun tail cells [P, s1_begin), soft_max_ext,
  kqv, gate, wo) so the softmax is available; the late GDN layers start from zero at s1_begin via init_replay's new
  zero_pos; no output head). The mass per KV cell (sum over heads, queries, late attention layers) is an F32 graph
  output, mapped cell -> position on the host (llama_kv_cache_context::get_cell_pos) and summed across ubatches in
  double. vtail: on the LATE ubatch that starts at P, per late attention layer, s_h = mean over its first c rows of
  ||V(attn_norm x)|| / ||V(attn_norm(W h_S + b))||, then cache V *= (bulk ? s_h : 1) via an in-graph cpy before the
  layer's own attention (bulk = positions < P not in the exact rows; F32 [n_kv] masks from the cell positions; both
  V layouts). Scales are read back for reporting.
- Refusals (named): S1_PROBE window starting in the prefix ("blind selection or a window longer than the tail is not
  supported"), S1_PROBE without a complete S1_FILL, LATE after S1_FILL without a complete probe, FULL mid-S1, S1 rows
  as outputs, vtail with the first tail ubatch < c_eff rows or n_ubatch < c_eff, vtail on a single-pass (FULL part)
  tail. llama_ridgefill_seq_set clears the S1 mass and resets vtail to off (per prompt, like the rows).
- Found while testing: llm_graph_input_mem_hybrid::set_input aborted on the unallocated rs s_copy of a graph that runs
  no recurrent layer (S1_FILL); guarded like the kq_mask / k_idxs guards.
- G1 (`rf27-gates/g1_s1_vtail.py REF_DIR`, lane T's dumps engine-ref/s1-vtail at lane T 4eb0a8d, SHA256SUMS checked):
  49/49. Both prompts x T3s, T3s-vtail-k0/k72, S1gw, S1gw-vtail-k0/k72: 25 arrays each (logits, final every layer,
  prefix late layers before vtail) worst 6.18e-07; S1 mass 1.21e-07; S1 rows identical (217 fixture, 222 selchat =
  220 + 2 special; k-th / k+1-th reference mass gap 1.6e-04 / 7.3e-05: no ties); vtail scales <= 1.2e-7 relative
  (k0 <-> c = 200, k72 <-> c = 128). Chunking: S1 with n_ubatch 32 (probe in 2 ubatches) and vtail with
  c = n_ubatch = 128 (tail in 2 ubatches) also match. Controls: vtail moves late V by 7% and logits by 3%; c changes
  the scales; S1 rows differ from T3s (88 / 95 shared). Refusals 5/5 by name. T3s rows and n_base come from the
  reference (the qattn part of T3s is the driver's, L-C), so this G1 covers libllama's mass, top-k input and fill.
- Regression on rf27-s1 (options off): canonical G1 speed 27/27, quality 27/27; G0 72/0; G0 server 8/8; L-C native
  Tier-3 G1 12/12 + 3/3; check_score 24/24; g1_select_chat 42/42; select_rows 23/23; dispatch 44/44; quality a0 27/27;
  rollback 11/11.
- Cost (for L-B's reporting): S1_FILL computes the projector + K/V of the late attention layers for P rows, and LATE
  computes them again (reuse is a follow-up); S1_PROBE runs W rows through every late layer with a non-FA attention
  over n_kv cells. vtail: per late attention layer, c rows x (projector + attn_norm + V projection) twice, plus one
  read-modify-write of that layer's V cache over n_kv cells.
- Graph reuse: the S1_PROBE and vtail graphs are never reused (their masks are sized by n_kv); other graphs unchanged.

## 2026-10-21 rf27-s1: L-C's S1/vtail orchestration (rf27-sweep-s1 cad20681e) merged
- Review against the C API v2 note: only L-C's files (src/llama-ridgefill-sweep.*, tools/server/server-ridgefill-sweep.*,
  tools/ridgefill/). Order LOWER -> n_base (base rows u qattn top) -> S1_FILL [0, P) -> S1_PROBE [end - W, end) ->
  top max(1, half-even(F x n_base)) capped at P, ties to the lower position, u special rows -> set_rows -> set_vtail ->
  LATE -> vtail off -> FULL (cancel goes through the same path). Driver checks: S1 not blind, W inside the known exact
  tail, API version >= 2 at build and run time, n_batch / n_ubatch >= c, deferred last token not consumed by
  calibration. The first tail decode is widened to c rows. vtail-only and S1 with qattn share 0 run the two-sweep flow
  (no window capture). No objection; merged as 202b047a1 (tree == cad20681e).
- Flag for L-C (not edited: their file): tools/ridgefill/README.md example commands carry this machine's absolute
  paths (/var/home/..., /var/mnt/..., ../llama.cpp-ridgefill-tools); some predate this commit, the new S1 example adds
  more. The portability grep would hit them before a release.
- Gates on the merged tree: G1 S1/vtail 49/49; L-C check_s1_score.py run here (baseline scorer built from a clean
  archive of b9a8e5add, same CMake options; fixture rf27-gates/lb copy): PASS 24 + 12 native, worst logprob 1.76e-07,
  mass 1.36e-07; canonical G1 speed 27/27, quality 27/27; G0 72/0; G0 server 8/8; L-C native Tier-3 G1 12/12 + 3/3;
  check_score 24/24; g1_select_chat 42/42; select_rows 23/23; dispatch 44/44; quality a0 27/27; rollback 11/11; L-C
  HTTP sweep (defaults) 3/3.

## 2026-10-21 rf27-s1: L-C 705d6bf3c (README portability) and L-B 0ca58254d (S1/vtail CLI, HTTP, reporting) merged
- 705d6bf3c: README only; machine paths replaced by placeholder variables; the portability patterns
  (rf27-gates/g5-portability-patterns.txt, positive control planted) find nothing in tools/ridgefill/README.md.
  Merged as 7d60dd6f1.
- 0ca58254d review (L-B's branch already contains 705d6bf3c through its own merge, which resolved conflicts in L-C's
  score.cpp / README and also edits check_s1_score.py and run_nll.py). The decisions hold:
  - S1/vtail run only in the quality+qattn two-sweep flow: common_ridgefill_validate_options refuses them by name in
    speed / quality; llama-cli, llama-bench and llama-perplexity refuse them; non-qwen35, header rows with S1, a
    blind window, qattn share 0 with S1, n_rs_seq > 0 and non-causal contexts are refused by name; API v2 is checked
    at run time.
  - Order: seq_set before vtail-on (the driver's); vtail off before FULL.
  - Server prompt cache: a cached fill is reused only with the same mode, rows and vtail row count (c), so a request
    with vtail off never reuses rescaled V; the cached prompt state carries c.
  - Reporting: common prices S1_FILL (P rows of projector + K/V, recomputed), the probe (W rows through every late
    layer + 2 x n_head x d x the dense query/cell pairs the driver counted) and vtail (c rows of projector + V
    projections) once; the scorer requires driver cost == report cost (1e-12) and no longer adds them itself.
  - Behaviour to know: with S1 or vtail enabled, a prompt too short to fill, or one whose tail is <= the vtail rows
    (the server defers its last token), is refused by name instead of running exactly (fail closed, no silent
    shortening).
  - Flags: (1) L-B's commits edit L-C-owned files (tools/ridgefill/score.cpp, check_s1_score.py, run_nll.py, README):
    the scorer's S1 flags now use the --ridgefill-* names, --s1-window defaults to 0 (--s1 on sets 256) and a refusal
    needle in check_s1_score.py is looser ("S1 window"); L-C should acknowledge. (2) L-B's commit messages carry
    "Assisted-by: Codex" instead of the campaign trailer; not changed here.
  Merged as abc4117e2 (tree == 0ca58254d + my NOTES / results).
- Gates on abc4117e2 (CPU, -j4 build clean): G1 S1/vtail 49/49; canonical G1 speed 27/27, quality 27/27; G0 72/0;
  G0 server 8/8; L-C native Tier-3 G1 12/12 + 3/3; check_score 24/24; g1_select_chat 42/42; select_rows 23/23;
  dispatch 44/44; quality a0 27/27; rollback 11/11; L-C check_s1_score PASS 24 + 12 native (worst logprob 1.76e-07,
  mass 1.36e-07; baseline = clean archive of b9a8e5add); L-C HTTP sweep defaults 3/3. L-B's checks run here
  independently: check_s1_tools 115/115 (architecture controls 3/3 with the dense tiny-qwen3 fixture),
  check_s1_http 72/72, check_s1_reporting 91 checks / 16 configs, worst relative error 0.
- Vulkan build of the merged tree (build only, nothing run on the GPU: it is busy with lane T; lane Q validates after
  a winner is confirmed): build-vulkan-d5916c0cf (ignored), lane Q's configure options (GGML_VULKAN ON, native,
  Release, Vulkan headers from ~/.local/include, loader /usr/lib64/libvulkan.so.1 1.4.341, glslc), -j4: rc 0, 0 errors,
  52 warnings all in upstream tools/mtmd / libstdc++ headers. llama-server, llama-ridgefill-score, llama-bench,
  llama-perplexity and libggml-vulkan built; libllama exports the v2 entry points (checked with nm, not executed).
- Observation (not new): llama-ext.h is upstream's C++-only extension header, so the whole RidgeFill "C API" (v1 and
  v2) has C++ linkage (mangled symbols). Every current caller is C++; a C caller or FFI binding would need extern "C"
  wrappers. Not changed: an API decision for the coordinator.
