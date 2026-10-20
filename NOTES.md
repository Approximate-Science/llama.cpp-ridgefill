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
