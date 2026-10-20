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
