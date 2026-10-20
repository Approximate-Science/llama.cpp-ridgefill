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
