# Streaming gap ledger

Card 13 only. Warm measurements use lane Q's fixed prompts, cold + full warmup excluded, three timed requests per arm. All paths are relative to this worktree. Source and binary SHA pins are retained beside each specification.

## Initial disjoint host-interval ledger

Controls: `trial1-16384` and `trial1b-32768`, legacy hash lookup and host mask; same frozen binary and flags per paired trace. Values are mean streamed minus resident ms. The signed API terms include useful GPU execution: synchronization is not a measurement of GPU idle. Async copy-worker time is excluded from this sum to avoid double counting.

| Host interval | 16K extra ms | 32K extra ms | Lower bound for added cost / transfer floor |
|---|---:|---:|---|
| carry_staging | 29.073 | 91.504 | H2D carry bytes / 7.04 GB/s; conversion adds work |
| compact_device_alloc | 0.113 | 0.118 | 0 added allocation time |
| compact_download | 62.065 | 116.086 | 39.6 / 79.1 ms (full hS / 4.24 GB/s) |
| compact_finish | 0.086 | 0.096 | 0 added bookkeeping time |
| compact_host_alloc | 0.026 | 0.034 | 0 added allocation time |
| compact_pack | 8.380 | 40.660 | packed payload / 26.25 GB/s; this memcpy rate already includes read+write |
| compact_upload | 20.327 | 41.817 | packed bytes / 7.04 GB/s |
| compact_wait | 0.026 | 0.056 | 0 additional wait |
| compute_submit | -2188.439 | -3580.440 | 0 additional API overhead; includes useful compute |
| graph_alloc | -517.061 | -658.266 | 0 additional API overhead; includes useful compute |
| graph_build | 16.508 | 33.201 | 0 additional host graph work |
| graph_reset | 27.976 | 58.941 | 0 additional host graph work |
| memory_apply | 0.515 | 0.387 | 0 additional host work |
| replay_lookup | 812.497 | 3210.015 | 0 map reconstruction; maintained-index lookup measured ~7.5 us/call |
| set_inputs | 26.860 | 166.436 | 0 extra mask upload for eligible dense layout |
| synchronize | 2499.965 | 2837.787 | 0 added wait; includes useful compute |
| tail_upload | 6.663 | 6.660 | 5.215 ms (35 MiB cold tail / 7.04 GB/s) |
| untraced | 13.616 | 406.583 | 0 additional dispatch; unresolved residual, not device idle |
| upload | 12.253 | 12.436 | 7.92 ms (two maps / 7.04 GB/s) |
| upload_wait | 0.146 | 0.177 | 0 exposed worker wait |
| **Sum / measured mean prompt gap** | **831.595** | **2784.289** | |

The sum is checked against the server prompt-time difference to 1e-5 ms in `ledger4.py`. Raw resident and streamed interval totals, counts and each repetition are in the two `ledger.json` files. `untraced` is the prompt-time residual after disjoint instrumented intervals; it is not attributed to a particular device operation. Combining allocation/submission/synchronization gives -205.535 ms at16K and -1400.919 ms at32K: long host lookup work overlapped useful GPU execution. Its large standalone time was therefore not all removable from TTFT.

## Measured transfer controls

`link-routes.json` and `ledger-16384/link.json`: five32MiB copies, mapped synchronous H2D median5.933ms (5.66GB/s), async plain6.741ms, async pinned4.768ms (7.04GB/s). Pinned D2H11.496ms was worse than the mapped read control7.913ms (4.24GB/s). CPU memcpy1.278ms (26.25GB/s). Transfer data were byte-checked. These are measured API routes, not theoretical bus bandwidth.

Map payload is1,672,396,800B/prefill at both lengths:60maps, two copied on the main thread,58 prefetched. Mapped-copy floor~295.7ms total, pinned-DMA floor~237.6ms total; most is already overlapped. Initial worker sums were294.97/287.76ms, exposed joins only0.146/0.177ms. Two main-thread maps total55,746,560B: mapped floor9.85ms, pinned floor7.92ms, measured12.25/12.44ms. Do not add the entire background payload floor to the critical-path sum.

hS link counters:16K H2D mean203,192,320B, D2H167,772,160B;32K H2D mean423,144,107B, D2H335,544,320B. These exclude model loading, normal inputs/masks and backend-internal transfers. Compact/carry/tail upload floors partition these bytes; they are not independent extra payloads.

Floor accounting corrections: the cold exact tail is3584 rows, not one512-row chunk:35MiB upload per prefill, floor5.215ms at the optimistic pinned route. The compact carry allocation includes10MiB for the first-tail residual, but its initial seed upload contains only exact prefix rows. At16K timed0: compact95610880B, seed70942720B, cold tail36700160B, totaling203253760B. Do not double-count the carry's uninitialized first-tail allocation as a transfer. CPU memcpy bandwidth is payload bytes/time; multiplying its payload by two again would double-count read/write traffic. Floors for graph building, queue stalls, bookkeeping and dispatch are ideal zero added work, not measured hardware minima. The remaining gap is not established to be a transfer floor.

## Causes and retained changes

The largest measured time lever was the full host attention mask: band-only device construction saves256.284/1747.028ms median at16K/32K in the same-binary `final-{16384,32768}` host-mask control. It uses existing ggml operations, with host fallback for incompatible cache layouts. These causal differences are separate comparisons, not extra terms to add to the disjoint ledger.

Rebuilding and destroying a full replay hash table every chunk consumed812/3210ms of host time, much of it overlapping compute. Reusing the maintained position index reduces512-query allocations16398 ->3 and avoids that work; its standalone TTFT gain was small. Dense row traversal also reduces CPU membership time13.335 ->0.637us/call (three reps), but no GPU gain is claimed: its16K auxiliary off-arm median was faster, and its third request overlapped an offline parser as recorded in NOTES-A.md.

Early hS readback contended with question/window reads on Vulkan's device mutex. Retiming it after the first S1 fill submission overlaps compute: the native trace overlaps7.271ms of an8.749ms read with compute submissions; compaction join is0.008ms. Warm16K synchronous/retimed medians12.660066/12.619858s. Moving the copy early without this retiming is rejected. No backend or kernel changed.

Ordered independent question-mass layer graphs preserve per-head FP32 operations and the original FP64 accumulation order. They reduce measured scratch74.376 ->72.876MiB. Offloading the four captures (5MiB total) to host and staging one1.25MiB window before its original device normalization saves another3.516MiB of global peak, bringing the intermediate32K overhead to509.375MiB. Pinned transfer code was not retained: its measured route advantage was small relative to mask work, and pinned D2H was slower.

## Final warm gate and reconciled ledger

Card13, same fixed lane-Q prompts, cold plus a full warmup excluded, three timed requests per arm. Server prompt timings are reported; one-token client wall times are retained in rows.json. Unprofiled arms decide the timing gate. Profiled arms supply the additive ledger and are reported separately.

| Length | Resident median [min,max] ms | Streamed median [min,max] ms | Extra | Stock median ms | Streamed-stock peak MiB |
|---|---:|---:|---:|---:|---:|
| 16384 | 12307.590 [12141.827,12368.287] | 12640.417 [12452.901,12715.866] | 2.704% | 15243.475 | 262.914062 |
| 32768 | 24925.490 [24745.203,24960.737] | 25439.724 [25366.909,25572.551] | 2.063% | 32567.352 | 509.140625 |

Both timing gates and the512MiB peak gate pass. This is the requested stop condition; the remaining gap is not claimed to be a transfer floor.

The following values are three-repetition means of streamed minus resident disjoint host intervals. Nested diagnostics and asynchronous workers are excluded. Queue APIs include useful GPU work, so their signed changes must be considered together, not called GPU idle. Zero floors denote ideal zero *added* work; no empirical CPU or queue minimum is claimed.

| Main-thread interval | 16K extra ms | 32K extra ms | Added-cost floor / scope |
|---|---:|---:|---|
| budget_estimate | 12.585 | 24.987 | ideal 0 added host work; tiny CPU control 0.623 us/call |
| carry_staging | 40.360 | 97.816 | exact-prefix F32 seed bytes / 7.04 GB/s; conversion adds work |
| compact_device_alloc | 0.108 | 0.119 | ideal 0 added allocation; hardware minimum not isolated |
| compact_download | 0.004 | 0.004 | 39.6 / 79.1 ms full read service; exposed join can be 0 |
| compact_finish | 0.065 | 0.100 | ideal 0 added bookkeeping |
| compact_pack | 20.098 | 43.258 | compact payload / 26.25 GB/s memcpy rate; scatter adds work |
| compact_upload | 19.932 | 43.279 | compact payload / 7.04 GB/s optimistic pinned route |
| compact_wait | 0.022 | 0.052 | ideal 0 extra wait |
| compute_submit | -2512.325 | -4672.145 | 0 extra API/stall cost; includes useful GPU work |
| graph_alloc | 209.584 | 1980.899 | 0 extra API/stall cost; includes useful GPU work |
| graph_build | 27.068 | 63.231 | ideal 0 added graph work; hardware minimum not isolated |
| graph_reset | 30.803 | 54.608 | ideal 0 added graph work; hardware minimum not isolated |
| memory_apply | 0.197 | 0.432 | ideal 0 added bookkeeping |
| replay_lookup | 34.339 | 67.673 | 0 full-index reconstruction; lookup service remains |
| row_lookup | 2.414 | 5.080 | 0 extra traversal; CPU micro median 0.637 us/call |
| set_inputs | -70.165 | -321.209 | 0 full host mask upload in eligible layouts; small indices remain |
| synchronize | 2503.360 | 2836.276 | 0 extra wait; includes useful GPU work |
| tail_upload | 6.677 | 6.688 | 5.215 ms (35 MiB / 7.04 GB/s) |
| untraced | 2.646 | 303.693 | ideal 0 added dispatch; unattributed, not measured GPU idle |
| upload | 12.550 | 12.720 | 9.85 ms actual mapped route; 7.92 ms optimistic pinned route |
| upload_wait | 0.151 | 0.187 | 0 exposed worker wait |
| window_offload | 1.551 | 1.561 | 1.237 ms (5 MiB / 4.24 GB/s) |
| window_upload | 1.198 | 1.252 | 0.745 ms (5 MiB / 7.04 GB/s optimistic route) |
| **Sum / measured mean prompt gap** | **343.224** | **550.561** | |

Allocation/submission/synchronization combined are +200.619/+145.029ms. The complete terms reconcile to the measured mean gap within1e-5ms before display rounding (`ledger4.py`). Untraced is an explicit residual; it is not assigned to transfers or stalls.

| Diagnostic service time (excluded from sum) | 16K streamed ms | 32K streamed ms |
|---|---:|---:|
| upload_worker | 287.545 | 287.553 |
| download_worker | 66.870 | 128.587 |
| row_lookup | 5.740 | 12.127 |
| carry_rows | 2.600 | 5.112 |
| replay_lookup | 35.523 | 70.068 |
| budget_estimate | 14.594 | 29.252 |

| Payload service floor (actual mapped / best tested pinned H2D) | 16K ms | 32K ms |
|---|---:|---:|
| exact seed | 12.522 / 10.063 | 30.975 / 24.893 |
| compact bulk | 16.917 / 13.595 | 37.355 / 30.020 |
| cold tail | 6.489 / 5.215 | 6.489 / 5.215 |

These are payload service floors, not extra elapsed-time terms. Seed conversion/scatter and allocation add work. D2H full hS floors are39.565/79.130ms; measured service is reported above, while its exposed join is almost zero. Main-thread maps have a9.856ms mapped service floor (7.922ms best pinned) plus at least2.123ms CPU staging on the current plain-source route. Background maps total1,616,650,240B and are already overlapped. Four lower captures add5MiB each direction, with1.237ms read plus0.927ms mapped-write service floors (0.745ms best pinned write).


Replay, row and budget diagnostics can be nested in other host intervals. Background read/upload service is not added to elapsed time a second time. The main-thread compact_download interval is now the future get/join, whereas download_worker records the actual full read.

| API count per prefill | 16K resident / streamed | 32K resident / streamed |
|---|---:|---:|
| graph_build | 63 / 1260 | 127 / 2828 |
| graph_alloc | 63 / 1260 | 127 / 2828 |
| compute_submit | 90 / 1529 | 186 / 3481 |
| synchronize | 12 / 143 | 12 / 143 |
| set_inputs | 90 / 1529 | 186 / 3481 |

Byte gates:80 native CPU checks default and traversal-off,24 separate-KV checks,80 Vulkan checks traversal-off/on; four readback/window combinations and joint/ordered mass controls match complete prefill/final state, five full-vocabulary logits, raw lower/S1 masses, selected rows and scales. The full27B pair contains four lower attention layers,2000 mass rows and419086772 serialized prefill bytes. Default C1 matches6519d9245 on128 generated tokens,515 characters and128 probability entries; peak changes -16384B. All warm RF response payloads at both lengths match resident.

Replay allocation regression:512 queries in occupied16K/32K caches allocate3 times; the legacy control fails at16398 allocations. Dense/sparse/mixed ownership, repeated/duplicate/negative positions, missing rows, highest duplicate cells, invalid owners, odd F16 mask offsets and cancellation during pending readback are checked. No timing threshold is used in unit tests.

Raw: `row-merge-{16384,32768}/{analysis,ledger,result}.json`, per-arm rows/responses/commands/environments/server logs/VRAM samples, `row-merge-*-spec.json`, `row-merge-pins.json`, `row-merge.patch`, `row-merge-smoke/identity.json`, `c1-row-merge-identity/identity.json`, and `link-routes.json` plus `ledger-16384/link.json`. Frozen binaries are `row-merge-bin`; builds are `build-cpu` and `build-vulkan-dev`. Bridge jobs are stream-row-merge-16/32, each card13 only, with same-job fresh stock. No long evaluation set was run.
