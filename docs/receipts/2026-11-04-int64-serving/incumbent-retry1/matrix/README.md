# Fresh incumbent baseline — November 4, 2026

The complete baseline passes its registered measurement and runtime gates.
Rebuilt-control and Int64-candidate serving trials are still pending; these
results alone establish no improvement or adoption decision.

The same benchmark invocation completed at 05:00:17 UTC with exit 0:
96 measured cells, 40 warmup cells and 510 recorded requests. The server
counted exactly 524 successful requests including its known functional and
initial warmup calls, with zero requests running or waiting. Both nodes retain
the incumbent image and the serving invocation is unchanged.

The full validator checks each request, not only aggregate success: registered
grid, matching warmup prompts, exact output budgets, cumulative usage/timing,
zero prefix-hit deltas in all 60 measured prefill cells, throughput arithmetic
and per-cell speculative-counter deltas all pass.

Maximum reasoning is explicit. Decode requests generate 200 tokens, including
reasoning, and need not finish an answer. These are throughput measurements,
not answer-quality scores. Concurrency 8 queues against max_num_seqs=4.

## Decode baseline

Each table entry is the median of three per-cell median rates. Rates use
(N−1)/(last−first usage update), the registered reciprocal-TPOT convention.

| Workload | Concurrency 1 | Concurrency 2 | Concurrency 4 | Concurrency 8 |
|---|---:|---:|---:|---:|
| code | 35.48 tok/s | 26.31 tok/s | 18.89 tok/s | 17.95 tok/s |
| prose | 38.04 tok/s | 25.67 tok/s | 18.64 tok/s | 17.64 tok/s |
| structured | 37.88 tok/s | 28.05 tok/s | 19.76 tok/s | 18.92 tok/s |

## Evidence and metric labels

- [Complete labeled table](matrix-labeled.md) preserves every measured value while correcting the presentation labels.
- [Original frozen-runner Markdown](matrix.md) is retained byte-for-byte, including its historical labels. Its aggregate decode column means sum(N−1)/cell wall time.
- [Original raw JSON, gzip](matrix.json.gz) retains all request timings, outputs, hashes and counter deltas. Uncompressed SHA256: `0e70c2dbf142bd1c3b71c5c176e172b7411e05512f5a8ab51df5d7886229e15b`.
- [Validation receipt](matrix-validation.json) identifies the exact validator and presentation renderer.
- [Terminal state and request accounting](matrix-terminal-state.json) records both image IDs, service identities, health and frozen-harness hash verification.
- [Supplementary streaming diagnostics](streaming-diagnostics.json) report bundle-aware and whole-cell generated-token rates. They do not replace the registered gate or infer individual token times inside streamed bundles.

The unchanged NVIDIA checkpoint revision is
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`: TP2, MTP5, FP8 KV 17 GiB/rank,
2,350,210 usable KV tokens, 1,000,000 context, chunk4096 and maxseq4.
See the [preceding runtime/API/prefix qualification](../../README.md#baseline-qualification-and-measured-run-activation)
and the [registered Int64 gate](../../../../gates/2026-11-03-dense-int64.md).
