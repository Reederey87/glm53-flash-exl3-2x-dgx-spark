# MTP3 versus MTP5 — November 3, 2026

MTP3 passes the registered performance and functional gates and is recommended
for adoption. MTP5 has been restored and its watchdog re-armed; final MTP3
adoption awaits the user’s decision. No image or kernel changed.

Both arms completed 96 measured cells and 40 warmups. The comparator finds no
prefill, decode or draft-acceptance regression exceeding 2% in two repetitions.
Code decode improves in all three repetitions at every concurrency; median
gains exceed the corresponding control range. Structured-output effects remain
inside substantial control variability and are not claimed as improvements.

These are fixed 200-generated-token workloads with explicit maximum reasoning,
often unfinished reasoning. They do not measure finished-answer quality or
latency. Concurrency 8 queues against max_num_seqs=4.

## Decode medians

Rates use the frozen reciprocal-TPOT convention: (N−1)/(last−first usage update).
Control range is (maximum−minimum)/median across three repetitions.

| Workload | Concurrency | MTP5 tok/s | MTP3 tok/s | Change | Control range | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| code | 1 | 36.20 | 40.03 | +10.58% | 0.54% | gain above control range |
| code | 2 | 25.93 | 27.31 | +5.31% | 2.09% | gain above control range |
| code | 4 | 18.67 | 20.89 | +11.89% | 3.22% | gain above control range |
| code | 8 | 19.27 | 20.44 | +6.06% | 1.93% | gain above control range |
| prose | 1 | 37.68 | 41.14 | +9.18% | 9.22% | within control range |
| prose | 2 | 26.35 | 28.67 | +8.79% | 0.25% | gain above control range |
| prose | 4 | 18.07 | 19.50 | +7.92% | 7.60% | gain above control range |
| prose | 8 | 17.54 | 19.72 | +12.45% | 5.37% | gain above control range |
| structured | 1 | 35.85 | 40.90 | +14.07% | 28.88% | within control range |
| structured | 2 | 27.86 | 27.97 | +0.40% | 5.44% | within control range |
| structured | 4 | 18.37 | 20.24 | +10.18% | 18.32% | within control range |
| structured | 8 | 19.62 | 19.89 | +1.38% | 9.18% | within control range |

## No-cache prefill medians

Every measured prefill has zero prefix-cache-hit delta. Observed median gains
range from 1.78% to 2.18%; no per-cell regression meets the rejection rule.

| Concurrency | Prompt tokens | MTP5 tok/s | MTP3 tok/s | Change |
|---:|---:|---:|---:|---:|
| 1 | 8192 | 2270.3 | 2313.0 | +1.88% |
| 1 | 16384 | 2251.8 | 2295.8 | +1.96% |
| 1 | 32768 | 2237.9 | 2281.3 | +1.94% |
| 1 | 65536 | 2212.3 | 2251.6 | +1.78% |
| 1 | 131072 | 2169.8 | 2208.8 | +1.80% |
| 2 | 8192 | 1327.6 | 1352.1 | +1.84% |
| 2 | 16384 | 1467.8 | 1493.9 | +1.78% |
| 2 | 32768 | 1556.8 | 1586.5 | +1.90% |
| 2 | 65536 | 1595.1 | 1625.8 | +1.93% |
| 2 | 131072 | 1595.2 | 1625.3 | +1.89% |
| 4 | 8192 | 781.6 | 796.1 | +1.86% |
| 4 | 16384 | 849.8 | 865.3 | +1.82% |
| 4 | 32768 | 884.9 | 902.4 | +1.98% |
| 4 | 65536 | 898.2 | 916.4 | +2.03% |
| 4 | 131072 | 892.0 | 910.7 | +2.09% |
| 8 | 8192 | 459.4 | 469.1 | +2.12% |
| 8 | 16384 | 479.9 | 489.9 | +2.08% |
| 8 | 32768 | 489.0 | 499.6 | +2.18% |
| 8 | 65536 | 490.4 | 500.9 | +2.14% |
| 8 | 131072 | 485.1 | 495.0 | +2.05% |

## Controls and functional evidence

- Same NVIDIA NVFP4 checkpoint revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` and image `a7f76ae9…` on both ranks; actual serving arguments verify the sole MTP5→MTP3 change.
- TP2, 1,000,000 context, FP8 KV, 17 GiB per rank, and 2,350,210 usable KV tokens remain unchanged.
- All six API smoke checks pass in both arms, including nonempty modern reasoning and parsed tool calls. Four cold/hot prefix pairs pass in each arm.
- An extra numeric-only-format diagnostic fails in both arms: the model returns a correct explanation and 391 rather than only the number. It is retained separately and is not called a format pass.
- Draft acceptance rates rise in all paired comparisons; mean acceptance length is reported but has different maxima at depth 3 and 5.

## Metric conventions and reproducibility

The raw tables’ aggregate decode column is sum(N−1)/cell wall time. It is a
normalized rate, not all-generated-token throughput. The comparator also
reports sum(N)/cell wall and (N−first-update-count)/(last−first update) as
separate diagnostics. Some initial updates contain several tokens; individual
token generation times inside those bundles are not observed. These diagnostics
do not replace the preregistered gate.

See [the preregistration](../../gates/2026-11-03-mtp3.md), [all paired comparisons](comparison.json),
[raw MTP5 tables](mtp5.md), [raw MTP3 tables](mtp3.md), and [source/receipt hashes](manifest.json).
The compressed JSON files retain the original bytes, including every request’s
token counts, timestamps, output text and every acceptance-counter delta.

```bash
mkdir -p local/mtp-depth-review
gzip -dc docs/receipts/2026-11-03-mtp-depth/mtp5.json.gz > local/mtp-depth-review/mtp5.json
gzip -dc docs/receipts/2026-11-03-mtp-depth/mtp3.json.gz > local/mtp-depth-review/mtp3.json
uv run --frozen --no-dev scripts/compare_mtp_matrix.py \
  local/mtp-depth-review/mtp5.json local/mtp-depth-review/mtp3.json \
  --out local/mtp-depth-review/comparison.json
```

## Verified restoration

MTP5 was restored through the unit after both containers stopped and both
nodes exceeded 90 GiB MemFree. The original recipe/unit hashes, actual MTP5
arguments, checkpoint metadata, image, environments and mounts match the saved
controls. All six API checks and four cold/hot prefix pairs pass; KV remains
17 GiB per rank and exactly 2,350,210 usable tokens. At 16:09:20 UTC the service
and watchdog timer were active, the watchdog probe exited successfully, and
`/health` returned 200. See [restoration evidence](restoration.json) and the
`restored-*` receipts. No final MTP3 adoption has been made.
