# Speed matrix — nvidia/GLM-5.3-Flash-FP4-Local

Started: 2026-10-17T13:25:49.014205+00:00

## Prefill, no cache (TTFT basis)

| Conc | Prompt tokens | Req | TTFT med s | TTFT p90 s | Prefill tok/s med | Aggregate tok/s | Mem avail after GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|

## Decode

| Conc | Decode tokens | Completion med | Decode tok/s med | Aggregate tok/s | Mean accept len | Draft accept rate | Drafts | Mem avail after GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 200 | 46.76 | 42.43 | 3.685 | 0.537 | 54 | 2.23 |
| 2 | 200 | 200 | 44.54 | 77.08 | 4.914 | 0.783 | 81 | 2.23 |
| 4 | 200 | 200 | 20.25 | 46.39 | 2.671 | 0.334 | 298 | 2.23 |
| 8 | 200 | 200 | 29.77 | 81.80 | 3.873 | 0.575 | 411 | 2.23 |

## Notes

- decode c=8 len=200: concurrency 8 > max_num_seqs 4; queued
