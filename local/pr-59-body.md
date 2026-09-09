## Summary

Adopts isolated env-only `EXL3_TEMP_ROWS_FUSED=32` vs production E3@128 (task 24 W2). No cubin rebuild. Independent variable is TRF only. Frozen: `IMAGE=e3-w3-zfill`, `EXL3_FAT_GROUPED=1`, `EXL3_MOE_ROW_TILE=0`, pin / C4 / k=7 / adaptive-k ema.

Fused `exl3_moe` skips `token_count > cap`. Decode `tokens <= cap` returns with **no fat fallback**. Unique-per-token top-k (`grouped_topk` / `torch.topk`) keeps hottest ≤ T. C4 T=32 therefore does **not** drop experts (`32 > 32` is false). CPU judge `scripts/probe_w2_unique_topk.py` fail-closes without live evidence and pins the production router AST fingerprint (`tests/fixtures/w2-grouped-topk-router.py`).

`start.sh` still defaults TRF=128 and refuses TRF below `MAX_NUM_SEQS*(DFLASH_TOKENS+1)` on dflash. Production last-wins 32.

## Cluster receipts (same-boot A vs B)

| Gate | A `TRF=128` | B `TRF=32` |
|---|---|---|
| Acceptance | (healthy) | 7/7 |
| Serving (:18000) | — | 6/6 |
| Pool | 1,396,551 / 1.40× | identical |
| 60k median tok/s | 1253.1 | **1454.3 (+16.1%)** |
| 240k median tok/s | 1242.4 | **1407.8 (+13.3%)** |
| Structured n=9 | 66.60 @ 7.0/1.000 | 69.10 @ 7.0/1.000 |
| Hashmap prose n=9 | 27.54 | **30.82** |
| Idle/post MemFree head/worker GiB | 5.00 / 3.67 post-A | 4.46 / 4.56 idle; 3.77 / 4.04 post-B |
| Health / bind | 200 / loopback | 200 / loopback |

Both ranks `temp_rows_fused()=32`, `grouped_ok`. No CUDA/Xid/IMA. Watchdog re-armed.

**Verdict: ADOPT.** Expected sign was cold-prefill lean-lose (S1 at TRF=64). Measured was a fat-path-share win at the C-decode floor. Rollback: last-wins `EXL3_TEMP_ROWS_FUSED=128` (`.env.bak-pre-task24-w2-20260909`).

## What this does **not** do

- Does not rebuild cubin or switch `IMAGE`.
- Does not combine with W1 (REVERTED) or W4 (REVERTED).
- Does not change launcher default (still 128). GHCR/old copies stay fail-closed until they set the env.
- Does not open W5 occupancy (ncu-gated).

Host tests: `pytest tests/test_exl3_grouped.py tests/test_numeric_config.py tests/test_w2_unique_topk_probe.py` → **44 passed**. Final review: APPROVED after fail-closed unique-topk judge (live evidence + AST fingerprint).
