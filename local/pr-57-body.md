## Summary

Measures W4 fused-gather on the adopted W3 image and **reverts**. Gate/up A-tile loads from `x` + `row_token` + SUH so `h13` is not allocated. Gather host entry is dropped. Down mainloop keeps the W3 zero-fill A-pad.

Independent variable was IMAGE `e3-w3-zfill` → `e3-w4-fgather` (`sha256:946d4feeeb2a…`). `EXL3_FAT_GROUPED` stays 1. Compile-time `CUDA_VISIBLE_DEVICES=` is a RUN prefix only.

Default `overlay/` and `Dockerfile` stay W3-matched (gather + h13). W4 sources live in `overlay-w4/` and are consumed only by `Dockerfile.e3-w4-layer`. Historical `Dockerfile.e3-cubin-layer` / `Dockerfile.e3-py-layer` keep version-matched interfaces.

## Cluster receipts (same-boot A vs B)

| Gate | A `e3-w3-zfill` | B `e3-w4-fgather` `946d4feeeb2a` |
|---|---|---|
| Acceptance | — | 7/7 |
| Serving (:18000) | — | 6/6 |
| Pool | 1,396,551 / 1.40x | identical |
| 60k median tok/s | 1329.6 | 1187.0 (**−10.7%**) |
| 240k median tok/s | 1187.3 | 1149.8 (−3.2%) |
| Structured median | 70.36 @ 7.0/1.000 | 68.79 @ 7.0/1.000 (−2.2%) |
| Idle MemFree head/worker GiB | 10.6 / 10.4 after A-ladder | 7.0 / 3.8 idle; 5.6 / 4.0 after B-ladder |
| Health / bind | 200 / loopback | 200 / loopback |

Unique-prompt oversize matches W3 (~80k / ~319k). Cubin `exl3_fat_moe_ext.so` sha `76a077fbf081…` vs W3 `0afdfca7806d…`. `cuobjdump`: LOCAL:0, gateup REG 125 / down 128, STACK:16. Both ranks `grouped_ok`. Watchdog timer re-armed.

**Verdict: REVERT.** Expected sign was MemFree win, speed unknown. 60k −10.7% is past the −5% abort. Idle MemFree did not rise. The 8× redundant gather on a 16-row tile of a 128-K Hadamard is the speed cost. Production restored `IMAGE=glm53-selfbuild:e3-w3-zfill`, `EXL3_FAT_GROUPED=1`. Overlay-w4 stays in-tree as opt-in.

Rollback used last-wins `IMAGE=glm53-selfbuild:e3-w3-zfill` (`.env.bak-pre-w4-revert-20260908`).

## What this does **not** do

- Does not leave production on `e3-w4-fgather`.
- Does not flip TRF (still 128).
- Does not spend the 224 MiB `h13` reclaim on capacity.
- Does not combine with W1 (already REVERTED) or W5.
- Does not re-arm this gather-as-reload. A later attempt needs a persistent SUH-scaled A-cache across K.

Host tests: `pytest tests/test_exl3_grouped.py tests/test_numeric_config.py tests/test_exl3_v147_qualification.py tests/test_exl3_w4_optin.py` → **38 passed**.

CUDA review: APPROVED (implementation candidate). Final review: APPROVED after isolating W4 behind `overlay-w4/` so the default Dockerfile and W1/W3 layer recipes keep the W3 gather/h13 interface.
