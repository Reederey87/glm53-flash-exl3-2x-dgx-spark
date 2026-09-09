## Summary

Follow-up to merged [#59](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark/pull/59). Aligns README standing numbers with the W2 TRF=32 cluster receipts. No runtime, overlay, or env change.

#59 adopted `EXL3_TEMP_ROWS_FUSED=32` and updated the cold-prefill table row, but left prose, structured, provenance, and the kernel callout on the 2026-09-07 E3 figures (240k 1286, structured 69.62, prose band 28–31). Those are no longer production.

## What changed

Standing claims now match the same-boot A/B on `e3-w3-zfill`:

| Claim | Before (E3 / task 25) | After (W2 TRF=32, 2026-09-09) |
|---|---|---|
| Cold prefill 240k / 60k | ~1408 / ~1454 (already in #59 table) | same, now also the standing receipt |
| Hashmap prose | ~28–31; task 25 29.70 | **30.82** vs same-boot 27.54; band **~29–31** |
| Structured | E3 adopt **69.62** | **69.10** @ 7.0/1.000 (uncontended 68.7–70.3) |
| Provenance / kernel callout | E3 1075 → 1286 as standing | W2 1242 → 1408 standing; E3 kept as previous stack |

## What this does **not** do

- Does not change `EXL3_TEMP_ROWS_FUSED`, image, or overlay.
- Does not re-run cluster benches.
- Does not reopen W5.

Rollback of the numbers is reverting this README commit; production stays last-wins TRF=32.
