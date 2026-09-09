## Summary

Follow-up to merged [#59](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark/pull/59). Aligns README standing numbers with the 2026-09-09 production stack: Task 25 verification-only adaptive-k (PR #58) plus W2 `EXL3_TEMP_ROWS_FUSED=32` (PR #59). No runtime, overlay, or env change.

#59 adopted TRF=32 and updated the cold-prefill table row, but left prose, structured, provenance, and the kernel callout on the 2026-09-07 E3 figures (240k 1286, structured 69.62, prose band 28–31). Those are no longer production. Adaptive-k decode impact (hashmap +7.1%, hard essay +10.7%) was in the Task 25 docs, not the standing README provenance.

## What changed

Standing claims now match the two isolated 2026-09-09 receipts on `e3-w3-zfill`:

| Claim | Before (E3 as standing) | After (Task 25 + W2) |
|---|---|---|
| Adaptive-k prose | mentioned only as a prior hop | isolated **+7.1% / +10.7%** (hashmap 27.72 → 29.70, essay 21.77 → 24.09); knob stays on |
| Cold prefill 240k / 60k | ~1408 / ~1454 (already in #59 table) | same, now also the standing prefill receipt |
| Hashmap prose | ~28–31 | standing W2 **30.82** vs same-boot 27.54 (adaptive-k frozen on); band **~29–31** |
| Structured | E3 adopt **69.62** | **69.10** @ 7.0/1.000 (uncontended 68.7–70.3) |
| Provenance / kernel callout | E3 1075 → 1286 as standing | W2 1242 → 1408 standing; Task 25 is the decode-lane receipt; E3 kept as previous kernel stack |

## What this does **not** do

- Does not change `EXL3_TEMP_ROWS_FUSED`, image, or overlay.
- Does not re-run cluster benches.
- Does not reopen W5.

Rollback of the numbers is reverting this README commit; production stays last-wins TRF=32.
