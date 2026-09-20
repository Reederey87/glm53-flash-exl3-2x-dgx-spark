# Task 42 lever 2 — gate/up Hadamard reuse: window receipts

Cluster: spark1 (head) + spark2 (worker), 2×DGX Spark GB10/SM121, CUDA 13.0.88.
Date: 2026-09-20 (EDT).
Candidate image: `glm53-selfbuild:e3-pipeline-f1s8-reuse`
(`sha256:1b866e26af4d…`, labels `glm53.task42.pipeline=1x8` `glm53.task42.reuse=1`
`glm53.recipe.stamp=task42-reuse-f1s8`).
Control of this window: same image, `GLM53_EXL3_MOE_PIPELINE=1`
`GLM53_EXL3_MOE_REUSE=0` (lever 1 behaviour).
Rollback named before the arm: `GLM53_EXL3_MOE_REUSE=0` on this image, or
`IMAGE=glm53-selfbuild:e3-pipeline-f1s8` if the rebuild itself were bad.

## 1. What was measured

Stock `had_gather_gu_in()` still runs two `had_hf_r_128_inner` calls on the same
`in_ptr`. This checkpoint's gate/up residual SUH is identical
(`torch.equal(w13_suh[:,0], w13_suh[:,1])`). Lever 2 compiles a second instance
of the already-adopted pipeline kernel with `bool shared_input`; the true
instance skips the up-lane Hadamard and feeds `gemm_up` from `temp_state_g`.
Python aliases the up-SUH pointer table onto the gate table only after the
equality proof; the host dispatch additionally requires the two SUH pointer
tables to be the same tensor.

Nothing about tile shapes, MMA, dequant, reduction order, or lever-1 geometry
(frag 1 / sh 8) changes. A and B stay on one cubin; the runtime knob is the
only independent variable.

## 2. Shipped-cubin signature

`cuobjdump -symbols` / `-res-usage` of the extracted
`exllamav3_ext.cpython-312-aarch64-linux-gnu.so` (338 MiB). Four pipeline
instances are present. Production geometry is K4/N256/mcg (`cb=1`):

| instance | mangled `shared_input` | REG | STACK | static SHARED |
|---|---|---|---|---|
| `glm53_exl3_moe_pipeline_kernel<4,256,1,false>` | `Lb0E` | 128 | 32 B | 1024 |
| `glm53_exl3_moe_pipeline_kernel<4,256,1,true>` | `Lb1E` | 128 | 32 B | 1024 |
| `glm53_exl3_moe_pipeline_kernel<4,256,2,false>` | `Lb0E` | 128 | 32 B | 1024 |
| `glm53_exl3_moe_pipeline_kernel<4,256,2,true>` | `Lb1E` | 128 | 32 B | 1024 |

Same 128 registers and the same 32-byte spill frame as adopted lever 1. Host
`nvdisasm` of the `.so` failed (`Unexpected section size`); STL/LDL per instance
is therefore not re-quoted from this window. The kprobe kernel names below are
the live proof that the two `shared_input` instantiations were actually selected.

## 3. Isolated kernel A/B (`kprobe.py`, torch profiler device time)

GPU exclusive, same image, `GLM53_EXL3_MOE_PIPELINE=1` on both arms.

| T | ctrl fused µs (`<4,256,1,false>`) | var fused µs (`<4,256,1,true>`) | var vs ctrl |
|---|---|---|---|
| 12 | 3197.6 | 3179.5 | **−0.57%** |
| 20 | 4435.9 | 4450.4 | **+0.33%** |
| 32 | 5740.2 | 5712.5 | **−0.48%** |

Control: `reuse_aliased=False`, boot line `shared_input=0`.
Variant: `reuse_aliased=True`, boot line `shared_input=1` plus
`gate/up Hadamard reuse`. The skip is armed and the kernel names differ. Device
time is noise around the already-adopted pipeline kernel; this lever does not
add another −6% device-time cut.

## 4. End-to-end decode, same image, knob the only variable

Both boots: `prod-start.sh` (settle + overlay-verify skipped on retry),
acceptance 7/7, pool `1,396,551 tokens / 1.40×`, `num_gpu_blocks=567`.
First-round-after-boot numbers discarded; warmup 3-run structured/hashmap/essay
then 240 s settle, then 9-run measured round.

| lane | control (REUSE=0) | variant (REUSE=1) | delta |
|---|---|---|---|
| structured, 9 runs | 73.88 (73.49–74.39), acc 1.0 / 7.0 | 74.19 (63.49–74.67), acc 1.0 / 7.0 | **+0.42%** |
| hashmap prose, 9 runs | 30.36 (26.70–34.58), acc 0.496 | 33.03 (28.10–35.43), acc 0.551 | **+8.79%** |
| essay, 9 runs | 25.18 (23.84–26.78), acc 0.427 | 25.92 (24.46–27.20), acc 0.436 | **+2.94%** |

No NaN on any lane. Hashmap coherent on both arms. Structured min on the variant
includes one 63.49 outlier; median still sits inside the control range.

## 5. Correctness

- Acceptance 7/7 on both arms (includes a ~32k-token needle retrieval).
- Kernel parity (`parity.py` / `cmp.py` on `apply_exl3_experts`, 8 cases over
  T ∈ {12,20,32,1024} × skew ∈ {1.0, 0.0}): **3 of 8 bit-exact**, worst
  normalized max-abs **1.646e-3**, worst NRMSE **7.437e-6**. Paths identical
  (`none` at decode T, `grouped` at T=1024). Verdict: **PARITY OK** (fp16
  recompilation tolerance). Receipt `parity-compare.txt`.
- Fail-closed observed live: control selected `shared_input=0` /
  `reuse_aliased=0`; variant selected `shared_input=1` / `reuse_aliased=1` on
  **both** nodes. Launcher refused REUSE=1 on the previous production image
  (no `glm53.task42.reuse` label).
- MemFree above the 2.5 GiB floor on both nodes while serving; pool
  byte-identical.

## 6. Gate verdict

Task 42's gate: *≥5% decode e2e on hashmap prose **and** hard essay, structured
non-inferior.* **NOT MET — essay is short.**

| lane | result | gate |
|---|---|---|
| structured | +0.42% | non-inferior ✓ |
| hashmap prose | +8.79% | ≥5% ✓ |
| hard essay | **+2.94%** (25.1813 → 25.9213 tok/s) | ≥5% ✗ |

Kernel device time is not a substitute for the essay lane. Adoption and gate
satisfaction are separate claims.

Acceptance is resolved by an **explicit user decision, not by reinterpreting
the measurement**: on 2026-09-20, after the per-lane result above was reported,
the user directed that the change be **adopted** as-is, with the essay
shortfall left recorded. That is the same class of requirement revision lever 1
used earlier the same day. This receipt keeps the two statements separate:

- The gate **as pre-registered is not met** (essay +2.94% < 5%). That remains
  true and is not restated as a pass.
- The change is **accepted and deployed** on the user's explicit decision, with
  no lane regressed, hashmap +8.79%, structured non-inferior, parity OK, and
  production booted on `glm53-selfbuild:e3-pipeline-f1s8-reuse` with
  `GLM53_EXL3_MOE_REUSE=1`.

Rollback remains a last-wins `.env` flip: `GLM53_EXL3_MOE_REUSE=0` (same image,
two Hadamards, pipeline kernel still selected), or
`IMAGE=glm53-selfbuild:e3-pipeline-f1s8`.

## 7. Adopted production state (after the decision)

- Image: `glm53-selfbuild:e3-pipeline-f1s8-reuse` on **both** nodes.
- Knobs: `GLM53_EXL3_MOE_PIPELINE=1` `GLM53_EXL3_MOE_REUSE=1`.
- Arming: `shared_input=1 (register-cut decode; gate/up Hadamard reuse)` on
  head and worker; overlay `reuse_aliased=1`.
- Pool: `1,396,551 tokens / 1.40×`, `num_gpu_blocks=567`.
- `/health` 200. Watchdog timer re-armed. Head/worker MemFree 4.54 / 4.52 GiB.

## 8. Method notes worth keeping

1. Overlay GPU self-check immediately before vLLM's fit check reproduced the
   0.25 GiB miss (`103.16 < 103.44 GiB`). Skip overlay-verify on a serving boot
   of an image that already passed it (build + kprobe + first attempt).
2. `prod-start.sh` is the right boot path (`NEED_GIB=90`, 3-attempt retry). A
   bare `start.sh start` after kprobe leftover memory fails the fit check.
3. `set -o pipefail` + `docker logs | grep -q` is SIGPIPE on a large stream and
   looks like a missing arming line. Capture logs, then search the string.
4. `head` on a grep under pipefail is the same class. Swallow or capture first.
5. First-round-after-boot numbers remain invalid. Warm both arms identically.
6. `kprobe.py` must record the `shared_input` kernel name, not only fused µs —
   otherwise a wash looks like the skip never fired.
