# Task 42 — thin-decode fused `exl3_moe` register/spill cut: window receipts

Cluster: spark1 (head) + spark2 (worker), 2×DGX Spark GB10/SM121, CUDA 13.0.88.
Date: 2026-09-19 → 2026-09-20 (EDT).
Candidate image: `glm53-selfbuild:e3-pipeline-f1s8` (`sha256:224cf63b2a55…`).
Control image: `glm53-selfbuild:e3-w3-zfill-v149` (`sha256:c9ab369e62a1…`).

## 1. What was measured

The fused `exl3_moe_kernel<4,256,1>` is the largest single decode kernel in this
deployment (49.70% of decode-step kernel time on rank0, task 29). Its stock
geometry (`MOE_FRAG_STAGES`/`MOE_SH_STAGES` = 3/3) carries a real local-memory
spill frame. The candidate re-instantiates the *same* kernel template with a
shallower fragment pipeline and a deeper shared-memory pipeline (1/8), which
re-partitions the same fixed 90 KiB dynamic smem budget and trades
register-resident fragment stages (which spill) for smem-resident `cp.async`
stages (which do not).

Nothing about the tile shapes, MMA instruction, dequant, or reduction order
changes. The variant is a separate symbol; the stock kernel and every stock
instance are untouched.

## 2. Shipped-cubin signature (both measured on the built images)

Per-function, from `cuobjdump -res-usage` + `nvdisasm` of the extracted cubin
(the vLLM image ships no `nvdisasm`, so the host toolkit was used):

| instance | REG | STACK | static SHARED | STL | LDL | SASS insts |
|---|---|---|---|---|---|---|
| stock `exl3_moe_kernel<4,256,1>` | 128 | 88 B | 1024 | 37 | 39 | 10,256 |
| variant `glm53_exl3_moe_pipeline_kernel<4,256,1>` | 128 | 32 B | 1024 | 9 | 4 | 5,888 |

Same 128 registers and the same 1 block/SM occupancy. `SMEM_MAX` (90 KiB) is a
fixed launch parameter, not stage-derived; `exl3_gemm_inner.cuh:68` asserts the
budget fits and the build compiled, so the dynamic smem footprint is unchanged.

Note: the TODO's research predicted a 16-byte frame with 6 STL / 0 LDL. Measured
on this revision and toolchain the cut is *larger* (88 → 32 B, 37 → 9 STL), and
the instruction count also falls 42.6%. The SASS numbers above are the ones to
use; the research figures were from a different build.

## 3. Isolated kernel A/B (`kprobe.py`, torch profiler device time)

Three arms, one image pair, no reboot. `prod` vs `ctrl` validates the rebuild.

| T | prod (stock) | ctrl (new img, knob 0) | var (new img, knob 1) | var vs ctrl |
|---|---|---|---|---|
| 12 | 3374.0 µs | 3364.9 µs | 3150.2 µs | **−6.38%** |
| 20 | 4696.7 µs | 4694.7 µs | 4375.9 µs | **−6.79%** |
| 32 | 6002.5 µs | 6032.6 µs | 5590.1 µs | **−7.34%** |

`prod ≈ ctrl` within 0.5% on every point → the image rebuild is neutral for the
stock path, so the runtime knob is a clean independent variable.

The pre-registered instrument (`tests/bench_e3_microbench.py`) was run first and
**rejected as insensitive**: at T ∈ {12,20,32} it reports ~3.71 ms for both the
stock and the grouped path, i.e. it cannot resolve the change. The profiler probe
above replaced it, and shows the fused kernel is 99% of layer device time.

## 4. End-to-end decode, same image, knob the only variable

Control boot: `./start.sh restart` with `GLM53_EXL3_MOE_PIPELINE=0`
(arming lines 0). Variant boot: same image with `=1` (arming lines 1 on **both**
nodes). Both boots: acceptance 7/7, pool `1,396,551 tokens / 1.40×`,
`num_gpu_blocks=567`.

| lane | control (knob 0) | variant (knob 1) | delta |
|---|---|---|---|
| structured, 9 runs | 71.17 (70.54–71.46) | 75.48 / 75.16 (75.11–76.11) | **+5.8%** |
| hashmap prose, 9 runs | 30.17 (28.52–34.59) | 33.89 / 35.04 | +12% |
| essay, 9 runs | 25.38 (23.42–27.78) | 26.01 | **+2.5%** |

The structured lane is the tight instrument and its two arms do not overlap.
Prose and essay are noisy (control prose spans 28.5–34.6; prose acceptance drifts
0.50–0.58 across runs, which is a direct throughput confound on that lane).

## 5. Correctness

- Acceptance 7/7 on both arms (includes a ~32k-token needle retrieval).
- Structured decode: accept ratio 1.0, 7.0 accepted per step, no NaN, 9/9 runs.
- Prose coherent, no NaN.
- Kernel parity (`parity.py`, real serving entry point `apply_exl3_experts`,
  8 cases over T ∈ {12,20,32,1024} × skew ∈ {1.0, 0.0}, both arms on identical
  inputs; `cmp.py` over the saved fp32 output tensors, receipt
  `parity-compare.txt`): **1 of 8 bit-exact**, worst normalized max-abs deviation
  **1.646e-3** (`max|Δ| / mean|a|`), worst NRMSE **7.438e-6**, both arms take
  identical paths. `absum` (Σ|y|) is bit-identical in all 8 cases and `sum`
  agrees to ≤1.9e-5 relative, so the deviation is small against the output scale
  rather than a shifted result. Not bit-exact: recompiling with a different
  register budget lets ptxas schedule/contract fp differently. Verdict: PARITY OK
  (within fp16 recompilation tolerance), and stated as such rather than as exact
  parity. Note the first attempt at this comparison ran on the host, which has no
  torch, and died with `ModuleNotFoundError` (`final.log` section 3), so the
  numbers above come from a re-run against the harness's saved tensors.
- MemFree above the 2.5 GiB floor on both nodes throughout; pool byte-identical.

## 6. Gate verdict

Task 42's gate: *≥5% decode e2e on hashmap prose **and** hard essay, structured
non-inferior.* **NOT MET — one lane is short.**

| lane | result | gate |
|---|---|---|
| structured | +5.8% | non-inferior ✓ |
| hashmap prose | +12% | ≥5% ✓ |
| hard essay | **+2.47%** (25.3845 → 26.0120 tok/s) | ≥5% ✗ |

The essay lane is the binding lane and it does not clear the bar. Kernel device
time (−6.4 to −7.3%) and the structured result do not substitute for it, because
the gate is stated per-lane on end-to-end decode.

What is true and what is not:

- The change is **deployed**: production boots it, the arming line is present on
  both nodes, acceptance is 7/7, the pool is byte-identical, and the rollback is
  a single knob. No lane regressed, and the essay lane did improve.
- The change is **not gate-passing** as pre-registered. Adoption and gate
  satisfaction are separate claims and this receipt does not merge them.

Acceptance therefore stays **pending**, not satisfied. The essay shortfall is a
measured outcome, not an implementation defect, so no code change follows from
it. Resolving it needs one of: a user decision to revise the gate or the scope
(a single-lane essay result on a noisy lane), or a further arm that clears ≥5%
on the essay lane. Until then this task must not be reported as a met gate.

## 7. Method findings worth keeping

1. **First-round-after-boot numbers are not usable.** The variant first measured
   54.63 structured / 25.25 prose ~2.5 min after `/health`; the same boot, once
   warm, measured 75.48 / 33.89. `MemAvailable` is ~4 GiB on this box, so the
   ~180 GB of weights cannot stay cached and the first runs pay disk I/O. Every
   arm must be warmed identically before benching.
2. **`./start.sh start` refuses while the previous container holds port 8000.**
   A control "boot" that used `start` silently kept the previous arm running and
   produced a bogus control (`start.sh exit=1`, arming line still 1). Use
   `restart`/`stop`, and always assert the arming line count per arm.
3. **A boot can fail on the 0.25 GiB memory margin.** The first variant boot died
   in vLLM's startup fit check (`free 103.19 GiB < 0.85 × 121.69 GiB`) with no
   relation to the kernel — the arming line count was 0, so the variant had never
   been selected. Cleaning up probe artifacts and rebooting cleared it. This is
   the task-50 margin, and a window must leave the box as clean as it found it.
4. **Run in-container comparisons in the container.** The parity step ran its
   `torch.load` comparison on the host, which has no torch, so it died with
   `ModuleNotFoundError` *after* both arms had already succeeded. The window's
   headline numbers were therefore reported without a retained receipt for a
   while; `parity-compare.txt` is the re-run. Any step that imports torch belongs
   in `docker run`, and a window script should assert its comparison step exited 0
   instead of leaving the failure in a log.
5. **A capability that lives in the image needs a capability check in the
   launcher.** "Fail-closed" inside the patched extension says nothing about an
   old image or a disabled code path: both reach stock with the knob armed. The
   build stamps `glm53.task42.pipeline` and the launcher refuses an armed knob
   without it. Assert the *end-to-end* contract, not the layer you wrote.
