# Native activation qualification preparation

Prepared only, after Exa research and review of the pinned FlashInfer/vLLM,
local Exllamav3 routing tests and TensorFold's per-path numerical verification.
No GPU qualification has run and no native container may run beside the active
Int64 serving matrix. This is a coverage review and one prepared geometry case,
not a complete executable gate.

## Live controls and consequences of the one-flag treatment

[Both-node process/source receipts](../../docs/receipts/2026-11-04-moe-activation-review/)
at 04:11 UTC show these additional variables unset:

- `VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE`: pinned default `0`.
- `VLLM_B12X_CSF_SCALE_PREFETCH`: pinned default `1`.

Installed `vllm/envs.py` and `nvfp4_csf.py` match the exact `8244d1c...`
source. Resolving those defaults preserves per-expert input/intermediate scale
vectors in the loader; it does **not** prove that every checkpoint vector is
nonuniform. The native suite must exercise uniform, nonuniform, and one-ULP
different vectors, and eventually inspect representative checkpoint scale
metadata before selecting its real-data cases.

The pinned [CSF adapter](https://github.com/local-inference-lab/vllm/blob/8244d1c44348ae8a5bbfc9f1b607021652283ed4/vllm/model_executor/layers/quantization/nvfp4_csf.py)
registers a layer for side-stream scale prefetch only when `use_a16` is true.
Thus `FORCE_A16=0` is expected to remove the 42 native-A4 target layers from
that registration as well as change their activation math. The existing
prefetch environment flag stays fixed. This is an inherent consequence of
the one-flag treatment; any speed result must describe the whole treatment,
not attribute it solely to faster matrix multiplication. Live selected-plan
and eager/graph behavior still need verification after a candidate start.

The hybrid-prefill oracle uses the minimum input scale across the layer,
whereas native A4 can retain per-expert scales. Layer-wide scale policy
`all`/`w13`/`w2`, disabling A16 prefetch, or a two-term residual hybrid prefill
are separate experimental arms, each with its own quality and serving gate.
Do not add any of them to the current patch.

## Pinned coverage and its limits

[`native-source-inventory.json`](native-source-inventory.json) records exact
test-file hashes and static function/decorator identities from FlashInfer
`91271757f7746f3dfb58f7c2aeb44a663389de9e`. This is not pytest collection or
an executed-case count. The four main files contain 35 test functions with
additional parametrization.

| Source | Evidence it can provide | What it does not prove |
|---|---|---|
| `test_nvfp4_csf.py` | Native A4/A16 compressed-versus-expanded scale equality; poisoned scratch/barriers; changed input/routes; graph replay and stable allocation count | An independent arithmetic oracle or actual GLM geometry: its principal case uses E8/H256/top2 |
| `test_nvfp4_shared_input_scales.py` | Static scale equality/storage contracts, nonuniform/invalid/one-ULP cases, mutation invalidation and operator aliasing | End-to-end large-shape GPU arithmetic |
| `test_nvfp4_auto.py` | Prepared launch ownership, unprepared live counts and changed-input replay; some larger expert shapes | Native-A4 decode coverage: the automatic small-decode path under test selects A16 |
| `test_w4a16_a4_prefill.py` | One-/two-term hybrid arithmetic versus float64 emulation; explicit precision selection; CSF readers, stream ordering, changed-input replay and stable pointers | The same numerical contract as native A4, or full GLM-width 4096-token coverage |

The [native NVFP4 reference](https://github.com/local-inference-lab/flashinfer/blob/91271757f7746f3dfb58f7c2aeb44a663389de9e/flashinfer/experimental/b12x/moe/_shared/kernels/reference.py)
distinguishes `micro`, `dynamic_precise` and `dynamic_fast` quantization order.
Direct division and reciprocal multiplication can disagree at FP4 rounding
ties. The hybrid oracle also rounds different intermediates. Use the oracle
for the executed backend and its scale contract; never require identical A4
and A16 output or transplant the hybrid tolerance to native A4 without proof.

The [FP4 numerics correction](https://github.com/flashinfer-ai/flashinfer/pull/3932)
motivates explicit E4M3 subnormal and independent activation/weight global-scale
coverage. The inventory includes additional source anchors for:

- `test_bf16_to_fp4_tma_fp8_scale_boundaries_graph_exact`: zero, subnormal,
  minimum-normal and rounding-transition scale bytes, exact payloads and replay.
- CSF packed-scale expansion and W4A16 packed-scale boundary tests. These
  exercise specific representation tables; their treatment of subnormals
  does not by itself prove that production selects that representation or
  loses checkpoint scales. Verify the actual representation first.

`test_a4_prefill_launchers_are_owned_by_each_device` requires two CUDA devices
in one process. Each Spark process has one GB10, so a skip cannot count as a
pass. Record that topology mismatch explicitly and separately test both nodes'
local ownership plus the serving system's two-process TP behavior. Do not
claim those are equivalent coverage.

### Dynamic native-A4 arithmetic and replay anchor

The pinned `test_nvfp4_swapped_token_tiles.py::test_swapped_token_tile_live_replay`
adds arithmetic coverage missing from the compressed-versus-expanded equality
tests. Its reference consumes the original packed weights and scales and selects
`dynamic_precise` or `dynamic_fast` to match the tested invocation. The static
parameter product is 130 cases (2 fast-math choices × 5 intermediate widths ×
13 tile/planner/shared-scale combinations); this is not a collected or passed
case count. H512 and at most 20 experts keep this component bounded, but do not
represent the GLM model's full geometry.

For each case it freezes preparation, binds live row counts 1/8/16/17/capacity,
changes activations and expert IDs after capture, and masks the final row's
routes. Three replays must preserve allocation counters and buffer addresses.
The active output must be finite and nonzero, mean per-row cosine must be at
least 0.9999, and RMSE divided by reference RMS must be at most 0.015. Inactive
output tails must retain their NaN poison and the masked row must be exactly
zero. These are the unchanged upstream thresholds for this bounded component;
they do not set the actual-GLM or completed-answer quality tolerance.

This test deliberately uses normal activation scales and forces the dynamic
backend. It does not qualify compact-kernel arithmetic, CSF storage, subnormal
activation scales, the GLM SwiGLU clamp, or automatic serving-plan selection.
Retain those as separate required coverage. Include this exact test and its
source dependencies in the future native bundle, without collecting the whole
phase-kernel module that includes large-offset allocations.

Exa review of the pinned [reference implementation](https://github.com/local-inference-lab/flashinfer/blob/91271757f7746f3dfb58f7c2aeb44a663389de9e/flashinfer/experimental/b12x/moe/_shared/kernels/reference.py)
and the upstream [quantizer exactness work](https://github.com/flashinfer-ai/flashinfer/pull/3387)
supports treating rounding order and subnormals explicitly. The latter concerns
different quantizer entry points; its fast-math environment variable is not
evidence of a control for this embedded B12X kernel. Local vLLM plan construction,
Exllamav3 routing-reference checks and TensorFold's separate serial-exactness and
trusted-model quality checks were reviewed after the lookup. No new runtime
flag or GPU test was applied during the Int64 baseline.

## Prepared actual-geometry hybrid case

[`native/test_glm_hybrid_geometry.py`](native/test_glm_hybrid_geometry.py) calls
the exact upstream float64-emulation test with H4096, local intermediate 1024,
E288, top8, SwiGLU limit 10 and one residual term. It requires SM121 and verifies
the upstream file hash and import owner. The unchanged upstream case uses
192 live tokens, nonuniform calibration-like scales and its relative-norm
threshold of 0.004. Those settings qualify this hybrid component only.

The upstream fixture allocates packed weights and dequantizes one routed
expert at a time for its reference. Its two packed projections contain
1,811,939,328 bytes total at this geometry; this is **not** a measured peak
allocation. Preparation clones/repacking, scale planes, scratch and compiler
state add memory. Measure peak host/device memory under the existing both-node
90 GiB idle/free-memory gate and a bounded native container before expanding
the matrix. Avoid the phase-test `_build_domain` for a naive full-size fixture:
it retains several full dense and dequantized expert tensors.

This wrapper has been syntax-checked locally, not collected or executed on a
GPU. Its source tree must be mounted at `/opt/moe-native-a4-tests` in the
future isolated native container; do not mount alternate kernel implementation
code over the qualified image. Freeze the complete bundle/dependency identities
and host runner before any native window. No container build is needed merely
to add Python test files.

## Remaining gate work

Prepare native-A4 arithmetic at the same H4096/N1024/E288/top8/limit10 geometry,
using per-expert scales and a backend-appropriate oracle. Cover actual decode,
MTP verification and prefill row counts, including 1/4/6/24, dispatch boundaries
and the 4096-token chunk; include uniform/nonuniform/subnormal calibration and
changed inputs/routes through graph replay with poisoned scratch and stable
allocations. Register exact tolerances, collected identities, bounded memory,
fresh cache paths, both-node invocation/source/artifact checks and the terminal
receipt validator before execution. The relevant phase tests include large
offset cases: do not indiscriminately collect them into this bounded suite.

Only after native qualification, the remaining completed-answer quality
components, actual runtime-treatment proof and the unchanged full speed
matrix may the activation experiment be considered for adoption.
