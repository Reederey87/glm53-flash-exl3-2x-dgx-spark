# Native activation qualification preparation

Prepared only, after Exa research and review of the pinned FlashInfer/vLLM,
local Exllamav3 routing tests and TensorFold's per-path numerical verification.
No GPU qualification has run and no native container may run beside the active
Int64 serving matrix. The hybrid case and 32 native-A4 cases are prepared and
locally collected without GPU execution. The isolated runner, complete source
bundle and terminal admission gate still need to be frozen.

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

The native TP2 component below still needs GPU execution. Extend the complete
bundle with actual-geometry CSF storage/expansion, shared scratch across layers,
real checkpoint calibration metadata and automatic serving-plan selection.
Freeze fresh cache paths, both-node idle/identity/source/artifact checks,
the isolated runner and terminal receipt validator before execution. The
relevant phase tests include large-offset cases: do not indiscriminately
collect them into this bounded suite.

Only after native qualification, the remaining completed-answer quality
components, actual runtime-treatment proof and the unchanged full speed
matrix may the activation experiment be considered for adoption.

## Prepared TP2 native-A4 component

[`test_glm_native_geometry.py`](native/test_glm_native_geometry.py) declares
32 cases at H4096/N1024/E288/top8 with SwiGLU limit10. Local pytest collection
finds all32 plus the existing hybrid case; **none has executed on a GPU**.
The [preparation manifest](native-preparation-manifest.json) records exact
source hashes, case IDs, thresholds, memory checks and remaining coverage.

| Backend | Live rows | Scale profiles | Fast math | Cases |
|---|---|---|---|---|
| Compact (`micro`) | 1, 4, 6, 8 | Nonuniform | False | 4 |
| Dynamic | 1, 4, 6, 8, 9, 16, 17, 24, 32, 33, 4096 | Nonuniform | False and true | 22 |
| Compact | 6 | Uniform and subnormal | False | 2 |
| Dynamic | 33 | Uniform and subnormal | False and true | 4 |

Exa reviewed the pinned numerical reference and live-replay tests, plus the
[SM12x backend synchronization](https://github.com/flashinfer-ai/flashinfer/pull/4285)
and [scale-numerics correction](https://github.com/flashinfer-ai/flashinfer/pull/3932).
Those older public entry points do not establish behavior of this installed
experimental pin. Subsequent local source review found that its
[`micro.py::is_supported`](https://github.com/local-inference-lab/flashinfer/blob/91271757f7746f3dfb58f7c2aeb44a663389de9e/flashinfer/experimental/b12x/moe/_shared/kernels/micro.py)
accepts at most eight live tokens. The 8/9 boundary is therefore explicit;
M24 must not be claimed as compact coverage. Dynamic tile boundaries16/17
and32/33 are separate checks.

The pinned standard corpus's GLM M1/42-layer cases use local intermediate512
(TP4). The new fixture uses1024 (TP2), preserves all288 experts and consumes
the original packed ModelOpt tensors in the independent Torch oracle. The
corpus is now the ninth test-source anchor, imported for fixture/assertion
helpers rather than collected wholesale. The actual serving loader's `w13`
layout and clamp are retained. Exact fixture and installed-reference hashes
are checked before use; the test tree must not supply kernel implementation.

Every row and route is evaluated, including all4096 rows of the full chunk.
The unchanged reference dequantizes one route's expert tensors at a time,
avoiding a full dense288-expert reference copy at the cost of many launches.
The compact oracle uses `micro` rounding; dynamic uses `dynamic_precise` or
`dynamic_fast` to match its declared invocation. Physical weight global scales
are retained while activation calibration changes. Torch's float32 matmul
precision must be `highest`; reduced-precision reference matmul is refused.
The test checks that the
first reference route actually activates the clamp and, for subnormal cases,
has positive subnormal input and intermediate block scales. A dormant fixture
fails rather than counting nominal flags as numerical coverage.

Prospective component thresholds are mean row cosine>=0.9999 and
RMSE/reference RMS<=0.015, with finite/nonzero output. These retain the dynamic
anchor's bounds and deliberately tighten the older compact corpus's0.999/0.03
bounds; they have not been established as passing at TP2. Investigate any
failure before changing a threshold. A16/A4 bit equality is not required.
Both eager and captured output must pass the initial oracle; three replays
must pass a second oracle after changing activations, routes and route weights.
Replays poison output and barriers, require an exactly zero masked row and an
untouched NaN tail, and forbid allocation-count/live-byte/pointer changes.
Selected backend and fast-math state are checked, so an unnoticed fallback
cannot satisfy a forced-path case.

Preallocation guards account for the sum of all scratch buffers, retain at
least24GiB MemFree after the requested allocation and refuse a planned tensor
footprint>=32GiB. The fixture reserves an8GiB planning allowance; observed
CUDA tensor peak must remain below48GiB. Five CPU fault checks verify that
oversized/low-free requests fail before any allocation, including a combined
scratch request whose individual buffers would fit. These checks are not a
hard cap on compiler/host allocations and do not replace the both-node90GiB
idle gate or the future bounded native runner.

This component uses expanded scale storage and forced backend selections.
It does not qualify the CSF representation, router top-k selection, checkpoint
quality, or the two-process TP collective. Exllamav3 routing tests and
TensorFold's teacher-receipt checks reinforced keeping those separate from
fixed-route arithmetic and from bitwise serial replay. No code from either
engine is installed and no cluster treatment changed during preparation.
