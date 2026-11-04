# Native NVFP4 MoE activations — candidate 18 preparation

Prepared only. No recipe or environment on either Spark has changed. This is
a future isolated trial after the active Int64 comparison and restoration.
Native correctness, model-quality and serving protocols must be completed
and frozen before staging; this directory is not authorization to skip them.

## Current behavior and exact scope

Both nodes currently set `VLLM_B12X_MOE_FP4_FORCE_A16=1` and
`B12X_W4A16_A4_PREFILL=1`. The one-/two-term setting and automatic A16 cutoff
are unset. The former defaults to one term in the installed code. Hybrid
prefill is already available; enabling the same flag would not be a treatment.

[NVIDIA's deployment recipe at this exact revision](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5/README.md)
sets the same pair of flags. This candidate deliberately tests a different
activation policy from that published recipe; checkpoint-format compatibility
alone cannot establish its model quality or justify adoption.

The exact NVIDIA revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` declares
42 routed-expert layers as `NVFP4` and its MTP layer 45 as `W4A16_NVFP4`.
Other mixed-quant assignments remain intact. The text geometry is hidden4096,
expert intermediate2048 (1024 per TP2 shard), 288 experts, top8 and SwiGLU
limit10. [Both-node receipts](../../docs/receipts/2026-11-04-moe-activation-review/)
record public config, whitelisted numeric controls and six installed source
hashes, all identical between nodes and matching pinned source.

The sole candidate change in `recipe.patch` is:

```diff
-  VLLM_B12X_MOE_FP4_FORCE_A16: "1"
+  VLLM_B12X_MOE_FP4_FORCE_A16: "0"
```

The pinned model loader passes `use_a16=True` for explicit W4A16 checkpoint
layers; `Nvfp4CsfMoEMethod` ORs that with the force flag. Therefore the expected
effect is native A4 for the 42 NVFP4 target layers, while the explicit MTP
W4A16 assignment stays intact. This is a source-derived expectation, not live
proof of the candidate. The MTP backend remains humming. Dense/vision formats,
checkpoint bytes, scale storage and all other recipe fields are unchanged.

Native A4 and hybrid prefill have different activation-scale contracts; this
can change both prefill and decode outputs. Do not describe it as merely a
faster identical kernel, or substitute an EXL3/requantized checkpoint.
The [native coverage review](native.md) also identifies the one-flag change's
effect on A16-only scale-prefetch registration and confirms both nodes retain
the pinned per-expert scale policy. Live selected-plan proof remains required.

## Exa research, stronger options and local review

Exa lookup preceded the patch and covered the original hybrid path, row-type
selection, scale numerics and native B12X integration. The canonical
[hybrid-prefill proposal](https://github.com/local-inference-lab/b12x/pull/474)
reports a speed/quality tradeoff on RTX PRO hardware; those numbers do not
predict GB10 results. The [vLLM row-type change](https://github.com/local-inference-lab/vllm/pull/977)
avoids letting batch size decide precision. Current pinned source uses
scheduler-owned token ranges and `B12X_W4A16_A4_PREFILL`, so the old threshold
variable is not the candidate switch.

[FlashInfer's quantization correction](https://github.com/flashinfer-ai/flashinfer/pull/3932)
shows why subnormal block scales and separate activation/weight global scales
need explicit numerical tests. [Native B12X MoE integration](https://github.com/vllm-project/vllm/pull/52018)
documents the forced-A16 override and graph/storage requirements. Its earlier
standalone-package evidence does not replace this image's qualification.

After Exa, reviewed updated local vLLM and pinned `8244d1c...` loader,
`nvfp4_csf.py`, MoE adapter and semantic row selection; FlashInfer
`9127175...` preparation, CSF and activation kernels; Exllamav3's routing
comparison by row class; and TensorFold's separate prompt/decode paths and
global-scale fields. Useful checks are row-class routing consistency,
explicit scale ownership and separate prompt/decode validation. No EXL3 or
TensorFold execution is introduced.

A two-term residual A4 prefill is a separate stronger-precision option;
disabling hybrid A4 is an A16 reference. Neither is bundled with this switch.
Any later trial of either needs its own native, quality and speed gate.

## Required qualification before serving

- Freeze exact test files and collected case identities from the installed
  FlashInfer pin: CSF decode/scale expansion, native A4/A16 MoE, nonuniform and
  subnormal calibration, shared scratch, changed-input graph replay and hybrid
  prefill selection. Existing `test_nvfp4_csf.py`,
  `test_nvfp4_shared_input_scales.py`, `test_nvfp4_auto.py` and
  `test_w4a16_a4_prefill.py` provide starting coverage, not a completed test plan.
  [Pinned source inventory, the hybrid case and 32 actual-TP2 native-A4 cases](native.md)
  are prepared, with explicit limits. Local collection and five preallocation
  guard tests pass; no GPU qualification result is claimed.
- Include actual TP2 geometry and decode/MTP/prefill boundaries, with bounded
  memory and quantization-aware references. Compile and inspect the executed
  SM121 artifacts on both Sparks. Small-shape tests or a source-only import do
  not prove real-layer correctness. Never run these GPU tests beside serving.
- Freeze a paired model-quality corpus and rejection thresholds before either
  serving arm: completed answers, long-context retrieval/checksums, tool/JSON
  behavior and task correctness. Include real inference scale/route coverage.
  The [full-split GSM8K component](quality.md) now has pinned data, requests,
  thresholds and an executable fail-closed runner/scorer; it is not run and
  does not replace the remaining quality components.
  The 200-token max-reasoning speed matrix measures throughput, not quality.
- Register a fresh matched baseline/candidate full matrix with unique seed,
  three reps, 8k–128k, c1/2/4/8, the same decode budgets and all API/cache gates.
  Keep MTP5, 1M context, cap4, chunk4096, FP8 KV17GiB/rank and exactly2,350,210
  usable tokens, transport, driver and kernel fixed. Confirm actual selected
  target/draft activation paths before timing and retain cache provenance.

Once eligible, snapshot both nodes and stage atomically only in the runtime
tree, with healers disarmed and both-node idle/memory gates passed. Start only
through the FP4 service. Rollback restores the saved baseline recipe and appends
the ledger decision; verify incumbent health/capacity and rearm the watchdog.
Final adoption requires the user's decision after complete receipts.

Local preparation verified forward/reverse patch application and parsed-YAML
equality after changing exactly this scalar. The manifest records all hashes.
No GPU correctness, model-quality, speed result or adoption is claimed.
