# Prefill budgets 2048 and 8192 — candidates 07 and 08

Prepared before either treatment is measured. These are two separate windows,
each comparing the incumbent 4096 budget with one candidate. Complete the
active Int64 window and restore the incumbent first. No image build is needed.
Neither patch has been staged to the runtime or tested with inference.

## Research and exact-source review

Exa review of [vLLM tuning guidance](https://docs.vllm.ai/en/stable/configuration/optimization/)
supports measuring smaller chunks for decode interruption and larger chunks
for prompt latency. Those recommendations do not establish a winning value
for this checkpoint or GB10. The [memory guide](https://docs.vllm.ai/en/stable/configuration/conserving_memory/)
also identifies CUDA graphs as a memory consumer; retain capture policy and
measure actual memory instead of compensating with a smaller KV pool.

Stronger options considered were separate solo/mixed budgets, decode cadence,
compute-share scheduling and disaggregated prefill. The fork's solo-prefill
switch has additional prerequisites documented in the [candidate 22 review](../2026-11-03-prefill-policy-review.md).
Exa fetched the primary tuning/memory documents; PR880 again timed out, so its
previously retained canonical review and the pinned local source remain the
evidence for that fork-specific alternative. Separate prefill/decode instances
would change this two-rank deployment and are not part of a chunk-budget arm.

Reviewed vLLM commit `8244d1c44348ae8a5bbfc9f1b607021652283ed4`:

- `vllm/config/vllm.py::_set_max_num_scheduled_tokens` makes the implicit
  scheduled-token cap equal to the batched budget under the fixed MTP config.
  This derived change belongs to the single budget treatment; do not add a
  second explicit scheduler-cap knob.
- `vllm/v1/worker/gpu/model_runner.py` sizes request/input buffers from the
  batched budget. `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` also sizes
  persistent per-token metadata and attention capacities from it. Thus this
  is not a live scheduler-only edit; restart and inspect the resolved settings.
- CUDA graph size selection also consults configured sequence/speculation
  limits. Retain graph flags, record resolved capture sizes and allocation;
  do not claim those derived effects are a separate kernel improvement.

Reviewed Exllamav3 `da68d57de61781e5b74ad68698ac4caf36408d7a`,
`generator/generator.py`: shared chunks, cooldown and rotation distinguish
ongoing decoding from solo prefill. Reviewed TensorFold
`e854c28001a2a23e86cc6dd22342e64c2807bab7`, GLM `cuda_state.zig` and
`cuda_prompt_solo.zig`: 2048 mixed/4096 solo rows, persistent buffer sizing,
ring-boundary checks and a drafter refusal. The useful technique carried into
these trials is explicit solo/mixed measurement. No quantization or serving
engine is substituted.

## Frozen treatments and controls

Only `defaults.max_num_batched_tokens` changes in the recipe. Both patches
apply to baseline SHA256
`cd0d994a55ceeae6b7dab2d02d59f09e461c105d3c7c2effd79dc6c14141589a`.

| Arm | Patch | Patch SHA256 | Resulting recipe SHA256 |
|---|---|---|---|
| 2048 | `experiments/prefill-chunks/2048.patch` | `5e1cc33e80da7475faa46f8b14c7643eb3da71fb04a3ac2cb046dc044f3c5890` | `107e65d94e9f32f2b3c9689feaa4e1f9612953c334116d51761fca4aa6a32007` |
| 8192 | `experiments/prefill-chunks/8192.patch` | `7648266f205e10ab6ad50acfd3e2e8e5b9867b83a9e02fd636af347002447eca` | `b029ee807a61e28d89728f549cd1fec78b59c1b4d312e9fccd82e5bc2adcbb50` |

Actual temporary application verified the exact single-field change and
byte-identical reverse application for each patch; [receipts](../receipts/2026-11-03-prefill-chunks/patch-check.json)
retain the hashes. Freeze this gate and both
patches in a commit before any window; verify all hashes again before staging.

Both arms use incumbent image
`sha256:a7f76ae9f18c7701c9cde5f72c606393d9a21ca94dd8e53b27589935f64fcbe2`,
NVIDIA checkpoint revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, TP2,
FP8 KV, 17 GiB per rank, **2,350,210 usable KV tokens**, 1,000,000 context,
MTP5 and sequence cap4. Preserve transport, reserve, driver/kernel, all other
recipe/environment values, attention/linear/MoE backends and graph flags.
The recipe's 500k default remains overridden by the live 1M environment.
MTP3 adoption and any new image adoption remain separate decisions.

## Common feasibility pilot before treatment

First run a valid mixed-load pilot on the restored 4096 incumbent, using the
frozen `f5d049e` probe already staged on spark1. Probe SHA256 is
`47594c5ca7e9bae172a8d723e571b4c5c7f76537fd6f1aa98d5467f66d3c3a95`;
its helper SHA256 is
`745f317ba7df8f1f78cf18388142b8d000b7b5a7d56f6fb1af0a078c125b2626`.
Explicitly override `--prefill-lengths 32768,65536,131072`. The 32k minimum
provides at least four prefill chunks even at budget8192; the default16k can
provide too few update opportunities to establish the required overlap.
This choice precedes all live measurements and does not relax validity rules.

Retain the other probe controls: decoders1/3, generated budget2048 including
reasoning, readiness32, three reps, max-seqs4, 32k/256 warmups and pilot seed
2026110306. Require all18 measured cells and both warmups valid. Preserve any
failed pilot. If a protocol change is necessary, freeze it and repeat the
baseline pilot before either candidate runs; do not adjust after seeing a
candidate result. Record the successful pilot receipt and exact command in
the ledger before the first treatment restart. Pilot data are not A/B controls.

## Measurement and decision

Run a fresh 4096 control for each candidate window. Full-matrix seeds are
2026110307 for 2048 and 2026110308 for 8192, equal across paired arms.
Use the unchanged full-matrix harness `05c6023a4b0df380aae5bdb159afab32c1bf516a`,
SHA256 `85371429e129aeb62f4f9f28b918d8453eb292267654e0e6d30489e0f4368401`.
Require three reps of no-cache 8k/16k/32k/64k/128k at c1/2/4/8 and
code/prose/structured decode at c1/2/4/8 with200 generated tokens, explicit
maximum reasoning and the existing discarded warmups:96 measured cells and
40 warmups per arm. Label c8 queueing against cap4. Match prompt hashes,
budgets and protocol settings; keep reciprocal-TPOT and normalized aggregate
labels, with full-cell/bundle-aware rates reported separately.

In addition, run the validated mixed protocol in both arms, with seeds
2026110309 for 2048 and 2026110310 for 8192. Each pair uses identical seeds,
lengths, prompts and budgets; neither pair reuses pilot output. Report long
TTFT/prefill rate, per-decoder observed token-count rate, complete-update-gap
p95/max and whole-cell speculative deltas. These gaps are not individual-token
latencies, and overlapping requests alone do not prove GPU co-execution.

Before numbers are collected, the primary benefit measures are:

- **2048:** for each mixed cell, median across decoders of their p95 usage-update
  gaps; then the geometric mean over the six length/decoder-count cells per
  repetition. Require at least2% reduction versus its matched control in at
  least two of three reps, with candidate median below the control range.
- **8192:** geometric mean of the20 per-request median prefill rates in each
  full-matrix repetition. Require at least2% gain in at least two paired reps,
  with candidate median above the control range.

Both must also pass the full mandatory non-regression gate: reject a >2%
prefill/decode/draft-acceptance-rate regression in at least two matched reps
at any measured full-matrix point. For the mixed grid, reject a >2% rise in
the cell's median p95 gap or worst complete-update gap, or a >2% fall in long
prefill rate or draft acceptance, in at least two matched reps at any grid
point. These are relative changes. Report other mixed token-rate effects as
diagnostics; within-bundle generation timing is unobserved. All required
metrics must be finite/valid, with positive comparison denominators.

A latency/throughput tradeoff that violates these gates is recorded as a
regression, not adopted. Effects inside control variability are inconclusive.
Require API/reasoning/tool and cold/hot prefix checks, identical measured KV
capacity, no corrupted/nonfinite output, CUDA/Xid errors, lost ranks or
sustained swapping. Incomplete streams, cache hits or unproven mixed overlap
cannot produce a passing result.

## Activation and rollback

Snapshot both nodes' recipe/environment/image/argv/unit/caches and actual KV.
Disarm the FP4 watchdog and any active legacy healers. Stop only through
`vllm-glm53fp4.service`; require both containers absent, both nodes MemFree
>=90GiB and reserve1048576KiB before starting. Keep the source-build/native
test window closed. Preserve fresh per-arm compiler caches and attention
state using the [existing cache-isolation procedure](2026-11-03-dense-int64.md#serving-image-and-cache-isolation-after-native-qualification).

Apply only one patch to a copy of the verified baseline recipe and stage it
atomically into `/home/nvidia/glm53-fp4-local`. Append the experiment ledger;
do not introduce duplicate active `.env` keys. Start through the unit.
Before timing, inspect actual container argv on both ranks, the resolved
batched/scheduled budgets, unchanged MTP depth and actual KV pool, capture
sizes and memory. A changed recipe file alone is not activation evidence.

After each window, atomically restore the saved baseline recipe, append the
rollback ledger entry, restore original cache directories, and start through
the unit after the same both-node gate. Verify image/recipe/MTP5/chunk4096,
health and capacity; re-arm healers. Keep all arm receipts and restore the
incumbent pending the user's separate adoption decision.
