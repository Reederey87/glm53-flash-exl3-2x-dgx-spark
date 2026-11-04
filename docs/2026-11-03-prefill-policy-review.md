# Solo and mixed prefill: candidate 22 prerequisites

The supplementary probe is now frozen at `f5d049e` and staged on spark1 at
`/home/nvidia/glm53-benchmarks/mixed-prefill-f5d049e`. Its five file hashes
match and its offline CLI/import check passes on the ARM64 host. These
[staging receipts](receipts/2026-11-03-mixed-prefill/) do not establish live
overlap or a scheduler benefit; the feasibility pilot remains pending.

This is source and saved-runtime evidence, not a cluster performance result.
No scheduling setting has been changed. It narrows the implementation options
for candidate 22 and the mixed-load qualification needed by chunk-size
candidates 07/08.

## Research and transferable techniques

Exa review of [vLLM tuning guidance](https://docs.vllm.ai/en/stable/configuration/optimization/)
confirms the expected tradeoff: smaller chunks reduce decode interruption,
while larger chunks can reduce prompt latency. Its large-GPU recommendations
do not establish a winning chunk size for this two-GB10 NVFP4 deployment.
The older [concurrent partial-prefill change](https://github.com/vllm-project/vllm/pull/10235)
is a stronger scheduling option, but its original flags must not be assumed
to match the deployed fork's current interfaces.

TensorFold's `zig/src/families/glm/cuda_prompt_solo.zig` uses 4096-row prompt
passes only when other streams are not decoding, with 2048-row mixed passes.
The choice also checks available buffers, prefix restoration and index-ring
boundaries. Exllamav3 `generator/generator.py` has separate `max_chunk_size`
and `shared_chunk_size`, with optional prefill cooldown and rotation across
prefilling jobs. These are useful policy ideas; neither implementation is
being imported into this NVIDIA NVFP4 runtime.

The deployed vLLM fork already contains a related opt-in path described in
[PR 880](https://github.com/local-inference-lab/vllm/pull/880). Exa located it,
but fetching the canonical PR and pinned scheduler page timed out; the
canonical PR was then read with `gh api`, and the exact deployed commit was
read from the local git object store. The PR remains open; the actual source
at `8244d1c44348ae8a5bbfc9f1b607021652283ed4` establishes presence in this
deployment. The PR's different-model measurements are not GLM results.

## Why the environment switch alone is ineffective here

At that exact commit, `vllm/v1/core/sched/scheduler.py` widens a prefill-only
step from `max_num_scheduled_tokens` to the input budget only if **all** of
these conditions hold:

- `VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS=1` (the default is 0).
- `needs_decode_count` is true and no eligible decode is present.
- The current token budget is positive.
- The scheduled-token cap is smaller than `max_num_batched_tokens`.

Decode counting itself is conditional: prefill cadence throttling, a compute
share controller, or parallel prefill with the decode-aware policy must
activate that path. Baseline defaults are interval 1, no compute share,
one parallel prefill and round-robin policy. The unchanged baseline recipe
does not override them. With MTP and no explicit scheduled-token cap,
`vllm/config/vllm.py::_set_max_num_scheduled_tokens` resolves the cap to the
batched budget, 4096. Both required distinctions are therefore absent.

Saved pre-build image `a7f76ae9...`, recipe SHA256 `cd0d994a...`, and the
selected container environment were checked read-only at 18:54 UTC. No
override for the six inspected scheduling controls is present. Receipt:
`local/prefill-policy-review/snapshot-controls.txt`. This supports the baseline
configuration finding, alongside the retained
[head](receipts/2026-11-03-mtp-depth/restored-head-runtime.json) and
[worker](receipts/2026-11-03-mtp-depth/restored-worker-runtime.json) actual
serving arguments. The branch has not been instrumented during GPU work.

The pinned upstream unit test also hides a relevant prerequisite in its
fixture: `test_prefill_only_step_may_use_the_batched_token_budget` constructs
`_create_fair_scheduler`, whose default compute share is 0.5. Its passing
condition cannot be generalized to a scheduler with all baseline defaults.
This test was inspected, not executed on the cluster.

## Experimental consequence

Do not register an environment-switch-only trial and call a successful boot
evidence that this policy was active. Candidate 22 needs separate qualification
of its prerequisite scheduling policy and smaller scheduled-token cap, then
a paired on/off switch trial with those prerequisites frozen. The fork also
provides `max_num_prefill_tokens_per_step`, but it requires compute sharing;
setting it alone is not an independent shortcut.

Keep chunk 2048/8192 trials separate: they change the overall batched-token
budget, which can alter static buffers and compilation, unlike merely sharing
that budget across work. Preserve the checkpoint and actual 17 GiB/rank KV
pool, and verify the effective scheduling configuration before measurement.

The extra workload must deliberately submit a long, unique prefill while
short requests are already decoding. Retain per-request token-progress times,
request start/end times, TTFT, output budgets and counter deltas. Confirm
observed request overlap; an all-prefill or queue-only run cannot establish
mixed-load improvement. SSE bundles do not reveal individual token times, so
report gaps between usage updates with that label. These mixed-load checks
supplement the required complete speed matrix and cold/hot recurrent-state
checks. The [supplementary mixed-load probe](../experiments/mixed-prefill/README.md)
now implements client-observed overlap checks, usage-update gaps and bounded
counter windows without changing the frozen full speed matrix. Local real-HTTP
and fault tests pass. Live feasibility, exact sub-arms and adoption gates
remain pending; no scheduling setting has changed.
