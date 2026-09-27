# Compute the GLM5Next router once

The deployed fork computes `self.gate(hidden_states)` in `Glm5NextMoE`
and passes the result into `MoERunner`. The runner holds the same gate and
overwrites those logits before expert dispatch. `GLM53_ROUTER_ONCE=1`
removes the first calculation only when `experts.gate is self.gate`.
A runner without that gate still receives model-computed logits.

This is a Python runtime patch, with no weight, precision, kernel, packing,
or KV-cache change. The installer rejects a missing/ambiguous source anchor
and partial installation. Default is off; rollback is `GLM53_ROUTER_ONCE=0`
followed by a new container. Turning the flag off does not undo edits inside
an existing container. Both ranks mount and apply the same installer.

## Evidence and scope

Exa research on 2026-09-27 found the same ownership rule in
[vLLM's optional-router-logits proposal](https://github.com/vllm-project/vllm/pull/41747):
only runners supplied an internal gate may omit externally computed logits.
This proposal is supporting design evidence, not proof of this fork's behavior.
The live `e3-armc-guards` image was read directly: model forward computes the
gate, `FusedMoEFactory` receives that gate, and `MoERunner._forward_impl`
replaces the incoming logits before dispatch. Local upstream GLM5Next already
passes hidden states as a placeholder. No upstream runner refactor is needed.

The discarded 288-by-4096 bf16 matrix is roughly 99 MB across 42 MoE layers
per forward. This suggests a small saving, not a large end-to-end speedup.
The actual result depends on graph compilation, memory reuse, and draft
acceptance. Benchmark warmed structured counting, hashmap prose, and a
low-acceptance technical essay on the same image and controls.

Behavioral tests exercise both internal and external gate paths and check
that routed output is preserved while the duplicate call disappears. Installer
tests cover disabled mode, invalid flags, repeat application, anchor drift,
partial installation and duplicate anchors. These do not replace cluster tests.

## Follow-up priorities

1. Per-request cache attribution and a mixed-agent benchmark. Verify the existing
   `--enable-prompt-tokens-details` path in HTTP and SSE before using it to
   attribute return hits. [The upstream report](https://github.com/vllm-project/vllm/issues/44961)
   illustrates why the flag alone is not proof. Include more than four callers
   and record waiting, preemptions, TTFT and completed tokens.
2. Known one-shot batch/eval clients can use the already shipped no-store API;
   qualify agent retention and chaff throughput before integrating callers.
3. Profile shared-expert stream overlap at the actual verification row counts.
   The deployed runner supports a separate stream, whose synchronization may
   help or hurt small batches. [Upstream's platform-specific fix](https://github.com/vllm-project/vllm/pull/30085)
   demonstrates sensitivity, but its AMD regression is not NVIDIA evidence.
   First check whether EXL3 owns shared experts and whether the overlap branch
   runs at all. Only then consider a separate one-knob window; no change here.
4. Keep the reused-prefix cap, dual-tail registration and waiting retention
   behind measured workload triggers. [Waiting-retention results](https://github.com/vllm-project/vllm/pull/54366)
   preserved hits but did not show a TTFT gain and added scheduler CPU cost.

TrellisMX TP2/SM121 feasibility and fused GDN metadata remain separate research
tasks. TensorFold's static graph buffers and ExLlamaV3's MoE implementations
were inspected as design references; neither supplies evidence to change this
deployment's attention state or expert kernels in the router window.

## Cluster qualification

Pending the pre-registered control/candidate window. Gate and raw receipts:
`local/router-once-20260927/`. Do not infer adoption from installer success.
