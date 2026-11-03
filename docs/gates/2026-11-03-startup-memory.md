# Both-node startup memory gate — preregistration

Prepared while the separate MTP3 experiment is running. No startup gate
treatment has been applied. This candidate may run only after MTP5 has been
restored and the MTP3 receipts and rollback are complete.

## Independent variable and frozen controls

Replace only the head-only `MemAvailable >=100 GiB` service precheck with the
read-only helper in `experiments/startup-memory`. Require both current-cycle
node observations: no `vllm_node` containers, `MemFree >=94371840 KiB`, and
`vm.min_free_kbytes=1048576`. Probe errors or the 600-second deadline prevent
serving startup. No sysctl, cache-reclamation, retry or cleanup changes.

Freeze the helper and service patch in a commit before staging. Preserve the
unit's SHA256 `af944e8695e5e6db8b4152f591857d1f67d2a7fb5e1432dc25cc4760c35e9631`
as the rollback. Record the candidate unit hash before its first start.

Hold the incumbent MTP5 recipe, image ID
`sha256:a7f76ae9f18c7701c9cde5f72c606393d9a21ca94dd8e53b27589935f64fcbe2`,
checkpoint `nvidia/GLM-5.3-Flash-FP4-Local` revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, TP2, max sequences 4, chunk 4096,
1,000,000 context, 17 GiB KV per rank and 2,350,210 usable KV tokens fixed.
Keep all effective launcher arguments, mounts, image, caches, network,
driver 580.173.02 and kernel 6.17.0-1031-nvidia unchanged. Verify actual vLLM
process arguments on both ranks; Docker's sleep-infinity command is not
evidence of serving arguments.

## Safety and readiness gates

1. Local fault tests must reject either low node, stale serving containers,
   reserve mismatch, unreachable worker, malformed output and deadlines.
   They must not combine observations from different polling cycles.
2. A frozen helper run with `--once` against the loaded cluster must return 1
   and retain both node observations without affecting the service.
3. Before the trial, snapshot the unit, helper if present, runtime config,
   hashes and container/image controls on both nodes. Stop healers and the
   actual FP4 unit, verify both containers gone, then independently confirm
   both MemFree thresholds before staging. No build is needed.
4. The helper must pass on the idle pair and the patched unit must reach
   healthy status within its existing startup timeout. Keep the actual
   ExecStartPre invocation, readiness observations, journal and timing.
   A successful separate helper run alone does not prove unit integration.
5. Reasoning, tool call, API, prefix-cache and KV checks must pass with the
   exact unchanged checkpoint and capacity. Any CUDA/Xid error, rank loss,
   corrupt output, sustained swapping or failed readiness aborts to rollback.

## Performance and decision

Use the same frozen `05c6023` speed harness and seed 2026110304 for incumbent
and candidate. The completed November 3 MTP5 control is eligible only after
verifying all frozen controls match; no changed harness or prompt identities.
Three reps, unique no-cache prefill 8192/16384/32768/65536/131072, c1/2/4/8,
cap 4; code/prose/structured decode 200 generated tokens including reasoning,
explicit max effort, per-cell warmup. Retain full raw receipts and acceptance
counter deltas. No unrelated inference during measurement.

A rate regression greater than 2% in at least two of three matched reps in
prefill, decode or speculative acceptance rejects the candidate. Changes
inside the observed control spread are inconclusive; remeasure paired cells
when needed. No inference-speed gain is expected. Neutral performance can
support this reliability improvement only when readiness and unit integration
are proven and all correctness/capacity gates pass.

After receipts, restore the original unit atomically, remove only the helper
introduced by this trial (or restore its snapshot), daemon-reload, and start
through the unit after independently confirming both nodes idle. Append the
trial/rollback outcome to the experiment ledger and `.env` as comments only;
the upstream parser rejects duplicate active keys. Verify MTP5, image,
checkpoint, capacity and health, then re-arm the FP4 watchdog. Ask for final
adoption only with the measured result and rollback receipt. Do not leave
this candidate permanently active on the strength of a helper-only check.

## Read-only qualification receipt

Helper/service patch frozen at `5bc557b`. The helper SHA256 is
`82a2f656de6c766f02e76b273cf537397589a2d9fefcc7622c5ebf8798b93745`;
resulting candidate unit SHA256 is
`d659d4e4fbc0d64801efa35716e7bb8a2fc3cb352db367e2a92bcddc11892cae`.
The candidate unit has not been staged or activated.

The experiment-only helper's `--once` probe returned exit 1 in 1.153s against
the loaded MTP3 pair. Both node reads succeeded, both showed one serving
container and reserve 1048576. Head/worker MemFree was 1,957,192/2,545,804 KiB;
MemAvailable was 1,802,144/2,102,352 KiB. Refusal was expected. The service hash
remained the original `af944e86...`; the MTP3 matrix remained active/running
with PID 188858. No inference request was sent by this probe.

Raw `loaded-refusal.{json,log}` and `loaded-refusal-exit.txt` are retained in
runtime `local/ab-20261103/startup-memory-5bc557b`, with local copies under
`local/cluster-20261103/startup-memory-5bc557b`. Local repository suite:
129 passed, including 22 readiness fault tests. This qualifies only the
read-only refusal path. Idle readiness, service integration, full serving
gates, rollback and adoption remain pending.
