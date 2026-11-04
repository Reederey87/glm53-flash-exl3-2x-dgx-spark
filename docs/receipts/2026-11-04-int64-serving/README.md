# Incumbent baseline recovery and recorded retry

The Int64 native qualification remains complete on both nodes. Full serving
qualification remains open: incumbent baseline, rebuilt control, then candidate
must each complete the frozen matrix. This directory records recovery and the
fresh incumbent startup; it does not claim a completed matrix or adoption.

## Same-value reserve recovery

Frozen helper `2e41e0f`, SHA256
`027731fcb6ef19ba7d334e1ba996f289e85b7e7fff51059f8b79858b0257e63c`,
passed read-only preflight and actual application on both nodes. Normal-zone
watermark boosts586588/7680 pages became0. The reserve1048576KiB, scale10,
boost factor15000, managed pages and base min/low/high watermarks all stayed
unchanged. The operation used the default container security profile, all
capabilities dropped and only the exact reserve proc file mounted writable.
No persistent sysctl file, model, driver, kernel or serving code changed.

The paired receipts measured head MemAvailable113209780→117927616KiB,
an increase of4717836KiB (**4.50GiB**); worker119530504→119561224KiB.
This is available-memory accounting, not a measured amount of file-cache
reclamation. The earlier approximate2.1GiB observation compared separate
snapshots; use these paired values. Raw zoneinfo/meminfo snapshots are retained
locally and on each node. The helper requires actual vendor-kernel verification,
not just the upstream source rationale in the [procedure](../../../experiments/startup-memory/watermark-recovery.md).

The first read-only head preflight could not read a mode0600 staged script
with all capabilities dropped. Making only that non-secret script mode0444
resolved file access; the same helper bytes then passed both preflights. No
security-profile override or broader privilege was introduced.

## State observed after the pause

On resumption at02:52UTC, the service was already healthy under invocation
`389165217f364c43b9ef01c46b5209a2`, started23:51:53UTC on November3 and active
since00:05:03UTC. Its start occurred while this work was paused; it is not
represented as our recorded retry. Both containers used incumbenta7f76ae9.
Effective serving arguments, exact NVIDIA checkpoint revision and metadata,
2,350,210KV tokens, active environment, host configuration and mounts matched
the saved control. At inspection, completed/running/waiting requests and
prefix hits were0. The watchdog was active.

All five compiler directories on each node still had birth times22:58:22UTC,
matching the initial empty-cache preparation; original pre-experiment backup
inodes were unchanged. These observations establish healthy rollback evidence.
They do not supply a contemporaneous paired MemFree gate for that intervening
start, so it was not substituted for the fully recorded baseline.

## Recorded retry

After confirming no requests, stopped the watchdog and inference through their
units. Both ranks terminated and both nodes then exceeded90GiB MemFree. The
prepared `release_closed_native_cache.py` was **not applied**: further file-cache
recovery was unnecessary. It remains a separately frozen recovery option with
exact closed-result identity checks, not a serving treatment or adopted feature.

Frozen cache helper `31456d3`, SHA256
`67642fe5d75374acd4b667bd39c06e9735c83cb23b3e44bd49f10da4dbea088a`,
preserved the intervening caches under `*.int64-before-incumbent-retry1-20261103`
on both nodes and created empty original paths with the same permissions.
Original `*.int64-before-incumbent-20261103` backups remain intact. Neither
Hugging Face cache nor checkpoint contents were changed.

Final pre-start readiness measured head MemFree120343688KiB and worker
120326992KiB, no containers, reserve1048576KiB on both. Recipe and launcher
hashes match the saved control; both aliases remain the incumbent. The active
`.env` assignments were compared without exporting them; only a comment ledger
was appended, and the new backup remains on the head.

Requested unit-only startup at **02:59:08UTC on November4**, invocation
**`b9c5ac7bd82a47c8a5e98424d537f5e8`**. This is the live startup to observe; an
observation timeout is not terminal state and must not trigger another start.
Watchdog is disarmed. Health, exact runtime/API/KV/cache qualification and the
full96-cell matrix remain required before any performance decision. Use frozen
benchmark05c6023, seed2026110304,3reps,40warmups,max reasoning,200generated tokens,
8k–128k and concurrency1/2/4/8 with cap4; retain all original rejection gates.

All323 repository tests passed before staging the retry helper. These tests
cover recovery guards and measurement contracts, not serving performance.
The manifest covers the selected non-secret receipts here; configuration
snapshots and private container inspection remain on the cluster.

## Baseline qualification and measured-run activation

The same recorded serving invocation `b9c5ac7bd82a47c8a5e98424d537f5e8`
reached active state with health 200. At 03:16:51 UTC, both actual container
image IDs, active environments, host configuration and mounts matched the
saved incumbent. Both ranks passed exact serving-argument and NVIDIA revision /
metadata checks. The recipe, launcher and frozen functional/benchmark hashes
also matched. The KV pool remains exactly **2,350,210 tokens**.

All six frozen API checks passed: health, model identity, chat, parsed reasoning,
tool call and prefix reuse. The separate prefix ladder returned all eight rows:
first requests had zero cached tokens and repeat requests reused all
1,024 / 4,096 / 8,192 / 16,384 tokens. All cache labels and geometry match the
previous verified control, including the 17 GiB FP8 pool. Local measurement /
comparison regression tests also passed.

The frozen full baseline matrix is running as
`glm53-int64-baseline-retry1-matrix-20261104.service`, invocation
`d9e6284b9a1349ea8c500b46693f28f8`. The exact command and initial process state
are retained under `incumbent-retry1/head/`. It uses seed 2026110304, three
repetitions, 96 measured cells and 40 warmups, with maximum reasoning and
200 generated tokens per decode request. These counts include reasoning.
The registered reciprocal-TPOT and normalized aggregate conventions remain
unchanged; usage-update timestamps do not imply individual token arrival times.

This activation is not a completed performance result. Rebuilt control and
candidate serving trials remain pending. Watchdog remains disarmed during the
window; restoration and rearming remain required before leaving the window.

## Live private runtime-verifier qualification

At 04:04:41 UTC, frozen verifier `2bea9e3` passed on both running incumbent
containers against their pre-matrix private snapshots. Actual image identity,
environment, launch configuration, host settings and all mount records match;
the serving invocation is still `b9c5ac7b`. Both nodes' four build-worker
settings equal their exact incumbent image values, so no existing override
contradicts the registered rebuilt-control inheritance check.

[`runtime-verifier-2bea9e3.json`](incumbent-retry1/head/runtime-verifier-2bea9e3.json)
contains only whitelisted results. Full inspection and environment values
remain on the cluster. The verifier ran as a read-only metadata check without
GPU calls, inference requests, or container/config changes. This proves the
unchanged-incumbent comparison path; rebuilt-control and candidate runtime
qualification still require their actual future containers.

## Original-cache restoration preflight

At 04:26 UTC, frozen helper `1f2eefe` passed read-only planning on both nodes
against the exact original preservation receipts. All five original directory
identities match and the prospective final-arm archive paths are absent.
Calling only the local idle guard while serving is live correctly refused on
both nodes. No restoration/rename function was called; `applied=false` in
both [preflight receipts](cache-restore-preflight-1f2eefe/).

The recorded active directories belong to the current incumbent baseline,
not the future candidate. Recompute the plan after the full window and prove
both nodes idle before applying. The helper's filesystem tests cover retention,
partial rename rollback and refusal to overwrite an intervening writer.
Actual original-cache restoration, incumbent startup/API/KV verification and
watchdog rearming remain pending at the end of the experiment.
