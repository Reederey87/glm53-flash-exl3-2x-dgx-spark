# Fresh incumbent startup — no serving measurements

The unchanged incumbent image `a7f76ae9…` started through
`vllm-glm53fp4.service` at 22:58:23 UTC on November 3, invocation
`b6a625d4142043c0a5222b0e85d7ede4`. The head worker failed device initialization
at 22:59:22: its vLLM memory snapshot reported **105.72 GiB available**, below
the unchanged **105.87 GiB** requirement from utilization 0.87. The worker
rank then lost its head connection. No health, functional or speed gate passed;
no matrix started and no candidate serving result exists.

Before starting, frozen helper `3e71306` preserved all five compiler-cache
directories on both nodes, verified their original inodes and permissions,
and created empty original paths. The checkpoint cache was excluded. Both
nodes were idle, reserve was 1048576 KiB, and the final manual gate measured
head MemFree 94,638,744 KiB and worker 117,439,296 KiB, above 94,371,840 KiB.
The recipe, image aliases and active environment assignments were unchanged;
only a comment ledger was appended. Original snapshots remain on the nodes.

The logs were preserved before stopping through the service. The container's
wrapper was still running when captured, so its `running 0` state is not a
successful engine exit. Explicit service cancellation leaves terminal
`Result=signal`, `ActiveState=failed`, MainPID 0. This is distinct from the
recorded engine initialization exception. Both serving containers are gone;
watchdog remains inactive for the experiment window.

The installed incumbent's `MemorySnapshot` already uses `psutil.available`
for integrated GPUs outside WSL. The quoted 105.72 GiB is therefore the
**vLLM startup snapshot**, not an independently measured raw CUDA-free value.
The 90 GiB host-free guard is necessary but did not guarantee this later
startup check. Keep utilization, checkpoint, KV capacity and runtime code fixed.

Exa research covered [NVIDIA UMA memory accounting](https://nvidia.custhelp.com/app/answers/detail/a_id/5728/~/unexpected-available-memory-reporting-on-dgx-spark),
[vLLM integrated-GPU accounting](https://github.com/vllm-project/vllm/pull/35356),
the stronger [allocator-budget proposal](https://github.com/vllm-project/vllm/pull/49760),
and [targeted file-cache advice](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html).
Afterward, reviewed pinned/local vLLM memory accounting and `request_memory`,
Exllamav3's integrated-device/reservation handling, and TensorFold's explicit
snapshot lifetime and allocator release. A new budget or allocator patch would
be a separate treatment, not a repair silently bundled into this baseline.

Existing frozen, CPU-only, read-only recovery helpers were reused for the exact
completed control image and freshly verified inactive BuildKit uv cache. Their
receipts show unchanged metadata but no material recovery. No global cache purge,
file deletion, driver/kernel change, model-content read or modification occurred.

Read-only `mincore` diagnostics then measured these finite scopes:

| Scope | Resident bytes | Coverage |
|---|---:|---|
| Completed build workspaces/exports | 536,444,928 | 94 files at least 16 MiB |
| Host dependencies/completed native results | 562,233,344 | 86 files at least 16 MiB |
| Exact checkpoint revision | 0 | All 42 indexed files; index hash verified |
| Declared layers of five completed images | 2,315,935,744 | 265 files at least 16 MiB across 82 layers |

These are scoped observations, not a complete accounting of host page cache;
do not sum them as disjoint totals. The image-layer result includes superseded
cuSPARSELt and NCCL libraries hidden by later layers. No advice was applied to
those newly identified files. The checkpoint uses shared, sharded Hub blobs;
an initial probe refused the assumed model-local blob layout before mapping
any weights. The corrected read-only probe checked the observed layout.

At 23:10 UTC the head has MemFree 93,705,364 KiB, below the unchanged manual
gate; worker has 116,733,076 KiB. Both aliases still resolve to the incumbent.
Recovery and a fresh baseline retry remain open. Preserve this failed invocation,
its logs, and the original cache backups. Any retry must start with empty compiler
caches, retain both-node guards, and pass exact API/KV/identity checks before the
frozen 96-cell matrix. Rebuilt-control and candidate serving tests remain pending.

Full logs and residency records remain under the runtime's
`local/ab-20261103/int64-serving/incumbent/`; selected non-secret receipts and
their manifest are published here. Configuration snapshots containing `.env`
remain only on the cluster.
