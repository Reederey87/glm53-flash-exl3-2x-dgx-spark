# Both-node startup readiness candidate

Candidate 26 changes only the service's readiness check. It is staged here;
the repository service and live service remain unchanged pending its own A/B
window. Do not combine this treatment with MTP depth or a new image.

The current live unit matches `systemd/vllm-glm53fp4.service`, SHA256
`af944e8695e5e6db8b4152f591857d1f67d2a7fb5e1432dc25cc4760c35e9631`.
It gates only head `MemAvailable >=100 GiB`. Workspace policy instead
requires **both** nodes' `MemFree >=90 GiB` before a start/build. This
conservative idle policy is not a general CUDA capacity estimate.

`wait_cluster_idle.py` reads Docker state, `/proc/meminfo` and the reserve
setting on the head and worker. It requires no serving container, including
stopped containers, and `vm.min_free_kbytes=1048576` on both nodes. Every
attempt rechecks both nodes. An unavailable worker, Docker failure, invalid
response or deadline fails closed. It never stops containers, reclaims cache,
changes sysctls, or uses a GPU. SSH uses the direct QSFP address, with proxy
commands and jumps disabled. Probes and the overall wait are bounded.

These are sequential readiness snapshots, not a reservation against other
operators starting work immediately afterward. Exclusive maintenance-window
ownership remains required. This helper gates serving startup only; the
separate full-source build wrapper already has its own both-node idle gate.

## Research before implementation

Exa lookup on November 3 covered Spark unified-memory accounting, multi-node
readiness, systemd failure handling, and alternatives such as cache reclamation
and changing memory policy. Sources:

- [NVIDIA Spark known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)
  distinguishes free CUDA memory from reclaimable unified memory. It does
  **not** prescribe our 90 GiB `MemFree` threshold; that is workspace policy.
- [systemd.service](https://freedesktop.org/software/systemd/man/latest/systemd.service.html)
  documents that a non-ignored `ExecStartPre` failure prevents `ExecStart`,
  and failed starts can still run `ExecStopPost` cleanup.

Local source review followed that lookup: vLLM `vllm/utils/mem_utils.py`
uses available memory for its integrated-GPU allocation estimate; this
candidate does not alter that profiler. Exllamav3's TP exception propagation
and collective timeout paths were reviewed without importing its quantization.
TensorFold `zig/src/families/glm/cuda_roce_multi.zig` requires every rank to
vote ready before a gather. The portable technique used here is to require
both current-cycle observations, never an old head success plus a later
worker success. This is not a port of TensorFold's server or transport.

Cache dropping, reserve changes and driver/kernel upgrades are excluded.
The current reserve remains pinned; a mismatch is reported rather than fixed
silently. The running NVIDIA NVFP4 checkpoint and KV pool are preserved.

## Frozen staging and qualification

Freeze this directory, tests and preregistration in a commit before copying
anything to a node. Record script, patch and resulting unit hashes. Apply
`service.patch` only to the exact baseline unit hash above. Check the patch
locally with `git apply --check experiments/startup-memory/service.patch`.

First stage the script under an experiment-only path and run `--once` while
the model is loaded. Expected exit is 1 with both node observations; this is
a read-only refusal check, not a service change. Keep its JSON receipt.
The local fault tests cover low free/high available memory, stale containers,
reserve mismatch, malformed responses, SSH failure, deadlines and stale
observations. Do not create real low-memory or network-failure incidents to
exercise these branches on the serving cluster.

The later service trial follows
[the preregistered gate](../../docs/gates/2026-11-03-startup-memory.md).
Stage the helper atomically as `scripts/wait_cluster_idle.py` only during that
window, along with the independently patched unit. Preserve the old unit and
any prior helper before replacing either. Run `systemd-analyze --user verify`,
then `systemctl --user daemon-reload`. Start only through the production unit.
The existing retry and cleanup behavior remains unchanged.

No native code, Python dependency or image build is involved. Tests are run
through the existing locked uv project. Adoption requires actual unit-start,
functional, capacity and speed receipts, then user approval.
