# 14 — Enabling GPU performance counters on the Spark pair (owner runbook)

**Status: prepared, not executed.** This change needs root on both nodes and a
reboot, so it is the owner's call. Nothing in this document has been run.

## Why

The counter lane for tasks 1, 29, W5 and #55061 is blocked by one driver
setting, not by tooling:

```
$ cat /proc/driver/nvidia/params | grep -i restrict
RmProfilingAdminOnly: 1
$ nsys profile --gpu-metrics-devices=help
GPU Metrics: None of the installed GPUs are supported:
  Blackwell GB20B | NVIDIA GB10 PCI[000f:01:00.0] - Insufficient privilege,
  see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

`ncu` is not installed on either node; `nsys` 2025.3.2 is, and it reports the
GB10 as *supported but privilege-denied*. `dcgmi`/`nv-hostengine` are not
installed. `perf_event_paranoid=4` is a second, unrelated restriction on host
perf events.

Consequence today: PR #64's in-process torch profiler is the only counter path
that works. It gives kernel durations and launch geometry, which is enough for
share questions, but it cannot produce DRAM/L2 byte counters. Task 29 was parked
(`GAP_CANDIDATE_UNMEASURED`) and W5 (occupancy) was gated on exactly that
missing measurement.

## What the change is

One module option, in a dedicated drop-in so no existing file is edited:

```
# /etc/modprobe.d/99-nvidia-profiling.conf
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```

`NVreg_RestrictProfilingToAdminUsers=1` is the upstream default that makes the
profiling APIs root-only. Setting it to 0 lets local users collect GPU
performance counters. It has no effect on clocks, power limits or ECC.

The `scripts/enable-gpu-profiling.sh` helper in this repo writes the drop-in,
rebuilds the initramfs, and reports the state. It never reboots on its own.

## Steps (owner, both nodes)

1. Stop the vLLM unit and the watchdog on that node so nothing is mid-flight:
   `systemctl --user stop vllm-glm53exl3-watchdog.timer vllm-glm53exl3.service`
   (as `nvidia`), and let the peer idle.
2. Write the drop-in and rebuild initramfs:
   `sudo bash scripts/enable-gpu-profiling.sh`
3. Reboot the node: `sudo systemctl reboot`. Repeat on the other node.
   Both nodes must be rebooted; the option is read at module load.
4. Verify on both nodes:
   - `cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly` -> `0`
   - `nsys profile --gpu-metrics-devices=help` -> lists `NVIDIA GB10` with no
     privilege error
   - bring production back with `local/prod-start.sh` and confirm `/health`
     200, pool line unchanged, and the watchdog timers re-armed.
5. Capture one proof-of-capability receipt before any A/B claims a counter:
   a short `nsys profile --gpu-metrics-device=0` of a throwaway workload, or an
   `ncu --metrics dram__bytes_read.sum` run if `ncu` is installed first
   (`sudo apt-get install -y nsight-compute`).

## Rollback

`sudo bash scripts/enable-gpu-profiling.sh --revert` (removes the drop-in and
rebuilds initramfs), then reboot. Counters go back to root-only.

## Risk

- The profiling APIs become available to every local user. These are
  single-tenant boxes; treat it as a local trust change, not a network one.
- A reboot is required, so this belongs in a maintenance slot. It does not
  change weights, the KV pin, the image, or any serving knob.
- Profiling while production serves still perturbs it. Keep the existing
  contract: `ncu` replay stays off the live CUDA-graph server; use
  `nsys --gpu-metrics` or a stopped window.

## What it unblocks

- Task 29: a measured DRAM-byte counter for the fused `exl3_moe` decode path,
  replacing the advisory 218 GB/s weight-streaming model.
- Task 24 W5: the achieved-occupancy measurement the cubin sweep is gated on.
- Task 1: the exact-workload traffic measurement that gates selective
  quantization.
- #55061: the `index_select` overflow-path probe.

Until this is run, those stay parked and the software-only oracles in
`scripts/audit_*_kernel_share.py` remain the only counter path.
