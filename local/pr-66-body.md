Task 24 W5 closes the E3 follow-up queue. W5 was the only item gated on a
hardware counter, so the first job was to get one. `RmProfilingAdminOnly` is
enforced as a `CAP_SYS_ADMIN` check: `docker run --gpus all --cap-add SYS_ADMIN`
clears `ERR_NVGPUCTRPERM` with no host drop-in and no reboot, and
`nsys --gpu-metrics-devices=help` then lists `0: Blackwell GB20B | NVIDIA GB10`.
What the capability does not remove is the window: production holds essentially
all unified memory, so a second CUDA process cannot even `cudaMalloc(1 KiB)` or
call `cudaMemGetInfo` while the server runs (measured, host and container), and
the capture has to run with production stopped. docs/14 is corrected for both
facts (ncu 2025.3.1 is installed at `/opt/nvidia/nsight-compute/2025.3.1/ncu`,
just not on `PATH`) and now documents the container route; the host drop-in plus
`scripts/enable-gpu-profiling.sh` stays prepared, not applied.

The gate was pre-registered in docs/11 §8 before the window: a sweep opens only
if min-across-kernels achieved occupancy is < 0.8 × theoretical **and**
registers/thread ≤ 96. Static accounting from the production cubin already said
`fm_gateup_kernel` / `fm_down_kernel` = `REG:128 STACK:16 SHARED:1024 LOCAL:0` on
`e3-w3-zfill`, so 256 threads × 128 regs × 2 blocks = 65,536 = the whole 64K
register file, and the register file binds at 2 blocks/SM.

Six independent ncu captures at production per-rank geometry (hidden 4096, TP2
inter 1024, TRF cap 32, 64 routed experts, `--launch-skip 1 --launch-count 3`)
agree, the last one with the frozen final bytes. That profile records
three launches in total, not three per kernel: `fm_down_kernel` twice (IDs 0 and
2) and `fm_gateup_kernel` once (ID 1). Measured gateup **33.01–33.02%** and down
**33.15–33.20%** achieved occupancy (the minimum across a kernel's launches; the
two down launches read 33.17/33.17, 33.17/33.18, 33.18/33.20, 33.20/33.18,
33.15/33.17 and 33.20/33.20
across the six
captures) against 33.33% theoretical (99.0 / 99.6%), block limits registers 2,
shared memory 3, warps 6, grids 2304 = (8, 288) and 4608 = (16, 288).
Registers/thread is 128, above the 96 stop threshold, and a third block/SM would
need ≤ 85 regs/thread and would spill.
**W5 STOP by measurement:** no cubin, no Python change, production stays
`EXL3_FAT_GROUPED=1` with last-wins `EXL3_TEMP_ROWS_FUSED=32` on `e3-w3-zfill`.

New tooling: `scripts/probe_e3_occupancy.py` (offline E3 grouped replica at
production per-rank geometry), `scripts/audit_e3_occupancy.py` (ncu CSV judge;
keeps launches separate, gates on the minimum across a kernel's launches, and
fails closed with ABORT on a missing kernel, a missing or out-of-range metric
(including the launch geometry it cites as evidence), an unreadable metric value
(`N/A`, `ERROR (...)`), a row that carries a metric but no kernel identity, a row
without a launch id, a launch that drops a metric its siblings report, or an
achieved value above the theoretical one, and holds the capture to the launch
count ncu was asked for; `STOP_REGISTER_HEADROOM` /
`STOP_NO_GAP` / `OPEN_GAP` / `ABORT`), and
`scripts/run_e3_occupancy_window.py` (guarded 8-phase window: preflight, disarm,
stop, capture, judge, start, gates, rearm; `--state` resume; ncu runs in a named
capped container whose removal is confirmed before production is (re)started,
including on a resume straight into `start`; SIGTERM/SIGHUP take the recovery
path; the restart and the timer re-arm are attempted independently, so a failed
restart cannot leave monitoring off; production is not started on top of a
profiler whose removal could not be confirmed; a resumed window keeps the
capture knobs its receipt recorded (`--launches`, `--n-exp`, `--iters`,
`--need-gib`, `--ncu-timeout`) and refuses a conflicting override with exit 2; a
zero
`--iters/--n-exp/--launches` is rejected before production is stopped). 59 new
CPU-only tests across two files, and CI now compiles `scripts/` as well as
`local overlay tests`.

Window integrity (`local/task24-w5-window-20260909-r7.json`, frozen final bytes,
plus `…-r6.json`, `…-r5.json`, `…-r3.json` and `…-202752.json`): all 8 phases ok,
acceptance
7/7 rc 0, `.env` never edited, image sha `d8144f02…` identical before and after,
JIT stamp `078835f1d75f` unchanged, `GPU KV cache size: 1,396,551 tokens … 1.40x`
unchanged, watchdog and metrics-alert re-armed. A window with a deliberately bad
`--iters 0` (`local/task24-w5-failpath-window-20260909.json`) validated the
failure path on the cluster: capture failed, the named container was removed,
production was restarted, the timers were re-armed, `"recovery": "ok"`, exit 1.
Windows with `--ncu-timeout 1`
(`local/task24-w5-timeout4-window-20260909.json` on the frozen final bytes, plus
`…-timeout3-window-20260909.json` and `…-timeout2-window-20260909.json`)
exercised the timeout path: the
Docker client was killed at the timeout, the hung profiling container was removed
(a container `--rm` alone had not cleaned, and recovery's own cleanup then
confirmed it was gone), capture failed with
`RuntimeError('ncu exceeded 1s')`, and recovery restarted production and re-armed
both timers. The first window's auditor exited 1 on
ncu's CSV log preamble; the parser was fixed and all
captures were re-judged with the final bytes (identical verdicts). Receipts:
`local/task24-w5-window-20260909-{201512,202752,r3,r5,r6,r7}.json` + logs,
`local/task24-w5-{failpath,timeout,timeout2,timeout3,timeout4}-window-20260909.json`
+ logs,
`local/task24-w5-occupancy-20260909-{201521,202800,211435,215941,221957,225704}/`,
docs/06,
docs/11 §8. Frozen sha256 — probe `5fceb5ee…`, auditor `698973f5…`, runner
`4e516a33…` — identical locally and on the node for both final-byte runs (r7 and
timeout4). Resume handling was cluster-checked separately with judge-only
resumes: a receipt recording 3 launches re-judges at 3, one recording 6 makes the
auditor reject the 3-launch capture ("expected 6"), and `--launches 6` against a
receipt recording 3 exits 2 without touching the capture.
