# Bring-up and operations

## First start

```bash
cp env.example .env      # edit for your nodes; .env is gitignored
bash download.sh         # verifies shard count + pinned revision
local/prod-start.sh      # NOT start.sh directly — see below
local/acceptance.sh      # 7/7 expected: tools, thinking, vision, 36k needle
bash local/serving-test.sh   # from a client machine, through your tunnel
```

## Why prod-start.sh, not start.sh

A bare `start.sh restart` tears down a pair holding 82 GiB/node and starts again before
the kernel has reclaimed it — measured failure at 99.33 GiB free against a 103.44 GiB
gate, with the memory back a minute later. `prod-start.sh` = stop → wait for
`MemFree ≥ 90 GiB` on **both** nodes → start. (90, not 100: steady-state idle MemFree is
93–97 GiB, the rest is page cache that never reclaims at idle.)

It also hashes the shape-affecting `.env` knobs (`MAX_MODEL_LEN`, `MNBT`, `MAX_NUM_SEQS`,
spec config, `IMAGE`, `EXTRA_ARGS`) and **wipes the persistent Triton/TileLang JIT caches
on both nodes when the hash changes**. This is not optional hygiene: a k=7→5→7 spec A/B
left mixed-shape kernels in the cache and structured acceptance collapsed 0.96→0.58 until
both nodes' caches were wiped. Expect the first boot after any shape change to be a slow
cold-JIT boot; the unit allows 2 hours.

## systemd (user units, linger on)

- `vllm-glm53exl3.service` — oneshot, owns both ranks. `ExecStart` must be
  **restart-shaped**: on a FAILED start systemd runs `ExecStopPost` (not `ExecStop`),
  the wreckage keeps running, and a start-shaped retry then kills itself on
  `check_port_free` forever.
- `vllm-glm53exl3-watchdog.timer` (60s) — deliberately reads no occupancy metric.
  Signals: container **absent** = deliberate stop → stand down; **exited** = crash →
  heal; **running but /health down** past the load grace = wedged → heal; **forward
  progress frozen** while requests pend → heal. That last one exists because in vLLM v1
  `/health` returns 200 through a stuck NCCL collective — health is NOT a hang signal.
  Heals via `systemctl restart --no-block` (a blocking restart would hang the timer for
  up to 2h).
- `glm53exl3-check-updates.timer` (weekly) — node parity (incl. `vm.min_free_kbytes`),
  DGX OTA, image digest vs pin, upstream repo drift. Exit 1 leaves the unit failed on
  purpose — that IS the nag; check `systemctl --user --failed`.
- `glm53exl3-metrics-alert.timer` (5 min) — DFlash2 acceptance floor + **live PID-1 argv
  integrity** (KV pin present, loopback bind, exl3 flags). Alert-only, never heals.
- `glm53exl3-xid-monitor.service` (both nodes) — kernel Xid watch. Categorically never
  bounces the service: off-the-bus errors need a cold power-cycle, and an automated
  bounce would just eat the start-limit budget.

## A/B windows — lessons already paid for

- `systemctl --user stop vllm-glm53exl3-watchdog.timer` first; nothing re-arms it
  behind your back.
- `systemctl --user reset-failed vllm-glm53exl3` **before each window restart** — churn
  eats `StartLimitBurst=3/3600` and systemd then refuses starts (`start-limit-hit`)
  that look exactly like boot failures.
- Bench only after `ActiveState=active`, never on `/health` alone — the post-ready
  warmup sweep runs between the two and pollutes both throughput and the spec-decode
  counters.
- Re-arm the watchdog when done. A pair started by hand is invisible to it.
