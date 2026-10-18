# 2026-10-17 — boot unit and watchdog receipt

Cluster: `spark-a183` (head) + `spark-7829` (worker), GB10 pair.

## Boot path

Installed and enabled on the head as user `nvidia`:

```text
vllm-glm53fp4.service              enabled  -> ~/.config/systemd/user/default.target.wants/
vllm-glm53fp4-watchdog.timer       enabled  -> ~/.config/systemd/user/timers.target.wants/
Linger=yes
```

The previous production boot unit is **disabled**:

```text
vllm-dsv41-two-sparks.service              disabled
vllm-dsv41-two-sparks-watchdog.timer       disabled
```

Cold start through the unit (not a physical reboot):

```bash
cd /home/nvidia/glm53-fp4-local
./launch.sh down
systemctl --user start vllm-glm53fp4.service
```

Result:

```text
ActiveState=active  SubState=exited  Result=success  UnitFileState=enabled
/health 200
GPU KV cache size: 2,078,358 tokens, Maximum concurrency for 500,000 tokens per request: 4.16x
```

The unit's `ExecStartPre` waited for Docker, the worker's Docker over SSH, and
`MemAvailable >= 100 GiB`; `ExecStartPost` held the unit in `activating` until
`/health` returned 200. `PULL_IMAGE=0` used the staged `vllm-node-b12x` image
and the 42-file checkpoint. A full cold start took about 5 minutes with warm
JIT/B12X caches on disk.

## Watchdog

Timer armed:

```text
vllm-glm53fp4-watchdog.timer
NEXT 2026-10-17 21:39:43 EDT
LAST 2026-10-17 21:39:44 EDT
ACTIVATES vllm-glm53fp4-watchdog.service
```

First scheduled run:

```text
vllm-glm53fp4-watchdog.service  ActiveState=inactive  SubState=dead  Result=success  ExecMainStatus=0
```

It exited 0 on the healthy `/health` 200 probe and wrote no log entry; the
serving unit stayed `active` and `/health` stayed 200.

The health-down branch was exercised without restarting production by using a
separate state directory, `PORT=9`, and `WATCHDOG_MAX_HEALS=0`:

```text
health=000 but both ranks are running; waiting one tick
not restarting: 0 heals in 21600s (cap 0); r0=true r1=true health=000
```

The serving unit remained `active`, `/health` 200. The first line proves the
one-tick grace for a long prefill; the second proves the 3-per-6h cap is
enforced before any `systemctl restart`.

## Not done

- **No physical reboot was performed.** The unit is enabled, linger is on, the
  symlinks exist, and the cold-start path was executed through systemd. A real
  reboot remains a maintenance action.
- The Mac-side `18000 -> head:8000` tunnel is not a cluster unit and must be
  recreated after a laptop reboot.
- The unit hardcodes worker `nvidia@192.168.177.11`; update it if the pair's
  addresses change.

## Files

- `systemd/vllm-glm53fp4.service`
- `systemd/vllm-glm53fp4-watchdog.service`
- `systemd/vllm-glm53fp4-watchdog.timer`
- `scripts/watchdog.sh`
- `docs/09-systemd.md`
