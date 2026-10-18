# Systemd boot unit and watchdog

The recipe is served by a **user** systemd unit on the head node. It starts the
same `launch.sh` path used manually, so a reboot and a manual start cannot
diverge.

## Units

The files live in [`systemd/`](../systemd/):

| File | Role |
|---|---|
| `vllm-glm53fp4.service` | Cold-start the head + worker through `launch.sh`; wait for `/health`; stop both ranks. |
| `vllm-glm53fp4-watchdog.service` | Run the health canary script once. |
| `vllm-glm53fp4-watchdog.timer` | Run the canary 15 min after boot, then every minute. |

Install on the head as the `nvidia` user:

```bash
cd /home/nvidia/glm53-fp4-local
cp systemd/vllm-glm53fp4.service \
   systemd/vllm-glm53fp4-watchdog.service \
   systemd/vllm-glm53fp4-watchdog.timer \
   ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable vllm-glm53fp4.service
systemctl --user enable --now vllm-glm53fp4-watchdog.timer
```

`loginctl` linger must be on so the user manager starts at boot:

```bash
loginctl show-user nvidia -p Linger     # Linger=yes
```

The previous production unit must be disabled **before** the first reboot.
Both units bind port 8000 and need the unified memory:

```bash
systemctl --user disable --now vllm-dsv41-two-sparks.service
systemctl --user disable --now vllm-dsv41-two-sparks-watchdog.timer
systemctl --user disable --now vllm-glm53exl3.service 2>/dev/null || true
```

## Boot path

1. The user manager starts the enabled unit.
2. `ExecStartPre` waits for the local Docker daemon (300 s), the worker's Docker
   over SSH (600 s), and **MemAvailable >= 100 GiB** on the head (600 s).
3. `ExecStart=launch.sh` runs with `PULL_IMAGE=0`: it checks the pinned upstream
   checkout, the local `vllm-node-b12x` image, the 42-file checkpoint, then
   dispatches both ranks in daemon mode.
4. `ExecStartPost` polls `/health` for up to 900 s. The unit stays `activating`
   until the API is ready.
5. If the start fails, `ExecStopPost` runs `launch.sh down` so a half-started
   pair does not hold the port or the pool, then systemd retries after 300 s
   (three attempts per hour).

`PULL_IMAGE=0` is deliberate: the image and checkpoint are staged from the
bring-up. If either is missing, run `./launch.sh setup` manually first, or
temporarily set `PULL_IMAGE=1` for that start.

## Watchdog

`scripts/watchdog.sh` runs as the one-shot watchdog service:

- exits if the unit is disabled, inactive, activating, or a manual stop left it
  inactive;
- passes immediately on `/health` 200 and clears the pending marker;
- if both ranks are running but health is down, waits one timer tick before
  acting (a long prefill can miss one probe);
- restarts the unit through systemd, which runs the memory-aware
  `ExecStartPre` and the health gate;
- caps restarts at **3 in 6 hours** so a broken checkpoint cannot thrash GB10
  memory.

Logs:

```bash
tail -f /home/nvidia/glm53-fp4-local/local/watchdog/watchdog.log
journalctl --user -u vllm-glm53fp4.service -n 100 --no-pager
```

## Verify without rebooting

```bash
systemctl --user is-enabled vllm-glm53fp4.service
systemctl --user is-enabled vllm-glm53fp4-watchdog.timer
systemctl --user list-timers vllm-glm53fp4-watchdog.timer
systemctl --user status vllm-glm53fp4.service --no-pager
curl -fsS http://127.0.0.1:8000/health
```

To test the cold path now, stop the ad-hoc pair and start the unit:

```bash
cd /home/nvidia/glm53-fp4-local
./launch.sh down
systemctl --user start vllm-glm53fp4.service     # blocks until /health
systemctl --user restart vllm-glm53fp4-watchdog.service
```

A real reboot can then be scheduled; the unit is already enabled and the
watchdog is armed.

## Manual stop is respected

`systemctl --user stop vllm-glm53fp4.service` leaves the unit inactive, and the
watchdog exits without restarting it. Start it again to re-arm:

```bash
systemctl --user start vllm-glm53fp4.service
```

## What is not automated

- The Mac-side SSH tunnel (for example `18000 -> head:8000`) is not a cluster
  unit. Recreate it after a laptop reboot.
- The unit hardcodes the worker SSH target `nvidia@192.168.177.11`; update it
  if the pair's addresses change.
- A true reboot test is a maintenance action; the unit's boot path is wired
  with linger, the enable symlinks, and the cold start above.
