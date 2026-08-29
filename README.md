# GLM-5.3-Flash-EXL3 on 2× NVIDIA DGX Spark

Reproduction kit for a **production** deployment of GLM-5.3-Flash (320B MoE / 18B active)
on two NVIDIA DGX Spark (GB10 Grace Blackwell, 121 GiB unified memory each), serving
**1M context** with DFlash2 speculative decoding at TP=2 over a direct 200Gb QSFP link.

This is the deployment I actually run, with every gotcha it cost to get here written down.

## What's in the box

| | |
|---|---|
| Weights | [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) — uniform-K4 EXL3/TR3, ~164 GiB, pinned revision |
| Runtime | Prebuilt serving image, **pinned by digest** in `env.example` (provenance and licenses: [NOTICE](NOTICE)) |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2`, k=7 (structured accept 1.0 post-xgrammar-fix, ~70 tok/s structured) |
| Base kit | Upstream serving recipe vendored at `32db610`, with local commits on top (credited in [NOTICE](NOTICE)) |

The vendored kit does the heavy lifting (`start.sh` owns both ranks over ssh, JIT cache
persistence, warmup, xgrammar termination backports). This repo adds what production
needed on top — all changes are marked `# LOCAL:` in-file:

- **`--host 127.0.0.1` hardcoded** — upstream binds `0.0.0.0` with `--network host`,
  which is an unauthenticated model on your LAN. Verify after any update:
  `ss -ltn | grep 8000` must show loopback only.
- **KV pool pinned** (`--kv-cache-memory-bytes`) — byte-identical pool every boot,
  no profiling variance. Never raise it; see `docs/02-parameters.md`.
- **Prefix-cache geometry for a hybrid KDA model** — `MAX_NUM_BATCHED_TOKENS` must equal
  the 3584-token page size and async scheduling must be OFF, or cache hits silently read
  0%. The full mechanism: `docs/04-prefix-caching.md`. This one took a day to find.
- **`local/` ops kit** — `prod-start.sh` (memory-gated restart with a config-shape hash
  that wipes stale JIT caches), a watchdog that distinguishes crash/wedge/deliberate-stop,
  metrics alerting with argv-integrity checks, GPU Xid monitoring, a driver-branch hold
  (590.x deadlocks CUDAGraph capture on GB10), acceptance and serving test batteries,
  and prefix-cache probes (`cache-burst.py`, `cache-probe.sh`, `toolcall-probe.py`).
- **systemd units** (user-level, linger on) — one oneshot unit owns the pair; timers for
  watchdog, weekly update/parity checks, and metrics alerts.

## Quickstart

```bash
# on the head node
git clone <this repo> glm53 && cd glm53
cp env.example .env         # read it top to bottom — every value is a decision
bash download.sh            # ~164 GiB of weights, verified against the pinned revision
local/prod-start.sh         # NOT start.sh directly — see docs/03-bringup.md
local/acceptance.sh         # 7 checks: tools, thinking, vision, long-context needle
```

Then install the units in `local/` (`systemctl --user enable ...`) so the pair survives
reboots and heals itself. `docs/03-bringup.md` has the full drill, including why
`ExecStart` must be restart-shaped and why the watchdog never blocks.

## Measured (2026-08-29, warm, temp 0, median of 3)

| Phase | tok/s |
|---|---|
| Structured (count 1→200) | 70.4 (acceptance 1.0, 7.0/step) |
| Prose | 29.5 |
| Production path (temp 1.0, thinking on) | ~30 |
| 200k cold prefill | ~940 tok/s, no OOM |
| 110k cached re-prefill | **3.5 s** (vs 132 s cold, 97–98% hit) |

The 2026-08-29 numbers include the xgrammar termination backports, which raised
structured acceptance from 0.98 to 1.0 by eliminating spurious tail-draft rejections —
`docs/06-improvement-plan.md` documents that change and the rest of the improvement
program.

## Known issues

- **Co-batched prefill inserts nothing into the prefix cache** (upstream vLLM bug, present
  in the pinned image): any request whose prefill overlaps another in-flight request gets
  zero cache retention. Mitigation and full repro data: `docs/05-known-issues.md`.
- **Two long sessions evict each other** (accepted regression of the 1M window): sessions
  ≥ ~68k each thrash cross-session retention to 0%; solo sessions retain 99%+ to ~340k.
  Keep one long-context client at a time; compact around 300k.

## Layout

```
env.example        the deployment's configuration, annotated (start here)
start.sh stop.sh   vendored launcher (LOCAL-patched: loopback bind, single env example)
overlay/           runtime patches applied to the pinned image at container start
local/             production ops: prod-start, watchdog, monitors, tests, cache probes
docs/              architecture, parameters, bringup, prefix caching, known issues,
                   improvement plan
tests/             decode benches + kit regression tests
```

## License

Original work in this repo: Apache-2.0 ([LICENSE](LICENSE)). Vendored serving-kit files:
MIT, reproduced in [NOTICE](NOTICE) together with full provenance.
