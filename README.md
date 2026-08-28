# GLM-5.3-Flash-EXL3 on 2× NVIDIA DGX Spark

Reproduction kit for a **production** deployment of GLM-5.3-Flash (320B MoE / 18B active)
on two NVIDIA DGX Spark (GB10 Grace Blackwell, 121 GiB unified memory each), serving
**262k context** with DFlash2 speculative decoding at TP=2 over a direct 200Gb QSFP link.

This is the deployment I actually run, with every gotcha it cost to get here written down.
It replaces the earlier NVFP4 recipe, which is preserved unchanged in
[`legacy-nvfp4/`](legacy-nvfp4/).

## What's in the box

| | |
|---|---|
| Weights | [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) — uniform-K4 EXL3/TR3, ~164 GiB, pinned revision |
| Runtime | [MiaAI-Lab/GLM-5.3-Flash-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-DGX-Sparks) prebuilt image, **pinned by digest** |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2`, k=7 (structured accept ~0.98, ~67–70 tok/s structured) |
| Base kit | Upstream recipe vendored at `c91754f`, with local commits on top (see below) |

The upstream kit does the heavy lifting (`start.sh` owns both ranks over ssh, JIT cache
persistence, warmup). This repo adds what production needed on top — all changes are
marked `# LOCAL:` in-file:

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
  metrics alerting with argv-integrity checks, GPU Xid monitoring, acceptance and serving
  test batteries, and prefix-cache probes (`cache-burst.py`, `cache-probe.sh`).
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

`.env.example` is upstream's untouched original; **`env.example` is this deployment's
annotated configuration** and the one to start from.

## Measured (2026-08-28, warm, temp 0)

| Phase | tok/s |
|---|---|
| Structured (count 1→200) | 69.8 |
| Prose | 27.7 |
| Production path (temp 1.0, thinking on) | ~30 |
| 200k cold prefill | ~940 tok/s, no OOM |
| 100k cached re-prefill | **7 s** (vs 110 s cold) |

## Known issues

- **Co-batched prefill inserts nothing into the prefix cache** (upstream vLLM bug, present
  in the pinned image): any request whose prefill overlaps another in-flight request gets
  zero cache retention. Sequential traffic hits 84–93%; concurrent agentic bursts hit 0%.
  Mitigation and full repro data: `docs/05-known-issues.md`.
- Requests larger than ~150k tokens prefill fine but are too big for the pool to retain
  as reusable prefix.

## Layout

```
env.example        the deployment's configuration, annotated (start here)
start.sh stop.sh   upstream launcher (LOCAL-patched: loopback bind, overridable NCCL knobs)
local/             production ops: prod-start, watchdog, monitors, tests, cache probes
docs/              architecture, parameters, bringup, prefix caching, known issues
tests/             decode benches + kit regression tests
legacy-nvfp4/      the previous NVFP4 deployment kit, frozen as shipped
```

## License

Original work in this repo: Apache-2.0 ([LICENSE](LICENSE)). Files vendored from the
MiaAI-Lab kit: MIT ([LICENSE.MiaAI-Lab-kit](LICENSE.MiaAI-Lab-kit)). See [NOTICE](NOTICE).
