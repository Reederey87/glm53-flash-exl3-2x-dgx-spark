# GLM-5.3-Flash-EXL3 on 2× NVIDIA DGX Spark

Reproduction kit for a **production** deployment of GLM-5.3-Flash (320B MoE / 18B active)
on two NVIDIA DGX Spark (GB10 Grace Blackwell, 121 GiB unified memory each), serving
**1M context** with DFlash2 speculative decoding at TP=2 over a direct 200Gb QSFP link.

This is the deployment I actually run, with every gotcha it cost to get here written down.

## Why this kit

- **Frontier-class model on desk hardware.** A 320B-parameter MoE with a **1,000,000-token
  context window**, served from two consumer-purchasable Sparks — no rack, no cloud bill,
  loopback-only by default.
- **Measured, not claimed.** Every number in this README comes off the running cluster:
  **70.4 tok/s structured / 29.5 tok/s prose** decode, **~893 tok/s** cold prefill,
  **1.0 speculative acceptance** on structured output (7 drafted tokens per step, zero
  wasted verifies), a 110k-token session re-prefilling in **~4 s** instead of 132 via
  prefix caching — and **multi-session caching that works**: two 68k sessions retain
  97.8%, four concurrent 60k sessions retain 95%. The bench scripts ship in `tests/`
  and `local/` — reproduce them in minutes.
- **Hardware-honest quantization.** GB10 lacks the `cvt.e2m1x2` instruction — NVFP4
  can never compile for this chip. EXL3's trellis kernels are Blackwell-native and keep
  the 320B experts packed at 82 GiB/node, which is what leaves room for the 1M-token KV
  pool. The full rationale: `docs/01-architecture.md`.
- **Reproducible by construction — and reproduce-TESTED.** Runtime image pinned **by
  digest**, weights pinned by revision, KV pool pinned to the byte (identical every
  boot — no profiling variance), a config-shape hash that wipes stale JIT caches before
  they poison your numbers. v1.0.2 was verified by booting the production cluster from
  a fresh clone of the tag: acceptance 7/7, byte-identical pool, loopback bind held.
- **Quality-gated like production, because it is production.** A 7-probe acceptance suite
  (tool calls, thinking, vision, 36k needle), a 6-probe serving suite (SSE, 4-way
  concurrency, sustained load), a 23-turn tool-call battery under concurrent cold-prefill
  load — all passing on the shipped config, all runnable from this repo.
- **Self-healing ops included.** Memory-gated restarts, a watchdog that tells crash from
  wedge from deliberate stop, spec-decode acceptance alerting, GPU Xid monitoring, weekly
  update/parity checks with a driver-branch hold. The failure modes are documented
  *because they happened here first* — `docs/` is the postmortem you don't have to write.

## What's in the box

| | |
|---|---|
| Weights | [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) — uniform-K4 EXL3/TR3, ~164 GiB, pinned revision |
| Runtime | Prebuilt serving image, **pinned by digest** in `env.example` (provenance and licenses: [NOTICE](NOTICE)) |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2`, k=7 (structured accept 1.0 post-xgrammar-fix, ~70 tok/s structured) |
| Base kit | Upstream serving recipe vendored at `0e2e78f`, with local commits on top (credited in [NOTICE](NOTICE)) |

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

## API surface notes (this build)

**Cache reset.** `overlay/patch_cache_reset.py` mounts the upstream dev cache
router on the head API server, so a genuinely cold prefix cache no longer
needs a container restart:

```bash
curl -s -X POST http://127.0.0.1:8000/reset_prefix_cache    # -> {"success": true}
```

It returns `{"success": bool}` and reports `false` while blocks are still
held (running requests, in-flight async KV offload) — retry after they
drain. `GLM53_EXPOSE_CACHE_RESET=0 ./start.sh` restores the stock surface
(restart is then the only reset); `VLLM_SERVER_DEV_MODE=1` still mounts the
whole dev set (`/sleep`, `/rlhf`, `/rpc`, `/server_info`) if ever needed.
Auth caveat: the bearer middleware only guards `/v1`, `/v2`, `/inference`,
`/cohere` (upstream `GUARDED_PREFIX`), so root-mounted routes — the stock
`/tokenize` / `/detokenize` and the cache-reset routes — answer without the
key even with `VLLM_API_KEY` set. Set `GLM53_EXPOSE_CACHE_RESET=0` on kits
that serve untrusted clients.

**Tokenize.** It is mounted at the **root** (`/v1/tokenize` is 404) and the
request validates `prompt` (or `messages` for the chat shape), not `text`.

## Measured (2026-08-29, warm, temp 0, median of 3)

| Phase | Value |
|---|---|
| Structured decode (count 1→200) | 70.4 tok/s (acceptance 1.0000, 7.0/step, all positions) |
| Prose decode | 29.5 tok/s |
| Production path (temp 1.0, thinking on) | ~30 tok/s |
| Cold prefill (solo, ~133–240k) | **~893 tok/s** (941 with `LONG_PREFILL_TOKEN_THRESHOLD` unset — the −5% is the price of HOL relief) |
| Short-request TTFT behind a 240k cold prefill | **7.9 s** (256 s without the threshold) |
| 110k cached re-prefill | **~4 s** (vs 132 s cold; 98% hit) |
| 2×68k sessions, cross-session retention | **97.8%** (5.8 s re-prefill) |
| 4×60k concurrent sessions ×3 rounds | **95.0%** (271k tokens re-prefilled in 17.4 s) |
| Bench-convergence caveat | first passes after a restart read low (prose −17%) from parked-swap fault-in — run 3–4 passes |

The 2026-08-29 numbers include the xgrammar termination backports (structured
acceptance 0.98 → 1.0 by eliminating spurious tail-draft rejections), the sparse-KDA
retention fix (the multi-session rows above were an exact 0% before it), and the
long-prefill chunk cap — `docs/06-improvement-plan.md` documents the full program,
including the experiments that measured *worse* and were reverted.

## Rebase track

Upstream vLLM is actively landing official GLM-5.3-Flash support (#53906, #53969,
`FLASHINFER_MLA_SPARSE_SM120` auto-selection, native DFlash2, sparse mamba
retention). `docs/07-rebase-plan.md` is the researched plan for moving this kit
onto that base and shrinking the fork delta to an EXL3 plugin + thin GLM ports.

## Layout

```
env.example        the deployment's configuration, annotated (start here)
start.sh stop.sh   vendored launcher (LOCAL-patched: loopback bind, single env example)
overlay/           runtime patches applied to the pinned image at container start
local/             production ops: prod-start, watchdog, monitors, tests, cache probes
docs/              architecture, parameters, bringup, prefix caching, known issues,
                   improvement plan, rebase plan
tests/             decode benches + kit regression tests
```

## Credits

The idea of serving GLM-5.3-Flash with **EXL3 weights on GB10** — and the serving
recipe this kit builds on — comes from
[Mia's AI Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks),
who also host the
[byte-identical weights mirror](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)
this recipe stays fetchable from. The EXL3/TR3 quantization is by
[brandonmusic](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
(format and kernels by [turboderp's exllamav3](https://github.com/turboderp-org/exllamav3));
the DFlash2 speculative-decode drafter is
[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
(CC BY-NC-ND 4.0, fetched separately — not redistributed here);
the base model is [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).

## License

Original work in this repo: Apache-2.0 ([LICENSE](LICENSE)). Vendored serving-kit files:
MIT, reproduced in [NOTICE](NOTICE) together with full provenance.
