# GLM-5.3-Flash-EXL3 on 2× NVIDIA DGX Spark

Reproduction kit for a **production** deployment of GLM-5.3-Flash (320B MoE / 18B
active) on two NVIDIA DGX Spark (GB10 Grace Blackwell, 121 GiB unified memory each):
a **1,000,000-token context window** with DFlash2 speculative decoding, TP=2 over a
direct 200Gb QSFP link, loopback-only by default.

This is the deployment I actually run, with every gotcha written down. Since
2026-08-30 the serving image is **built by this repo's `Dockerfile`** — production
runs the local build, not a pulled artifact.

## Why this kit, in numbers

The most frustrating thing about every GLM-5.3-Flash config I ran before this one
was not decode speed — it was **prefix-cache misses and prefill latency**. A config
that passed every acceptance check would read **0% cache hits** under real agentic
traffic (each turn re-read the whole history), and with a few coding agents attached
time-to-first-token ran **80–160 s** while effective prefill collapsed from ~900 to
**~160 tok/s**. Nothing was logged; the counters just read zero.

The cause is the model, not a mis-set flag: GLM-5.3-Flash is a hybrid KDA(mamba)+MLA
architecture, cached in **3,584-token pages** whose KDA state is checkpointed only
when a scheduler step ends exactly on a page boundary — and one missing checkpoint
vetoes every attention hit. On top of that, the DFlash2 drafter's eagle-style prune
silently dropped the last page of every hit. Fixing this is most of what separates
this tree from the recipe it started from:

| Measured before | Fix in this tree | Measured after |
|---|---|---|
| Hits read 0% at upstream `MNBT=1024` (chunk ends miss the page boundary) | `MAX_NUM_BATCHED_TOKENS` = the 3,584 page size, async scheduling OFF | solo 110k replay **97–98%** |
| Every hit lost its last page (N−1 of N) | `overlay/patch_hybrid_prefix_hit.py` — prune scoped to the drafter's own group | full-N-page hits |
| Multi-session retention collapse — 2×68k sessions **0%** (163 s) | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` (sparse KDA retention, extended to the mamba groups) | **97.8%** (5.8 s) |
| Co-batch zero-insertion — 4×60k concurrent **0%** (288 s) | same knob | **95.0%** (17.4 s) |
| Short request stuck behind a 240k read — **256 s** TTFT | `LONG_PREFILL_TOKEN_THRESHOLD=1792` fairness cap | **7.9 s** |
| First turn after every restart cold | `local/content-warmup.sh` pre-reads the shared system prompt at boot | warm on turn 1 |

Mechanism, the remaining cautions, and how to verify on your own pair (a lifetime
hit-rate on a dashboard hides all of this): `docs/04-prefix-caching.md`,
`docs/08-concurrent-prefill.md`; probes `local/cache-burst.py`, `local/cache-probe.sh`,
`local/ttft-probe.py`.

The headline figures, same pair:

| | |
|---|---|
| Context window | **1,000,000 tokens**, with speculation active — on two desk machines |
| Structured decode | **~70 tok/s** at speculative acceptance **1.0000** (7/7 drafted tokens accepted, every converged pass) |
| Prose decode | **27.9 tok/s** at the 1M window |
| Cold prefill | **~893 tok/s** solo (133–240k prompts) |
| 500k prompt, drafter on | **854 tok/s** cold; same prompt replayed from cache **111× faster** (5.3 s) |
| Short request behind a 240k read | **7.9 s** to first token (256 s without this kit's fairness cap) |
| Multi-session caching | 2×68k sessions retain **97.8%**; 4×60k concurrent retain **95%** |

No other public recipe serves this model on this hardware with all four of: EXL3
(the only quantization GB10 can actually run — it lacks the instruction NVFP4
compiles to), a 1M window that *coexists* with speculative decoding, prefix caching
that survives the hybrid-KDA architecture and the drafter, and perfect structured
acceptance. Each of those is a specific fix in this tree, and removing any one of
them has a measured cost (`docs/10-selfbuild-production.md`, "load-bearing set").

Metric provenance, honestly: decode figures are medians of 3–4 converged temp-0
passes; the ~70 structured is the 2026-08-30 self-built image (the earlier pulled
image read 70.2–72.8 across boots, its 72.8 high-water after four hours warm — a
long-warm re-bench of the self-build is the open follow-up); 27.9 prose and the
caching rows were measured 08-29 on the previous image, whose cache mechanisms are
baked into this build and spot-verified by the 500k/30k replay runs (15–17× at 30k);
prefill and replay figures are single timed runs. Every bench script ships in
`tests/` and `local/` — reproduce any row in minutes.

## The serving image: preview vLLM, pinned and completed

The base is `vllm/vllm-openai:glm53-flash-arm64-cu130` — the **day-0 GLM-5.3 preview
image**, carrying a *pre-release* vLLM dev build (`0.1.dev20051+g487ecf187`) cut from
the official enablement lineage **before** it merged upstream (#53906 is still open,
and the tree predates vLLM's native DFlash2). Preview code is why this kit pins the
base **by digest** and adds every capability explicitly, verified on the real pair:

- **EXL3 kernels** — `exllamav3` built for aarch64/sm_121 at a pinned commit; keeps
  the 320B experts packed at 82 GiB/node, which is what leaves room for the 1M pool.
- **The DFlash2 drafter end to end** — model, speculator, aux-hidden-state capture;
  none of it exists in the preview tree (we booted the raw base nine times to prove
  exactly what's missing — `docs/09-rebase-draft-test.md`).
- **KV slot-share** — without it the drafter caps the servable window near 358k;
  with it, 1M + speculation coexist.
- **Correctness backports** the preview tree predates: xgrammar termination
  (#52805/#53046 — acceptance 0.98→1.0000), the #54282 draft-noise salt, a
  long-generation kernel clamp.
- **Hybrid-KDA prefix caching under speculation** — page-aligned geometry, sparse
  retention, drafter-group fixes; upstream is converging on the same (#54163).

When official support merges, `docs/07`/`docs/09` are the map forward — with the
known landmine flagged (upstream's Aug-22 refactor broke DFlash2 loading on main).

## What's in the box

| | |
|---|---|
| Weights | [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) — uniform-K4 EXL3/TR3, ~164 GiB, pinned revision |
| Runtime | **Built by `Dockerfile` here** from the digest-pinned preview base ([NOTICE](NOTICE)) |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2`, k=7, BF16 (keep it BF16 — quantized-drafter handling is broken upstream) |
| Ops | memory-gated restarts, crash/wedge/stop-aware watchdog, acceptance alerting, Xid monitoring, systemd units |

Key local hardening, all marked `# LOCAL:` in-file: **loopback bind hardcoded**
(upstream ships `0.0.0.0` on `--network host` — an open model on your LAN; verify
with `ss -ltn | grep 8000` after any update), **KV pool pinned to the byte** (never
raise it — `docs/02`), and **`MAX_NUM_BATCHED_TOKENS` = the 3,584-token page size
with async scheduling OFF** — get either wrong and cache hits silently read 0%
(`docs/04`).

## Quickstart

```bash
# on the head node
git clone <this repo> glm53 && cd glm53
cp env.example .env         # read it top to bottom — every value is a decision
docker build -t glm53-selfbuild .   # from the digest-pinned base; ship to the worker too
bash download.sh            # ~164 GiB of weights, verified against the pinned revision
local/prod-start.sh         # NOT start.sh directly — see docs/03-bringup.md
local/acceptance.sh         # 7 checks: tools, thinking, vision, long-context needle
```

Then install the units in `local/` (`systemctl --user enable ...`) so the pair
survives reboots and heals itself. Full drill: `docs/03-bringup.md`.

This release is reproduce-tested: this exact tree was rebuilt on the production head
and booted **as production**, passing acceptance 7/7, serving 6/6, a byte-identical
KV pool (1,396,551 tokens), and 1.0000/7.0 structured acceptance on first boot.

## API surface notes (this build)

**Cache reset** — `POST /reset_prefix_cache` (from `overlay/patch_cache_reset.py`)
empties a cold cache without a restart; returns `{"success": false}` while blocks
are held, retry after requests drain. Auth caveat: the bearer middleware guards only
`/v1`-style prefixes, so root-mounted routes (`/tokenize`, `/detokenize`, the cache
reset) answer without the key — set `GLM53_EXPOSE_CACHE_RESET=0` for untrusted
clients. **Tokenize** is mounted at the root (`/v1/tokenize` is 404) and validates
`prompt`/`messages`, not `text`.

## The experiments that lost (read before "optimizing")

The improvement program (`docs/06`, `docs/08`, `docs/10`) records every tested
change, including the rejections — with numbers, so you don't re-pay for them:
a community-recommended prefill config (+3.7% cold prefill, but −3.3–4% on every
decoded token and −12% pool); FlashInfer's radix top-k (zero gain at draft-batch
sizes); a bigger draft length (k=8: prose −9%); dual-rail NCCL (loses at production
geometry). If a knob isn't set the way upstream defaults it, there's a measured
reason in those three docs.

## Layout

```
env.example        the deployment's configuration, annotated (start here)
Dockerfile         builds the serving image from the digest-pinned day-0 base
start.sh stop.sh   vendored launcher (LOCAL-patched: loopback bind, single env example)
overlay/           runtime patches applied at container start
local/             production ops: prod-start, watchdog, monitors, tests, cache probes
docs/              01 architecture · 02 parameters · 03 bringup · 04 prefix caching ·
                   05 known issues · 06 improvement plan · 07 rebase plan ·
                   08 concurrent prefill · 09 rebase field test · 10 self-build cutover
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

Original work in this repo: Apache-2.0 ([LICENSE](LICENSE)). Vendored serving-kit
files: MIT, reproduced in [NOTICE](NOTICE) together with full provenance.
