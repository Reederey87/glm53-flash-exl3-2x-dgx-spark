# Parameters

Every value in `env.example`, and why. The short version: **four knobs form a load-bearing
system — `MAX_MODEL_LEN=1000000`, `MAX_NUM_BATCHED_TOKENS=3584`, the KV pin, and
`--no-async-scheduling`. Change one and you must re-derive the others.**

## MAX_MODEL_LEN=1000000

This deployment started at 262k because the pinned pool then counted 318,640 tokens
(1.22× one full request) and 900k meant every concurrent session evicted every other
session's cache. Two allocator changes later (the drafter/MLA page **slot-share** plus
the hybrid prefix-hit fix — see `docs/06-improvement-plan.md`), the **same pinned
bytes** count **1,396,551 tokens = 1.40× a full 1M request**, and 1M allocates cleanly.
Pool "tokens" are geometry-dependent; the bytes never move.

Cost of the 1M window, measured: ~6–9% prose decode and longer cold prefill; structured
decode unaffected. Solo-session cache retention holds ~340k tokens at 99%; two ≥68k
sessions still thrash each other to 0% (see `docs/04-prefix-caching.md`) — the window
does not repeal the one-long-context-client rule.

## MAX_NUM_BATCHED_TOKENS=3584 — do not "round" this number

3584 **is the cache page size** vLLM derives for this hybrid model ("Setting attention
block size to 3584 tokens…" in the boot log). Align-mode prefix caching checkpoints the
KDA state only when a prefill chunk ends exactly on a page boundary:

- MNBT **below** 3584 (upstream ships 1024): chunk ends drift, checkpoints almost never
  land, and a missing KDA checkpoint **vetoes all attention-layer cache hits** — the
  dashboard reads exactly 0% with nothing in the logs.
- MNBT **8192**: OOMs the GB10 indexer's shared memory on long prefill.
- **3584**: one page per chunk; verified clean on a 200k cold prefill at ~940 tok/s.

If a future image changes the page size, MNBT must move with it.

## EXTRA_ARGS: the KV pin + async off

`--kv-cache-memory-bytes 15414698763` (14.36 GiB) pins the pool: vLLM skips memory
profiling and the pool is byte-identical every boot. The value is vLLM's own suggestion
from the first unpinned boot. Rules:

- **Never raise it.** With the pin there is no fit protection; under a 4×60k concurrent
  burst the head node has **3.6 GiB MemFree**. An overshoot is a mid-flight OOM kill,
  not a clean error. If boot-time free memory ever shrinks, LOWER the pin.
- **Keep the quoted form.** Unquoted, `source` under `set -a` parses the number as a
  command and `start.sh` dies with exit 127.

`--no-async-scheduling` is mandatory, twice over: the tri-state default resolves to
ENABLED if you merely omit the flag, and under async the sliding-window-family
reservation is double-counted (vLLM #47728 class), which inflates the MNBT-3584
admission check to 17.51 GiB > the pin — the engine refuses to boot. Async-off passes
the same check and benched at/above the async reference here (structured 69.8 vs 67.4).

## GPU_MEM_UTIL=0.85

With the pin this sizes **nothing** — it is only the boot gate (`free >= total × util`),
and the gate reads CUDA device-free, which tracks `MemFree`, NOT `MemAvailable` (page
cache is invisible to it). 0.87 demands 105.87 GiB free against boots measured at
104.1–106.98 and crash-loops. Leave it.

## The rest

- `MAX_NUM_SEQS=4` — matches the cudagraph capture set (4 seqs × 8 spec tokens = 32).
- `DFLASH_TOKENS=7` — k=5 was rejected in A/B (structured −28%; the k+1 ceiling is
  arithmetic). The capture sizes are sized for k=7; they are not free memory.
- `READY_TIMEOUT=4800` / `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` — a cold JIT rebuild
  after a cache wipe is slow, not dead. Timeouts that "look generous" prevent the
  watchdog from shooting a booting engine.
- `IMAGE` names an **immutable** artifact + `SKIP_PULL=1` — since 2026-08-30 that is a local build tag produced by this repo's `Dockerfile` (previously a registry digest); either way, a restart must never silently upgrade
  the runtime. The weekly check-updates timer diffs the registry digest against the pin
  so upgrades are deliberate.

## Added 2026-08-31

- `DFLASH_REVISION=7d74cdd…` — the drafter checkpoint, pinned. The Hub repo mutates
  under a fixed name; without the pin, a re-download quietly swaps drafter weights,
  which is the same stale-kernel hazard as changing `DFLASH_TOKENS` (it is in the JIT
  shape hash for that reason). The newer checkpoints were benched and rejected.
- `DEFAULT_MAX_NEW_TOKENS=65536` — server-side ceiling for requests that omit
  `max_tokens`. Without it, one forgetful client can decode toward a million tokens
  and starve everyone. Passed as its own `--override-generation-config` argument so
  the JIT shape hash ignores it; the model's own sampling defaults (temperature 1.0)
  survive the merge — verify the boot log line says so.
- `GLM53_SPINWAIT_2MS=1` — cuts vLLM's shared-memory reader spin from 1 s to 2 ms.
  On a 20-core GB10 the default keeps two threads busy-spinning at ~90% through every
  decode; with the patch the spinner parks and the head's hot zones drop ~5 °C.
  Throughput measured unchanged against a same-day control.
- `GLM53_FINE_GRAINED_APC=1` — lets prefix-cache hits land on 64-token boundaries
  instead of full 3,584-token pages (the coordinator's veto came from a scratch
  buffer that never caches anyway — `docs/04`). This is what makes small prompts and
  follow-up turns cache; it costs ~3% structured decode, accepted deliberately.

## Added 2026-09-01

- `GLM53_DEFAULT_REASONING_EFFORT=high` — server-side default reasoning effort
  (W27). The W16 template maps *unset* to `max`, so any client sending only
  `enable_thinking` ran at the most expensive setting; this knob injects
  `--default-chat-template-kwargs '{"reasoning_effort":"high"}'` as one argv
  element. Strict enum (`low|high|max`, empty = off), validated start/restart
  only, not in the JIT shape hash. Per-request `chat_template_kwargs` win.
- `GLM53_ALIGN_FLOOR=1` — align-floor overlay (dormant at LPTT=1792). Stops the
  mamba align split from zeroing a sub-block chunk when `LPTT >= block_size`;
  required before any `LPTT >= 3584` arm. Read once at import.
- `GLM53_KV_CAPACITY_LOG=1` — honest KV-capacity boot log (W41, log-only).
  After the stock line, prints one line per KV-cache group plus the usable
  block-id count and the aligned dense-retention cached-conversation capacity —
  the stock "GPU KV cache size" number is a concurrency figure in token units
  for a multi-group hybrid, not a cache capacity. Derivation errors are logged,
  never boot-fatal. Not in the JIT shape hash.
- `GLM53_APC_NO_STORE=1` — kill switch for the per-request prefix-cache
  no-store surface (W42). A request sent with `vllm_xargs
  {"skip_writing_prefix_cache": 1}` (or the typed `SamplingParams` field) never
  inserts its blocks into the GPU prefix cache; lookups are unaffected and its
  blocks stay unhashed, recycling LIFO-first, so one-off batch lanes stop
  evicting other sessions' cached prefixes. Malformed values are an HTTP 400 at
  the API boundary. Not in the JIT shape hash.
