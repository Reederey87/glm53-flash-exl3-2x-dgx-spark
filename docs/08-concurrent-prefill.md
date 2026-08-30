# Concurrent agents and slow prefill — the problem, in plain language

*Written 2026-08-30, after a measured A/B window on the production pair. All numbers
below come off the running cluster.*

## The problem

When several coding agents talk to the server at the same time, new requests wait a
very long time before their first token — often 80 to 160 seconds. Live telemetry from
a real 4-agent working session showed a mean time-to-first-token of **39 seconds**
(11.6 s of queueing plus 27.6 s of prompt reading), and the effective prompt-reading
speed collapsed from ~900 tokens/second (server free) to **~160 tokens/second**
(agents busy).

## Why it happens

The server has a protection rule that was written for a different situation. The rule
(`GLM53_MIXED_PREFILL_CHUNK=skip`, in `overlay/patch_scheduler_decode_floor.py`) says:

> *If anyone is currently receiving an answer (decoding), don't spend this engine step
> reading new prompts (prefill) — wait.*

That rule keeps answers streaming smoothly when one person uses the server. But with
four agents working at once, someone is **always** receiving an answer — so prompt
reading only happens in the small gaps between answers. A context that takes ~30
seconds to read on a free server takes over two minutes on a busy one. That is the
entire mechanism; the cache, the network, and the GPU are all fine.

## What we tested (and measured)

The probe: one request decoding a long answer, plus three fresh ~27k-token prompts
arriving five seconds later, prefix cache reset first. Three configurations:

| Configuration | Prompt-read time per lane | Aggregate read speed | Decoder speed during |
|---|---|---|---|
| `skip` (production today) | 136–164 s | 486 tok/s | 18.3 tok/s |
| Mixed cap `896` | **88 s (−46%)** | **907 tok/s** | 9.1 tok/s (−50%) |
| Mixed cap `448` | 93 s | 856 tok/s | 8.7 tok/s |
| `MAX_NUM_BATCHED_TOKENS=7168` (skip kept) | 157 s | 509 tok/s (+5%) | 19.1 tok/s |

Three conclusions:

1. **Allowing mixed reading is a trade, not a free win.** Waiting times nearly halve,
   but whoever is receiving an answer slows to half speed while the reading happens.
   In this probe the total wall time came out the same either way (165–172 s — the
   decoder was the last to finish in both runs), but that equality is
   workload-specific: per-request end-to-end time depends on how much reading vs
   answering each request does (see the worked example below, where mixing wins by
   ~14%).
2. **The trade cannot be tuned away with a smaller bite size.** Reading *any* amount
   of new text in a step forces the server to stream all 288 expert weight matrices
   through memory (~63% of the step cost, nearly flat in chunk size — profiled in the
   vendored upstream kit's prefill study,
   [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks `docs/improve-prefill.md` §P0](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/b5ab8091/docs/improve-prefill.md)).
   A 448-token bite taxes the
   decoder just as much as an 896-token bite and reads slower. If you mix at all,
   use the biggest sensible cap.
3. **A bigger step budget does not help while the skip rule stands.** Doubling
   `MAX_NUM_BATCHED_TOKENS` to 7168 (two cache pages) let four prompts share each
   reading step instead of two — but each step took proportionally longer, so the
   aggregate speed moved only +5%, and the starvation rule was untouched. It also
   read ~2% slower solo (266k-token stress read: 856–872 tok/s, no crash — so 7168
   is *safe*, just not useful). Reverted; 3584 stands.

## What is implemented

**Nothing changed in production.** The skip rule stays, because the pre-registered
gate said the decoder should not lose more than 15% and it lost 50%. The mitigation
is documented and one line away for operators who prefer faster response starts over
smooth streaming during busy moments:

```bash
# in .env, then restart (scheduler-only: no JIT cache wipe, instantly reversible)
GLM53_MIXED_PREFILL_CHUNK=896
```

For an interactive multi-agent workload the end-to-end turn math slightly favors
turning this on: a turn that pays a 26k-token read plus a 500-token answer completes
in ~143 s mixed vs ~167 s under skip. For a workload that mostly streams long answers,
leave it off.

## What actually reduces the pain (already shipped or queued)

- **Prefix-cache retention** (`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`, shipped): most
  of an agent's prompt is unchanged between turns, and 85–97% of it now hits the cache
  instead of being re-read. The collapse above only applies to the *missed* portion.
- **Content warmup** (shipped): the agents' shared system prompt is pre-read at boot,
  so even the first turn after a restart hits the cache.
- **Long-prefill chunking** (`LONG_PREFILL_TOKEN_THRESHOLD=1792`, shipped): a short
  request no longer waits behind a 240k read (7.9 s instead of 256 s).
- **Client habits** (free): stagger agent launches by ~15 s so their cold reads don't
  collide; keep agent system prompts byte-identical so they share cache pages; avoid
  many tiny (<3,584-token) requests — 3,584 tokens is this stack's prefix-cache page
  (block) size for the hybrid KDA model (which is also why `MAX_NUM_BATCHED_TOKENS`
  is pinned to the same value — see `docs/04-prefix-caching.md`), so shorter prompts
  can never produce a cache hit.
- **Upstream watch**: vLLM #52789 (mid-forward mamba checkpoints, 9–25% faster
  prompt reading) and the kit's own P7 idea (co-scheduling a second prefill into a
  capped step) are the structural fixes; both arrive via a future image rebase, not
  an overlay.
