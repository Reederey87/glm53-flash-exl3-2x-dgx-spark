# Prefix caching on a hybrid KDA model

This page exists because a concurrent agentic workload measured **0% prefix-cache
savings and 0.44 tok/s aggregate output** on a config that passed every other check.
If you serve agents (Claude Code, openclaw, Hermes, droid, …) read this before tuning.

## The mechanism

GLM-5.3-Flash is hybrid: KDA linear-attention layers cache a **recurrent state**, not
per-token KV. vLLM sizes the whole cache in **3584-token pages** and runs
`mamba_cache_mode=align`, where a page's KDA state is checkpointed only when a
scheduler step **ends exactly on a page boundary**. The hybrid coordinator requires
every KV group to hit, so one missing KDA checkpoint **vetoes all attention hits** —
the counters read exactly 0% and nothing is logged (vLLM #42317 / #45238 class).

Three practical consequences, all measured on this cluster:

1. **MNBT must be ≥ the page size** (here: exactly 3584). At upstream's MNBT=1024,
   chunk ends land on a boundary only by arithmetic accident — near-zero hits.
2. **Hits align to 3584-token pages**, and DFlash2's eagle-style prune costs one page:
   expect N−1 of N complete pages. Prompts under ~3.6k tokens can never hit at all.
3. **`vllm:kv_cache_usage_perc` counts only RUNNING requests' blocks** (0% at idle does
   not mean a cold cache) and page-granular accounting inflates it ~2.4× vs tokens.
   Do not read it as "pool underutilised".

## What works after the fix

Sequential agentic traffic caches properly: a 100k-token session replays 26/27 pages
(93% hit) and re-prefills in **7 seconds instead of 110**. Retention verified to ~150k
per session; a 200k session prefills fine but exceeds what the 89-page pool retains.

## What still doesn't (upstream bug)

**Any prefill that overlaps another in-flight request — even one merely decoding —
inserts nothing into the cache.** See `05-known-issues.md` for the isolation data.
Until it's fixed upstream, keep long-prompt agent clients at ~1 in-flight request and
size client-side compaction so sessions stay ≤ ~150k (we compact at 110k).

## Verifying, not assuming

The lifetime hit-rate on a dashboard hides all of this (it averages over benches and
retries). Use the probes:

```bash
# per-round hit% under synthetic multi-session agentic load (rounds >= 2!)
uv run python local/cache-burst.py --sessions 4 --ctx-tokens 60000 --rounds 3

# windowed delta hit-rate against live traffic
local/cache-probe.sh 30
```

Round 1 is cold by construction; rounds 2+ are the verdict. A healthy config shows
>90% on round 2 sequential and the same sessions re-prefilling in seconds.
