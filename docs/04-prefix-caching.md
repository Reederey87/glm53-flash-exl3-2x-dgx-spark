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
2. **Hits align to 3584-token pages** — prompts under ~3.6k tokens can never hit at
   all. (DFlash2's eagle-style prune used to cost the last page — N−1 of N — until the
   `patch_hybrid_prefix_hit.py` overlay scoped the prune to the drafter's own group;
   solo replays now hit the full N pages, measured 97–98%.)
3. **`vllm:kv_cache_usage_perc` counts only RUNNING requests' blocks** (0% at idle does
   not mean a cold cache) and page-granular accounting inflates it ~2.4× vs tokens.
   Do not read it as "pool underutilised".

## What works after the fixes

Solo agentic traffic caches properly: a 110k-token session replays at 97–98% hit and
re-prefills in **3.5 seconds instead of 132**; the post-power-on sweep verified solo
replay retention at 99%+ out to **~340k real tokens** on the 1M-window pool
(1,396,551 tokens since the slot-share).

## What still doesn't (upstream bugs + one accepted regression)

1. **Any prefill that overlaps another in-flight request — even one merely decoding —
   inserts nothing into the cache.** See `05-known-issues.md` for the isolation data.
2. **Two long sessions evict each other.** Since the slot-share merge, two sessions of
   ~68k+ each thrash cross-session retention to an exact 0% (pre-merge 84–93%); only
   ~2×20k coexists. Accepted as the price of the 1M window — the suspect is cached
   pages costing ~2× in pool accounting under slot-share.

Both land on the same operating rule: keep long-prompt agent clients at **~1 in-flight
request**, and size client-side compaction so sessions stay under the solo ceiling
(we compact at 300k).

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
