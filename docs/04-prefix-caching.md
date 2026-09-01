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

## The retention fix (2026-08-29) — both remaining failure modes resolved

Two failure modes survived the hit-boundary fixes, and one env var resolved both:
**`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`** (in `env.example`, forwarded by
`start.sh`). This fork extends the knob to MambaSpec groups: instead of densely
keeping a KDA state per 3584-token page, retention keeps **only replay boundaries
and shared-prefix junctions** (Marconi-style), which are always retained — so exact
prefix replays (the append-only agentic shape) still hit, while cached prefixes stop
being too expensive to coexist in the pool. Measured, same-day controls → arm:

| Shape | Dense (unset) | Sparse (`0`) |
|---|---|---|
| 2×68k sessions, round-2 hits | **0.0%** (163 s) | **97.8%** (5.8 s) |
| 4×60k concurrent ×3 rounds (the co-batch shape) | **0%** all rounds (288 s) | **95.0%** (17.4 s) |
| Solo 110k replay | 98.0% | 98.0% (held) |

The co-batch "inserts nothing" bug turned out to be a **side effect of dense KDA
retention**, not a scheduler defect — a code audit had already cleared the chunking
path (`_mamba_block_aligned_split` floors every intermediate chunk to the 3584 grid).
Decode was unaffected (structured acceptance 1.0000, throughput within noise), the
pinned pool byte-identical, and the env is not in the JIT shape hash (no cache wipe).

## Per-group retention (2026-09-01) — the dense collapse was the drafter's

Re-read the table above with upstream PR #83's finding in hand: block ids are one
global LRU pool, and a cached 3,584-token segment costs **38 ids, 33 of them DFlash2
drafter sliding-window blocks** — hashed and freed to the LRU tail for a group whose
hit length the hybrid `min()` discards anyway. "Dense = 0%" was the drafter starving
MLA/mamba of ids, not dense KDA retention being too expensive in itself.
`overlay/patch_apc_per_group_retention.py` makes retention per-group: the drafter
takes `VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA=0` (boundaries only), every other
group keeps the global value — and production now runs the global knob **dense**
(commented out in `.env`). Boot line: `retention_by_group=[None,None,None,None,None,None,0]`.

| Shape | global `0` (W3) | **dense + drafter `0` (W25)** |
|---|---|---|
| 2×68k / 4×60k round-2 hits | 100% / 98.7% | **100% (1.2 s) / 98.7%** |
| Solo 110k replay · 4×120k | 98% · — | **98.0% · 100%** |
| **Shared prefix, different length** (subagent fork) | **0%** | **37–49%** |
| 4×200k concurrent replay (86% of pool) | **75.0%** | 49.9% |

Dense retention now costs nothing on every standing shape and buys the one shape
boundaries-only retention can never hit — a subagent forking from a parent's context
(shared prefix, shorter or longer tail). Its measured cost, from a same-day control,
is ~25 pp only when four sessions concurrently replay ~200k each at the pool's edge.
Decode, acceptance and the pinned pool are unchanged. Rollback is one line:
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` back in `.env` (the overlay then resolves
every group to 0 — byte-for-byte the previous behavior).

Residual cautions: soak the concurrent-hit path under real load (vLLM issue #54199
reports a GB10 illegal-memory-access when a cache-hit request is admitted while the
donor is in flight — zero errors in our probes, but that is the crash class that
lives behind concurrent hits), and re-validate at the next image rebase (upstream
#52216 changes the retention default and promotes the env to a real argument).
Client-side compaction around **300k** (the measured solo ceiling) still stands.

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

## 2026-08-31 addendum: the sub-page floor has a candidate fix (W18)

The "hits align to 3584-token pages" rule above is not a property of the model —
it is the coordinator's *partial-hash veto*: fine-grained hits are disabled when
any KV-cache manager lacks `supports_fine_grained_hash_lookup`, and on this hybrid
the only such manager is `KpoolTailManager`, the 4-token indexer tail scratch that
never prefix-caches at all (its lookup returns 0). The boot log says so verbatim:
`Disabling fine-grained prefix-cache hits because these KV cache managers require
block-aligned lookups: KpoolTailManager.` Upstream kit PR #59 exempts that manager
from the veto so MLA + mamba(align) hits reconcile at the hash grain (64 tokens).
Ported here as `overlay/patch_fine_grained_apc.py` behind `GLM53_FINE_GRAINED_APC=1`.
**ADOPTED in production 2026-08-31 (W18):** follow-ups reuse at grain 64 — a 6.5k
prompt goes 3,584 → 6,464 reused tokens (96.2%), a 2.6k prompt 0 → 2,816; follow-up
wall ~4.1 s → ~1.0 s; 2×68k retention 100%, 4×60k concurrent 98.7%, zero IMA in the
soak. Cost: −2.7% structured decode (66.5–66.7 vs 68.4–68.7), accepted for agentic
traffic. Details + gates: `06-improvement-plan.md` (W18).
