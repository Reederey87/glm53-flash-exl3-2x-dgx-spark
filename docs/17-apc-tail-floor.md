# Task 45 site 1 — the prefix-cache tail was registered one hash unit too high

**Verdict: ADOPTED 2026-09-19.** Overlay `overlay/patch_apc_tail_boundary.py`,
knob `GLM53_APC_TAIL_FLOOR=1`. Boundary arithmetic only: no kernel, no numerics,
no extra CUDA graphs, no JIT-shape change, no image rebuild.

## The defect, measured before anything was changed

`KVCacheManager.get_computed_blocks` sets

```python
max_cache_hit_length = request.num_tokens - 1
```

because the last token has to be recomputed to obtain logits. So for a request of
`n` prompt tokens the deepest hash unit the finder can hand back is
`floor((n - 1) / unit) * unit`, with `unit = hash_block_size` (64 here, because
fine-grained APC is on).

Three sites that register reusable state floored from `n` instead:

| site | file | line |
|---|---|---|
| the mamba partial-tail prefill stop | `v1/core/sched/scheduler.py` | `_mamba_block_aligned_split` |
| `FullAttentionManager._cache_partial_tail_block` | `v1/core/single_type_kv_cache_manager.py` | `boundary_tokens` |
| `MambaManager._cache_partial_tail_block` | `v1/core/single_type_kv_cache_manager.py` | `latest_prompt_hash_boundary` |

Whenever a prompt length lands on the 64-token hash grid, `n // 64 * 64 == n`,
which is one unit **above** every position a lookup can request. The registered
state is unreachable and the hybrid `min()` falls back to the previous 3,584-token
page. When the length is not on the grid the two expressions agree, which is why
this hid for so long.

Measured on the live cluster before the change
(`scripts/probe_apc_boundary_reachability.py`):

| case | prompt | reused | ceiling | short | warm |
|---|---:|---:|---:|---:|---:|
| `exact@7168` | 7,168 | 3,584 | 7,104 | **3,520** | 2.668 s |
| `exact@10752` | 10,752 | 7,168 | 10,688 | **3,520** | 2.673 s |
| `exact@14336` | 14,336 | 10,752 | 14,272 | **3,520** | 2.679 s |
| `exact@6464` | 6,464 | 3,584 | 6,400 | **2,816** | 2.268 s |
| `exact@7360` | 7,360 | 7,168 | 7,296 | 128 | 0.570 s |
| `exact@6500` | 6,500 | 6,464 | 6,464 | 0 | 0.217 s |
| `exact@10000` | 10,000 | 9,984 | 9,984 | 0 | 0.197 s |
| `exact@20000` | 20,000 | 19,968 | 19,968 | 0 | 0.262 s |

Every length that is **not** a multiple of 64 already reached its ceiling; every
one that is fell a whole page short, at ~2.67 s instead of ~0.25 s. A prompt that
is an exact multiple of the page size is worse: `_cache_partial_tail_block`
refuses a block-aligned tail outright (`num_tokens % block_size == 0`), so no
partial entry was registered at all and the guard
`last_cache_position < tail_boundary < num_prompt_tokens` failed too, because on a
page-aligned prompt `last_cache_position` backs off a whole page under EAGLE while
the stock `tail_boundary` equals `num_prompt_tokens`.

## The change

Floor all three registrations from `(n - 1)`:

```python
# scheduler._mamba_block_aligned_split
tail_boundary = (
    (request.num_prompt_tokens - 1) // self.hash_block_size
    * self.hash_block_size
    if self.mamba_partial_cache_hit
    else 0
)
```

```python
# FullAttentionManager._cache_partial_tail_block
boundary_tokens = (
    (request.num_prompt_tokens - 1) // hash_block_size * hash_block_size
)
```

```python
# MambaManager._cache_partial_tail_block
latest_prompt_hash_boundary = (
    (request.num_prompt_tokens - 1) // hash_block_size
) * hash_block_size
```

The scheduler stop is **additional**, not a replacement: the existing
block-aligned stops and the shared-prefix junction stop are untouched, so no
intermediate chunk can end off-grid. Replacing them is the state-poisoning
failure mode (upstream #43559; 14 tests fail that way in the sibling lineage).

## Why it is safe

The published entry is keyed by the prefix `[0, q)` and holds the state after
exactly `q` tokens, for `q = floor((n - 1) / unit) * unit`. A consumer resuming at
`q` gets a state that matches the position its own key proves. The change moves a
registration to a reachable position whose state is already materialised there; it
does not invent reuse, relax a key, or weaken a proof. For every prompt length that
already worked the emitted value is **identical**, so the only lengths whose
behaviour changes are the ones that were broken.

### Difference from upstream #52244's formula — deliberate

Upstream's `mamba_state_cache_position(num_tokens, block_size, hash_block_size)`
returns `(num_tokens - 1 - unit) // unit * unit` with `unit = min(hash_block_size,
block_size)` — one further unit down. That extra back-off compensates for the
EAGLE last-block drop applied to the **full-attention** group.

It is not needed here, and applying it would cost a hash unit for nothing: this
kit's `overlay/patch_hybrid_prefix_hit.py` scopes the EAGLE drop to the drafter's
`SlidingWindowSpec` group only (boot line `eagle_group_ids=[6]`, MLA and Mamba
`use_eagle=False`), so the reachable ceiling really is `(n - 1) // unit * unit`.
The probe measures that directly: `reused == ceiling` in every fixed case.

## Receipts

All on the armed boot, image `glm53-selfbuild:e3-w3-zfill-v149`, pool
`1,396,551 tokens / 1.40×` unchanged, `retention_by_group=[None,None,None,None,None,None,0]`
unchanged.

| gate | result | receipt |
|---|---|---|
| boundary reachability | every exact replay reaches its ceiling; 2.67 s → 0.265 s | `local/task45-apc-tail-boundary-armed-20260919.json` |
| correctness (26k needle, hash-grid prompt) | exact replay hits **26,176 = the exact ceiling**, right code, 2.041 s vs 19.76 s cold; no stale leak | `local/task45-apc-tail-correctness-20260919.json` |
| acceptance | 7/7, incl. the ~32k needle | `local/task45-gates-20260919.txt` |
| serving | 6/6 | `local/task45-gates-20260919.txt` |
| structured decode | median **71.20 tok/s** @ 1.0/7.0, no NaN (standing band 69–70) | `local/task45-armed-structured-20260919.json` |
| prose decode | median **32.31 tok/s** over 5 runs, accept 0.542 (standing band 29–32) | `local/task45-armed-prose5-20260919.json` |
| MemFree | head 5 GiB, worker 3 GiB (floor 2.5 GiB) | `local/task45-gates-20260919.txt` |
| boot | `Result=success`, `NRestarts=0`, 8 m 55 s, health 200 | — |

Post-change ladder (`scripts/probe_apc_boundary_reachability.py`), same shapes:

| case | reused | ceiling | ratio | warm |
|---|---:|---:|---:|---:|
| `exact@6464` | 6,400 | 6,400 | 1.0000 | 0.260 s |
| `exact@7168` | 7,104 | 7,104 | 1.0000 | 0.264 s |
| `exact@7360` | 7,296 | 7,296 | 1.0000 | 0.264 s |
| `exact@10752` | 10,688 | 10,688 | 1.0000 | 0.265 s |
| `exact@14336` | 14,272 | 14,272 | 1.0000 | 0.272 s |
| `exact@6500` / `@10000` / `@20000` | unchanged | | 1.0000 | unchanged |

## The measured cost, stated plainly

On the **append** shape (a longer consumer, `producer + 32` tokens) at a
hash-grid prompt length the deepest entry is now `n - 64` instead of `n`, so the
consumer recomputes one extra hash unit:

| case | before | after |
|---|---:|---:|
| `append32@6464` | 6,464 / 6,464 | 6,400 / 6,464 (−64) |
| `append32@7360` | 7,360 / 7,360 | 7,296 / 7,360 (−64) |
| `append32@6500` | 6,464 / 6,528 | 6,464 / 6,528 (unchanged) |
| `append32@7168`, `@10000`, `@10752`, `@14336`, `@20000` | 1.0000 | 1.0000 |

One hash unit is 64 tokens, about 20 ms of prefill, against 2,816–3,520 tokens
(~2.4 s) recovered per exact replay. This is the same arithmetic upstream #53802
adopts, and it is accepted deliberately. **Not** taken: registering both `n` and
`n - 64` would remove the 64-token append cost but adds a cache entry per request
on a pool that is the binding capacity constraint (14 aligned segments, 566 usable
block ids).

## Not covered — task 45 remains partially open

- **Successor-aware hashing** (the TODO's site (a), upstream #50897 / RFC #50438)
  is not implemented. It removes the conservative last-block drop by proving the
  successor token instead of assuming it. Its impact here is bounded because this
  kit already scopes the EAGLE drop to the drafter SWA group; it stays open as its
  own task.
- **The EAGLE registration rounding site** in the TODO (`num_finalized_computed_tokens`)
  **does not exist in this lineage.** A tree-wide search of the deployed
  `vllm/v1/` finds no occurrence, and after this change no
  `num_prompt_tokens //` floor remains anywhere under `v1/`. Recorded as
  `NOT_APPLICABLE` for this fork, not as an unported fix.
- The `hashmap` prose lane remains the noisy lane: a 3-run sample medians 26.6
  with one 18.6 stall, a 5-run sample medians 32.3. Decode is not touched by this
  change by construction — `_mamba_block_aligned_split` returns immediately once
  `start >= prefill_end`, which is every decode step.

## Rollback

`GLM53_APC_TAIL_FLOOR=0` in `.env` and restart, or remove the line (the launcher
default is 0, which leaves every file byte-identical). Backups on the node:
`.env.bak-pre-task45-apc-tail-20260919`,
`start.sh.bak-pre-task45-apc-tail-20260919`.
