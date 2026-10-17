# 2026-10-17 — prefix-cache WARN investigation and fix

## Symptom

`./launch.sh smoke` reported:

```
[WARN] prefix_cache: HTTP 200/200; cached_tokens=0
```

while the server was otherwise healthy and `--enable-prefix-caching` was on.

## Live probe (before the fix)

`scripts/probe_prefix_cache.py` sends the same unique random token-ID prompt
twice and records the API field and the `/metrics` counters per request.

```
cache: block_size=4864 mamba_block_size=256 fine_grained_mamba=False
       match_unit=None enabled=True
    len req  prompt  cached   q_delta hit_delta
   1024   1    1024       -      1024         0
   1024   2    1024       -      1024      1024
   4096   1    4096       -      4096         0
   4096   2    4096       -      4096      4096
   8192   1    8192       -      8192         0
   8192   2    8192       -      8192      8192
  16384   1   16384       -     16384         0
  16384   2   16384       -     16384     16384
```

The cache is working: every second request hits 100%. The API field is `-`
because the server was not started with `--enable-prompt-tokens-details`.
The smoke's `details.get(...) or 0` then converted the missing field into a
zero, producing a false WARN.

## Why the field is null

In vLLM's chat-completion path,
`_make_prompt_tokens_details()` returns `None` unless
`enable_prompt_tokens_details` is true; the flag is default-off in
`vllm/entrypoints/launchers/cli_args.py`. The completion path has the same
guard. This is a reporting switch, not a cache switch.

Upstream context:

- [`issues/44377`](https://github.com/vllm-project/vllm/issues/44377) — zero
  cached tokens omitted from the details.
- [`pull/44383`](https://github.com/vllm-project/vllm/pull/44383) — merged
  fix for that omission; the 2026-10-15 image includes it. The flag is still
  required to emit the object at all.
- [`issues/54607`](https://github.com/vllm-project/vllm/issues/54607) — a 0%
  hit can just mean the shared prefix is shorter than one block.
- [`issues/53749`](https://github.com/vllm-project/vllm/issues/53749) — hybrid
  checkpoints can require one attention block; relevant background, but **not
  this case**, because 1k/4k prompts hit here.
- [`pull/30877`](https://github.com/vllm-project/vllm/pull/30877) — Mamba
  prefix caching with `align` mode, the path this model uses.
- vLLM's own hybrid test
  `tests/v1/core/prefix_cache/test_hybrid_prefix_cache_hit_rate.py` starts the
  server with `--enable-prompt-tokens-details` and asserts on
  `prompt_tokens_details.cached_tokens`; it also reads the block geometry from
  `vllm:cache_config_info`.

## Fix

1. `recipes/glm-5.3-flash-fp4-local.yaml` now passes
   `--enable-prompt-tokens-details`.
2. `scripts/smoke.py` now:
   - reads `vllm:prefix_cache_hits_total` before/after the probe and treats a
     positive delta as a pass even when the API field is absent;
   - reads `cached_tokens` when the field is present;
   - probes a ~6k-token prompt rather than ~800 tokens;
   - reports the cache geometry (`block`, `mamba`, `fine_grained`) in the
     detail.
3. `scripts/probe_prefix_cache.py` is the reproducible diagnostic.

## Verification (after restart)

To be filled from the live pair after the recipe restart: expected
`cached_tokens` equal to the prompt length on the second request, and
`[PASS] prefix_cache` with `hits +…; cached_tokens=…`.

## Files

- `scripts/probe_prefix_cache.py`
- `scripts/smoke.py`
- `recipes/glm-5.3-flash-fp4-local.yaml`
- `docs/02-parameters.md`, `docs/04-kv-cache.md`
