# The TP=2 kernel fix

This is the contribution of this repository. Everything else here is
orchestration.

## Background

GLM-5.3-Flash's 11 full-attention layers are **NoPE sparse MLA**. On an
sm_121 GB10 there is no stock kernel for them:

- vLLM's GLM-5.3 support (PR #53906) ships NoPE sparse decode for **SM90** and
  **SM100** only. For compute capability 12 the MLA backend list is
  `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`.
- `FLASHINFER_MLA_SPARSE_SM120` requires the packed `fp8_ds_mla` layout, whose
  `concat_and_cache_mla` asserts `pe_dim == 64`. GLM is NoPE — `pe_dim` is 0.

[cyijun/glm-5.3-flash-nvfp4-gb10](https://github.com/cyijun/glm-5.3-flash-nvfp4-gb10)
solved this for a **single** GB10: pad the NoPE query into the 656-byte GLM_NSA
ABI, and rebuild FlashInfer's AOT sparse module with a `(64, 2176)` decode
specialization. This repo builds on that image and all credit for that work is
theirs.

## What breaks at TP=2

FlashInfer dispatches sparse-MLA decode on a `(num_heads, topk)` key.

`topk` is **2176**, not 2048: GLM's kpool path adds an always-selected tail and
the buffer is rounded up to FlashInfer's `BLOCK_N=128`.

`num_heads` is where tensor parallelism changes the answer. GLM has **64**
attention heads:

| | heads per rank | needs specialization |
|---|---:|---|
| TP=1 (one GB10) | 64 | `(64, 2176)` — present in the base image |
| **TP=2 (two GB10s)** | **32** | **`(32, 2176)` — absent everywhere** |

With the lookup missing, FlashInfer silently falls through to the paged/prefill
kernel, which refuses any batch of 64 tokens or fewer. A 4-token decode batch
during warmup trips it:

```
tvm.error.InternalError: Check failed: num_tokens > 64 (4 vs. 64) :
Decode (num_tokens <= 64) must go through sparse_mla_sm120_decode_dsv3_2
```

The engine loads the full 181 GiB checkpoint, sizes a KV pool, and *then* dies —
which makes it an expensive failure to diagnose.

## The fix

`image/patch_tp2.py` makes three edits, then the Dockerfile rebuilds the AOT
module for `sm_121a`:

1. `flashinfer/mla/_sparse_mla_sm120.py` — add `(32, 2176)` to the dispatch set.
2. `sparse_mla_sm120_decode_dsv3_2.cu` — add `DSV3_2_DISPATCH(32, 2176)`.
3. `sparse_mla_sm120_prefill.cu` — the GLM_NSA 2176 branch hardcodes
   `if (num_heads != 64) return false;`. Widen it to a switch over `{32, 64}`.

**This is not a new kernel shape.** Upstream FlashInfer already instantiates
`launch_prefill_mg` at `NH=32` for `topk=2048`, and the decode side already
carries `(32, 2048)`. We are adding a `topk` to head counts that already exist —
which is why it compiles and runs without further work.

## Why it must be a rebuild

FlashInfer ships this module as a **precompiled AOT `.so`** that takes
precedence over its Python and CUDA sources. Editing the sources alone changes
nothing at runtime. The Dockerfile therefore moves the shipped `.so` aside,
rebuilds with `FLASHINFER_CUDA_ARCH_LIST=12.1a`, and installs the result back
into the same slot.

This is also why a pure-Python overlay can never fix the sparse path — an
attempt that pads the query in Python still calls the old compiled kernel.

## Verification

The build fails if the rebuilt module lacks either specialization
(`image/verify_tp2.py`). To check an image by hand:

```bash
docker run --rm --entrypoint bash <image> -lc \
  'GLM53_GB10_PATCH_DISABLE=1 python3 -c "
import flashinfer.mla._sparse_mla_sm120 as m
print(sorted(k for k in m._DECODE_DSV3_2_DISPATCH if k[1]==2176))"'
# expect: [(32, 2176), (64, 2176)]
```

## Extending to other topologies

TP=4 would need `(16, 2176)`. Add it the same way: another insert next to
`(16, 2048)` in the two dispatch tables, and another `case 16:` in the prefill
switch in `patch_tp2.py`. Each entry costs compile time and register pressure,
so add only the topology you actually serve.
