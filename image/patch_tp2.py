#!/usr/bin/env python3
"""Add the FlashInfer sparse-MLA specialization GLM-5.3-Flash needs at TP=2.

FlashInfer dispatches sparse-MLA decode on a (num_heads, topk) key. GLM's kpool
buffer is 2176 wide (index_topk 2048 + always-selected tail, rounded up to
BLOCK_N=128), so a 2176 specialization is required for every head count served.

The base image supplies (64, 2176) -- correct for TP=1, where one rank owns all
64 attention heads. **At TP=2 each rank owns 32.** The decode lookup then misses,
FlashInfer falls through to the paged/prefill kernel, and that kernel rejects any
batch of <=64 tokens:

    tvm.error.InternalError: Check failed: num_tokens > 64 (4 vs. 64) :
    Decode (num_tokens <= 64) must go through sparse_mla_sm120_decode_dsv3_2

This script adds 32 alongside 64. NH=32 is not a new shape for these kernels:
upstream FlashInfer already instantiates launch_prefill_mg at NH=32 for
topk=2048, and the decode side already carries (32, 2048). We are adding a topk
to existing head counts, not inventing one.

Every edit asserts its anchor, so an unexpected base image fails the build loudly
instead of silently producing an unpatched image.
"""

from __future__ import annotations

import pathlib
import sys

SITE = pathlib.Path("/usr/local/lib/python3.12/dist-packages")
PY = SITE / "flashinfer/mla/_sparse_mla_sm120.py"
DECODE = SITE / "flashinfer/data/csrc/sparse_mla_sm120_decode_dsv3_2.cu"
PREFILL = SITE / "flashinfer/data/csrc/sparse_mla_sm120_prefill.cu"


def die(msg: str) -> None:
    sys.exit(f"FAIL: {msg}")


def insert_after(path: pathlib.Path, anchor: str, addition: str) -> None:
    text = path.read_text()
    if addition in text:
        die(f"{path.name}: already contains {addition.strip()!r}")
    if text.count(anchor) != 1:
        die(f"{path.name}: anchor {anchor.strip()!r} appears {text.count(anchor)}x, expected 1")
    path.write_text(text.replace(anchor, anchor + addition, 1))
    print(f"  ok  {path.name}: + {addition.strip()!r}")


print("== decode dispatch set (python)")
insert_after(PY, "        (32, 2048),\n", "        (32, 2176),\n")

print("== decode instantiation (cuda)")
insert_after(DECODE, "  DSV3_2_DISPATCH(32, 2048)\n", "  DSV3_2_DISPATCH(32, 2176)\n")

print("== prefill GLM_NSA branch: widen 64-only to {32, 64}")
text = PREFILL.read_text()
anchor = """    if constexpr (MT == ModelType::GLM_NSA) {
      if (num_heads != 64) return false;
      launch_prefill_mg<MT, ComputeMode::FP8, 64, 2176, 64>(
          Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens,
          stride_kv_block, topk_length_ptr, stream);
      return true;
    }"""
replacement = """    if constexpr (MT == ModelType::GLM_NSA) {
#define GLM_NSA_MG_2176(NH)                                                             \\
  launch_prefill_mg<MT, ComputeMode::FP8, NH, 2176, 64>(                                \\
      Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens,                 \\
      stride_kv_block, topk_length_ptr, stream)
      switch (num_heads) {
        case 32:
          GLM_NSA_MG_2176(32);
          return true;
        case 64:
          GLM_NSA_MG_2176(64);
          return true;
        default:
          return false;
      }
#undef GLM_NSA_MG_2176
    }"""
# Guard on OUR marker, not on "case 32:" -- upstream's own topk=2048 dispatch
# already contains `case 32:`, so testing for that aborts on a clean base image.
if "GLM_NSA_MG_2176" in text:
    die("prefill.cu: already patched (GLM_NSA_MG_2176 present)")
if text.count(anchor) != 1:
    die("prefill.cu: GLM_NSA 2176 branch not found in the expected form")
PREFILL.write_text(text.replace(anchor, replacement, 1))
print("  ok  sparse_mla_sm120_prefill.cu: heads {32, 64} at topk=2176")

print("\npatched for TP=2 (num_heads=32) alongside the base image's TP=1 (64)")
