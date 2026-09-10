#!/usr/bin/env python3
"""Offline E3 grouped-kernel occupancy probe (task 24 W5 counter lane).

Builds the production E3 grouped path (`EXL3_FAT_GROUPED=1`, TRF cap) in an
isolated process at production per-rank geometry and runs `apply_exl3_experts`
a fixed number of times, so an external profiler (ncu / nsys) can measure the
`fm_gateup_kernel` / `fm_down_kernel` launches without touching the live
server. Kernel resource usage (registers, SMEM) and the per-SM occupancy limit
are properties of the compiled cubin and the launch geometry, not of the
routed-expert count, so a reduced `--n-exp` is representative while it keeps
the footprint inside the unified memory the running production server leaves
free.

This is a counter vehicle, not an end-to-end benchmark. It makes no speed
claim; use `tests/bench_e3_microbench.py` for isolated timing.

Exit codes:
  0 = grouped path engaged and the requested iterations completed
  2 = the grouped path did not engage (probe is not a valid counter vehicle)
  1 = usage/runtime failure
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("EXL3_FAT_EXPERT_LOG", "0")

import torch  # noqa: E402

# Production per-rank geometry (config.json: hidden 4096, moe_intermediate
# 2048, TP2 -> 1024 per rank; docs/11 §8: live scratch 224 MiB h13 + 56 MiB
# h2 = 280 MiB/rank at MNBT 3584 x topk 8).
HID = 4096
INTER = 1024
TOPK = 8


def make_layer(n_exp: int, hidden: int = HID, inter: int = INTER, seed: int = 0):
    """Small EXL3 MoE layer with the production weight layout and kernels."""
    from vllm.model_executor.layers.quantization.exl3 import (
        MCG_MARKER_SIGNED_INT32,
        Exl3Config,
        Exl3MoEMethod,
    )

    moe = types.SimpleNamespace(swiglu_limit=10.0)
    method = Exl3MoEMethod(moe, Exl3Config())
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        num_experts=n_exp,
        hidden_size=hidden,
        intermediate_size_per_partition=inter,
        params_dtype=torch.float16,
    )
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    with torch.no_grad():
        layer.w13_trellis.copy_(
            torch.randint(
                -30000, 30000, tuple(layer.w13_trellis.shape),
                dtype=torch.int16, generator=g,
            )
        )
        layer.w2_trellis.copy_(
            torch.randint(
                -30000, 30000, tuple(layer.w2_trellis.shape),
                dtype=torch.int16, generator=g,
            )
        )
        for p in (layer.w13_suh, layer.w13_svh, layer.w2_suh, layer.w2_svh):
            p.copy_((torch.randn(tuple(p.shape), generator=g) * 0.5).half())
        layer.w13_suh[:, 1].copy_(layer.w13_suh[:, 0])
        layer.w13_mcg.fill_(MCG_MARKER_SIGNED_INT32)
        layer.w2_mcg.fill_(MCG_MARKER_SIGNED_INT32)
    layer = layer.to("cuda:0")
    method.process_weights_after_loading(layer)
    return layer


def routing(tokens: int, n_exp: int, topk: int = TOPK, skew: float = 1.0, seed: int = 0):
    """Zipf-skewed unique-per-token top-k routing (the live router shape)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    p = 1.0 / torch.arange(1, n_exp + 1).float() ** skew
    p = p[torch.randperm(n_exp, generator=g)]
    ids = torch.multinomial(p.expand(tokens, -1), topk, replacement=False, generator=g)
    w = torch.rand(tokens, topk, generator=g).softmax(-1).half()
    return ids.cuda(), w.cuda()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=1792,
                    help="prefill chunk rows (production LPTT chunk is 1792)")
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--n-exp", type=int, default=64,
                    help="routed experts in the replica (footprint only)")
    ap.add_argument("--hidden", type=int, default=HID)
    ap.add_argument("--inter", type=int, default=INTER,
                    help="per-rank intermediate (h2 width)")
    ap.add_argument("--cap", type=int, default=32,
                    help="EXL3_TEMP_ROWS_FUSED, production last-wins 32")
    ap.add_argument("--skew", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    os.environ.update({
        "EXL3_FAT_KERNEL": "1",
        "EXL3_FAT_SORTED": "0",
        "EXL3_FAT_BATCHED": "0",
        "EXL3_MOE_ROW_TILE": "0",
        "EXL3_FAT_GROUPED": "1",
        "EXL3_TEMP_ROWS_FUSED": str(args.cap),
    })

    from vllm.model_executor.layers.quantization.exl3 import (
        _FUSED_TEMP_CACHE,
        _record_exl3_fat_resolution,
        apply_exl3_experts,
        build_exl3_fused_state,
        exl3_fat_diag,
        exl3_fat_moe_symbols,
    )

    if not exl3_fat_moe_symbols():
        print("error: E3 grouped kernels are not loaded in this image", file=sys.stderr)
        return 1

    rec: dict = {
        "schema": 1,
        "tokens": args.tokens,
        "topk": args.topk,
        "n_exp": args.n_exp,
        "hidden": args.hidden,
        "inter": args.inter,
        "cap": args.cap,
        "skew": args.skew,
        "iters": args.iters,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability()),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
    }

    layer = make_layer(args.n_exp, args.hidden, args.inter)
    # The TRF cap is read when the fused temp state is built; rebuild it so the
    # capture really runs the production cap rather than the build-time default.
    _FUSED_TEMP_CACHE.clear()
    build_exl3_fused_state(layer, layer._exl3_inners)
    _record_exl3_fat_resolution(layer)
    rec["tier"] = layer._exl3_fat_effective_tier
    rec["tier_reason"] = layer._exl3_fat_tier_reason
    if rec["tier"] != "grouped":
        print(f"error: tier is {rec['tier']} ({rec['tier_reason']}), expected grouped",
              file=sys.stderr)
        return 2

    x = torch.randn(args.tokens, args.hidden, dtype=torch.float16, device="cuda")
    ids, w = routing(args.tokens, args.n_exp, args.topk, args.skew)
    counts = torch.bincount(ids.reshape(-1), minlength=args.n_exp)
    rec["routing"] = {
        "max_rows": int(counts.max()),
        "mean_rows": float(counts.float().mean()),
        "n_gt_cap": int((counts > args.cap).sum()),
        "fat_rows": int(counts[counts > args.cap].sum()),
    }

    times: list[float] = []
    for i in range(args.warm + args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = apply_exl3_experts(x, ids, w, layer)
        torch.cuda.synchronize()
        if i >= args.warm:
            times.append((time.perf_counter() - t0) * 1e3)
    rec["last_fat_fallback"] = layer._exl3_last_fat_fallback
    if rec["last_fat_fallback"] != "grouped":
        print(f"error: last fallback is {rec['last_fat_fallback']}, expected grouped",
              file=sys.stderr)
        return 2
    rec["times_ms"] = times
    rec["median_ms"] = statistics.median(times) if times else None
    rec["out_finite"] = bool(torch.isfinite(y).all().item())
    rec["out_shape"] = list(y.shape)
    rec["mem_allocated_bytes"] = int(torch.cuda.memory_allocated())
    rec["mem_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    diag = exl3_fat_diag()
    rec["grouped_scratch_bytes"] = int(diag.get("grouped_scratch_bytes", 0))
    rec["grouped_calls"] = int(diag.get("grouped_calls", 0))

    payload = json.dumps(rec, indent=2)
    if args.out is not None:
        args.out.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
