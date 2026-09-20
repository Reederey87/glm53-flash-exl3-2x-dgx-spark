#!/usr/bin/env python3
"""Task 42 kernel-level probe: attribute GPU device time by kernel name.

Copied from the 2026-09-20 lever-1 window and extended to record
GLM53_EXL3_MOE_REUSE so the shared_input instance is an independent variable.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

os.environ.setdefault("EXL3_FAT_EXPERT_LOG", "0")

import torch  # noqa: E402

HID, INTER, NEXP, TOPK = 4096, 2048, 288, 8


def make_layer(n_exp=NEXP, hidden=HID, inter=INTER, seed=0):
    import types
    from vllm.model_executor.layers.quantization.exl3 import (
        MCG_MARKER_SIGNED_INT32, Exl3Config, Exl3MoEMethod,
    )

    moe = types.SimpleNamespace(swiglu_limit=10.0)
    method = Exl3MoEMethod(moe, Exl3Config())
    layer = torch.nn.Module()
    method.create_weights(layer, num_experts=n_exp, hidden_size=hidden,
                          intermediate_size_per_partition=inter, params_dtype=torch.float16)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    with torch.no_grad():
        layer.w13_trellis.copy_(torch.randint(-30000, 30000, tuple(layer.w13_trellis.shape),
                                              dtype=torch.int16, generator=g))
        layer.w2_trellis.copy_(torch.randint(-30000, 30000, tuple(layer.w2_trellis.shape),
                                              dtype=torch.int16, generator=g))
        for p in (layer.w13_suh, layer.w13_svh, layer.w2_suh, layer.w2_svh):
            p.copy_((torch.randn(tuple(p.shape), generator=g) * 0.5).half())
        layer.w13_suh[:, 1].copy_(layer.w13_suh[:, 0])
        layer.w13_mcg.fill_(MCG_MARKER_SIGNED_INT32)
        layer.w2_mcg.fill_(MCG_MARKER_SIGNED_INT32)
    layer = layer.to("cuda:0")
    method.process_weights_after_loading(layer)
    return layer


def routing(tokens, n_exp=NEXP, topk=TOPK, skew=1.0, seed=0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    p = 1.0 / torch.arange(1, n_exp + 1).float() ** skew
    p = p[torch.randperm(n_exp, generator=g)]
    ids = torch.multinomial(p.expand(tokens, -1), topk, replacement=False, generator=g)
    w = torch.rand(tokens, topk, generator=g).softmax(-1).half()
    return ids.cuda(), w.cuda()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default="12,20,32")
    ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    from vllm.model_executor.layers.quantization.exl3 import (
        _FUSED_TEMP_CACHE, _record_exl3_fat_resolution, apply_exl3_experts,
        build_exl3_fused_state,
    )

    rec = {
        "pipeline": os.environ.get("GLM53_EXL3_MOE_PIPELINE", "0"),
        "reuse": os.environ.get("GLM53_EXL3_MOE_REUSE", "0"),
        "arm": os.environ.get("GLM53_EXL3_MOE_REUSE",
                              os.environ.get("GLM53_EXL3_MOE_PIPELINE", "0")),
        "cap": args.cap,
        "iters": args.iters,
        "reuse_aliased": None,
        "cases": [],
    }

    layer = make_layer()
    rec["reuse_aliased"] = bool(getattr(layer, "_exl3_reuse_aliased", False))
    os.environ.update({"EXL3_FAT_KERNEL": "1", "EXL3_FAT_SORTED": "0",
                       "EXL3_FAT_BATCHED": "0", "EXL3_MOE_ROW_TILE": "0"})
    os.environ["EXL3_FAT_GROUPED"] = "1"
    os.environ["EXL3_TEMP_ROWS_FUSED"] = str(args.cap)
    _FUSED_TEMP_CACHE.clear()
    build_exl3_fused_state(layer, layer._exl3_inners)
    _record_exl3_fat_resolution(layer)

    for t in [int(x) for x in args.tokens.split(",")]:
        x = torch.randn(t, HID, generator=torch.Generator().manual_seed(7)).half().cuda()
        ids, w = routing(t, skew=1.0, seed=5)
        for _ in range(5):
            apply_exl3_experts(x, ids, w, layer)
        torch.cuda.synchronize()

        ev = []
        for _ in range(args.iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            apply_exl3_experts(x, ids, w, layer)
            e.record()
            torch.cuda.synchronize()
            ev.append(s.elapsed_time(e))

        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                apply_exl3_experts(x, ids, w, layer)
            torch.cuda.synchronize()
        by_kernel: dict[str, list[float]] = {}
        for evt in prof.key_averages():
            if evt.device_type == torch.autograd.DeviceType.CUDA or evt.self_device_time_total > 0:
                by_kernel[evt.key] = [evt.self_device_time_total, evt.count]
        tot_us = sum(v[0] for v in by_kernel.values()) / max(1, args.iters)
        fused = {k: v for k, v in by_kernel.items()
                 if "exl3_moe" in k or "pipeline_kernel" in k}
        fused_us = sum(v[0] for v in fused.values()) / max(1, args.iters)
        top = sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:6]
        case = {
            "tokens": t,
            "path": layer._exl3_last_fat_fallback,
            "wall_median_ms": statistics.median(ev),
            "device_total_us_per_call": tot_us,
            "fused_us_per_call": fused_us,
            "fused_share_pct": (100.0 * fused_us / tot_us) if tot_us else 0.0,
            "fused_kernels": {k: v[0] / max(1, args.iters) for k, v in fused.items()},
            "top_kernels": [{"name": k[:90], "us_per_call": v[0] / max(1, args.iters), "n": v[1]}
                            for k, v in top],
        }
        rec["cases"].append(case)
        print(f"  t={t:5d} wall={case['wall_median_ms']:8.3f} ms  device={tot_us:9.1f} us  "
              f"fused={fused_us:8.1f} us ({case['fused_share_pct']:5.1f}%)  path={case['path']}",
              flush=True)
        for k, v in fused.items():
            print(f"      fused kernel: {k[:110]}  {v[0]/max(1, args.iters):.1f} us/call",
                  flush=True)
        del x
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print("wrote", args.out)
    print("reuse_aliased", rec["reuse_aliased"], "pipeline", rec["pipeline"], "reuse", rec["reuse"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
