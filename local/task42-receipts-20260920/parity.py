#!/usr/bin/env python3
"""Task 42 kernel-parity harness.

Builds the production-shape MoE layer deterministically, runs the *real* serving
entry point (`apply_exl3_experts`) at decode token counts, and dumps the output
tensors so two arms (stock vs variant kernel selection) can be compared
elementwise. Nothing here is random at run time: the layer weights and the
routing come from fixed seeds, so the same bytes must come out of both arms.

Run once per arm with a different --out; compare with --compare.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("EXL3_FAT_EXPERT_LOG", "0")

import torch  # noqa: E402

HID, INTER, NEXP, TOPK = 4096, 2048, 288, 8


def make_layer(n_exp=NEXP, hidden=HID, inter=INTER, seed=0):
    from vllm.model_executor.layers.quantization.exl3 import (
        MCG_MARKER_SIGNED_INT32, Exl3Config, Exl3MoEMethod,
    )
    import types

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
    ap.add_argument("--tokens", default="12,20,32,1024")
    ap.add_argument("--cap", type=int, default=32)
    args = ap.parse_args()

    from vllm.model_executor.layers.quantization.exl3 import (
        _FUSED_TEMP_CACHE, _record_exl3_fat_resolution, apply_exl3_experts,
        build_exl3_fused_state,
    )

    arm = os.environ.get("GLM53_EXL3_MOE_PIPELINE", "0")
    rec = {"arm": arm, "tokens": args.tokens, "cap": args.cap, "cases": []}

    layer = make_layer()
    os.environ.update({"EXL3_FAT_KERNEL": "1", "EXL3_FAT_SORTED": "0",
                       "EXL3_FAT_BATCHED": "0", "EXL3_MOE_ROW_TILE": "0"})
    os.environ["EXL3_FAT_GROUPED"] = "1"
    os.environ["EXL3_TEMP_ROWS_FUSED"] = str(args.cap)
    _FUSED_TEMP_CACHE.clear()
    build_exl3_fused_state(layer, layer._exl3_inners)
    _record_exl3_fat_resolution(layer)

    for t in [int(x) for x in args.tokens.split(",")]:
        for skew in (1.0, 0.0):
            x = torch.randn(t, HID, generator=torch.Generator().manual_seed(7)).half().cuda()
            ids, w = routing(t, skew=skew, seed=5)
            y = apply_exl3_experts(x, ids, w, layer)
            torch.cuda.synchronize()
            yc = y.detach().float().cpu()
            raw = yc.numpy().tobytes()
            rec["cases"].append({
                "tokens": t, "skew": skew,
                "path": layer._exl3_last_fat_fallback,
                "shape": list(y.shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "finite": bool(torch.isfinite(yc).all()),
                "sum": float(yc.sum()), "absum": float(yc.abs().sum()),
                "maxabs": float(yc.abs().max()),
            })
            torch.save(yc, f"{args.out}.t{t}.s{skew}.pt")
            print(f"  t={t:5d} skew={skew} path={layer._exl3_last_fat_fallback:8} "
                  f"sum={float(yc.sum()):+.6f} sha={rec['cases'][-1]['sha256'][:16]}", flush=True)
            del x, y, yc
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
