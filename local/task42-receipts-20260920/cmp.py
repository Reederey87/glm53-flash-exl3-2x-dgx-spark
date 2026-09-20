#!/usr/bin/env python3
"""Task 42 parity comparison: stock vs variant layer output, elementwise."""
import json
import pathlib
import statistics

import torch

out = pathlib.Path("/out")
c = json.loads((out / "parity-ctrl.json").read_text())
v = json.loads((out / "parity-var.json").read_text())

print(f"{'tokens':>7} {'skew':>5} {'bit-exact':>10} {'maxabs delta':>14} {'rel maxabs':>12} {'nrmse':>12}")
worst_rel = 0.0
bit_exact = 0
n = 0
for cc, vv in zip(c["cases"], v["cases"]):
    a = torch.load(f"{out}/parity-ctrl.json.t{cc['tokens']}.s{cc['skew']}.pt")
    b = torch.load(f"{out}/parity-var.json.t{vv['tokens']}.s{vv['skew']}.pt")
    same = bool(torch.equal(a, b))
    d = (a - b).abs()
    scale = float(a.abs().mean().clamp_min(1e-6))
    mx = float(d.max())
    rel = mx / scale
    rms = float(d.square().mean().sqrt())
    nrmse = rms / scale
    worst_rel = max(worst_rel, rel)
    bit_exact += int(same)
    n += 1
    print(f"{cc['tokens']:>7} {cc['skew']:>5} {str(same):>10} {mx:>14.4e} {rel:>12.3e} {nrmse:>12.3e}")

print()
print(f"cases: {n}   bit-exact: {bit_exact}   worst rel maxabs: {worst_rel:.3e}")
print("ctrl paths:", [x["path"] for x in c["cases"]])
print("var  paths:", [x["path"] for x in v["cases"]])
# fp16 has ~3 decimal digits; a tolerance of 2e-3 on the mean-abs scale is the
# conventional "same numerics" bar used elsewhere in this kit.
print("VERDICT:", "PARITY OK" if worst_rel < 2e-3 else "PARITY FAIL")
