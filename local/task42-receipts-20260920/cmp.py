#!/usr/bin/env python3
"""Task 42 parity comparison: stock vs variant layer output, elementwise.

Usage:  cmp.py [out_dir]        (default /out; the harness writes there)

Reads `parity-ctrl.json` / `parity-var.json` and the per-case output tensors the
harness saved beside them (`<out>.t<tokens>.s<skew>.pt`).

Fail-closed. This is used as a gate, so it exits non-zero rather than printing a
verdict and returning 0: a missing or non-finite tensor, a differing case list,
a shape mismatch, a non-finite metric, or a failed tolerance check all abort.
Note that a NaN metric would otherwise pass silently -- `max(0.0, nan)` is 0.0 in
Python -- which is exactly the case this guards.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import torch

# fp16 has ~3 decimal digits; a tolerance of 2e-3 on the mean-abs scale is the
# conventional "same numerics" bar used elsewhere in this kit.
TOL = 2e-3

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/out")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_case(arm: str, case: dict) -> torch.Tensor:
    path = OUT / f"parity-{arm}.json.t{case['tokens']}.s{case['skew']}.pt"
    if not path.is_file():
        fail(f"missing tensor {path}")
    tensor = torch.load(path)
    if not torch.is_tensor(tensor):
        fail(f"{path} did not load a tensor")
    if not bool(torch.isfinite(tensor).all()):
        fail(f"{path} contains non-finite values")
    return tensor


def case_ids(doc: dict) -> list[tuple]:
    return [(x["tokens"], x["skew"], x["path"], tuple(x["shape"])) for x in doc["cases"]]


def main() -> int:
    ctrl = json.loads((OUT / "parity-ctrl.json").read_text())
    var = json.loads((OUT / "parity-var.json").read_text())

    # Compare the case identity lists instead of zipping: a short or reordered
    # list must not read as a pass.
    ids_ctrl, ids_var = case_ids(ctrl), case_ids(var)
    if ids_ctrl != ids_var:
        fail(f"case lists differ:\n  ctrl={ids_ctrl}\n  var ={ids_var}")
    if not ids_ctrl:
        fail("no cases")

    print(f"{'tokens':>7} {'skew':>5} {'bit-exact':>10} {'maxabs delta':>14} "
          f"{'rel maxabs':>12} {'nrmse':>12}")
    worst_rel = 0.0
    bit_exact = 0
    for cc in ctrl["cases"]:
        a = load_case("ctrl", cc)
        b = load_case("var", cc)
        if a.shape != b.shape:
            fail(f"shape mismatch at t={cc['tokens']} skew={cc['skew']}: "
                 f"{tuple(a.shape)} vs {tuple(b.shape)}")
        same = bool(torch.equal(a, b))
        d = (a - b).abs()
        scale = float(a.abs().mean().clamp_min(1e-6))
        mx = float(d.max())
        rel = mx / scale
        nrmse = float(d.square().mean().sqrt()) / scale
        for name, value in (("maxabs delta", mx), ("rel maxabs", rel), ("nrmse", nrmse)):
            if not math.isfinite(value):
                fail(f"non-finite {name} at t={cc['tokens']} skew={cc['skew']}: {value}")
        worst_rel = max(worst_rel, rel)
        bit_exact += int(same)
        print(f"{cc['tokens']:>7} {cc['skew']:>5} {str(same):>10} {mx:>14.4e} "
              f"{rel:>12.3e} {nrmse:>12.3e}")

    print()
    print(f"cases: {len(ids_ctrl)}   bit-exact: {bit_exact}   "
          f"worst rel maxabs: {worst_rel:.3e}")
    print("ctrl paths:", [x["path"] for x in ctrl["cases"]])
    print("var  paths:", [x["path"] for x in var["cases"]])
    if worst_rel >= TOL:
        fail(f"worst rel maxabs {worst_rel:.3e} >= tolerance {TOL:.1e}")
    print("VERDICT: PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
