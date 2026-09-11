#!/usr/bin/env python3
"""Task 38 item 2 — silicon probe for `cvt.rn.satfinite.e2m1x2.f32` on GB10.

Item 1 established the *toolchain* half of the FP4-e2m1 correction: with CUDA
13.0.88's ptxas, `cvt.rn.satfinite.e2m1x2.f32` is rejected on `.target sm_121`
but assembles on `.target sm_121a` and lowers to real SASS
(`F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`). That proves the instruction is
**target-gated, not silicon-absent**. This probe closes the remaining half: does
the silicon *execute* it and produce architecturally correct values?

It does not need a compiler on the host and it does not need a stopped window.
A one-kernel PTX module is generated here, JIT'd by the CUDA **driver**
(`cuModuleLoadDataEx`), launched, and read back; the host side is plain `ctypes`
against `libcuda.so.1`. Measured 2026-09-11: a 1-byte `cuMemAlloc` and a
single-thread launch both succeed while production holds the GPU, so the probe
runs alongside the serving stack. (This is the opposite of the original probe,
which needed the primary context and could not get it — see `--assemble`.)

Three measured facts are load-bearing and are asserted by the tests:

1. **The destination is `.b8` and the store is `st.global.u8`.** Declaring
   `.b32`/`st.global.u32` makes ptxas reject the instruction ("Arguments
   mismatch for instruction 'cvt'"), and the driver JIT then rejects the module
   with `CUDA_ERROR_INVALID_PTX` (218).
2. **The operands must come from memory, not immediates.** With `mov.f32 %f1,
   0f3F800000` ptxas materializes only the top byte of each constant and every
   case reads `0x00`; passing the two floats as kernel *parameters* produces the
   correct values. The probe therefore passes them as parameters.
3. **`a` lands in the upper nibble and `b` in the lower**, matching the PTX ISA
   ("the value converted from input a is stored in the upper 4 bits of d and the
   value converted from input b is stored in the lower 4 bits of d").

Verdicts: `SILICON_CONFIRMED` (every case matches the spec), `SILICON_MISMATCH`
(some case does not, or the nibble order is reversed), `PROBE_UNAVAILABLE` (the
driver refused to load or launch — reported, never silently swallowed).

Run `--ptx-only` to emit the PTX on a machine with no free GPU.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import struct
import subprocess
import sys
from pathlib import Path

# One kernel, three parameters. Immediates are deliberately not used: ptxas
# mis-materializes them for this instruction (see the module docstring).
PTX = """.version 9.0
.target sm_121a
.address_size 64

.visible .entry probe_e2m1(.param .u64 p_out, .param .f32 pa, .param .f32 pb)
{
    .reg .b64  %rd<4>;
    .reg .b8   %rs<2>;
    .reg .f32  %f<4>;

    ld.param.u64      %rd1, [p_out];
    cvta.to.global.u64 %rd2, %rd1;
    ld.param.f32 %f1, [pa];
    ld.param.f32 %f2, [pb];
    cvt.rn.satfinite.e2m1x2.f32 %rs1, %f1, %f2;
    st.global.u8 [%rd2], %rs1;
    ret;
}
"""

# E2M1 code -> value. Code bit 3 is the sign; bits 2:0 index the magnitude.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0

# (name, a, b, is_tie). Ties are recorded but excluded from the gate: `cvt.rn`'s
# tie rule is the one thing a hand-written reference is least entitled to assume.
CASES: tuple[tuple[str, float, float, bool], ...] = (
    ("recorded-reversed-2.0-1.0", 2.0, 1.0, False),
    ("zero", 0.0, 0.0, False),
    ("half", 0.5, 0.5, False),
    ("distinct-1.0-2.0", 1.0, 2.0, False),
    ("distinct-4.0-1.0", 4.0, 1.0, False),
    ("top-of-range", 6.0, 6.0, False),
    ("negative", -1.0, -6.0, False),
    ("saturate-high", 100.0, 1e30, False),
    ("saturate-low", -100.0, -1e30, False),
    ("round-down", 0.4, 1.2, False),
    ("round-up", 0.6, 2.7, False),
    ("tie-0.25-0.75", 0.25, 0.75, True),
)


def e2m1_code(value: float, *, ties_to_even: bool = True) -> int:
    """Round-to-nearest e2m1 code for one f32 (saturating)."""
    sign = 1 if value < 0 else 0
    magnitude = abs(value)
    if magnitude != magnitude:  # NaN is not in the e2m1 alphabet
        raise ValueError("NaN has no e2m1 encoding")
    if magnitude > E2M1_MAX:
        return (sign << 3) | 0b111
    best_index = 0
    best_delta = None
    for index, candidate in enumerate(E2M1_VALUES):
        delta = abs(magnitude - candidate)
        if best_delta is None or delta < best_delta - 1e-12:
            best_index, best_delta = index, delta
        elif abs(delta - best_delta) <= 1e-12 and ties_to_even:
            # Exact midpoint: cvt.rn is round-to-nearest-even, so prefer the
            # code with an even mantissa bit (an even magnitude index).
            if index % 2 == 0:
                best_index = index
    return (sign << 3) | best_index


def expected_pair(a: float, b: float) -> int:
    """The PTX-specified packing: `a` in the upper nibble, `b` in the lower."""
    return (e2m1_code(a) << 4) | e2m1_code(b)


def reversed_pair(a: float, b: float) -> int:
    """The opposite nibble order, used only to detect a reversed result."""
    return (e2m1_code(b) << 4) | e2m1_code(a)


def build_ptx() -> str:
    return PTX


# --- driver API -------------------------------------------------------------

def _load_libcuda() -> ctypes.CDLL:
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("libcuda not found")


def _check(result: int, what: str) -> None:
    if result != 0:
        raise RuntimeError(f"{what} failed with CUDA error {result}")


def run_on_gpu(ptx: str, cases=CASES) -> list[int]:
    """JIT the PTX with the driver and return one packed byte per case."""
    lib = _load_libcuda()
    _check(lib.cuInit(0), "cuInit")
    device = ctypes.c_int()
    _check(lib.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    context = ctypes.c_void_p()
    _check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), "cuDevicePrimaryCtxRetain")
    _check(lib.cuCtxSetCurrent(context), "cuCtxSetCurrent")
    module = ctypes.c_void_p()
    source = ctypes.create_string_buffer(ptx.encode())
    _check(lib.cuModuleLoadDataEx(ctypes.byref(module), source, 0, None, None),
           "cuModuleLoadDataEx")
    function = ctypes.c_void_p()
    _check(lib.cuModuleGetFunction(ctypes.byref(function), module, b"probe_e2m1"),
           "cuModuleGetFunction")

    observed: list[int] = []
    for _name, a, b, _tie in cases:
        device_ptr = ctypes.c_ulonglong()
        _check(lib.cuMemAlloc_v2(ctypes.byref(device_ptr), 1), "cuMemAlloc")
        try:
            out_val = ctypes.c_ulonglong(device_ptr.value)
            a_val = ctypes.c_float(a)
            b_val = ctypes.c_float(b)
            # kernelParams is an array of POINTERS, one per kernel parameter.
            params = (ctypes.c_void_p * 3)(
                ctypes.cast(ctypes.byref(out_val), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(a_val), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(b_val), ctypes.c_void_p),
            )
            _check(lib.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, None, params, None),
                   "cuLaunchKernel")
            _check(lib.cuCtxSynchronize(), "cuCtxSynchronize")
            host = (ctypes.c_ubyte * 1)()
            _check(lib.cuMemcpyDtoH_v2(host, device_ptr, 1), "cuMemcpyDtoH")
            observed.append(host[0])
        finally:
            lib.cuMemFree_v2(device_ptr)
    return observed


# --- judging ----------------------------------------------------------------

def judge(observed: list[int]) -> dict:
    if len(observed) != len(CASES):
        raise RuntimeError(f"expected {len(CASES)} results, got {len(observed)}")
    rows = []
    errors: list[str] = []
    reversed_cases = 0
    gated = 0
    for (name, a, b, tie), seen in zip(CASES, observed):
        expect = expected_pair(a, b)
        flipped = reversed_pair(a, b)
        matches_spec = seen == expect
        matches_reversed = seen == flipped and flipped != expect
        row = {
            "case": name, "a": a, "b": b, "tie": tie, "observed": f"0x{seen:02x}",
            "expected_spec": f"0x{expect:02x}", "matches_spec": matches_spec,
            "matches_reversed_order": matches_reversed,
        }
        rows.append(row)
        if tie:
            continue  # recorded, not gated
        gated += 1
        if matches_reversed:
            reversed_cases += 1
        if not matches_spec:
            errors.append(
                f"{name} (a={a}, b={b}): silicon produced 0x{seen:02x}, "
                f"PTX ISA expects 0x{expect:02x} (a in the upper nibble, b in the lower)"
            )
    result: dict = {
        "schema": 1,
        "probe": "cvt.rn.satfinite.e2m1x2.f32",
        "cases": rows,
        "gated_cases": gated,
        "cases_matching_spec": sum(1 for row in rows if row["matches_spec"]),
        "reversed_order_cases": reversed_cases,
        "errors": errors,
    }
    if reversed_cases:
        errors.append(
            f"{reversed_cases} case(s) match the REVERSED nibble order: the silicon "
            "packs a into the lower nibble, contradicting the PTX ISA"
        )
    if gated and not any(row["matches_spec"] for row in rows if not row["tie"]):
        errors.append("no non-tie case matched the PTX-specified encoding")
    result["verdict"] = "SILICON_CONFIRMED" if not errors else "SILICON_MISMATCH"
    return result


def assemble_with_ptxas(ptx: str, workdir: Path) -> dict:
    """Compiler-only check: does ptxas accept this kernel for sm_121a?"""
    source = workdir / "run_probe.ptx"
    source.write_text(ptx)
    try:
        proc = subprocess.run(
            ["ptxas", "-arch=sm_121a", "-o", str(workdir / "run_probe.cubin"), str(source)],
            capture_output=True, text=True, timeout=300,
        )
    except FileNotFoundError:
        return {"ptxas": "not found", "rc": None}
    return {
        "ptxas": "ptxas",
        "rc": proc.returncode,
        "stdout": proc.stdout.strip()[-2000:],
        "stderr": proc.stderr.strip()[-2000:],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="write the result JSON here")
    ap.add_argument("--ptx-only", action="store_true", help="emit the PTX and stop")
    ap.add_argument("--assemble", action="store_true",
                    help="also run ptxas -arch=sm_121a over the PTX (item 1 re-check)")
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/isa_probe_p8"))
    args = ap.parse_args(argv)

    ptx = build_ptx()
    if args.ptx_only:
        args.workdir.mkdir(parents=True, exist_ok=True)
        (args.workdir / "run_probe.ptx").write_text(ptx)
        print(ptx)
        return 0

    result: dict = {"schema": 1, "probe": "cvt.rn.satfinite.e2m1x2.f32", "ptx": ptx}
    if args.assemble:
        args.workdir.mkdir(parents=True, exist_ok=True)
        result["ptxas"] = assemble_with_ptxas(ptx, args.workdir)
    try:
        observed = run_on_gpu(ptx)
    except Exception as exc:  # noqa: BLE001  (a probe that cannot run is a result)
        result.update({"verdict": "PROBE_UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"})
        _emit(result, args.out)
        print(f"[isa-probe] PROBE_UNAVAILABLE: {exc}", file=sys.stderr)
        return 2
    result.update(judge(observed))
    _emit(result, args.out)
    print(f"[isa-probe] verdict={result['verdict']} "
          f"spec_matches={result['cases_matching_spec']}/{len(CASES)} "
          f"reversed={result['reversed_order_cases']}", flush=True)
    return 0 if result["verdict"] == "SILICON_CONFIRMED" else 1


def _emit(result: dict, out: Path | None) -> None:
    text = json.dumps(result, indent=1) + "\n"
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    sys.exit(main())
