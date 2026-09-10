#!/usr/bin/env python3
"""Task 34 correctness gate: numeric parity of the FlashKDA fused prefill.

The A/B contract (docs/15 §4) makes this a **required** step before any speed
claim, because the arm replaces a prefill kernel. It compares, on identical
inputs:

  reference  vllm.third_party.flash_linear_attention.ops.kda.chunk_kda_with_fused_gate
             (the Triton chunked path production runs today), called with the
             exact conventions ``Glm5NextLinearAttention.forward`` uses; and
  candidate  torch.ops._flashkda_C.fwd
             (vLLM #55737), called with the exact conventions the overlay's
             ``_flashkda_prefill`` uses.

Conventions that differ and are therefore reproduced explicitly:

  * ``beta``   the Triton path wants it **pre-sigmoided fp32**
               (``_cast_sigmoid``); the fused kernel takes **raw** logits and
               requires bf16 (fp32 raises ``beta must be bfloat16``). The
               ``--beta sigmoid`` variant tests the other convention in bf16.
  * ``g``      both take the **raw** gate, but the Triton path names it
               ``raw_g``.
  * ``A_log``  the Triton path takes the 4-D parameter; the fused kernel takes
               ``A_log.view(-1)`` and ``dt_bias.view(-1, head_dim)``.

Run inside the serving container (needs the GPU, ``vllm._flashkda_C`` and the
Triton kernels). Exits non-zero on divergence so a caller can gate on it. The
image's entrypoint is ``vllm``, so ``--entrypoint python3`` is required:

  docker run --rm --gpus all --ipc=host \
    -v <this>:/parity.py:ro --entrypoint python3 <image> /parity.py

The GPU is reachable alongside the serving container, so this runs without
stopping production.

RESULT 2026-09-10: PARITY_FAILED and the arm was reverted. ``--beta``,
``--pre-l2norm``, ``--reference`` and ``--trivial-gate`` are the isolating
variants that were tried; none of them recovers parity. Full numbers in
``local/task34-parity-gate-20260910.txt``.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

# The bounded gate this deployment runs (kda.py: safe_gate=True, lower_bound=-5).
LOWER_BOUND = -5.0


def _cast_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """kda.py's helper, restated so the harness does not import the layer."""
    return x.float().sigmoid()


def build_inputs(tokens: int, heads: int, head_dim: int, seed: int, device: str,
                 trivial_gate: bool = False):
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    mk = lambda: torch.randn(1, tokens, heads, head_dim, device=device, dtype=dtype)  # noqa: E731
    q, k, v = mk(), mk(), mk()
    # Raw gate logits; kept small so exp(A_log) * (g + dt_bias) stays in a range
    # where both implementations are exercised without saturating the bound.
    g = torch.randn(1, tokens, heads, head_dim, device=device, dtype=dtype) * 0.5
    beta_raw = torch.randn(1, tokens, heads, device=device, dtype=dtype)
    if trivial_gate:
        # Constant decay -lower_bound/2 everywhere: isolates the recurrence
        # and output projection from the gate/lower_bound handling.
        g = torch.zeros_like(g)
        a_log = torch.zeros(heads, device=device, dtype=torch.float32)
        dt_bias = torch.zeros(heads, head_dim, device=device, dtype=torch.float32)
    else:
        a_log = torch.randn(heads, device=device, dtype=torch.float32)
        dt_bias = torch.randn(heads, head_dim, device=device, dtype=torch.float32)
    initial_state = torch.zeros(1, heads, head_dim, head_dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    return {
        "q": q, "k": k, "v": v, "g": g, "beta_raw": beta_raw,
        "a_log": a_log, "dt_bias": dt_bias,
        "initial_state": initial_state, "cu_seqlens": cu_seqlens,
    }


def run_reference(inp, heads: int, head_dim: int, which: str = "glm"):
    if which == "kimi":
        # kimi_k3's copy: raw beta, sigmoid in-kernel, no safe_gate flag.
        from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
            chunk_kda_with_fused_gate as kimi_chunk,
        )

        out, final_state = kimi_chunk(
            q=inp["q"], k=inp["k"], v=inp["v"], raw_g=inp["g"],
            raw_beta=inp["beta_raw"],
            A_log=inp["a_log"].view(1, 1, heads, 1), g_bias=inp["dt_bias"],
            initial_state=inp["initial_state"], output_final_state=True,
            use_qk_l2norm_in_kernel=True, cu_seqlens=inp["cu_seqlens"],
            lower_bound=LOWER_BOUND,
        )
        return out, final_state

    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda_with_fused_gate

    out, final_state = chunk_kda_with_fused_gate(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        raw_g=inp["g"],
        # Pre-sigmoided fp32: the Triton kernels do not sigmoid.
        beta=_cast_sigmoid(inp["beta_raw"].squeeze(0)).unsqueeze(0),
        A_log=inp["a_log"].view(1, 1, heads, 1),
        g_bias=inp["dt_bias"],
        initial_state=inp["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu_seqlens"],
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )
    return out, final_state


def run_candidate(inp, tokens: int, heads: int, head_dim: int, beta_mode: str = "raw",
                  pre_l2norm: bool = False):
    import vllm._flashkda_C  # noqa: F401  — registers the ops

    q = inp["q"]
    if pre_l2norm:
        from vllm.third_party.flash_linear_attention.ops.kda import l2norm_fwd
        q = l2norm_fwd(q.contiguous())
        inp = dict(inp)
        inp["k"] = l2norm_fwd(inp["k"].contiguous())
    max_seqs = 1
    workspace_size = torch.ops._flashkda_C.get_workspace_size(tokens, heads, max_seqs)
    final_state = torch.zeros(
        max_seqs, heads, head_dim, head_dim, device=q.device, dtype=torch.float32
    )
    workspace = torch.zeros(workspace_size, device=q.device, dtype=torch.uint8)
    out = torch.empty(1, tokens, heads, head_dim, device=q.device, dtype=q.dtype)

    # The overlay passes raw beta on the claim that the fused kernel sigmoids
    # in-kernel. ``sigmoid`` is the Triton path's convention, tested here to
    # tell a one-line convention bug apart from a genuine kernel mismatch.
    if beta_mode == "sigmoid":
        # bf16, because the kernel rejects any other dtype.
        beta = _cast_sigmoid(inp["beta_raw"]).to(torch.bfloat16)
    else:
        beta = inp["beta_raw"]

    torch.ops._flashkda_C.fwd(
        q.contiguous(),
        inp["k"].contiguous(),
        inp["v"].contiguous(),
        inp["g"].contiguous(),
        beta,
        head_dim ** -0.5,
        out,
        workspace,
        inp["a_log"].view(-1),
        inp["dt_bias"].view(-1, head_dim),
        LOWER_BOUND,
        inp["initial_state"].contiguous(),
        final_state,
        inp["cu_seqlens"].contiguous(),
    )
    return out, final_state


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    a32 = a.detach().float()
    b32 = b.detach().float()
    diff = (a32 - b32).abs()
    denom = a32.abs().clamp_min(1e-6)
    a_flat = a32.flatten()
    b_flat = b32.flatten()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    denom_c = (a_c.norm() * b_c.norm()).clamp_min(1e-12)
    corr = float((a_c @ b_c / denom_c).item())
    ref_scale = float(a_flat.abs().mean().item())
    return {
        "correlation": corr,
        "reference_mean_abs": ref_scale,
        "candidate_mean_abs": float(b_flat.abs().mean().item()),
        "name": name,
        "shape": list(a.shape),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "max_rel_diff": float((diff / denom).max().item()),
        "reference_abs_max": float(a32.abs().max().item()),
        "all_finite": bool(torch.isfinite(a32).all().item() and torch.isfinite(b32).all().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    # bf16 accumulation differs by reduction order, so the gate is a tolerance
    # on the *relative* error, not bit equality. 1e-2 is loose enough for bf16
    # and still catches a wrong gate, wrong l2norm, or transposed state.
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--pre-l2norm", action="store_true",
                        help="normalize q/k before the fused call (the Triton path does this in-kernel)")
    parser.add_argument("--trivial-gate", action="store_true",
                        help="zero g/dt_bias/A_log: constant decay, gate handling removed")
    parser.add_argument("--reference", choices=("glm", "kimi"), default="glm",
                        help="glm = the deployed production path; kimi = the fork's native pair")
    parser.add_argument("--beta", choices=("raw", "sigmoid"), default="raw",
                        help="raw = the overlay's convention; sigmoid = the Triton path's")
    args = parser.parse_args()

    device = "cuda"
    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA unavailable in this container"}))
        return 2
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}", file=sys.stderr)

    inp = build_inputs(args.tokens, args.heads, args.head_dim, args.seed, device,
                       trivial_gate=args.trivial_gate)
    ref_out, ref_state = run_reference(
        inp, args.heads, args.head_dim, which=args.reference
    )
    cand_out, cand_state = run_candidate(
        inp, args.tokens, args.heads, args.head_dim, beta_mode=args.beta,
        pre_l2norm=args.pre_l2norm,
    )

    results = [
        compare("out", ref_out, cand_out),
        compare("final_state", ref_state, cand_state),
    ]
    verdict = "PARITY_OK"
    for r in results:
        if not r["all_finite"] or r["max_rel_diff"] > args.tol:
            verdict = "PARITY_FAILED"
    report = {
        "verdict": verdict,
        "tokens": args.tokens,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "seed": args.seed,
        "tolerance_max_rel_diff": args.tol,
        "lower_bound": LOWER_BOUND,
        "reference": args.reference,
        "trivial_gate": args.trivial_gate,
        "candidate_beta": args.beta,
        "candidate_pre_l2norm": args.pre_l2norm,
        "comparisons": results,
    }
    print(json.dumps(report, indent=2))
    return 0 if verdict == "PARITY_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
