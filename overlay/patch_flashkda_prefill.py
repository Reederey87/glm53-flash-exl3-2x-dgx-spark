#!/usr/bin/env python3
"""Task 34 arm: FlashKDA fused CUDA chunked prefill for GLM-5.3-Flash (vLLM #55737).

Prefill is this deployment's measured weak lane (240k cold ~1408 tok/s, 60k
~1454). #55737 replaces the ~15-kernel Triton ``chunk_kda_with_fused_gate``
path with ``vllm._flashkda_C`` and reports 1.7-3.8x on the KDA layer and TTFT
-7.9% to -13.2%. That extension **is** present and functional in the deployed
image (``torch.ops._flashkda_C.get_workspace_size(1792, 32, 4)`` -> 51,314,688 B
on this GB10), and the auto-selection gate in the upstream patch accepts
SM12x + bf16 + head_dim 128 + a bounded gate -- all true here.

This overlay ports the upstream change onto the **deployed fork**, which does
not carry upstream's ``glm5next/nvidia/ops/third_party/`` tree; the live KDA
layer is ``vllm/models/glm5next/nvidia/kda.py`` (653 lines, v1.4.7-lineage
fork ``487ecf187``). It is a *port*, not a rebase: the three insertion points
below are this fork's anchors, not upstream's line numbers.

Default OFF. With ``GLM53_KDA_PREFILL_BACKEND`` unset the target file is left
byte-identical, so this overlay cannot change production by merely being
installed. Fail-closed: a drifted anchor raises and the boot aborts rather
than silently serving an unpatched (or half-patched) KDA layer.

Knob (container runtime):
  GLM53_KDA_PREFILL_BACKEND   unset|triton = stock; flashkda = this arm

Not yet cluster-armed: the numeric parity of the fused kernel against the
Triton chunk path, and the e2e prefill/acceptance gates, still need a stopped
maintenance window (see docs/13 and spec/TODO.md task 34). This file is the
prepared candidate, not a receipt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MARK = "# [glm53-flashkda-prefill]"

TARGET = Path(
    os.environ.get(
        "GLM53_GLM_KDA_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/"
        "nvidia/kda.py",
    )
)

KNOB = "GLM53_KDA_PREFILL_BACKEND"

# --- insertion point 1: module-level helper block -------------------------
CLASS_ANCHOR = "class Glm5NextLinearAttention(GatedDeltaNetAttention):\n"

HELPER_BLOCK = f'''from vllm.v1.worker.workspace import current_workspace_manager  {MARK}


def _glm53_flashkda_supported(head_dim, dtype, lower_bound) -> bool:  {MARK}
    """Upstream #55737's auto-selection gate, restated for this fork."""
    capability = current_platform.get_device_capability()
    return bool(
        current_platform.is_cuda()
        and capability is not None
        and capability.major in (9, 10, 12)
        and head_dim == 128
        and dtype == torch.bfloat16
        and lower_bound is not None
    )


'''

# --- insertion point 2: __init__ tail ------------------------------------
INIT_ANCHOR = "        self._conv_state_dim_first = is_conv_state_dim_first()\n"

INIT_BLOCK = f'''        # {MARK} FlashKDA chunked-prefill arm (task 34 / vLLM #55737)
        if not _glm53_flashkda_supported(
            self.head_dim, vllm_config.model_config.dtype, self.kda_lower_bound
        ):
            raise RuntimeError(
                "{KNOB}=flashkda requires CUDA SM90/SM10x/SM12x, bfloat16, "
                "head_dim=128 and a bounded KDA gate"
            )
        import vllm._flashkda_C  # noqa: F401  {MARK}

        _max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        _max_seqs = vllm_config.scheduler_config.max_num_seqs
        _ws = torch.ops._flashkda_C.get_workspace_size(
            _max_tokens, self.local_num_heads, _max_seqs
        )
        self._flashkda_buffer_specs = (
            (
                (_max_seqs, self.local_num_heads, self.head_dim, self.head_dim),
                self.get_state_dtype()[1],
            ),
            ((_ws,), torch.uint8),
            (
                (1, _max_tokens, self.local_num_heads, self.head_dim),
                vllm_config.model_config.dtype,
            ),
        )
        self._glm53_flashkda_prefill = True
'''

# --- insertion point 3: the non-spec prefill call ------------------------
CALL_ANCHOR = """            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda_with_fused_gate(
                q=_rearr(q_ns),
                k=_rearr(k_ns),
                v=_rearr(v_ns),
                raw_g=g1_ns,
                # Chunk path wants the pre-sigmoided fp32 beta (its kernels
                # don't sigmoid); beta_ns is raw bf16 from forward.
                beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0),
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
            )
"""

FLASHKDA_METHOD = f'''
    def _flashkda_prefill(  {MARK}
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        out=None,
    ):
        """Fused KDA chunked prefill. Takes raw gate logits ``g`` and raw
        ``beta`` logits, l2-normalizes q/k in-kernel and applies the bounded
        gate ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``, matching
        ``chunk_kda_with_fused_gate(..., safe_gate=True)``."""
        final_state, workspace, workspace_out = (
            current_workspace_manager().get_simultaneous(*self._flashkda_buffer_specs)
        )
        final_state = final_state[: initial_state.shape[0]]
        if out is None:
            out = workspace_out[:, : q.shape[1]]
        torch.ops._flashkda_C.fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta,
            self.head_dim ** -0.5,
            out,
            workspace,
            self.A_log.view(-1),
            self.dt_bias.view(-1, self.head_dim),
            self.kda_lower_bound,
            initial_state.contiguous(),
            final_state,
            cu_seqlens.contiguous(),
            None,
            None,
        )
        return out, final_state
'''

CALL_NEW = (
    f"            if self._glm53_flashkda_prefill:  {MARK}\n"
    "                core_attn_out_non_spec, last_recurrent_state = "
    "self._flashkda_prefill(\n"
    "                    q=_rearr(q_ns),\n"
    "                    k=_rearr(k_ns),\n"
    "                    v=_rearr(v_ns),\n"
    "                    g=g1_ns,\n"
    "                    beta=beta_ns,\n"
    "                    initial_state=initial_state,\n"
    "                    cu_seqlens=non_spec_query_start_loc,\n"
    "                    out=None,\n"
    "                )\n"
    "            else:\n"
    + "".join("    " + line + "\n" for line in CALL_ANCHOR.rstrip("\n").split("\n"))
)


def armed() -> bool:
    raw = os.environ.get(KNOB, "").strip().lower()
    if raw in ("", "triton"):
        return False
    if raw == "flashkda":
        return True
    raise SystemExit(f"{KNOB} must be unset, 'triton' or 'flashkda' (got {raw!r})")


def apply_to(source: str) -> str:
    if MARK in source:
        return source
    for anchor, name in (
        (CLASS_ANCHOR, "Glm5NextLinearAttention class"),
        (INIT_ANCHOR, "__init__ conv-state tail"),
        (CALL_ANCHOR, "non-spec chunk_kda_with_fused_gate call"),
    ):
        count = source.count(anchor)
        if count != 1:
            raise SystemExit(
                f"glm53 flashkda: anchor {name!r} matched {count} times, expected 1 — "
                "refusing to patch a drifted kda.py"
            )
    source = source.replace(CLASS_ANCHOR, HELPER_BLOCK + CLASS_ANCHOR, 1)
    source = source.replace(INIT_ANCHOR, INIT_BLOCK + INIT_ANCHOR, 1)
    source = source.replace(CALL_ANCHOR, CALL_NEW + "\n", 1)
    # the method goes at the end of the class body, before the module's tail
    source = source.rstrip("\n") + "\n" + FLASHKDA_METHOD + "\n"
    return source


def main() -> int:
    if not armed():
        print(f"glm53: flashkda prefill off ({KNOB} unset/triton)", file=sys.stderr)
        return 0
    if not TARGET.is_file():
        raise SystemExit(f"glm53 flashkda: missing {TARGET}")
    original = TARGET.read_text(encoding="utf-8")
    patched = apply_to(original)
    if patched == original:
        print("glm53: flashkda prefill already installed", file=sys.stderr)
        return 0
    compile(patched, str(TARGET), "exec")
    TARGET.write_text(patched, encoding="utf-8")
    cache = TARGET.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob("kda*.pyc"):
            pyc.unlink(missing_ok=True)
    print("glm53: flashkda chunked prefill installed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
