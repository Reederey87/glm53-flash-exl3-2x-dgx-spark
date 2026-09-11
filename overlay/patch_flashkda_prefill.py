#!/usr/bin/env python3
"""Task 34 arm: FlashKDA fused CUDA chunked prefill for GLM-5.3-Flash (vLLM #55737).

Prefill is this deployment's measured weak lane (240k cold ~1408 tok/s, 60k
~1454). #55737 replaces the ~15-kernel Triton ``chunk_kda_with_fused_gate``
path with ``vllm._flashkda_C`` and reports 1.7-3.8x on the KDA layer and TTFT
-7.9% to -13.2%. That extension is present in the deployed image and its
workspace allocator works (``torch.ops._flashkda_C.get_workspace_size(1792,
32, 4)`` -> 51,314,688 B on this GB10), and the auto-selection gate in the
upstream patch accepts SM12x + bf16 + head_dim 128 + a bounded gate -- all
true here.

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

DO NOT ARM. The task 34 numeric parity gate failed; see
``docs/15-flashkda-prefill-arm.md`` and
``local/task34-parity-gate-20260910.txt``. Two independent reasons:

1. *Arity.* This port's first cut passed **16** positional arguments to
   ``torch.ops._flashkda_C.fwd``. The deployed op declares **14**
   (``q, k, v, g, beta, scale, out, workspace, A_log, dt_bias, lower_bound,
   initial_state, final_state, cu_seqlens``), so arming raised
   ``RuntimeError: expected at most 14 argument(s) but received 16`` at the
   first prefill -- a hard failure, not a silent one. Fixed here, and
   ``EXPECTED_FWD_ARITY`` now pins the emitted call.

2. *Numerics.* With the arity corrected, ``_flashkda_C.fwd`` does not
   reproduce the Triton chunk path. Against production's
   ``vllm.third_party`` kernel its output is ~150x smaller in mean magnitude
   and essentially uncorrelated (Pearson ~0.0), and the same holds against
   the fork's own ``kimi_k3`` copy. The divergence survives sequence lengths
   64/512, a zeroed gate, both beta conventions, pre-normalized q/k, and
   ``lower_bound`` in {-1,-2,-3,-5}, and the ratio does not track
   ``exp(lower_bound)``. Root cause not isolated; the fork's only sanctioned
   caller (``kimi_k3/nvidia/kda.py``) pairs the fused kernel with a
   *different* Triton kernel (``raw_beta``, no ``safe_gate``), a 1-D
   ``A_log``, and workspace-manager buffers, so this port is not equivalent
   to it.

Consequence: no speed comparison is meaningful until parity passes, so the
arm stays unwired on the cluster and the task 34 A/B window was not run.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

MARK = "# [glm53-flashkda-prefill]"

CLASS_NAME = "Glm5NextLinearAttention"
METHOD_NAME = "_flashkda_prefill"

TARGET = Path(
    os.environ.get(
        "GLM53_GLM_KDA_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/"
        "nvidia/kda.py",
    )
)

KNOB = "GLM53_KDA_PREFILL_BACKEND"

# The deployed op's declaration, verbatim from the v1.4.7 image:
#   _flashkda_C::fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta,
#                    float scale, Tensor(a!) out, Tensor(c!) workspace,
#                    Tensor A_log, Tensor dt_bias, float lower_bound,
#                    Tensor? initial_state=None, Tensor(b!)? final_state=None,
#                    Tensor? cu_seqlens=None) -> ()
# A first cut of this port passed two extra trailing ``None``s and raised
# ``expected at most 14 argument(s) but received 16`` at the first prefill, so
# the count is now pinned rather than assumed.
EXPECTED_FWD_ARITY = 14

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


# Every structure a *fully* installed arm must contain. A bare marker is not
# evidence of installation: the marker is written by several of these lines, so
# a truncated or hand-edited file can carry it while the arm is absent.
REQUIRED_STRUCTURES = (
    ("module helper", "def _glm53_flashkda_supported("),
    ("capability gate call", "if not _glm53_flashkda_supported("),
    ("workspace-manager import", "from vllm.v1.worker.workspace import current_workspace_manager"),
    ("extension import", "import vllm._flashkda_C"),
    ("workspace sizing call", "torch.ops._flashkda_C.get_workspace_size("),
    ("workspace sizing", "self._flashkda_buffer_specs = ("),
    ("arm flag", "self._glm53_flashkda_prefill = True"),
    ("dispatch", "if self._glm53_flashkda_prefill:"),
    ("method definition", f"def {METHOD_NAME}("),
    ("workspace acquisition", "current_workspace_manager().get_simultaneous("),
    ("fused kernel call", "torch.ops._flashkda_C.fwd("),
)


def _class_node(source: str) -> ast.ClassDef:
    """The target class, or fail closed if it is absent or ambiguous."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SystemExit(f"glm53 flashkda: target does not parse: {exc}") from exc
    found = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == CLASS_NAME
    ]
    if len(found) != 1:
        raise SystemExit(
            f"glm53 flashkda: expected exactly one module-level class "
            f"{CLASS_NAME!r}, found {len(found)}"
        )
    return found[0]


def _method_is_a_class_member(source: str) -> bool:
    try:
        klass = _class_node(source)
    except SystemExit:
        return False
    return any(
        isinstance(stmt, ast.FunctionDef) and stmt.name == METHOD_NAME
        for stmt in klass.body
    )


def _has_required_imports(source: str) -> bool:
    """Require the arm's imports as *statements*, not as substrings.

    A commented-out `# from vllm.v1.worker.workspace import
    current_workspace_manager` leaves the substring present, so a textual check
    certifies a file that compiles and then raises NameError at prefill time.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    need_from = ("vllm.v1.worker.workspace", "current_workspace_manager")
    need_plain = "vllm._flashkda_C"
    got_from = got_plain = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == need_from[0]:
            if any(alias.name == need_from[1] for alias in node.names):
                got_from = True
        elif isinstance(node, ast.Import):
            if any(alias.name == need_plain for alias in node.names):
                got_plain = True
    return got_from and got_plain


def _fwd_call_arity(source: str) -> int | None:
    """Positional-argument count of the ``torch.ops._flashkda_C.fwd`` call.

    The op is a fixed-arity TorchScript binding: passing the wrong number of
    arguments raises at the first prefill, not at import. Counting them here
    turns that into an install-time failure. Returns None if the call is
    absent, so the caller decides what that means.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if ast.unparse(node.func) == "torch.ops._flashkda_C.fwd":
            return len(node.args)
    return None


def is_complete(source: str) -> bool:
    """True only when every arm structure is present, the required imports are
    real import statements, the method is a real member of the class, and the
    fused call has the deployed op's arity."""
    if any(needle not in source for _, needle in REQUIRED_STRUCTURES):
        return False
    if not _has_required_imports(source):
        return False
    if _fwd_call_arity(source) != EXPECTED_FWD_ARITY:
        return False
    return _method_is_a_class_member(source)


def _insert_method(source: str) -> str:
    """Insert the method at the end of the *target class body*.

    Appending at EOF is not equivalent: a module-level function or class after
    the target class would silently swallow the method as a nested definition,
    leaving the arm compiling but non-functional.
    """
    klass = _class_node(source)
    if not klass.body:
        raise SystemExit(f"glm53 flashkda: class {CLASS_NAME!r} has an empty body")
    last = max(stmt.end_lineno or stmt.lineno for stmt in klass.body)
    lines = source.splitlines(keepends=True)
    if last > len(lines):
        raise SystemExit("glm53 flashkda: class body extends past end of file")
    block = FLASHKDA_METHOD
    if not lines[last - 1].endswith("\n"):
        block = "\n" + block
    lines.insert(last, block)
    return "".join(lines)


def armed() -> bool:
    raw = os.environ.get(KNOB, "").strip().lower()
    if raw in ("", "triton"):
        return False
    if raw == "flashkda":
        return True
    raise SystemExit(f"{KNOB} must be unset, 'triton' or 'flashkda' (got {raw!r})")


def apply_to(source: str) -> str:
    # The template is indented as a class member; wrapping it in a class makes
    # it parseable so its arity can be checked before anything is written.
    template_arity = _fwd_call_arity("class _Probe:\n" + FLASHKDA_METHOD)
    if template_arity != EXPECTED_FWD_ARITY:
        raise SystemExit(
            f"glm53 flashkda: the method template passes {template_arity} "
            f"positional arguments to _flashkda_C.fwd, but the deployed op "
            f"declares {EXPECTED_FWD_ARITY} — refusing to install"
        )
    if MARK in source:
        if is_complete(source):
            return source
        raise SystemExit(
            "glm53 flashkda: the marker is present but the arm is incomplete — "
            "refusing to accept a partial installation as already installed"
        )
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
    patched = source.replace(CLASS_ANCHOR, HELPER_BLOCK + CLASS_ANCHOR, 1)
    patched = patched.replace(INIT_ANCHOR, INIT_BLOCK + INIT_ANCHOR, 1)
    patched = patched.replace(CALL_ANCHOR, CALL_NEW + "\n", 1)
    patched = _insert_method(patched)
    if not is_complete(patched):
        raise SystemExit(
            "glm53 flashkda: the assembled patch failed its own completeness "
            "check — refusing to write"
        )
    compile(patched, "kda.py", "exec")
    return patched


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
