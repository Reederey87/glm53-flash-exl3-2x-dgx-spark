#!/usr/bin/env python3
"""CPU tests for the task 34 FlashKDA chunked-prefill overlay.

The overlay is a *port* onto this fork's ``glm5next/nvidia/kda.py``, so the
tests pin the three insertion points and the fail-closed/idempotence contract.
A trimmed but structurally faithful fixture keeps the suite hermetic; when the
gitignored live-image dump is present the same assertions run against the real
deployed file.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "overlay/patch_flashkda_prefill.py"
SPEC = importlib.util.spec_from_file_location("flashkda_overlay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

LIVE_DUMP = (
    ROOT
    / "tests/fixtures/live-image-vllm/models/glm5next/nvidia/kda.py"
)

FIXTURE = '''\
from vllm.platforms import current_platform


def _cast_sigmoid(x):
    return x.float().sigmoid()


class Glm5NextLinearAttention(GatedDeltaNetAttention):
    def __init__(self, config, vllm_config, prefix=""):
        self.head_dim = 128
        self.local_num_heads = 32
        self.kda_lower_bound = -5.0
        self._conv_state_dim_first = is_conv_state_dim_first()

    def _forward(self):
        ns_out = None
        if attn_metadata_narrowed.num_prefills > 0:
            initial_state = gather_initial_states(
                recurrent_state, non_spec_state_indices_tensor, has_initial_state
            )
            (
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
'''


def patched(source: str = FIXTURE) -> str:
    return MODULE.apply_to(source)


def test_overlay_applies_and_compiles():
    out = patched()
    assert out != FIXTURE
    compile(out, "kda.py", "exec")


def test_marker_makes_it_idempotent():
    once = patched()
    assert MODULE.apply_to(once) == once
    assert once.count(MODULE.MARK) > 0


def test_helper_block_lands_before_the_class():
    out = patched()
    assert out.index("_glm53_flashkda_supported") < out.index(MODULE.CLASS_ANCHOR)


def test_flashkda_method_is_a_class_method():
    tree = ast.parse(patched())
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Glm5NextLinearAttention"
    )
    methods = [n.name for n in klass.body if isinstance(n, ast.FunctionDef)]
    assert "_flashkda_prefill" in methods


def test_init_gains_the_arm_flag_and_buffer_specs():
    out = patched()
    assert "self._glm53_flashkda_prefill = True" in out
    assert "self._flashkda_buffer_specs = (" in out
    assert "get_workspace_size" in out


def test_call_site_becomes_a_dispatch_and_keeps_the_triton_path():
    out = patched()
    assert "if self._glm53_flashkda_prefill:" in out
    assert "self._flashkda_prefill(" in out
    # the original Triton call survives, re-indented into the else branch
    assert "                ) = chunk_kda_with_fused_gate(" in out
    assert "                    lower_bound=lower_bound," in out


def test_gate_matches_the_upstream_selection_predicate():
    out = patched()
    for needle in (
        "capability.major in (9, 10, 12)",
        "head_dim == 128",
        "torch.bfloat16",
        "lower_bound is not None",
    ):
        assert needle in out, needle


@pytest.mark.parametrize(
    "anchor",
    [
        MODULE.CLASS_ANCHOR,
        MODULE.INIT_ANCHOR,
        MODULE.CALL_ANCHOR,
    ],
)
def test_fail_closed_on_each_drifted_anchor(anchor):
    drifted = FIXTURE.replace(anchor, "    pass  # drifted\n")
    assert drifted != FIXTURE
    with pytest.raises(SystemExit):
        MODULE.apply_to(drifted)


def test_fail_closed_on_ambiguous_anchor():
    ambiguous = FIXTURE + "\n\n" + MODULE.CLASS_ANCHOR + "    pass\n"
    with pytest.raises(SystemExit):
        MODULE.apply_to(ambiguous)


def test_unarmed_leaves_the_file_untouched(tmp_path, monkeypatch):
    target = tmp_path / "kda.py"
    target.write_text(FIXTURE)
    monkeypatch.setattr(MODULE, "TARGET", target)
    for value in ("", "triton", "TRITON"):
        monkeypatch.setenv(MODULE.KNOB, value)
        assert MODULE.main() == 0
        assert target.read_text() == FIXTURE, value


def test_invalid_knob_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(MODULE.KNOB, "cuda")
    with pytest.raises(SystemExit):
        MODULE.main()


def test_armed_writes_a_compiling_file(tmp_path, monkeypatch):
    target = tmp_path / "kda.py"
    target.write_text(FIXTURE)
    monkeypatch.setattr(MODULE, "TARGET", target)
    monkeypatch.setenv(MODULE.KNOB, "flashkda")
    assert MODULE.main() == 0
    written = target.read_text()
    compile(written, str(target), "exec")
    assert MODULE.MARK in written
    # second run is a no-op
    assert MODULE.main() == 0
    assert target.read_text() == written


def test_missing_target_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "TARGET", tmp_path / "absent.py")
    monkeypatch.setenv(MODULE.KNOB, "flashkda")
    with pytest.raises(SystemExit):
        MODULE.main()


@pytest.mark.skipif(not LIVE_DUMP.is_file(), reason="live-image dump not present")
def test_applies_to_the_real_deployed_kda_py():
    source = LIVE_DUMP.read_text(encoding="utf-8")
    out = MODULE.apply_to(source)
    compile(out, str(LIVE_DUMP), "exec")
    tree = ast.parse(out)
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Glm5NextLinearAttention"
    )
    methods = [n.name for n in klass.body if isinstance(n, ast.FunctionDef)]
    assert "_flashkda_prefill" in methods
    assert MODULE.apply_to(out) == out
