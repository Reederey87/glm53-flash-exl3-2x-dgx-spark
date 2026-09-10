#!/usr/bin/env python3
"""CPU tests for the task 39 persistent-top-k reachability audit."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_persistent_topk_reachability.py"
SPEC = importlib.util.spec_from_file_location("ptopk_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

GLM_ATTENTION = """\
from vllm.model_executor.layers.sparse_attn_indexer_kpool import SparseAttnIndexerKpool


class Glm5NextAttention:
    def __init__(self):
        self.indexer_op = SparseAttnIndexerKpool()
"""

KPOOL_DEAD = """\
def _decode_topk(logits, select_k, topk_dst, seq_lens, next_n, num_rows):
    if False and current_platform.is_cuda() and select_k in (512, 1024, 2048):
        workspace_manager = current_workspace_manager()
        (topk_workspace,) = workspace_manager.get_simultaneous(
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )
        torch.ops._C.persistent_topk(
            logits,
            seq_lens,
            topk_dst,
            topk_workspace,
            select_k,
            max_seq_len,
        )
    else:
        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            seq_lens,
            topk_dst,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            select_k,
        )
"""

KPOOL_LIVE = KPOOL_DEAD.replace(
    "if False and current_platform.is_cuda()",
    "if current_platform.is_cuda()",
)

KPOOL_NO_TOPK = """\
def _decode_topk(logits):
    return logits
"""

DEEPSEEK_ATTENTION = """\
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer


class DeepseekV4Attention:
    def __init__(self):
        self.indexer_op = SparseAttnIndexer()
"""

OVERLAY = """\
KPOOL_OLD = "if current_platform.is_cuda() and select_k in (512, 1024, 2048):"
KPOOL_NEW = (
    "if False and current_platform.is_cuda() and "
    "select_k in (512, 1024, 2048):  # GB10 persistent_topk smem"
)


def _disable_gb10_persistent_topk() -> None:
    text = KPOOL.read_text()
    if KPOOL_NEW in text:
        pass
    elif KPOOL_OLD in text:
        KPOOL.write_text(text.replace(KPOOL_OLD, KPOOL_NEW, 1))
    else:
        raise RuntimeError(
            "glm53: kpool persistent_topk pattern not found — patch the file by hand"
        )
"""

OVERLAY_NO_GUARD = OVERLAY.replace(
    "kpool persistent_topk pattern not found", "boom"
)


def build_site(tmp_path: Path, *, kpool: str = KPOOL_DEAD) -> Path:
    site = tmp_path / "vllm"
    (site / "models/glm5next/nvidia").mkdir(parents=True)
    (site / "models/deepseek_v4").mkdir(parents=True)
    (site / "model_executor/layers").mkdir(parents=True)
    (site / "models/glm5next/nvidia/attention.py").write_text(GLM_ATTENTION)
    (site / "models/deepseek_v4/attention.py").write_text(DEEPSEEK_ATTENTION)
    (site / "model_executor/layers/sparse_attn_indexer_kpool.py").write_text(kpool)
    return site


def build_overlay(tmp_path: Path, *, source: str = OVERLAY) -> Path:
    overlay = tmp_path / "overlay"
    overlay.mkdir(exist_ok=True)
    (overlay / MODULE.OVERLAY_NAME).write_text(source)
    return overlay


def test_dead_branch_is_found():
    branches = MODULE.dead_persistent_topk_branches(KPOOL_DEAD, "kpool")
    assert len(branches) == 1
    assert branches[0]["persistent_topk_calls"] == [7]
    assert branches[0]["live_kernel_calls"] == [16]


def test_live_branch_is_not_reported_as_dead():
    assert MODULE.dead_persistent_topk_branches(KPOOL_LIVE, "kpool") == []


def test_live_persistent_topk_detects_re_enabled_call():
    assert MODULE.live_persistent_topk(KPOOL_DEAD, "kpool") == []
    assert MODULE.live_persistent_topk(KPOOL_LIVE, "kpool") == [7]


def test_verdict_not_applicable(tmp_path):
    site = build_site(tmp_path)
    report = MODULE.audit(site, build_overlay(tmp_path), None)
    assert report["verdict"] == "NOT_APPLICABLE"
    assert report["live_persistent_topk_lines"] == []
    assert report["glm_indexer"]["uses_plain_indexer"] is False
    assert report["overlay_guard"]["fail_closed"] is True


def test_verdict_reachable_when_guard_is_gone(tmp_path):
    site = build_site(tmp_path, kpool=KPOOL_LIVE)
    report = MODULE.audit(site, build_overlay(tmp_path), None)
    assert report["verdict"] == "REACHABLE"
    assert report["live_persistent_topk_lines"] == [7]


def test_abort_when_no_persistent_topk_at_all(tmp_path):
    site = build_site(tmp_path, kpool=KPOOL_NO_TOPK)
    report = MODULE.audit(site, build_overlay(tmp_path), None)
    assert report["verdict"] == "ABORT"


def test_abort_when_glm_no_longer_uses_the_kpool_indexer(tmp_path):
    site = build_site(tmp_path)
    (site / "models/glm5next/nvidia/attention.py").write_text(
        GLM_ATTENTION.replace("SparseAttnIndexerKpool", "SomethingElse")
    )
    with pytest.raises(MODULE.Abort):
        MODULE.audit(site, build_overlay(tmp_path), None)


def test_plain_indexer_prefix_is_not_confused_with_kpool():
    glm = MODULE.glm_uses_kpool(GLM_ATTENTION)
    assert glm["imports_kpool_class"] is True
    assert glm["uses_plain_indexer"] is False


def test_plain_indexer_users_excludes_kpool_models(tmp_path):
    site = build_site(tmp_path)
    users = MODULE.plain_indexer_users(site)
    assert users == ["models/deepseek_v4/attention.py"]


def test_overlay_without_fail_closed_guard_aborts(tmp_path):
    site = build_site(tmp_path)
    overlay = build_overlay(tmp_path, source=OVERLAY_NO_GUARD)
    with pytest.raises(MODULE.Abort):
        MODULE.audit(site, overlay, None)


def test_missing_overlay_aborts(tmp_path):
    site = build_site(tmp_path)
    with pytest.raises(MODULE.Abort):
        MODULE.audit(site, tmp_path / "absent", None)


def test_missing_kpool_source_aborts(tmp_path):
    site = build_site(tmp_path)
    (site / MODULE.KPOOL_REL).unlink()
    with pytest.raises(MODULE.Abort):
        MODULE.audit(site, build_overlay(tmp_path), None)


def test_unparseable_kpool_source_aborts(tmp_path):
    site = build_site(tmp_path, kpool="def broken(:\n")
    with pytest.raises(MODULE.Abort):
        MODULE.audit(site, build_overlay(tmp_path), None)


def test_cli_exit_codes(tmp_path):
    site = build_site(tmp_path)
    overlay = build_overlay(tmp_path)

    ok = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--vllm-site",
            str(site),
            "--overlay-dir",
            str(overlay),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["verdict"] == "NOT_APPLICABLE"

    reachable_site = build_site(tmp_path / "live", kpool=KPOOL_LIVE)
    reachable = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--vllm-site",
            str(reachable_site),
            "--overlay-dir",
            str(overlay),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert reachable.returncode == 2
    assert json.loads(reachable.stdout)["verdict"] == "REACHABLE"

    missing = subprocess.run(
        [sys.executable, str(SCRIPT), "--vllm-site", str(tmp_path / "nope")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 1
    assert json.loads(missing.stdout)["verdict"] == "ABORT"
