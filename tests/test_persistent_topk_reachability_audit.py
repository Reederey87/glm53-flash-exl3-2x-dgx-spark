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


GLM_ATTENTION_PLAIN_CONSTRUCT = """\
from vllm.model_executor.layers.sparse_attn_indexer_kpool import SparseAttnIndexerKpool
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer


class Glm5NextAttention:
    def __init__(self):
        self.indexer_op = SparseAttnIndexer()
"""

GLM_ATTENTION_SHADOWED = """\
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer as Indexer


def unrelated():
    from vllm.model_executor.layers.sparse_attn_indexer_kpool import SparseAttnIndexerKpool as Indexer
    return Indexer


class Glm5NextAttention:
    def __init__(self):
        self.indexer_op = Indexer()
"""

GLM_ATTENTION_MISSING_SYMBOL = """\
from vllm.model_executor.layers.sparse_attn_indexer_kpool import MissingIndexer


class Glm5NextAttention:
    def __init__(self):
        self.indexer_op = MissingIndexer()
"""

GLM_ATTENTION_PLAIN_ALIASED = """\
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer as SparseAttnIndexerKpool


class Glm5NextAttention:
    def __init__(self):
        self.indexer_op = SparseAttnIndexerKpool()
"""

GLM_ATTENTION_KPOOL_ALIASED = """\
from vllm.model_executor.layers.sparse_attn_indexer_kpool import SparseAttnIndexerKpool as Kpool


class Glm5NextAttention:
    def __init__(self):
        self.indexer_op = Kpool()
"""

GLM_ATTENTION_COMMENT_ONLY = """\
# This deployment deliberately avoids SparseAttnIndexerKpool.


class Glm5NextAttention:
    def __init__(self):
        pass
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


def test_abort_when_glm_imports_the_indexer_from_an_unknown_module(tmp_path):
    """Import provenance decides identity; an unknown source is unresolved."""
    site = build_site(tmp_path)
    (site / "models/glm5next/nvidia/attention.py").write_text(
        GLM_ATTENTION.replace(
            "from vllm.model_executor.layers.sparse_attn_indexer_kpool import",
            "from my.own.layers import",
        )
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


# --- finding 3: a live else/elif is not a dead branch ----------------------

KPOOL_LIVE_ELSE = """\
def _decode_topk(logits, select_k, topk_dst, seq_lens):
    if False and current_platform.is_cuda() and select_k in (512, 1024, 2048):
        pass
    else:
        torch.ops._C.persistent_topk(
            logits,
            seq_lens,
            topk_dst,
            select_k,
        )
"""

KPOOL_DEAD_BODY_LIVE_ELSE = """\
def _decode_topk(logits):
    if False and current_platform.is_cuda():
        torch.ops._C.persistent_topk(logits)
    else:
        torch.ops._C.persistent_topk(logits)
"""

KPOOL_DEAD_BODY_LIVE_ELIF = """\
def _decode_topk(logits):
    if False and current_platform.is_cuda():
        torch.ops._C.persistent_topk(logits)
    elif other_platform.is_cuda():
        torch.ops._C.persistent_topk(logits)
"""


def test_live_else_call_is_not_classified_dead():
    """The `else` branch is executable; walking the whole `ast.If` hid that."""
    assert MODULE.dead_persistent_topk_branches(KPOOL_LIVE_ELSE, "kpool") == []
    assert MODULE.live_persistent_topk(KPOOL_LIVE_ELSE, "kpool") == [5]


def test_dead_body_with_a_live_else_keeps_the_else_call():
    assert MODULE.dead_persistent_topk_branches(KPOOL_DEAD_BODY_LIVE_ELSE, "kpool") == [
        {"line": 2, "persistent_topk_calls": [3], "else_branch": True, "live_kernel_calls": []}
    ]
    assert MODULE.live_persistent_topk(KPOOL_DEAD_BODY_LIVE_ELSE, "kpool") == [5]


def test_dead_body_with_a_live_elif_keeps_the_elif_call():
    assert MODULE.live_persistent_topk(KPOOL_DEAD_BODY_LIVE_ELIF, "kpool") == [5]


def test_verdict_reachable_when_only_the_else_branch_calls_topk(tmp_path):
    site = build_site(tmp_path, kpool=KPOOL_LIVE_ELSE)
    report = MODULE.audit(site, build_overlay(tmp_path), None)
    assert report["verdict"] == "REACHABLE", report["reason"]
    assert report["live_persistent_topk_lines"] == [5]


# --- finding 4: parse the GLM module, do not pattern-match it --------------


def test_plain_indexer_construction_is_not_given_the_kpool_verdict():
    """Importing the kpool name is not the same as constructing it."""
    glm = MODULE.glm_uses_kpool(GLM_ATTENTION_PLAIN_CONSTRUCT)
    assert glm["instantiates_kpool_class"] is False
    assert glm["plain_indexer_constructed"] is True
    assert glm["uses_plain_indexer"] is True


def test_comment_only_mention_is_not_an_import_or_a_construction():
    glm = MODULE.glm_uses_kpool(GLM_ATTENTION_COMMENT_ONLY)
    assert glm["imports_kpool_class"] is False
    assert glm["instantiates_kpool_class"] is False
    assert glm["kpool_reference_lines"] == []


def test_abort_when_glm_constructs_the_plain_indexer_instead(tmp_path):
    site = build_site(tmp_path)
    (site / MODULE.GLM_ATTENTION_REL).write_text(GLM_ATTENTION_PLAIN_CONSTRUCT)
    with pytest.raises(MODULE.Abort):
        MODULE.audit(site, build_overlay(tmp_path), None)


def test_abort_when_glm_only_mentions_kpool_in_a_comment(tmp_path):
    site = build_site(tmp_path)
    (site / MODULE.GLM_ATTENTION_REL).write_text(GLM_ATTENTION_COMMENT_ONLY)
    with pytest.raises(MODULE.Abort):
        MODULE.audit(site, build_overlay(tmp_path), None)


# --- finding 3 (second pass): constructor identity is import provenance ----


def test_aliased_plain_import_is_not_given_the_kpool_verdict():
    """The plain indexer imported under the kpool *name* is still the plain one."""
    glm = MODULE.glm_uses_kpool(GLM_ATTENTION_PLAIN_ALIASED)
    assert glm["instantiates_kpool_class"] is False
    assert glm["plain_indexer_constructed"] is True
    assert glm["uses_plain_indexer"] is True


def test_abort_when_the_plain_indexer_is_aliased_to_the_kpool_name(tmp_path):
    site = build_site(tmp_path)
    (site / MODULE.GLM_ATTENTION_REL).write_text(GLM_ATTENTION_PLAIN_ALIASED)
    with pytest.raises(MODULE.Abort):
        MODULE.audit(site, build_overlay(tmp_path), None)


def test_kpool_import_under_an_arbitrary_alias_still_resolves_to_kpool():
    glm = MODULE.glm_uses_kpool(GLM_ATTENTION_KPOOL_ALIASED)
    assert glm["instantiates_kpool_class"] is True
    assert glm["uses_plain_indexer"] is False


def test_import_provenance_beats_the_local_spelling():
    """The same local name resolves differently depending on its source."""
    plain = MODULE.glm_uses_kpool(GLM_ATTENTION_PLAIN_ALIASED)
    kpool = MODULE.glm_uses_kpool(GLM_ATTENTION_KPOOL_ALIASED)
    assert plain["instantiates_kpool_class"] is not kpool["instantiates_kpool_class"]


# --- finding 3 (third pass): bindings live in scopes, not in a flat dict ---


def test_shadowed_binding_is_unresolved():
    """A function-local kpool import must not overwrite a global plain one."""
    glm = MODULE.glm_uses_kpool(GLM_ATTENTION_SHADOWED)
    assert glm["instantiates_kpool_class"] is False
    assert glm["unresolved_indexer_names"] == ["Indexer"]


def test_missing_exported_symbol_is_unresolved():
    """Importing from the kpool module is not enough; the symbol must exist."""
    glm = MODULE.glm_uses_kpool(GLM_ATTENTION_MISSING_SYMBOL)
    assert glm["instantiates_kpool_class"] is False
    assert glm["unresolved_indexer_names"] == ["MissingIndexer"]


def test_abort_on_a_shadowed_indexer_binding(tmp_path):
    site = build_site(tmp_path)
    (site / MODULE.GLM_ATTENTION_REL).write_text(GLM_ATTENTION_SHADOWED)
    with pytest.raises(MODULE.Abort):
        MODULE.audit(site, build_overlay(tmp_path), None)


def test_abort_on_a_missing_exported_symbol(tmp_path):
    site = build_site(tmp_path)
    (site / MODULE.GLM_ATTENTION_REL).write_text(GLM_ATTENTION_MISSING_SYMBOL)
    with pytest.raises(MODULE.Abort):
        MODULE.audit(site, build_overlay(tmp_path), None)
