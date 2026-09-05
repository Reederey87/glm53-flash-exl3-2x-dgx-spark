#!/usr/bin/env python3
"""CPU-only tests for the GLM-5.3 W28 indexer-workspace overlay."""
from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PINNED_FIXTURE_ROOT = HERE / "fixtures" / "w28-pinned-preview"
PATCH = next(
    path
    for path in (
        HERE / "patch_indexer_workspace.py",
        ROOT / "overlay" / "patch_indexer_workspace.py",
    )
    if path.is_file()
)
sys.path.insert(0, str(PATCH.parent))
import patch_indexer_workspace as P  # noqa: E402


LIVE_MAX_MODEL_LEN = 1_000_000
LIVE_KPOOL = 4
LIVE_MAX_NUM_SEQS = 4
LIVE_SPEC = 7
MAX_LOGITS_BYTES = 512 * 1024 * 1024
RADIX_BYTES = 1024 * 1024


class Ns:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def cfg(
    *,
    max_model_len: int = LIVE_MAX_MODEL_LEN,
    max_num_seqs: int = LIVE_MAX_NUM_SEQS,
    spec: int | None = LIVE_SPEC,
    mnbt: int = 3584,
):
    return Ns(
        model_config=Ns(max_model_len=max_model_len),
        scheduler_config=Ns(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=mnbt,
        ),
        speculative_config=(
            Ns(num_speculative_tokens=spec) if spec is not None else None
        ),
    )


def set_mode(value: str | None) -> None:
    if value is None:
        os.environ.pop(P.ENV_NAME, None)
    else:
        os.environ[P.ENV_NAME] = value


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def test_mode_is_literal_and_fail_closed() -> None:
    saved = os.environ.get(P.ENV_NAME)
    try:
        set_mode(None)
        assert P.workspace_mode() == "stock"
        for good in ("stock", "rightsize"):
            set_mode(good)
            assert P.workspace_mode() == good
        for bad in ("", "RIGHTSIZE", " rightsize ", "1", "on"):
            set_mode(bad)
            try:
                P.workspace_mode()
            except ValueError as exc:
                assert P.ENV_NAME in str(exc)
                assert repr(bad) in str(exc)
            else:
                raise AssertionError(f"{bad!r} must fail")
    finally:
        set_mode(saved)


def test_stock_receipt_and_rightsize_formula() -> None:
    saved = os.environ.get(P.ENV_NAME)
    try:
        stock = LIVE_MAX_MODEL_LEN * P.STOCK_MULTIPLIER
        total_bytes = stock * P.BYTES_PER_ENTRY + RADIX_BYTES
        assert stock == 40_000_000
        assert total_bytes == 5_281_048_576
        assert f"{total_bytes / 1024**2:.2f}" == "5036.40"

        for mode in (None, "stock"):
            set_mode(mode)
            assert P.glm5next_workspace_entries(cfg(), LIVE_KPOOL) == stock

        set_mode("rightsize")
        per_req = cdiv(LIVE_MAX_MODEL_LEN + LIVE_SPEC, LIVE_KPOOL)
        assert per_req == 250_002
        assert (
            P.glm5next_workspace_entries(cfg(), LIVE_KPOOL)
            == per_req * LIVE_MAX_NUM_SEQS
            == 1_000_008
        )
        assert (
            P.glm5next_workspace_entries(
                cfg(max_num_seqs=16),
                LIVE_KPOOL,
            )
            == 4_000_032
        )
    finally:
        set_mode(saved)


def test_formula_does_not_depend_on_mnbt() -> None:
    """Codex finding: do not use the unproven MNBT request multiplier."""
    saved = os.environ.get(P.ENV_NAME)
    try:
        set_mode("rightsize")
        values = {
            P.glm5next_workspace_entries(cfg(mnbt=mnbt), LIVE_KPOOL)
            for mnbt in (1, 4, 64, 3584, 8192)
        }
        assert values == {1_000_008}
        assert "max_num_batched_tokens" not in P.HELPERS_SRC
    finally:
        set_mode(saved)


def test_rightsize_must_be_glm_kpool_and_must_reclaim() -> None:
    saved = os.environ.get(P.ENV_NAME)
    try:
        set_mode("rightsize")
        for ratio in (0, 1):
            try:
                P.glm5next_workspace_entries(cfg(), ratio)
            except ValueError as exc:
                assert "index_kpool > 1" in str(exc)
            else:
                raise AssertionError("non-kpool rightsize must fail")

        # 160 * cdiv(1_000_007, 4) exceeds the stock 40M entries.
        try:
            P.glm5next_workspace_entries(
                cfg(max_num_seqs=160),
                LIVE_KPOOL,
            )
        except ValueError as exc:
            assert "does not narrow" in str(exc)
        else:
            raise AssertionError("a no-reclaim arm must fail readiness")
    finally:
        set_mode(saved)


def test_splitter_rejects_a_row_wider_than_workspace() -> None:
    try:
        P.split_prefill_chunks(
            [250_003],
            [1],
            250_002,
            MAX_LOGITS_BYTES,
        )
    except ValueError as exc:
        assert "row exceeds workspace" in str(exc)
    else:
        raise AssertionError("the stock end==start fail-open must be closed")


def legal_batches(seed: int = 20260904, count: int = 400):
    rng = random.Random(seed)
    for _ in range(count):
        count_reqs = rng.randint(1, LIVE_MAX_NUM_SEQS)
        query_lens = [rng.randint(1, 896) for _ in range(count_reqs)]
        seq_lens = [
            rng.randint(query_len, LIVE_MAX_MODEL_LEN) // LIVE_KPOOL
            for query_len in query_lens
        ]
        yield seq_lens, query_lens
    yield [LIVE_MAX_MODEL_LEN // LIVE_KPOOL] * LIVE_MAX_NUM_SEQS, [1] * 4
    yield [LIVE_MAX_MODEL_LEN // LIVE_KPOOL], [3584]


def test_rightsize_keeps_legal_chunking_identical_to_stock() -> None:
    saved = os.environ.get(P.ENV_NAME)
    try:
        set_mode("rightsize")
        rightsize = P.glm5next_workspace_entries(cfg(), LIVE_KPOOL)
        stock = LIVE_MAX_MODEL_LEN * P.STOCK_MULTIPLIER
        assert rightsize < stock
        for seq_lens, query_lens in legal_batches():
            assert P.split_prefill_chunks(
                seq_lens,
                query_lens,
                rightsize,
                MAX_LOGITS_BYTES,
            ) == P.split_prefill_chunks(
                seq_lens,
                query_lens,
                stock,
                MAX_LOGITS_BYTES,
            )
    finally:
        set_mode(saved)


PINNED_INDEXER = (
    """# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig


class Logger:
    def info(self, *args):
        return None


logger = Logger()

def split():
    while True:
"""
    + P.ANCHOR_SPLITTER
    + "\n"
    + P.ANCHOR_HELPERS
    + """

class Builder:
    def build(self):
        for metadata in (None,):
            if metadata is not None:
"""
    + P.ANCHOR_CHUNK_ASSERT
)

PINNED_GLM = (
    """class Indexer:
    def __init__(self, vllm_config):
        self.index_kpool = 4
"""
    + P.ANCHOR_GLM_IMPORT.replace("        ", "        ", 1)
)

PINNED_KPOOL = """from __future__ import annotations


class Buffer:
    def __init__(self, size):
        self.size = size

    def __getitem__(self, item):
        return range(min(item.stop, self.size))


def gather(prefill_metadata, short_prefill, total_seq_lens, index_kpool):
    k_quant_full = Buffer(total_seq_lens)
    k_scale_full = Buffer(total_seq_lens)
    if True:
        for chunk in prefill_metadata.chunks if not short_prefill else ():
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
        return len(k_quant), len(k_scale)
"""


def run_patch(
    indexer: Path,
    glm: Path,
    kpool: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_INDEXER_BACKEND_PY"] = str(indexer)
    env["GLM53_GLM5NEXT_ATTENTION_PY"] = str(glm)
    env["GLM53_INDEXER_KPOOL_PY"] = str(kpool)
    return subprocess.run(
        [sys.executable, str(PATCH), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def write_fixture(tmp: Path) -> tuple[Path, Path, Path]:
    indexer = tmp / "indexer.py"
    glm = tmp / "attention.py"
    kpool = tmp / "sparse_attn_indexer_kpool.py"
    indexer.write_text(PINNED_INDEXER)
    glm.write_text(PINNED_GLM)
    kpool.write_text(PINNED_KPOOL)
    return indexer, glm, kpool


def test_apply_preflight_idempotence_and_glm_scope() -> None:
    with tempfile.TemporaryDirectory() as raw:
        indexer, glm, kpool = write_fixture(Path(raw))
        before_indexer = indexer.read_text()
        before_glm = glm.read_text()
        before_kpool = kpool.read_text()

        preflight = run_patch(indexer, glm, kpool, "--preflight")
        assert preflight.returncode == 0, preflight.stderr
        assert indexer.read_text() == before_indexer
        assert glm.read_text() == before_glm
        assert kpool.read_text() == before_kpool

        first = run_patch(indexer, glm, kpool)
        assert first.returncode == 0, first.stderr
        patched_indexer = indexer.read_text()
        patched_glm = glm.read_text()
        patched_kpool = kpool.read_text()
        assert P._verified(patched_indexer, P.INDEXER_SITES)
        assert P._verified(
            patched_glm,
            ((
                "GLM call site",
                P.MARK_GLM_IMPORT,
                P.ANCHOR_GLM_IMPORT,
                P.PATCHED_GLM_IMPORT,
            ),),
        )
        assert P._verified(
            patched_kpool,
            ((
                "actual allocation guard",
                P.MARK_KPOOL_CAP,
                P.ANCHOR_KPOOL_CAP,
                P.PATCHED_KPOOL_CAP,
            ),),
        )
        # The generic helper remains stock. Only the GLM call site opts in.
        assert P.MARK_IMPORT in patched_indexer
        assert P.ANCHOR_HELPERS in patched_indexer
        assert "_glm53_glm5next_workspace_entries(" in patched_glm
        assert "role=target rank=%d mode=rightsize" in patched_indexer
        assert "deepseek_v4" not in patched_glm

        second = run_patch(indexer, glm, kpool)
        assert second.returncode == 0, second.stderr
        assert "already present" in second.stdout
        assert indexer.read_text() == patched_indexer
        assert glm.read_text() == patched_glm
        assert kpool.read_text() == patched_kpool


def test_runtime_operator_uses_actual_glm_allocation_bound() -> None:
    with tempfile.TemporaryDirectory() as raw:
        indexer, glm, kpool = write_fixture(Path(raw))
        result = run_patch(indexer, glm, kpool)
        assert result.returncode == 0, result.stderr
        namespace: dict = {}
        exec(compile(kpool.read_text(), str(kpool), "exec"), namespace)
        gather = namespace["gather"]

        saved = os.environ.get(P.ENV_NAME)
        try:
            set_mode("rightsize")
            rightsize = P.glm5next_workspace_entries(cfg(), LIVE_KPOOL)
            stock = LIVE_MAX_MODEL_LEN * P.STOCK_MULTIPLIER
            assert gather(
                Ns(chunks=[Ns(total_seq_lens=rightsize)]),
                False,
                rightsize,
                LIVE_KPOOL,
            ) == (rightsize, rightsize)
            try:
                gather(
                    Ns(chunks=[Ns(total_seq_lens=rightsize + 1)]),
                    False,
                    rightsize,
                    LIVE_KPOOL,
                )
            except ValueError as exc:
                assert "gather chunk exceeds allocated workspace" in str(exc)
            else:
                raise AssertionError("rightsize + 1 must fail before gather slicing")

            # The same chunk remains legal under the stock GLM allocation.
            assert gather(
                Ns(chunks=[Ns(total_seq_lens=rightsize + 1)]),
                False,
                stock,
                LIVE_KPOOL,
            ) == (rightsize + 1, rightsize + 1)

            # Non-kpool callers retain the pinned source behavior.
            assert gather(
                Ns(chunks=[Ns(total_seq_lens=9)]),
                False,
                8,
                1,
            ) == (8, 8)
        finally:
            set_mode(saved)


def test_each_anchor_drift_fails_without_writing_any_file() -> None:
    drifts = (
        (
            "indexer",
            "from dataclasses import dataclass\n\nimport torch",
            "from dataclasses import dataclass\n\nimport sys\nimport torch",
        ),
        ("indexer", "return max_model_len * 40", "return max_model_len * 48"),
        ("indexer", "if end == start:", "if end <= start:"),
        ("indexer", "chunks.append(metadata)", "chunks += [metadata]"),
        (
            "glm",
            "self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)",
            "self.max_total_seq_len = 1",
        ),
        (
            "kpool",
            "k_quant = k_quant_full[: chunk.total_seq_lens]",
            "k_quant = k_quant_full",
        ),
    )
    for label, old, new in drifts:
        with tempfile.TemporaryDirectory() as raw:
            indexer, glm, kpool = write_fixture(Path(raw))
            targets = {"indexer": indexer, "glm": glm, "kpool": kpool}
            target = targets[label]
            changed = target.read_text().replace(old, new, 1)
            assert changed != target.read_text()
            target.write_text(changed)
            before_indexer = indexer.read_text()
            before_glm = glm.read_text()
            before_kpool = kpool.read_text()
            result = run_patch(indexer, glm, kpool)
            assert result.returncode != 0, (old, result.stdout, result.stderr)
            assert "preflight failed" in result.stderr
            assert indexer.read_text() == before_indexer
            assert glm.read_text() == before_glm
            assert kpool.read_text() == before_kpool


def test_pinned_container_sources_apply() -> None:
    indexer_src = Path(
        os.environ.get(
            "GLM53_INDEXER_BACKEND_PY_SRC",
            PINNED_FIXTURE_ROOT / "indexer.py",
        )
    )
    glm_src = Path(
        os.environ.get(
            "GLM53_GLM5NEXT_ATTENTION_PY_SRC",
            PINNED_FIXTURE_ROOT / "glm-attention.py",
        )
    )
    kpool_src = Path(
        os.environ.get(
            "GLM53_INDEXER_KPOOL_PY_SRC",
            PINNED_FIXTURE_ROOT / "sparse-attn-indexer-kpool.py",
        )
    )
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        indexer = tmp / "indexer.py"
        glm = tmp / "attention.py"
        kpool = tmp / "sparse_attn_indexer_kpool.py"
        indexer.write_text(indexer_src.read_text())
        glm.write_text(glm_src.read_text())
        kpool.write_text(kpool_src.read_text())
        result = run_patch(indexer, glm, kpool)
        assert result.returncode == 0, result.stderr
        assert P.MARK_IMPORT in indexer.read_text()
        compile(indexer.read_text(), str(indexer), "exec")
        compile(glm.read_text(), str(glm), "exec")
        compile(kpool.read_text(), str(kpool), "exec")


def test_recipe_wiring() -> None:
    start = (ROOT / "start.sh").read_text()
    env_example = (ROOT / "env.example").read_text()
    assert '_glm53_cli_indexer_workspace_set="${GLM53_INDEXER_WORKSPACE+a}"' in start
    assert 'GLM53_INDEXER_WORKSPACE="${GLM53_INDEXER_WORKSPACE-stock}"' in start
    assert "GLM53_INDEXER_WORKSPACE must be exactly one of: stock rightsize" in start
    assert start.count("python3 -S /opt/glm53/patch_indexer_workspace.py") == 2
    assert start.count(
        "-v \"$INDEXER_WORKSPACE_PATCH_HOST:"
        "/opt/glm53/patch_indexer_workspace.py:ro\""
    ) == 1
    assert "GLM53_INDEXER_WORKSPACE=stock" in env_example


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"indexer workspace tests: PASS ({len(tests)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
