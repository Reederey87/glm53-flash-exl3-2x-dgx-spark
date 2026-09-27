"""Compact drafter page: derivation and the flag-off no-op."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "overlay" / "patch_draft_compact_page.py"

# Live boot 2026-09-27: MLA block 3584, MLA page 2,351,104, draft 2048 B/token.
MLA_BLOCK = 3584
MLA_PAGE = 2_351_104
DRAFT_BYTES = 2048


def _load():
    spec = importlib.util.spec_from_file_location("patch_draft_compact_page", PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_geometry_picks_896_when_the_backend_accepts_it_whole():
    mod = _load()
    assert mod.choose_draft_block(
        MLA_BLOCK, MLA_PAGE, DRAFT_BYTES, accepts=lambda _b: True
    ) == 896
    # 896 * 2048 fits; 1792 * 2048 does not; 896 divides 3584.
    assert 896 * DRAFT_BYTES <= MLA_PAGE
    assert MLA_BLOCK % 896 == 0
    assert (896 * 2) * DRAFT_BYTES > MLA_PAGE


def test_backend_that_splits_falls_through_to_the_next_divisor():
    mod = _load()

    def accepts(block: int) -> bool:
        return block <= 512

    assert mod.choose_draft_block(MLA_BLOCK, MLA_PAGE, DRAFT_BYTES, accepts) == 512


def test_nothing_above_the_floor_stays_at_64():
    mod = _load()
    assert (
        mod.choose_draft_block(
            MLA_BLOCK, MLA_PAGE, DRAFT_BYTES, accepts=lambda _b: False
        )
        == 64
    )


def test_a_target_language_model_layer_stays_at_the_floor():
    mod = _load()
    assert (
        mod.choose_draft_block(
            MLA_BLOCK,
            MLA_PAGE,
            DRAFT_BYTES,
            accepts=lambda _b: True,
            names=("model.language_model.layers.0.self_attn",),
        )
        == 64
    )


def test_untagged_drafter_layers_can_grow():
    """get_model() loads the drafter with an empty prefix, so the layer
    names are model.layers.*, not eagle_head.*."""
    mod = _load()
    assert (
        mod.choose_draft_block(
            MLA_BLOCK,
            MLA_PAGE,
            DRAFT_BYTES,
            accepts=lambda _b: True,
            names=tuple(f"model.layers.{i}.self_attn" for i in range(5)),
        )
        == 896
    )


def test_page_too_small_for_128_stays_at_the_floor():
    mod = _load()
    # 128 * 2048 = 262144, so a page of one byte under that cannot grow.
    assert (
        mod.choose_draft_block(
            MLA_BLOCK, 128 * DRAFT_BYTES - 1, DRAFT_BYTES, accepts=lambda _b: True
        )
        == 64
    )


def _kv_fixture() -> str:
    return (
        "def build(draft_specs, mla_block, mla_page, draft_bytes_per_token, logger):\n"
        "            compact_block = 64\n"
        "            return compact_block\n"
        "def validate(draft_inner, mla_page, attn_group, cast, UniformTypeKVCacheSpecs):\n"
        "            if any(\n"
        "                s.block_size != 64 or s.page_size_padded != mla_page\n"
        "                for s in draft_inner.values()\n"
        "            ):\n"
        "                return None\n"
        "            return 1\n"
    )


def _coord_fixture() -> str:
    return (
        "def _glm53_is_draft_swa_spec(spec):\n"
        "    return True\n"
        "def find(self, spec, use_eagle, idx, eagle_verified, first_group_id):\n"
        "                drop_eagle_block = use_eagle and idx not in eagle_verified\n"
        "                return drop_eagle_block\n"
    )


def test_armed_patch_rewrites_both_files_and_is_idempotent(tmp_path, monkeypatch):
    mod = _load()
    kv = tmp_path / "kv.py"
    coord = tmp_path / "coord.py"
    kv.write_text(_kv_fixture())
    coord.write_text(_coord_fixture())
    monkeypatch.setenv("GLM53_DRAFT_COMPACT_PAGE", "1")
    monkeypatch.setenv("GLM53_KV_CACHE_UTILS_PY", str(kv))
    monkeypatch.setenv("GLM53_KV_COORDINATOR_PY", str(coord))
    # Reload paths from the env the module reads at call time.
    mod.KV_FILE = str(kv)
    mod.COORD_FILE = str(coord)
    assert mod.main() == 0
    kv_text = kv.read_text()
    coord_text = coord.read_text()
    assert "compact page chose block=%d" in kv_text
    assert "unpadded_page_size_bytes" in kv_text
    assert "block_size != 64" not in kv_text
    assert "block_size > 64" in coord_text
    compile(kv_text, str(kv), "exec")
    compile(coord_text, str(coord), "exec")
    assert mod.main() == 0
    assert kv.read_text() == kv_text
    assert coord.read_text() == coord_text


def test_flag_off_does_not_read_or_write(tmp_path, monkeypatch):
    mod = _load()
    missing = tmp_path / "nope.py"
    monkeypatch.setenv("GLM53_DRAFT_COMPACT_PAGE", "0")
    monkeypatch.setenv("GLM53_KV_CACHE_UTILS_PY", str(missing))
    monkeypatch.setenv("GLM53_KV_COORDINATOR_PY", str(missing))
    mod.KV_FILE = str(missing)
    mod.COORD_FILE = str(missing)
    assert mod.main() == 0
    assert not missing.exists()


def test_injected_selector_matches_the_pure_function():
    """The string written into kv_cache_utils is the same rule as
    choose_draft_block, including the eagle_head fail-closed."""
    text = PATCH.read_text()
    for needle in (
        '"language_model" in str(n)',
        "mla_block % 64 == 0",
        "_glm53_cand * draft_bytes_per_token > mla_page",
        "_glm53_kernel = _glm53_sel(_glm53_cand, [_GLM53FA])",
        "_glm53_ok = _glm53_kernel == _glm53_cand",
        "range(_glm53_steps, 1, -1)",
        "s.unpadded_page_size_bytes > mla_page",
        "_glm53_mla_block % s.block_size != 0",
        "block_size > 64",
    ):
        assert needle in text, needle
    assert os.environ.get("GLM53_DRAFT_COMPACT_PAGE", "0") != "1"
