from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "overlay" / "patch_glm5_drafter_group.py"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("patch_glm5_drafter_group", PATCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_padding_waste_arithmetic() -> None:
    mla_page = 2_351_104
    draft_bytes_per_token = 2_048
    kernel_block = 64

    draft_tokens_per_page = mla_page // draft_bytes_per_token
    usable_draft_tokens = draft_tokens_per_page // kernel_block * kernel_block
    waste = (draft_tokens_per_page - usable_draft_tokens) * draft_bytes_per_token
    exact_fit_page = (
        (draft_tokens_per_page + kernel_block - 1)
        // kernel_block
        * kernel_block
        * draft_bytes_per_token
    )

    assert draft_tokens_per_page == 1_148
    assert usable_draft_tokens == 1_088
    assert waste == 122_880
    assert round(100 * waste / mla_page, 2) == 5.23
    assert exact_fit_page == 2_359_296
    assert exact_fit_page - mla_page == 8_192


def test_fresh_patch_and_legacy_upgrade_include_exact_waste_receipt(tmp_path: Path) -> None:
    module = _load_patch_module()
    source = module.EDIT_GROUPS_RETURN_NEW

    assert "structural_padding_waste_bytes = (" in source
    assert '"structural_padding_waste_bytes=%d (%.2f%% of mla_page) "' in source
    assert '"exact_fit_page=%d growth_bytes=%d (%.2f%%)"' in source

    legacy = source.replace(
        """            compact_block = 64
            kernel_page_bytes = compact_block * draft_bytes_per_token
            draft_tokens_per_mla_page = mla_page // draft_bytes_per_token
            usable_draft_tokens = (
                draft_tokens_per_mla_page // compact_block * compact_block
            )
            structural_padding_waste_bytes = (
                draft_tokens_per_mla_page - usable_draft_tokens
            ) * draft_bytes_per_token
            exact_fit_draft_tokens = (
                (draft_tokens_per_mla_page + compact_block - 1)
                // compact_block
                * compact_block
            )
            exact_fit_page_bytes = exact_fit_draft_tokens * draft_bytes_per_token
            logger.info(
                "DFlash2 drafter KV: padded slot-share block=%d "
                "mla_page=%d (was block=%d); exact-fit page mismatch "
                "draft_bytes/token=%d kernel_page=%d "
                "draft_tokens/page=%d usable_draft_tokens=%d "
                "structural_padding_waste_bytes=%d (%.2f%% of mla_page) "
                "exact_fit_page=%d growth_bytes=%d (%.2f%%)",
                compact_block,
                mla_page,
                any_draft.block_size,
                draft_bytes_per_token,
                kernel_page_bytes,
                draft_tokens_per_mla_page,
                usable_draft_tokens,
                structural_padding_waste_bytes,
                100.0 * structural_padding_waste_bytes / mla_page,
                exact_fit_page_bytes,
                exact_fit_page_bytes - mla_page,
                100.0 * (exact_fit_page_bytes - mla_page) / mla_page,
            )
""",
        """            compact_block = 64
            logger.info(
                "DFlash2 drafter KV: padded slot-share block=%d "
                "mla_page=%d (was block=%d); exact-fit page mismatch "
                "draft_bytes/token=%d",
                compact_block,
                mla_page,
                any_draft.block_size,
                draft_bytes_per_token,
            )
""",
        1,
    )
    target = tmp_path / "kv_cache_utils.py"
    target.write_text("def patched_function():\n" + legacy)

    assert module.patch_file(str(target)) == 0
    upgraded = target.read_text()
    assert "structural_padding_waste_bytes=%d" in upgraded
    assert "exact_fit_page=%d growth_bytes=%d" in upgraded

    before = target.read_bytes()
    assert module.patch_file(str(target)) == 0
    assert target.read_bytes() == before
