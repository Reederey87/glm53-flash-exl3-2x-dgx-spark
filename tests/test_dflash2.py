#!/usr/bin/env python3
"""Host-runnable installer tests for the GLM DFlash2 overlay."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_dflash2.py",
        ROOT / "overlay" / "patch_dflash2.py",
    )
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
from patch_dflash2 import (  # noqa: E402
    CAUSAL_NEW,
    CAUSAL_OLD,
    DECODER_ATTR_NEW,
    DECODER_ATTR_OLD,
    DECODER_CALL_NEW,
    DECODER_CALL_OLD,
    MARK,
    MODEL_ATTR_NEW,
    MODEL_ATTR_OLD,
    MODEL_CALL_NEW,
    MODEL_CALL_OLD,
    REGISTRY_ENTRY,
    REGISTRY_NEW,
    REGISTRY_OLD,
    SPEC_NEW,
    SPEC_OLD,
    UTILS_NEW,
    UTILS_OLD,
    prepare_qwen,
    prepare_registry,
    prepare_spec_init,
    prepare_utils,
    require_kit_model,
    require_kit_spec,
)

KIT_MODEL_SRC = next(
    p
    for p in (
        HERE / "dflash2_model.py",
        ROOT / "overlay" / "dflash2_model.py",
    )
    if p.is_file()
)
KIT_SPEC_SRC = next(
    p
    for p in (
        HERE / "dflash2_speculator.py",
        ROOT / "overlay" / "dflash2_speculator.py",
    )
    if p.is_file()
)

PINNED_QWEN = f'''import torch
from torch import nn

class Qwen3Config:
    pass

class WeightsMapper:
    def __init__(self, **kwargs):
        pass

def support_torch_compile(cls):
    return cls

{CAUSAL_OLD}    return bool(override)

@support_torch_compile
class DFlashQwen3DecoderLayer:
    pass

{DECODER_ATTR_OLD}        orig_to_new_substr={{"midlayer.": "layers.0."}},
    )
    def __init__(self):
{DECODER_CALL_OLD}                    None,
                )
            ]
        )

{MODEL_ATTR_OLD}        self.draft_model_config = None
{MODEL_CALL_OLD}
class Qwen3ForCausalLM:
    pass
'''

PINNED_REGISTRY = f'''_SPECULATIVE_DECODING_MODELS = {{
{REGISTRY_OLD}    "DSparkDraftModel": ("vllm.models.deepseek_v4", "DSparkDeepseekV4ForCausalLM"),
}}
'''

PINNED_UTILS = f'''from dataclasses import replace

def load_dflash_model(vllm_config, target_model):
{UTILS_OLD}    return draft_vllm_config
'''

PINNED_SPEC_INIT = f'''def build_speculator(vllm_config, device):
    speculative_config = vllm_config.speculative_config
{SPEC_OLD}    return None
'''


def _run_patch(site: Path, opt: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_VLLM_SITE"] = str(site)
    env["GLM53_OPT"] = str(opt)
    return subprocess.run(
        [sys.executable, str(PATCH)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _layout(raw: Path) -> tuple[Path, Path]:
    site = raw / "vllm"
    opt = raw / "opt"
    (site / "model_executor/models").mkdir(parents=True)
    (site / "v1/worker/gpu/spec_decode/dflash").mkdir(parents=True)
    opt.mkdir()
    (site / "model_executor/models/qwen3_dflash.py").write_text(PINNED_QWEN)
    (site / "model_executor/models/registry.py").write_text(PINNED_REGISTRY)
    (site / "v1/worker/gpu/spec_decode/dflash/utils.py").write_text(PINNED_UTILS)
    (site / "v1/worker/gpu/spec_decode/__init__.py").write_text(PINNED_SPEC_INIT)
    shutil.copyfile(KIT_MODEL_SRC, opt / "dflash2_model.py")
    shutil.copyfile(KIT_SPEC_SRC, opt / "dflash2_speculator.py")
    return site, opt


def test_kit_source_is_glm_named() -> None:
    assert KIT_MODEL_SRC.is_file(), KIT_MODEL_SRC
    assert not (ROOT / "overlay" / "qwen3_dflash2.py").exists()
    text = KIT_MODEL_SRC.read_text()
    require_kit_model(text)
    require_kit_spec(KIT_SPEC_SRC.read_text())
    assert MARK in text
    assert "dflash2_grouped_conv" in text
    assert "direct_register_custom_op" in text
    assert "torch.topk" in text
    assert ".get_top_k_tokens(" not in text
    overlay_old = ROOT / "overlay" / "qwen3_dflash2.py"
    if overlay_old.parent.is_dir():
        assert not overlay_old.exists()
    dockerfile = ROOT / "Dockerfile"
    if dockerfile.is_file():
        image = dockerfile.read_text()
        assert "COPY overlay/dflash2_model.py /opt/glm53/dflash2_model.py" in image
        assert "COPY overlay/qwen3_dflash2.py" not in image
        assert "RUN python3 /opt/glm53/patch_dflash2.py" in image


def test_fixture_installs_to_registry_path() -> None:
    with tempfile.TemporaryDirectory() as raw:
        site, opt = _layout(Path(raw))
        first = _run_patch(site, opt)
        assert first.returncode == 0, first.stderr + first.stdout
        dst = site / "model_executor/models/qwen3_dflash2.py"
        assert dst.is_file()
        installed = dst.read_text()
        assert installed == (opt / "dflash2_model.py").read_text()
        assert "dflash2_grouped_conv" in installed
        assert "_dflash2_grouped_conv_kernel" in installed
        assert "class DFlash2Qwen3ForCausalLM" in installed
        spec = (site / "v1/worker/gpu/spec_decode/dflash2/speculator.py").read_text()
        assert "class DFlash2Speculator" in spec
        assert (site / "v1/worker/gpu/spec_decode/dflash2/__init__.py").is_file()
        qwen = (site / "model_executor/models/qwen3_dflash.py").read_text()
        assert 'getattr(config, "is_causal", None)' in qwen
        assert "self.decoder_layer_cls(" in qwen
        assert "DFlashQwen3DecoderLayer(" not in qwen.split("self.layers")[1]
        registry = (site / "model_executor/models/registry.py").read_text()
        assert REGISTRY_ENTRY in registry
        utils = (site / "v1/worker/gpu/spec_decode/dflash/utils.py").read_text()
        assert 'draft_kv = "auto"' in utils
        assert '"fp8_ds_mla"' in utils
        spec_init = (site / "v1/worker/gpu/spec_decode/__init__.py").read_text()
        assert "DFlash2Speculator" in spec_init
        assert "DFlash2DraftModel" in spec_init
        second = _run_patch(site, opt)
        assert second.returncode == 0, second.stderr + second.stdout
        assert dst.read_text() == installed
        assert "already present" in second.stdout


def test_fail_closed_on_anchor_drift() -> None:
    with tempfile.TemporaryDirectory() as raw:
        site, opt = _layout(Path(raw))
        qwen = site / "model_executor/models/qwen3_dflash.py"
        qwen.write_text(PINNED_QWEN.replace("dflash_config.causal", "causal_flag", 1))
        result = _run_patch(site, opt)
        assert result.returncode != 0
        assert "preflight failed" in result.stderr
        assert "causal_flag" in qwen.read_text()
        assert not (site / "model_executor/models/qwen3_dflash2.py").exists()


def test_prepare_helpers_are_idempotent() -> None:
    qwen_patched, action = prepare_qwen(PINNED_QWEN)
    assert action == "patched"
    again, again_action = prepare_qwen(qwen_patched)
    assert again_action == "already present"
    assert again == qwen_patched
    assert CAUSAL_NEW in qwen_patched
    assert DECODER_ATTR_NEW in qwen_patched
    assert DECODER_CALL_NEW in qwen_patched
    assert MODEL_ATTR_NEW in qwen_patched
    assert MODEL_CALL_NEW in qwen_patched

    registry_patched, _ = prepare_registry(PINNED_REGISTRY)
    assert REGISTRY_NEW in registry_patched
    again_registry, again_registry_action = prepare_registry(registry_patched)
    assert again_registry_action == "already present"
    assert again_registry == registry_patched
    utils_patched, _ = prepare_utils(PINNED_UTILS)
    assert UTILS_NEW in utils_patched
    spec_patched, _ = prepare_spec_init(PINNED_SPEC_INIT)
    assert SPEC_NEW in spec_patched


def test_registry_rejects_conflicting_mapping() -> None:
    conflicting = PINNED_REGISTRY.replace(
        REGISTRY_OLD,
        REGISTRY_OLD
        + '    "DFlash2DraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n',
        1,
    )
    try:
        prepare_registry(conflicting)
    except ValueError as exc:
        assert "conflicting DFlash2DraftModel mapping" in str(exc)
    else:
        raise AssertionError("conflicting mapping must fail closed")

    duplicate = PINNED_REGISTRY.replace(
        REGISTRY_OLD,
        REGISTRY_NEW
        + '    "DFlash2DraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n',
        1,
    )
    try:
        prepare_registry(duplicate)
    except ValueError as exc:
        assert "conflicting DFlash2DraftModel mapping" in str(exc)
    else:
        raise AssertionError("duplicate mapping must fail closed")

    single_quoted = PINNED_REGISTRY.replace(
        REGISTRY_OLD,
        REGISTRY_OLD
        + "    'DFlash2DraftModel': ('qwen3_dflash', 'DFlashQwen3ForCausalLM'),\n",
        1,
    )
    try:
        prepare_registry(single_quoted)
    except ValueError as exc:
        assert "conflicting DFlash2DraftModel mapping" in str(exc)
    else:
        raise AssertionError("single-quoted conflicting mapping must fail closed")

    already_patched_single = PINNED_REGISTRY.replace(
        REGISTRY_OLD,
        REGISTRY_NEW
        + "    'DFlash2DraftModel': ('qwen3_dflash', 'DFlashQwen3ForCausalLM'),\n",
        1,
    )
    try:
        prepare_registry(already_patched_single)
    except ValueError as orig:
        assert "conflicting DFlash2DraftModel mapping" in str(orig)
    else:
        raise AssertionError(
            "already-patched single-quoted duplicate must fail closed"
        )


def test_conflicting_registry_does_not_write_destinations() -> None:
    with tempfile.TemporaryDirectory() as raw:
        site, opt = _layout(Path(raw))
        registry = site / "model_executor/models/registry.py"
        original = registry.read_text()
        registry.write_text(
            original.replace(
                REGISTRY_OLD,
                REGISTRY_OLD
                + '    "DFlash2DraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n',
                1,
            )
        )
        result = _run_patch(site, opt)
        assert result.returncode != 0
        assert "preflight failed" in result.stderr
        assert not (site / "model_executor/models/qwen3_dflash2.py").exists()
        assert not (site / "v1/worker/gpu/spec_decode/dflash2/speculator.py").exists()
        assert '"DFlash2DraftModel": ("qwen3_dflash"' in registry.read_text()

    with tempfile.TemporaryDirectory() as raw:
        site, opt = _layout(Path(raw))
        registry = site / "model_executor/models/registry.py"
        original = registry.read_text()
        registry.write_text(
            original.replace(
                REGISTRY_OLD,
                REGISTRY_OLD
                + "    'DFlash2DraftModel': ('qwen3_dflash', 'DFlashQwen3ForCausalLM'),\n",
                1,
            )
        )
        result = _run_patch(site, opt)
        assert result.returncode != 0
        assert "preflight failed" in result.stderr
        assert not (site / "model_executor/models/qwen3_dflash2.py").exists()
        assert not (site / "v1/worker/gpu/spec_decode/dflash2/speculator.py").exists()
        assert "'DFlash2DraftModel': ('qwen3_dflash'" in registry.read_text()


def test_kit_compile_without_vllm() -> None:
    compile(KIT_MODEL_SRC.read_text(), str(KIT_MODEL_SRC), "exec")
    compile(KIT_SPEC_SRC.read_text(), str(KIT_SPEC_SRC), "exec")
    compile(PATCH.read_text(), str(PATCH), "exec")


def main() -> int:
    test_kit_source_is_glm_named()
    test_fixture_installs_to_registry_path()
    test_fail_closed_on_anchor_drift()
    test_prepare_helpers_are_idempotent()
    test_registry_rejects_conflicting_mapping()
    test_conflicting_registry_does_not_write_destinations()
    test_kit_compile_without_vllm()
    print("dflash2 overlay OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
