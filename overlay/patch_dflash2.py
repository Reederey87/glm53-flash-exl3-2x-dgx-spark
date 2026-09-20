#!/usr/bin/env python3
"""Install DFlash2 onto the glm53-flash vLLM pin (fail-closed, idempotent).

The kit source is ``overlay/dflash2_model.py`` (GLM-5.3-Flash-DFlash2). vLLM's
registry still loads architecture ``DFlash2DraftModel`` from module
``qwen3_dflash2``, so this installer copies the kit file to that destination
and does not rename checkpoint class names (``DFlash2Qwen3*``).

Also patches the pin so DFlash2 can load:

* honor checkpoint ``is_causal`` (incoai GLM DFlash2 ships ``false``)
* restore ``decoder_layer_cls`` / ``model_cls`` indirection (#53449 / #52816)
* register ``DFlash2DraftModel``
* keep draft KV in the model dtype on MLA/FP8 (SM121 has no FA3/FA4 for
  plain FP8 KV and cannot use the target's ``fp8_ds_mla`` layout)
* dispatch ``DFlash2Speculator`` when the draft architecture is DFlash2

Fail closed if a pinned anchor drifts or a partial patch is found. Written
atomically after ``compile()``.
"""
from __future__ import annotations

import ast
import os
import stat
import sys
from pathlib import Path


SITE = Path(
    os.environ.get(
        "GLM53_VLLM_SITE",
        "/usr/local/lib/python3.12/dist-packages/vllm",
    )
)
OPT = Path(os.environ.get("GLM53_OPT", "/opt/glm53"))

KIT_MODEL = OPT / "dflash2_model.py"
KIT_SPEC = OPT / "dflash2_speculator.py"
DST_MODEL = SITE / "model_executor/models/qwen3_dflash2.py"
DST_DIR = SITE / "v1/worker/gpu/spec_decode/dflash2"
DST_SPEC = DST_DIR / "speculator.py"
DST_INIT = DST_DIR / "__init__.py"
QWEN = SITE / "model_executor/models/qwen3_dflash.py"
REGISTRY = SITE / "model_executor/models/registry.py"
DFLASH_UTILS = SITE / "v1/worker/gpu/spec_decode/dflash/utils.py"
SPEC_INIT = SITE / "v1/worker/gpu/spec_decode/__init__.py"

MARK = "# [glm53-dflash2]"

CAUSAL_OLD = (
    "def _dflash_layer_causal(config: Qwen3Config, layer_idx: int) -> bool:\n"
    '    """``dflash_config.causal`` overrides all layers; else only SWA '
    'layers causal."""\n'
    '    override = (getattr(config, "dflash_config", None) or {}).get("causal")\n'
)
CAUSAL_NEW = (
    "def _dflash_layer_causal(config: Qwen3Config, layer_idx: int) -> bool:\n"
    '    """Honor checkpoint ``is_causal`` (incoai DFlash2: false) '
    'before SWA=causal."""\n'
    '    is_causal = getattr(config, "is_causal", None)\n'
    "    if is_causal is not None:\n"
    "        return bool(is_causal)\n"
    '    override = (getattr(config, "dflash_config", None) or {}).get("causal")\n'
)

DECODER_ATTR_OLD = (
    "@support_torch_compile\n"
    "class DFlashQwen3Model(nn.Module):\n"
    "    hf_to_vllm_mapper = WeightsMapper(\n"
)
DECODER_ATTR_NEW = (
    "@support_torch_compile\n"
    "class DFlashQwen3Model(nn.Module):\n"
    "    decoder_layer_cls = DFlashQwen3DecoderLayer\n"
    "    hf_to_vllm_mapper = WeightsMapper(\n"
)

DECODER_CALL_OLD = (
    "        self.layers = nn.ModuleList(\n"
    "            [\n"
    "                DFlashQwen3DecoderLayer(\n"
)
DECODER_CALL_NEW = (
    "        self.layers = nn.ModuleList(\n"
    "            [\n"
    "                self.decoder_layer_cls(\n"
)

MODEL_ATTR_OLD = (
    "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
    '    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):\n'
)
MODEL_ATTR_NEW = (
    "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
    "    model_cls = DFlashQwen3Model\n"
    "\n"
    '    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):\n'
)

MODEL_CALL_OLD = (
    "        self.model = DFlashQwen3Model(\n"
    "            vllm_config=vllm_config,\n"
    '            prefix=maybe_prefix(prefix, "model"),\n'
    "            start_layer_id=target_layer_num,\n"
    "        )\n"
)
MODEL_CALL_NEW = (
    "        self.model = self.model_cls(\n"
    "            vllm_config=vllm_config,\n"
    '            prefix=maybe_prefix(prefix, "model"),\n'
    "            start_layer_id=target_layer_num,\n"
    "        )\n"
)

REGISTRY_OLD = (
    '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n'
)
REGISTRY_NEW = (
    '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n'
    '    "DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),\n'
)
REGISTRY_ENTRY = (
    '    "DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),\n'
)

UTILS_OLD = """    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
"""
UTILS_NEW = """    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Dense DFlash2 attention cannot use the target's MLA-only fp8_ds_mla
    # layout, and SM121 has no FA3/FA4 for plain FP8 KV. Keep draft KV in
    # the model dtype unless speculative_config.kv_cache_dtype is set.
    draft_kv = speculative_config.kv_cache_dtype
    if draft_kv is None and vllm_config.cache_config.cache_dtype in (
        "fp8_ds_mla",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
    ):
        draft_kv = "auto"
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=draft_kv,
            )
            if draft_kv is not None
            else vllm_config.cache_config
        ),
    )
"""

SPEC_OLD = '''    if speculative_config.method == "dflash":
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator(vllm_config, device)
'''
SPEC_NEW = '''    if speculative_config.method == "dflash":
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
                DFlash2Speculator,
            )

            return DFlash2Speculator(vllm_config, device)
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator(vllm_config, device)
'''


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-dflash2.tmp")
    try:
        tmp.write_text(source)
        if target.exists():
            os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def qwen_counts(text: str) -> tuple[list[int], list[int]]:
    old = [
        text.count(x)
        for x in (
            CAUSAL_OLD,
            DECODER_ATTR_OLD,
            DECODER_CALL_OLD,
            MODEL_ATTR_OLD,
            MODEL_CALL_OLD,
        )
    ]
    new = [
        text.count(CAUSAL_NEW),
        text.count(DECODER_ATTR_NEW),
        text.count(DECODER_CALL_NEW),
        text.count(MODEL_ATTR_NEW),
        text.count(MODEL_CALL_NEW),
    ]
    return old, new


def verified_qwen(text: str) -> bool:
    old, new = qwen_counts(text)
    return (
        old == [0, 0, 0, 0, 0]
        and new == [1, 1, 1, 1, 1]
        and 'getattr(config, "is_causal", None)' in text
        and "self.decoder_layer_cls(" in text
        and "self.model = self.model_cls(" in text
    )


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _speculative_decoding_models_dict(text: str) -> ast.Dict:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ValueError(f"registry is not valid Python: {exc}") from exc
    found: ast.Dict | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id != "_SPECULATIVE_DECODING_MODELS":
            continue
        if found is not None:
            raise ValueError("multiple _SPECULATIVE_DECODING_MODELS assignments")
        if not isinstance(node.value, ast.Dict):
            raise ValueError("_SPECULATIVE_DECODING_MODELS is not a dict literal")
        found = node.value
    if found is None:
        raise ValueError("missing _SPECULATIVE_DECODING_MODELS dict")
    return found


def dflash2_registry_mappings(
    text: str,
) -> list[tuple[str, tuple[str, str] | None]]:
    """Architecture-key mappings, independent of quote style."""
    found = _speculative_decoding_models_dict(text)
    mappings: list[tuple[str, tuple[str, str] | None]] = []
    for key, value in zip(found.keys, found.values):
        key_s = _const_str(key)
        if key_s != "DFlash2DraftModel":
            continue
        mapped: tuple[str, str] | None = None
        if isinstance(value, ast.Tuple) and len(value.elts) == 2:
            module = _const_str(value.elts[0])
            cls = _const_str(value.elts[1])
            if module is not None and cls is not None:
                mapped = (module, cls)
        mappings.append(("DFlash2DraftModel", mapped))
    return mappings


def dflash2_registry_key_count(text: str) -> int:
    return len(dflash2_registry_mappings(text))


def verified_registry(text: str) -> bool:
    mappings = dflash2_registry_mappings(text)
    return mappings == [
        ("DFlash2DraftModel", ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"))
    ] and text.count(REGISTRY_ENTRY) == 1


def verified_utils(text: str) -> bool:
    return (
        text.count(UTILS_OLD) == 0
        and text.count(UTILS_NEW) == 1
        and 'draft_kv = "auto"' in text
        and '"fp8_ds_mla"' in text
    )


def verified_spec_init(text: str) -> bool:
    return (
        text.count(SPEC_OLD) == 0
        and text.count(SPEC_NEW) == 1
        and "DFlash2Speculator" in text
        and "DFlash2DraftModel" in text
    )


def prepare_qwen(source: str) -> tuple[str, str]:
    old, new = qwen_counts(source)
    if verified_qwen(source):
        return source, "already present"
    if any(n > 0 for n in new) or any(n not in (0, 1) for n in old):
        raise ValueError(
            "partial/inconsistent dflash2 qwen3_dflash patch "
            f"(old={old}, new={new})"
        )
    if old != [1, 1, 1, 1, 1] or new != [0, 0, 0, 0, 0]:
        raise ValueError(
            "pinned dflash2 qwen3_dflash anchors drifted "
            f"(old={old}, new={new})"
        )
    patched = source
    patched = patched.replace(CAUSAL_OLD, CAUSAL_NEW, 1)
    patched = patched.replace(DECODER_ATTR_OLD, DECODER_ATTR_NEW, 1)
    patched = patched.replace(DECODER_CALL_OLD, DECODER_CALL_NEW, 1)
    patched = patched.replace(MODEL_ATTR_OLD, MODEL_ATTR_NEW, 1)
    patched = patched.replace(MODEL_CALL_OLD, MODEL_CALL_NEW, 1)
    if not verified_qwen(patched):
        raise ValueError("dflash2 qwen3_dflash post-patch verification failed")
    return patched, "patched"


def prepare_registry(source: str) -> tuple[str, str]:
    if verified_registry(source):
        return source, "already present"
    n_keys = dflash2_registry_key_count(source)
    if n_keys != 0:
        raise ValueError(
            "conflicting DFlash2DraftModel mapping "
            f"(keys={n_keys}, expected_entry={source.count(REGISTRY_ENTRY)})"
        )
    n_old = source.count(REGISTRY_OLD)
    if n_old != 1:
        raise ValueError(
            "pinned DFlash2 registry anchor drifted "
            f"(DFlashDraftModel={n_old}, DFlash2DraftModel="
            f"{source.count(REGISTRY_ENTRY)})"
        )
    patched = source.replace(REGISTRY_OLD, REGISTRY_NEW, 1)
    if not verified_registry(patched):
        raise ValueError("dflash2 registry post-patch verification failed")
    return patched, "patched"


def prepare_utils(source: str) -> tuple[str, str]:
    if 'draft_kv = "auto"' in source or UTILS_NEW in source:
        if not verified_utils(source):
            raise ValueError("partial/inconsistent dflash2 draft-kv patch")
        return source, "already present"
    n_old = source.count(UTILS_OLD)
    if n_old != 1:
        raise ValueError(
            "pinned dflash2 draft-kv anchor drifted "
            f"(anchor={n_old})"
        )
    patched = source.replace(UTILS_OLD, UTILS_NEW, 1)
    if not verified_utils(patched):
        raise ValueError("dflash2 draft-kv post-patch verification failed")
    return patched, "patched"


def prepare_spec_init(source: str) -> tuple[str, str]:
    if "DFlash2Speculator" in source:
        if not verified_spec_init(source):
            raise ValueError("partial/inconsistent dflash2 speculator dispatch")
        return source, "already present"
    n_old = source.count(SPEC_OLD)
    if n_old != 1:
        raise ValueError(
            "pinned dflash2 speculator-dispatch anchor drifted "
            f"(anchor={n_old})"
        )
    patched = source.replace(SPEC_OLD, SPEC_NEW, 1)
    if not verified_spec_init(patched):
        raise ValueError("dflash2 speculator-dispatch post-patch verification failed")
    return patched, "patched"


def require_kit_model(text: str) -> None:
    required = (
        "dflash2_grouped_conv",
        "direct_register_custom_op",
        "_dflash2_grouped_conv_kernel",
        "torch.topk",
        "class DFlash2Qwen3ForCausalLM",
        "class DFlash2Qwen3Model",
        "decoder_layer_cls = DFlash2Qwen3DecoderLayer",
        "block_size=1 + speculative_config.num_speculative_tokens",
        MARK,
    )
    missing = [name for name in required if name not in text]
    if missing:
        raise ValueError(f"kit dflash2 model missing {missing}")
    if ".get_top_k_tokens(" in text:
        raise ValueError(
            "kit dflash2 model still calls get_top_k_tokens; "
            "the pin LogitsProcessor does not export that helper"
        )


def require_kit_spec(text: str) -> None:
    required = (
        "class DFlash2Speculator",
        "_DRAFT_NOISE_SALT",
        "def gumbel_noised_argmax",
        'float("-inf")',
    )
    missing = [name for name in required if name not in text]
    if missing:
        raise ValueError(f"kit dflash2 speculator missing {missing}")
    for line in text.splitlines():
        stripped = line.lstrip()
        if "gumbel_noised_argmax" in line and stripped.startswith("from "):
            raise ValueError(
                "kit dflash2 speculator imports pin-missing gumbel_noised_argmax"
            )
    if "def draft_logits_spec" in text or ".draft_logits_spec(" in text:
        raise ValueError("kit dflash2 speculator uses pin-missing draft_logits_spec")


def copy_kit_files() -> None:
    if not KIT_MODEL.is_file():
        raise SystemExit(f"missing kit DFlash2 model {KIT_MODEL}")
    if not KIT_SPEC.is_file():
        raise SystemExit(f"missing kit DFlash2 speculator {KIT_SPEC}")
    model_text = KIT_MODEL.read_text()
    spec_text = KIT_SPEC.read_text()
    try:
        require_kit_model(model_text)
        require_kit_spec(spec_text)
    except ValueError as exc:
        raise SystemExit(f"dflash2 kit preflight failed: {exc}") from exc
    compile(model_text, str(KIT_MODEL), "exec")
    compile(spec_text, str(KIT_SPEC), "exec")
    DST_DIR.mkdir(parents=True, exist_ok=True)
    replace_file(DST_MODEL, model_text)
    replace_file(DST_SPEC, spec_text)
    if not DST_INIT.exists():
        replace_file(DST_INIT, "# SPDX-License-Identifier: Apache-2.0\n")


def main() -> int:
    missing = [
        path
        for path in (QWEN, REGISTRY, DFLASH_UTILS, SPEC_INIT)
        if not path.is_file()
    ]
    if missing:
        raise SystemExit("missing " + ", ".join(str(path) for path in missing))

    qwen_source = QWEN.read_text()
    registry_source = REGISTRY.read_text()
    utils_source = DFLASH_UTILS.read_text()
    spec_source = SPEC_INIT.read_text()
    try:
        qwen_patched, qwen_action = prepare_qwen(qwen_source)
        registry_patched, registry_action = prepare_registry(registry_source)
        utils_patched, utils_action = prepare_utils(utils_source)
        spec_patched, spec_action = prepare_spec_init(spec_source)
    except ValueError as exc:
        raise SystemExit(f"dflash2 patch preflight failed: {exc}") from exc

    compile(qwen_patched, str(QWEN), "exec")
    compile(registry_patched, str(REGISTRY), "exec")
    compile(utils_patched, str(DFLASH_UTILS), "exec")
    compile(spec_patched, str(SPEC_INIT), "exec")
    copy_kit_files()

    if qwen_patched != qwen_source:
        replace_file(QWEN, qwen_patched)
    if registry_patched != registry_source:
        replace_file(REGISTRY, registry_patched)
    if utils_patched != utils_source:
        replace_file(DFLASH_UTILS, utils_patched)
    if spec_patched != spec_source:
        replace_file(SPEC_INIT, spec_patched)

    print(f"{DST_MODEL.name}: installed from {KIT_MODEL.name} (registry path kept)")
    print(f"{QWEN.name}: is_causal/decoder_layer_cls {qwen_action}")
    print(f"{REGISTRY.name}: DFlash2DraftModel {registry_action}")
    print(f"{DFLASH_UTILS.name}: draft KV auto {utils_action}")
    print(f"{SPEC_INIT.name}: DFlash2Speculator {spec_action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
