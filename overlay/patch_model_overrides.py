#!/usr/bin/env python3
"""Add ``exl3`` to ModelConfig's ordered quantization overrides.

Exl3Config.override_quantization_method only wins if ``exl3`` sits in this
list. Fail-closed, idempotent, env-overridable.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_MODEL_CONFIG_PY",
        os.path.join(
            os.environ.get(
                "GLM53_VLLM_SITE",
                "/usr/local/lib/python3.12/dist-packages/vllm",
            ),
            "config/model.py",
        ),
    )
)

OLD = '            overrides = [\n                "auto_gptq",\n'
NEW = '            overrides = [\n                "exl3",\n                "auto_gptq",\n'


def prepare(source: str) -> tuple[str, str]:
    if NEW in source:
        if source.count(NEW) != 1 or source.count(OLD) != 0:
            raise ValueError("partial/inconsistent exl3 ModelConfig override")
        return source, "already present"
    if source.count(OLD) != 1:
        raise ValueError(
            "pinned ModelConfig overrides list missing or not unique"
        )
    patched = source.replace(OLD, NEW, 1)
    if patched.count(NEW) != 1 or '"exl3"' not in patched:
        raise ValueError("exl3 ModelConfig override post-patch verification failed")
    return patched, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-overrides.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"exl3 override preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")
    if patched != source:
        replace_file(TARGET, patched)
    print(f"{TARGET.name}: exl3 ModelConfig override {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
