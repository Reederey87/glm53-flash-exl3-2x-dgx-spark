#!/usr/bin/env python3
"""Stdlib-only LinearEXL3 namespace stub for the vLLM EXL3 overlay.

v1.4.7 LinearEXL3 imports ``Config`` from ``exllamav3.model.config`` and, when
``config is None``, constructs ``NullConfig`` so ``forward()`` can read
``config.infer_params``. Loading the real ``exllamav3`` package pulls
FlashAttention, so the overlay stubs only the packages that do that and
injects compatible ``NullConfig``/``InferParams``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path


class InferParams:
    """Minimal InferParams compatible with v1.4.7 LinearEXL3.

    Real ``exllamav3.model.config.InferParams`` also owns CPU-MoE/vision/n-gram
    knobs. This serving path never constructs those modules; LinearEXL3 only
    reads ``no_reconstruct`` on the reconstruct-vs-GEMM gate. Defaults match an
    unset InferParams (``no_reconstruct=False``).
    """

    no_reconstruct: bool = False
    mgemm_K_threshold: int = 0
    mgemm_n_threshold: int = 0

    def __init__(self) -> None:
        self.no_reconstruct = False
        self.mgemm_K_threshold = 0
        self.mgemm_n_threshold = 0


class NullConfig:
    """Stand-in for ``exllamav3.model.config.NullConfig``.

    v1.4.7 LinearEXL3 replaces ``config=None`` with ``NullConfig()`` because
    ``forward()`` reads ``config.infer_params``. The previous Config-only stub
    lacked both names and would raise AttributeError at first LinearEXL3
    construction.
    """

    def __init__(self) -> None:
        self.infer_params = InferParams()


def inject_config_stub(package_root: Path, modules: dict | None = None) -> types.ModuleType:
    """Install package stubs plus ``NullConfig``/``InferParams`` into ``modules``."""
    modules = sys.modules if modules is None else modules
    package_root = Path(package_root)
    for name, path in (
        ("exllamav3", package_root),
        ("exllamav3.modules", package_root / "modules"),
        ("exllamav3.model", package_root / "model"),
    ):
        if name in modules:
            continue
        module = types.ModuleType(name)
        module.__file__ = str(path / "__init__.py")
        module.__package__ = name
        module.__path__ = [str(path)]
        modules[name] = module

    existing = modules.get("exllamav3.model.config")
    if existing is not None:
        if not hasattr(existing, "NullConfig") or not hasattr(existing, "InferParams"):
            raise RuntimeError(
                "exllamav3.model.config is already loaded without NullConfig/InferParams"
            )
        return existing
    config = types.ModuleType("exllamav3.model.config")
    config.__file__ = str(package_root / "model/config.py")
    config.__package__ = "exllamav3.model"
    config.Config = type("Config", (), {})
    config.InferParams = InferParams
    config.NullConfig = NullConfig
    modules[config.__name__] = config
    return config


def install_exllamav3_namespace() -> None:
    """Load LinearEXL3 without running ``exllamav3/__init__.py`` (FlashAttention)."""
    if "exllamav3.modules.quant.exl3" in sys.modules:
        return
    import exllamav3_ext  # noqa: F401  — compiled extension must exist

    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("exllamav3 package is not installed in this image")
    package_root = Path(list(spec.submodule_search_locations)[0])
    inject_config_stub(package_root)


def load_linear_exl3_cls():
    install_exllamav3_namespace()
    return importlib.import_module("exllamav3.modules.quant.exl3").LinearEXL3
