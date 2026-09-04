#!/usr/bin/env python3
"""Backport the two correctness fixes bundled with the GLM W28 restart.

This overlay ports:

* vLLM #53798: seed resumed align-mode Mamba state indices in
  ``MambaSpec.block_size`` units, not the generic scheduler block size;
* vLLM #54057: declare ``masked_mha_available = False`` on the SM120 sparse
  MLA implementation used by GLM-5.3 on GB10, while also pinning its missing
  dense-prefill capability to ``False``.

The #53798 port is adapted to the pinned preview build. The runner binds the
resolved KV cache config to ``ModelState`` immediately after attention backend
initialization and before any request can be admitted. The hybrid model state
then resolves and validates its Mamba groups once and uses the Mamba block size
for resumed-request state indexing.

All five Python targets are preflighted and compiled before any write.
Re-application is idempotent, anchor drift fails closed, writes use atomic
replacement, and stale pyc files are removed.
"""
from __future__ import annotations

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

TARGETS = {
    "interface": SITE / "v1/worker/gpu/model_states/interface.py",
    "runner": SITE / "v1/worker/gpu/model_runner.py",
    "mamba": SITE / "v1/worker/gpu/model_states/mamba_hybrid.py",
    "sm120": SITE / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
}


INTERFACE_MARK = "    def set_kv_cache_config(  # [glm53-w28-correctness]\n"
INTERFACE_ANCHOR = """    def apply_staged_writes(self) -> None:
        return None

    def get_additional_cg_support(self) -> tuple[AttentionCGSupport, str | None]:
"""
INTERFACE_PATCHED = """    def apply_staged_writes(self) -> None:
        return None

    def set_kv_cache_config(  # [glm53-w28-correctness]
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        \"\"\"Bind the resolved KV cache config before requests are admitted.\"\"\"
        return None

    def get_additional_cg_support(self) -> tuple[AttentionCGSupport, str | None]:
"""

RUNNER_MARK = (
    "        self.model_state.set_kv_cache_config(  "
    "# [glm53-w28-correctness]\n"
)
RUNNER_ANCHOR = """        self.attn_groups, attn_cg_support, self.kernel_block_sizes = init_attn_backend(
            self.kv_cache_config, self.vllm_config, self.device
        )
        attn_cg_support = attn_cg_support.narrow(
"""
RUNNER_PATCHED = """        self.attn_groups, attn_cg_support, self.kernel_block_sizes = init_attn_backend(
            self.kv_cache_config, self.vllm_config, self.device
        )
        self.model_state.set_kv_cache_config(  # [glm53-w28-correctness]
            self.kv_cache_config
        )
        attn_cg_support = attn_cg_support.narrow(
"""

MAMBA_BIND_MARK = "    def set_kv_cache_config(  # [glm53-w28-correctness]\n"
MAMBA_BIND_ANCHOR = """    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
"""
MAMBA_BIND_PATCHED = """    def set_kv_cache_config(  # [glm53-w28-correctness]
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        if not self._align_mode:
            return
        group_ids: list[int] = []
        specs: list[MambaSpec] = []
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                group_ids.append(group_id)
                specs.append(spec)
        assert specs, "no mamba layers in the model"
        representative = specs[0]
        assert all(
            spec.block_size == representative.block_size
            and spec.num_speculative_blocks
            == representative.num_speculative_blocks
            and spec.mamba_cache_mode == representative.mamba_cache_mode
            for spec in specs
        ), "all mamba groups must share cache scheduling parameters"
        self._mamba_group_ids = group_ids
        self._mamba_spec = representative

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
"""

MAMBA_SEED_MARK = (
    "            # [glm53-w28-correctness] The align table is in Mamba blocks.\n"
)
MAMBA_SEED_ANCHOR = """        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
            )

    def _get_mamba_group_info(
        self, kv_cache_config: KVCacheConfig
    ) -> tuple[list[int], MambaSpec]:
        if self._mamba_spec is None:
            group_ids: list[int] = []
            specs: list[MambaSpec] = []
            for i, group in enumerate(kv_cache_config.kv_cache_groups):
                spec = group.kv_cache_spec
                if isinstance(spec, MambaSpec):
                    group_ids.append(i)
                    specs.append(spec)
            assert specs, "no mamba layers in the model"
            assert all(specs[0] == s for s in specs)
            self._mamba_group_ids = group_ids
            self._mamba_spec = specs[0]
        return self._mamba_group_ids, self._mamba_spec
"""
MAMBA_SEED_PATCHED = """        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            # [glm53-w28-correctness] The align table is in Mamba blocks.
            _, mamba_spec = self._get_mamba_group_info()
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // mamba_spec.block_size
            )

    def _get_mamba_group_info(self) -> tuple[list[int], MambaSpec]:
        assert self._mamba_spec is not None, "KV cache config not bound"
        return self._mamba_group_ids, self._mamba_spec
"""

MAMBA_CALLS_MARK = (
    "        mamba_group_ids, mamba_spec = self._get_mamba_group_info()  "
    "# [glm53-w28-correctness]\n"
)
MAMBA_CALLS_ANCHOR = """        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        ctx = self._ensure_align_ctx(kv_cache_config, mamba_group_ids, block_tables)
"""
MAMBA_CALLS_PATCHED = """        mamba_group_ids, mamba_spec = self._get_mamba_group_info()  # [glm53-w28-correctness]
        ctx = self._ensure_align_ctx(kv_cache_config, mamba_group_ids, block_tables)
"""

SM120_MARK = "    masked_mha_available = False  # [glm53-w28-correctness]\n"
SM120_ANCHOR = """    is_sparse = True

    def __init__(
"""
SM120_PATCHED = """    is_sparse = True
    supports_dense_mha_prefill = False
    # The sparse-MQA-only SM120 implementation does not inherit the generic
    # initializer that sets this capability, but the prefill dispatcher reads
    # it for every sparse backend once num_mha_tokens > 0.
    masked_mha_available = False  # [glm53-w28-correctness]

    def __init__(
"""

SITES = {
    "interface": (
        ("KV config hook", INTERFACE_MARK, INTERFACE_ANCHOR, INTERFACE_PATCHED),
    ),
    "runner": (
        ("bind KV config", RUNNER_MARK, RUNNER_ANCHOR, RUNNER_PATCHED),
    ),
    "mamba": (
        ("resolve Mamba groups", MAMBA_BIND_MARK, MAMBA_BIND_ANCHOR, MAMBA_BIND_PATCHED),
        ("Mamba-block seed", MAMBA_SEED_MARK, MAMBA_SEED_ANCHOR, MAMBA_SEED_PATCHED),
        ("bound group lookup", MAMBA_CALLS_MARK, MAMBA_CALLS_ANCHOR, MAMBA_CALLS_PATCHED),
    ),
    "sm120": (
        ("masked MHA capability", SM120_MARK, SM120_ANCHOR, SM120_PATCHED),
    ),
}


def verified_state(text: str, sites) -> bool:
    return all(
        text.count(mark) == 1
        and text.count(patched) == 1
        and text.count(anchor) == patched.count(anchor)
        for _name, mark, anchor, patched in sites
    )


def prepare(text: str, sites, label: str) -> tuple[str, str]:
    marks = sum(text.count(mark) for _name, mark, _anchor, _patched in sites)
    if marks:
        if marks != len(sites) or not verified_state(text, sites):
            raise ValueError(
                f"partial/inconsistent {label} patch "
                f"(marks={marks}, expected={len(sites)})"
            )
        return text, "already present"

    out = text
    for name, _mark, anchor, patched in sites:
        count = out.count(anchor)
        if count != 1:
            raise ValueError(
                f"pinned {label} anchor {name!r} drifted "
                f"(found {count}, expected 1)"
            )
        out = out.replace(anchor, patched, 1)
    if not verified_state(out, sites):
        raise ValueError(f"{label} post-patch verification failed")
    return out, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-w28-correctness.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob(f"{target.stem}*.pyc"):
        pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    preflight_only = "--preflight" in argv[1:]

    original: dict[str, str] = {}
    patched: dict[str, str] = {}
    actions: dict[str, str] = {}
    for label, target in TARGETS.items():
        if not target.is_file():
            raise SystemExit(f"missing {target}")
        original[label] = target.read_text()
        try:
            patched[label], actions[label] = prepare(
                original[label], SITES[label], label
            )
        except ValueError as exc:
            raise SystemExit(f"W28 correctness preflight failed: {exc}") from exc
        compile(patched[label], str(target), "exec")

    if preflight_only:
        print(
            "W28 correctness preflight OK "
            + " ".join(f"{name}={actions[name]}" for name in TARGETS)
        )
        return 0

    for label, target in TARGETS.items():
        if patched[label] != original[label]:
            replace_file(target, patched[label])
            clear_pyc(target)
    print(
        "W28 correctness "
        + " ".join(f"{name}={actions[name]}" for name in TARGETS)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
