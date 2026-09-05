#!/usr/bin/env python3
"""CPU-only tests for the W28 correctness backport overlay."""
from __future__ import annotations

import os
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
        HERE / "patch_w28_correctness.py",
        ROOT / "overlay" / "patch_w28_correctness.py",
    )
    if path.is_file()
)
sys.path.insert(0, str(PATCH.parent))
import patch_w28_correctness as P  # noqa: E402


FIXTURES = {
    "interface": (
        """class ModelState:
"""
        + P.INTERFACE_ANCHOR
        + "        return None\n"
    ),
    "runner": (
        """class Runner:
    def initialize_kv_cache(self):
"""
        + P.RUNNER_ANCHOR
        + "            0, None\n        )\n"
    ),
    "mamba": (
        """class MambaHybridModelState:
"""
        + P.MAMBA_BIND_ANCHOR
        + P.MAMBA_SEED_ANCHOR
        + """
    def preprocess_state(self, kv_cache_config, block_tables):
"""
        + P.MAMBA_CALLS_ANCHOR
        + "        return ctx\n"
    ),
    "sm120": (
        """class FlashInferMLASparseSM120Impl:
"""
        + P.SM120_ANCHOR
        + "        self,\n    ):\n        pass\n"
    ),
}


def write_tree(root: Path) -> dict[str, Path]:
    paths = {
        "interface": root / "v1/worker/gpu/model_states/interface.py",
        "runner": root / "v1/worker/gpu/model_runner.py",
        "mamba": root / "v1/worker/gpu/model_states/mamba_hybrid.py",
        "sm120": (
            root
            / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
        ),
    }
    for label, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(FIXTURES[label])
    return paths


def run_patch(
    root: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_VLLM_SITE"] = str(root)
    return subprocess.run(
        [sys.executable, str(PATCH), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_apply_preflight_and_idempotence() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        paths = write_tree(root)
        before = {label: path.read_text() for label, path in paths.items()}

        preflight = run_patch(root, "--preflight")
        assert preflight.returncode == 0, preflight.stderr
        assert {label: path.read_text() for label, path in paths.items()} == before

        first = run_patch(root)
        assert first.returncode == 0, first.stderr
        after = {label: path.read_text() for label, path in paths.items()}
        for label, path in paths.items():
            assert P.verified_state(after[label], P.SITES[label])
            compile(after[label], str(path), "exec")

        mamba = after["mamba"]
        assert "// mamba_spec.block_size" in mamba
        assert "// self.cache_config.block_size" not in mamba
        assert '"KV cache config not bound"' in mamba
        assert "all mamba groups must share cache scheduling parameters" in mamba

        sm120 = after["sm120"]
        assert "supports_dense_mha_prefill = False" in sm120
        assert "masked_mha_available = False" in sm120

        second = run_patch(root)
        assert second.returncode == 0, second.stderr
        assert "already present" in second.stdout
        assert {label: path.read_text() for label, path in paths.items()} == after


def test_mamba_divisor_regression_arithmetic() -> None:
    """The resumed request must select Mamba column 121, not scheduler 6709."""
    num_computed_tokens = 107_360
    scheduler_block_size = 16
    mamba_block_size = 880
    assert (num_computed_tokens - 1) // mamba_block_size == 121
    assert (num_computed_tokens - 1) // scheduler_block_size == 6709


def test_each_anchor_drift_fails_before_any_write() -> None:
    for label, sites in P.SITES.items():
        for name, _mark, anchor, _patched in sites:
            with tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                paths = write_tree(root)
                target = paths[label]
                drifted_anchor = anchor.replace("\n", "\n# drift\n", 1)
                drifted = target.read_text().replace(anchor, drifted_anchor, 1)
                target.write_text(drifted)
                before = {
                    item_label: path.read_text()
                    for item_label, path in paths.items()
                }
                result = run_patch(root)
                assert result.returncode != 0, (
                    label,
                    name,
                    result.stdout,
                    result.stderr,
                )
                assert "preflight failed" in result.stderr
                assert {
                    item_label: path.read_text()
                    for item_label, path in paths.items()
                } == before


def test_pinned_container_sources_apply() -> None:
    source_root_path = Path(
        os.environ.get("GLM53_W28_PINNED_SOURCE_ROOT", PINNED_FIXTURE_ROOT)
    )
    source_names = {
        "interface": "model-state-interface.py",
        "runner": "gpu-model-runner.py",
        "mamba": "mamba-hybrid.py",
        "sm120": "sm120-mla.py",
    }
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        paths = write_tree(root)
        for label, name in source_names.items():
            paths[label].write_text((source_root_path / name).read_text())
        result = run_patch(root)
        assert result.returncode == 0, result.stderr
        for path in paths.values():
            compile(path.read_text(), str(path), "exec")


def test_recipe_wiring() -> None:
    start = (ROOT / "start.sh").read_text()
    assert start.count("python3 -S /opt/glm53/patch_w28_correctness.py") == 2
    assert start.count(
        "-v \"$W28_CORRECTNESS_PATCH_HOST:"
        "/opt/glm53/patch_w28_correctness.py:ro\""
    ) == 1


def main() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
    print(f"W28 correctness tests: PASS ({len(tests)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
