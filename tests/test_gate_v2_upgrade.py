#!/usr/bin/env python3
"""The gate patcher must handle every deployed predecessor state: pristine ->
v3.1, v1-baked -> v3.1, v2 (applied or baked) -> v3.1, v3 -> v3.1,
v3.1 -> no-op. Discovered live 2026-08-31: the original
PR #80 installer saw the v1 marker and skipped, leaving production on the v1 gate."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "overlay" / "patch_scheduler_decode_floor.py"

PRISTINE = (
    "import itertools\n"
    "import time\n"
    "\n"
    "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
    "\n"
    "class Scheduler:\n"
    "    def _sched_running(self):\n"
    "        if True:\n"
    "            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n"
    "                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n"
    "            num_new_tokens = min(\n"
    "                num_new_tokens, token_budget, input_budget - draft_slots\n"
    "            )\n"
    "\n"
    "            # Make sure the input position does not exceed the max model len.\n"
    "\n"
    "    def _sched_waiting(self):\n"
    "        while True:\n"
    "            if True:\n"
    "                if True:\n"
    "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
    "                    if 0 < threshold < num_new_tokens:\n"
    "                        num_new_tokens = threshold\n"
    "\n"
    "                    # chunked prefill has to be enabled explicitly to allow\n"
    "                    pass\n"
)


def load_overlay(target: Path):
    os.environ["GLM53_SCHEDULER_PY"] = str(target)
    spec = importlib.util.spec_from_file_location("gate_overlay", OVERLAY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_patcher(target: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy(); env["GLM53_SCHEDULER_PY"] = str(target)
    return subprocess.run([sys.executable, str(OVERLAY)], env=env, capture_output=True, text=True)


def build_v1_file(mod) -> str:
    """Reproduce what the OLD v1 overlay produced, from the current overlay's own anchors."""
    v1 = PRISTINE.replace(mod.IMPORT_OLD, mod.IMPORT_NEW, 1)
    helper_v1 = "\ndef _glm53_mixed_prefill_policy(" + mod.HELPER.split(
        "def _glm53_mixed_prefill_policy(", 1)[1]
    v1 = v1.replace(
        "from vllm.compilation.cuda_graph import CUDAGraphStat\n",
        helper_v1 + "from vllm.compilation.cuda_graph import CUDAGraphStat\n", 1)
    run_v1 = mod.RUNNING_OLD.replace(
        ")\n\n            # Make sure",
        ")\n" + mod.V1_RUNNING + "\n            # Make sure", 1)
    v1 = v1.replace(mod.RUNNING_OLD, run_v1, 1)
    wait_body = (
        "                        if mixed_cap <= 0:\n"
        "                            request_queue.pop_request()\n"
        "                            step_skipped_waiting.prepend_request(request)\n"
        "                            continue\n"
        "                        num_new_tokens = min(num_new_tokens, mixed_cap)\n"
    )
    wait_v1 = mod.WAITING_OLD.replace(
        "num_new_tokens = threshold\n\n                    # chunked prefill",
        "num_new_tokens = threshold\n" + mod.V1_WAITING + wait_body
        + "\n                    # chunked prefill", 1)
    v1 = v1.replace(mod.WAITING_OLD, wait_v1, 1)
    return v1


def test_pristine_v31_then_noop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "scheduler.py"
        mod = load_overlay(t)
        assert mod.RUNNING_OLD in PRISTINE and mod.WAITING_OLD in PRISTINE, "fixture drifted"
        t.write_text(PRISTINE)
        r = run_patcher(t); assert r.returncode == 0, r.stderr
        v31 = t.read_text()
        assert mod.MARK_V31 in v31
        assert v31.count("_glm53_mixed_prefill_gate(") >= 3
        compile(v31, "v31.py", "exec")
        r = run_patcher(t); assert r.returncode == 0 and "already present" in r.stdout, r.stdout
        assert t.read_text() == v31


def test_v1_baked_upgrades_then_noop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "scheduler.py"
        mod = load_overlay(t)
        v1 = build_v1_file(mod)
        assert mod.V1_RUNNING in v1 and mod.V1_WAITING in v1, "v1 fixture drifted"
        assert mod.MARK in v1 and mod.MARK_V2 not in v1
        compile(v1, "v1.py", "exec")
        t.write_text(v1)
        r = run_patcher(t); assert r.returncode == 0, r.stderr
        assert "upgraded v1 -> v3.1" in r.stdout, r.stdout
        up = t.read_text()
        assert mod.V1_RUNNING not in up and mod.V1_WAITING not in up
        assert up.count("_glm53_mixed_prefill_gate(") >= 3
        assert up.count("def _glm53_mixed_prefill_policy(") == 1
        assert up.count("def _glm53_mixed_prefill_gate(") == 1
        compile(up, "up.py", "exec")
        r = run_patcher(t); assert r.returncode == 0 and "already present" in r.stdout, r.stdout
        assert t.read_text() == up


def build_v2_file(mod) -> str:
    """What the v2 overlay produced at image build (the w24 production image bakes this)."""
    v2 = PRISTINE.replace(mod.IMPORT_OLD, mod.IMPORT_NEW, 1)
    v2 = v2.replace(
        "from vllm.compilation.cuda_graph import CUDAGraphStat\n",
        mod.HELPER_V2_ONLY + mod.HELPER_POLICY + "from vllm.compilation.cuda_graph import CUDAGraphStat\n", 1)
    v2 = v2.replace(mod.RUNNING_OLD, mod.RUNNING_NEW, 1).replace(mod.WAITING_OLD, mod.WAITING_NEW, 1)
    return v2


def test_v2_baked_upgrades_to_v3_then_noop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "scheduler.py"
        mod = load_overlay(t)
        v2 = build_v2_file(mod)
        assert mod.MARK_V2 in v2 and mod.MARK_V3 not in v2
        compile(v2, "v2.py", "exec")
        t.write_text(v2)
        r = run_patcher(t); assert r.returncode == 0, r.stderr
        assert "upgraded v2 -> v3.1" in r.stdout, r.stdout
        up = t.read_text()
        assert mod.HELPER_V2_ONLY not in up and mod.HELPER_V31_ONLY in up
        assert up.count("def _glm53_mixed_prefill_gate(") == 1
        assert up.count("def _glm53_gate_state(") == 1
        assert up.count("_glm53_mixed_prefill_gate(") >= 3   # call sites untouched
        compile(up, "up.py", "exec")
        r = run_patcher(t); assert r.returncode == 0 and "already present" in r.stdout, r.stdout
        assert t.read_text() == up


def test_v3_baked_upgrades_to_v31_then_noop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "scheduler.py"
        mod = load_overlay(t)
        v3 = PRISTINE.replace(mod.IMPORT_OLD, mod.IMPORT_NEW, 1)
        v3 = v3.replace(
            "from vllm.compilation.cuda_graph import CUDAGraphStat\n",
            mod.HELPER_V3_ONLY
            + mod.HELPER_POLICY
            + "from vllm.compilation.cuda_graph import CUDAGraphStat\n",
            1,
        )
        v3 = v3.replace(mod.RUNNING_OLD, mod.RUNNING_NEW, 1).replace(
            mod.WAITING_OLD, mod.WAITING_NEW, 1
        )
        assert mod.MARK_V3 in v3 and mod.MARK_V31 not in v3
        compile(v3, "v3.py", "exec")
        t.write_text(v3)
        r = run_patcher(t)
        assert r.returncode == 0 and "upgraded v3 -> v3.1" in r.stdout, r.stdout
        up = t.read_text()
        assert mod.HELPER_V3_ONLY not in up and mod.HELPER_V31_ONLY in up
        compile(up, "v31.py", "exec")
        r = run_patcher(t)
        assert r.returncode == 0 and "already present" in r.stdout, r.stdout
        assert t.read_text() == up


def test_damaged_v3_predecessor_fails_without_writing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "scheduler.py"
        mod = load_overlay(t)
        v3 = PRISTINE.replace(mod.IMPORT_OLD, mod.IMPORT_NEW, 1)
        v3 = v3.replace(
            "from vllm.compilation.cuda_graph import CUDAGraphStat\n",
            mod.HELPER_V3_ONLY
            + mod.HELPER_POLICY
            + "from vllm.compilation.cuda_graph import CUDAGraphStat\n",
            1,
        )
        v3 = v3.replace(mod.RUNNING_OLD, mod.RUNNING_NEW, 1).replace(
            mod.WAITING_OLD, mod.WAITING_NEW, 1
        )
        damaged = v3.replace(mod.V2_RUNNING, "            mixed_cap = None\n", 1)
        t.write_text(damaged)
        before = t.read_bytes()
        r = run_patcher(t)
        assert r.returncode != 0
        assert "complete running gate" in r.stderr
        assert t.read_bytes() == before


def test_damaged_v31_noop_fails_without_writing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "scheduler.py"
        mod = load_overlay(t)
        t.write_text(PRISTINE)
        r = run_patcher(t)
        assert r.returncode == 0, r.stderr
        damaged = t.read_text().replace(
            "_glm53_set_capped(st, now, True)",
            "_glm53_set_capped(st, now, False)",
            1,
        )
        t.write_text(damaged)
        before = t.read_bytes()
        r = run_patcher(t)
        assert r.returncode != 0
        assert "v3.1 helper" in r.stderr
        assert t.read_bytes() == before


def test_damaged_waiting_body_fails_without_writing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "scheduler.py"
        load_overlay(t)
        t.write_text(PRISTINE)
        r = run_patcher(t)
        assert r.returncode == 0, r.stderr
        damaged = t.read_text().replace(
            "                        num_new_tokens = min(num_new_tokens, mixed_cap)\n",
            "                        pass\n",
            1,
        )
        t.write_text(damaged)
        before = t.read_bytes()
        r = run_patcher(t)
        assert r.returncode != 0
        assert "complete waiting gate" in r.stderr
        assert t.read_bytes() == before


def test_commented_import_fails_without_writing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "scheduler.py"
        load_overlay(t)
        t.write_text(PRISTINE)
        r = run_patcher(t)
        assert r.returncode == 0, r.stderr
        damaged = t.read_text().replace("import os\n", "# import os\n", 1)
        t.write_text(damaged)
        before = t.read_bytes()
        r = run_patcher(t)
        assert r.returncode != 0
        assert "top-level unaliased import os" in r.stderr
        assert t.read_bytes() == before


if __name__ == "__main__":
    test_pristine_v31_then_noop()
    test_v1_baked_upgrades_then_noop()
    test_v2_baked_upgrades_to_v3_then_noop()
    test_v3_baked_upgrades_to_v31_then_noop()
    test_damaged_v3_predecessor_fails_without_writing()
    test_damaged_v31_noop_fails_without_writing()
    test_damaged_waiting_body_fails_without_writing()
    test_commented_import_fails_without_writing()
    print("gate v2 upgrade paths OK")
