"""A resident cached tail keeps the step until its prefill is done."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "overlay" / "patch_resident_tail.py"
START = (ROOT / "start.sh").read_text()

SPEC = importlib.util.spec_from_file_location("patch_resident_tail", INSTALLER)
assert SPEC is not None and SPEC.loader is not None
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)

PAGE = 3584

FIXTURE = (
    "logger = None\n\n\n"
    "class Scheduler(SchedulerInterface):\n"
    "    def schedule(self):\n"
    "        req_index = 0\n"
    "        while req_index < len(self.running) and token_budget > 0:\n"
    "            request = self.running[req_index]\n"
    "            if input_budget <= draft_slots:\n"
    "                break\n"
    "            num_new_tokens = 0\n"
    "            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n"
    "                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n"
    "            num_new_tokens = min(\n"
    "                num_new_tokens, token_budget, input_budget - draft_slots\n"
    "            )\n"
    "            while self.waiting:\n"
    "                if True:\n"
    "                    num_computed_tokens = 0\n"
    "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
    "                    if 0 < threshold < num_new_tokens:\n"
    "                        num_new_tokens = threshold\n"
    "                break\n"
)


def _req(**kwargs):
    base = dict(
        request_id="r",
        num_prompt_tokens=0,
        num_computed_tokens=0,
        num_tokens=0,
        block_hashes=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _run(tmp: Path, flag: str, text: str | None) -> subprocess.CompletedProcess[str]:
    model = tmp / "scheduler.py"
    if text is not None:
        model.write_text(text)
    env = os.environ.copy()
    env["GLM53_RESIDENT_TAIL"] = flag
    env["GLM53_SCHEDULER_PY"] = str(model)
    return subprocess.run(
        [sys.executable, str(INSTALLER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_resident_tail_boundaries() -> None:
    assert PATCH._glm53_is_resident_tail(PAGE - 1, PAGE - 1 + 10, PAGE, PAGE) is False
    assert PATCH._glm53_is_resident_tail(PAGE, PAGE + PAGE, PAGE, PAGE) is True
    assert PATCH._glm53_is_resident_tail(PAGE, PAGE + PAGE + 1, PAGE, PAGE) is False
    assert PATCH._glm53_is_resident_tail(PAGE, PAGE, PAGE, PAGE) is False
    assert PATCH._glm53_is_resident_tail(0, 8000, PAGE, PAGE) is False
    assert PATCH._glm53_is_resident_tail(PAGE, PAGE + 10, 0, PAGE) is False
    assert PATCH._glm53_is_resident_tail(PAGE, PAGE + 10, PAGE, 0) is False
    assert PATCH._glm53_is_resident_tail("nope", PAGE, PAGE, PAGE) is False


def test_defer_rules() -> None:
    common = dict(
        prefill_left=3000,
        is_tail=False,
        num_running=4,
        max_running=4,
        defer_count=0,
        step_cap=4,
    )
    assert PATCH._glm53_should_defer_cold(
        tail_waiting=False, tail_running=True, **common
    )
    assert not PATCH._glm53_should_defer_cold(
        tail_waiting=True, tail_running=False, **common
    )
    assert PATCH._glm53_should_defer_cold(
        tail_waiting=True,
        tail_running=False,
        **{**common, "num_running": 3},
    )
    assert not PATCH._glm53_should_defer_cold(
        tail_waiting=False, tail_running=True, **{**common, "defer_count": 4}
    )
    assert not PATCH._glm53_should_defer_cold(
        tail_waiting=False, tail_running=True, **{**common, "is_tail": True}
    )
    assert not PATCH._glm53_should_defer_cold(
        tail_waiting=False, tail_running=True, **{**common, "prefill_left": 64}
    )
    assert not PATCH._glm53_should_defer_cold(
        tail_waiting=False,
        tail_running=False,
        **{**common, "num_running": 1},
    )


def test_note_stamps_only_a_fresh_qualifying_admission(monkeypatch) -> None:
    monkeypatch.delenv("GLM53_RESIDENT_TAIL_PAGE", raising=False)
    monkeypatch.delenv("GLM53_RESIDENT_TAIL_TOKENS", raising=False)
    fresh = _req(num_prompt_tokens=PAGE + 700, num_computed_tokens=0)
    PATCH._glm53_note_resident_tail(fresh, PAGE)
    assert fresh._glm53_resident_tail == 1

    cold = _req(num_prompt_tokens=20000, num_computed_tokens=0)
    cold._glm53_resident_tail = 1
    PATCH._glm53_note_resident_tail(cold, 0)
    assert cold._glm53_resident_tail == 0

    running = _req(num_prompt_tokens=PAGE + 100, num_computed_tokens=PAGE)
    PATCH._glm53_note_resident_tail(running, PAGE)
    assert not hasattr(running, "_glm53_resident_tail")


def test_open_tail_ignores_a_stale_or_finished_stamp(monkeypatch) -> None:
    monkeypatch.delenv("GLM53_RESIDENT_TAIL_PAGE", raising=False)
    monkeypatch.delenv("GLM53_RESIDENT_TAIL_TOKENS", raising=False)
    request = _req(num_prompt_tokens=PAGE + 500, _glm53_resident_tail=1)
    assert PATCH._glm53_tail_still_open(request, PAGE)
    assert not PATCH._glm53_tail_still_open(request, 0)
    assert not PATCH._glm53_tail_still_open(request, PAGE + 500)
    request._glm53_resident_tail = 0
    assert not PATCH._glm53_tail_still_open(request, PAGE)


def test_limits_fall_back_on_bad_values(monkeypatch) -> None:
    monkeypatch.setenv("GLM53_RESIDENT_TAIL_PAGE", "nope")
    assert PATCH._glm53_resident_tail_page() == PAGE
    monkeypatch.setenv("GLM53_RESIDENT_TAIL_PAGE", "8")
    assert PATCH._glm53_resident_tail_page() == PAGE
    monkeypatch.setenv("GLM53_RESIDENT_TAIL_PAGE", "4096")
    assert PATCH._glm53_resident_tail_page() == 4096
    monkeypatch.setenv("GLM53_RESIDENT_TAIL_STEPS", "0")
    assert PATCH._glm53_resident_tail_steps() == 4
    monkeypatch.setenv("GLM53_RESIDENT_TAIL_STEPS", "9")
    assert PATCH._glm53_resident_tail_steps() == 9


class _Hit:
    def __init__(self, hit: int) -> None:
        self.hit = hit

    def find_longest_cache_hit(self, hashes, length):
        return ((), self.hit, 0)


class _Sched:
    def __init__(self, running, waiting=None, hit=0, max_running=4) -> None:
        self.running = running
        self.waiting = waiting or []
        self.skipped_waiting = []
        self.kv_cache_manager = SimpleNamespace(coordinator=_Hit(hit))
        self.block_size = PAGE
        self.hash_block_size = 896
        self.max_num_running_reqs = max_running


def test_running_tail_defers_a_full_batch(monkeypatch) -> None:
    monkeypatch.delenv("GLM53_RESIDENT_TAIL_PAGE", raising=False)
    monkeypatch.delenv("GLM53_RESIDENT_TAIL_TOKENS", raising=False)
    monkeypatch.delenv("GLM53_RESIDENT_TAIL_STEPS", raising=False)
    cold = _req(request_id="cold", num_prompt_tokens=20000, num_computed_tokens=1000)
    tail = _req(
        request_id="tail",
        num_prompt_tokens=PAGE + 1000,
        num_computed_tokens=PAGE,
        _glm53_resident_tail=1,
    )
    assert PATCH._glm53_defer_cold_for_resident_tail(_Sched([cold, tail], max_running=2), cold)
    assert cold._glm53_resident_tail_defers == 1
    assert not PATCH._glm53_defer_cold_for_resident_tail(_Sched([tail, cold], max_running=2), tail)


def test_waiting_tail_needs_a_free_slot_and_ignores_a_held_waiter(monkeypatch) -> None:
    monkeypatch.delenv("GLM53_RESIDENT_TAIL_PAGE", raising=False)
    monkeypatch.delenv("GLM53_RESIDENT_TAIL_TOKENS", raising=False)
    cold = _req(request_id="cold", num_prompt_tokens=20000, num_computed_tokens=1000)
    waiter = _req(
        request_id="warm",
        num_prompt_tokens=7029,
        num_tokens=7029,
        num_computed_tokens=0,
        block_hashes=("h",),
    )
    assert PATCH._glm53_defer_cold_for_resident_tail(
        _Sched([cold], [waiter], hit=6272, max_running=4), cold
    )
    full = _req(request_id="cold2", num_prompt_tokens=20000, num_computed_tokens=1000)
    assert not PATCH._glm53_defer_cold_for_resident_tail(
        _Sched([full], [waiter], hit=6272, max_running=1), full
    )

    def _hold(*_args):
        return PAGE

    PATCH._glm53_hold_for_inflight_prefix = _hold
    try:
        held = _req(request_id="cold3", num_prompt_tokens=20000, num_computed_tokens=1000)
        assert not PATCH._glm53_defer_cold_for_resident_tail(
            _Sched([held], [waiter], hit=6272, max_running=4), held
        )
    finally:
        del PATCH._glm53_hold_for_inflight_prefix


def test_step_cap_and_lookup_failure_do_not_defer(monkeypatch) -> None:
    monkeypatch.delenv("GLM53_RESIDENT_TAIL_STEPS", raising=False)
    cold = _req(
        request_id="cold",
        num_prompt_tokens=20000,
        num_computed_tokens=1000,
        _glm53_resident_tail_defers=4,
    )
    tail = _req(
        request_id="tail",
        num_prompt_tokens=PAGE + 100,
        num_computed_tokens=PAGE,
        _glm53_resident_tail=1,
    )
    assert not PATCH._glm53_defer_cold_for_resident_tail(_Sched([cold, tail]), cold)
    assert cold._glm53_resident_tail_defers == 4

    class _Boom:
        def find_longest_cache_hit(self, hashes, length):
            raise RuntimeError("lookup")

    waiter = _req(
        request_id="warm",
        num_prompt_tokens=7029,
        num_tokens=7029,
        block_hashes=("h",),
    )
    sched = _Sched([_req(request_id="c", num_prompt_tokens=20000, num_computed_tokens=100)], [waiter])
    sched.kv_cache_manager = SimpleNamespace(coordinator=_Boom())
    assert not PATCH._glm53_defer_cold_for_resident_tail(sched, sched.running[0])


def test_flag_off_leaves_the_file(tmp_path: Path) -> None:
    result = _run(tmp_path, "0", FIXTURE)
    assert result.returncode == 0
    assert "unchanged" in result.stdout
    assert (tmp_path / "scheduler.py").read_text() == FIXTURE


def test_flag_on_applies_once_and_compiles(tmp_path: Path) -> None:
    result = _run(tmp_path, "1", FIXTURE)
    assert result.returncode == 0, result.stderr
    patched = (tmp_path / "scheduler.py").read_text()
    assert PATCH._complete(patched)
    compile(patched, "scheduler.py", "exec")
    assert "request.num_computed_tokens" in patched
    assert "_glm53_note_resident_tail(request, num_computed_tokens)" in patched
    again = _run(tmp_path, "1", None)
    assert again.returncode == 0
    assert "already present" in again.stdout
    assert (tmp_path / "scheduler.py").read_text() == patched


def test_bad_flag_and_partial_marker_write_nothing(tmp_path: Path) -> None:
    bad = _run(tmp_path, "yes", FIXTURE)
    assert bad.returncode == 1
    assert "exactly 0 or 1" in bad.stderr
    assert (tmp_path / "scheduler.py").read_text() == FIXTURE

    partial = FIXTURE.replace(
        "logger = None",
        "logger = None\n# [glm53-resident-tail]\n",
        1,
    )
    refused = _run(tmp_path, "1", partial)
    assert refused.returncode == 1
    assert "partial" in refused.stderr
    assert (tmp_path / "scheduler.py").read_text() == partial


def test_anchor_drift_writes_nothing(tmp_path: Path) -> None:
    drifted = FIXTURE.replace(
        "            request = self.running[req_index]\n",
        "            request = self.running[0]\n",
        1,
    )
    result = _run(tmp_path, "1", drifted)
    assert result.returncode == 1
    assert "anchor drift" in result.stderr
    assert (tmp_path / "scheduler.py").read_text() == drifted


def test_launcher_wires_both_ranks_after_prefix_admit() -> None:
    apply = "python3 -S /opt/glm53/patch_resident_tail.py"
    admit = "python3 -S /opt/glm53/patch_prefix_admit.py"
    assert START.count(apply) == 2
    assert START.count('-e "GLM53_RESIDENT_TAIL=$GLM53_RESIDENT_TAIL"') == 1
    assert 'scp -q -o BatchMode=yes "$RESIDENT_TAIL_PATCH_HOST" ' in START
    assert "'/tmp/patch_resident_tail.py:/opt/glm53/patch_resident_tail.py:ro'" in START
    assert '"$RESIDENT_TAIL_PATCH_HOST:/opt/glm53/patch_resident_tail.py:ro"' in START
    assert 'GLM53_RESIDENT_TAIL="${GLM53_RESIDENT_TAIL-0}"' in START
    assert "GLM53_RESIDENT_TAIL must be exactly 0 or 1" in START
    first_admit = START.index(admit)
    first_apply = START.index(apply)
    assert first_admit < first_apply
    second_admit = START.index(admit, first_admit + 1)
    second_apply = START.index(apply, first_apply + 1)
    assert second_admit < second_apply
    # The shared docker env array is what the head container actually receives.
    assert START.index('-e "GLM53_PREFIX_ADMIT=$GLM53_PREFIX_ADMIT"') < START.index(
        '-e "GLM53_RESIDENT_TAIL=$GLM53_RESIDENT_TAIL"'
    )
    assert START.index('"${nccl_common[@]}"') > START.index(
        '-e "GLM53_RESIDENT_TAIL=$GLM53_RESIDENT_TAIL"'
    )
