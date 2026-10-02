"""Hold a waiter until a running partner publishes the shared prefix."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "overlay" / "patch_prefix_admit.py"
START = (ROOT / "start.sh").read_text()

SPEC = importlib.util.spec_from_file_location("patch_prefix_admit", INSTALLER)
assert SPEC is not None and SPEC.loader is not None
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)

PAGE = 3584

FIXTURE = (
    "logger = init_logger(__name__)\n\n\n"
    "class Scheduler(SchedulerInterface):\n"
    "    def schedule(self):\n"
    "        if self.waiting:\n"
    "            while self.waiting:\n"
    "                request = request_queue.peek_request()\n"
    "                # Get already-cached tokens.\n"
    "                if request.num_computed_tokens == 0:\n"
    "                    pass\n"
)


def _req(tokens, computed=0, request_id="r"):
    return SimpleNamespace(
        prompt_token_ids=list(tokens),
        num_computed_tokens=computed,
        request_id=request_id,
    )


def _hold(waiter, running, block_size=PAGE, hash_block_size=896):
    return PATCH._glm53_hold_for_inflight_prefix(
        waiter, running, block_size, hash_block_size
    )


def _run(tmp: Path, flag: str, text: str | None) -> subprocess.CompletedProcess[str]:
    model = tmp / "scheduler.py"
    if text is not None:
        model.write_text(text)
    env = os.environ.copy()
    env["GLM53_PREFIX_ADMIT"] = flag
    env["GLM53_SCHEDULER_PY"] = str(model)
    return subprocess.run(
        [sys.executable, str(INSTALLER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_aligned_prefix_matches_a_linear_scan() -> None:
    samples = [
        ([1, 2, 3], [9], 4),
        ([1, 2, 3, 4], [1, 2, 9, 4], 2),
        (list(range(20)), list(range(20)), 8),
        (list(range(20)), list(range(15)) + [0, 0, 0, 0, 0], 4),
        ([7] * 5000, [7] * 5000, PAGE),
        ([7] * 4000, [7] * 3999 + [1], PAGE),
        ([], [1, 2], PAGE),
        ([1], [], PAGE),
    ]
    for mine, theirs, block in samples:
        limit = min(len(mine), len(theirs))
        matched = 0
        while matched < limit and mine[matched] == theirs[matched]:
            matched += 1
        expect = (matched // block) * block
        assert PATCH._glm53_aligned_prefix_tokens(mine, theirs, block) == expect


def test_cold_identical_partner_holds_for_the_full_page() -> None:
    tokens = [3] * (PAGE + 10)
    waiter = _req(tokens, request_id="wait")
    partner = _req(tokens, computed=0, request_id="run")
    assert _hold(waiter, [partner]) == PAGE


def test_release_when_the_longest_share_is_published() -> None:
    tokens = [3] * (PAGE * 4 + 50)
    waiter = _req(tokens, request_id="wait")
    partner = _req(tokens, computed=PAGE * 4, request_id="run")
    assert _hold(waiter, [partner]) == 0


def test_one_published_page_does_not_release_the_rest() -> None:
    tokens = [3] * (PAGE * 4)
    waiter = _req(tokens, request_id="wait")
    partner = _req(tokens, computed=PAGE, request_id="run")
    assert _hold(waiter, [partner]) == PAGE * 4


def test_a_shorter_ready_partner_does_not_hide_a_longer_inflight_share() -> None:
    long = [5] * (PAGE * 3)
    short = long[:PAGE]
    waiter = _req(long, request_id="wait")
    ready = _req(short, computed=PAGE, request_id="short")
    behind = _req(long, computed=PAGE, request_id="long")
    assert _hold(waiter, [ready, behind]) == PAGE * 3
    assert _hold(waiter, [behind, ready]) == PAGE * 3


def test_a_longer_ready_partner_releases_even_if_another_is_behind() -> None:
    long = [5] * (PAGE * 3)
    waiter = _req(long, request_id="wait")
    ready = _req(long, computed=PAGE * 3, request_id="ready")
    behind = _req(long, computed=0, request_id="behind")
    assert _hold(waiter, [behind, ready]) == 0


def test_different_prefix_and_sub_page_share_are_not_held() -> None:
    waiter = _req([1] * 20 + [2] * PAGE, request_id="wait")
    other = _req([9] * PAGE, computed=0, request_id="other")
    short = _req([1] * 20 + [8] * PAGE, computed=0, request_id="short")
    assert _hold(waiter, [other, short]) == 0
    assert _hold(waiter, []) == 0


def test_hash_grid_cannot_release_before_the_coarser_page() -> None:
    tokens = [4] * 2000
    waiter = _req(tokens, request_id="wait")
    partner = _req(tokens, computed=1000, request_id="run")
    # 896 would call this shared; the MLA page is 3584, so there is no full page.
    assert _hold(waiter, [partner], block_size=896, hash_block_size=PAGE) == 0
    assert _hold(waiter, [partner], block_size=PAGE, hash_block_size=896) == 0
    wide = [4] * (PAGE + 100)
    waiter = _req(wide, request_id="wait2")
    partner = _req(wide, computed=900, request_id="run2")
    assert _hold(waiter, [partner], block_size=896, hash_block_size=PAGE) == PAGE


def test_self_empty_and_bad_block_do_not_hold() -> None:
    tokens = [1] * (PAGE * 2)
    waiter = _req(tokens, request_id="same")
    assert _hold(waiter, [waiter]) == 0
    assert _hold(_req([], request_id="empty"), [_req(tokens, request_id="run")]) == 0
    assert _hold(waiter, [_req(tokens, request_id="run")], block_size=0, hash_block_size=0) == 0
    assert _hold(waiter, [_req(tokens, request_id="run")], block_size="no", hash_block_size=PAGE) == 0


def test_cached_share_follows_later_progress() -> None:
    tokens = [6] * (PAGE * 2)
    waiter = _req(tokens, request_id="wait")
    partner = _req(tokens, computed=0, request_id="run")
    assert _hold(waiter, [partner]) == PAGE * 2
    partner.num_computed_tokens = PAGE * 2
    assert _hold(waiter, [partner]) == 0


def test_flag_off_leaves_a_missing_file(tmp_path: Path) -> None:
    result = _run(tmp_path, "0", None)
    assert result.returncode == 0
    assert "unchanged" in result.stdout
    assert not (tmp_path / "scheduler.py").exists()


def test_flag_off_does_not_edit(tmp_path: Path) -> None:
    result = _run(tmp_path, "0", FIXTURE)
    assert result.returncode == 0
    assert (tmp_path / "scheduler.py").read_text() == FIXTURE


def test_flag_on_patches_once(tmp_path: Path) -> None:
    result = _run(tmp_path, "1", FIXTURE)
    assert result.returncode == 0, result.stderr
    text = (tmp_path / "scheduler.py").read_text()
    assert text.count("[glm53-prefix-admit]") == 2
    assert text.count("def _glm53_hold_for_inflight_prefix(") == 1
    assert "step_skipped_waiting.prepend_request(request)" in text
    assert text.index("def _glm53_hold_for_inflight_prefix(") < text.index(
        "# Get already-cached tokens."
    )
    assert text.index("prepend_request(request)") < text.index(
        "# Get already-cached tokens."
    )
    compile(text, "scheduler.py", "exec")
    again = _run(tmp_path, "1", None)
    assert again.returncode == 0, again.stderr
    assert (tmp_path / "scheduler.py").read_text() == text


def test_drift_writes_nothing(tmp_path: Path) -> None:
    original = "logger = init_logger(__name__)\n\n\nclass Scheduler(SchedulerInterface):\n    pass\n"
    result = _run(tmp_path, "1", original)
    assert result.returncode == 1
    assert "anchor drift" in result.stderr
    assert (tmp_path / "scheduler.py").read_text() == original


def test_duplicate_anchor_writes_nothing(tmp_path: Path) -> None:
    original = FIXTURE + FIXTURE
    result = _run(tmp_path, "1", original)
    assert result.returncode == 1
    assert (tmp_path / "scheduler.py").read_text() == original


def test_garbage_flag_is_rejected(tmp_path: Path) -> None:
    result = _run(tmp_path, "yes", FIXTURE)
    assert result.returncode == 1
    assert "exactly 0 or 1" in result.stderr
    assert (tmp_path / "scheduler.py").read_text() == FIXTURE


def test_partial_marker_writes_nothing(tmp_path: Path) -> None:
    original = FIXTURE.replace(
        "# Get already-cached tokens.",
        "# [glm53-prefix-admit]\n                # Get already-cached tokens.",
    )
    result = _run(tmp_path, "1", original)
    assert result.returncode == 1
    assert "partial" in result.stderr
    assert (tmp_path / "scheduler.py").read_text() == original


def test_launcher_wires_the_patch_after_the_other_scheduler_edits() -> None:
    assert START.count("python3 -S /opt/glm53/patch_prefix_admit.py") == 2
    assert START.count('-e "GLM53_PREFIX_ADMIT=$GLM53_PREFIX_ADMIT"') == 1
    assert 'scp -q -o BatchMode=yes "$PREFIX_ADMIT_PATCH_HOST" ' in START
    assert "'/tmp/patch_prefix_admit.py:/opt/glm53/patch_prefix_admit.py:ro'" in START
    assert '"$PREFIX_ADMIT_PATCH_HOST:/opt/glm53/patch_prefix_admit.py:ro"' in START
    assert 'GLM53_PREFIX_ADMIT="${GLM53_PREFIX_ADMIT-0}"' in START
    assert "GLM53_PREFIX_ADMIT must be exactly 0 or 1" in START
    merge = "python3 -S /opt/glm53/patch_kv_merge_assert.py"
    admit = "python3 -S /opt/glm53/patch_prefix_admit.py"
    assert START.index(merge) < START.index(admit)
    # The worker heredoc is the second copy. It must be ordered the same way.
    second_merge = START.index(merge, START.index(merge) + 1)
    second_admit = START.index(admit, START.index(admit) + 1)
    assert second_merge < second_admit
