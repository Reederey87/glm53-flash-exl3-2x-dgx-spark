"""Tool-return grace: band order, TTL, cap, and the installer."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay"))

from cache_tail_evict import (  # noqa: E402
    GRACE_CAP,
    GRACE_TTL_S,
    confirm_tool_grace,
    grace_live,
    mark_reused,
    note_grace,
    rank_key,
    remember_snapshot,
    reset_grace_state,
    select_blocks,
)

INSTALLER = ROOT / "overlay" / "patch_tool_return_grace.py"


class Block:
    def __init__(self, block_id: int, num_tokens: int | None, hashed: bool = True):
        self.block_id = block_id
        self.block_hash = ("h", block_id) if hashed else None
        self.block_hash_num_tokens = num_tokens
        self.prev_free_block = None
        self.next_free_block = None


class Queue:
    def __init__(self, blocks: list[Block]):
        self.fake_free_list_head = Block(-1, None, hashed=False)
        self.fake_free_list_tail = Block(-2, None, hashed=False)
        prev = self.fake_free_list_head
        for block in blocks:
            prev.next_free_block = block
            block.prev_free_block = prev
            prev = block
        prev.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = prev

    def remove(self, block: Block) -> None:
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = block.next_free_block = None


class Pool:
    def __init__(self, blocks: list[Block]):
        self.blocks = blocks


def test_grace_sits_between_one_shot_and_reused(monkeypatch):
    monkeypatch.setenv("GLM53_CACHE_HOT_PROTECT", "1")
    monkeypatch.setenv("GLM53_TOOL_RETURN_GRACE", "1")
    reset_grace_state()
    one_shot = Block(1, 90_000)
    paused = Block(2, 90_000)
    reused = Block(3, 90_000)
    note_grace(2, 90_000)
    mark_reused(3)
    try:
        queue = Queue([reused, paused, one_shot])
        taken = [block.block_id for block in select_blocks(queue, 1)]
        assert taken == [1]
        assert rank_key(True, 1, 0) < rank_key(True, 1, 0, grace=True)
        assert rank_key(True, 1, 0, grace=True) < rank_key(True, 1, 0, reused=True)
    finally:
        reset_grace_state()


def test_expired_grace_is_spent_as_one_shot(monkeypatch):
    monkeypatch.setenv("GLM53_TOOL_RETURN_GRACE", "1")
    reset_grace_state()
    # The paused page is the deep one. Live grace would spend the shallow
    # one-shot instead. Expired grace spends the deep page.
    paused = Block(1, 90_000)
    chaff = Block(2, 3_584)
    note_grace(1, 90_000, now=time.monotonic() - (GRACE_TTL_S + 1))
    try:
        queue = Queue([chaff, paused])
        taken = [block.block_id for block in select_blocks(queue, 1)]
        assert taken == [1]
        assert not grace_live(1, time.monotonic())
    finally:
        reset_grace_state()


def test_cap_drops_the_deepest_page_of_the_oldest_mark():
    reset_grace_state()
    try:
        now = 1_000.0
        for block_id in range(GRACE_CAP):
            note_grace(block_id, 3_584, now=now)
        note_grace(GRACE_CAP, 200_000, now=now)
        assert GRACE_CAP not in _grace_ids()
        assert 0 in _grace_ids()
        note_grace(GRACE_CAP + 1, 3_584, now=now + 5)
        assert GRACE_CAP + 1 in _grace_ids()
    finally:
        reset_grace_state()


def test_confirm_uses_explicit_external_id_with_overlapping_finishes(monkeypatch):
    monkeypatch.setenv("GLM53_TOOL_RETURN_GRACE", "1")
    reset_grace_state()
    kept = Block(0, 3_584)
    other = Block(1, 7_168)
    pool = Pool([kept, other])
    remember_snapshot("engine-id", [(0, kept.block_hash, 3_584)], "chat-id")
    remember_snapshot("engine-other", [(1, other.block_hash, 7_168)], "chat-other")
    try:
        assert confirm_tool_grace("chat-id", pool) == 1
        assert grace_live(0, time.monotonic())
        assert not grace_live(1, time.monotonic())
        assert confirm_tool_grace("chat-other", pool) == 1
    finally:
        reset_grace_state()


def test_unknown_id_cannot_claim_an_unrelated_recent_finish(monkeypatch):
    monkeypatch.setenv("GLM53_TOOL_RETURN_GRACE", "1")
    reset_grace_state()
    block = Block(0, 3_584)
    remember_snapshot("plain-stop", [(0, block.block_hash, 3_584)])
    assert confirm_tool_grace("unrelated-tool-call", Pool([block])) == 0
    assert not grace_live(0, time.monotonic())
    reset_grace_state()


def test_more_than_sixteen_unconfirmed_finishes_stay_bounded(monkeypatch):
    import cache_tail_evict as evict
    monkeypatch.setenv("GLM53_TOOL_RETURN_GRACE", "1")
    reset_grace_state()
    block = Block(0, 3_584)
    for index in range(128):
        remember_snapshot(f"engine-{index}", [(0, block.block_hash, 3_584)])
        assert len(evict._SNAPSHOTS) <= evict._SNAPSHOT_LIMIT
    assert confirm_tool_grace("engine-0", Pool([block])) == 0
    assert confirm_tool_grace("engine-127", Pool([block])) == 1
    reset_grace_state()


def test_duplicate_external_ids_fail_closed(monkeypatch):
    monkeypatch.setenv("GLM53_TOOL_RETURN_GRACE", "1")
    reset_grace_state()
    blocks = [Block(0, 3_584), Block(1, 7_168)]
    for index, block in enumerate(blocks):
        remember_snapshot(str(index), [(index, block.block_hash, 3_584)], "duplicate")
    assert confirm_tool_grace("duplicate", Pool(blocks)) == 0
    assert confirm_tool_grace("duplicate", Pool(blocks)) == 0
    assert not _grace_ids()
    reset_grace_state()


def test_pending_snapshot_expires_and_reset_drops_it(monkeypatch):
    import cache_tail_evict as evict
    monkeypatch.setenv("GLM53_TOOL_RETURN_GRACE", "1")
    reset_grace_state()
    block = Block(0, 3_584)
    monkeypatch.setattr(evict.time, "monotonic", lambda: 1000.0)
    remember_snapshot("engine", [(0, block.block_hash, 3_584)], "chat")
    monkeypatch.setattr(evict.time, "monotonic", lambda: 1001.0 + GRACE_TTL_S)
    assert confirm_tool_grace("chat", Pool([block])) == 0
    remember_snapshot("engine", [(0, block.block_hash, 3_584)], "chat")
    reset_grace_state()
    assert confirm_tool_grace("chat", Pool([block])) == 0


def test_snapshot_and_confirm_are_inert_when_disabled(monkeypatch):
    import cache_tail_evict as evict
    monkeypatch.setenv("GLM53_TOOL_RETURN_GRACE", "0")
    reset_grace_state()
    block = Block(0, 3_584)
    remember_snapshot("engine", [(0, block.block_hash, 3_584)], "chat")
    assert not evict._SNAPSHOTS
    assert confirm_tool_grace("chat", Pool([block])) == 0


def test_confirm_skips_a_block_whose_hash_changed(monkeypatch):
    monkeypatch.setenv("GLM53_TOOL_RETURN_GRACE", "1")
    reset_grace_state()
    kept = Block(0, 3_584)
    lost = Block(1, 7_168)
    pool = Pool([kept, lost])
    remember_snapshot(
        "req",
        [(0, kept.block_hash, 3_584), (1, lost.block_hash, 7_168)],
    )
    lost.block_hash = ("h", "other")
    try:
        assert confirm_tool_grace("req", pool) == 1
        assert grace_live(0, time.monotonic())
        assert not grace_live(1, time.monotonic())
        assert confirm_tool_grace("req", pool) == 0
    finally:
        reset_grace_state()


def test_a_real_hit_clears_grace():
    reset_grace_state()
    try:
        note_grace(4, 3_584)
        mark_reused(4)
        assert not grace_live(4, time.monotonic())
    finally:
        reset_grace_state()


def _grace_ids() -> set[int]:
    from cache_tail_evict import _GRACE

    return set(_GRACE)


def _write_anchors(tmp: Path) -> None:
    (tmp / "kv.py").write_text(
        "def free(self, request):\n        \"\"\"placeholder\n" + FREE_ANCHOR + "        return None\n"
    )
    (tmp / "core.py").write_text(
        "class C:\n    def reset_prefix_cache(self):\n"
        + CORE_ANCHOR
        + "        return None\n"
        + "    def preprocess(self, request):\n"
        + "        req = Request.from_engine_core_request(request, self.request_block_hasher)\n"
        + "        if req.use_structured_output:\n            pass\n"
    )
    (tmp / "serving.py").write_text(
        "async def stream():\n" + STREAM_ANCHOR + "async def full():\n" + FULL_ANCHOR
    )


def _run_installer(tmp: Path, flag: str, *, rewrite: bool = True) -> subprocess.CompletedProcess[str]:
    if rewrite:
        _write_anchors(tmp)
    env = os.environ.copy()
    env["GLM53_TOOL_RETURN_GRACE"] = flag
    env["GLM53_KV_MANAGER_PY"] = str(tmp / "kv.py")
    env["GLM53_ENGINE_CORE_PY"] = str(tmp / "core.py")
    env["GLM53_CHAT_SERVING_PY"] = str(tmp / "serving.py")
    return subprocess.run(
        [sys.executable, str(INSTALLER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


FREE_ANCHOR = """\
            request: The request to free the blocks.
        \"\"\"
        pins = self._partial_tail_pins.pop(request.request_id, None)
"""
CORE_ANCHOR = """\
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
"""
STREAM_ANCHOR = """\
                        if tools_streamed[i] and not tool_choice_function_name:
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )
"""
FULL_ANCHOR = """\
            is_finish_reason_tool_calls = auto_tools_called or (
                request.tool_choice
                and request.tool_choice == "required"
                and output.finish_reason == "stop"
            )
"""


def test_installer_flag_off_leaves_files(tmp_path: Path):
    result = _run_installer(tmp_path, "0")
    assert result.returncode == 0, result.stderr
    assert "unchanged" in result.stdout
    assert "[glm53-tool-return-grace]" not in (tmp_path / "kv.py").read_text()


def test_installer_flag_on_is_idempotent(tmp_path: Path):
    result = _run_installer(tmp_path, "1")
    assert result.returncode == 0, result.stderr
    text = (tmp_path / "serving.py").read_text()
    assert text.count("[glm53-tool-return-grace]") == 4
    assert (tmp_path / "core.py").read_text().count("def glm53_mark_tool_grace") == 1
    again = _run_installer(tmp_path, "1", rewrite=False)
    assert again.returncode == 0, again.stderr
    assert "already present" in again.stdout
    assert (tmp_path / "serving.py").read_text() == text


def test_installer_rejects_garbage_and_drift(tmp_path: Path):
    for name in ("kv.py", "core.py", "serving.py"):
        (tmp_path / name).write_text("drift\n")
    bad = _run_installer(tmp_path, "1", rewrite=False)
    assert bad.returncode == 1
    assert (tmp_path / "kv.py").read_text() == "drift\n"
    garbage = _run_installer(tmp_path, "yes", rewrite=False)
    assert garbage.returncode == 1
    assert (tmp_path / "kv.py").read_text() == "drift\n"


def test_installer_refuses_marker_only_or_incomplete_upgrade(tmp_path: Path):
    assert _run_installer(tmp_path, "1").returncode == 0
    path = tmp_path / "core.py"
    path.write_text(path.read_text().replace(
        "req._glm53_external_request_id = request.external_req_id", "pass"
    ))
    before = [(tmp_path / name).read_text() for name in ("kv.py", "core.py", "serving.py")]
    result = _run_installer(tmp_path, "1", rewrite=False)
    assert result.returncode == 1
    assert [(tmp_path / name).read_text() for name in ("kv.py", "core.py", "serving.py")] == before
