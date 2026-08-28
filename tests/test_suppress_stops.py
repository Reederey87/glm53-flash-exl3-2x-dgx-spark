#!/usr/bin/env python3
"""CPU test: fail-closed #42 patcher matches glm53-flash detokenizer anchors."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _d in (HERE, ROOT / "overlay"):
    if (_d / "patch_suppress_stops_in_reasoning.py").is_file():
        sys.path.insert(0, str(_d))
        break
from patch_suppress_stops_in_reasoning import (  # noqa: E402
    FACTORY_OLD,
    IMPORT_OLD,
    INIT_OLD,
    MARK,
    STOP_OLD,
    apply_text,
)

MINIMAL = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    f"{IMPORT_OLD}"
    "class IncrementalDetokenizer:\n"
    "    def from_new_request(cls, tokenizer, request):\n"
    f"{FACTORY_OLD}"
    "class BaseIncrementalDetokenizer:\n"
    "    def __init__(self, request):\n"
    f"{INIT_OLD}"
    "    def update(self, new_token_ids, stop_terminated):\n"
    "        stop_check_offset = 0\n"
    f"{STOP_OLD}"
    "                output_text=self.output_text,\n"
    "            )\n"
)


def test_apply_then_skip() -> None:
    out, status = apply_text(MINIMAL)
    assert status == "applied", status
    assert MARK in out
    assert "_maybe_enable_reasoning_stop_guard" in out
    assert "not self._reasoning_stop_guard or self._reasoning_closed" in out
    assert "TokenizersBackend" in FACTORY_OLD
    out2, status2 = apply_text(out)
    assert status2 == "skipped", status2
    assert out2 == out


def test_missing_anchor_fails_closed() -> None:
    _, status = apply_text("class IncrementalDetokenizer:\n    pass\n")
    assert status.startswith("missing:"), status
    assert "factory" in status


if __name__ == "__main__":
    test_apply_then_skip()
    test_missing_anchor_fails_closed()
    print("test_suppress_stops: ok")
