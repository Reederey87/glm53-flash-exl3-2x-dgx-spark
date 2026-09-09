#!/usr/bin/env python3
"""Apply overlay/patch_adaptive_k.py and unit-test the verification-only EMA.

Hard gates:
  * default off never trims
  * capture-only graphs do not trim
  * structured and MIN_STEPS pin full k=7
  * batch minimum, saturate=max climb, saturate=n ratchet
  * extra target query lenses 3 and 5
  * installer never writes batch_k into num_spec_tokens_to_schedule
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = HERE.parent / "overlay" / "patch_adaptive_k.py"

SYNTHETIC_SCHED = '''import itertools
import os
import time
from vllm.compilation.cuda_graph import CUDAGraphStat


class Scheduler:
    def update_from_output(self):
        for req_id in ():
            scheduled_spec_token_ids = []
            generated_token_ids = []
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                _ = num_rejected

    def schedule(self):
        num_scheduled_tokens = {}
        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
        return num_spec_tokens_to_schedule

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids
'''

SYNTHETIC_CG = '''from dataclasses import dataclass


class CUDAGraphManager:
    def _init_candidates(self):
        if False:
            decode_query_lens = [1, 2, 3]
        else:
            decode_query_lens = [self.decode_query_len]
        return decode_query_lens


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    num_tokens: int
'''


class _Req:
    def __init__(self, rid, k=7):
        self.request_id = rid
        self.spec_token_ids = [-1] * k


def _patch(tmp: Path, env: dict[str, str] | None = None) -> tuple[str, str]:
    sched = tmp / "scheduler.py"
    cg = tmp / "cudagraph_utils.py"
    sched.write_text(SYNTHETIC_SCHED)
    cg.write_text(SYNTHETIC_CG)
    run_env = os.environ.copy()
    run_env["GLM53_SCHEDULER_PY"] = str(sched)
    run_env["GLM53_CUDAGRAPH_UTILS_PY"] = str(cg)
    if env:
        run_env.update(env)
    subprocess.check_call([sys.executable, str(PATCH)], env=run_env)
    return sched.read_text(), cg.read_text()


def policy_tests(helper_src: str) -> None:
    def make(env):
        ns = {"os": os}
        old = dict(os.environ)
        os.environ.update(env)
        try:
            exec(helper_src, ns)
            inst = ns["_Glm53AdaptiveK"]()
        finally:
            os.environ.clear()
            os.environ.update(old)
        return inst

    p = make({"GLM53_ADAPTIVE_K": "off"})
    r = _Req("a")
    for _ in range(10):
        p.observe("a", 7, 0)
    p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 7 and not p.enabled

    p = make({"GLM53_ADAPTIVE_K": "off", "GLM53_ADAPTIVE_K_CAPTURE": "1"})
    r = _Req("a")
    for _ in range(15):
        p.observe("a", 7, 1)
    p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 7 and not p.enabled
    assert p.graphs_enabled is True

    p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0"})
    r = _Req("a")
    for _ in range(3):
        p.observe("a", 7, 1)
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
        assert len(r.spec_token_ids) == 7, "min_steps guard"
    for _ in range(12):
        p.observe("a", len(r.spec_token_ids), 1)
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 2, r.spec_token_ids
    s = _Req("s")
    r.spec_token_ids = [-1] * 7
    p.apply([(r, False), (s, True)], {"a": r, "s": s})
    assert len(r.spec_token_ids) == 7 and len(s.spec_token_ids) == 7

    r.spec_token_ids = [-1] * 2
    for _ in range(20):
        p.observe("a", len(r.spec_token_ids), len(r.spec_token_ids))
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 7, r.spec_token_ids

    p = make(
        {
            "GLM53_ADAPTIVE_K": "ema",
            "GLM53_ADAPTIVE_K_SATURATE": "n",
            "GLM53_ADAPTIVE_K_HIST": "0",
        }
    )
    r = _Req("a")
    for _ in range(15):
        p.observe("a", 7, 1)
    r.spec_token_ids = [-1] * 7
    p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 2
    for _ in range(30):
        p.observe("a", 2, 2)
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 2, "saturate=n never climbs"

    class _SReq:
        def __init__(self, rid, structured=False, prefill=False):
            self.request_id = rid
            self.use_structured_output = structured
            self.is_prefill_chunk = prefill

    p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0"})
    a, b, s, pf = _SReq("a"), _SReq("b"), _SReq("s", structured=True), _SReq(
        "p", prefill=True
    )
    live = {"a": a, "b": b, "s": s, "p": pf}
    for _ in range(15):
        p.observe("a", 7, 1)
        p.observe("b", 7, 7)
    assert p.batch_k(7, [a], live) == 2
    assert p.batch_k(7, [b], live) == 7
    assert p.batch_k(7, [a, b], live) == 2
    assert p.batch_k(7, [a, s], live) == 7
    assert p.batch_k(7, [a, pf], live) == 2

    p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0"})
    a, b = _Req("a"), _Req("b")
    for _ in range(15):
        p.observe("a", 7, 7)
        p.observe("b", 7, 1)
    p.apply([(a, False), (b, False)], {"a": a, "b": b})
    assert len(a.spec_token_ids) == 2 and len(b.spec_token_ids) == 2

    with tempfile.TemporaryDirectory() as cfg_dir:
        cfg = Path(cfg_dir) / "glm53_adaptive_k.json"
        cfg.write_text('{"mode":"ema","set":"2,4,7","alpha":0.9,"min_steps":1}')
        p = make(
            {
                "GLM53_ADAPTIVE_K": "off",
                "GLM53_ADAPTIVE_K_CAPTURE": "1",
                "GLM53_ADAPTIVE_K_FILE": str(cfg),
            }
        )
        r = _Req("a")
        for _ in range(12):
            p.observe("a", 7, 1)
            r.spec_token_ids = [-1] * 7
            p.apply([(r, False)], {"a": r})
        assert not p.enabled
        assert p.graphs_enabled is True
        assert len(r.spec_token_ids) == 7, "capture-only must ignore runtime ema files"


def extra_capture_sizes() -> list[int]:
    stock = {1, 2, 4, 8, 16, 24, 32}
    query = {3, 5, 8}
    extra = {s * q for s in (1, 2, 3, 4) for q in query}
    return sorted(stock | extra)


def test_install_and_policy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        st, ct = _patch(tmp_path)
        assert "_GLM53_ADAPTIVE_K.batch_k(" not in st
        assert "num_spec_tokens_to_schedule = self.num_spec_tokens" in st
        assert st.count(
            "# Dynamic speculative decoding: compute optimal K\n"
            "        num_spec_tokens_to_schedule = self.num_spec_tokens\n"
            "        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:\n"
            "            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n"
            "                len(num_scheduled_tokens)\n"
            "            ]\n"
        ) == 1
        assert "_GLM53_ADAPTIVE_K.observe(" in st
        assert "_GLM53_ADAPTIVE_K.apply(" in st
        compile(st, "scheduler.py", "exec")
        compile(ct, "cudagraph_utils.py", "exec")
        st2, ct2 = _patch(tmp_path)
        assert st2 == st and ct2 == ct
        start = st.index("class _Glm53AdaptiveK:")
        end = st.index("_GLM53_ADAPTIVE_K = _Glm53AdaptiveK()")
        policy_tests(st[start:end])
        cstart = ct.index("def _glm53_adaptive_k_query_lens(")
        cend = ct.index("@dataclass(frozen=True)\nclass BatchExecutionDescriptor:")
        ns: dict = {}
        exec(ct[cstart:cend], ns)
        fn = ns["_glm53_adaptive_k_query_lens"]
        os.environ["GLM53_ADAPTIVE_K"] = "off"
        os.environ.pop("GLM53_ADAPTIVE_K_CAPTURE", None)
        assert fn([8], 8) == [8]
        os.environ["GLM53_ADAPTIVE_K_CAPTURE"] = "1"
        assert fn([8], 8) == [3, 5, 8]
        os.environ["GLM53_ADAPTIVE_K"] = "ema"
        os.environ["GLM53_ADAPTIVE_K_CAPTURE"] = "0"
        assert fn([8], 8) == [3, 5, 8]


def test_extra_capture_list_matches_b0() -> None:
    assert extra_capture_sizes() == [
        1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 32
    ]


def test_refuses_draft_hook_source() -> None:
    text = PATCH.read_text()
    assert 'num_spec_tokens_to_schedule = _GLM53_ADAPTIVE_K.batch_k(' not in text
    assert "refuses to write" in text or "Do not wire" in text


def test_live_extract_install_if_present() -> None:
    live_sched = Path("/tmp/ak-live/v1__core__sched__scheduler.py")
    live_cg = Path("/tmp/ak-live/v1__worker__gpu__cudagraph_utils.py")
    if not (live_sched.is_file() and live_cg.is_file()):
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sched = tmp_path / "scheduler.py"
        cg = tmp_path / "cudagraph_utils.py"
        shutil.copyfile(live_sched, sched)
        shutil.copyfile(live_cg, cg)
        env = os.environ.copy()
        env["GLM53_SCHEDULER_PY"] = str(sched)
        env["GLM53_CUDAGRAPH_UTILS_PY"] = str(cg)
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        st, ct = sched.read_text(), cg.read_text()
        assert "_GLM53_ADAPTIVE_K.batch_k(" not in st
        assert "_GLM53_ADAPTIVE_K.apply(" in st
        assert "_glm53_adaptive_k_query_lens(decode_query_lens" in ct
        compile(st, "scheduler.py", "exec")
        compile(ct, "cudagraph_utils.py", "exec")
        subprocess.check_call([sys.executable, str(PATCH)], env=env)


if __name__ == "__main__":
    test_install_and_policy()
    test_extra_capture_list_matches_b0()
    test_refuses_draft_hook_source()
    test_live_extract_install_if_present()
    print("adaptive-k patch OK")
