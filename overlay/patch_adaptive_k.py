#!/usr/bin/env python3
"""Adaptive *verification* length for DFlash2 (env-gated, default OFF).

Kit #139 inspiration, **not** a drop-in. On this --no-async-scheduling V2
stack, ``num_spec_tokens_to_schedule`` is the DFlash drafter length
(``num_query_per_req = 1 + num_speculative_steps``, grouped conv, selector,
draft-logit cache). Target verify query length already follows
``len(request.spec_token_ids)`` via ``scheduled_spec_decode_tokens``.

This overlay therefore:
  * trims only ``request.spec_token_ids`` in ``update_draft_token_ids``
  * observes accepted drafts in ``update_from_output``
  * captures extra **target** FULL graphs for verify query lens 3 and 5
  * **refuses** to write ``batch_k()`` into ``num_spec_tokens_to_schedule``

Knobs (read at container runtime):
  GLM53_ADAPTIVE_K            off (default) | ema | on | 1
  GLM53_ADAPTIVE_K_CAPTURE    0 (default) | 1  extra target graphs without
                              enabling the EMA policy (B0 capture-only)
  GLM53_ADAPTIVE_K_SET        candidate draft lengths, default "2,4,7"
  GLM53_ADAPTIVE_K_ALPHA      EMA alpha, default 0.25
  GLM53_ADAPTIVE_K_MARGIN     n = largest set value <= ceil(ema + margin)
  GLM53_ADAPTIVE_K_MIN_STEPS  full-k steps before trimming, default 4
  GLM53_ADAPTIVE_K_SATURATE   "max" (default) or "n"
  GLM53_ADAPTIVE_K_HIST       histogram every N steps, default 200
  GLM53_ADAPTIVE_K_FILE       optional JSON override; only when graphs existed
                              at boot

Structured-output requests and MIN_STEPS stay at full k=7. Batch-min keeps
FLASHINFER_MLA_SPARSE_SM120 / KDA FULL graphs uniform.

Install after patch_scheduler_decode_floor.py and patch_align_floor.py.
Fail closed on drifted schedule() / update_draft_token_ids / CUDA-graph
anchors. Idempotent. Not in the JIT shape hash by itself; extra capture
sizes are.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SCHED = Path(os.environ.get("GLM53_SCHEDULER_PY", SITE / "v1/core/sched/scheduler.py"))
CG = Path(
    os.environ.get(
        "GLM53_CUDAGRAPH_UTILS_PY", SITE / "v1/worker/gpu/cudagraph_utils.py"
    )
)
MARK = "# [glm53-adaptive-k]"
DRAFT_HOOK = "_GLM53_ADAPTIVE_K.batch_k("

SCHED_HELPER = '''
class _Glm53AdaptiveK:  # [glm53-adaptive-k]
    """CPU-only EMA policy for the *verified* draft prefix length.

    ``batch_k`` exists for unit tests and must not be wired into
    ``num_spec_tokens_to_schedule`` on this image (that field sizes the
    trained eight-row DFlash draft).
    """

    def __init__(self) -> None:
        mode = os.environ.get("GLM53_ADAPTIVE_K", "off").strip().lower()
        capture = os.environ.get("GLM53_ADAPTIVE_K_CAPTURE", "0").strip().lower()
        self.enabled = mode in ("ema", "on", "1")
        self.graphs_enabled = self.enabled or capture in ("1", "on", "true", "yes")
        self.alpha = float(os.environ.get("GLM53_ADAPTIVE_K_ALPHA", "0.25"))
        self.margin = float(os.environ.get("GLM53_ADAPTIVE_K_MARGIN", "1.0"))
        self.min_steps = int(os.environ.get("GLM53_ADAPTIVE_K_MIN_STEPS", "4"))
        raw = os.environ.get("GLM53_ADAPTIVE_K_SET", "2,4,7")
        self.k_set = sorted({int(x) for x in raw.split(",") if x.strip()})
        self.saturate = os.environ.get("GLM53_ADAPTIVE_K_SATURATE", "max").strip().lower()
        self.hist_every = int(os.environ.get("GLM53_ADAPTIVE_K_HIST", "200"))
        self.state: dict[str, list[float]] = {}  # req_id -> [ema, observed_steps]
        self.hist: dict[int, int] = {}
        self.steps = 0
        self.k_max = max(self.k_set) if self.k_set else 0
        self.boot_set = list(self.k_set)
        self.boot_enabled = self.enabled
        self.boot_graphs = self.graphs_enabled
        self.file = os.environ.get(
            "GLM53_ADAPTIVE_K_FILE", "/root/.cache/vllm/glm53_adaptive_k.json"
        )
        self.file_mtime = None
        self._reload()
        if self.enabled:
            print(
                f"[glm53-adaptive-k] enabled set={self.k_set} alpha={self.alpha} "
                f"margin={self.margin} min_steps={self.min_steps} "
                f"saturate={self.saturate} graphs={self.graphs_enabled}",
                flush=True,
            )
        elif self.graphs_enabled:
            print(
                f"[glm53-adaptive-k] capture-only graphs set={self.k_set} "
                "(policy off)",
                flush=True,
            )

    def _reload(self) -> None:
        if not self.boot_enabled:
            return
        try:
            mtime = os.stat(self.file).st_mtime
        except OSError:
            mtime = None
        if mtime == self.file_mtime:
            return
        self.file_mtime = mtime
        if mtime is None:
            return
        try:
            import json

            with open(self.file) as fh:
                cfg = json.load(fh)
            mode = str(cfg.get("mode", "ema")).strip().lower()
            self.enabled = mode in ("ema", "on", "1")
            self.alpha = float(cfg.get("alpha", self.alpha))
            self.margin = float(cfg.get("margin", self.margin))
            self.min_steps = int(cfg.get("min_steps", self.min_steps))
            self.saturate = str(cfg.get("saturate", self.saturate)).strip().lower()
            if "set" in cfg:
                want = {
                    int(x)
                    for x in (
                        cfg["set"]
                        if isinstance(cfg["set"], list)
                        else str(cfg["set"]).split(",")
                    )
                }
                self.k_set = sorted(want & set(self.boot_set)) or list(self.boot_set)
            self.state.clear()
            self.hist.clear()
            print(
                f"[glm53-adaptive-k] reloaded {self.file}: enabled={self.enabled} "
                f"set={self.k_set} alpha={self.alpha} margin={self.margin} "
                f"min_steps={self.min_steps} saturate={self.saturate}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[glm53-adaptive-k] override file ignored: {exc!r}", flush=True)

    def observe(self, req_id: str, num_draft: int, num_accepted: int) -> None:
        if not self.enabled or num_draft <= 0:
            return
        if num_accepted >= num_draft and self.saturate != "n":
            obs = float(max(self.k_max, num_draft))
        else:
            obs = float(num_accepted)
        st = self.state.get(req_id)
        if st is None:
            self.state[req_id] = [
                obs * self.alpha + float(self.k_max) * (1.0 - self.alpha),
                1.0,
            ]
        else:
            st[0] = obs * self.alpha + st[0] * (1.0 - self.alpha)
            st[1] += 1.0

    def choose(self, req_id: str, k: int, structured: bool):
        """Draft length for one request, or None when it must stay at full k."""
        if not self.enabled or structured or k <= 0:
            return None
        st = self.state.get(req_id)
        if st is None or st[1] < self.min_steps:
            return None
        import math

        target = int(math.ceil(st[0] + self.margin))
        cands = [v for v in self.k_set if v <= min(target, k)]
        n = max(cands) if cands else min(self.k_set)
        return max(1, min(n, k))

    def apply(self, reqs, live_ids) -> None:
        """Trim spec_token_ids to a uniform verified prefix.

        Structured-output or not-yet-observed requests pin the batch at
        the full length (uniform FULL graphs, nothing trimmed).
        """
        if self.steps % 50 == 0:
            self._reload()
        if not self.enabled or not reqs:
            self.steps += 1
            return
        ns = []
        for r, s in reqs:
            n_i = self.choose(r.request_id, len(r.spec_token_ids), s)
            if n_i is None:
                ns = None
                break
            ns.append(n_i)
        n = max(len(r.spec_token_ids) for r, _ in reqs) if ns is None else min(ns)
        for r, _ in reqs:
            if len(r.spec_token_ids) > n:
                r.spec_token_ids = r.spec_token_ids[:n]
        self._count(n, live_ids)

    def batch_k(self, k: int, reqs, live_ids) -> int:
        """Policy-only helper. Do not wire this into the drafter count."""
        if self.steps % 50 == 0:
            self._reload()
        if not self.enabled or k <= 0:
            self.steps += 1
            return k
        ns = []
        for r in reqs:
            if r is None or getattr(r, "is_prefill_chunk", False):
                continue
            n_i = self.choose(
                r.request_id, k, bool(getattr(r, "use_structured_output", False))
            )
            if n_i is None:
                ns = None
                break
            ns.append(n_i)
        n = k if not ns else min(ns)
        self._count(n, live_ids)
        return n

    def _count(self, n: int, live_ids) -> None:
        self.hist[n] = self.hist.get(n, 0) + 1
        self.steps += 1
        if self.hist_every > 0 and self.steps % self.hist_every == 0:
            total = sum(self.hist.values())
            parts = " ".join(f"{k}:{v}" for k, v in sorted(self.hist.items()))
            emas = " ".join(
                f"{rid[:8]}={st[0]:.2f}/{int(st[1])}"
                for rid, st in list(self.state.items())[:4]
            )
            print(
                f"[glm53-adaptive-k] step {self.steps} chosen-length hist "
                f"({total}): {parts} | ema {emas}",
                flush=True,
            )
            self.state = {rid: st for rid, st in self.state.items() if rid in live_ids}


_GLM53_ADAPTIVE_K = _Glm53AdaptiveK()  # [glm53-adaptive-k]


'''

OBS_OLD = """                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
"""
OBS_NEW = """                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                _GLM53_ADAPTIVE_K.observe(req_id, num_draft_tokens, num_accepted)  # [glm53-adaptive-k]
"""

UPD_OLD = """    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
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
"""
UPD_NEW = """    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        _ak_reqs = []  # [glm53-adaptive-k]
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
            _ak_structured = self.structured_output_manager.should_advance(request)  # [glm53-adaptive-k]
            if _ak_structured:
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids
            if _GLM53_ADAPTIVE_K.enabled and spec_token_ids:  # [glm53-adaptive-k]
                _ak_reqs.append((request, _ak_structured))
        if _ak_reqs:  # [glm53-adaptive-k]
            _GLM53_ADAPTIVE_K.apply(_ak_reqs, self.requests)
"""

# Present on this image; we verify it and deliberately do not rewrite it.
SCHED_K_OLD = """        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
"""

CG_HELPER = '''
def _glm53_adaptive_k_query_lens(lens, decode_query_len):  # [glm53-adaptive-k]
    """Extra *target* uniform decode graph lengths (k+1 for 2 and 4).

    Drafter graphs stay native query_len=8. Enable when the EMA policy is
    on or when GLM53_ADAPTIVE_K_CAPTURE=1 (B0 capture-only control).
    """
    import os

    mode = os.environ.get("GLM53_ADAPTIVE_K", "off").strip().lower()
    capture = os.environ.get("GLM53_ADAPTIVE_K_CAPTURE", "0").strip().lower()
    graphs = mode in ("ema", "on", "1") or capture in ("1", "on", "true", "yes")
    if not graphs:
        return lens
    raw = os.environ.get("GLM53_ADAPTIVE_K_SET", "2,4,7")
    ks = {int(x) for x in raw.split(",") if x.strip()}
    extra = {k + 1 for k in ks if 0 < k + 1 <= decode_query_len}
    out = sorted(set(lens) | extra | {decode_query_len})
    print(f"[glm53-adaptive-k] uniform decode graph query lens: {out}", flush=True)
    return out


'''
CG_ANCHOR = "@dataclass(frozen=True)\nclass BatchExecutionDescriptor:\n"
CG_OLD = """        else:
            decode_query_lens = [self.decode_query_len]
"""
CG_NEW = """        else:
            decode_query_lens = [self.decode_query_len]
        decode_query_lens = _glm53_adaptive_k_query_lens(decode_query_lens, self.decode_query_len)  # [glm53-adaptive-k]
"""

SCHED_NEEDLE = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"


def replace_once(path: Path, text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def _assert_no_draft_hook(path: Path, text: str) -> None:
    if DRAFT_HOOK in text:
        raise SystemExit(
            f"{path}: refused draft-facing {DRAFT_HOOK} wiring into "
            "num_spec_tokens_to_schedule (parked DFlash2 eight-row contract)"
        )
    if text.count(SCHED_K_OLD) != 1:
        raise SystemExit(
            f"{path}: num_spec_tokens_to_schedule anchor drifted; "
            "refusing to guess the drafter hook"
        )


def validate_scheduler(text: str) -> None:
    ast.parse(text, filename=str(SCHED))
    if text.count(MARK) < 4:
        raise SystemExit(f"{SCHED}: adaptive-k marker incomplete ({text.count(MARK)})")
    if text.count(SCHED_HELPER) != 1:
        raise SystemExit(f"{SCHED}: helper missing, duplicated, or drifted")
    if "_GLM53_ADAPTIVE_K.observe(" not in text:
        raise SystemExit(f"{SCHED}: observe hook missing")
    if "_GLM53_ADAPTIVE_K.apply(" not in text:
        raise SystemExit(f"{SCHED}: update_draft_token_ids apply hook missing")
    _assert_no_draft_hook(SCHED, text)
    if "import os\n" not in text.split("import time\n", 1)[0]:
        raise SystemExit(f"{SCHED}: helper requires a top-level plain import os")


def validate_cudagraph(text: str) -> None:
    ast.parse(text, filename=str(CG))
    if text.count(MARK) < 2:
        raise SystemExit(f"{CG}: adaptive-k graph marker incomplete")
    if text.count(CG_HELPER) != 1:
        raise SystemExit(f"{CG}: graph helper missing, duplicated, or drifted")
    if "_glm53_adaptive_k_query_lens(decode_query_lens" not in text:
        raise SystemExit(f"{CG}: decode_query_lens hook missing")


def patch_scheduler() -> None:
    if not SCHED.is_file():
        raise SystemExit(f"missing {SCHED}")
    text = SCHED.read_text()
    if MARK in text:
        validate_scheduler(text)
        print(f"{SCHED.name}: {MARK} already present and complete - skipping")
        return
    if "def _Glm53AdaptiveK" in text or "_GLM53_ADAPTIVE_K" in text:
        raise SystemExit(f"{SCHED}: partial or conflicting adaptive-k install")
    _assert_no_draft_hook(SCHED, text)
    if text.count(OBS_OLD) != 1:
        raise SystemExit(f"{SCHED}: observe target not unique ({text.count(OBS_OLD)})")
    if text.count(UPD_OLD) != 1:
        raise SystemExit(
            f"{SCHED}: update_draft_token_ids target not unique ({text.count(UPD_OLD)})"
        )
    if text.count(SCHED_NEEDLE) != 1:
        raise SystemExit(f"{SCHED}: helper insert point not unique")
    if "import os\n" not in text.split("import time\n", 1)[0]:
        raise SystemExit(
            f"{SCHED}: missing top-level import os "
            "(decode-floor / align-floor must install first)"
        )
    text = text.replace(SCHED_NEEDLE, SCHED_HELPER + SCHED_NEEDLE, 1)
    text = replace_once(SCHED, text, OBS_OLD, OBS_NEW, "observe")
    text = replace_once(SCHED, text, UPD_OLD, UPD_NEW, "update_draft_token_ids")
    validate_scheduler(text)
    tmp = SCHED.with_suffix(SCHED.suffix + ".glm53-adaptive-k.tmp")
    tmp.write_text(text)
    os.replace(tmp, SCHED)
    print(
        f"patched {SCHED.name} (GLM53_ADAPTIVE_K="
        f"{os.environ.get('GLM53_ADAPTIVE_K', 'off')})"
    )


def patch_cudagraph_utils() -> None:
    if not CG.is_file():
        raise SystemExit(f"missing {CG}")
    text = CG.read_text()
    if MARK in text:
        validate_cudagraph(text)
        print(f"{CG.name}: {MARK} already present and complete - skipping")
        return
    if "_glm53_adaptive_k_query_lens" in text:
        raise SystemExit(f"{CG}: partial or conflicting adaptive-k graph install")
    if text.count(CG_ANCHOR) != 1:
        raise SystemExit(f"{CG}: helper insert point not unique")
    if text.count(CG_OLD) != 1:
        raise SystemExit(f"{CG}: decode_query_lens target not unique")
    text = text.replace(CG_ANCHOR, CG_HELPER + CG_ANCHOR, 1)
    text = replace_once(CG, text, CG_OLD, CG_NEW, "decode_query_lens")
    validate_cudagraph(text)
    tmp = CG.with_suffix(CG.suffix + ".glm53-adaptive-k.tmp")
    tmp.write_text(text)
    os.replace(tmp, CG)
    print("patched cudagraph_utils.py (adaptive-k target graph lengths)")


def main() -> int:
    patch_scheduler()
    patch_cudagraph_utils()
    return 0


if __name__ == "__main__":
    sys.exit(main())
