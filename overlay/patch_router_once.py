#!/usr/bin/env python3
"""Skip the router GEMM that MoERunner discards.

On image glm53-selfbuild:e3-armc-guards (fork 487ecf187), Glm5NextMoE.forward
always runs ``self.gate(hidden_states)`` and passes the logits to the runner.
MoERunner._forward_impl then overwrites those logits whenever ``self.gate``
is set, which this model does: FusedMoEFactory is called with ``gate=self.gate``
and returns the runner. The first GEMM is a 288x4096 bf16 read on every MoE
layer, and its result is never routed.

Current upstream already passes the hidden state through and lets the runner
own the GEMM. This overlay does the same, with one extra check: the skip
happens only when ``experts.gate is self.gate``. A runner that does not hold
the gate still receives logits computed here.

GLM53_ROUTER_ONCE=1 applies the edit. 0 leaves the installed file untouched.
Rollback is the flag at 0 and a new container. No Triton specialization.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MARK = "[glm53-router-once]"
MODEL = Path(
    os.environ.get(
        "GLM53_GLM5NEXT_MODEL_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/"
        "glm5next/nvidia/model.py",
    )
)

OLD = """\
        # The router is always external (self.gate); main's MoERunner expects
        # pre-computed router_logits, so compute them here unconditionally.
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
"""

NEW = """\
        # [glm53-router-once] The runner overwrites router_logits when it
        # holds this gate (MoERunner._forward_impl). The GEMM here is then
        # a discarded 288x4096 bf16 read on every MoE layer. Pass the hidden
        # state through in that case. A runner without this gate still
        # consumes the logits computed here.
        if getattr(self.experts, "gate", None) is self.gate:
            router_logits = hidden_states
        else:
            router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
"""


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".glm53-tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _already_once(text: str) -> bool:
    """Upstream already lets the runner own the GEMM."""
    if "self.gate(hidden_states)" in text:
        return False
    return "router_logits=hidden_states" in text or "router_logits = hidden_states" in text


def main() -> int:
    flag = os.environ.get("GLM53_ROUTER_ONCE", "0").strip()
    if flag == "0":
        print("GLM53_ROUTER_ONCE is off — installed sources unchanged", flush=True)
        return 0
    if flag != "1":
        print(
            f"GLM53_ROUTER_ONCE must be exactly 0 or 1 (got: {flag!r})",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if not MODEL.is_file():
        print(f"{MARK} missing {MODEL}", file=sys.stderr, flush=True)
        return 1
    text = MODEL.read_text()
    if MARK in text and text.count(NEW) == 1 and text.count(OLD) == 0:
        print(f"{MARK} already present in {MODEL}", flush=True)
        return 0
    if MARK in text or (NEW in text and OLD in text):
        print(f"{MARK} partial install in {MODEL}", file=sys.stderr, flush=True)
        return 1
    if text.count(OLD) == 1:
        patched = text.replace(OLD, NEW, 1)
        compile(patched, str(MODEL), "exec")
        _write(MODEL, patched)
        print(f"{MARK} applied to {MODEL}", flush=True)
        return 0
    if _already_once(text):
        print(f"{MARK} runner already owns the router GEMM in {MODEL}", flush=True)
        return 0
    print(
        f"{MARK} anchor drift in {MODEL}: old={text.count(OLD)}",
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
