#!/usr/bin/env python3
"""Launch shared experts before the routed dispatch, not after it.

On image glm53-selfbuild:e3-armc-guards (fork 487ecf187),
``SharedExperts.maybe_sync_shared_experts_stream`` only marks a start point on
the auxiliary stream, and the shared experts are actually enqueued in
``forward`` -- which ``MoERunner._apply_quant_method`` calls *after*
``forward_modular`` has enqueued the routed experts. The aux stream is therefore
handed the work last. On a CPU-bound decode step the routed experts are already
enqueued, and usually already running, before the aux stream sees the launch, so
the two do not overlap in practice: the shared expert serializes behind the
routed MoE on every sparse layer.

This model makes that expensive. GLM-5.3-Flash-ExL3 keeps the shared expert in
bf16 (``scope: glm53_routed_experts_only``, ``n_shared_experts: 1``,
``moe_intermediate_size: 2048``) while the routed experts are 4-bit trellis.
Per token the shared expert reads ``3 * 4096 * 2048`` bf16 weights = 50.3 MB per
layer, against ~100 MB for the eight activated 4-bit experts, so it is roughly a
third of the MoE weight traffic per step on each of the 42 sparse layers.

The edit moves the launch to the sync point, before the gate and the routed
dispatch, and carries the dependency on explicit CUDA events instead of a raw
``wait_stream``: the aux stream waits only for the activation to be produced,
not for the gate or the routed experts. ``forward`` then becomes a join.

Correctness notes:

* the activation is read-only on both paths -- ``apply_exl3_experts`` reshapes
  and returns a new tensor, it never writes into ``x`` -- so no input clone is
  required;
* the event pair and the in-flight flag are allocated in ``__init__``, and the
  in-flight flag is cleared inside the same forward, so a captured CUDA graph
  replays the recorded fork/join with no host state to re-enter;
* ``forward`` joins before returning, so the caller still sees a finished
  output, and the main stream is never left waiting on the aux stream.

GLM53_SHARED_EXPERTS_EARLY=1 applies the edit. 0 leaves the installed file
untouched. Rollback is the flag at 0 and a new container.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MARK = "[glm53-shared-experts-early]"
TARGET = Path(
    os.environ.get(
        "GLM53_SHARED_EXPERTS_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
        "fused_moe/runner/shared_experts.py",
    )
)

# --- anchor 1: allocate the event pair and the in-flight flag ----------------
# The trailing comment line is part of the anchor so that the replacement is not
# a superstring of the original text (which would make an applied file look
# pristine again).
INIT_OLD = """\
            self._stream = aux_stream()
            if self._stream is not None:
                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")

    # TODO(bnell): Hack for elastic_ep. Get rid of this
"""

INIT_NEW = """\
            self._stream = aux_stream()
            if self._stream is not None:
                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")
        # [glm53-shared-experts-early] One event pair and one in-flight flag per
        # DBO ubatch id, allocated unconditionally: forward() reads the flag even
        # when the aux stream is disabled (VLLM_DISABLE_SHARED_EXPERTS_STREAM) or
        # unavailable on a non-cuda-alike platform, where stock runs the layer
        # inline. Allocated at init, so nothing is created inside a captured
        # region.
        self._early_input_ready = [torch.cuda.Event(), torch.cuda.Event()]
        self._early_output_ready = [torch.cuda.Event(), torch.cuda.Event()]
        self._early_pending = [False, False]

    # TODO(bnell): Hack for elastic_ep. Get rid of this
"""

# --- anchor 2: launch at the sync point -------------------------------------
SYNC_OLD = """\
            shared_experts_input.record_stream(self._stream)

            # Mark sync start point for the aux stream since we will
            # run in parallel with router/gate.
            self._stream.wait_stream(current_stream())
"""

SYNC_NEW = """\
            shared_experts_input.record_stream(self._stream)

            # [glm53-shared-experts-early] Launch here, at the sync point
            # before the gate and the routed dispatch, instead of in forward()
            # after the routed experts. Enqueued last, this work arrives on the
            # aux stream only once the routed experts are already enqueued, so
            # the two do not overlap on a CPU-bound decode step. The events
            # carry the real dependency: the aux stream waits for this tensor,
            # not for the gate or the routed experts.
            _early_idx = self._output_idx
            assert self._output[_early_idx] is None
            self._early_input_ready[_early_idx].record(current_stream())
            with torch.cuda.stream(self._stream):
                self._early_input_ready[_early_idx].wait(self._stream)
                self._output[_early_idx] = self._layer(shared_experts_input)
                self._early_output_ready[_early_idx].record(self._stream)
            self._early_pending[_early_idx] = True
"""

# --- anchor 3: forward joins the launch instead of repeating it -------------
FORWARD_OLD = """\
        if order != experts_order:
            return None

        assert self._output[self._output_idx] is None

        if order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            self._output[self._output_idx] = self._run_in_aux_stream(
                shared_experts_input
            )
        else:
            self._output[self._output_idx] = self._layer(shared_experts_input)
"""

FORWARD_NEW = """\
        if order != experts_order:
            return None

        if self._early_pending[self._output_idx]:
            # [glm53-shared-experts-early] Already enqueued on the aux stream at
            # the sync point. Only join it here: the output slot is filled and
            # re-running the layer would duplicate the work.
            self._early_pending[self._output_idx] = False
            self._early_output_ready[self._output_idx].wait(current_stream())
            assert self._output[self._output_idx] is not None
            return

        assert self._output[self._output_idx] is None

        if order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            self._output[self._output_idx] = self._run_in_aux_stream(
                shared_experts_input
            )
        else:
            self._output[self._output_idx] = self._layer(shared_experts_input)
"""

EDITS = (
    ("init", INIT_OLD, INIT_NEW),
    ("sync", SYNC_OLD, SYNC_NEW),
    ("forward", FORWARD_OLD, FORWARD_NEW),
)


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".glm53-tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _drop_pyc(path: Path) -> None:
    cache = path.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob(f"{path.stem}.*.pyc"):
        try:
            pyc.unlink()
        except OSError:
            pass


def _state(text: str) -> str:
    """Classify the installed file: 'clean', 'applied', or 'refuse:<reason>'."""
    marks = text.count(MARK)
    if marks not in (0, len(EDITS)):
        return f"refuse:partial install, marker appears {marks} time(s)"
    expect = ("clean" if marks == 0 else "applied")
    for name, old, new in EDITS:
        old_n, new_n = text.count(old), text.count(new)
        if expect == "clean":
            if old_n == 1 and new_n == 0:
                continue
            return (
                f"refuse:anchor drift at '{name}': old={old_n} new={new_n} "
                "(expected old=1 new=0)"
            )
        if old_n == 0 and new_n == 1:
            continue
        return (
            f"refuse:partial install at '{name}': old={old_n} new={new_n} "
            "(expected old=0 new=1)"
        )
    return expect


def main() -> int:
    flag = os.environ.get("GLM53_SHARED_EXPERTS_EARLY", "0").strip()
    if flag == "0":
        print(
            "GLM53_SHARED_EXPERTS_EARLY is off — installed sources unchanged",
            flush=True,
        )
        return 0
    if flag != "1":
        print(
            "GLM53_SHARED_EXPERTS_EARLY must be exactly 0 or 1 "
            f"(got: {flag!r})",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if not TARGET.is_file():
        print(f"{MARK} missing {TARGET}", file=sys.stderr, flush=True)
        return 1

    text = TARGET.read_text()
    state = _state(text)
    if state.startswith("refuse:"):
        print(
            f"{MARK} refusing to patch {TARGET}: {state[len('refuse:'):]}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if state == "applied":
        print(f"{MARK} already present in {TARGET}", flush=True)
        return 0

    patched = text
    for _, old, new in EDITS:
        patched = patched.replace(old, new, 1)

    try:
        compile(patched, str(TARGET), "exec")
    except SyntaxError as exc:
        print(f"{MARK} patched source does not compile: {exc}", file=sys.stderr, flush=True)
        return 1

    _write(TARGET, patched)
    _drop_pyc(TARGET)
    print(f"{MARK} applied to {TARGET}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
