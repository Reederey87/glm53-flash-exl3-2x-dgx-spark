#!/usr/bin/env python3
"""Grace a prefix for the tool round-trip, without pinning it.

The engine frees a request's blocks before the chat layer knows the stop
was a tool call. This patch snapshots the hashed ids at that free, and the
chat layer confirms them only when it rewrites ``finish_reason`` to
``tool_calls``. ``cache_tail_evict`` then spends one-shot pages before
those ids, and a reused page after them. A plain stop never confirms, so
it stays one-shot.

GLM53_TOOL_RETURN_GRACE=1 applies the three edits. 0 leaves the installed
sources untouched. Rollback is the flag at 0 and a new container. The
grace state itself lives in the bind-mounted ``cache_tail_evict`` module.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MARK = "[glm53-tool-return-grace]"

KV = Path(
    os.environ.get(
        "GLM53_KV_MANAGER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_manager.py",
    )
)
CORE = Path(
    os.environ.get(
        "GLM53_ENGINE_CORE_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py",
    )
)
SERVING = Path(
    os.environ.get(
        "GLM53_CHAT_SERVING_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/"
        "chat_completion/serving.py",
    )
)

FREE_OLD = """\
            request: The request to free the blocks.
        \"\"\"
        pins = self._partial_tail_pins.pop(request.request_id, None)
"""

FREE_NEW = """\
            request: The request to free the blocks.
        \"\"\"
        # [glm53-tool-return-grace] snapshot hashed ids before they return
        # to the pool. The confirm is a later utility call, and only for a
        # tool-call finish. A failure here must not skip the free.
        try:
            import sys as _glm53_sys
            if "/opt/glm53" not in _glm53_sys.path:
                _glm53_sys.path.insert(0, "/opt/glm53")
            from cache_tail_evict import remember_snapshot as _glm53_remember
            _glm53_pairs = []
            _glm53_seen: set[int] = set()
            _glm53_pins = self._partial_tail_pins.get(request.request_id) or ()
            _glm53_groups = self.get_blocks(request.request_id).blocks
            for _glm53_block in list(_glm53_pins) + [
                block for group in _glm53_groups for block in group
            ]:
                if (
                    _glm53_block.block_hash is None
                    or _glm53_block.block_id in _glm53_seen
                ):
                    continue
                _glm53_seen.add(_glm53_block.block_id)
                _glm53_pairs.append(
                    (
                        _glm53_block.block_id,
                        _glm53_block.block_hash,
                        _glm53_block.block_hash_num_tokens or 0,
                    )
                )
            _glm53_remember(
                request.request_id, _glm53_pairs,
                getattr(request, "_glm53_external_request_id", None),
            )
        except Exception:
            import traceback
            traceback.print_exc()
        pins = self._partial_tail_pins.pop(request.request_id, None)
"""

CORE_OLD = """\
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
"""

CORE_NEW = """\
        _glm53_reset = self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )
        if _glm53_reset:
            import sys as _glm53_sys
            if "/opt/glm53" not in _glm53_sys.path:
                _glm53_sys.path.insert(0, "/opt/glm53")
            from cache_tail_evict import reset_grace_state
            reset_grace_state()
        return _glm53_reset

    def glm53_mark_tool_grace(self, request_id: str) -> int:
        \"\"\"[glm53-tool-return-grace] Mark the request's still-matching pages.

        Called by the chat layer only after finish_reason becomes
        tool_calls. The block pool lives in this process, so the mark has
        to cross from the API process as a utility call.
        \"\"\"
        import sys as _glm53_sys

        if "/opt/glm53" not in _glm53_sys.path:
            _glm53_sys.path.insert(0, "/opt/glm53")
        from cache_tail_evict import confirm_tool_grace as _glm53_confirm

        return _glm53_confirm(
            request_id, self.scheduler.kv_cache_manager.block_pool
        )

    def reset_encoder_cache(self) -> None:
"""

ID_OLD = """\
        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
"""
ID_NEW = """\
        req = Request.from_engine_core_request(request, self.request_block_hasher)
        # [glm53-tool-return-grace] preserve the explicit API/engine id mapping.
        req._glm53_external_request_id = request.external_req_id
        if req.use_structured_output:
"""

STREAM_OLD = """\
                        if tools_streamed[i] and not tool_choice_function_name:
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )
"""

STREAM_NEW = """\
                        if tools_streamed[i] and not tool_choice_function_name:
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )
                        # [glm53-tool-return-grace] confirm only this rewrite.
                        # A mark failure must not drop the finished stream.
                        if finish_reason_ == "tool_calls":
                            try:
                                await self.engine_client.engine_core.call_utility_async(
                                    "glm53_mark_tool_grace",
                                    getattr(res, "request_id", None) or request_id,
                                )
                            except Exception:
                                logger.exception(
                                    "[glm53-tool-return-grace] mark failed"
                                )
"""

FULL_OLD = """\
            is_finish_reason_tool_calls = auto_tools_called or (
                request.tool_choice
                and request.tool_choice == "required"
                and output.finish_reason == "stop"
            )
"""

FULL_NEW = """\
            is_finish_reason_tool_calls = auto_tools_called or (
                request.tool_choice
                and request.tool_choice == "required"
                and output.finish_reason == "stop"
            )
            # [glm53-tool-return-grace] confirm only this rewrite.
            # A mark failure must not drop the finished response.
            if is_finish_reason_tool_calls:
                try:
                    await self.engine_client.engine_core.call_utility_async(
                        "glm53_mark_tool_grace",
                        getattr(final_res, "request_id", None) or request_id,
                    )
                except Exception:
                    logger.exception(
                        "[glm53-tool-return-grace] mark failed"
                    )
"""


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".glm53-tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main() -> int:
    flag = os.environ.get("GLM53_TOOL_RETURN_GRACE", "0").strip()
    if flag == "0":
        print(
            "GLM53_TOOL_RETURN_GRACE is off — installed sources unchanged",
            flush=True,
        )
        return 0
    if flag != "1":
        print(
            f"GLM53_TOOL_RETURN_GRACE must be exactly 0 or 1 (got: {flag!r})",
            file=sys.stderr,
            flush=True,
        )
        return 1
    paths = (KV, CORE, SERVING)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        print(f"{MARK} missing {missing}", file=sys.stderr, flush=True)
        return 1
    texts = [path.read_text() for path in paths]
    installed = (
        texts[0].count(FREE_NEW) == 1,
        texts[1].count(CORE_NEW) == 1 and texts[1].count(ID_NEW) == 1,
        texts[2].count(STREAM_NEW) == 1 and texts[2].count(FULL_NEW) == 1,
    )
    if all(installed):
        print(f"{MARK} already present", flush=True)
        return 0
    if any(MARK in text for text in texts):
        print(f"{MARK} partial install", file=sys.stderr, flush=True)
        return 1
    counts = (
        texts[0].count(FREE_OLD),
        texts[1].count(CORE_OLD),
        texts[2].count(STREAM_OLD),
        texts[2].count(FULL_OLD),
        texts[1].count(ID_OLD),
    )
    if counts != (1, 1, 1, 1, 1):
        print(f"{MARK} anchor drift: {counts}", file=sys.stderr, flush=True)
        return 1
    new_kv = texts[0].replace(FREE_OLD, FREE_NEW, 1)
    new_core = texts[1].replace(CORE_OLD, CORE_NEW, 1).replace(ID_OLD, ID_NEW, 1)
    new_serving = texts[2].replace(STREAM_OLD, STREAM_NEW, 1).replace(
        FULL_OLD, FULL_NEW, 1
    )
    if (
        new_kv.count(MARK) != 1
        or new_core.count(MARK) != 2
        or new_serving.count(MARK) != 4
    ):
        print(f"{MARK} replacement did not apply cleanly", file=sys.stderr, flush=True)
        return 1
    for path, text in ((KV, new_kv), (CORE, new_core), (SERVING, new_serving)):
        compile(text, str(path), "exec")
    _write(KV, new_kv)
    _write(CORE, new_core)
    _write(SERVING, new_serving)
    print(f"{MARK} applied to kv manager, engine core, and chat serving", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
