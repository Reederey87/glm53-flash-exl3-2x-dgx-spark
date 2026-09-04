#!/usr/bin/env python3
"""Right-size the GLM-5.3 sparse-indexer prefill workspace (opt-in).

The pinned vLLM preview sizes the shared sparse-indexer gather workspace as
``max_model_len * 40`` entries. At the production 1M-token window, with
132 bytes per FP8 entry plus the radix scratch, that locks 5036.40 MiB during
memory profiling.

GLM-5.3's kpool indexer consumes compressed sequence lengths. Its safe
per-step upper bound is:

    max_num_seqs * cdiv(max_model_len + num_speculative_tokens, index_kpool)

The patch is deliberately GLM-scoped. It adds a helper to the shared indexer
module, then changes only ``models/glm5next/nvidia/attention.py`` to call it
with that model instance's authoritative ``self.index_kpool``. The generic
``get_max_prefill_buffer_size`` function stays byte-for-byte stock, so other
models and the rank-0 DFlash2 drafter cannot enter the experiment.

Safety hardening required by the local review:

* a single row wider than the workspace raises instead of the stock
  ``if end == start`` fail-open;
* every emitted chunk is checked again before metadata construction and GPU
  gather;
* the GLM kpool operator compares each chunk against its actual allocation
  argument immediately before slicing the gather buffers;
* the disputed ``max_num_batched_tokens`` request multiplier is not used;
* rightsize mode must actually narrow the workspace, otherwise boot fails
  rather than reporting a successful experiment that kept stock sizing.

``GLM53_INDEXER_WORKSPACE``:

* ``stock`` (default) keeps the exact stock allocation;
* ``rightsize`` activates the GLM-only bound;
* any other value fails at process import.

All three target files are preflighted before any is written. Re-application is
idempotent, drift fails closed, writes use atomic replacement, and stale pyc
files are removed.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


INDEXER_TARGET = Path(
    os.environ.get(
        "GLM53_INDEXER_BACKEND_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/"
        "indexer.py",
    )
)
GLM_TARGET = Path(
    os.environ.get(
        "GLM53_GLM5NEXT_ATTENTION_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/"
        "attention.py",
    )
)
KPOOL_TARGET = Path(
    os.environ.get(
        "GLM53_INDEXER_KPOOL_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
        "sparse_attn_indexer_kpool.py",
    )
)

ENV_NAME = "GLM53_INDEXER_WORKSPACE"
STOCK_MULTIPLIER = 40
BYTES_PER_ENTRY = 132


HELPERS_SRC = '''

# [glm53-indexer-workspace] GLM-5.3-only prefill gather sizing.
_GLM53_WORKSPACE_ENV = "GLM53_INDEXER_WORKSPACE"


def _glm53_workspace_mode() -> str:
    """Return the literal workspace mode; only an unset value defaults."""
    raw = os.environ.get(_GLM53_WORKSPACE_ENV)
    if raw is None:
        return "stock"
    if raw not in ("stock", "rightsize"):
        raise ValueError(
            f"{_GLM53_WORKSPACE_ENV} must be exactly one of: stock rightsize "
            f"(got: {raw!r})"
        )
    return raw


def _glm53_glm5next_workspace_entries(
    vllm_config: VllmConfig,
    index_kpool: int,
) -> int:
    """Return the GLM kpool workspace, never a generic model workspace."""
    stock = get_max_prefill_buffer_size(vllm_config)
    if _glm53_workspace_mode() == "stock":
        return stock

    ratio = int(index_kpool)
    if ratio <= 1:
        raise ValueError(
            "[glm53-indexer-workspace] rightsize requires GLM index_kpool > 1 "
            f"(got {ratio})"
        )
    max_model_len = int(vllm_config.model_config.max_model_len)
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    if max_model_len <= 0 or max_num_seqs <= 0:
        raise ValueError(
            "[glm53-indexer-workspace] rightsize requires positive "
            f"max_model_len/max_num_seqs (got {max_model_len}/{max_num_seqs})"
        )
    spec_cfg = getattr(vllm_config, "speculative_config", None)
    num_spec = int(getattr(spec_cfg, "num_speculative_tokens", 0) or 0)
    span = max_model_len + max(0, num_spec)
    per_req = -(-span // ratio)
    entries = per_req * max_num_seqs

    # A rightsize arm that silently kept stock would produce a false-positive
    # deployment receipt. Fail readiness instead.
    if entries >= stock:
        raise ValueError(
            "[glm53-indexer-workspace] rightsize does not narrow the stock "
            f"workspace: computed={entries}, stock={stock}, "
            f"index_kpool={ratio}, max_num_seqs={max_num_seqs}"
        )

    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available()
        and torch.distributed.is_initialized()
        else -1
    )
    logger.info(
        "[glm53-indexer-workspace] role=target rank=%d mode=rightsize "
        "index_kpool=%d max_num_seqs=%d entries=%d stock=%d bytes=%d "
        "reclaimed_mib=%.1f",
        rank,
        ratio,
        max_num_seqs,
        entries,
        stock,
        entries * 132,
        (stock - entries) * 132 / (1024 * 1024),
    )
    return entries
'''

class _HostDistributed:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def is_initialized() -> bool:
        return False


class _HostTorch:
    distributed = _HostDistributed()


_HELPER_NS: dict = {"os": os, "torch": _HostTorch()}


def _stock_size(config) -> int:
    return int(config.model_config.max_model_len) * STOCK_MULTIPLIER


_HELPER_NS["get_max_prefill_buffer_size"] = _stock_size
_HELPER_NS["logger"] = type(
    "_NullLogger",
    (),
    {"info": staticmethod(lambda *_args, **_kwargs: None)},
)()
_HELPER_NS["VllmConfig"] = object
exec(
    compile(HELPERS_SRC, "<glm53-indexer-workspace helpers>", "exec"),
    _HELPER_NS,
)
workspace_mode = _HELPER_NS["_glm53_workspace_mode"]
glm5next_workspace_entries = _HELPER_NS["_glm53_glm5next_workspace_entries"]


def split_prefill_chunks(
    compressed_seq_lens: list[int],
    query_lens: list[int],
    workspace_size: int,
    max_logits_bytes: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Host replica of the patched splitter used by CPU-only tests."""
    chunks: list[tuple[tuple[int, int], tuple[int, int]]] = []
    n = len(compressed_seq_lens)
    max_logits_elems = max_logits_bytes // 4
    end = 0
    while end < n:
        start, chunk_m, chunk_n = end, 0, 0
        while end < n:
            q, s = query_lens[end], compressed_seq_lens[end]
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break
        if end == start:
            chunk_m, chunk_n = query_lens[end], compressed_seq_lens[end]
            if chunk_n > workspace_size:
                raise ValueError(
                    "sparse-indexer row exceeds workspace: "
                    f"row={chunk_n}, workspace={workspace_size}"
                )
            end += 1
        if chunk_n > workspace_size:
            raise AssertionError(
                "sparse-indexer chunk exceeds workspace: "
                f"chunk={chunk_n}, workspace={workspace_size}"
            )
        req_slice = (start, end)
        max_q = (
            max(1, max_logits_elems // chunk_n)
            if chunk_n > 0
            else max(1, chunk_m)
        )
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, (q_off, q_off + sub_m)))
    return chunks


MARK_HELPERS = "# [glm53-indexer-workspace] GLM-5.3-only prefill gather sizing.\n"
MARK_IMPORT = "import os  # [glm53-indexer-workspace]\n"
ANCHOR_IMPORT = """from dataclasses import dataclass

import torch
"""
PATCHED_IMPORT = """from dataclasses import dataclass

import os  # [glm53-indexer-workspace]
import torch
"""
ANCHOR_HELPERS = """def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40
"""
PATCHED_HELPERS = ANCHOR_HELPERS + HELPERS_SRC

MARK_SPLITTER = "            # [glm53-indexer-workspace] N cannot be sub-chunked.\n"
ANCHOR_SPLITTER = """        # A single request can exceed the budget, requiring sub-chunking
        # on the query dimension.
        if end == start:
            chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            end += 1

        req_slice = slice(start + request_offset, end + request_offset)
"""
PATCHED_SPLITTER = """        # A single request can exceed the logits budget, requiring
        # sub-chunking on the query dimension. N cannot be sub-chunked: the
        # stock fail-open admitted the oversized row anyway, so right-sizing
        # could overrun the locked gather workspace.
        if end == start:
            chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            # [glm53-indexer-workspace] N cannot be sub-chunked.
            if chunk_n > workspace_size:
                raise ValueError(
                    "[glm53-indexer-workspace] sparse-indexer row exceeds "
                    f"workspace: row={chunk_n}, workspace={workspace_size}, "
                    f"request={end + request_offset}"
                )
            end += 1

        if chunk_n > workspace_size:
            raise AssertionError(
                "[glm53-indexer-workspace] emitted chunk exceeds workspace: "
                f"chunk={chunk_n}, workspace={workspace_size}, "
                f"requests={start + request_offset}:{end + request_offset}"
            )
        req_slice = slice(start + request_offset, end + request_offset)
"""

MARK_CHUNK_ASSERT = (
    "                # [glm53-indexer-workspace] Final guard before GPU gather.\n"
)
ANCHOR_CHUNK_ASSERT = """                # Skip when total_seq_lens is 0 (i.e., no compressed token).
                if metadata is not None:
                    chunks.append(metadata)
"""
PATCHED_CHUNK_ASSERT = """                # Skip when total_seq_lens is 0 (i.e., no compressed token).
                if metadata is not None:
                    # [glm53-indexer-workspace] Final guard before GPU gather.
                    if metadata.total_seq_lens > self.max_prefill_buffer_size:
                        raise AssertionError(
                            "[glm53-indexer-workspace] metadata chunk exceeds "
                            f"workspace: chunk={metadata.total_seq_lens}, "
                            f"workspace={self.max_prefill_buffer_size}"
                        )
                    chunks.append(metadata)
"""

INDEXER_SITES = (
    ("workspace environment import", MARK_IMPORT, ANCHOR_IMPORT, PATCHED_IMPORT),
    ("helpers", MARK_HELPERS, ANCHOR_HELPERS, PATCHED_HELPERS),
    ("splitter fail-closed", MARK_SPLITTER, ANCHOR_SPLITTER, PATCHED_SPLITTER),
    (
        "pre-gather chunk assertion",
        MARK_CHUNK_ASSERT,
        ANCHOR_CHUNK_ASSERT,
        PATCHED_CHUNK_ASSERT,
    ),
)

MARK_GLM_IMPORT = (
    "            _glm53_glm5next_workspace_entries,  "
    "# [glm53-indexer-workspace]\n"
)
ANCHOR_GLM_IMPORT = """        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
"""
PATCHED_GLM_IMPORT = """        from vllm.v1.attention.backends.mla.indexer import (
            _glm53_glm5next_workspace_entries,  # [glm53-indexer-workspace]
        )

        # GLM-scoped by construction: the generic helper remains untouched,
        # and the authoritative kpool value is the same instance value used
        # to build the compressed KV-cache spec above.
        self.max_total_seq_len = _glm53_glm5next_workspace_entries(
            vllm_config,
            self.index_kpool,
        )
"""

MARK_KPOOL_CAP = (
    "            # [glm53-indexer-workspace] Guard the actual GLM allocation.\n"
)
ANCHOR_KPOOL_CAP = """        for chunk in prefill_metadata.chunks if not short_prefill else ():
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
"""
PATCHED_KPOOL_CAP = """        for chunk in prefill_metadata.chunks if not short_prefill else ():
            # [glm53-indexer-workspace] Guard the actual GLM allocation.
            if index_kpool > 1 and chunk.total_seq_lens > total_seq_lens:
                raise ValueError(
                    "[glm53-indexer-workspace] gather chunk exceeds allocated "
                    f"workspace: chunk={chunk.total_seq_lens}, "
                    f"allocated={total_seq_lens}"
                )
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
"""


def _verified(text: str, sites) -> bool:
    return all(
        text.count(mark) == 1
        and text.count(patched) == 1
        and text.count(anchor) == patched.count(anchor)
        for _name, mark, anchor, patched in sites
    )


def _prepare(text: str, sites, label: str) -> tuple[str, str]:
    marks = sum(text.count(mark) for _name, mark, _anchor, _patched in sites)
    if marks:
        if marks != len(sites) or not _verified(text, sites):
            raise ValueError(
                f"partial/inconsistent {label} patch "
                f"(marks={marks}, expected={len(sites)})"
            )
        return text, "already present"

    out = text
    for name, _mark, anchor, patched in sites:
        count = out.count(anchor)
        if count != 1:
            raise ValueError(
                f"pinned {label} anchor {name!r} drifted "
                f"(found {count}, expected 1)"
            )
        out = out.replace(anchor, patched, 1)
    if not _verified(out, sites):
        raise ValueError(f"{label} post-patch verification failed")
    return out, "patched"


def prepare_indexer(text: str) -> tuple[str, str]:
    return _prepare(text, INDEXER_SITES, "indexer")


def prepare_glm(text: str) -> tuple[str, str]:
    sites = (("GLM call site", MARK_GLM_IMPORT, ANCHOR_GLM_IMPORT, PATCHED_GLM_IMPORT),)
    return _prepare(text, sites, "GLM attention")


def prepare_kpool(text: str) -> tuple[str, str]:
    sites = (
        (
            "actual allocation guard",
            MARK_KPOOL_CAP,
            ANCHOR_KPOOL_CAP,
            PATCHED_KPOOL_CAP,
        ),
    )
    return _prepare(text, sites, "kpool operator")


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-indexer-workspace.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob(f"{target.stem}*.pyc"):
        pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    preflight_only = "--preflight" in argv[1:]

    for target in (INDEXER_TARGET, GLM_TARGET, KPOOL_TARGET):
        if not target.is_file():
            raise SystemExit(f"missing {target}")

    indexer_source = INDEXER_TARGET.read_text()
    glm_source = GLM_TARGET.read_text()
    kpool_source = KPOOL_TARGET.read_text()
    try:
        indexer_patched, indexer_action = prepare_indexer(indexer_source)
        glm_patched, glm_action = prepare_glm(glm_source)
        kpool_patched, kpool_action = prepare_kpool(kpool_source)
    except ValueError as exc:
        raise SystemExit(f"indexer workspace preflight failed: {exc}") from exc

    compile(indexer_patched, str(INDEXER_TARGET), "exec")
    compile(glm_patched, str(GLM_TARGET), "exec")
    compile(kpool_patched, str(KPOOL_TARGET), "exec")
    if preflight_only:
        print(
            "indexer workspace preflight OK "
            f"(indexer={indexer_action}, glm={glm_action}, "
            f"kpool={kpool_action})"
        )
        return 0

    if indexer_patched != indexer_source:
        replace_file(INDEXER_TARGET, indexer_patched)
        clear_pyc(INDEXER_TARGET)
    if glm_patched != glm_source:
        replace_file(GLM_TARGET, glm_patched)
        clear_pyc(GLM_TARGET)
    if kpool_patched != kpool_source:
        replace_file(KPOOL_TARGET, kpool_patched)
        clear_pyc(KPOOL_TARGET)
    print(
        "indexer workspace "
        f"(indexer={indexer_action}, glm={glm_action}, "
        f"kpool={kpool_action}, "
        f"{ENV_NAME}={os.environ.get(ENV_NAME, 'stock (unset)')!r})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
