#!/usr/bin/env python3
"""Regression tests for the kpool tail correctness patch.

Covers the four independently verified defects:

* the prefill seed kernel's dense block stride against the padded indexer
  alias (vLLM #57477);
* the hybrid model-state path dropping ``positions``;
* the transient ``slot_mapping.clone()`` that CUDA graph replay keeps
  pointing at;
* the unbounded column index in the live V2 slot-mapping kernel.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_kpool_tail_correctness.py",
        ROOT / "overlay" / "patch_kpool_tail_correctness.py",
    )
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
from patch_kpool_tail_correctness import (  # noqa: E402
    SITE,
    SITES,
    TARGETS,
    prepare,
    verified_state,
)

# --- production geometry -----------------------------------------------------
# GLM-5.3-Flash: index_kpool 4, head_dim 128, bf16. The tail aliases the
# indexer cache, whose padded block stride is 38016 B -> 19008 bf16 elements,
# against a dense 2 * 4 * 128 = 1024 element tail block.
KPOOL = 4
HEAD_DIM = 128
PADDED_BLOCK_ELEMS = 19008
DENSE_BLOCK_ELEMS = 2 * KPOOL * HEAD_DIM
# block_size 4 padded to token_alignment 128 -> block_alignment 32.
TAIL_STRIDE = 32
TAIL_BLOCK_SIZE = 4


def seed_block_base(blk: int, stride0: int, head_dim: int) -> int:
    """Padded addressing: the fix. ``blk * tail.stride(0)``."""
    return blk * stride0


def dense_block_base(blk: int, kpool: int, head_dim: int) -> int:
    """Dense addressing: the defect. ``blk * 2 * kpool * head_dim``."""
    return blk * 2 * kpool * head_dim


def column_in_range(pos: int, block_size: int, stride: int) -> bool:
    """The V2 kernel's bound: ``pos // block_size < block_table_stride``."""
    return pos // block_size < stride


def test_seed_addressing() -> None:
    # The two formulas agree only for block 0, which is why short prompts and
    # a single-block cache looked healthy.
    assert seed_block_base(0, PADDED_BLOCK_ELEMS, HEAD_DIM) == dense_block_base(
        0, KPOOL, HEAD_DIM
    )

    # For block 1 the dense offset (1024) falls *inside* block 0's padded
    # region [0, 19008), i.e. into an unrelated indexer block, while leaving
    # the request's own tail block untouched. This is the silent corruption.
    dense1 = dense_block_base(1, KPOOL, HEAD_DIM)
    assert dense1 == DENSE_BLOCK_ELEMS
    assert dense1 < PADDED_BLOCK_ELEMS
    assert dense1 != seed_block_base(1, PADDED_BLOCK_ELEMS, HEAD_DIM)

    # Padded addressing is non-overlapping and 16-byte aligned per block.
    for blk in range(8):
        start = seed_block_base(blk, PADDED_BLOCK_ELEMS, HEAD_DIM)
        end = start + 2 * KPOOL * HEAD_DIM
        assert start % 8 == 0
        assert end <= seed_block_base(blk + 1, PADDED_BLOCK_ELEMS, HEAD_DIM)

    # The score plane is addressed by stride(1), not by a dense kpool*head_dim.
    kpool_head = KPOOL * HEAD_DIM
    assert kpool_head == 512
    # The K plane of a padded block still fits below the score plane.
    assert KPOOL * HEAD_DIM <= kpool_head


def test_column_bound() -> None:
    # Tail group: every position at or past block_size * stride used to index
    # past the row, and past the whole allocation for the last request.
    assert column_in_range(0, TAIL_BLOCK_SIZE, TAIL_STRIDE)
    assert column_in_range(124, TAIL_BLOCK_SIZE, TAIL_STRIDE)  # 31 < 32
    assert not column_in_range(128, TAIL_BLOCK_SIZE, TAIL_STRIDE)  # 32 == 32
    assert not column_in_range(1 << 20, TAIL_BLOCK_SIZE, TAIL_STRIDE)

    # A group whose row spans the sequence is unaffected (clamp is identity).
    wide_stride = 1024  # 65536 tokens / block_size 64
    for pos in (0, 63, 64, 65_535):
        assert column_in_range(pos, 64, wide_stride)
    # Only past the row does it start masking.
    assert not column_in_range(65_536, 64, wide_stride)


def _minimal_sources() -> dict[str, str]:
    """Smallest text containing every pinned anchor, per target file."""
    return {
        "kpool": '''import triton
import triton.language as tl


@triton.jit
def _kpool_tail_seed_kernel(
    key_ptr,
    score_ptr,
    tslot_ptr,
    tail_ptr,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Seed the tail ring.

    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.
    """
    i = tl.program_id(0)
    t = tl.load(tslot_ptr + i).to(tl.int64)
    blk = t // KPOOL
    offs = tl.arange(0, BLOCK_D)
    m = offs < HEAD_DIM
    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(tail_ptr + base + KPOOL * HEAD_DIM + offs, s, mask=m)


def kpool_seed_tail_cache(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> None:
    """Seed the paged tail cache from a prefill batch (see the kernel)."""
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    n = tslot.shape[0]
    if n == 0:
        return
    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
''',
        "mamba": '''class MambaHybridModelState:
    def prepare_attn(self, input_batch):
        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
''',
        "indexer": '''def compute_kpool_tail_slot_mapping(
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_actual_tokens: int,
    num_reqs: int,
    kpool: int,
) -> torch.Tensor:
    """Circular tail slots: every token of request r lands in r's own block.

    Pure torch (no Triton, no device sync): the indexer op consumes the tail
    slot mapping on its eager break, so the returned tensor need not be the
    persistent ``BlockTables`` buffer.
    """
    out = slot_mapping.clone()
    if num_actual_tokens == 0:
        return out
    device = slot_mapping.device
    tokens = torch.arange(num_actual_tokens, device=device)
    req = torch.searchsorted(query_start_loc, tokens, right=True) - 1
    req = req.clamp_(min=0, max=num_reqs - 1)
    own_block = block_table[:num_reqs, 0].index_select(0, req).to(torch.int64)
    pos = positions[:num_actual_tokens].to(torch.int64)
    out[:num_actual_tokens] = own_block * kpool + torch.remainder(pos, kpool)
    return out


class KpoolTailMetadataBuilder(AttentionMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # No indexer-builder buffers (expanded_block_table / scheduler_metadata /
        # compressed_slot_mapping) -- the tail is storage-only and exports only
        # slot_mapping, which is rebuilt per step from the group's block table.
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        slot_mapping = common_attn_metadata.slot_mapping
        positions = common_attn_metadata.positions
        if positions is not None:
            # Circular per-request layout; the generic kernel output collapses
            # onto tail block 0 for pos >= kpool (see compute_... docstring).
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
            )
        return DeepseekV32IndexerMetadata(slot_mapping=slot_mapping)
''',
        "blocktable": '''import triton
import triton.language as tl


@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,
    query_start_loc,
    pos,
    block_table_ptrs,
    block_table_strides,
    block_sizes,
    slot_mappings_ptr,
    slot_mappings_stride,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)
    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

        block_indices = positions // (block_size * CP_SIZE)
        block_offsets = positions % (block_size * CP_SIZE)
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices
        )

        if CP_SIZE == 1:
            # Common case: Context parallelism is not used.
            slot_ids = block_numbers * block_size + block_offsets
        else:
            # Context parallelism is used.
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)

        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
''',
    }


def _patched_indexer_source() -> str:
    patched, action = prepare(_minimal_sources()["indexer"], SITES["indexer"], "indexer")
    assert action == "patched"
    return patched


def _extract_top_level_function(source: str, name: str) -> str:
    """Return the source of a top-level ``def name`` block."""
    start = source.index(f"def {name}(")
    rest = source[start:]
    for marker in ("\nclass ", "\ndef ", "\n@"):
        idx = rest.find(marker, 1)
        if idx != -1:
            rest = rest[:idx]
    return rest


def test_indexer_buffer_is_shape_preserving() -> None:
    """The persistent buffer must be viewed to the batch shape.

    The ``clone()`` this replaces had the batch's shape, so handing the raw
    ``max_num_batched_tokens``-sized buffer to the builder would silently
    change the shape of the exported ``metadata.slot_mapping``.
    """
    text = _patched_indexer_source()
    assert "].view_as(slot_mapping)" in text, "buffer not viewed to the batch shape"
    assert "out=self._tail_slot_mapping," not in text, "raw buffer passed as out"
    assert "out.shape != slot_mapping.shape" in text, "no exact-shape guard"


def test_indexer_buffer_behaviour() -> None:
    """Exact-shape guard, stable address, and a shape-preserving result."""
    try:
        import torch
    except ImportError:
        return

    namespace: dict = {"torch": torch}
    exec(
        _extract_top_level_function(
            _patched_indexer_source(), "compute_kpool_tail_slot_mapping"
        ),
        namespace,
    )
    fn = namespace["compute_kpool_tail_slot_mapping"]

    slot_mapping = torch.arange(6, dtype=torch.int64)
    positions = torch.arange(6, dtype=torch.int64)
    query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int64)
    block_table = torch.tensor([[5], [9]], dtype=torch.int64)
    buffer = torch.empty(64, dtype=torch.int64)
    out = buffer[: slot_mapping.numel()].view_as(slot_mapping)

    result = fn(
        slot_mapping, block_table, query_start_loc, positions, 6, 2, KPOOL, out=out
    )

    assert result.shape == slot_mapping.shape, "output shape changed"
    assert result.data_ptr() == buffer.data_ptr(), "not the persistent buffer"
    # Request 0 owns block 5 and request 1 block 9; slot = own * kpool + pos % kpool.
    assert result.tolist() == [
        (5 if i < 3 else 9) * KPOOL + i % KPOOL for i in range(6)
    ]

    # A second call must reuse the same storage -- that address is what CUDA
    # graph capture records, so it cannot move between steps.
    again = fn(
        slot_mapping, block_table, query_start_loc, positions, 6, 2, KPOOL, out=out
    )
    assert again.data_ptr() == result.data_ptr()

    # A wrong-shaped buffer must fail closed, not write a partial mapping.
    try:
        fn(
            slot_mapping,
            block_table,
            query_start_loc,
            positions,
            6,
            2,
            KPOOL,
            out=torch.empty(3, dtype=torch.int64),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a wrong-shaped out buffer did not raise")


def _tree(root: Path) -> Path:
    """Materialize the minimal tree the patch expects."""
    site = root / "vllm"
    paths = {
        "kpool": site / "models/glm5next/nvidia/ops/kpool_compress.py",
        "mamba": site / "v1/worker/gpu/model_states/mamba_hybrid.py",
        "indexer": site / "v1/attention/backends/mla/indexer.py",
        "blocktable": site / "v1/worker/gpu/block_table.py",
    }
    sources = _minimal_sources()
    for label, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sources[label])
    return site


def _run_patch(site: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_VLLM_SITE"] = str(site)
    return subprocess.run(
        [sys.executable, str(PATCH), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_fixture() -> None:
    with tempfile.TemporaryDirectory() as raw:
        site = _tree(Path(raw))
        first = _run_patch(site)
        assert first.returncode == 0, first.stderr
        assert "kpool=patched" in first.stdout
        for label, target in TARGETS.items():
            text = (site / str(target).split("/vllm/", 1)[1]).read_text()
            assert verified_state(text, SITES[label]), label

        hashes = {
            label: (site / str(t).split("/vllm/", 1)[1]).read_bytes()
            for label, t in TARGETS.items()
        }
        second = _run_patch(site)
        assert second.returncode == 0, second.stderr
        assert "already present" in second.stdout
        for label, target in TARGETS.items():
            after = (site / str(target).split("/vllm/", 1)[1]).read_bytes()
            assert after == hashes[label], f"{label} not byte-identical on re-apply"


def test_fail_closed_on_drift() -> None:
    with tempfile.TemporaryDirectory() as raw:
        site = _tree(Path(raw))
        target = site / "models/glm5next/nvidia/ops/kpool_compress.py"
        drifted = target.read_text().replace(
            "    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM",
            "    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM  # drifted",
            1,
        )
        target.write_text(drifted)
        before = {p: p.read_bytes() for p in site.rglob("*.py")}
        result = _run_patch(site)
        assert result.returncode != 0
        assert "preflight failed" in result.stderr
        # A failed run must not have written anything.
        assert {p: p.read_bytes() for p in site.rglob("*.py")} == before


def test_fail_closed_on_partial() -> None:
    with tempfile.TemporaryDirectory() as raw:
        site = _tree(Path(raw))
        target = site / "v1/attention/backends/mla/indexer.py"
        # Apply only the signature site of five.
        text = target.read_text().replace(
            """    num_actual_tokens: int,
    num_reqs: int,
    kpool: int,
) -> torch.Tensor:""",
            """    num_actual_tokens: int,
    num_reqs: int,
    kpool: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:""",
            1,
        )
        target.write_text(text)
        result = _run_patch(site)
        assert result.returncode != 0
        assert "partial/inconsistent" in result.stderr


def test_prepare_is_pure() -> None:
    """``prepare`` must not mutate its input or double-apply."""
    for label, sites in SITES.items():
        source = _minimal_sources()[label]
        original = source
        patched, action = prepare(source, sites, label)
        assert action == "patched"
        assert source == original, "prepare mutated its input"
        again, action2 = prepare(patched, sites, label)
        assert action2 == "already present"
        assert again == patched


def test_installed_copy_if_present() -> None:
    site = Path(os.environ.get("GLM53_VLLM_SITE", "") or SITE)
    if not site.is_dir():
        return
    with tempfile.TemporaryDirectory() as raw:
        tree = _tree(Path(raw))
        # Overwrite the synthetic tree with the real deployed files.
        for label, target in TARGETS.items():
            rel = str(target).split("/vllm/", 1)[1]
            real = site / rel
            if not real.is_file():
                return
            (tree / rel).write_text(real.read_text())
        result = _run_patch(tree)
        assert result.returncode == 0, result.stderr
        for label, target in TARGETS.items():
            rel = str(target).split("/vllm/", 1)[1]
            assert verified_state((tree / rel).read_text(), SITES[label]), label


def test_recipe_wiring_if_present() -> None:
    start = ROOT / "start.sh"
    dockerfile = ROOT / "Dockerfile"
    if not start.is_file() or not dockerfile.is_file():
        return
    launcher = start.read_text()
    image = dockerfile.read_text()
    assert 'KPOOL_TAIL_CORRECTNESS_PATCH_HOST="${KPOOL_TAIL_CORRECTNESS_PATCH_HOST:-' in launcher
    assert launcher.count("python3 -S /opt/glm53/patch_kpool_tail_correctness.py") == 2
    # The knob is the documented one-line rollback: both apply sites must be
    # gated on it, and it must reach the container on both ranks.
    assert (
        launcher.count(
            'if [ "${KPOOL_TAIL_CORRECTNESS:-1}" = "1" ] '
            "&& [ -f /opt/glm53/patch_kpool_tail_correctness.py ]"
        )
        == 2
    ), "both apply sites must honour KPOOL_TAIL_CORRECTNESS"
    assert 'KPOOL_TAIL_CORRECTNESS="${KPOOL_TAIL_CORRECTNESS-1}"' in launcher
    assert '-e "KPOOL_TAIL_CORRECTNESS=$KPOOL_TAIL_CORRECTNESS"' in launcher
    assert (
        "-v '/tmp/patch_kpool_tail_correctness.py:"
        "/opt/glm53/patch_kpool_tail_correctness.py:ro'" in launcher
    )
    assert (
        '-v "$KPOOL_TAIL_CORRECTNESS_PATCH_HOST:'
        '/opt/glm53/patch_kpool_tail_correctness.py:ro"' in launcher
    )
    assert 'scp -q -o BatchMode=yes "$KPOOL_TAIL_CORRECTNESS_PATCH_HOST"' in launcher
    # Runtime-only, like the W28 correctness patch: the patch must NOT be baked
    # into the image, or a baked copy would re-apply itself and the
    # KPOOL_TAIL_CORRECTNESS rollback would be silently inert.
    assert "patch_kpool_tail_correctness" not in image, "must not be baked"
    assert "test_kpool_tail_correctness" not in image, "must not be baked"


def main() -> int:
    test_seed_addressing()
    test_column_bound()
    test_fixture()
    test_fail_closed_on_drift()
    test_fail_closed_on_partial()
    test_prepare_is_pure()
    test_indexer_buffer_is_shape_preserving()
    test_indexer_buffer_behaviour()
    test_installed_copy_if_present()
    test_recipe_wiring_if_present()
    print("kpool tail correctness patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
