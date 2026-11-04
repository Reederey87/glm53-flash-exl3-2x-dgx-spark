"""Exercise >2**31 FP32 partials using the pinned planner's supported split-K8.

The installed upstream regression uses split-K16, introduced by a separate
planner change. Keep that source intact and reuse its verified helpers here.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


def load_upstream_helpers(path: Path):
    expected = "e2576cfef66c128cb5aff959f6fa0b85573ec2f7dbec6e83ddd12224d160dff7"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("installed candidate A16 test source identity differs")
    spec = importlib.util.spec_from_file_location("qualified_upstream_a16", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_nvfp4_split_partials_exceed_int32_elements():
    upstream = load_upstream_helpers(Path(
        "/opt/dense-int64-tests/tests/experimental/b12x/gemm/test_blockscaled_a16.py"
    ))
    upstream.require_b12x()
    torch, blockscaled = upstream.torch, upstream.blockscaled
    m, n, k, slices = 8192, 34816, 1024, 8
    source = torch.ones((m, k), device="cuda", dtype=torch.bfloat16)
    packed = torch.full((n, k // 2), 0x22, device="cuda", dtype=torch.uint8)
    scales = torch.ones((n, k // 16), device="cuda", dtype=torch.float8_e4m3fn)
    weight = blockscaled.pack_weight(
        packed, upstream.swizzle_block_scale(scales), recipe="nvfp4",
        global_scale=torch.ones(1, device="cuda", dtype=torch.float32),
    )
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    config = upstream.BlockscaledConfig(
        mode="a16", tile_m=16, tile_n=64, tile_k=64, split_k=slices
    )
    workspace = upstream.make_workspace(
        source, weight, activation_mode="a16", out=out, config=config
    )
    assert workspace.dtype == torch.uint8
    assert workspace.numel() == slices * m * n * 4 == 9_126_805_504
    assert workspace.numel() // 4 > 2**31
    with upstream.prepared_execution(
        source, weight, activation_mode="a16", out=out,
        workspace=workspace, config=config,
    ) as (query, plan):
        assert upstream._preparation.effective_a16_config(query, config)[2] == slices

        def run():
            blockscaled.mm(source, weight, plan=plan, out=out, workspace=workspace)

        run()
        torch.testing.assert_close(out, torch.full_like(out, k), rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        source.fill_(0.5)
        graph.replay()
        torch.testing.assert_close(out, torch.full_like(out, k / 2), rtol=0, atol=0)
