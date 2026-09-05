#!/usr/bin/env python3
"""Static launcher regressions for P0 runtime hardening."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "start.sh").read_text()
README = (ROOT / "README.md").read_text()


def test_patch_installers_skip_site_initialization() -> None:
    invocations = re.findall(
        r"^\s*(python3(?: -S)? /opt/glm53/patch_[^\s]+\.py)\s*$",
        SOURCE,
        re.MULTILINE,
    )
    assert invocations
    assert all(command.startswith("python3 -S ") for command in invocations)
    assert sum("patch_mamba_null_gap_retirement.py" in command for command in invocations) == 2


def test_api_key_is_head_only() -> None:
    worker_start = SOURCE.index('worker_ssh "docker run -d')
    head_start = SOURCE.index('docker run -d --name "$CONTAINER_HEAD"', worker_start)
    worker_command = SOURCE[worker_start:head_start]
    head_command = SOURCE[head_start:SOURCE.index('log "containers up', head_start)]
    assert "VLLM_API_KEY" not in worker_command
    assert head_command.count('-e VLLM_API_KEY="$VLLM_API_KEY"') == 1


def test_shared_head_guidance_uses_effective_launcher_names() -> None:
    assert "`GPU_MEM_UTIL=0.78`" in README
    assert "`CG_ESTIMATE=0`" in README
    assert "GPU_MEMORY_UTILIZATION=0.78" not in README
    assert "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0" not in README
    assert '--gpu-memory-utilization "${GPU_MEM_UTIL}"' in SOURCE
    assert '-e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=$CG_ESTIMATE"' in SOURCE
