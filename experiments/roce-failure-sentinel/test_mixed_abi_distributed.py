"""Run with torchrun on two idle nodes, one control image and one candidate."""
from datetime import timedelta
import os
import time

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    os.environ.get("ROCE_MIXED_ABI_QUALIFICATION") != "1",
    reason="requires explicit two-image, two-node ABI qualification window",
)


def test_actual_mixed_proxy_abis_refuse_setup_on_both_ranks():
    from flashinfer.experimental.b12x.comm import roce
    from flashinfer.experimental.b12x.comm.roce import _proxy

    # CPU bootstrap can report a setup refusal without running an NCCL or
    # RoCE GPU collective. Real registered regions and local proxy setup are used.
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        assert dist.get_world_size() == 2 and roce.API_VERSION == 1
        abis = [None, None]
        dist.all_gather_object(abis, _proxy.load().roce_abi_version())
        assert sorted(abis) == [7, 8], "must use actual old/new image pair"
        torch.cuda.set_device(0)
        started = time.monotonic()
        runtime = None
        error = ""
        try:
            runtime = roce.AllReduce.from_exchange_group(
                exchange_group=dist.group.WORLD, device=torch.device("cuda", 0),
                max_size=4096, max_gather_bytes=4096)
        except RuntimeError as exc:
            error = str(exc)
        finally:
            if runtime is not None:
                runtime.close()
        results = [None, None]
        dist.all_gather_object(results, {"error": error, "elapsed_s": time.monotonic() - started})
        for result in results:
            assert "configuration differs" in result["error"] and "proxy_abi" in result["error"]
            assert result["elapsed_s"] < 60
    finally:
        dist.destroy_process_group()
