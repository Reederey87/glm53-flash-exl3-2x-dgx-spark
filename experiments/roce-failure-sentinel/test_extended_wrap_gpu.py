"""Candidate-only native two-shot/varlen wrap checks; no RDMA proxy is started."""
from types import SimpleNamespace
import time

import pytest
import torch


@pytest.mark.parametrize("scenario", ["twoshot_success", "twoshot_rs_timeout", "twoshot_ag_timeout",
                                      "varlen_success", "varlen_timeout", "varlen_invalid"])
@pytest.mark.parametrize("blocks", [1, 2, 4])
@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
def test_extended_sequence_zero(scenario, blocks, graph_mode):
    from flashinfer.experimental.b12x.comm.roce import _twoshot_cute, _varlen_cute
    from flashinfer.experimental.b12x.comm.roce._proxy import ABI_VERSION
    from flashinfer.experimental.b12x.comm.roce.roce_oneshot import RoceOneshotAllReduce

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1):
        pytest.skip("requires idle native GB10 qualification")
    assert ABI_VERSION == 8
    twoshot = scenario.startswith("twoshot")
    success = scenario.endswith("success")
    host = torch.zeros(32768, dtype=torch.uint8, pin_memory=True)
    assert RoceOneshotAllReduce._device_pointer(host.data_ptr()) == host.data_ptr()
    recv, send, flags, control = 0, 16384, 24576, 28672
    ctrl = host[control:].view(torch.int32)
    ctrl[0] = -1
    for word in (64, 128, 256):
        ctrl[word] = -1  # old readiness, not the wrapped sequence zero
    counters = torch.zeros(16, dtype=torch.int32, device="cuda")
    counters[0] = -1  # epoch; counters[1] is poison
    inp = torch.ones(16 if twoshot else 8, dtype=torch.bfloat16, device="cuda")
    guarded = torch.full((32,), -123, dtype=torch.bfloat16, device="cuda")
    out = guarded[8:-8]
    base = host.data_ptr()
    counts = None
    if twoshot:
        host[8192:8208].view(torch.bfloat16).fill_(2)  # peer's reduce contribution
        host[12288:12304].view(torch.bfloat16).fill_(3)  # peer's reduced shard
        if scenario == "twoshot_rs_timeout":
            host[flags + 128:flags + 132].view(torch.int32).fill_(1)
        if scenario == "twoshot_ag_timeout":
            host[flags + 384:flags + 388].view(torch.int32).fill_(1)
        launch = _twoshot_cute.get_launcher("bfloat16", 2, 0, 512, 128, 1, 1, 0)
        args = [inp.data_ptr(), out.data_ptr(), 2, 32, 1, 1,
                base + send, base + recv, base + flags, 4096, base + control,
                counters.data_ptr(), counters.data_ptr() + 8, counters.data_ptr() + 4,
                1024, blocks]
    else:
        host[4096:4112].view(torch.bfloat16).fill_(2)
        counts = torch.tensor([16, 17 if scenario == "varlen_invalid" else 16],
                              dtype=torch.int64, device="cuda")
        if scenario == "varlen_timeout":
            host[flags + 128:flags + 132].view(torch.int32).fill_(1)
        launch = _varlen_cute.get_launcher(2, 0, 512, 128, 1, 1, 0)
        args = [inp.data_ptr(), out.data_ptr(), counts.data_ptr(), 1, 16, 1, 1,
                base + recv, base + flags, 4096, base + control, counters.data_ptr(),
                counters.data_ptr() + 8, counters.data_ptr() + 4, 1024, blocks]
    torch.cuda.synchronize()
    if graph_mode:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch(*args)
        run = graph.replay
    else:
        run = lambda: launch(*args)
    started = time.monotonic()
    run()
    torch.cuda.synchronize()
    assert time.monotonic() - started < 10
    assert torch.all(guarded[:8] == -123) and torch.all(guarded[-8:] == -123)
    state = SimpleNamespace(_proxy=None, _ctrl_np=ctrl.numpy(), rank=0)
    expected = torch.full_like(out, -123)
    if success:
        expected[:8] = 3 if twoshot else 1
        expected[8:] = 3 if twoshot else 2
        assert int(counters[0]) == 0 and int(counters[1]) == 0 and int(ctrl[14]) == 0
        RoceOneshotAllReduce.check_health(state)
    else:
        # A local partial result can precede the later peer failure. The whole
        # result is unusable, and the host must report poison before returning it.
        partial = {"twoshot_ag_timeout": 3, "varlen_timeout": 1}.get(scenario)
        if partial is not None:
            # Another block can publish poison before block zero writes its
            # local pack. Both orderings are valid; the peer half must stay
            # guarded, and a poisoned replay must preserve the observed bytes.
            assert torch.all((out[:8] == -123) | (out[:8] == partial))
            if blocks == 1:
                assert torch.all(out[:8] == partial)
            expected[:8] = out[:8]
        assert int(counters[0]) == -1 and int(counters[1]) == 1
        assert int(ctrl[2]) == 0 and int(ctrl[14]) == 1
        message = "unusable byte count" if scenario == "varlen_invalid" else "sequence 0;"
        with pytest.raises(RuntimeError, match=message):
            RoceOneshotAllReduce.check_health(state)
        assert RoceOneshotAllReduce.poisoned.fget(state)
        if scenario == "twoshot_rs_timeout":
            assert int(ctrl[128]) == -1, "failed reduction published a ready chunk"
        if scenario == "varlen_invalid":
            assert int(ctrl[256]) == -1 and int(ctrl[7]) == 1
    assert torch.equal(out, expected)
    if not success:
        host[flags:control].zero_()
        if counts is not None:
            counts.fill_(16)
        torch.cuda.synchronize()
        run()
        torch.cuda.synchronize()
        # Tail bookkeeping can advance on a captured poisoned launch; payload,
        # epoch, failure and poison must remain unchanged.
        assert torch.equal(out, expected)
        assert int(counters[0]) == -1 and int(counters[1]) == 1 and int(ctrl[14]) == 1
