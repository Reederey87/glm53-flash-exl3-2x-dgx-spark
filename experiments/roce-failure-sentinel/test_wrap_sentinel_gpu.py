"""Native bounded wrap regression, run only in an idle SM121 container.

No network proxy is created. Pinned peer mailboxes exercise the installed
compiled one-shot/gather kernels, including their timeout and poison paths.
Actual two-node fabric, two-shot and varlen tests are separate required gates.
"""
from types import SimpleNamespace
import time

import pytest
import torch


@pytest.mark.parametrize("operation", ["reduce", "gather"])
@pytest.mark.parametrize("blocks", [1, 2, 4])
@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("timeout", [False, True], ids=["success", "timeout"])
def test_sequence_zero_kernel_failure_and_success(operation, blocks, graph_mode, timeout):
    from flashinfer.experimental.b12x.comm.roce import _allgather_cute, _oneshot_cute
    from flashinfer.experimental.b12x.comm.roce.roce_oneshot import RoceOneshotAllReduce

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1):
        pytest.skip("requires idle native GB10 SM121 qualification")
    # Exactly two simulated ranks, two slots, one rail. Separate page-aligned
    # spans make invalid pointer arithmetic visible without large allocations.
    slot_bytes, flag_stride = 4096, 128
    recv_offset, flag_offset, send_offset, ctrl_offset = 0, 16384, 20480, 28672
    host = torch.zeros(32768, dtype=torch.uint8, pin_memory=True)
    assert RoceOneshotAllReduce._device_pointer(host.data_ptr()) == host.data_ptr()
    ctrl = host[ctrl_offset:ctrl_offset + 60].view(torch.int32)
    # Peer 1, slot 0 is the wrapped sequence's source; never use zero flags
    # for a timeout test, since zero is exactly the expected sequence.
    peer_slot = recv_offset + 2 * slot_bytes
    host[peer_slot:peer_slot + 16].view(torch.bfloat16).fill_(2)
    peer_flag = flag_offset + 2 * flag_stride
    host[peer_flag:peer_flag + 4].view(torch.int32).fill_(1 if timeout else 0)
    ctrl[0] = -1
    counters = torch.zeros(4, dtype=torch.int32, device="cuda")
    counters[0] = -1  # epoch 0xffffffff, next sequence is zero
    inp = torch.ones(8, dtype=torch.bfloat16, device="cuda")
    output_size = 8 if operation == "reduce" else 16
    guarded = torch.full((output_size + 16,), -123, dtype=torch.bfloat16, device="cuda")
    out = guarded[8:-8]
    if operation == "reduce":
        launch = _oneshot_cute.get_launcher("bfloat16", 2, 0, 512, 2, flag_stride, 1, 0)
    else:
        launch = _allgather_cute.get_launcher(2, 0, 512, 2, flag_stride, 1, 0)
    addresses = [host.data_ptr() + offset for offset in
                 (recv_offset, flag_offset, send_offset, ctrl_offset)]
    prefix = [inp.data_ptr(), out.data_ptr(), 1, 16]
    if operation == "gather":
        prefix.append(1)  # row_packs
    args = [*prefix, *addresses, slot_bytes, counters.data_ptr(),
            counters.data_ptr() + 4, counters.data_ptr() + 8,
            counters.data_ptr() + 12, 1024, blocks]
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
    assert time.monotonic() - started < 10, "short-spin fault did not complete promptly"
    assert torch.all(guarded[:8] == -123) and torch.all(guarded[-8:] == -123)
    host_state = SimpleNamespace(_proxy=None, _ctrl_np=ctrl.numpy(), rank=0)
    if timeout:
        assert int(ctrl[2]) == 0 and int(ctrl[14]) == 1
        assert int(counters[0]) == -1 and int(counters[3]) == 1
        assert torch.all(out == -123), "failed collective published output"
        with pytest.raises(RuntimeError, match="sequence 0;.*epoch stopped at 4294967295"):
            RoceOneshotAllReduce.check_health(host_state)
        assert RoceOneshotAllReduce.poisoned.fget(host_state)
        before = counters.clone()
        # A later launch must remain a no-op even if the peer becomes ready.
        host[peer_flag:peer_flag + 4].view(torch.int32).zero_()
        run()
        torch.cuda.synchronize()
        assert torch.equal(counters, before) and torch.all(out == -123)
    else:
        assert int(ctrl[14]) == 0 and int(counters[0]) == 0 and int(counters[3]) == 0
        expected = (torch.full_like(out, 3) if operation == "reduce" else
                    torch.cat((torch.ones_like(inp), torch.full_like(inp, 2))))
        assert torch.equal(out, expected)
        RoceOneshotAllReduce.check_health(host_state)
        assert not RoceOneshotAllReduce.poisoned.fget(host_state)
