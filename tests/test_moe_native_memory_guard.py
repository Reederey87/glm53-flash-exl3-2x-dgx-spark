"""CPU fault checks for the future GPU test's preallocation refusal only."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).parents[1] / 'experiments/moe-native-a4/native/test_glm_native_geometry.py'
spec = importlib.util.spec_from_file_location('native_geometry_guard', SOURCE)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
GIB = 1024**3


def fake_runtime(monkeypatch, *, free, allocated):
    class MemoryPath:
        def __init__(self, path):
            assert path == '/proc/meminfo'

        def read_text(self):
            return f'MemFree: {free // 1024} kB\nMemAvailable: 123456 kB\n'

    monkeypatch.setattr(native, 'Path', MemoryPath)
    allocations = []

    def empty(shape, **kwargs):
        allocations.append((shape, kwargs))
        return object()

    return SimpleNamespace(cuda=SimpleNamespace(memory_allocated=lambda: allocated), empty=empty), allocations


@pytest.mark.parametrize('free,allocated,shape,reason', [
    (24 * GIB, 0, (1024, 1024), 'free memory'),
    (120 * GIB, 32 * GIB, (1,), 'tensor footprint'),
    (120 * GIB, 0, (32 * GIB,), 'tensor footprint'),
])
def test_scratch_refuses_before_any_allocation(monkeypatch, free, allocated, shape, reason):
    torch, allocations = fake_runtime(monkeypatch, free=free, allocated=allocated)
    specs = [SimpleNamespace(shape=(1,), dtype=SimpleNamespace(itemsize=1)),
             SimpleNamespace(shape=shape, dtype=SimpleNamespace(itemsize=1))]
    with pytest.raises(AssertionError, match=reason):
        native._scratch(torch, specs, 'cuda')
    assert allocations == []


def test_uses_total_scratch_bytes_before_first_allocation(monkeypatch):
    torch, allocations = fake_runtime(monkeypatch, free=120 * GIB, allocated=0)
    specs = [SimpleNamespace(shape=(10 * GIB,), dtype=SimpleNamespace(itemsize=2))] * 2
    with pytest.raises(AssertionError, match='tensor footprint'):
        native._scratch(torch, specs, 'cuda')
    assert allocations == []


def test_bounded_allocation_can_proceed(monkeypatch):
    torch, allocations = fake_runtime(monkeypatch, free=80 * GIB, allocated=GIB)
    specs = [SimpleNamespace(shape=(128, 4096), dtype=SimpleNamespace(itemsize=2))]
    result = native._scratch(torch, specs, 'cuda')
    assert len(result) == len(allocations) == 1
