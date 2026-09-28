"""Installer and ordering for the early shared-expert launch.

The claim under test is an ordering claim, not a speed claim: with the stock
source the shared experts are enqueued on the auxiliary stream only after the
routed experts have been enqueued, so the two cannot overlap; with the patch
they are enqueued at the sync point, before the gate and the routed dispatch,
and ``forward`` only joins.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "overlay" / "patch_shared_experts_overlap.py"
START = ROOT / "start.sh"
ENV_EXAMPLE = ROOT / "env.example"
KNOB = "GLM53_SHARED_EXPERTS_EARLY"
DEPLOYED_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "shared-experts-487ecf187"
    / "shared_experts.py"
)


# --------------------------------------------------------------------------
# installer behaviour
# --------------------------------------------------------------------------
def _run(tmp: Path, flag: str, text: str | None) -> subprocess.CompletedProcess[str]:
    target = tmp / "shared_experts.py"
    if text is not None:
        target.write_text(text)
    env = os.environ.copy()
    env["GLM53_SHARED_EXPERTS_EARLY"] = flag
    env["GLM53_SHARED_EXPERTS_PY"] = str(target)
    return subprocess.run(
        [sys.executable, str(INSTALLER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _installer():
    return _load(INSTALLER, "installer")


def _synthesized_stock() -> str:
    """Build a stock shared_experts.py out of the installer's own anchors.

    ``tests/fixtures/`` is gitignored (it holds local live-image dumps), so CI
    has no pinned copy of the deployed file. The three ``*_OLD`` anchors are the
    exact deployed bytes the installer rewrites; wrapping them in the
    surrounding class yields a source that is stock by construction and is
    available everywhere, so the ordering proof runs in CI too.
    """
    installer = _installer()
    return (
        "from collections.abc import Callable\n"
        "from enum import IntEnum\n"
        "\n"
        "import torch\n"
        "\n"
        "import vllm.envs as envs\n"
        "from vllm.logger import init_logger\n"
        "from vllm.utils.torch_utils import (\n"
        "    aux_stream,\n"
        "    current_stream,\n"
        ")\n"
        "\n"
        "\n"
        "logger = init_logger(__name__)\n"
        "\n"
        "_EPLB_OVERLAP_SAFE_BACKENDS = (\n"
        '    "allgather_reducescatter",\n'
        '    "flashinfer_nvlink_one_sided",\n'
        ")\n"
        "\n"
        "\n"
        "class SharedExpertsOrder(IntEnum):\n"
        "    NONE = 0\n"
        "    NO_OVERLAP = 1\n"
        "    MK_INTERNAL_OVERLAPPED = 2\n"
        "    MULTI_STREAM_OVERLAPPED = 3\n"
        "\n"
        "\n"
        "class SharedExperts(torch.nn.Module):\n"
        "    def __init__(\n"
        "        self,\n"
        "        layer,\n"
        "        moe_config,\n"
        "        enable_dbo: bool,\n"
        "        mk_can_overlap_shared_experts: Callable[[], bool],\n"
        "    ):\n"
        "        super().__init__()\n"
        "        self.enable_dbo = enable_dbo\n"
        "        self._output = [None, None]\n"
        "        self._layer = layer\n"
        "        self._moe_config = moe_config\n"
        "        self._mk_can_overlap_shared_experts = mk_can_overlap_shared_experts\n"
        "        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:\n"
        '            logger.debug_once("Disabling MoE shared_experts cuda stream")\n'
        "            self._stream = None\n"
        "        else:\n"
        + installer.INIT_OLD
        + "    @property\n"
        "    def _disable_shared_experts_overlap(self) -> bool:\n"
        "        parallel_config = self._moe_config.moe_parallel_config\n"
        '        if getattr(self._layer, "shard_sequence_parallel", False):\n'
        "            return True\n"
        "        return (\n"
        "            parallel_config.enable_eplb\n"
        "            and parallel_config.all2all_backend not in _EPLB_OVERLAP_SAFE_BACKENDS\n"
        "        ) or parallel_config.use_fi_nvl_two_sided_kernels\n"
        "\n"
        "    def _determine_shared_experts_order(self, hidden_states):\n"
        "        if self._disable_shared_experts_overlap:\n"
        "            return SharedExpertsOrder.NO_OVERLAP\n"
        "        if self._mk_can_overlap_shared_experts():\n"
        "            return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED\n"
        "        if self._stream is not None:\n"
        "            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED\n"
        "        return SharedExpertsOrder.NO_OVERLAP\n"
        "\n"
        "    def maybe_sync_shared_experts_stream(\n"
        "        self,\n"
        "        shared_experts_input,\n"
        "    ):\n"
        "        experts_order = self._determine_shared_experts_order(shared_experts_input)\n"
        "        if experts_order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:\n"
        "            assert self._stream is not None\n"
        + installer.SYNC_OLD
        + "\n"
        "    def _run_in_aux_stream(self, shared_experts_input):\n"
        "        with torch.cuda.stream(self._stream):\n"
        "            output = self._layer(shared_experts_input)\n"
        "        current_stream().wait_stream(self._stream)\n"
        "        return output\n"
        "\n"
        "    @property\n"
        "    def _output_idx(self) -> int:\n"
        "        return 0\n"
        "\n"
        "    @property\n"
        "    def output(self):\n"
        "        assert self._output[self._output_idx] is not None\n"
        "        output = self._output[self._output_idx]\n"
        "        self._output[self._output_idx] = None\n"
        "        return output\n"
        "\n"
        "    def forward(\n"
        "        self,\n"
        "        shared_experts_input,\n"
        "        order: SharedExpertsOrder,\n"
        "    ):\n"
        "        experts_order = self._determine_shared_experts_order(shared_experts_input)\n"
        + installer.FORWARD_OLD
        + "        assert self._output[self._output_idx] is not None\n"
    )


def _fixture() -> str:
    """The deployed file when the local audit fixture is present, else a stub."""
    if DEPLOYED_FIXTURE.is_file():
        return DEPLOYED_FIXTURE.read_text()
    return _synthesized_stock()


def test_fixture_is_the_deployed_shape() -> None:
    if not DEPLOYED_FIXTURE.is_file():
        pytest.skip("local audit fixture tests/fixtures/ is not shipped")
    text = DEPLOYED_FIXTURE.read_text()
    assert text.count("self._stream.wait_stream(current_stream())") == 1
    assert "self._early_pending" not in text
    assert "MULTI_STREAM_OVERLAPPED" in text


def test_synthesized_stock_carries_every_anchor() -> None:
    """The CI stand-in must be stock by construction, not by luck."""
    installer = _installer()
    text = _synthesized_stock()
    for name, old, _ in installer.EDITS:
        assert text.count(old) == 1, f"anchor '{name}' must appear exactly once"
    assert installer.MARK not in text
    assert text.count("self._stream.wait_stream(current_stream())") == 1


def test_flag_off_leaves_bytes_identical(tmp_path: Path) -> None:
    original = _fixture()
    result = _run(tmp_path, "0", original)
    assert result.returncode == 0
    assert "unchanged" in result.stdout
    assert (tmp_path / "shared_experts.py").read_text() == original


def test_flag_on_rewrites_all_three_anchors_once(tmp_path: Path) -> None:
    result = _run(tmp_path, "1", _fixture())
    assert result.returncode == 0, result.stderr
    text = (tmp_path / "shared_experts.py").read_text()
    assert text.count("[glm53-shared-experts-early]") == 3
    # the raw fork/join is gone from the sync point
    assert "self._stream.wait_stream(current_stream())" not in text
    # the aux stream is fed an explicit input dependency
    assert "self._early_input_ready[_early_idx].record(current_stream())" in text
    assert "self._early_output_ready[_early_idx].record(self._stream)" in text
    # forward joins instead of re-running the layer
    assert "self._early_pending[self._output_idx] = False" in text
    assert "self._early_output_ready[self._output_idx].wait(current_stream())" in text
    compile(text, "shared_experts.py", "exec")


def test_reapply_is_idempotent(tmp_path: Path) -> None:
    first = _run(tmp_path, "1", _fixture())
    assert first.returncode == 0, first.stderr
    applied = (tmp_path / "shared_experts.py").read_text()
    second = _run(tmp_path, "1", None)
    assert second.returncode == 0, second.stderr
    assert "already present" in second.stdout
    assert (tmp_path / "shared_experts.py").read_text() == applied


def test_garbage_flag_is_rejected_without_writing(tmp_path: Path) -> None:
    original = _fixture()
    result = _run(tmp_path, "yes", original)
    assert result.returncode == 1
    assert "exactly 0 or 1" in result.stderr
    assert (tmp_path / "shared_experts.py").read_text() == original


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.replace("self._stream = aux_stream()", "self._stream = mk_stream()"),
        lambda t: t.replace(
            "self._stream.wait_stream(current_stream())",
            "self._stream.wait_stream(other_stream())",
        ),
        lambda t: t.replace(
            "        if order != experts_order:\n            return None\n",
            "        if order == experts_order:\n            return None\n",
        ),
    ],
)
def test_drift_writes_nothing(tmp_path: Path, mutate) -> None:
    original = mutate(_fixture())
    assert original != _fixture()
    result = _run(tmp_path, "1", original)
    assert result.returncode == 1
    assert "anchor drift" in result.stderr
    assert (tmp_path / "shared_experts.py").read_text() == original


def test_partial_install_is_refused(tmp_path: Path) -> None:
    text = _fixture().replace(
        "            self._stream.wait_stream(current_stream())\n",
        "            self._stream.wait_stream(current_stream())  # [glm53-shared-experts-early]\n",
    )
    result = _run(tmp_path, "1", text)
    assert result.returncode == 1
    assert "partial install" in result.stderr


# --------------------------------------------------------------------------
# ordering behaviour, driven through a stubbed torch + vllm
# --------------------------------------------------------------------------
LOG: list[str] = []


class _Stream:
    def __init__(self, name: str) -> None:
        self.name = name

    def wait_stream(self, other: "_Stream") -> None:
        LOG.append(f"{self.name}.wait_stream({other.name})")

    def __enter__(self) -> "_Stream":
        LOG.append(f"enter({self.name})")
        return self

    def __exit__(self, *exc: object) -> bool:
        LOG.append(f"exit({self.name})")
        return False


class _Event:
    def record(self, stream: _Stream) -> None:
        LOG.append(f"event.record({stream.name})")

    def wait(self, stream: _Stream) -> None:
        LOG.append(f"event.wait({stream.name})")


class _Tensor:
    def __init__(self, rows: int) -> None:
        self.shape = (rows,)

    def record_stream(self, stream: _Stream) -> None:
        LOG.append(f"tensor.record_stream({stream.name})")


MAIN = _Stream("main")
AUX = _Stream("aux")


class _Module:
    def __init__(self) -> None:
        pass


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch, *, disable_stream: bool = False
) -> None:
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    nn.Module = _Module  # type: ignore[attr-defined]
    cuda = types.ModuleType("torch.cuda")
    cuda.Stream = _Stream  # type: ignore[attr-defined]
    cuda.Event = _Event  # type: ignore[attr-defined]
    cuda.stream = lambda s: s  # type: ignore[attr-defined]
    torch.nn = nn  # type: ignore[attr-defined]
    torch.cuda = cuda  # type: ignore[attr-defined]
    torch.Tensor = _Tensor  # type: ignore[attr-defined]

    envs = types.ModuleType("vllm.envs")
    envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM = disable_stream  # type: ignore[attr-defined]
    envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD = 256  # type: ignore[attr-defined]

    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: types.SimpleNamespace(  # type: ignore[attr-defined]
        debug_once=lambda *a, **k: None
    )

    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = types.SimpleNamespace(is_cuda=lambda: True)  # type: ignore[attr-defined]

    torch_utils = types.ModuleType("vllm.utils.torch_utils")
    torch_utils.aux_stream = lambda: AUX  # type: ignore[attr-defined]
    torch_utils.current_stream = lambda: MAIN  # type: ignore[attr-defined]

    config_mod = types.ModuleType("vllm.model_executor.layers.fused_moe.config")
    config_mod.FusedMoEConfig = object  # type: ignore[attr-defined]

    ubatching = types.ModuleType("vllm.v1.worker.ubatching")
    ubatching.dbo_current_ubatch_id = lambda: 0  # type: ignore[attr-defined]

    for name, mod in {
        "torch": torch,
        "torch.nn": nn,
        "torch.cuda": cuda,
        "vllm": types.ModuleType("vllm"),
        "vllm.envs": envs,
        "vllm.logger": logger_mod,
        "vllm.platforms": platforms,
        "vllm.utils": types.ModuleType("vllm.utils"),
        "vllm.utils.torch_utils": torch_utils,
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.fused_moe": types.ModuleType(
            "vllm.model_executor.layers.fused_moe"
        ),
        "vllm.model_executor.layers.fused_moe.config": config_mod,
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.ubatching": ubatching,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


def _load(path: Path, tag: str):
    spec = importlib.util.spec_from_file_location(f"se_{tag}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Layer:
    """Stands in for the shared-expert MLP; records each execution."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, value: _Tensor) -> _Tensor:
        self.calls += 1
        LOG.append("layer.run")
        return value


def _moe_config() -> types.SimpleNamespace:
    """TP-only, no EPLB and no flashinfer two-sided kernels: overlap allowed."""
    parallel = types.SimpleNamespace(
        enable_eplb=False,
        all2all_backend="allgather_reducescatter",
        use_fi_nvl_two_sided_kernels=False,
    )
    return types.SimpleNamespace(moe_parallel_config=parallel)


def _drive(module) -> tuple[list[str], int, object]:
    layer = _Layer()
    experts = module.SharedExperts(
        layer=layer,
        moe_config=_moe_config(),
        enable_dbo=False,
        mk_can_overlap_shared_experts=lambda: False,
    )
    activation = _Tensor(8)
    order = module.SharedExpertsOrder.MULTI_STREAM_OVERLAPPED

    LOG.clear()
    experts.maybe_sync_shared_experts_stream(activation)
    sync_log = list(LOG)

    LOG.clear()
    # forward() stores into the output slot; the caller reads `.output`.
    experts.forward(activation, order)
    forward_log = list(LOG)
    out = experts.output

    # The routed experts run between the sync point and forward(); nothing in
    # this harness enqueues them, so their absence is what "before the dispatch"
    # means here: the launch must already be in sync_log.
    return sync_log + ["--forward--"] + forward_log, layer.calls, out


def test_stock_enqueues_the_layer_in_forward(tmp_path: Path, monkeypatch) -> None:
    _install_stubs(monkeypatch)
    stock = tmp_path / "stock.py"
    stock.write_text(_fixture())
    log, calls, out = _drive(_load(stock, "stock"))
    assert calls == 1
    assert out is not None
    # stock: the sync point only marks a start; the layer runs in forward
    assert "layer.run" not in log[: log.index("--forward--")]
    assert "layer.run" in log[log.index("--forward--") :]


def test_patched_enqueues_the_layer_at_the_sync_point(tmp_path: Path, monkeypatch) -> None:
    _install_stubs(monkeypatch)
    result = _run(tmp_path, "1", _fixture())
    assert result.returncode == 0, result.stderr
    log, calls, out = _drive(_load(tmp_path / "shared_experts.py", "patched"))
    before = log[: log.index("--forward--")]
    after = log[log.index("--forward--") :]
    assert out is not None
    # the layer is enqueued before the dispatch ...
    assert "layer.run" in before
    # ... exactly once ...
    assert calls == 1
    # ... and forward only joins.
    assert "layer.run" not in after
    assert any("event.wait(main)" in line for line in after)


def test_knob_is_wired_on_both_nodes() -> None:
    text = START.read_text()
    assert f'-e "{KNOB}=${KNOB}"' in text, "head container env must carry the knob"
    assert f" {KNOB} \\\n" in text, "the worker serve_env loop must carry the knob"
    # The strict-bool loop is a moving list of knobs, so assert membership in
    # whatever `for _v in ...; do` loop the launcher carries, not a fixed line.
    loops = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("for _v in ") and line.strip().endswith("; do")
    ]
    assert loops, "the strict-bool validate loop must exist"
    assert any(KNOB in loop for loop in loops), (
        f"{KNOB} must be in the strict-bool validate loop, found: {loops}"
    )
    # the installer reaches both containers, by bind mount on the head and by
    # scp + bind mount on the worker
    assert text.count(
        f'$SHARED_EXPERTS_EARLY_PATCH_HOST:/opt/glm53/patch_shared_experts_overlap.py:ro'
    ) == 1
    assert text.count(
        "'/tmp/patch_shared_experts_overlap.py:"
        "/opt/glm53/patch_shared_experts_overlap.py:ro'"
    ) == 1
    assert (
        'scp -q -o BatchMode=yes "$SHARED_EXPERTS_EARLY_PATCH_HOST" '
        '"${WORKER_SSH}:/tmp/patch_shared_experts_overlap.py"' in text
    )
    # it runs inside both inner scripts, gated on the flag
    assert text.count("python3 -S /opt/glm53/patch_shared_experts_overlap.py") == 2


def test_knob_is_documented_in_env_example() -> None:
    text = ENV_EXAMPLE.read_text()
    assert f"{KNOB}=0" in text, "env.example must ship the knob defaulted off"


def test_patched_survives_a_disabled_aux_stream(tmp_path: Path, monkeypatch) -> None:
    """Stock runs the layer inline with the aux stream off; so must the patch.

    ``forward`` reads the in-flight flag unconditionally, so ``__init__`` has to
    allocate it even when it takes the disabled branch (no aux stream, or a
    non-cuda-alike platform where ``aux_stream()`` returns None).
    """
    _install_stubs(monkeypatch, disable_stream=True)
    result = _run(tmp_path, "1", _fixture())
    assert result.returncode == 0, result.stderr
    module = _load(tmp_path / "shared_experts.py", "nostream")
    layer = _Layer()
    experts = module.SharedExperts(
        layer=layer,
        moe_config=_moe_config(),
        enable_dbo=False,
        mk_can_overlap_shared_experts=lambda: False,
    )
    assert experts._stream is None
    activation = _Tensor(8)
    experts.maybe_sync_shared_experts_stream(activation)
    assert experts._early_pending == [False, False]
    LOG.clear()
    experts.forward(activation, module.SharedExpertsOrder.NO_OVERLAP)
    # no aux stream, so the layer runs inline, exactly as stock does
    assert layer.calls == 1
    assert LOG == ["layer.run"]
    assert experts.output is not None


def test_patched_never_runs_the_layer_twice(tmp_path: Path, monkeypatch) -> None:
    """A join that re-ran the layer would double the shared-expert work."""
    _install_stubs(monkeypatch)
    result = _run(tmp_path, "1", _fixture())
    assert result.returncode == 0, result.stderr
    module = _load(tmp_path / "shared_experts.py", "twice")
    layer = _Layer()
    experts = module.SharedExperts(
        layer=layer,
        moe_config=_moe_config(),
        enable_dbo=False,
        mk_can_overlap_shared_experts=lambda: False,
    )
    activation = _Tensor(8)
    order = module.SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
    experts.maybe_sync_shared_experts_stream(activation)
    experts.forward(activation, order)
    assert layer.calls == 1
    assert experts._early_pending == [False, False]
    # the caller consumed the single output the launch produced
    assert experts.output is not None
    assert experts._output == [None, None]
