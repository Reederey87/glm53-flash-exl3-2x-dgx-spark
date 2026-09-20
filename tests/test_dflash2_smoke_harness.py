#!/usr/bin/env python3
"""Host regression tests for the DFlash2 cluster smoke harnesses.

These exist because both harnesses once exited 0 unconditionally: a failed
acceptance, a failed identity probe, an unsupported benchmark flag, and a
missing lane all still reached "DONE" with status 0. An always-zero harness
cannot be used as a gate, so the exit-status contract is pinned here.

The scripts are driven against a fixture kit directory with PATH shims for
`docker`, `ssh` and `curl`, so no container, cluster or GPU is needed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "local" / "dflash2-smoke.sh"
BENCH = ROOT / "local" / "dflash2-bench.sh"

STUB_PROBE_PASS = "print('IDENTITY PASS')\n"
STUB_PROBE_FAIL = "import sys\nprint('IDENTITY FAIL')\nsys.exit(1)\n"

STUB_ACCEPT_PASS = "#!/usr/bin/env bash\necho '  ACCEPTANCE PASSED'\nexit 0\n"
STUB_ACCEPT_FAIL = "#!/usr/bin/env bash\necho '  ACCEPTANCE FAILED'\nexit 1\n"

STUB_BENCH = """#!/usr/bin/env python3
import json
import os
import sys

out = sys.argv[sys.argv.index("--out") + 1]
mode = os.environ.get("STUB_BENCH_MODE", "ok")
if mode == "fail":
    print("bench_decode: unrecognized arguments: --essay")
    sys.exit(2)
if mode == "empty":
    sys.exit(0)
json.dump(
    {
        "tok_s_median": 70.0,
        "tok_s_min": 69.0,
        "tok_s_max": 71.0,
        "accept_ratio_median": 1.0,
        "accepted_per_step_median": 7.0,
        "any_nan": False,
        "runs": [],
    },
    open(out, "w"),
)
sys.exit(0)
"""


def _write_shim(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _make_fixture(tmp: Path, *, accept_pass: bool, probe_pass: bool) -> tuple[Path, Path]:
    """Build a fixture kit dir plus a PATH shim dir. Returns (kit, shim)."""
    kit = tmp / "kit"
    (kit / "local").mkdir(parents=True)
    (kit / "tests").mkdir(parents=True)

    shutil.copy(SMOKE, kit / "local" / "dflash2-smoke.sh")
    shutil.copy(BENCH, kit / "local" / "dflash2-bench.sh")
    (kit / "local" / "dflash2-smoke.sh").chmod(0o755)
    (kit / "local" / "dflash2-bench.sh").chmod(0o755)

    _write_shim(
        kit / "local" / "acceptance.sh",
        STUB_ACCEPT_PASS if accept_pass else STUB_ACCEPT_FAIL,
    )
    _write_shim(
        kit / "local" / "smoke_identity.py",
        STUB_PROBE_PASS if probe_pass else STUB_PROBE_FAIL,
    )
    _write_shim(kit / "tests" / "bench_decode.py", STUB_BENCH)

    shim = tmp / "shim"
    shim.mkdir()
    # `docker exec -i <ctr> python3 -` must actually run the streamed probe so
    # the probe's own exit status reaches the harness.
    _write_shim(
        shim / "docker",
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "exec" ]; then exec python3 -; fi\n'
        'if [ "${1:-}" = "inspect" ]; then echo "image=stub id=stub"; fi\n'
        "exit 0\n",
    )
    _write_shim(shim / "ssh", "#!/usr/bin/env bash\necho 'worker: 1000 kB'\nexit 0\n")
    _write_shim(shim / "curl", "#!/usr/bin/env bash\nexit 0\n")
    return kit, shim


def _run(script: Path, kit: Path, shim: Path, **env_extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = f"{shim}{os.pathsep}{env['PATH']}"
    env["GLM53_KIT_DIR"] = str(kit)
    env.update(env_extra)
    return subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def test_smoke_fails_when_acceptance_fails(tmp_path: Path) -> None:
    kit, shim = _make_fixture(tmp_path, accept_pass=False, probe_pass=True)
    result = _run(kit / "local" / "dflash2-smoke.sh", kit, shim)
    assert result.returncode != 0, result.stdout
    assert "FAILED: acceptance.sh" in result.stdout
    assert "SMOKE FAIL" in result.stdout


def test_smoke_fails_when_identity_probe_fails(tmp_path: Path) -> None:
    kit, shim = _make_fixture(tmp_path, accept_pass=True, probe_pass=False)
    result = _run(kit / "local" / "dflash2-smoke.sh", kit, shim)
    assert result.returncode != 0, result.stdout
    assert "FAILED: identity probe" in result.stdout
    assert "SMOKE FAIL" in result.stdout


def test_smoke_passes_when_required_checks_pass(tmp_path: Path) -> None:
    kit, shim = _make_fixture(tmp_path, accept_pass=True, probe_pass=True)
    result = _run(kit / "local" / "dflash2-smoke.sh", kit, shim)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SMOKE PASS" in result.stdout


def test_bench_fails_when_driver_errors(tmp_path: Path) -> None:
    kit, shim = _make_fixture(tmp_path, accept_pass=True, probe_pass=True)
    out = tmp_path / "bench-fail"
    result = _run(
        kit / "local" / "dflash2-bench.sh",
        kit,
        shim,
        BENCH_DECODE=str(kit / "tests" / "bench_decode.py"),
        BENCH_OUT=str(out),
        BENCH_SETTLE="0",
        STUB_BENCH_MODE="fail",
    )
    assert result.returncode != 0, result.stdout
    assert "FAILED: bench_decode" in result.stdout
    assert "BENCH FAIL" in result.stdout


def test_bench_fails_when_a_lane_is_missing(tmp_path: Path) -> None:
    """The original bug: an unsupported flag left a lane absent and still exited 0."""
    kit, shim = _make_fixture(tmp_path, accept_pass=True, probe_pass=True)
    out = tmp_path / "bench-empty"
    result = _run(
        kit / "local" / "dflash2-bench.sh",
        kit,
        shim,
        BENCH_DECODE=str(kit / "tests" / "bench_decode.py"),
        BENCH_OUT=str(out),
        BENCH_SETTLE="0",
        STUB_BENCH_MODE="empty",
    )
    assert result.returncode != 0, result.stdout
    assert "MISSING LANES" in result.stdout
    assert "BENCH FAIL" in result.stdout


def test_bench_passes_when_all_lanes_land(tmp_path: Path) -> None:
    kit, shim = _make_fixture(tmp_path, accept_pass=True, probe_pass=True)
    out = tmp_path / "bench-ok"
    result = _run(
        kit / "local" / "dflash2-bench.sh",
        kit,
        shim,
        BENCH_DECODE=str(kit / "tests" / "bench_decode.py"),
        BENCH_OUT=str(out),
        BENCH_SETTLE="0",
        STUB_BENCH_MODE="ok",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BENCH PASS" in result.stdout
    for lane in ("structured", "hashmap", "essay"):
        rec = json.loads((out / f"{lane}.json").read_text())
        assert rec["tok_s_median"] == 70.0


def test_receipted_bench_json_has_the_fields_the_summary_reads() -> None:
    """Guard the field names the summary depends on, against the real receipts."""
    bench = ROOT / "local" / "dflash2-smoke-receipts-20260920" / "bench"
    for lane in ("structured", "hashmap", "essay"):
        rec = json.loads((bench / f"{lane}.json").read_text())
        assert isinstance(rec["tok_s_median"], (int, float))
        assert isinstance(rec["accept_ratio_median"], (int, float))
        # The per-run acceptance is nested under spec, not at run level.
        assert "accept" not in rec["runs"][0]
        assert isinstance(rec["runs"][0]["spec"]["accept_ratio"], (int, float))


if __name__ == "__main__":
    import tempfile

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        with tempfile.TemporaryDirectory() as td:
            try:
                if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                    fn(Path(td))
                else:
                    fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    if failures:
        print(f"test_dflash2_smoke_harness: {failures} failure(s)")
        sys.exit(1)
    print("test_dflash2_smoke_harness: ok")
