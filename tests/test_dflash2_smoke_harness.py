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


def test_bench_rejects_stale_receipts_from_a_previous_run(tmp_path: Path) -> None:
    """A prior success in the same BENCH_OUT must not satisfy the current run."""
    kit, shim = _make_fixture(tmp_path, accept_pass=True, probe_pass=True)
    out = tmp_path / "bench-reused"
    args = {
        "BENCH_DECODE": str(kit / "tests" / "bench_decode.py"),
        "BENCH_OUT": str(out),
        "BENCH_SETTLE": "0",
    }

    first = _run(kit / "local" / "dflash2-bench.sh", kit, shim, STUB_BENCH_MODE="ok", **args)
    assert first.returncode == 0, first.stdout
    for lane in ("structured", "hashmap", "essay"):
        assert (out / f"{lane}.json").is_file()

    # Same output directory, driver exits 0 and writes nothing.
    second = _run(kit / "local" / "dflash2-bench.sh", kit, shim, STUB_BENCH_MODE="empty", **args)
    assert second.returncode != 0, second.stdout
    assert "MISSING LANES" in second.stdout
    assert "BENCH FAIL" in second.stdout
    for lane in ("structured", "hashmap", "essay"):
        assert not (out / f"{lane}.json").exists()


def test_bench_fails_when_stale_receipt_cleanup_fails(tmp_path: Path) -> None:
    """Unremovable stale receipts must abort, not silently continue to PASS.

    Setup mirrors the reachable failure: the lane JSON stays READABLE while
    `rm` cannot unlink it. That needs a non-writable output directory, which
    still allows the per-lane log redirect because those files already exist and
    stay owner-writable -- exactly the case where an unguarded harness would go
    on to consume the stale receipt and print BENCH PASS.
    """
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return  # root bypasses directory permissions, so the setup is not meaningful

    kit, shim = _make_fixture(tmp_path, accept_pass=True, probe_pass=True)
    out = tmp_path / "bench-locked"
    args = {
        "BENCH_DECODE": str(kit / "tests" / "bench_decode.py"),
        "BENCH_OUT": str(out),
        "BENCH_SETTLE": "0",
    }

    first = _run(kit / "local" / "dflash2-bench.sh", kit, shim, STUB_BENCH_MODE="ok", **args)
    assert first.returncode == 0, first.stdout
    assert (out / "hashmap.json").is_file()

    # Directory read-only (so unlink fails) but the logs stay writable (so the
    # per-lane redirect still succeeds).
    out.chmod(0o555)
    try:
        second = _run(kit / "local" / "dflash2-bench.sh", kit, shim, STUB_BENCH_MODE="empty", **args)
        assert second.returncode != 0, second.stdout
        assert "cannot clear stale receipt" in second.stderr + second.stdout
        assert "BENCH PASS" not in second.stdout
        # The stale lane is still on disk: the run refused rather than consumed it.
        assert (out / "hashmap.json").is_file()
    finally:
        out.chmod(0o755)


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
