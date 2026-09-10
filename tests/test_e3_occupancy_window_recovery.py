#!/usr/bin/env python3
"""Recovery-path tests for the task 24 W5 guarded window runner (CPU only).

The window stops production, so every failure path must put it back. These tests
drive ``main()`` and ``phase_capture()`` with mocked helpers and assert that
production and the timers are restored independently, that an interrupt takes the
same path, and that a profiling container is removed before the restart.
"""

import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).parents[1]


def load_runner():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "run_e3_occupancy_window", ROOT / "scripts" / "run_e3_occupancy_window.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed(rc: int = 0, out: str = "", err: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["cmd"], rc, out, err)


class WindowRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = load_runner()
        self.win = self.runner.win
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.receipt = Path(self.tmp.name) / "window.json"
        self.calls: list[str] = []
        self.original_signals = {
            sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)
        }
        for sig, handler in self.original_signals.items():
            self.addCleanup(signal.signal, sig, handler)
        self.rearm = mock.patch.object(
            self.win, "phase_rearm", side_effect=lambda state: self.calls.append("rearm")
        )
        self.rearm.start()
        self.addCleanup(self.rearm.stop)
        self.start = mock.patch.object(
            self.win, "guarded_start", side_effect=lambda: self.calls.append("start")
        )
        self.start.start()
        self.addCleanup(self.start.stop)
        # Defaults: a healthy head and no polling. Individual tests override
        # these; nothing here may reach the real server or block on a wait loop.
        self.curl = mock.patch.object(self.win, "curl", return_value=(200, ""))
        self.curl.start()
        self.addCleanup(self.curl.stop)
        self.health = mock.patch.object(self.win, "wait_health", return_value=True)
        self.health.start()
        self.addCleanup(self.health.stop)

    def run_window(self, handlers: dict, argv: list[str] | None = None) -> int:
        with mock.patch.dict(self.runner.HANDLERS, handlers, clear=False):
            return self.runner.main(
                argv or ["--from", "preflight", "--to", "rearm",
                         "--receipt", str(self.receipt)]
            )

    def fake_phases(self, failing: str, exc: BaseException) -> dict:
        def make(name):
            def handler(state):
                self.calls.append(name)
                if name == failing:
                    raise exc
            return handler
        return {name: make(name) for name in self.runner.PHASES}

    def receipt_state(self) -> dict:
        return json.loads(self.receipt.read_text())

    def test_sigterm_during_capture_takes_the_recovery_path(self) -> None:
        handlers = self.fake_phases("capture", KeyboardInterrupt("signal 15"))
        with mock.patch.object(self.win, "curl", return_value=(200, "")):
            rc = self.run_window(handlers)
        state = self.receipt_state()
        self.assertEqual(rc, 1)
        self.assertEqual(state["recovery"], "ok")
        self.assertEqual(state["recovery_start"], "skipped (already healthy)")
        self.assertIn("rearm", self.calls)
        self.assertTrue(state["phases"]["capture"].startswith("FAILED"))

    def test_signal_handlers_are_installed(self) -> None:
        seen: dict[int, object] = {}

        def capture(state):
            for sig in (signal.SIGTERM, signal.SIGHUP):
                seen[sig] = signal.getsignal(sig)

        handlers = {name: (lambda state: None) for name in self.runner.PHASES}
        handlers["capture"] = capture
        with mock.patch.object(self.win, "curl", return_value=(200, "")):
            self.run_window(handlers)
        self.assertEqual(seen[signal.SIGTERM], self.win._on_signal)
        self.assertEqual(seen[signal.SIGHUP], self.win._on_signal)

    def test_rearm_runs_even_when_the_restart_fails(self) -> None:
        handlers = self.fake_phases("judge", RuntimeError("auditor exited 1"))
        with mock.patch.object(self.win, "curl", return_value=(503, "")), \
                mock.patch.object(self.win, "guarded_start",
                                  side_effect=RuntimeError("prod-start.sh exited 1")):
            rc = self.run_window(handlers)
        state = self.receipt_state()
        self.assertEqual(rc, 1)
        self.assertIn("rearm", self.calls)
        self.assertTrue(state["recovery"].startswith("FAILED"))
        self.assertIn("restart failed", state["recovery"])

    def test_unhealthy_head_is_a_recovery_failure(self) -> None:
        handlers = self.fake_phases("judge", RuntimeError("auditor exited 1"))
        with mock.patch.object(self.win, "curl", return_value=(503, "")), \
                mock.patch.object(self.win, "wait_health", return_value=False):
            rc = self.run_window(handlers)
        state = self.receipt_state()
        self.assertEqual(rc, 1)
        self.assertIn("rearm", self.calls)
        self.assertTrue(state["recovery"].startswith("FAILED"))
        self.assertIn("did not become healthy", state["recovery"])

    def test_healthy_head_skips_the_restart_but_still_rearms(self) -> None:
        handlers = self.fake_phases("judge", RuntimeError("auditor exited 1"))
        with mock.patch.object(self.win, "curl", return_value=(200, "")):
            rc = self.run_window(handlers)
        state = self.receipt_state()
        self.assertEqual(rc, 1)
        self.assertNotIn("start", self.calls)
        self.assertIn("rearm", self.calls)
        self.assertEqual(state["recovery"], "ok")

    def test_container_is_removed_before_production_is_restarted(self) -> None:
        def capture(state):
            state["container_name"] = "w5-occupancy-test"
            raise RuntimeError("ncu exited 1")

        handlers = {name: (lambda state: None) for name in self.runner.PHASES}
        handlers["capture"] = capture
        with mock.patch.object(self.win, "curl", return_value=(503, "")), \
                mock.patch.object(self.win, "guarded_start",
                                  side_effect=lambda: self.calls.append("start")), \
                mock.patch.object(self.win, "run",
                                  side_effect=lambda *a, **k: (self.calls.append("docker rm"),
                                                               completed(0))[1]):
            self.run_window(handlers)
        self.assertEqual(self.calls.index("docker rm"), self.calls.index("start") - 1)
        state = self.receipt_state()
        self.assertEqual(state["container_cleanup"][0]["name"], "w5-occupancy-test")

    def test_stale_container_problem_is_reported(self) -> None:
        def capture(state):
            state["container_name"] = "w5-occupancy-test"
            raise RuntimeError("ncu exited 1")

        handlers = {name: (lambda state: None) for name in self.runner.PHASES}
        handlers["capture"] = capture
        with mock.patch.object(self.win, "curl", return_value=(200, "")), \
                mock.patch.object(self.win, "run",
                                  return_value=completed(1, err="Error: permission denied")):
            self.run_window(handlers)
        state = self.receipt_state()
        self.assertTrue(state["recovery"].startswith("FAILED"))
        self.assertIn("may still hold the GPU", state["recovery"])

    def test_gone_container_is_not_a_problem(self) -> None:
        def capture(state):
            state["container_name"] = "w5-occupancy-test"
            raise RuntimeError("ncu exited 1")

        handlers = {name: (lambda state: None) for name in self.runner.PHASES}
        handlers["capture"] = capture
        with mock.patch.object(self.win, "curl", return_value=(200, "")), \
                mock.patch.object(self.win, "run",
                                  return_value=completed(1, err="Error: No such container: x")):
            self.run_window(handlers)
        self.assertEqual(self.receipt_state()["recovery"], "ok")

    def test_cleanup_that_raises_still_rearms_the_timers(self) -> None:
        """A hung `docker rm -f` must not skip the timer restore."""
        def capture(state):
            state["container_name"] = "w5-occupancy-test"
            raise RuntimeError("ncu exited 1")

        handlers = {name: (lambda state: None) for name in self.runner.PHASES}
        handlers["capture"] = capture
        with mock.patch.object(self.win, "curl", return_value=(503, "")), \
                mock.patch.object(self.win, "guarded_start",
                                  side_effect=lambda: self.calls.append("start")), \
                mock.patch.object(self.win, "run",
                                  side_effect=subprocess.TimeoutExpired("docker", 120)):
            self.run_window(handlers)
        state = self.receipt_state()
        self.assertNotIn("start", self.calls)
        self.assertIn("rearm", self.calls)
        self.assertTrue(state["recovery"].startswith("FAILED"))
        self.assertIn("could not confirm removal", state["recovery"])

    def test_unconfirmed_container_does_not_start_a_competing_workload(self) -> None:
        def capture(state):
            state["container_name"] = "w5-occupancy-test"
            raise RuntimeError("ncu exited 1")

        handlers = {name: (lambda state: None) for name in self.runner.PHASES}
        handlers["capture"] = capture
        with mock.patch.object(self.win, "curl", return_value=(503, "")), \
                mock.patch.object(self.win, "run",
                                  return_value=completed(1, err="Error: permission denied")):
            self.run_window(handlers)
        state = self.receipt_state()
        self.assertNotIn("start", self.calls)
        self.assertIn("rearm", self.calls)
        self.assertTrue(state["recovery"].startswith("FAILED"))
        self.assertIn("may still hold the GPU", state["recovery"])
        self.assertIn("not started", state["recovery"])

    def test_unconfirmed_container_still_skips_the_restart_when_healthy(self) -> None:
        def capture(state):
            state["container_name"] = "w5-occupancy-test"
            raise RuntimeError("ncu exited 1")

        handlers = {name: (lambda state: None) for name in self.runner.PHASES}
        handlers["capture"] = capture
        with mock.patch.object(self.win, "curl", return_value=(200, "")), \
                mock.patch.object(self.win, "run",
                                  return_value=completed(1, err="Error: permission denied")):
            self.run_window(handlers)
        state = self.receipt_state()
        self.assertNotIn("start", self.calls)
        self.assertIn("rearm", self.calls)
        self.assertEqual(state["recovery_start"], "skipped (already healthy)")
        self.assertIn("may still hold the GPU", state["recovery"])

    def test_orphan_container_blocks_a_resumed_start(self) -> None:
        """--state … --from start must not race a profiler left by the capture."""
        state = {
            "backup": "/tmp/env.bak", "phases": {"disarm": "ok", "stop": "ok",
                                                 "capture": "running"},
            "container_name": "w5-occupancy-test",
        }
        self.receipt.write_text(json.dumps(state))
        with mock.patch.object(self.win, "curl", return_value=(503, "")) as curl, \
                mock.patch.object(self.win, "run",
                                  return_value=completed(1, err="Error: permission denied")):
            rc = self.run_window({}, ["--from", "start", "--to", "start",
                                      "--receipt", str(self.receipt)])
        self.assertEqual(rc, 1)
        self.assertNotIn("start", self.calls)
        self.assertIn("rearm", self.calls)
        # Only the recovery health check ran; the start phase never got that far.
        self.assertEqual(curl.call_count, 1)
        self.assertIn("refusing to start production", self.receipt_state()["error"])

    def test_resumed_start_cleans_up_before_restarting(self) -> None:
        state = {
            "backup": "/tmp/env.bak", "phases": {"disarm": "ok", "stop": "ok",
                                                 "capture": "running"},
            "container_name": "w5-occupancy-test",
        }
        self.receipt.write_text(json.dumps(state))

        def fake_run(argv, **kwargs):
            if argv[:3] == ["docker", "rm", "-f"]:
                self.calls.append("docker rm")
            return completed(0)

        with mock.patch.object(self.win, "curl", return_value=(503, "")), \
                mock.patch.object(self.win, "guarded_start",
                                  side_effect=lambda: self.calls.append("start")), \
                mock.patch.object(self.win, "run", side_effect=fake_run):
            rc = self.run_window({}, ["--from", "start", "--to", "start",
                                      "--receipt", str(self.receipt)])
        self.assertEqual(rc, 0)
        self.assertEqual(self.calls, ["docker rm", "start"])
        self.assertEqual(self.receipt_state()["phases"]["start"], "ok")

    def test_failure_before_disarm_does_not_restart_production(self) -> None:
        handlers = self.fake_phases("preflight", RuntimeError("no .env"))
        with mock.patch.object(self.win, "curl", return_value=(200, "")) as curl:
            rc = self.run_window(handlers, ["--from", "preflight", "--to", "disarm",
                                            "--receipt", str(self.receipt)])
        self.assertEqual(rc, 1)
        self.assertNotIn("start", self.calls)
        self.assertNotIn("rearm", self.calls)
        self.assertEqual(curl.call_count, 0)
        self.assertEqual(self.receipt_state()["error"], "RuntimeError('no .env')")


class CaptureCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = load_runner()
        self.win = self.runner.win
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "local").mkdir()
        # phase_capture checkpoints through the shared save(); a receipt path
        # left behind by another test class must not be written here.
        self.receipt = mock.patch.object(self.win, "_RECEIPT", None)
        self.receipt.start()
        self.addCleanup(self.receipt.stop)
        self.patches = [
            mock.patch.object(self.runner, "ROOT", self.root),
            mock.patch.object(self.runner, "NCU_HOST_DIR", self.root),
            mock.patch.object(self.win, "memfree_gib", return_value=100.0),
            mock.patch.object(self.win, "image_id", return_value="sha256:test"),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_timeout_removes_the_profiling_container(self) -> None:
        state = {"need_gib": 90.0, "launches": 3, "n_exp": 64, "iters": 3,
                 "ncu_timeout": 5.0}
        removed: list[list[str]] = []

        def fake_run(argv, **kwargs):
            if argv[:3] == ["docker", "rm", "-f"]:
                removed.append(argv)
                return completed(0)
            raise subprocess.TimeoutExpired(argv, 5.0)

        with mock.patch.object(self.runner.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(RuntimeError) as caught:
                self.runner.phase_capture(state)
        self.assertIn("exceeded", str(caught.exception))
        self.assertEqual(removed, [["docker", "rm", "-f", state["container_name"]]])
        self.assertTrue(state["container_name"].startswith("w5-occupancy-"))

    def test_nonzero_ncu_removes_the_profiling_container(self) -> None:
        state = {"need_gib": 90.0, "launches": 3, "n_exp": 64, "iters": 3,
                 "ncu_timeout": 5.0}
        removed: list[list[str]] = []

        def fake_run(argv, **kwargs):
            if argv[:3] == ["docker", "rm", "-f"]:
                removed.append(argv)
                return completed(0)
            return completed(1, err="ncu: no kernels profiled")

        with mock.patch.object(self.runner.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(RuntimeError) as caught:
                self.runner.phase_capture(state)
        self.assertIn("ncu exited 1", str(caught.exception))
        self.assertEqual(len(removed), 1)

    def test_capture_waits_for_memory_before_profiling(self) -> None:
        state = {"need_gib": 90.0, "launches": 3, "n_exp": 64, "iters": 3,
                 "ncu_timeout": 5.0}
        seen: list[list[str]] = []

        def fake_run(argv, **kwargs):
            seen.append(argv)
            return completed(1, err="stop after the memory wait")

        with mock.patch.object(self.win, "memfree_gib", return_value=1.0), \
                mock.patch.object(self.runner.time, "monotonic",
                                  side_effect=[0.0, 0.0, 1000.0]), \
                mock.patch.object(self.runner.time, "sleep"):
            with self.assertRaises(RuntimeError) as caught:
                self.runner.phase_capture(state)
        self.assertIn("MemFree did not reach", str(caught.exception))
        self.assertEqual(seen, [])
        self.assertNotIn("container_name", state)

    def test_unusable_probe_json_fails_closed(self) -> None:
        """A capture whose probe JSON has no usable median must not reach the judge."""
        state = {"need_gib": 90.0, "launches": 3, "n_exp": 64, "iters": 0,
                 "ncu_timeout": 5.0}

        def fake_run(argv, **kwargs):
            outdir = None
            for index, item in enumerate(argv):
                if item == "-v" and argv[index + 1].endswith(":/out"):
                    outdir = Path(argv[index + 1][: -len(":/out")])
            if outdir is not None:
                (outdir / "probe.json").write_text(json.dumps(
                    {"tier": "grouped", "last_fat_fallback": "grouped",
                     "out_finite": True, "routing": {"fat_rows": 0},
                     "median_ms": None}
                ))
            return completed(0)

        with mock.patch.object(self.runner.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(RuntimeError) as caught:
                self.runner.phase_capture(state)
        self.assertIn("median_ms", str(caught.exception))

    def test_judge_requires_the_launch_count_it_asked_ncu_for(self) -> None:
        """The auditor must be told how many launches ncu was asked to capture."""
        state = {"outdir": str(self.root), "launches": 3}
        (self.root / "ncu.csv").write_text("csv")
        seen: list[list[str]] = []

        def fake_run(argv, **kwargs):
            seen.append(argv)
            (self.root / "audit.json").write_text(json.dumps(
                {"decision": "STOP_REGISTER_HEADROOM", "reason": "registers/thread 128"}
            ))
            return completed(0)

        with mock.patch.object(self.runner.subprocess, "run", side_effect=fake_run):
            self.runner.phase_judge(state)
        self.assertEqual(seen[0][seen[0].index("--expected-launches") + 1], "3")
        self.assertIn("--out", seen[0])
        self.assertEqual(state["audit"]["decision"], "STOP_REGISTER_HEADROOM")

    def test_judge_failure_is_reported(self) -> None:
        state = {"outdir": str(self.root), "launches": 3}
        (self.root / "ncu.csv").write_text("csv")

        def fake_run(argv, **kwargs):
            (self.root / "audit.json").write_text(json.dumps(
                {"decision": "ABORT", "reason": "capture holds 2 launch record(s)"}
            ))
            return completed(1)

        with mock.patch.object(self.runner.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(RuntimeError) as caught:
                self.runner.phase_judge(state)
        self.assertIn("auditor exited 1", str(caught.exception))


class ArgumentGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = load_runner()

    def test_zero_counts_are_rejected_before_any_phase(self) -> None:
        for flag in ("--iters", "--n-exp", "--launches"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit) as caught:
                self.runner.main([flag, "0", "--receipt",
                                  str(Path(tempfile.mkdtemp()) / "w.json")])
            self.assertEqual(caught.exception.code, 2)


class ResumeMetadataTests(unittest.TestCase):
    """A resume must judge the capture it took, not argparse's defaults."""

    def setUp(self) -> None:
        self.runner = load_runner()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.receipt = self.root / "window.json"

    def write_state(self, **recorded) -> None:
        (self.root / "ncu.csv").write_text("csv")
        self.receipt.write_text(json.dumps({
            "backup": "/tmp/env.bak",
            "phases": {"disarm": "ok", "stop": "ok", "capture": "ok"},
            "outdir": str(self.root),
            **recorded,
        }))

    def judge_calls(self, argv: list[str]) -> tuple[int, list[list[str]]]:
        seen: list[list[str]] = []

        def fake_run(command, **kwargs):
            seen.append(command)
            (self.root / "audit.json").write_text(json.dumps(
                {"decision": "STOP_REGISTER_HEADROOM", "reason": "registers/thread 128"}
            ))
            return completed(0)

        with mock.patch.object(self.runner.subprocess, "run", side_effect=fake_run):
            return self.runner.main(argv), seen

    def test_resume_keeps_the_captured_launch_count(self) -> None:
        self.write_state(launches=6, n_exp=64, iters=3, ncu_timeout=60.0,
                         need_gib=90.0)
        rc, seen = self.judge_calls(
            ["--from", "judge", "--to", "judge", "--receipt", str(self.receipt)]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(seen[0][seen[0].index("--expected-launches") + 1], "6")
        self.assertEqual(json.loads(self.receipt.read_text())["launches"], 6)

    def test_conflicting_override_on_resume_is_refused(self) -> None:
        self.write_state(launches=6, ncu_timeout=60.0)
        err = io.StringIO()
        with mock.patch.object(self.runner.subprocess, "run") as run, \
                mock.patch.object(sys, "stderr", err):
            rc = self.runner.main(
                ["--from", "judge", "--to", "judge", "--launches", "3",
                 "--receipt", str(self.receipt)]
            )
        self.assertEqual(rc, 2)
        run.assert_not_called()
        self.assertIn("conflicts", err.getvalue())
        self.assertEqual(json.loads(self.receipt.read_text())["launches"], 6)

    def test_matching_override_on_resume_is_accepted(self) -> None:
        self.write_state(launches=6, ncu_timeout=60.0)
        rc, seen = self.judge_calls(
            ["--from", "judge", "--to", "judge", "--launches", "6",
             "--receipt", str(self.receipt)]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(seen[0][seen[0].index("--expected-launches") + 1], "6")

    def test_missing_metadata_falls_back_to_the_defaults(self) -> None:
        self.write_state()
        rc, seen = self.judge_calls(
            ["--from", "judge", "--to", "judge", "--receipt", str(self.receipt)]
        )
        self.assertEqual(rc, 0)
        state = json.loads(self.receipt.read_text())
        self.assertEqual(
            (state["launches"], state["n_exp"], state["iters"], state["need_gib"]),
            (3, 64, 3, 90.0),
        )
        self.assertEqual(seen[0][seen[0].index("--expected-launches") + 1], "3")


if __name__ == "__main__":
    unittest.main()
