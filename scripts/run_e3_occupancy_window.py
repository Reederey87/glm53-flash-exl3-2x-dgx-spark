#!/usr/bin/env python3
"""Guarded ncu occupancy window for the E3 grouped kernels (task 24 W5; run on spark1).

Production holds essentially all of the unified memory, so a second CUDA
process cannot start while it serves: measured 2026-09-09, even a 1 KiB
``cudaMalloc`` and ``cudaMemGetInfo`` fail with ``out of memory`` on the head
while ``vllm-glm53exl3`` runs. The hardware-counter lane therefore needs a
stopped window, exactly like the earlier kernel windows. The window itself is
the no-reboot counter path: ``docker run --cap-add SYS_ADMIN`` gives the
profiler ``ERR_NVGPUCTRPERM`` clearance without touching
``NVreg_RestrictProfilingToAdminUsers`` or rebooting (docs/14).

Phases run in order and can be bounded with --from/--to for a stepwise window:

  preflight  hashes, health, drain check, JIT stamp, image id (shared with the
             decode/prefill window runner)
  disarm     stop watchdog + metrics-alert timers, reset-failed
  stop       ./start.sh stop, wait for the container to disappear
  capture    wait for >= --need-gib MemFree on both nodes, run ncu on the
             offline E3 replica, keep the CSV + probe JSON in local/
  judge      run the occupancy auditor over the captured CSV (offline; a
             parser fix can re-judge the same capture with --state)
  start      local/prod-start.sh (validates, waits for memory, starts); a
             resume after the recovery path already restarted production
             skips the restart when /health is already 200
  gates      acceptance.sh, MemFree tripwire, pool line, JIT stamp unchanged
  rearm      re-enable the timers

Any failure after disarm restarts production and re-arms the timers before
exiting non-zero; the two are attempted independently, so a failed restart
cannot skip the re-arm. SIGTERM/SIGHUP/SIGINT take the same recovery path. A
profiling container is named and removed before production is restarted, because
``--rm`` only removes it once its workload exits and a killed Docker client does
not necessarily stop a hung one. The window never edits ``.env``. Resume a
partial window with ``--state <receipt> --from <phase>``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_decode_profile_window as win  # noqa: E402  (shared guarded-window helpers)

ROOT = win.ROOT
PROBE = ROOT / "scripts" / "probe_e3_occupancy.py"
AUDITOR = ROOT / "scripts" / "audit_e3_occupancy.py"
NCU_HOST_DIR = Path("/opt/nvidia/nsight-compute")
NCU_BIN = str(NCU_HOST_DIR / "2025.3.1" / "ncu")
HEAD_CONTAINER = "glm53-exl3-head"
PHASES = ("preflight", "disarm", "stop", "capture", "judge", "start", "gates", "rearm")


def log(message: str) -> None:
    print(f"[w5-occupancy] {message}", flush=True)


def tail(text: str, lines: int = 12) -> list[str]:
    return (text or "").strip().splitlines()[-lines:]


def cleanup_container(state: dict) -> str | None:
    """Remove the profiling container, if one may still exist.

    ``docker run --rm`` removes the container when its workload exits, but a
    killed Docker client does not stop a hung container, and that container still
    owns GPU memory. Call this before production is restarted. Returns a problem
    string when removal could not be confirmed, otherwise None.
    """
    name = state.get("container_name")
    if not name:
        return None
    try:
        proc = win.run(["docker", "rm", "-f", name], timeout=120, check=False)
    except Exception as exc:  # noqa: BLE001  (a hung docker must not hide the timer restore)
        state.setdefault("container_cleanup", []).append({"name": name, "error": repr(exc)})
        log(f"container cleanup failed: {exc!r}")
        return f"could not confirm removal of profiling container {name} ({exc!r})"
    out = (proc.stdout + proc.stderr).strip()
    state.setdefault("container_cleanup", []).append(
        {"name": name, "rc": proc.returncode, "tail": tail(out, 3)}
    )
    if proc.returncode == 0:
        log(f"removed profiling container {name}")
        return None
    if "No such container" in out:
        log(f"profiling container {name} is already gone")
        return None
    log(f"container cleanup rc={proc.returncode}: {out}")
    return f"profiling container {name} may still hold the GPU"


def recover(state: dict) -> None:
    """Put production and the timers back, each independently of the other.

    A failed restart must not skip the re-arm, a raise from the cleanup must not
    skip either, and production is not started on top of a profiler whose removal
    could not be confirmed (that would race it for the GPU). A head that never
    becomes healthy is a recovery failure, not a success.
    """
    log("recovering: drop any profiling container, ensure production is up, re-arm the timers")
    problems: list[str] = []
    orphan = cleanup_container(state)
    if orphan:
        problems.append(orphan)
    healthy = False
    try:
        code, _ = win.curl("/health", timeout=10)
        healthy = code == 200
    except Exception as exc:  # noqa: BLE001
        problems.append(f"health check failed: {exc!r}")
    if healthy:
        state["recovery_health"] = True
        state["recovery_start"] = "skipped (already healthy)"
    elif orphan:
        state["recovery_health"] = False
        state["recovery_start"] = "skipped (profiling container may still hold the GPU)"
        problems.append(
            "production is not running and was not started while the container is unconfirmed"
        )
    else:
        try:
            win.guarded_start()
            ok = win.wait_health()
            state["recovery_health"] = ok
            if not ok:
                problems.append("head did not become healthy after the recovery restart")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"restart failed: {exc!r}")
    try:
        win.phase_rearm(state)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"re-arm failed: {exc!r}")
    if problems:
        state["recovery"] = "FAILED: " + "; ".join(problems)
        log("RECOVERY FAILED: " + "; ".join(problems) + " — operator action required")
    else:
        state["recovery"] = "ok"
        log("recovery ok: production healthy and timers re-armed")


def phase_stop(state: dict) -> None:
    proc = subprocess.run([str(ROOT / "start.sh"), "stop"], check=False,
                          capture_output=True, text=True, timeout=900)
    state["stop_rc"] = proc.returncode
    state["stop_tail"] = tail(proc.stdout + proc.stderr)
    log("stop tail:\n" + "\n".join(state["stop_tail"]))
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        out = win.run(["docker", "ps", "-a", "--filter", f"name={HEAD_CONTAINER}",
                       "--format", "{{.Names}}"], timeout=30, check=False).stdout.strip()
        if not out:
            log("head container gone")
            return
        time.sleep(5)
    raise RuntimeError("head container still present after stop")


def phase_capture(state: dict) -> None:
    need = float(state["need_gib"])
    deadline = time.monotonic() + 900
    while True:
        head = win.memfree_gib()
        worker = win.memfree_gib(win.WORKER)
        if head >= need and worker >= need:
            state["memfree_before_capture"] = {"head": head, "worker": worker}
            log(f"memory free: head {head:.2f} GiB worker {worker:.2f} GiB")
            break
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"MemFree did not reach {need} GiB (head={head:.2f} worker={worker:.2f})"
            )
        time.sleep(10)

    if not NCU_HOST_DIR.is_dir():
        raise RuntimeError(f"missing {NCU_HOST_DIR}")
    image = win.image_id() or win.effective_env().get("IMAGE", "")
    if not image:
        raise RuntimeError("cannot resolve the production image id")
    outdir = ROOT / "local" / f"task24-w5-occupancy-{time.strftime('%Y%m%d-%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"w5-occupancy-{time.strftime('%Y%m%d-%H%M%S')}"
    state["container_name"] = name
    win.save(state)
    cmd = [
        "docker", "run", "--rm", "--name", name, "--gpus", "all", "--cap-add", "SYS_ADMIN",
        "-v", f"{NCU_HOST_DIR}:/opt/nvidia/nsight-compute:ro",
        "-v", f"{PROBE}:/probe.py:ro",
        "-v", f"{outdir}:/out",
        "--entrypoint", NCU_BIN, image,
        "--csv", "--log-file", "/out/ncu.csv",
        "--clock-control", "none", "--cache-control", "none",
        "--kernel-name", "regex:fm_(gateup|down)_kernel",
        "--launch-skip", "1", "--launch-count", str(state["launches"]),
        "--section", "LaunchStats", "--section", "Occupancy",
        "python3", "/probe.py",
        "--n-exp", str(state["n_exp"]), "--iters", str(state["iters"]),
        "--warm", "1", "--out", "/out/probe.json",
    ]
    state["ncu_argv"] = cmd
    state["outdir"] = str(outdir)
    log("running ncu: " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True,
                              timeout=float(state["ncu_timeout"]))
    except subprocess.TimeoutExpired as exc:
        problem = cleanup_container(state)
        raise RuntimeError(
            f"ncu exceeded {state['ncu_timeout']:.0f}s" + (f"; {problem}" if problem else "")
        ) from exc
    state["ncu_rc"] = proc.returncode
    state["ncu_stdout_tail"] = tail(proc.stdout)
    state["ncu_stderr_tail"] = tail(proc.stderr)
    log("ncu rc=" + str(proc.returncode))
    if proc.returncode != 0:
        log("ncu stderr:\n" + "\n".join(state["ncu_stderr_tail"]))
        problem = cleanup_container(state)
        raise RuntimeError(
            f"ncu exited {proc.returncode}" + (f"; {problem}" if problem else "")
        )

    probe_json = outdir / "probe.json"
    if not probe_json.is_file():
        raise RuntimeError("probe wrote no /out/probe.json")
    probe = json.loads(probe_json.read_text())
    state["probe"] = probe
    if probe.get("tier") != "grouped" or probe.get("last_fat_fallback") != "grouped":
        raise RuntimeError(
            f"probe did not exercise the grouped path: tier={probe.get('tier')} "
            f"fallback={probe.get('last_fat_fallback')}"
        )
    if not probe.get("out_finite"):
        raise RuntimeError("probe output is not finite")
    routing = probe.get("routing")
    if not isinstance(routing, dict):
        raise RuntimeError(f"probe JSON has no routing block: {routing!r}")
    median = probe.get("median_ms")
    if not isinstance(median, (int, float)) or isinstance(median, bool) \
            or not math.isfinite(median) or median <= 0:
        raise RuntimeError(
            f"probe median_ms is not a positive finite number: {median!r} "
            "(check --iters/--warm)"
        )
    log(f"probe: {routing} median={median:.2f} ms")


def phase_judge(state: dict) -> None:
    outdir = Path(state["outdir"])
    csv_path = outdir / "ncu.csv"
    if not csv_path.is_file():
        raise RuntimeError(f"missing capture {csv_path}")
    audit = subprocess.run(
        [sys.executable, str(AUDITOR), "--ncu-csv", str(csv_path),
         "--expected-launches", str(state["launches"]),
         "--out", str(outdir / "audit.json")],
        check=False, capture_output=True, text=True, timeout=600,
    )
    state["audit_rc"] = audit.returncode
    state["audit_stdout_tail"] = tail(audit.stdout)
    log("audit tail:\n" + "\n".join(state["audit_stdout_tail"]))
    if audit.returncode != 0:
        raise RuntimeError(f"occupancy auditor exited {audit.returncode}")
    state["audit"] = json.loads((outdir / "audit.json").read_text())
    log(f"verdict: {state['audit']['decision']} — {state['audit']['reason']}")


def phase_start(state: dict) -> None:
    # Enforced here as well as in recovery: a window resumed straight into
    # `start` (--state … --from start) must not race an orphaned profiler.
    orphan = cleanup_container(state)
    if orphan:
        raise RuntimeError(f"{orphan}; refusing to start production")
    code, _ = win.curl("/health", timeout=10)
    if code == 200:
        state["start_skipped"] = "already healthy (recovery path or resume)"
        log("production already healthy; skipping the restart")
    else:
        win.guarded_start()
        if not win.wait_health():
            raise RuntimeError("head did not become healthy after the window")
    state["image_after"] = win.image_id()
    log("production healthy after the window")


HANDLERS = {
    "preflight": win.phase_preflight,
    "disarm": win.phase_disarm,
    "stop": phase_stop,
    "capture": phase_capture,
    "judge": phase_judge,
    "start": phase_start,
    "gates": win.phase_gates,
    "rearm": win.phase_rearm,
}


def phase_preflight(state: dict) -> None:
    win.phase_preflight(state)
    # The shared preflight records the decode-window auditor; this window's
    # decision artifact is the E3 occupancy auditor.
    state["auditor_sha256"] = win.sha256(AUDITOR)
    win.save(state)


HANDLERS["preflight"] = phase_preflight


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="start", choices=PHASES, default="preflight")
    ap.add_argument("--to", dest="end", choices=PHASES, default="rearm")
    ap.add_argument("--tag", default="task24-w5")
    ap.add_argument("--receipt", type=Path, default=None)
    ap.add_argument("--state", type=Path, default=None,
                    help="resume/checkpoint file (loaded if present)")
    ap.add_argument("--n-exp", type=int, default=None,
                    help="routed experts in the probe (default 64; a resumed "
                         "window keeps the receipt's value)")
    ap.add_argument("--iters", type=int, default=None,
                    help="probe iterations (default 3; a resumed window keeps the "
                         "receipt's value)")
    ap.add_argument("--launches", type=int, default=None,
                    help="ncu --launch-count, i.e. the launches the capture must "
                         "hold (default 3; a resumed window keeps the receipt's value)")
    ap.add_argument("--need-gib", type=float, default=None,
                    help="MemFree to wait for before profiling (default 90; a "
                         "resumed window keeps the receipt's value)")
    ap.add_argument("--ncu-timeout", type=float, default=None,
                    help="seconds before the ncu client is killed (default 1800; a "
                         "resumed window keeps the receipt's value)")
    args = ap.parse_args(argv)
    # A zero iteration/expert/launch count cannot produce a judgeable capture;
    # reject it before the window stops production.
    overrides = {
        "n_exp": args.n_exp,
        "iters": args.iters,
        "launches": args.launches,
        "need_gib": args.need_gib,
        "ncu_timeout": args.ncu_timeout,
    }
    for name, value in overrides.items():
        if value is not None and value < 1:
            ap.error(f"--{name.replace('_', '-')} must be >= 1 (got {value})")

    win._TAG = args.tag
    win._PROBE = PROBE
    if not PROBE.is_file() or not AUDITOR.is_file():
        raise SystemExit(f"missing {PROBE} or {AUDITOR}")

    receipt = args.receipt or args.state or (
        ROOT / "local" / f"{args.tag}-window-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    win._RECEIPT = receipt
    order = PHASES[PHASES.index(args.start):PHASES.index(args.end) + 1]
    state: dict = {}
    if receipt.is_file():
        try:
            state = json.loads(receipt.read_text())
        except json.JSONDecodeError:
            state = {}
    if args.start != PHASES[0] and "backup" not in state:
        print(f"--from {args.start} needs a prior preflight state in {receipt}",
              file=sys.stderr)
        return 2
    if state:
        # A resumed window keeps the failure that forced the resume, and drops
        # the pre-rename phase key so the receipt has one phase vocabulary.
        if "error" in state:
            state.setdefault("previous_errors", []).append(state.pop("error"))
        state.get("phases", {}).pop("measure", None)
    # These knobs describe the capture that was actually taken, and the receipt
    # is the record of it. A resume must not silently re-label that capture with
    # an argparse default (that would, for example, judge a six-launch profile
    # against the default count of three), so a recorded value wins and a
    # conflicting override is refused rather than applied.
    for name, given in overrides.items():
        recorded = state.get(name)
        if recorded is not None and given is not None and given != recorded:
            print(f"--{name.replace('_', '-')} {given} conflicts with {recorded} already "
                  f"recorded in {receipt}; resume without the flag to keep the capture's "
                  "own setting", file=sys.stderr)
            return 2
    state.update({
        "tag": args.tag,
        **{name: (state[name] if state.get(name) is not None
                  else (given if given is not None else default))
           for name, given, default in (
               ("n_exp", args.n_exp, 64),
               ("iters", args.iters, 3),
               ("launches", args.launches, 3),
               ("need_gib", args.need_gib, 90.0),
               ("ncu_timeout", args.ncu_timeout, 1800.0),
           )},
        "receipt": str(receipt),
        "started": state.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    state.setdefault("phases_requested", order)
    state.setdefault("runs", []).append(order)
    state.setdefault("phases", {})
    win._ACTIVE = state
    win.save(state)
    log(f"receipt {receipt}")

    # The shared runner registers these in its own main(); importing it is not
    # enough. Without them a SIGTERM kills this process outright and leaves
    # production stopped with the timers disarmed.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, win._on_signal)

    rc = 0
    phase = order[0]
    failure: BaseException | None = None
    try:
        for phase in order:
            log(f"phase {phase}")
            state.setdefault("phases", {})[phase] = "running"
            win.save(state)
            HANDLERS[phase](state)
            state["phases"][phase] = "ok"
            win.save(state)
    except BaseException as exc:  # noqa: BLE001  (must still restore production)
        failure = exc
        state["error"] = repr(exc)
        state["phases"][phase] = f"FAILED: {exc!r}"
        win.save(state)
        log(f"FAILED in {phase}: {exc!r}")
        rc = 1
    finally:
        # Recovery runs for an interrupt as well as for a phase failure, and the
        # receipt is written even if recovery itself blows up.
        if failure is not None:
            try:
                if "disarm" in state.get("phases", {}):
                    recover(state)
                else:
                    cleanup_container(state)
            except Exception as exc:  # noqa: BLE001
                state["recovery"] = f"FAILED: unexpected {exc!r}"
                log(f"RECOVERY FAILED: unexpected {exc!r} — operator action required")
        state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        win.save(state)
        log(f"receipt written to {receipt}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
