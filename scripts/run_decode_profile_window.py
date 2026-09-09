#!/usr/bin/env python3
"""Guarded decode-profile window for the task 29/31 oracle (runs on spark1).

Adds ONLY the two profiler knobs to ``.env`` (hash-neutral: ProfilerConfig
.compute_hash is a constant), restarts the pair through the validated
``local/prod-start.sh``, captures one C4 decode window with the in-process
torch profiler, copies the per-rank traces into ``local/``, then restores the
exact pre-window ``.env`` and reboots production. Any failure on the arm side
triggers the same restore unless ``--keep-armed`` is passed.

Phases run in order and can be bounded with --from/--to for a stepwise window:

  preflight  record hashes, back up .env, drain check, JIT stamp
  disarm     stop watchdog + metrics-alert timers, reset-failed
  arm        append the profiler knobs, boot the pair, wait for health
  profile    mkdir trace dirs, run the C4 probe, collect both-rank traces
  restore    restore .env, boot production, wait for health
  gates      acceptance.sh, MemFree tripwire, pool/shape/MemFree receipt
  rearm      re-enable the timers
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
STAMP = Path.home() / ".cache" / "vllm-glm53-flash" / ".config-shape"
TRACE_HOST_DIR = Path.home() / ".cache" / "vllm-glm53-flash" / "profiler"
PROBE = ROOT / "scripts" / "probe_decode_profile.py"
BASE = "http://127.0.0.1:8000"
WORKER = os.environ.get("WORKER_SSH", "nvidia@192.168.177.11")
ARM_KEYS = ("GLM53_PROFILE_TORCH_DIR", "GLM53_PROFILE_MAX_ITERS")
SELECTED = (
    "IMAGE",
    "MODEL",
    "DFLASH_MODEL",
    "DFLASH_REVISION",
    "DFLASH_TOKENS",
    "MAX_NUM_SEQS",
    "MAX_NUM_BATCHED_TOKENS",
    "MAX_MODEL_LEN",
    "LONG_PREFILL_TOKEN_THRESHOLD",
    "EXL3_FAT_GROUPED",
    "EXL3_TEMP_ROWS_FUSED",
    "GLM53_ADAPTIVE_K",
    "GLM53_ADAPTIVE_K_CAPTURE",
    "GLM53_INDEXER_WORKSPACE",
)
PHASES = ("preflight", "disarm", "arm", "profile", "restore", "gates", "rearm")
TIMERS = ("vllm-glm53exl3-watchdog.timer", "glm53exl3-metrics-alert.timer")
MIN_TRACE_BYTES = 1 << 20

# Set by main(); the interrupt path needs them without threading state through
# every handler.
_RECEIPT: Path | None = None
_ACTIVE: dict | None = None
_KEEP_ARMED = False
_RESTORE_DONE = False


def log(message: str) -> None:
    print(f"[decode-profile] {message}", flush=True)


def save(state: dict) -> None:
    """Atomically persist the receipt so an interrupted window stays recoverable."""
    if _RECEIPT is None:
        return
    tmp = _RECEIPT.with_suffix(_RECEIPT.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, default=str) + "\n")
    os.replace(tmp, _RECEIPT)


def armed_now() -> bool:
    try:
        env = effective_env()
    except OSError:
        return False
    return any(key in env for key in ARM_KEYS)


def needs_env_restore(state: dict) -> bool:
    """True when production may be running the armed boot and needs a reboot."""
    if "backup" not in state:
        return False
    return bool(state.get("armed_attempted")) or armed_now()


def needs_timer_restore(state: dict) -> bool:
    """True when the disarm phase may have left a timer stopped."""
    return bool(state.get("disarm_attempted"))


def recover_timers(state: dict) -> None:
    """Re-enable the watchdog timers after a failed or interrupted disarm.

    Independent of the .env restore: a window can stop a timer and then die
    before it ever arms the profiler, and that must not leave monitoring off.
    """
    if not needs_timer_restore(state):
        return
    try:
        phase_rearm(state)
        state["timer_restore"] = "ok"
    except Exception as exc:  # noqa: BLE001
        state["timer_restore"] = f"FAILED: {exc!r}"
        log(f"TIMER RESTORE FAILED: {exc} — operator action required")
    save(state)


def emergency_restore() -> None:
    """Restore production when the window is interrupted outside its own flow."""
    global _RESTORE_DONE
    state = _ACTIVE
    if _RESTORE_DONE or _KEEP_ARMED or not state:
        return
    if not (needs_env_restore(state) or needs_timer_restore(state)):
        return
    _RESTORE_DONE = True
    if needs_env_restore(state):
        log("interrupted — restoring the pre-window .env and rebooting production")
        try:
            phase_restore(state)
            state["auto_restore"] = "ok (interrupt)"
        except Exception as exc:  # noqa: BLE001
            state["auto_restore"] = f"FAILED: {exc!r}"
            log(f"EMERGENCY RESTORE FAILED: {exc} — operator action required")
    else:
        state["auto_restore"] = "not needed (production was never armed)"
    recover_timers(state)
    save(state)


def run(argv: list[str], timeout: float = 600, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=check, capture_output=True, text=True, timeout=timeout)


def curl(path: str, method: str = "GET", timeout: float = 15) -> tuple[int, str]:
    proc = subprocess.run(
        ["curl", "-s", "-o", "-", "-w", "\n%{http_code}", "--max-time", str(int(timeout)),
         "-X", method, BASE + path],
        check=False, capture_output=True, text=True, timeout=timeout + 10,
    )
    body, _, code = proc.stdout.rpartition("\n")
    try:
        return int(code), body
    except ValueError:
        return 0, proc.stdout


def effective_env() -> dict[str, str]:
    text = ENV_FILE.read_text()
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key in SELECTED or key in ARM_KEYS:
            out[key] = value.strip().strip("'\"")
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memfree_gib(host: str | None = None) -> float:
    cmd = ["awk", "/^MemFree:/ {printf \"%.2f\", $2/1048576}", "/proc/meminfo"]
    if host:
        text = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                    "awk '/^MemFree:/ {printf \"%.2f\", $2/1048576}' /proc/meminfo"], timeout=20).stdout
    else:
        text = Path("/proc/meminfo").read_text()
        text = f"{int([l for l in text.splitlines() if l.startswith('MemFree:')][0].split()[1])/1048576:.2f}"
    return float(text.strip())


def wait_health(timeout: float = 2400) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, _ = curl("/health", timeout=10)
        if code == 200:
            return True
        time.sleep(15)
    return False


def drain(timeout: float = 300) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, body = curl("/metrics", timeout=10)
        if code != 200:
            return True  # nothing serving
        running = sum(float(m) for m in re.findall(r"vllm:num_requests_(?:running|waiting)\{[^}]*\}\s+(\S+)", body))
        if running == 0:
            return True
        log(f"drain: {running:.0f} in flight, waiting")
        time.sleep(10)
    return False


def guarded_start() -> None:
    log("starting pair through local/prod-start.sh (validates, waits for memory)")
    proc = subprocess.run([str(ROOT / "local" / "prod-start.sh")], check=False, text=True, timeout=5400)
    if proc.returncode != 0:
        raise RuntimeError(f"prod-start.sh exited {proc.returncode}")


def pool_line() -> str:
    proc = run(["docker", "logs", "--tail", "4000", "glm53-exl3-head"], timeout=60, check=False)
    for line in proc.stderr.splitlines() + proc.stdout.splitlines():
        if "KV cache" in line or "kv_cache" in line.lower():
            return line.strip()
    return ""


def image_id() -> str:
    return run(["docker", "inspect", "-f", "{{.Image}}", "glm53-exl3-head"], timeout=30, check=False).stdout.strip()


def arm_env() -> None:
    text = ENV_FILE.read_text()
    if not text.endswith("\n"):
        text += "\n"
    text += (
        "\n# task 29/31 decode-profile oracle window (removed by restore)\n"
        "GLM53_PROFILE_TORCH_DIR=/root/.cache/vllm/profiler\n"
        "GLM53_PROFILE_MAX_ITERS=2000\n"
    )
    ENV_FILE.write_text(text)


def restore_env(backup: Path) -> None:
    shutil.copy2(backup, ENV_FILE)


def phase_preflight(state: dict) -> None:
    if not ENV_FILE.is_file():
        raise RuntimeError(f"missing {ENV_FILE}")
    if not PROBE.is_file():
        raise RuntimeError(f"missing {PROBE}")
    stamp_before = STAMP.read_text().strip() if STAMP.is_file() else ""
    backup = ROOT / f".env.bak-pre-task29-profile-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(ENV_FILE, backup)
    # Persist the recovery coordinates before anything else can fail: an
    # interrupted window must still know which file to restore from.
    state.update({"backup": str(backup), "env_sha256": sha256(ENV_FILE)})
    save(state)
    env = effective_env()
    if any(k in env for k in ARM_KEYS):
        raise RuntimeError("profiler knobs already present in .env — window not clean")
    code, _ = curl("/health", timeout=10)
    if code == 200 and not drain():
        raise RuntimeError("server did not drain; refusing to take it down")
    state.update(
        {
            "start_sha256": sha256(ROOT / "start.sh"),
            "probe_sha256": sha256(PROBE),
            "auditor_sha256": sha256(ROOT / "scripts" / "audit_decode_kernel_share.py"),
            "jit_stamp_before": stamp_before,
            "image_before": image_id(),
            "env_effective": env,
        }
    )
    log(f"preflight OK backup={backup.name} stamp={stamp_before[:12]}")


def timer_states() -> dict[str, str]:
    out: dict[str, str] = {}
    for unit in TIMERS:
        proc = run(["systemctl", "--user", "is-active", unit], timeout=30, check=False)
        out[unit] = proc.stdout.strip() or "unknown"
    return out


def phase_disarm(state: dict) -> None:
    # Record the intent before the first stop: a partial disarm (or an interrupt
    # mid-phase) must still re-enable monitoring on the way out.
    state["disarm_attempted"] = True
    save(state)
    for unit in TIMERS:
        proc = run(["systemctl", "--user", "stop", unit], timeout=60, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"systemctl stop {unit} exited {proc.returncode}: {proc.stderr.strip()}")
    run(["systemctl", "--user", "reset-failed"], timeout=60, check=False)
    states = timer_states()
    state["timers_after_disarm"] = states
    bad = {unit: status for unit, status in states.items() if status == "active"}
    if bad:
        raise RuntimeError(f"timers still active after disarm: {bad}")
    log(f"watchdog + metrics-alert timers disarmed {states}, failed units reset")


def phase_arm(state: dict) -> None:
    # Record the intent to arm before touching .env: if the window dies between
    # here and restore, the recovery path must know production was modified.
    state["armed_attempted"] = True
    save(state)
    arm_env()
    state["env_sha256_armed"] = sha256(ENV_FILE)
    save(state)
    guarded_start()
    if not wait_health():
        raise RuntimeError("head did not become healthy on the armed boot")
    code, _ = curl("/start_profile", method="GET", timeout=15)
    if code != 405:
        raise RuntimeError(f"armed boot did not mount /start_profile (GET -> {code})")
    state["armed_env"] = effective_env()
    state["armed_image"] = image_id()
    log("armed boot healthy and /start_profile mounted (405 on GET)")


def phase_profile(state: dict) -> None:
    TRACE_HOST_DIR.mkdir(parents=True, exist_ok=True)
    for old in TRACE_HOST_DIR.glob("*"):
        old.unlink()
    proc = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", WORKER,
                "mkdir -p ~/.cache/vllm-glm53-flash/profiler && rm -f ~/.cache/vllm-glm53-flash/profiler/*"],
               timeout=60, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"worker trace-dir reset failed: {proc.stderr.strip()}")
    receipt = ROOT / "local" / f"task29-profile-probe-{time.strftime('%Y%m%d-%H%M%S')}.json"
    proc = subprocess.run(
        [sys.executable, str(PROBE), "--warmup", "--out", str(receipt),
         "--trace-dir", str(TRACE_HOST_DIR)],
        check=False, capture_output=True, text=True, timeout=3600,
        env={**os.environ, "GLM53_PROFILE_MAX_ITERS": "2000",
             "GLM53_PROFILE_TORCH_DIR": "/root/.cache/vllm/profiler"},
    )
    state["probe_stdout_tail"] = proc.stdout.strip().splitlines()[-12:]
    state["probe_stderr_tail"] = proc.stderr.strip().splitlines()[-12:]
    log("probe stdout:\n" + "\n".join(state["probe_stdout_tail"]))
    if proc.returncode != 0:
        log("probe stderr:\n" + "\n".join(state["probe_stderr_tail"]))
        raise RuntimeError(f"probe exited {proc.returncode}")
    outdir = ROOT / "local" / f"task29-traces-{time.strftime('%Y%m%d-%H%M%S')}"
    (outdir / "head").mkdir(parents=True)
    (outdir / "worker").mkdir(parents=True)
    for trace in TRACE_HOST_DIR.glob("*"):
        shutil.copy2(trace, outdir / "head" / trace.name)
    proc = run(["rsync", "-a", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10",
                f"{WORKER}:~/.cache/vllm-glm53-flash/profiler/", str(outdir / "worker" / "")],
               timeout=900, check=False)
    state["probe_receipt"] = str(receipt)
    state["trace_dir"] = str(outdir)
    state["trace_collect_rc"] = {"worker_reset": 0, "rsync": proc.returncode}
    if proc.returncode != 0:
        raise RuntimeError(f"rsync of worker traces exited {proc.returncode}: {proc.stderr.strip()}")
    state["head_traces"] = require_traces(outdir, "head")
    state["worker_traces"] = require_traces(outdir, "worker")
    log(f"traces collected head={len(state['head_traces'])} worker={len(state['worker_traces'])}")


def require_traces(outdir: Path, rank: str) -> list[dict[str, object]]:
    """Fail closed when a rank produced no usable profiler trace."""
    files = sorted(p for p in (outdir / rank).glob("*") if p.is_file())
    sizes = [{"name": p.name, "bytes": p.stat().st_size} for p in files]
    traces = [entry for entry in sizes if str(entry["name"]).endswith(".pt.trace.json.gz")]
    if len(traces) != 1:
        raise RuntimeError(f"expected exactly one profiler trace for {rank}, got {sizes}")
    small = [entry for entry in traces if int(entry["bytes"]) < MIN_TRACE_BYTES]
    if small:
        raise RuntimeError(f"{rank} trace too small to contain a decode window: {small}")
    return sizes


def phase_restore(state: dict) -> None:
    backup = Path(state["backup"])
    restore_env(backup)
    if sha256(ENV_FILE) != state["env_sha256"]:
        raise RuntimeError("restored .env does not match the pre-window hash")
    guarded_start()
    if not wait_health():
        raise RuntimeError("production did not become healthy after restore")
    code, _ = curl("/start_profile", method="GET", timeout=15)
    if code != 404:
        raise RuntimeError(f"restored boot still mounts /start_profile (GET -> {code})")
    state["restored_env"] = effective_env()
    state["restored_image"] = image_id()
    log("production restored: .env hash matches, /start_profile gone (404)")


def phase_gates(state: dict) -> None:
    acc = subprocess.run(["bash", str(ROOT / "local" / "acceptance.sh")],
                         check=False, capture_output=True, text=True, timeout=3600)
    tail = (acc.stdout or "").strip().splitlines()[-6:]
    state["acceptance_tail"] = tail
    state["acceptance_rc"] = acc.returncode
    log("acceptance tail:\n" + "\n".join(tail))
    state["memfree_head_gib"] = memfree_gib()
    state["memfree_worker_gib"] = memfree_gib(WORKER)
    state["pool_line"] = pool_line()
    state["jit_stamp_after"] = STAMP.read_text().strip() if STAMP.is_file() else ""
    log(f"acceptance rc={acc.returncode} memfree head={state['memfree_head_gib']:.2f} "
        f"worker={state['memfree_worker_gib']:.2f} GiB")
    if acc.returncode != 0:
        raise RuntimeError("acceptance battery failed after restore")
    if state["memfree_head_gib"] < 2.5 or state["memfree_worker_gib"] < 2.5:
        raise RuntimeError("MemFree tripwire: below 2.5 GiB on a node")
    if state["jit_stamp_after"] != state["jit_stamp_before"]:
        raise RuntimeError("JIT shape stamp changed; profiler knob must be hash-neutral")


def phase_rearm(state: dict) -> None:
    for unit in TIMERS:
        proc = run(["systemctl", "--user", "start", unit], timeout=60, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"systemctl start {unit} exited {proc.returncode}: {proc.stderr.strip()}")
    states = timer_states()
    state["timers_after_rearm"] = states
    bad = {unit: status for unit, status in states.items() if status != "active"}
    if bad:
        raise RuntimeError(f"timers not active after rearm: {bad}")
    log(f"watchdog + metrics-alert timers re-armed {states}")


HANDLERS = {
    "preflight": phase_preflight,
    "disarm": phase_disarm,
    "arm": phase_arm,
    "profile": phase_profile,
    "restore": phase_restore,
    "gates": phase_gates,
    "rearm": phase_rearm,
}


def _on_signal(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv: list[str] | None = None) -> int:
    global _ACTIVE, _KEEP_ARMED, _RECEIPT, _RESTORE_DONE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="first", choices=PHASES, default=PHASES[0])
    ap.add_argument("--to", dest="last", choices=PHASES, default=PHASES[-1])
    ap.add_argument("--keep-armed", action="store_true",
                    help="on failure leave the profiler boot running (debug only)")
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--state", type=Path,
                    help="resume/checkpoint file (loaded if present, written at exit)")
    args = ap.parse_args(argv)

    lo, hi = PHASES.index(args.first), PHASES.index(args.last)
    if lo > hi:
        print("--from must not come after --to", file=sys.stderr)
        return 2
    receipt = args.receipt or args.state or (
        ROOT / "local" / f"task29-profile-window-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    state: dict = {}
    if receipt.is_file():
        try:
            state = json.loads(receipt.read_text())
        except json.JSONDecodeError:
            state = {}
    state.update({"schema": 2, "started": state.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    state.setdefault("phases", [])
    if args.first != PHASES[0] and "backup" not in state:
        print(f"--from {args.first} needs a prior preflight state (no backup in {receipt})", file=sys.stderr)
        return 2
    _RECEIPT, _ACTIVE, _KEEP_ARMED = receipt, state, args.keep_armed
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    atexit.register(emergency_restore)
    save(state)

    failure: BaseException | None = None
    for name in PHASES[lo:hi + 1]:
        # Persist the intent before the transition so an interruption between
        # here and the handler still leaves a recoverable receipt on disk.
        state["phase_in_progress"] = name
        save(state)
        log(f"--- phase {name} ---")
        try:
            HANDLERS[name](state)
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
            state["phases"].append({"phase": name, "ok": False, "error": repr(exc)})
            state["phase_in_progress"] = None
            failure = exc
            log(f"phase {name} FAILED: {exc}")
            save(state)
            break
        state["phases"].append({"phase": name, "ok": True})
        state["phase_in_progress"] = None
        save(state)

    if failure is not None and not args.keep_armed:
        if needs_env_restore(state):
            log("failure — restoring the pre-window .env and rebooting production")
            try:
                phase_restore(state)
                state["auto_restore"] = "ok"
            except Exception as exc:  # noqa: BLE001
                state["auto_restore"] = f"FAILED: {exc!r}"
                log(f"AUTO-RESTORE FAILED: {exc} — operator action required")
        else:
            state["auto_restore"] = "not needed (production was never armed)"
        recover_timers(state)

    _RESTORE_DONE = True
    state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save(state)
    log(f"receipt: {receipt}")
    return 1 if failure is not None else 0


if __name__ == "__main__":
    sys.exit(main())
