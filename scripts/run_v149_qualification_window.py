#!/usr/bin/env python3
"""Guarded A-B-B-A §6 qualification window for the task 35 ExLlamaV3 v1.4.9 pin.

Task 35 moved the ExLlamaV3 pin from v1.4.7 (`ca13bdd`) to v1.4.9 (`5be8865`)
and deployed the candidate on the correctness gates, but the end-to-end
throughput comparison was never run to `docs/13` §6's observation contract: the
original window ran one three-observation structured block per arm. This runner
produces that missing evidence.

One independent variable: the `IMAGE=` tag, i.e. the ExLlamaV3 revision baked
into the serving image. Every other knob is frozen by the pre-window `.env`.

  A  = glm53-selfbuild:e3-w3-zfill       (ExLlamaV3 1.4.7, control)
  B  = glm53-selfbuild:e3-w3-zfill-v149  (ExLlamaV3 1.4.9, candidate)

Sequence and sample sizes are pre-registered in `audit_v149_qualification.py`
and are not tunable from the command line: A-B-B-A with 9 decode observations
per arm on each of three fixed prompts and 5 cold-prefill observations per arm
at ~60k and ~240k. Each arm gets its own boot, because the revision is baked
into the process.

Phases run in order and can be bounded with `--from/--to`:

  preflight   hashes, drain check, both-node arm-image presence, `.env` backup,
              KV pool line, JIT shape stamp (shared helpers from the decode
              profile window runner)
  disarm      stop the watchdog + metrics-alert timers, reset-failed
  arm_*       append the arm's `IMAGE=` last-wins line, start through
              `local/prod-start.sh`, wait for health, verify the container's
              image tag and the in-container `exllamav3` version on BOTH nodes
  measure_*   run the pre-registered probe blocks, sample MemFree and the
              preemption counter around them
  restore     put the pre-window `.env` back (production is the v1.4.9
              candidate), reboot, verify the restore
  gates       acceptance.sh, both-node MemFree tripwire, KV pool line, JIT stamp
  rearm       re-enable the timers
  judge       run `audit_v149_qualification.py` over the receipt

Any failure after disarm reboots production from the pre-window `.env` and
re-arms the timers before exiting non-zero; the two are attempted independently
so a failed reboot cannot skip the re-arm. SIGTERM/SIGHUP/SIGINT take the same
recovery path. The window never edits any `.env` line other than appending
`IMAGE=`. Resume a partial window with `--state <receipt> --from <phase>`.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_decode_profile_window as win  # noqa: E402  (shared guarded-window helpers)

ROOT = win.ROOT
ENV_FILE = ROOT / ".env"
PROBE = ROOT / "scripts" / "probe_v149_qualification.py"
AUDITOR = ROOT / "scripts" / "audit_v149_qualification.py"
HEAD_CONTAINER = "glm53-exl3-head"
WORKER_CONTAINER = "glm53-exl3-worker"
WORKER = os.environ.get("WORKER_SSH", "nvidia@192.168.177.11")
TAG = "task35b"
TIMERS = ("vllm-glm53exl3-watchdog.timer", "glm53exl3-metrics-alert.timer")
# The services those timers trigger. Stopping a timer does not stop a service it
# already started, so disarm must wait for these to go inactive too.
TIMER_SERVICES = ("vllm-glm53exl3-watchdog.service", "glm53exl3-metrics-alert.service")

# Pre-registered arms. `tag` is the last-wins IMAGE= value; `exllamav3` is what
# the running container must report, checked in-container on both nodes.
ARMS: dict[str, dict[str, str]] = {
    "a": {"tag": "glm53-selfbuild:e3-w3-zfill", "exllamav3": "1.4.7"},
    "b": {"tag": "glm53-selfbuild:e3-w3-zfill-v149", "exllamav3": "1.4.9"},
    "b2": {"tag": "glm53-selfbuild:e3-w3-zfill-v149", "exllamav3": "1.4.9"},
    "a2": {"tag": "glm53-selfbuild:e3-w3-zfill", "exllamav3": "1.4.7"},
}
# lane -> (runs, subprocess timeout seconds). The counts are the §6 contract.
LANES: dict[str, tuple[int, float]] = {
    "structured": (9, 1800.0),
    "essay": (9, 1800.0),
    "hashmap": (9, 1800.0),
    "prefill60k": (5, 3600.0),
    "prefill240k": (5, 5400.0),
}
WARMUP_TOKENS = 32
TRIPWIRE_GIB = 2.5
PHASES = (
    "preflight",
    "disarm",
    "arm_a",
    "measure_a",
    "arm_b",
    "measure_b",
    "arm_b2",
    "measure_b2",
    "arm_a2",
    "measure_a2",
    "restore",
    "gates",
    "rearm",
    "judge",
)
PRODUCTION_IMAGE = ARMS["b"]["tag"]

_RECEIPT: Path | None = None
_ACTIVE: dict | None = None
_KEEP_ARMED = False
_RESTORE_DONE = False


def log(message: str) -> None:
    print(f"[task35b-window] {message}", flush=True)


def save(state: dict) -> None:
    """Atomically persist the receipt so an interrupted window stays recoverable."""
    if _RECEIPT is None:
        return
    tmp = _RECEIPT.with_suffix(_RECEIPT.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, default=str) + "\n")
    os.replace(tmp, _RECEIPT)


# --- environment ------------------------------------------------------------

def effective_env_all() -> dict[str, str]:
    """Last-wins value for every `KEY=value` line in `.env`."""
    out: dict[str, str] = {}
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        out[key] = value.strip().strip("'\"")
    return out


def set_image(tag: str) -> None:
    text = ENV_FILE.read_text()
    if not text.endswith("\n"):
        text += "\n"
    ENV_FILE.write_text(text + f"\n# {TAG} qualification window: arm image (removed by restore)\nIMAGE={tag}\n")


def jit_stamp() -> str:
    return win.STAMP.read_text().strip() if win.STAMP.is_file() else ""


POOL_CAPACITY_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
POOL_CONCURRENCY_RE = re.compile(r"Maximum concurrency[^:]*:\s*([\d.]+)x")


def pool_line_raw() -> str:
    """The raw KV-capacity log line, for the audit trail.

    Deliberately more specific than the shared `win.pool_line()`, which returns
    the first line containing `kv_cache`. In this container the *first* such line
    is a startup patch message carrying the file path `kv_cache_utils.py`
    (`[patch_glm5_drafter_group] ... kv_cache_utils.py: already patched`), which
    is identical on every arm, so comparing it would make the pool gate vacuous.

    `grep -m1` exits at the first match, so the pipeline stops early rather than
    streaming the whole log.
    """
    for pattern in ("GPU KV cache size:", "glm53-kv-capacity-log"):
        proc = win.run(
            ["sh", "-c", f"docker logs {HEAD_CONTAINER} 2>&1 | grep -m1 {pattern!r}"],
            timeout=300, check=False,
        )
        line = proc.stdout.strip()
        if line:
            return line
    return ""


def pool_capacity() -> str:
    """The KV pool's identity: token count and concurrency, NOT the log line.

    The raw line carries a timestamp, a PID and a source-location prefix
    (`(EngineCore pid=237) INFO 09-11 09:21:22 [kv_cache_utils.py:2598] ...`)
    that differ on every boot even when the pool is byte-identical, so comparing
    raw lines would abort a healthy window. Compare the parsed capacity instead.
    Returns "" when no capacity line is found, which the auditor rejects.
    """
    raw = pool_line_raw()
    if not raw:
        return ""
    match = POOL_CAPACITY_RE.search(raw)
    if not match:
        return ""
    tokens = match.group(1).replace(",", "")
    concurrency = POOL_CONCURRENCY_RE.search(raw)
    suffix = f"; concurrency {concurrency.group(1)}x" if concurrency else ""
    return f"{tokens} tokens{suffix}"


# --- container identity -----------------------------------------------------

def container_image(container: str, host: str | None = None) -> str:
    argv = ["docker", "inspect", "-f", "{{.Config.Image}}", container]
    if host:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, " ".join(argv)]
    proc = win.run(argv, timeout=60, check=False)
    return proc.stdout.strip()


def exllamav3_version(container: str = HEAD_CONTAINER, host: str | None = None) -> str:
    """Read the installed distribution version; `exllamav3.__version__` is unset."""
    argv = ["docker", "exec", container, "python3", "-m", "pip", "show", "exllamav3"]
    if host:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, " ".join(argv)]
    proc = win.run(argv, timeout=180, check=False)
    for line in (proc.stdout + proc.stderr).splitlines():
        match = re.match(r"^Version:\s*(\S+)", line.strip())
        if match:
            return match.group(1)
    return ""


def container_started_at(container: str = HEAD_CONTAINER, host: str | None = None) -> str:
    """The container's start time, used as a per-boot identity token."""
    argv = ["docker", "inspect", "-f", "{{.State.StartedAt}}", container]
    if host:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, " ".join(argv)]
    return win.run(argv, timeout=60, check=False).stdout.strip()


def preemptions() -> float:
    proc = win.run(["curl", "-fsS", f"{win.BASE}/metrics"], timeout=30, check=False)
    total = 0.0
    for line in proc.stdout.splitlines():
        if not line.startswith("vllm:num_preemptions_total"):
            continue
        try:
            total += float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return total


def memfree_pair() -> tuple[float, float]:
    return win.memfree_gib(), win.memfree_gib(WORKER)


def check_tripwire(where: str) -> tuple[float, float]:
    head, worker = memfree_pair()
    if head < TRIPWIRE_GIB or worker < TRIPWIRE_GIB:
        raise RuntimeError(f"{where}: MemFree tripwire head={head:.2f} worker={worker:.2f} GiB")
    return head, worker


def verify_arm(arm: str, container: str, host: str | None = None) -> dict:
    """Confirm the boot is the arm it claims to be, on the node it runs on."""
    expected = ARMS[arm]
    seen_image = container_image(container, host)
    record = {"container": container, "image_tag": seen_image, "expected_tag": expected["tag"]}
    if seen_image != expected["tag"]:
        raise RuntimeError(f"arm {arm}: {container} runs {seen_image!r}, expected {expected['tag']!r}")
    version = exllamav3_version(container, host)
    record["exllamav3_version"] = version
    if version != expected["exllamav3"]:
        raise RuntimeError(
            f"arm {arm}: in-container exllamav3 is {version!r}, expected {expected['exllamav3']!r}"
        )
    return record


# --- phases -----------------------------------------------------------------

def phase_preflight(state: dict) -> None:
    if not ENV_FILE.is_file():
        raise RuntimeError(f"missing {ENV_FILE}")
    for path in (PROBE, AUDITOR):
        if not path.is_file():
            raise RuntimeError(f"missing {path}")
    stamp_before = jit_stamp()
    backup = ROOT / f".env.bak-pre-{TAG}-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(ENV_FILE, backup)
    state.update({"backup": str(backup), "env_sha256": win.sha256(ENV_FILE)})
    save(state)
    env = effective_env_all()
    if env.get("IMAGE") != PRODUCTION_IMAGE:
        raise RuntimeError(
            f"preflight expects production on {PRODUCTION_IMAGE!r}, .env says {env.get('IMAGE')!r}"
        )
    code, _ = win.curl("/health", timeout=10)
    if code != 200:
        raise RuntimeError(f"production is not healthy before the window (/health -> {code})")
    if not win.drain():
        raise RuntimeError("server did not drain; refusing to take it down")
    # Both arm images must already exist on BOTH nodes: shipping one mid-window
    # would add a variable (and a multi-gigabyte transfer) to the measurement.
    presence: dict[str, dict[str, bool]] = {}
    for arm, spec in ARMS.items():
        tag = spec["tag"]
        head_ok = win.run(["docker", "image", "inspect", tag], timeout=60, check=False).returncode == 0
        worker_ok = win.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", WORKER,
             f"docker image inspect {tag}"], timeout=90, check=False,
        ).returncode == 0
        presence[arm] = {"tag": tag, "head": head_ok, "worker": worker_ok}
        if not (head_ok and worker_ok):
            raise RuntimeError(f"arm {arm} image {tag!r} missing (head={head_ok} worker={worker_ok})")
    state.update(
        {
            "window": TAG,
            "runner_sha256": win.sha256(Path(__file__)),
            "probe_sha256": win.sha256(PROBE),
            "auditor_sha256": win.sha256(AUDITOR),
            "prod_start_sha256": win.sha256(ROOT / "local" / "prod-start.sh"),
            "start_sha256": win.sha256(ROOT / "start.sh"),
            "env_effective_before": env,
            "image_before": win.image_id(),
            "pool_line_before": pool_line_raw(),
            "pool_capacity_before": pool_capacity(),
            "jit_stamp_before": stamp_before,
            "arm_image_presence": presence,
            "contract": {"arms": ARMS, "lanes": {k: v[0] for k, v in LANES.items()}},
        }
    )
    log(f"preflight OK backup={backup.name} stamp={stamp_before[:12]} pool={state['pool_line_before'][:60]}")


def timer_states() -> dict[str, str]:
    out: dict[str, str] = {}
    for unit in TIMERS:
        proc = win.run(["systemctl", "--user", "is-active", unit], timeout=30, check=False)
        out[unit] = proc.stdout.strip() or "unknown"
    return out


def phase_disarm(state: dict) -> None:
    state["disarm_attempted"] = True
    save(state)
    for unit in TIMERS:
        proc = win.run(["systemctl", "--user", "stop", unit], timeout=60, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"systemctl stop {unit} exited {proc.returncode}: {proc.stderr.strip()}")
    win.run(["systemctl", "--user", "reset-failed"], timeout=60, check=False)
    states = timer_states()
    state["timers_after_disarm"] = states
    bad = {unit: status for unit, status in states.items() if status == "active"}
    if bad:
        raise RuntimeError(f"timers still active after disarm: {bad}")
    # Stopping a timer does not stop a service it already started, nor cancel a
    # queued restart job. §6 requires waiting for in-flight watchdog/start work:
    # the watchdog can enqueue `systemctl restart --no-block`, and that job
    # would tear down the boot this runner is about to perform.
    quiescent = wait_quiescent()
    state["quiescent_after_disarm"] = quiescent
    save(state)
    if not quiescent:
        raise RuntimeError(
            "watchdog/metrics services or queued jobs still active after disarm; "
            "refusing to start the window"
        )
    log(f"watchdog + metrics-alert timers disarmed and quiescent {states}")


def _active_services() -> dict[str, str]:
    out: dict[str, str] = {}
    for unit in TIMER_SERVICES:
        proc = win.run(["systemctl", "--user", "is-active", unit], timeout=30, check=False)
        out[unit] = proc.stdout.strip() or "unknown"
    return out


def _pending_jobs() -> int:
    proc = win.run(["systemctl", "--user", "list-jobs"], timeout=30, check=False)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    # The header line ("No jobs running." or a column header) is not a job.
    return len([ln for ln in lines if "No jobs running" not in ln and not ln.startswith("JOB")])


def wait_quiescent(timeout: float = 300.0) -> bool:
    """Wait until the timer services are inactive and no jobs are queued."""
    deadline = time.monotonic() + timeout
    while True:
        busy = {
            unit: status
            for unit, status in _active_services().items()
            if status in ("active", "activating", "reloading", "deactivating")
        }
        jobs = _pending_jobs()
        if not busy and jobs == 0:
            return True
        if time.monotonic() >= deadline:
            log(f"quiescence timed out: services={busy} jobs={jobs}")
            return False
        log(f"waiting for quiescence: services={busy} jobs={jobs}")
        time.sleep(5)


def phase_arm(state: dict, arm: str) -> None:
    state.setdefault("arms", {})
    state["arms"].setdefault(arm, {})["arm_attempted"] = True
    state["env_touched"] = True
    save(state)
    log(f"arming {arm}: IMAGE={ARMS[arm]['tag']}")
    set_image(ARMS[arm]["tag"])
    win.guarded_start()
    if not win.wait_health():
        raise RuntimeError(f"arm {arm}: head did not become healthy")
    head = verify_arm(arm, HEAD_CONTAINER)
    worker = verify_arm(arm, WORKER_CONTAINER, host=WORKER)
    head_mem, worker_mem = check_tripwire(f"arm {arm} boot")
    # One predeclared warmup pass before the measured block.
    warm = win.run(
        ["python3", str(PROBE), "--kind", "structured", "--runs", "1",
         "--max-tokens", str(WARMUP_TOKENS), "--out", "/dev/null"],
        timeout=900, check=False,
    )
    if warm.returncode != 0:
        raise RuntimeError(f"arm {arm}: warmup probe failed rc={warm.returncode}: {warm.stderr[-400:]}")
    record = {
        **head,
        "worker_image_tag": worker["image_tag"],
        "worker_exllamav3_version": worker["exllamav3_version"],
        "health": 200,
        "jit_stamp": jit_stamp(),
        "pool_line": pool_line_raw(),
        "container_started_at": container_started_at(),
        "memfree_head_gib": head_mem,
        "memfree_worker_gib": worker_mem,
        "preemptions_before": preemptions(),
        "warmup_rc": warm.returncode,
    }
    state["arms"][arm].update(record)
    save(state)
    log(f"arm {arm} up: image={record['image_tag']} exllamav3={record['exllamav3_version']} "
        f"worker={record['worker_image_tag']} stamp={record['jit_stamp'][:12]} "
        f"MemFree head={head_mem:.2f} worker={worker_mem:.2f}")


def phase_measure(state: dict, arm: str) -> None:
    # Revalidate the running arm before measuring. A resume with `--from
    # measure_a` after an automatic recovery would otherwise measure whatever is
    # actually running (the restored production image) and file it under arm A.
    # `verify_arm` raises on any image or `exllamav3` mismatch.
    head_now = verify_arm(arm, HEAD_CONTAINER)
    worker_now = verify_arm(arm, WORKER_CONTAINER, host=WORKER)
    record = state.setdefault("arms", {}).setdefault(arm, {})
    boot_now = container_started_at()
    recorded_boot = record.get("container_started_at")
    if recorded_boot and boot_now and boot_now != recorded_boot:
        raise RuntimeError(
            f"arm {arm}: the head container restarted since the arm phase "
            f"({recorded_boot} -> {boot_now}); the observations would not belong "
            "to the verified boot"
        )
    record["measure_verified_image"] = head_now["image_tag"]
    record["measure_verified_worker_image"] = worker_now["image_tag"]
    record["measure_verified_exllamav3"] = head_now["exllamav3_version"]
    record["measure_container_started_at"] = boot_now
    # A fresh attempt tag per invocation: a retry after a failed block writes new
    # files instead of overwriting the evidence an earlier receipt points at.
    attempt = time.strftime("%Y%m%d-%H%M%S")
    record["measure_attempt"] = attempt
    save(state)

    state.setdefault("probes", {}).setdefault(arm, {})
    for lane, (runs, timeout) in LANES.items():
        # Window-, arm- and attempt-specific path, so no run of this harness can
        # ever clobber another's evidence.
        out = _RECEIPT.parent / (
            f"{_RECEIPT.stem}-{arm}-{lane}-{attempt}.json"
        )
        head_before, worker_before = check_tripwire(f"arm {arm} {lane} before")
        started = time.time()
        proc = subprocess.run(
            ["python3", str(PROBE), "--kind", lane, "--runs", str(runs), "--out", str(out)],
            check=False, text=True, timeout=timeout,
        )
        elapsed = time.time() - started
        head_after, worker_after = check_tripwire(f"arm {arm} {lane} after")
        if proc.returncode != 0:
            raise RuntimeError(
                f"arm {arm} lane {lane}: probe exited {proc.returncode}; "
                f"last output: {(proc.stdout or proc.stderr)[-400:]}"
            )
        state["probes"][arm][lane] = out.name
        state.setdefault("probe_blocks", []).append(
            {
                "arm": arm, "lane": lane, "runs": runs, "seconds": round(elapsed, 1),
                "memfree_head_gib": [head_before, head_after],
                "memfree_worker_gib": [worker_before, worker_after],
            }
        )
        save(state)
        log(f"arm {arm} lane {lane}: {runs} runs in {elapsed:.0f}s")
    state["arms"].setdefault(arm, {})["preemptions_after"] = preemptions()
    delta = state["arms"][arm]["preemptions_after"] - state["arms"][arm].get("preemptions_before", 0.0)
    state["arms"][arm]["preemptions_delta"] = delta
    save(state)
    if delta:
        raise RuntimeError(f"arm {arm}: preemptions increased by {delta} during measurement")


def phase_restore(state: dict) -> None:
    backup = Path(state["backup"])
    shutil.copy2(backup, ENV_FILE)
    if win.sha256(ENV_FILE) != state["env_sha256"]:
        raise RuntimeError("restored .env does not match the pre-window hash")
    save(state)
    win.guarded_start()
    if not win.wait_health():
        raise RuntimeError("production did not become healthy after restore")
    record = verify_arm("b", HEAD_CONTAINER)
    worker_record = verify_arm("b", WORKER_CONTAINER, host=WORKER)
    state["restored_image_tag"] = record["image_tag"]
    state["restored_exllamav3_version"] = record["exllamav3_version"]
    # Recovery intent is cleared ONLY once production is verified healthy and
    # running the expected image on both nodes. Clearing it earlier (right after
    # copying the backup) would make `needs_env_restore` report "production was
    # never moved" if `guarded_start` then failed, leaving production down with
    # no automatic retry.
    state["env_touched"] = False
    state["restored_worker_image_tag"] = worker_record["image_tag"]
    save(state)
    log(f"production restored: {record['image_tag']} exllamav3={record['exllamav3_version']} "
        f"worker={worker_record['image_tag']}")


def phase_gates(state: dict) -> None:
    acc = subprocess.run(
        ["bash", str(ROOT / "local" / "acceptance.sh")],
        check=False, capture_output=True, text=True, timeout=3600,
    )
    head, worker = check_tripwire("post-restore")
    state["gates"] = {
        "acceptance_rc": acc.returncode,
        "acceptance_tail": (acc.stdout or "").strip().splitlines()[-6:],
        "memfree_head_gib": head,
        "memfree_worker_gib": worker,
        "pool_line": pool_line_raw(),
        "pool_capacity": pool_capacity(),
        "pool_capacity_before": state.get("pool_capacity_before", ""),
        "jit_stamp": jit_stamp(),
        # The pre-window stamp, NOT arm B's: `prod-start.sh` hashes every raw
        # `IMAGE=` line including overridden ones, so arm B's `.env` (original
        # line + appended A + appended B) and the restored `.env` (original line
        # only) hash differently even though both run B. Comparing against arm
        # B's stamp would deterministically abort a successful window.
        "jit_stamp_before": state.get("jit_stamp_before", ""),
        "jit_stamp_arm_b": (state.get("arms", {}).get("b", {}) or {}).get("jit_stamp", ""),
    }
    save(state)
    log(f"acceptance rc={acc.returncode} MemFree head={head:.2f} worker={worker:.2f} GiB")
    if acc.returncode != 0:
        raise RuntimeError("acceptance battery failed after restore")


def phase_rearm(state: dict) -> None:
    for unit in TIMERS:
        proc = win.run(["systemctl", "--user", "start", unit], timeout=60, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"systemctl start {unit} exited {proc.returncode}: {proc.stderr.strip()}")
    states = timer_states()
    state["timers_after_rearm"] = states
    bad = {unit: status for unit, status in states.items() if status != "active"}
    if bad:
        raise RuntimeError(f"timers not active after rearm: {bad}")
    log(f"watchdog + metrics-alert timers re-armed {states}")


def phase_judge(state: dict) -> None:
    audit_out = _RECEIPT.with_name(_RECEIPT.stem.replace("-window-", "-audit-") + ".json")
    proc = subprocess.run(
        ["python3", str(AUDITOR), "--receipt", str(_RECEIPT), "--out", str(audit_out)],
        check=False, capture_output=True, text=True, timeout=900,
    )
    state["audit_receipt"] = audit_out.name
    state["audit_tail"] = (proc.stdout or "").strip().splitlines()[-20:]
    try:
        state["verdict"] = json.loads(audit_out.read_text()).get("verdict")
    except (OSError, json.JSONDecodeError):
        state["verdict"] = None
    save(state)
    log(f"judge rc={proc.returncode} verdict={state['verdict']} receipt={audit_out.name}")
    if state["verdict"] is None:
        raise RuntimeError(f"auditor produced no verdict: {(proc.stdout or proc.stderr)[-400:]}")


HANDLERS = {
    "preflight": phase_preflight,
    "disarm": phase_disarm,
    "arm_a": lambda state: phase_arm(state, "a"),
    "measure_a": lambda state: phase_measure(state, "a"),
    "arm_b": lambda state: phase_arm(state, "b"),
    "measure_b": lambda state: phase_measure(state, "b"),
    "arm_b2": lambda state: phase_arm(state, "b2"),
    "measure_b2": lambda state: phase_measure(state, "b2"),
    "arm_a2": lambda state: phase_arm(state, "a2"),
    "measure_a2": lambda state: phase_measure(state, "a2"),
    "restore": phase_restore,
    "gates": phase_gates,
    "rearm": phase_rearm,
    "judge": phase_judge,
}


# --- recovery ---------------------------------------------------------------

def needs_env_restore(state: dict) -> bool:
    if "backup" not in state:
        return False
    if state.get("env_touched"):
        return True
    env = effective_env_all() if ENV_FILE.is_file() else {}
    return env.get("IMAGE") != PRODUCTION_IMAGE


def needs_timer_restore(state: dict) -> bool:
    return bool(state.get("disarm_attempted"))


def recover_timers(state: dict) -> None:
    if not needs_timer_restore(state):
        return
    try:
        phase_rearm(state)
        state["timer_restore"] = "ok"
    except Exception as exc:  # noqa: BLE001
        state["timer_restore"] = f"FAILED: {exc!r}"
        log(f"TIMER RESTORE FAILED: {exc} — operator action required")
    save(state)


def restore_production(state: dict) -> None:
    """Put the pre-window `.env` back and reboot, independently of the timers."""
    if not needs_env_restore(state):
        state["auto_restore"] = "not needed (production was never moved)"
        return
    log("restoring the pre-window .env and rebooting production")
    try:
        phase_restore(state)
        state["auto_restore"] = "ok"
    except Exception as exc:  # noqa: BLE001
        state["auto_restore"] = f"FAILED: {exc!r}"
        log(f"AUTO-RESTORE FAILED: {exc} — operator action required")


def _recovery_ok(state: dict) -> bool:
    """True when recovery left nothing outstanding, so atexit need not retry."""
    return not str(state.get("auto_restore", "")).startswith("FAILED") and not str(
        state.get("timer_restore", "")
    ).startswith("FAILED")


def emergency_restore() -> None:
    global _RESTORE_DONE
    state = _ACTIVE
    if _RESTORE_DONE or _KEEP_ARMED or not state:
        return
    if not (needs_env_restore(state) or needs_timer_restore(state)):
        return
    _RESTORE_DONE = True
    restore_production(state)
    recover_timers(state)
    save(state)


def _on_signal(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv: list[str] | None = None) -> int:
    global _ACTIVE, _KEEP_ARMED, _RECEIPT, _RESTORE_DONE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="first", choices=PHASES, default=PHASES[0])
    ap.add_argument("--to", dest="last", choices=PHASES, default=PHASES[-1])
    ap.add_argument("--state", type=Path, help="receipt/checkpoint file")
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--keep-armed", action="store_true",
                    help="on failure leave the arm boot running (debug only)")
    args = ap.parse_args(argv)
    lo, hi = PHASES.index(args.first), PHASES.index(args.last)
    if lo > hi:
        print("--from must not come after --to", file=sys.stderr)
        return 2
    receipt = args.receipt or args.state or (
        ROOT / "local" / f"{TAG}-window-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    state: dict = {}
    if receipt.is_file():
        try:
            state = json.loads(receipt.read_text())
        except json.JSONDecodeError:
            state = {}
    state.update({"schema": 1, "window": TAG, "started": state.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S%z")})
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
    try:
        for name in PHASES[lo:hi + 1]:
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
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        # Anything raised outside a phase handler — a `save()` failure, or a
        # SIGTERM landing between phases — must still trigger recovery. Without
        # this the `finally` would see failure=None, skip the restore, and set
        # _RESTORE_DONE, which disables the atexit handler as well.
        failure = failure or exc
        log(f"window aborted outside a phase handler: {exc!r}")
    finally:
        if failure is not None and not args.keep_armed:
            restore_production(state)
            recover_timers(state)
            # Only disarm the atexit safety net once recovery actually
            # succeeded, so a failed restore gets a second attempt at exit.
            _RESTORE_DONE = _recovery_ok(state)
        else:
            _RESTORE_DONE = True
        state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            save(state)
        except OSError as exc:  # the receipt write must not mask the real failure
            log(f"could not write the final receipt: {exc!r}")

    log(f"receipt: {receipt}")
    return 1 if failure is not None else 0


if __name__ == "__main__":
    sys.exit(main())
