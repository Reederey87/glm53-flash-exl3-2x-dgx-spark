#!/usr/bin/env python3
"""The read-only contract of `start.sh preflight` (review round five).

`local/prod-start.sh` calls `./start.sh preflight` to decide whether a failed
start is a DETERMINISTIC environment problem (retrying is futile and leaves
partial containers behind) or a transient one. That makes preflight a
classifier, and a classifier that mutates state is not a classifier:

  * it must not write `.env` (the bootstrap would otherwise create it and change
    the answer on the next call);
  * it must not create the HF cache directories, locally or on the worker;
  * a failure anywhere inside it must reach the caller, so a non-zero exit
    cannot be masked by how the dispatcher invokes it.

These tests run the real bootstrap block and the real dispatch arm, lifted
verbatim from start.sh, with no cluster and no docker.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = (ROOT / "start.sh").read_text(encoding="utf-8")

# --- the .env bootstrap, lifted verbatim -------------------------------------

BOOT_BEGIN = "# `validate` and `preflight` are read-only diagnostics"
BOOT_END = "# Caller exports must win over .env"


def bootstrap_block() -> str:
    return START[START.index(BOOT_BEGIN) : START.index(BOOT_END)]


def run_bootstrap(tmp: Path, argv1: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the real bootstrap with `$1=argv1` and no .env present."""
    work = tmp / "kit"
    work.mkdir(exist_ok=True)
    (work / "env.example").write_text("HEAD_IP=192.168.177.10\n")
    script = (
        "set -euo pipefail\n"
        f'SCRIPT_DIR="{work}"\n'
        + bootstrap_block()
        + "\necho BOOTSTRAP-DONE\n"
    )
    r = subprocess.run(
        ["bash", "-c", script, "_", argv1], capture_output=True, text=True
    )
    return r, work / ".env"


def test_preflight_does_not_create_env() -> None:
    """Regression: a missing .env was silently written from env.example by
    every subcommand but `validate`, so a preflight run created configuration
    and the second call answered differently from the first."""
    with tempfile.TemporaryDirectory() as t:
        r, env = run_bootstrap(Path(t), "preflight")
        assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
        assert not env.exists(), "preflight wrote .env"
        assert "does not create configuration files" in r.stderr
        assert "BOOTSTRAP-DONE" not in r.stdout


def test_validate_still_does_not_create_env() -> None:
    with tempfile.TemporaryDirectory() as t:
        r, env = run_bootstrap(Path(t), "validate")
        assert r.returncode == 1
        assert not env.exists()


def test_start_still_bootstraps_env() -> None:
    """The read-only guard must not disable first-run bootstrap for the
    commands that legitimately need it."""
    with tempfile.TemporaryDirectory() as t:
        r, env = run_bootstrap(Path(t), "start")
        assert r.returncode == 0, (r.returncode, r.stderr)
        assert env.exists(), "start no longer bootstraps .env"
        assert env.read_text().strip() == "HEAD_IP=192.168.177.10"


def test_default_command_bootstraps_env() -> None:
    """No argv at all defaults to `start`, which is not read-only."""
    with tempfile.TemporaryDirectory() as t:
        r, env = run_bootstrap(Path(t), "")
        assert env.exists(), "default (start) no longer bootstraps .env"


# --- the dispatch arm, lifted verbatim ---------------------------------------

DISPATCH_BEGIN = "        preflight)\n"
DISPATCH_END = '        status)   status ;;'


def dispatch_arm() -> str:
    block = START[START.index(DISPATCH_BEGIN) : START.index(DISPATCH_END)]
    return block.strip()


def run_arm(fake_preflight: str) -> subprocess.CompletedProcess[str]:
    """Run the real dispatch arm against a stand-in preflight().

    The stand-in fails *implicitly* mid-body (a bare command that returns 23,
    then a succeeding command), which is exactly the shape errexit catches and
    an `&&`/`||` list does not.
    """
    script = (
        "set -euo pipefail\n"
        "log() { printf 'LOG: %s\\n' \"$*\"; }\n"
        f"preflight() {{\n{fake_preflight}\n}}\n"
        'cmd=preflight\n'
        'case "$cmd" in\n'
        + dispatch_arm()
        + "\n        *)\n            echo UNEXPECTED\n            exit 99 ;;\n"
        "    esac\n"
        'echo "REACHED-END"\n'
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_dispatch_propagates_a_failure_from_inside_preflight() -> None:
    """Regression: `preflight && log "preflight OK"` put the call in an AND-list,
    which suppresses errexit for the whole call. preflight then reported success
    even though a check inside it had failed."""
    r = run_arm("return 23\n")
    assert r.returncode == 23, (r.returncode, r.stdout, r.stderr)
    assert "REACHED-END" not in r.stdout, "the dispatcher continued past a failure"
    assert "preflight OK" not in r.stdout, "reported OK on a failed preflight"


def test_dispatch_propagates_an_implicit_mid_body_failure() -> None:
    """The failure is not the function's last command, so only errexit can
    catch it. Under `preflight && log` this returned 0."""
    r = run_arm("false\n:\n")
    assert r.returncode != 0, (r.returncode, r.stdout, r.stderr)
    assert "preflight OK" not in r.stdout


def test_dispatch_reports_ok_only_on_success() -> None:
    r = run_arm(":\n")
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert "preflight OK" in r.stdout
    assert "REACHED-END" in r.stdout


def test_dispatch_does_not_use_an_and_or_list_for_preflight() -> None:
    """Source-shape pin so the errexit suppression cannot come back."""
    arm = dispatch_arm()
    assert "preflight && " not in arm
    assert "preflight || " not in arm
    assert any(line.strip() == "preflight" for line in arm.splitlines()), arm


def test_dispatch_enables_read_only_before_calling_preflight() -> None:
    arm = dispatch_arm()
    assert "READ_ONLY=1" in arm
    assert arm.index("READ_ONLY=1") < arm.index("\n            preflight\n")


# --- the cache-preparation guards -------------------------------------------

def test_preflight_guards_cache_creation_behind_read_only() -> None:
    """`mkdir -p "$HF_CACHE_DIR"` and the worker `mkdir -p .../hub` are
    preparation, not checking. Both must be conditional on READ_ONLY."""
    begin = START.index("# Creating the cache directories is preparation")
    end = START.index('log "preflight OK (head=')
    block = START[begin:end]
    assert '[ "$READ_ONLY" = 1 ]' in block
    # The head mkdir is inside the else branch of the READ_ONLY test.
    assert 'mkdir -p "$HF_CACHE_DIR"' in block
    assert block.index('[ "$READ_ONLY" = 1 ]') < block.index('mkdir -p "$HF_CACHE_DIR"')
    # The worker: test-only under READ_ONLY, mkdir otherwise.
    assert 'worker_ssh "test -w' in block
    assert "elif ! worker_ssh \"mkdir -p" in block


def test_preflight_reads_the_gid_tables_fail_closed() -> None:
    """Review round five: a command substitution that prints a usable row and
    THEN returns non-zero was swallowed, so an incomplete check read as a pass.
    Both table reads carry an explicit `|| die`."""
    begin = START.index("head_table=\"$(gid_table")
    end = START.index("worker_table=\"$(worker_gid_table")
    head_read = START[begin:end]
    assert "|| die" in head_read, head_read
    # The read and its `|| die` span a line continuation, so take the whole
    # statement rather than the first line.
    worker_read = START[end : end + 200]
    assert "|| die" in worker_read, worker_read


if __name__ == "__main__":
    test_preflight_does_not_create_env()
    test_validate_still_does_not_create_env()
    test_start_still_bootstraps_env()
    test_default_command_bootstraps_env()
    test_dispatch_propagates_a_failure_from_inside_preflight()
    test_dispatch_propagates_an_implicit_mid_body_failure()
    test_dispatch_reports_ok_only_on_success()
    test_dispatch_does_not_use_an_and_or_list_for_preflight()
    test_dispatch_enables_read_only_before_calling_preflight()
    test_preflight_guards_cache_creation_behind_read_only()
    test_preflight_reads_the_gid_tables_fail_closed()
    print("start.sh preflight read-only tests OK")
