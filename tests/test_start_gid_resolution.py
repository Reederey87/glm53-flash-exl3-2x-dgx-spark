#!/usr/bin/env python3
"""Identity-based RoCE v2 GID resolution in start.sh, run standalone.

A GID index is a runtime table SLOT, not a stable property of an IP address: it
has drifted across reboots on this kit (the worker's RoCE v2 entry moved 4 -> 3)
and each drift took production down until `.env` was hand-edited. These tests pin
the resolver that replaced the hardcoded index, using synthetic GID tables so no
cluster is needed.

The tables below are the real ones captured during the 2026-09-11 incident.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = (ROOT / "start.sh").read_text(encoding="utf-8")

BEGIN = "# ------------------------- RoCE v2 GID resolution"
END = "# ------------------------------ preflight"

HEAD_IP = "192.168.177.10"
WORKER_IP = "192.168.177.11"

# \t/\n are left literal so they land inside a bash $'...' literal.
REAL_HEAD_ROWS = [
    r"2\t0000:0000:0000:0000:0000:ffff:c0a8:b10a\tIB/RoCE v1",
    r"3\t0000:0000:0000:0000:0000:ffff:c0a8:b10a\tRoCE v2",
]
REAL_WORKER_ROWS = [
    r"0\tfe80:0000:0000:0000:4ebb:47ff:fee8:782b\tIB/RoCE v1",
    r"1\tfe80:0000:0000:0000:4ebb:47ff:fee8:782b\tRoCE v2",
    r"2\t0000:0000:0000:0000:0000:ffff:c0a8:b10b\tIB/RoCE v1",
    r"3\t0000:0000:0000:0000:0000:ffff:c0a8:b10b\tRoCE v2",
]


def table(rows: list[str]) -> str:
    return "$'" + r"\n".join(rows) + "'"


def gid_helpers() -> str:
    """The resolver block, lifted verbatim from start.sh."""
    begin = START.index(BEGIN)
    end = START.index(END)
    return START[begin:end]


def run_bash(body: str) -> subprocess.CompletedProcess[str]:
    script = (
        "set -u\n"
        "warn() { printf 'WARN: %s\\n' \"$*\" >&2; }\n"
        "die() { printf 'DIE: %s\\n' \"$*\" >&2; return 1; }\n"
        "worker_ssh() { return 1; }\n"
        f"HEAD_IP={HEAD_IP}\n"
        f"WORKER_IP={WORKER_IP}\n"
        "HEAD_CX7_IB=rocep1s0f1\n"
        "WORKER_CX7_IB=rocep1s0f1\n"
        f"T_HEAD={table(REAL_HEAD_ROWS)}\n"
        f"T_WORKER={table(REAL_WORKER_ROWS)}\n"
        + gid_helpers()
        + "\n"
        + body
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def resolve(tbl_var: str, ip: str) -> str:
    r = run_bash(f'{tbl_var}=${tbl_var}\ngid_index_for_ip "${tbl_var}" {ip} && echo "|OK" || echo "|FAIL"')
    out = r.stdout.strip()
    idx, _, status = out.rpartition("|")
    return f"{idx}:{status}"


# --- address -> kernel GID form ---------------------------------------------

def test_ipv4_mapped_gid_matches_the_kernel_form():
    r = run_bash(
        f'ipv4_mapped_gid {HEAD_IP}; echo; ipv4_mapped_gid {WORKER_IP}; echo; '
        "ipv4_mapped_gid 10.0.0.214; echo"
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [
        "0000:0000:0000:0000:0000:ffff:c0a8:b10a",
        "0000:0000:0000:0000:0000:ffff:c0a8:b10b",
        "0000:0000:0000:0000:0000:ffff:0a00:00d6",
    ]


def test_leading_zero_octet_is_read_as_decimal_not_octal():
    """`printf '%02x' 010` would yield 08 (octal). 10# must force decimal."""
    r = run_bash("ipv4_mapped_gid 010.0.0.1; echo")
    assert r.stdout.strip() == "0000:0000:0000:0000:0000:ffff:0a00:0001"


def test_malformed_addresses_are_rejected():
    for bad in ("not.an.ip", "10.0.0", "10.0.0.999", "", "10.0.0.-1", "a.b.c.d"):
        r = run_bash(f'ipv4_mapped_gid "{bad}" >/dev/null 2>&1 && echo ACCEPTED || echo REJECTED')
        assert "REJECTED" in r.stdout, bad


def test_malformed_addresses_that_a_loose_parser_would_accept():
    """Review round five. Each of these survived the original `read -a` +
    per-octet loop and resolved to a PLAUSIBLE gid, so a bad HEAD_IP/WORKER_IP
    in .env would have silently matched the wrong fabric address instead of
    failing closed."""
    cases = [
        # Trailing dot: `read -a` yields four fields and drops the empty fifth,
        # so the count check passed.
        "192.168.177.11.",
        # 20-digit octet: `$((10#$o))` overflows 64-bit and WRAPS to a small
        # value, so the <=255 check passed. 18446744073709551627 -> 11, i.e.
        # the real worker address.
        "192.168.177.18446744073709551627",
        # Embedded newline: `read` consumes only the first line and discards
        # the rest, so this passed as 192.168.177.11.
        "192.168.177.11\nignored",
        # Five groups, leading/trailing space, sign prefixes, non-digits.
        "192.168.177.11.12",
        " 192.168.177.11",
        "192.168.177.11 ",
        "192.168.177.+11",
        "192.168.177.1a",
        "192.168.177.",
        "192.168.177.11;id",
        "192.168.177.11/24",
    ]
    for bad in cases:
        lit = "$'" + bad.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"
        r = run_bash(
            f"V={lit}\n"
            'ipv4_mapped_gid "$V" >/dev/null 2>&1 && echo ACCEPTED || echo REJECTED'
        )
        assert "REJECTED" in r.stdout, f"accepted {bad!r}: {r.stdout!r}"
        assert "ACCEPTED" not in r.stdout, f"accepted {bad!r}"


def test_the_overflowing_octet_is_not_silently_wrapped():
    """Pin the exact wrap the loose parser produced, so the digit-count bound
    cannot be relaxed back to a value-only check."""
    r = run_bash(
        "V='192.168.177.18446744073709551627'\n"
        'ipv4_mapped_gid "$V" 2>/dev/null; echo "|rc=$?"'
    )
    assert "c0a8:b10b" not in r.stdout, "wrapped to the real worker address"
    assert "|rc=1" in r.stdout


# --- index selection ---------------------------------------------------------

def test_resolver_picks_the_roce_v2_entry_on_both_real_tables():
    assert resolve("T_HEAD", HEAD_IP) == "3:OK"
    assert resolve("T_WORKER", WORKER_IP) == "3:OK"


def test_resolver_rejects_a_roce_v1_only_table():
    """A populated v1 entry passes a naive "is it zero" check but breaks NCCL."""
    v1 = table([r"2\t0000:0000:0000:0000:0000:ffff:c0a8:b10a\tIB/RoCE v1"])
    r = run_bash(
        f"T={v1}\n"
        f'gid_index_for_ip "$T" {HEAD_IP} >/dev/null 2>&1 && echo OK || echo REJECTED'
    )
    assert "REJECTED" in r.stdout


def test_resolver_rejects_a_populated_entry_for_the_wrong_address():
    wrong = table([r"3\t0000:0000:0000:0000:0000:ffff:c0a8:b10b\tRoCE v2"])
    r = run_bash(
        f"T={wrong}\n"
        f'gid_index_for_ip "$T" {HEAD_IP} >/dev/null 2>&1 && echo OK || echo REJECTED'
    )
    assert "REJECTED" in r.stdout


def test_resolver_rejects_an_ambiguous_table():
    dup = table(
        [
            r"3\t0000:0000:0000:0000:0000:ffff:c0a8:b10a\tRoCE v2",
            r"5\t0000:0000:0000:0000:0000:ffff:c0a8:b10a\tRoCE v2",
        ]
    )
    r = run_bash(f'T={dup}\ngid_index_for_ip "$T" {HEAD_IP} 2>&1; echo "|rc=$?"')
    assert "ambiguous" in r.stdout
    assert "|rc=1" in r.stdout


def test_resolver_rejects_an_empty_table():
    r = run_bash(f'gid_index_for_ip "" {HEAD_IP} >/dev/null 2>&1 && echo OK || echo REJECTED')
    assert "REJECTED" in r.stdout


def test_resolver_ignores_a_stale_hardcoded_hint():
    """The regression that caused the outage: .env said WORKER_GID=4 while the
    fabric had moved the worker's RoCE v2 entry to gid3. Resolution must come
    from the table, so the stale hint cannot influence the answer."""
    r = run_bash(
        "WORKER_GID=4\n"
        'gid_index_for_ip "$T_WORKER" 192.168.177.11; echo " (hint was $WORKER_GID)"'
    )
    assert r.stdout.startswith("3 ")
    assert "hint was 4" in r.stdout


# --- pre-launch recheck ------------------------------------------------------

def test_require_gid_index_passes_when_the_worker_entry_still_matches():
    r = run_bash(
        "worker_ssh() {\n"
        '  case "$*" in\n'
        '    *gid_attrs/types*) echo "RoCE v2" ;;\n'
        '    *gids*) echo "0000:0000:0000:0000:0000:ffff:c0a8:b10b" ;;\n'
        "  esac\n"
        "}\n"
        "require_gid_index worker rocep1s0f1 3 192.168.177.11 && echo PASSED || echo FAILED"
    )
    assert "PASSED" in r.stdout


def test_require_gid_index_fails_closed_on_a_mismatch():
    """A moved entry must abort before launch, not surface later as a bare
    NCCL errno 61 about 60 s into the boot."""
    r = run_bash(
        "worker_ssh() {\n"
        '  case "$*" in\n'
        '    *gid_attrs/types*) echo "RoCE v2" ;;\n'
        '    *gids*) echo "0000:0000:0000:0000:0000:0000:0000:0000" ;;\n'
        "  esac\n"
        "}\n"
        "require_gid_index worker rocep1s0f1 3 192.168.177.11 && echo PASSED || echo FAILED"
    )
    assert "FAILED" in r.stdout
    assert "GID table changed" in r.stderr


def test_require_gid_index_fails_closed_when_the_entry_is_unreadable():
    """An absent/unreadable entry must not read as success."""
    r = run_bash(
        "require_gid_index head rocep1s0f1 3 192.168.177.10 && echo PASSED || echo FAILED"
    )
    assert "FAILED" in r.stdout
    assert "empty" in r.stderr


def test_require_gid_index_fails_closed_on_the_wrong_roce_version():
    r = run_bash(
        "worker_ssh() {\n"
        '  case "$*" in\n'
        '    *gid_attrs/types*) echo "IB/RoCE v1" ;;\n'
        '    *gids*) echo "0000:0000:0000:0000:0000:ffff:c0a8:b10b" ;;\n'
        "  esac\n"
        "}\n"
        "require_gid_index worker rocep1s0f1 3 192.168.177.11 && echo PASSED || echo FAILED"
    )
    assert "FAILED" in r.stdout
    assert "not RoCE v2" in r.stderr


# --- the launcher actually calls it ------------------------------------------

def test_preflight_resolves_instead_of_trusting_the_env_value():
    begin = START.index("# ------------------------------ preflight")
    end = START.index("[ \"$TP\" = \"2\" ]")
    block = START[begin:end]
    assert "gid_index_for_ip" in block
    assert "STALE" in block
    assert 'HEAD_GID="$gid_head"' in block
    assert 'WORKER_GID="$gid_worker"' in block


def test_launch_rechecks_the_resolved_index_before_starting_containers():
    begin = START.index("# ------------------------------- launch")
    end = START.index("mkdir -p \"$CACHE_ROOT\"")
    block = START[begin:end]
    assert 'require_gid_index head   "$HEAD_CX7_IB"' in block
    assert 'require_gid_index worker "$WORKER_CX7_IB"' in block
