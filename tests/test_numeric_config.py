#!/usr/bin/env python3
"""CPU-only tests for launcher numeric type/range validation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"


def guard_source() -> str:
    source = START.read_text()
    begin = source.index("# GLM53 numeric config guard (begin)")
    end_marker = "# GLM53 numeric config guard (end)"
    end = source.index(end_marker, begin) + len(end_marker)
    # Mirror production ordering: the W41/W42 knob-defaults block always runs
    # before validate_numeric_config, so the strict-bool loop never sees an
    # UNSET knob (unset -> 1 happens in the defaults, "" stays a value).
    d_begin = source.index("# LOCAL: W41/W42 knob defaults (begin)")
    d_end_marker = "# LOCAL: W41/W42 knob defaults (end)"
    d_end = source.index(d_end_marker, d_begin) + len(d_end_marker)
    return source[d_begin:d_end] + "\n" + source[begin:end]


def validate(
    util: str,
    model: str,
    seqs: str,
    batch: str,
    spec_method: str = "dflash",
    dflash_tokens: str = "7",
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    script = (
        guard_source()
        + '\nGPU_MEM_UTIL="$1"; MAX_MODEL_LEN="$2"; MAX_NUM_SEQS="$3"; '
        + 'MAX_NUM_BATCHED_TOKENS="$4"; SPEC_METHOD="$5"; DFLASH_TOKENS="$6"\n'
        + 'validate_numeric_config || exit $?\n'
        + 'printf "%s|%s|%s|%s|%s\\n" "$GPU_MEM_UTIL" "$MAX_MODEL_LEN" '
        + '"$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" "$DFLASH_TOKENS"\n'
    )
    env = {**os.environ, "LC_ALL": "C"}
    env.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [
            "bash",
            "-c",
            script,
            "test",
            util,
            model,
            seqs,
            batch,
            spec_method,
            dflash_tokens,
        ],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def expect_rc(values: tuple[str, str, str, str], expected: int) -> None:
    result = validate(*values)
    assert result.returncode == expected, (values, result.returncode, result.stdout, result.stderr)


def test_matrix() -> None:
    expect_rc(("0.87", "1000000", "4", "1024"), 0)
    expect_rc((".87", "01000000", "0004", "01024"), 0)
    expect_rc(("1.0", "1000000", "4096", "8388608"), 0)
    expect_rc(("0", "1000000", "4", "1024"), 2)
    expect_rc(("8.7", "1000000", "4", "1024"), 2)
    expect_rc(("nope", "1000000", "4", "1024"), 2)
    expect_rc(("0.87", "0", "4", "1024"), 2)
    expect_rc(("0.87", "1000001", "4", "1024"), 2)
    expect_rc(("0.87", "1000000", "O4", "1024"), 2)
    expect_rc(("0.87", "1000000", "4097", "1024"), 2)
    expect_rc(("0.87", "1000000", "4", "1024\r"), 2)
    expect_rc(("0.87", "1000000", "4", "18446744073709551615"), 2)


def test_decimal_normalization() -> None:
    result = validate(".87", "01000000", "0004", "01024", dflash_tokens="07")
    assert result.returncode == 0
    assert result.stdout.strip() == ".87|1000000|4|1024|7"


def test_dflash2_rejects_non_native_block_lengths() -> None:
    for tokens in ("3", "4", "8", "nope"):
        result = validate("0.87", "1000000", "4", "1024", dflash_tokens=tokens)
        assert result.returncode == 2
    result = validate(
        "0.87", "1000000", "4", "1024", spec_method="none", dflash_tokens="4"
    )
    assert result.returncode == 0


def test_v2_runner_force_off_is_refused() -> None:
    refused = validate(
        "0.87",
        "1000000",
        "4",
        "1024",
        extra_env={"VLLM_USE_V2_MODEL_RUNNER": "0"},
    )
    assert refused.returncode == 2
    assert "VLLM_USE_V2_MODEL_RUNNER=0 is refused" in refused.stderr
    allowed = validate(
        "0.87",
        "1000000",
        "4",
        "1024",
        extra_env={"VLLM_USE_V2_MODEL_RUNNER": "1"},
    )
    assert allowed.returncode == 0
    invalid = validate(
        "0.87",
        "1000000",
        "4",
        "1024",
        extra_env={"VLLM_USE_V2_MODEL_RUNNER": "true"},
    )
    assert invalid.returncode == 2


def test_fat_grouped_is_strict_bool() -> None:
    refused = validate(
        "0.87",
        "1000000",
        "4",
        "1024",
        extra_env={"EXL3_FAT_GROUPED": "2"},
    )
    assert refused.returncode == 2
    assert "EXL3_FAT_GROUPED must be exactly 0 or 1" in refused.stderr
    allowed = validate(
        "0.87",
        "1000000",
        "4",
        "1024",
        extra_env={"EXL3_FAT_GROUPED": "1"},
    )
    assert allowed.returncode == 0
    off = validate(
        "0.87",
        "1000000",
        "4",
        "1024",
        extra_env={"EXL3_FAT_GROUPED": "0"},
    )
    assert off.returncode == 0


def test_fat_scratch_rows_optional_int() -> None:
    allowed = validate(
        "0.87",
        "1000000",
        "4",
        "1024",
        extra_env={"EXL3_FAT_SCRATCH_ROWS": "28672"},
    )
    assert allowed.returncode == 0
    empty = validate(
        "0.87",
        "1000000",
        "4",
        "1024",
        extra_env={"EXL3_FAT_SCRATCH_ROWS": ""},
    )
    assert empty.returncode == 0
    refused = validate(
        "0.87",
        "1000000",
        "4",
        "1024",
        extra_env={"EXL3_FAT_SCRATCH_ROWS": "nope"},
    )
    assert refused.returncode == 2
    assert "EXL3_FAT_SCRATCH_ROWS must be empty" in refused.stderr


def test_mtp_above_12_seqs_is_refused() -> None:
    refused = validate("0.87", "1000000", "13", "1024", spec_method="mtp")
    assert refused.returncode == 2
    assert "SPEC_METHOD=mtp refuses to boot when MAX_NUM_SEQS > 12" in refused.stderr
    allowed = validate("0.87", "1000000", "12", "1024", spec_method="mtp")
    assert allowed.returncode == 0
    small = validate("0.87", "1000000", "4", "1024", spec_method="mtp")
    assert small.returncode == 0


def test_restart_validates_before_stop() -> None:
    source = START.read_text()
    main = source.index("main() {")
    validation = source.index("start|restart|validate) validate_numeric_config", main)
    restart = source.index("restart)  stop; start", main)
    assert validation < restart


if __name__ == "__main__":
    test_matrix()
    test_decimal_normalization()
    test_restart_validates_before_stop()
    print("numeric config tests: PASS")
