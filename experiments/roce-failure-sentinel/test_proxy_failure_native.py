"""Compile and execute the installed C proxy's bounded failure branches."""
from pathlib import Path
import subprocess


def test_installed_proxy_failure_predicates(tmp_path):
    from flashinfer.experimental.b12x.comm.roce import _proxy

    source = Path(_proxy.__file__).parent
    assert _proxy.ABI_VERSION == 8
    harness = Path(__file__).with_name("proxy_failure_harness.c")
    executable = tmp_path / "proxy-failure"
    subprocess.run(["cc", "-std=c11", "-O2", "-I", str(source), str(harness),
                    "-libverbs", "-lpthread", "-o", str(executable)],
                   check=True, timeout=60, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, timeout=10,
                            capture_output=True, text=True)
    assert result.stdout.strip() == "proxy_failure_cases_passed=12"
