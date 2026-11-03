from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def qualification(tmp_path):
    path = Path(__file__).resolve().parents[1] / "experiments/dense-int64/qualification/results.py"
    spec = importlib.util.spec_from_file_location("native_qualification_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "collected-bounded.txt").write_text("tests/test_kernel.py::test_bounded\n1 test collected\n")
    (tmp_path / "bounded.xml").write_text('<testsuites><testsuite><testcase classname="tests.test_kernel" name="test_bounded"/></testsuite></testsuites>')
    for name in ("high-offset.xml", "high-offset-memcheck.xml"):
        (tmp_path / name).write_text('<testsuites><testsuite><testcase classname="tests.test_kernel" name="test_nvfp4_split_partials_exceed_int32_elements"/></testsuite></testsuites>')
    (tmp_path / "memcheck-100.log").write_text("========= ERROR SUMMARY: 0 errors\n")
    return module, tmp_path


@pytest.mark.parametrize("arm", ["control", "candidate"])
def test_complete_native_receipts_pass_test_gate_but_require_resource_review(qualification, arm):
    module, root = qualification
    result = module.collect_results(root, arm)
    assert result["tests_pass_without_skips"]
    assert "manual inspection required" in result["resource_review"]


@pytest.mark.parametrize("fault", ["missing_suite", "skip", "failure", "partial", "missing_collection",
                                    "missing_high", "wrong_high", "missing_memcheck", "nonzero_memcheck",
                                    "incomplete_child"])
def test_incomplete_or_failing_native_evidence_never_passes(qualification, fault):
    module, root = qualification
    if fault == "missing_suite":
        (root / "bounded.xml").unlink()
    elif fault in ("skip", "failure"):
        tag = "skipped" if fault == "skip" else "failure"
        (root / "bounded.xml").write_text(f'<testsuite><testcase classname="tests.test_kernel" name="test_bounded"><{tag}/></testcase></testsuite>')
    elif fault == "partial":
        with (root / "collected-bounded.txt").open("a") as stream:
            stream.write("tests/test_kernel.py::test_missing\n")
    elif fault == "missing_collection":
        (root / "collected-bounded.txt").unlink()
    elif fault == "missing_high":
        (root / "high-offset.xml").unlink()
    elif fault == "wrong_high":
        (root / "high-offset.xml").write_text((root / "bounded.xml").read_text())
    elif fault == "missing_memcheck":
        (root / "memcheck-100.log").unlink()
    elif fault == "nonzero_memcheck":
        (root / "memcheck-100.log").write_text("========= ERROR SUMMARY: 1 error\n")
    elif fault == "incomplete_child":
        (root / "memcheck-101.log").write_text("========= COMPUTE-SANITIZER\n")
    assert not module.collect_results(root, "candidate")["tests_pass_without_skips"]
