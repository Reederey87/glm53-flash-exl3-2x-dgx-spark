#!/usr/bin/env python3
"""Unit tests for the task 24 W5 ncu occupancy auditor (CPU only)."""

import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "ncu-e3-occupancy-20260909.csv"

REQUIRED_OK = {
    "Registers Per Thread": "128 register/thread",
    "Block Size": "256",
    "Grid Size": "2304",
    "Block Limit Registers": "2 block",
    "Block Limit Shared Mem": "3 block",
    "Block Limit Warps": "6 block",
    "Theoretical Occupancy": "33.33 %",
    "Achieved Occupancy": "33.02 %",
    "Waves Per SM": "24",
}


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        "audit_e3_occupancy", ROOT / "scripts" / "audit_e3_occupancy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(metrics: dict[str, str], kernel: str, launch_id: str = "0") -> str:
    head = (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",'
        '"Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"'
    )
    body = [
        f'"{launch_id}","1","python3","spark-a183","{kernel}","t","1","7","S","{name}","","{value}"'
        for name, value in metrics.items()
    ]
    return "\n".join([head, *body]) + "\n"


def one_launch(capture: dict, **overrides: float) -> dict:
    """Fixture capture with a single launch per kernel, optionally overridden."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for kernel, launches in capture.items():
        launch = sorted(launches, key=lambda key: (key.isdigit(), key))[0]
        out[kernel] = {launch: {**launches[launch], **overrides}}
    return out


def only(launches: dict) -> dict[str, float]:
    """The single launch record of a one-launch kernel."""
    return next(iter(launches.values()))


def fixture_rows() -> tuple[list[str], list[list[str]]]:
    """The real capture as (header, data rows), past ncu's log preamble."""
    lines = FIXTURE.read_text().splitlines()
    start = next(index for index, line in enumerate(lines) if "Metric Name" in line)
    parsed = list(csv.reader(lines[start:]))
    return parsed[0], parsed[1:]


def render(header: list[str], body: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(body)
    return buf.getvalue()


def edit_fixture(edit) -> str:
    """The real capture with ``edit(row, header) -> row`` applied to every row."""
    header, body = fixture_rows()
    return render(header, [edit(row, header) for row in body])


def column(header: list[str], name: str) -> int:
    return [item.strip().lower() for item in header].index(name.lower())


class ParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audit = load_auditor()

    def test_production_fixture_parses_both_kernels(self) -> None:
        capture = self.audit.parse_csv(FIXTURE.read_text())
        self.assertEqual(set(capture), {"fm_gateup_kernel", "fm_down_kernel"})
        gateup = only(capture["fm_gateup_kernel"])
        self.assertEqual(gateup["registers_per_thread"], 128.0)
        self.assertEqual(gateup["block_size"], 256.0)
        self.assertEqual(gateup["grid_size"], 2304.0)
        self.assertEqual(gateup["block_limit_registers"], 2.0)
        self.assertEqual(gateup["block_limit_shared_mem"], 3.0)
        self.assertEqual(gateup["block_limit_warps"], 6.0)
        self.assertEqual(gateup["smem_dynamic"], 32768.0)
        self.assertAlmostEqual(gateup["achieved_occupancy_pct"], 33.02)
        self.assertAlmostEqual(gateup["theoretical_occupancy_pct"], 33.33)
        self.assertEqual(sorted(capture["fm_down_kernel"]), ["0", "2"])
        self.assertEqual(only(capture["fm_down_kernel"])["grid_size"], 4608.0)

    def test_launches_are_kept_separate(self) -> None:
        text = (
            rows(REQUIRED_OK, "fm_gateup_kernel", "1")
            + rows({**REQUIRED_OK, "Achieved Occupancy": "33.17 %"}, "fm_down_kernel", "0")
            + rows({**REQUIRED_OK, "Achieved Occupancy": "33.18 %"}, "fm_down_kernel", "2")
        )
        capture = self.audit.parse_csv(text)
        self.assertEqual(sorted(capture["fm_down_kernel"]), ["0", "2"])
        self.assertAlmostEqual(
            capture["fm_down_kernel"]["0"]["achieved_occupancy_pct"], 33.17
        )
        self.assertAlmostEqual(
            capture["fm_down_kernel"]["2"]["achieved_occupancy_pct"], 33.18
        )

    def test_raw_metric_names_match_labels(self) -> None:
        text = rows(
            {
                "launch__registers_per_thread": "128",
                "launch__block_size": "256",
                "launch__occupancy_limit_registers": "2",
                "launch__occupancy_limit_shared_mem": "3",
                "launch__occupancy_limit_warps": "6",
                "sm__maximum_warps_per_active_cycle_pct": "33.33",
                "sm__warps_active.avg.pct_of_peak_sustained_active": "31.42",
                "launch__waves_per_multiprocessor": "5.33",
            },
            "fm_gateup_kernel",
        ) + rows(
            {
                "launch__registers_per_thread": "128",
                "launch__block_size": "256",
                "launch__occupancy_limit_registers": "2",
                "launch__occupancy_limit_shared_mem": "3",
                "launch__occupancy_limit_warps": "6",
                "sm__maximum_warps_per_active_cycle_pct": "33.33",
                "sm__warps_active.avg.pct_of_peak_sustained_active": "30.87",
                "launch__waves_per_multiprocessor": "10.67",
            },
            "fm_down_kernel",
        )
        capture = self.audit.parse_csv(text)
        self.assertEqual(capture["fm_down_kernel"]["0"]["registers_per_thread"], 128.0)
        self.assertAlmostEqual(
            only(capture["fm_gateup_kernel"])["theoretical_occupancy_pct"], 33.33
        )

    def test_unknown_kernels_ignored(self) -> None:
        text = rows({"Registers Per Thread": "64"}, "some_other_kernel")
        self.assertEqual(self.audit.parse_csv(text), {})

    def test_ncu_log_preamble_is_skipped(self) -> None:
        text = (
            "==WARNING== Note: Running with uncontrolled GPU caches. "
            "Profiling results may be inconsistent.\n"
            "==PROF== Connected to process 66 (/usr/bin/python3.12)\n"
            + rows({"Registers Per Thread": "128"}, "fm_gateup_kernel")
        )
        capture = self.audit.parse_csv(text)
        self.assertEqual(only(capture["fm_gateup_kernel"])["registers_per_thread"], 128.0)

    def test_missing_header_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            self.audit.parse_csv("==PROF== no csv here\n")

    def test_missing_columns_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            self.audit.parse_csv('"Metric Name","Metric Value"\n"a","2"\n')

    def test_missing_launch_id_column_fails_closed(self) -> None:
        """Without launch ids two profiles of one kernel would merge silently."""
        lines = rows(REQUIRED_OK, "fm_gateup_kernel").splitlines()
        text = "\n".join(
            ",".join(part for index, part in enumerate(line.split(",")) if index != 0)
            for line in lines
        ) + "\n"
        with self.assertRaises(ValueError) as caught:
            self.audit.parse_csv(text)
        self.assertIn("launch-id column", str(caught.exception))

    def test_blank_launch_id_fails_closed(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.audit.parse_csv(rows(REQUIRED_OK, "fm_gateup_kernel", ""))
        self.assertIn("no launch id", str(caught.exception))

    def test_unreadable_metric_values_are_not_numbers(self) -> None:
        for raw in ("N/A", "", "ERROR (33.02)", "-", "n/a"):
            with self.subTest(raw=raw):
                self.assertIsNone(self.audit._number(raw))
        for raw, expected in (("128 register/thread", 128.0), ("33.33 %", 33.33),
                              ("4608", 4608.0), ("0 byte/block", 0.0)):
            with self.subTest(raw=raw):
                self.assertEqual(self.audit._number(raw), expected)

    def test_launch_with_unreadable_metrics_still_becomes_a_record(self) -> None:
        text = (
            rows(REQUIRED_OK, "fm_down_kernel", "0")
            + rows({name: "N/A" for name in REQUIRED_OK}, "fm_down_kernel", "2")
        )
        capture = self.audit.parse_csv(text)
        self.assertEqual(sorted(capture["fm_down_kernel"]), ["0", "2"])
        record = capture["fm_down_kernel"]["2"]
        self.assertTrue(record)
        self.assertTrue(all(value is None for value in record.values()), record)

    def test_populated_row_without_a_kernel_name_fails_closed(self) -> None:
        """Blanking Kernel Name on a launch's rows must not drop that launch."""
        text = edit_fixture(
            lambda row, header: ["" if index == column(header, "Kernel Name") else cell
                                 for index, cell in enumerate(row)]
            if row[0] == "2" else row
        )
        with self.assertRaises(ValueError) as caught:
            self.audit.parse_csv(text)
        self.assertIn("has no kernel name", str(caught.exception))

    def test_blank_rows_are_still_skipped(self) -> None:
        """Blank lines and cells that carry no launch id or metric stay ignorable."""
        text = rows(REQUIRED_OK, "fm_gateup_kernel", "1") + '""\n\n"","",""\n'
        capture = self.audit.parse_csv(text)
        self.assertEqual(sorted(capture["fm_gateup_kernel"]), ["1"])


class JudgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audit = load_auditor()
        self.capture = self.audit.parse_csv(FIXTURE.read_text())

    def test_register_headroom_stops_w5(self) -> None:
        report = self.audit.judge(self.capture)
        self.assertEqual(report["decision"], "STOP_REGISTER_HEADROOM")
        for name, launches in (("fm_gateup_kernel", 1), ("fm_down_kernel", 2)):
            self.assertEqual(report["kernels"][name]["verdict"], "STOP_REGISTER_HEADROOM")
            self.assertEqual(report["kernels"][name]["launch_count"], launches)
        self.assertIn("register file binds", report["reason"])

    def test_low_register_high_achieved_stops_without_gap(self) -> None:
        report = self.audit.judge(
            one_launch(
                self.capture,
                registers_per_thread=64.0,
                theoretical_occupancy_pct=50.0,
                achieved_occupancy_pct=47.0,
            )
        )
        self.assertEqual(report["decision"], "STOP_NO_GAP")

    def test_low_register_low_achieved_opens_w5(self) -> None:
        report = self.audit.judge(
            one_launch(
                self.capture,
                registers_per_thread=64.0,
                theoretical_occupancy_pct=50.0,
                achieved_occupancy_pct=15.0,
            )
        )
        self.assertEqual(report["decision"], "OPEN_GAP")
        self.assertAlmostEqual(
            report["kernels"]["fm_gateup_kernel"]["achieved_over_theoretical"], 0.3
        )

    def test_gate_uses_the_minimum_across_launches(self) -> None:
        capture = one_launch(self.capture)
        base = only(capture["fm_down_kernel"])
        capture["fm_down_kernel"] = {
            "9": {**base, "registers_per_thread": 64.0,
                  "theoretical_occupancy_pct": 50.0, "achieved_occupancy_pct": 47.0},
            "10": {**base, "registers_per_thread": 64.0,
                   "theoretical_occupancy_pct": 50.0, "achieved_occupancy_pct": 20.0},
        }
        report = self.audit.judge(capture)
        down = report["kernels"]["fm_down_kernel"]
        self.assertEqual(down["launch_count"], 2)
        self.assertAlmostEqual(down["achieved_occupancy_pct"], 20.0)
        self.assertEqual(down["verdict"], "OPEN_GAP")
        self.assertAlmostEqual(
            report["kernels"]["fm_down_kernel"]["launches"]["9"]["achieved_occupancy_pct"], 47.0
        )

    def test_missing_kernel_aborts(self) -> None:
        capture = {"fm_gateup_kernel": self.capture["fm_gateup_kernel"]}
        report = self.audit.judge(capture)
        self.assertEqual(report["decision"], "ABORT")
        self.assertIn("fm_down_kernel", report["reason"])

    def test_missing_metric_aborts(self) -> None:
        capture = {
            name: {
                launch: {k: v for k, v in metrics.items() if k != "achieved_occupancy_pct"}
                for launch, metrics in launches.items()
            }
            for name, launches in self.capture.items()
        }
        report = self.audit.judge(capture)
        self.assertEqual(report["decision"], "ABORT")
        self.assertIn("achieved_occupancy_pct", report["reason"])

    def test_incomplete_second_launch_aborts_instead_of_borrowing(self) -> None:
        capture = one_launch(self.capture)
        base = only(capture["fm_down_kernel"])
        capture["fm_down_kernel"] = {
            "9": base,
            "10": {k: v for k, v in base.items() if k != "achieved_occupancy_pct"},
        }
        report = self.audit.judge(capture)
        self.assertEqual(report["decision"], "ABORT")
        self.assertIn("fm_down_kernel", report["reason"])
        self.assertIn("same metrics", report["reason"])

    def test_unreadable_grid_size_aborts(self) -> None:
        """A recognised metric spelled N/A is not a measurement."""
        text = edit_fixture(
            lambda row, header: row[: column(header, "Metric Value")] + ["N/A"]
            if row[column(header, "Metric Name")] == "Grid Size" else row
        )
        report = self.audit.judge(self.audit.parse_csv(text))
        self.assertEqual(report["decision"], "ABORT")
        self.assertIn("unreadable metric value(s) ['grid_size']", report["reason"])

    def test_missing_grid_size_aborts(self) -> None:
        """Deleting the geometry rows must not let an unverifiable capture through."""
        header, body = fixture_rows()
        body = [row for row in body if row[column(header, "Metric Name")] != "Grid Size"]
        report = self.audit.judge(self.audit.parse_csv(render(header, body)))
        self.assertEqual(report["decision"], "ABORT")
        self.assertIn("grid_size", report["reason"])

    def test_expected_launch_count_is_enforced(self) -> None:
        self.assertEqual(
            self.audit.judge(self.capture, expected_launches=3)["decision"],
            "STOP_REGISTER_HEADROOM",
        )
        report = self.audit.judge(self.capture, expected_launches=4)
        self.assertEqual(report["decision"], "ABORT")
        self.assertIn("expected 4", report["reason"])

    def test_dropped_launch_aborts_when_the_count_is_known(self) -> None:
        """A launch that vanishes from the capture must not weaken the gate."""
        capture = {
            name: {launch: metrics for launch, metrics in launches.items() if launch != "2"}
            for name, launches in self.capture.items()
        }
        self.assertEqual(self.audit.judge(capture)["decision"], "STOP_REGISTER_HEADROOM")
        report = self.audit.judge(capture, expected_launches=3)
        self.assertEqual(report["decision"], "ABORT")
        self.assertIn("2 launch record(s)", report["reason"])

    def test_zero_theoretical_aborts(self) -> None:
        report = self.audit.judge(
            one_launch(self.capture, theoretical_occupancy_pct=0.0)
        )
        self.assertEqual(report["decision"], "ABORT")
        self.assertIn("theoretical occupancy", report["reason"])

    def test_zero_achieved_aborts(self) -> None:
        report = self.audit.judge(one_launch(self.capture, achieved_occupancy_pct=0.0))
        self.assertEqual(report["decision"], "ABORT")
        self.assertIn("achieved occupancy", report["reason"])

    def test_achieved_above_theoretical_aborts(self) -> None:
        report = self.audit.judge(
            one_launch(
                self.capture, theoretical_occupancy_pct=33.33, achieved_occupancy_pct=90.0
            )
        )
        self.assertEqual(report["decision"], "ABORT")
        self.assertIn("exceeds theoretical", report["reason"])

    def test_zero_registers_aborts(self) -> None:
        report = self.audit.judge(one_launch(self.capture, registers_per_thread=0.0))
        self.assertEqual(report["decision"], "ABORT")
        self.assertIn("registers/thread", report["reason"])

    def test_all_metrics_unreadable_aborts(self) -> None:
        """A launch whose every value is N/A is a missing-metric ABORT, not invisible."""
        text = (
            rows(REQUIRED_OK, "fm_gateup_kernel", "1")
            + rows(REQUIRED_OK, "fm_down_kernel", "0")
            + rows({name: "N/A" for name in REQUIRED_OK}, "fm_down_kernel", "2")
        )
        report = self.audit.judge(self.audit.parse_csv(text))
        self.assertEqual(report["decision"], "ABORT")
        self.assertIn("fm_down_kernel launch 2", report["reason"])

    def test_error_text_in_a_value_aborts(self) -> None:
        text = (
            rows(REQUIRED_OK, "fm_gateup_kernel", "1")
            + rows({**REQUIRED_OK, "Achieved Occupancy": "ERROR (33.02) %"},
                   "fm_down_kernel", "0")
        )
        report = self.audit.judge(self.audit.parse_csv(text))
        self.assertEqual(report["decision"], "ABORT")
        self.assertIn("achieved_occupancy_pct", report["reason"])


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audit = load_auditor()

    def test_abort_writes_the_artifact_and_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ncu.csv"
            out_path = Path(tmp) / "audit.json"
            csv_path.write_text(
                rows(REQUIRED_OK, "fm_gateup_kernel")
                + rows({**REQUIRED_OK, "Theoretical Occupancy": "0 %"}, "fm_down_kernel")
            )
            rc = self.audit.main(["--ncu-csv", str(csv_path), "--out", str(out_path)])
            self.assertEqual(rc, 1)
            self.assertEqual(json.loads(out_path.read_text())["decision"], "ABORT")

    def test_unparseable_capture_writes_the_artifact_and_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ncu.csv"
            out_path = Path(tmp) / "audit.json"
            csv_path.write_text("==PROF== no csv here\n")
            rc = self.audit.main(["--ncu-csv", str(csv_path), "--out", str(out_path)])
            self.assertEqual(rc, 1)
            self.assertEqual(json.loads(out_path.read_text())["decision"], "ABORT")

    def test_real_capture_still_stops_with_the_expected_launch_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ncu.csv"
            out_path = Path(tmp) / "audit.json"
            csv_path.write_text(FIXTURE.read_text())
            rc = self.audit.main(
                ["--ncu-csv", str(csv_path), "--expected-launches", "3",
                 "--out", str(out_path)]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(
                json.loads(out_path.read_text())["decision"], "STOP_REGISTER_HEADROOM"
            )

    def test_malformed_launch_rows_exit_nonzero(self) -> None:
        """The two silent-launch-loss repros must both fail the capture closed."""
        cases = {
            "blanked kernel name": lambda row, header: [
                "" if index == column(header, "Kernel Name") else cell
                for index, cell in enumerate(row)
            ] if row[0] == "2" else row,
            "unreadable grid size": lambda row, header: (
                row[: column(header, "Metric Value")] + ["N/A"]
                if row[column(header, "Metric Name")] == "Grid Size" else row
            ),
        }
        for name, edit in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                csv_path = Path(tmp) / "ncu.csv"
                out_path = Path(tmp) / "audit.json"
                csv_path.write_text(edit_fixture(edit))
                rc = self.audit.main(
                    ["--ncu-csv", str(csv_path), "--expected-launches", "3",
                     "--out", str(out_path)]
                )
                self.assertEqual(rc, 1)
                report = json.loads(out_path.read_text())
                self.assertEqual(report["decision"], "ABORT")
                self.assertTrue(report["reason"])


if __name__ == "__main__":
    unittest.main()
