from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

def collect_results(root: Path, arm: str) -> dict:
    if arm not in ("control", "candidate"):
        raise ValueError("unknown arm")
    required = ["bounded.xml"]
    if arm == "candidate":
        required += ["high-offset.xml", "high-offset-memcheck.xml"]
    collection = root / "collected-bounded.txt"
    expected = set()
    if collection.is_file():
        for line in collection.read_text().splitlines():
            if line.startswith("tests/") and "::" in line:
                path, name = line.split("::", 1)
                expected.add((path.removesuffix(".py").replace("/", "."), name))
    results = {}
    valid = bool(expected)
    for name in required:
        path = root / name
        if not path.is_file():
            results[name] = {"missing": True}
            valid = False
            continue
        cases = list(ET.parse(path).getroot().iter("testcase"))
        failed = [c.attrib for c in cases if c.find("failure") is not None or c.find("error") is not None]
        skipped = [dict(c.attrib, reason=c.find("skipped").get("message", ""))
                   for c in cases if c.find("skipped") is not None]
        results[name] = {"cases": len(cases), "failed": failed, "skipped": skipped}
        valid &= bool(cases) and not failed and not skipped
        if name == "bounded.xml":
            actual = {(c.get("classname"), c.get("name")) for c in cases}
            results[name]["missing_cases"] = sorted(expected - actual)
            results[name]["unexpected_cases"] = sorted(actual - expected)
            valid &= actual == expected and len(cases) == len(expected)
        else:
            valid &= len(cases) == 1
            valid &= all(c.get("name") == "test_nvfp4_split_partials_exceed_int32_elements" for c in cases)

    sanitizer = []
    if arm == "candidate":
        for path in sorted(root.glob("memcheck-*.log")):
            summaries = [int(n) for n in re.findall(r"ERROR SUMMARY: (\d+) errors?", path.read_text())]
            sanitizer.append({"path": path.name, "error_summaries": summaries})
        valid &= bool(sanitizer) and all(item["error_summaries"] for item in sanitizer)
        valid &= all(n == 0 for item in sanitizer for n in item["error_summaries"])

    artifacts = []
    for path in sorted((root / "cache").rglob("*")):
        if not path.is_file() or path.suffix not in (".o", ".so", ".cubin", ".ptx"):
            continue
        item = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix != ".ptx":
            report = root / "resources" / (item["sha256"] + ".txt")
            report.parent.mkdir(exist_ok=True)
            output = subprocess.run(["cuobjdump", "--dump-resource-usage", str(path)],
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            report.write_text(output.stdout)
            item["cuobjdump_status"] = output.returncode
            item["resource_report"] = str(report.relative_to(root))
        artifacts.append(item)
    return {"arm": arm, "tests_pass_without_skips": bool(valid),
            "tests": results, "sanitizer": sanitizer, "native_artifacts": artifacts,
            "resource_review": "manual inspection required; collection is not a qualification pass"}


if __name__ == "__main__":
    root = Path("/receipts")
    result = collect_results(root, sys.argv[1])
    (root / "native-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "native_artifacts"}, indent=2))
    sys.exit(0 if result["tests_pass_without_skips"] else 1)
