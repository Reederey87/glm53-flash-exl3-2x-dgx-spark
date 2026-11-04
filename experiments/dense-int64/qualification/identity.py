from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

import torch
import flashinfer
import cutlass.cute as cute
from b12x._lib import dense_gemm

arm = sys.argv[1]
assert arm in ("control", "candidate")
manifest = json.loads(Path("/checks/manifest.json").read_text())
treatment = json.loads(Path("/workspace/dense-int64.json").read_text())
assert treatment["arm"] == arm
assert treatment["flashinfer_commit"] == manifest["flashinfer_base"]
assert treatment["patch_sha256"] == manifest["patch_sha256"]
package = Path(flashinfer.__file__).resolve().parent
source = Path(dense_gemm.__file__).resolve()
assert source == package / "experimental/b12x/_lib/dense_gemm.py", str(source)
key = "candidate_sha256" if arm == "candidate" else "base_sha256"
hashes = {}
for item in manifest["files"]:
    relative = item["path"]
    path = (package.parent / relative if relative.startswith("flashinfer/")
            else Path("/opt/dense-int64-tests") / relative)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == item[key] == treatment["files"][relative], (path, actual)
    hashes[str(path)] = actual
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability(0) == (12, 1)
props = torch.cuda.get_device_properties(0)
assert "GB10" in props.name
result = {
    "arm": arm, "treatment": treatment, "source_hashes": hashes,
    "dense_import": str(source), "torch": torch.__version__,
    "torch_cuda": torch.version.cuda, "device_name": props.name,
    "device_uuid": str(getattr(props, "uuid", "unavailable")),
    "capability": list(torch.cuda.get_device_capability(0)),
    "device_total_memory": props.total_memory,
    "mxf8_mma_available": hasattr(cute.nvgpu.warp, "MmaMXF8Op"),
    "packages": sorted((d.metadata["Name"], d.version) for d in metadata.distributions()),
}
Path("/receipts/identity.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "packages"}, indent=2))
assert result["mxf8_mma_available"], "MXFP8 native coverage unavailable"
