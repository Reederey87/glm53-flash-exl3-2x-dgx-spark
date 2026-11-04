"""Read source identity and narrowly whitelisted process controls; no GPU imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

site = Path("/usr/local/lib/python3.12/dist-packages")
expected = {
    "vllm/envs.py": "d05fcbe03ff55bc027c7557bc9e05ff18245907c83bd815a08ffb284441fe673",
    "vllm/model_executor/layers/quantization/nvfp4_csf.py": "0ae697a86efb41a1149885b88499c878636da54cafd9f2bb5710ea4626062c5c",
}
actual = {name: hashlib.sha256((site / name).read_bytes()).hexdigest() for name in expected}
assert actual == expected, "installed source identity differs"
allowed = {
    "VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE": {"0", "1", "all", "w13", "w2"},
    "VLLM_B12X_CSF_SCALE_PREFETCH": {"0", "1"},
    "VLLM_B12X_MOE_FP4_FORCE_A16": {"0", "1"},
    "B12X_W4A16_A4_PREFILL": {"0", "1"},
}
found = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        argv = (proc / "cmdline").read_bytes().split(b"\0")
        if b"serve" not in argv or b"/usr/local/bin/vllm" not in argv:
            continue
        env = dict(part.decode().split("=", 1) for part in (proc / "environ").read_bytes().split(b"\0") if b"=" in part)
    except (OSError, UnicodeDecodeError):
        continue
    assert all(key not in env or env[key] in values for key, values in allowed.items()), "noncanonical policy value; no values exported"
    found.append({key: env.get(key, "unset") for key in allowed})
assert len(found) == 1, "expected one serving process"
controls = found[0]
assert controls["VLLM_B12X_MOE_FP4_FORCE_A16"] == "1"
assert controls["B12X_W4A16_A4_PREFILL"] == "1"
defaults = {"VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE": "0", "VLLM_B12X_CSF_SCALE_PREFETCH": "1"}
effective = {key: default if controls[key] == "unset" else controls[key] for key, default in defaults.items()}
print(json.dumps({
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "controls": controls,
    "source_resolved_effective_policy": effective,
    "installed_source_sha256": actual,
    "scope": "Read-only source hash and process-environment whitelist. Effective defaults inferred from exact pinned source; no live tensor/plan inspection or GPU calls.",
}, indent=2))
