from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_live_weight_bytes.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_live_weight_bytes", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_safetensors(path: Path, tensors: dict[str, tuple[str, int]]) -> None:
    dtype_bytes = load_module().DTYPE_BYTES
    offset = 0
    header = {}
    payload = bytearray()
    for name, (dtype, size) in tensors.items():
        assert size % dtype_bytes[dtype] == 0
        header[name] = {
            "dtype": dtype,
            "shape": [size // dtype_bytes[dtype]],
            "data_offsets": [offset, offset + size],
        }
        payload.extend(b"x" * size)
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def fixture_snapshots(tmp_path: Path) -> tuple[Path, Path]:
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    target_tensors = {
        "model.language_model.embed_tokens.weight": ("BF16", 100),
        "lm_head.weight": ("BF16", 100),
        "model.language_model.layers.0.mlp.gate_proj.weight": ("BF16", 80),
        "model.language_model.layers.0.mlp.down_proj.weight": ("BF16", 80),
        "model.language_model.layers.0.self_attn.f_a_proj.weight": ("BF16", 20),
        "model.language_model.layers.0.self_attn.q_proj.weight": ("BF16", 80),
        "model.language_model.layers.3.self_attn.q_a_proj.weight": ("BF16", 40),
        "model.language_model.layers.3.self_attn.o_proj.weight": ("BF16", 80),
        "model.language_model.layers.3.mlp.shared_experts.up_proj.weight": (
            "BF16",
            80,
        ),
        "model.language_model.layers.3.mlp.experts.0.gate_proj.trellis": (
            "I16",
            40,
        ),
        "model.language_model.layers.3.mlp.experts.0.gate_proj.suh": ("F16", 8),
        "model.language_model.layers.3.mlp.experts.0.down_proj.trellis": (
            "I16",
            40,
        ),
        "model.language_model.layers.3.mlp.experts.0.down_proj.svh": ("F16", 8),
        "model.language_model.layers.4.mlp.experts.0.gate_proj.trellis": (
            "I16",
            40,
        ),
        "model.language_model.layers.4.mlp.experts.0.down_proj.trellis": (
            "I16",
            40,
        ),
        "model.language_model.layers.45.mlp.gate_proj.weight": ("BF16", 80),
        "model.visual.blocks.0.attn.qkv.weight": ("BF16", 80),
        "model.visual.patch_embed.proj.weight": ("BF16", 20),
    }
    shard = target / "model-00001-of-00001.safetensors"
    write_safetensors(shard, target_tensors)
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in target_tensors}})
    )
    (target / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_hidden_layers": 45,
                    "first_k_dense_replace": 3,
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "deepseek_sparse_attention",
                    ],
                }
            }
        )
    )
    (target / "README.md").write_text("---\nlicense: shapleymcg-1.0\n---\n")
    write_safetensors(
        draft / "model.safetensors",
        {
            "layers.0.self_attn.q_proj.weight": ("BF16", 80),
            "candidate_selector.hidden_projection.weight": ("BF16", 20),
            "hidden_norm.weight": ("BF16", 4),
        },
    )
    (draft / "README.md").write_text("---\nlicense: cc-by-nc-nd-4.0\n---\n")
    return target, draft


def test_mixed_tp_geometry_and_unused_tensors(tmp_path: Path) -> None:
    module = load_module()
    target, _ = fixture_snapshots(tmp_path)
    tensors = module.collect_target(target, 2)
    by_name = {tensor.name: tensor for tensor in tensors}
    assert by_name[
        "model.language_model.layers.0.self_attn.f_a_proj.weight"
    ].rank_fraction == 1
    assert by_name[
        "model.language_model.layers.0.self_attn.q_proj.weight"
    ].rank_fraction == 0.5
    assert by_name[
        "model.language_model.layers.3.self_attn.q_a_proj.weight"
    ].rank_fraction == 1
    assert by_name[
        "model.language_model.layers.3.mlp.experts.0.gate_proj.trellis"
    ].rank_fraction == 0.5
    assert by_name[
        "model.language_model.layers.3.mlp.experts.0.gate_proj.suh"
    ].rank_fraction == 1
    assert by_name[
        "model.language_model.layers.45.mlp.gate_proj.weight"
    ].rank_fraction == 0
    assert by_name["model.visual.blocks.0.attn.qkv.weight"].rank_fraction == 0.5
    assert by_name["model.visual.patch_embed.proj.weight"].rank_fraction == 1


def test_report_separates_residency_traffic_and_permissions(tmp_path: Path) -> None:
    module = load_module()
    target, draft = fixture_snapshots(tmp_path)
    args = type(
        "Args",
        (),
        {
            "target_snapshot": target,
            "draft_snapshot": draft,
            "tp_size": 2,
            "draft_tp_size": 1,
            "active_experts": [1],
            "accepted_per_step": 3.0,
        },
    )()
    report = module.build_report(args)
    assert report["estimated_resident_weight_bytes_by_rank"][0] > report[
        "estimated_resident_weight_bytes_by_rank"
    ][1]
    assert report["target"]["categories"]["target.vision"]["disk_bytes"] == 100
    assert report["target"]["categories"]["target.mtp_unused_dflash"][
        "rank_bytes"
    ] == 0
    scenario = report["traffic"]["scenarios"][0]
    assert scenario["emitted_tokens_per_verify_step"] == 4
    assert scenario["routed_weight_bytes_per_verify_step"] == 96
    assert report["permissions"]["draft_quantization"].startswith("BLOCKED")
    assert "explicitly replaces" in report["permissions"]["quality_gate"]


def test_invalid_header_fails_closed(tmp_path: Path) -> None:
    module = load_module()
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"\x10\x00\x00")
    try:
        module.read_header(broken)
    except ValueError as exc:
        assert "missing safetensors header length" in str(exc)
    else:
        raise AssertionError("truncated safetensors header was accepted")


def test_malformed_tensor_records_and_payloads_fail_closed(tmp_path: Path) -> None:
    module = load_module()
    malformed = tmp_path / "malformed.safetensors"
    header = json.dumps({"tensor": "not-an-object"}).encode()
    malformed.write_bytes(struct.pack("<Q", len(header)) + header)
    try:
        module.read_header(malformed)
    except ValueError as exc:
        assert "record is not an object" in str(exc)
    else:
        raise AssertionError("malformed tensor record was accepted")

    oversized = tmp_path / "oversized.safetensors"
    header = json.dumps(
        {
            "tensor": {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 1_000_000],
            }
        }
    ).encode()
    oversized.write_bytes(struct.pack("<Q", len(header)) + header)
    try:
        module.read_header(oversized)
    except ValueError as exc:
        assert "invalid data_offsets" in str(exc)
    else:
        raise AssertionError("out-of-bounds tensor payload was accepted")

    wrong_size = tmp_path / "wrong-size.safetensors"
    header = json.dumps(
        {
            "tensor": {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        }
    ).encode()
    wrong_size.write_bytes(struct.pack("<Q", len(header)) + header + b"xxxx")
    try:
        module.read_header(wrong_size)
    except ValueError as exc:
        assert "shape/dtype length" in str(exc)
    else:
        raise AssertionError("shape/dtype mismatch was accepted")


def test_custom_license_name_is_reported(tmp_path: Path) -> None:
    module = load_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "README.md").write_text(
        "---\nlicense: other\nlicense_name: shapleymcg-1.0\n---\n"
    )
    assert module.read_license(snapshot) == "shapleymcg-1.0"


def test_unknown_target_and_draft_tensors_make_cli_fail(tmp_path: Path) -> None:
    target, draft = fixture_snapshots(tmp_path)
    target_shard = target / "model-00001-of-00001.safetensors"
    write_safetensors(
        target_shard,
        {
            "model.unrecognized.weight": ("BF16", 8),
        },
    )
    (target / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.unrecognized.weight": target_shard.name,
                }
            }
        )
    )
    write_safetensors(
        draft / "model.safetensors",
        {"mystery.weight": ("BF16", 8)},
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--target-snapshot",
            str(target),
            "--draft-snapshot",
            str(draft),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert {item["name"] for item in report["unclassified"]} == {
        "model.unrecognized.weight",
        "mystery.weight",
    }


def test_unused_mtp_is_excluded_at_tp1(tmp_path: Path) -> None:
    module = load_module()
    target, _ = fixture_snapshots(tmp_path)
    tensors = module.collect_target(target, 1)
    mtp = next(
        tensor
        for tensor in tensors
        if tensor.name == "model.language_model.layers.45.mlp.gate_proj.weight"
    )
    assert mtp.rank_fraction == 0
