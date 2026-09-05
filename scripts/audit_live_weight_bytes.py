#!/usr/bin/env python3
"""Audit GLM-5.3-Flash weight residency and decode-step traffic.

The audit reads safetensors headers only. It does not load tensor payloads.
Traffic figures are a weight-streaming model, not hardware-counter receipts.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    disk_bytes: int
    category: str
    rank_fraction: float
    geometry: str


def read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: missing safetensors header length")
        header_len = struct.unpack("<Q", raw)[0]
        header = handle.read(header_len)
        if len(header) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
    decoded = json.loads(header)
    if not isinstance(decoded, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    payload_bytes = path.stat().st_size - 8 - header_len
    ranges = []
    for name, entry in decoded.items():
        if name == "__metadata__":
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: __metadata__ is not an object")
            continue
        validate_tensor_entry(path, name, entry, payload_bytes)
        ranges.append((entry["data_offsets"][0], entry["data_offsets"][1], name))
    ranges.sort()
    previous_end = 0
    for start, end, name in ranges:
        if start != previous_end:
            kind = "overlap" if start < previous_end else "gap"
            raise ValueError(f"{path}: tensor {name!r} has payload {kind}")
        previous_end = end
    if previous_end != payload_bytes:
        raise ValueError(
            f"{path}: payload size {payload_bytes} != tensor extent {previous_end}"
        )
    return decoded


def validate_tensor_entry(
    path: Path, name: str, entry: Any, payload_bytes: int
) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: tensor {name!r} record is not an object")
    dtype = entry.get("dtype")
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"{path}: tensor {name!r} has unsupported dtype {dtype!r}")
    shape = entry.get("shape")
    if not isinstance(shape, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in shape
    ):
        raise ValueError(f"{path}: tensor {name!r} has invalid shape {shape!r}")
    elements = 1
    for value in shape:
        elements *= value
    offsets = entry.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in offsets
        )
        or offsets[0] < 0
        or offsets[1] < offsets[0]
        or offsets[1] > payload_bytes
    ):
        raise ValueError(
            f"{path}: tensor {name!r} has invalid data_offsets {offsets!r}"
        )
    expected = elements * DTYPE_BYTES[dtype]
    actual = offsets[1] - offsets[0]
    if actual != expected:
        raise ValueError(
            f"{path}: tensor {name!r} byte length {actual} != "
            f"shape/dtype length {expected}"
        )


def read_indexed_tensors(snapshot: Path) -> list[tuple[str, dict[str, Any]]]:
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: missing non-empty weight_map")
    headers = {
        filename: read_header(snapshot / filename)
        for filename in sorted(set(weight_map.values()))
    }
    tensors = []
    for name, filename in weight_map.items():
        entry = headers[filename].get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"{snapshot / filename}: indexed tensor {name!r} missing")
        tensors.append((name, entry))
    return tensors


def read_single_tensors(path: Path) -> list[tuple[str, dict[str, Any]]]:
    header = read_header(path)
    return [
        (name, entry)
        for name, entry in header.items()
        if name != "__metadata__"
    ]


def tensor_bytes(entry: dict[str, Any]) -> int:
    offsets = entry.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) for value in offsets)
        or offsets[0] < 0
        or offsets[1] < offsets[0]
    ):
        raise ValueError(f"invalid tensor data_offsets: {offsets!r}")
    return offsets[1] - offsets[0]


def layer_index(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def target_category(name: str, config: dict[str, Any]) -> str:
    text = config.get("text_config", config)
    base_layers = int(text.get("num_hidden_layers", 0))
    idx = layer_index(name)
    if name.startswith("model.visual."):
        return "target.vision"
    if name == "model.language_model.embed_tokens.weight":
        return "target.embedding"
    if name == "lm_head.weight":
        return "target.lm_head"
    if idx is not None and idx >= base_layers:
        return "target.mtp_unused_dflash"
    if ".mlp.experts." in name:
        return "target.routed_experts"
    if ".mlp.shared_experts." in name:
        return "target.shared_experts"
    if idx is not None and idx < int(text.get("first_k_dense_replace", 0)):
        if ".mlp." in name:
            return "target.dense_mlp"
    if ".mlp.gate." in name:
        return "target.router"
    if ".self_attn." in name and idx is not None:
        layer_types = text.get("layer_types") or []
        if idx < len(layer_types) and layer_types[idx] == "linear_attention":
            return "target.kda"
        return "target.mla"
    if any(
        token in name
        for token in (
            "layernorm",
            ".norm.",
            ".norm.weight",
            ".hc_",
            ".eh_proj",
            ".hnorm",
            ".enorm",
        )
    ):
        return "target.norm_mhc_aux"
    return "target.unknown"


def routed_geometry(name: str, tp_size: int) -> tuple[float, str]:
    projection = ""
    for candidate in ("gate_proj", "up_proj", "down_proj"):
        if f".{candidate}." in name:
            projection = candidate
            break
    suffix = name.rsplit(".", 1)[-1]
    if suffix == "mcg":
        return 1.0, "replicated marker"
    if projection in {"gate_proj", "up_proj"}:
        if suffix in {"trellis", "svh"}:
            return 1 / tp_size, "column-sharded packed expert"
        return 1.0, "replicated packed expert metadata"
    if projection == "down_proj":
        if suffix in {"trellis", "suh"}:
            return 1 / tp_size, "row-sharded packed expert"
        return 1.0, "replicated packed expert metadata"
    return 1.0, "unknown routed tensor geometry"


def target_geometry(
    name: str, category: str, tp_size: int
) -> tuple[float, str]:
    if category == "target.mtp_unused_dflash":
        return 0.0, "not loaded by DFlash target path"
    if category == "target.unknown":
        return 1.0, "unknown target geometry"
    if tp_size == 1:
        return 1.0, "single rank"
    if category == "target.routed_experts":
        return routed_geometry(name, tp_size)
    if category in {"target.embedding", "target.lm_head"}:
        return 1 / tp_size, "vocabulary-parallel"
    if category == "target.vision":
        if any(
            token in name
            for token in (
                ".attn.qkv.weight",
                ".attn.proj.weight",
                ".mlp.gate_proj.weight",
                ".mlp.up_proj.weight",
                ".mlp.down_proj.weight",
                ".merger.proj.weight",
                ".merger.gate_proj.weight",
                ".merger.up_proj.weight",
                ".merger.down_proj.weight",
            )
        ):
            return 1 / tp_size, "vision tensor-parallel linear"
        return 1.0, "vision replicated/nonlinear"
    if any(token in name for token in (".f_a_proj.", ".g_a_proj.")):
        return 1.0, "replicated KDA fused shard"
    if category == "target.mla" and any(
        token in name
        for token in (
            ".q_a_proj.",
            ".kv_a_proj_with_mqa.",
            ".indexer.",
        )
    ):
        return 1.0, "replicated MLA/indexer projection"
    if name.endswith((".gate_proj.weight", ".up_proj.weight")):
        return 1 / tp_size, "column-sharded linear"
    if name.endswith(".down_proj.weight"):
        return 1 / tp_size, "row-sharded linear"
    if category in {"target.kda", "target.mla"} and name.endswith(".weight"):
        if any(
            token in name
            for token in (
                ".q_proj.",
                ".k_proj.",
                ".v_proj.",
                ".b_proj.",
                ".f_b_proj.",
                ".g_b_proj.",
                ".q_b_proj.",
                ".kv_b_proj.",
                ".o_proj.",
                "_conv1d.",
            )
        ):
            return 1 / tp_size, "attention tensor-parallel"
    return 1.0, "replicated"


def draft_category(name: str) -> str:
    if name.startswith("candidate_selector."):
        return "draft.selector"
    if name == "fc.weight" or any(
        token in name for token in (".self_attn.", ".mlp.")
    ):
        return "draft.backbone_linears"
    if any(
        token in name
        for token in (
            "attention_conv.base_kernel",
            "attention_conv.kernel_projection.weight",
            "mlp_conv.base_kernel",
            "mlp_conv.kernel_projection.weight",
        )
    ):
        return "draft.conv"
    if (
        name in {"hidden_norm.weight", "norm.weight"}
        or name.endswith(".input_layernorm.weight")
        or name.endswith(".post_attention_layernorm.weight")
    ):
        return "draft.norm"
    return "draft.unknown"


def read_license(snapshot: Path) -> str | None:
    card = snapshot / "README.md"
    if not card.is_file():
        return None
    text = card.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end < 0:
        return None
    frontmatter = text[4:end]
    match = re.search(r"(?m)^license:\s*([^\n#]+)", frontmatter)
    if not match:
        return None
    license_id = match.group(1).strip()
    if license_id.lower() != "other":
        return license_id
    name = re.search(r"(?m)^license_name:\s*([^\n#]+)", frontmatter)
    return name.group(1).strip() if name else license_id


def collect_target(snapshot: Path, tp_size: int) -> list[TensorInfo]:
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    result = []
    for name, entry in read_indexed_tensors(snapshot):
        category = target_category(name, config)
        fraction, geometry = target_geometry(name, category, tp_size)
        result.append(
            TensorInfo(
                name=name,
                dtype=str(entry.get("dtype", "unknown")),
                disk_bytes=tensor_bytes(entry),
                category=category,
                rank_fraction=fraction,
                geometry=geometry,
            )
        )
    return result


def collect_draft(snapshot: Path, draft_tp_size: int) -> list[TensorInfo]:
    result = []
    for name, entry in read_single_tensors(snapshot / "model.safetensors"):
        category = draft_category(name)
        result.append(
            TensorInfo(
                name=name,
                dtype=str(entry.get("dtype", "unknown")),
                disk_bytes=tensor_bytes(entry),
                category=category,
                rank_fraction=1 / draft_tp_size,
                geometry=(
                    "draft tensor-parallel"
                    if draft_tp_size > 1
                    else "draft rank 0 only"
                ),
            )
        )
    return result


def aggregate(tensors: list[TensorInfo]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"disk_bytes": 0, "rank_bytes": 0.0, "tensor_count": 0}
    )
    for tensor in tensors:
        group = groups[tensor.category]
        group["disk_bytes"] += tensor.disk_bytes
        group["rank_bytes"] += tensor.disk_bytes * tensor.rank_fraction
        group["tensor_count"] += 1
    return dict(sorted(groups.items()))


def routed_bytes_by_layer(
    tensors: list[TensorInfo],
) -> dict[int, dict[int, float]]:
    layers: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for tensor in tensors:
        if tensor.category != "target.routed_experts":
            continue
        match = re.search(r"\.layers\.(\d+)\..*\.experts\.(\d+)\.", tensor.name)
        if match:
            layers[int(match.group(1))][int(match.group(2))] += (
                tensor.disk_bytes * tensor.rank_fraction
            )
    return {layer: dict(experts) for layer, experts in layers.items()}


def estimate_traffic(
    target: list[TensorInfo],
    draft: list[TensorInfo],
    active_experts: list[int],
    accepted_per_step: float | None,
) -> dict[str, Any]:
    target_groups = aggregate(target)
    draft_groups = aggregate(draft)
    always_target = (
        "target.shared_experts",
        "target.dense_mlp",
        "target.kda",
        "target.mla",
        "target.router",
        "target.norm_mhc_aux",
    )
    base = sum(target_groups.get(name, {}).get("rank_bytes", 0.0) for name in always_target)
    base += 2 * target_groups.get("target.lm_head", {}).get("rank_bytes", 0.0)
    base += sum(group["rank_bytes"] for group in draft_groups.values())

    routed = routed_bytes_by_layer(target)
    scenarios = []
    for active in active_experts:
        routed_bytes = 0.0
        for experts in routed.values():
            sizes = sorted(experts.values(), reverse=True)
            routed_bytes += sum(sizes[:active])
        total = base + routed_bytes
        scenario = {
            "active_experts_per_moe_layer": active,
            "rank0_weight_bytes_per_verify_step": round(total),
            "routed_weight_bytes_per_verify_step": round(routed_bytes),
        }
        if accepted_per_step is not None:
            emitted = 1.0 + accepted_per_step
            scenario["estimated_weight_bytes_per_emitted_token"] = round(
                total / emitted
            )
            scenario["emitted_tokens_per_verify_step"] = emitted
        scenarios.append(scenario)
    return {
        "model": (
            "Each non-routed weight is counted once per verify step, the shared "
            "lm_head twice (draft plus verify), and routed experts once when active. "
            "Embeddings are sparse row lookups and excluded. This is not a DRAM "
            "counter measurement."
        ),
        "rank0_non_routed_and_draft_bytes_per_verify_step": round(base),
        "scenarios": scenarios,
    }


def permission_report(target_license: str | None, draft_license: str | None) -> dict[str, Any]:
    draft_blocked = bool(
        draft_license and "nc-nd" in draft_license.lower().replace("_", "-")
    )
    return {
        "target_license": target_license,
        "target_quantization": (
            "manual review required; preserve all base-model and pack attribution"
        ),
        "draft_license": draft_license,
        "draft_quantization": (
            "BLOCKED pending explicit derivative/commercial permission"
            if draft_blocked
            else "manual review required"
        ),
        "quality_gate": (
            "Target dense/head quantization remains blocked until the owner explicitly "
            "replaces the standing bit-exact gate with the task-specific quality gate."
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    target_snapshot = args.target_snapshot.resolve()
    draft_snapshot = args.draft_snapshot.resolve()
    target = collect_target(target_snapshot, args.tp_size)
    draft = collect_draft(draft_snapshot, args.draft_tp_size)
    target_groups = aggregate(target)
    draft_groups = aggregate(draft)
    rank_bytes = []
    for rank in range(args.tp_size):
        target_bytes = sum(item.disk_bytes * item.rank_fraction for item in target)
        draft_bytes = (
            sum(item.disk_bytes * item.rank_fraction for item in draft)
            if rank < args.draft_tp_size
            else 0.0
        )
        rank_bytes.append(round(target_bytes + draft_bytes))
    return {
        "target_snapshot": str(target_snapshot),
        "draft_snapshot": str(draft_snapshot),
        "tp_size": args.tp_size,
        "draft_tp_size": args.draft_tp_size,
        "target": {
            "disk_bytes": sum(item.disk_bytes for item in target),
            "categories": target_groups,
        },
        "draft": {
            "disk_bytes": sum(item.disk_bytes for item in draft),
            "categories": draft_groups,
        },
        "estimated_resident_weight_bytes_by_rank": rank_bytes,
        "traffic": estimate_traffic(
            target,
            draft,
            args.active_experts,
            args.accepted_per_step,
        ),
        "permissions": permission_report(
            read_license(target_snapshot), read_license(draft_snapshot)
        ),
        "unclassified": [
            asdict(item)
            for item in [*target, *draft]
            if item.category.endswith(".unknown")
            or item.geometry.startswith("unknown")
        ],
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def active_expert_list(value: str) -> list[int]:
    result = sorted({positive_int(part.strip()) for part in value.split(",")})
    if not result:
        raise argparse.ArgumentTypeError("must contain at least one count")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-snapshot", type=Path, required=True)
    parser.add_argument("--draft-snapshot", type=Path, required=True)
    parser.add_argument("--tp-size", type=positive_int, default=2)
    parser.add_argument("--draft-tp-size", type=positive_int, default=1)
    parser.add_argument(
        "--active-experts",
        type=active_expert_list,
        default=active_expert_list("8,16,32,64"),
        help="comma-separated unique routed experts per MoE layer",
    )
    parser.add_argument(
        "--accepted-per-step",
        type=float,
        help="accepted draft tokens per verify step; emitted tokens are 1 + this value",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.draft_tp_size > args.tp_size:
        parser.error("--draft-tp-size cannot exceed --tp-size")
    if args.accepted_per_step is not None and args.accepted_per_step < 0:
        parser.error("--accepted-per-step must be >= 0")
    report = build_report(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 2 if report["unclassified"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
