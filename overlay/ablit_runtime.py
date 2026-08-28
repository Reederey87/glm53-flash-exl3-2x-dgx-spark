# SPDX-License-Identifier: MIT
"""GLM-5.3-Flash o_proj abliteration (ABLIT) — load-time refusal-direction edit.

Applies the classic weight orthogonalization to every ``self_attn.o_proj`` in
the configured layer range:

    W' = (I - alpha * r r^T / ||r||^2) W

``r`` is the refusal direction in the residual/output space of o_proj
(hidden_size=4096). Components of the attention output orthogonal to ``r``
are preserved exactly; the component along ``r`` is scaled by ``1 - alpha``
(alpha > 1 over-projects, which is what the published recipe uses).

TP notes: o_proj is a RowParallelLinear — vLLM shards the *input* dim, the
4096 output rows are replicated on every rank. The edit only touches rows,
so each rank applies the identical formula to its own shard and the
post-allreduce result stays consistent. No collective is needed.

The edit runs at the end of ``Glm5NextModel.load_weights`` / ``Glm5NextMTP.
load_weights`` (installed by overlay/patch_ablit.py). o_proj stays native
BF16 in this serve (attn is unquantized), so the parameter is final at that
point — quantized-expert post-processing never touches it, and CUDA graph
capture happens later.

Artifacts (shipped in ``ablit/`` and mounted at /opt/glm53/ablit):
  LAYER_MAP.json                              layer/shard map + published recipe
  refusal_direction_glm53_dealign_late.pt     published dealign direction
  refusal_direction_glm53_bf_oproj.pt         blackfrost direction (alpha_ref 3.0)

Source: drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock
(published method "dealign-oproj-transplant": layers 15-45 edited, 0-14 kept
as stock safety anchors, MTP block included).

Env knobs (see .env.example):
  ABLIT=1                  enable (default off — hook is a no-op)
  ABLIT_DIR                artifact dir (default /opt/glm53/ablit)
  ABLIT_DIRECTION          dealign | bf_oproj | /path/to/dir.pt (default dealign)
  ABLIT_LAYERS             inclusive ranges, e.g. "15-45" or "15,17-19"
  ABLIT_ALPHA              projection scale (default 3.0)
  ABLIT_INCLUDE_MTP        also edit the checkpoint MTP block when it exists
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import torch

try:  # inside the vLLM image
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # standalone (tests)
    import logging

    logger = logging.getLogger("glm53_ablit")

DEFAULT_ABLIT_DIR = "/opt/glm53/ablit"

DIRECTION_FILES = {
    "dealign": "refusal_direction_glm53_dealign_late.pt",
    "bf_oproj": "refusal_direction_glm53_bf_oproj.pt",
}

# paths inside the hooked model that carry an o_proj to edit
_LAYER_O_PROJ_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.o_proj$")
_MTP_O_PROJ_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mtp_block\.self_attn\.o_proj$")


class AblitError(RuntimeError):
    """ABLIT was explicitly enabled but the edit cannot be applied."""


def env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def parse_layers(spec: str) -> list[int]:
    """Parse "15-45" / "15,17-19" into a sorted list of layer indices."""
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise AblitError(f"ABLIT_LAYERS: bad range {part!r}") from exc
            if hi < lo:
                raise AblitError(f"ABLIT_LAYERS: inverted range {part!r}")
            layers.update(range(lo, hi + 1))
        else:
            try:
                layers.add(int(part))
            except ValueError as exc:
                raise AblitError(f"ABLIT_LAYERS: bad index {part!r}") from exc
    if not layers:
        raise AblitError("ABLIT_LAYERS resolved to an empty set")
    return sorted(layers)


def resolve_direction_path(ablit_dir: str, direction: str) -> Path:
    if "/" in direction:  # explicit path to a .pt
        return Path(direction)
    fname = DIRECTION_FILES.get(direction)
    if fname is None:
        raise AblitError(
            f"ABLIT_DIRECTION={direction!r} not one of "
            f"{sorted(DIRECTION_FILES)} (or a .pt path)"
        )
    return Path(ablit_dir) / fname


def load_direction(path: Path) -> torch.Tensor:
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AblitError(f"cannot load ablit direction {path}: {exc}") from exc
    if not isinstance(obj, dict) or "directions" not in obj:
        raise AblitError(f"{path}: expected a dict with 'directions'")
    r = obj["directions"]
    if not torch.is_tensor(r):
        raise AblitError(f"{path}: 'directions' must be a tensor, got {type(r)}")
    if r.dim() == 2 and r.shape[0] == 1:  # some exports store a [1, N] row
        r = r.squeeze(0)
    if r.dim() != 1:
        raise AblitError(f"{path}: 'directions' must be 1-D (or [1, N]), got {tuple(r.shape)}")
    r = r.to(torch.float32)
    norm = r.norm()
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise AblitError(f"{path}: direction has non-positive/invalid norm {norm}")
    return r


def apply_to_o_proj(mod: Any, r: torch.Tensor, alpha: float) -> dict[str, Any]:
    """Orthogonalize one o_proj module in place. Returns a report dict."""
    weight = getattr(mod, "weight", None)
    if weight is None or not torch.is_tensor(weight) or weight.dim() != 2:
        return {"edited": False, "reason": "no 2-D .weight"}
    if weight.shape[0] != r.numel():
        return {
            "edited": False,
            "reason": f"out_features={weight.shape[0]} != direction dim {r.numel()}",
        }
    r_dev = r.to(device=weight.device, dtype=torch.float32)
    with torch.no_grad():
        w32 = weight.data.to(torch.float32)
        # W' = W - alpha * outer(r, r @ W) / ||r||^2
        rw = r_dev @ w32  # [in_local]
        delta = alpha * torch.outer(r_dev, rw) / float(r_dev @ r_dev)
        w32.sub_(delta)
        weight.data.copy_(w32.to(weight.dtype))
        residual = float((r_dev @ weight.data.to(torch.float32)).abs().max())
    return {
        "edited": True,
        "shape": tuple(weight.shape),
        "residual_max": residual,
    }


def walk_o_proj(model: Any) -> list[tuple[str, int | None, Any]]:
    """Collect (name, layer_idx or None-if-pure-mtp, module) o_proj candidates."""
    found: list[tuple[str, int | None, Any]] = []
    seen: set[int] = set()
    for name, mod in model.named_modules():
        if id(mod) in seen:
            continue
        m = _LAYER_O_PROJ_RE.search(name)
        if m is not None:
            seen.add(id(mod))
            found.append((name, int(m.group(1)), mod))
            continue
        m = _MTP_O_PROJ_RE.search(name)
        if m is not None:
            seen.add(id(mod))
            found.append((name, int(m.group(1)), mod))
    return found


def unwrap_text_model(model: Any) -> Any:
    """Accept the multimodal wrapper and hand back the text model."""
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", None)
        if inner is not None:
            return inner
        return lm
    return model


def apply_ablit(
    model: Any,
    r: torch.Tensor,
    layers: list[int],
    alpha: float,
    include_mtp: bool,
) -> dict[str, Any]:
    """Edit every configured o_proj under ``model``. Fails hard on surprises."""
    text_model = unwrap_text_model(model)
    candidates = walk_o_proj(text_model)
    want = set(layers)
    report: dict[str, Any] = {
        "edited_layers": [],
        "skipped": [],
        "mtp_edited": False,
    }
    seen_ids: set[int] = set()
    for name, idx, mod in candidates:
        if id(mod) in seen_ids:
            continue
        seen_ids.add(id(mod))
        is_mtp = _MTP_O_PROJ_RE.search(name) is not None
        if is_mtp:
            if not include_mtp:
                report["skipped"].append({"name": name, "reason": "include_mtp=0"})
                continue
            if idx not in want:
                report["skipped"].append(
                    {"name": name, "reason": f"layer {idx} not in ABLIT_LAYERS"}
                )
                continue
        else:
            if idx not in want:
                continue
        rep = apply_to_o_proj(mod, r, alpha)
        if rep.get("edited"):
            if is_mtp:
                report["mtp_edited"] = True
            else:
                report["edited_layers"].append(idx)
            logger.info(
                "ablit: orthogonalized %s shape=%s residual_max=%.3e",
                name,
                rep.get("shape"),
                rep.get("residual_max", float("nan")),
            )
        else:
            report["skipped"].append({"name": name, "reason": rep.get("reason")})
            logger.warning("ablit: skipped %s: %s", name, rep.get("reason"))
    report["edited_layers"].sort()
    return report


def maybe_apply(model: Any) -> dict[str, Any] | None:
    """Hook entrypoint. No-op unless ABLIT=1; raises AblitError if enabled
    but the recipe cannot be honored (fail loud beats silent stock weights)."""
    if not env_flag("ABLIT", False):
        return None

    ablit_dir = os.environ.get("ABLIT_DIR") or DEFAULT_ABLIT_DIR
    direction = os.environ.get("ABLIT_DIRECTION") or "dealign"
    layers_spec = os.environ.get("ABLIT_LAYERS") or "15-45"
    include_mtp = env_flag("ABLIT_INCLUDE_MTP", True)

    try:
        alpha = float(os.environ.get("ABLIT_ALPHA") or "3.0")
    except ValueError as exc:
        raise AblitError(f"ABLIT_ALPHA is not a number: {exc}") from exc
    if not math.isfinite(alpha) or alpha <= 0:
        raise AblitError(f"ABLIT_ALPHA must be a positive finite number, got {alpha}")

    path = resolve_direction_path(ablit_dir, direction)
    if not path.is_file():
        raise AblitError(f"ABLIT=1 but direction file missing: {path}")
    r = load_direction(path)

    layers = parse_layers(layers_spec)
    report = apply_ablit(model, r, layers, alpha, include_mtp)

    if not report["edited_layers"] and not report["mtp_edited"]:
        raise AblitError(
            "ABLIT=1 but no o_proj was edited — ABLIT_LAYERS="
            f"{layers_spec} matched nothing under this model"
        )
    logger.info(
        "ablit: ON direction=%s (%s) alpha=%s layers=%s edited=%s mtp=%s "
        "(skipped=%d) — early safety-anchor layers stay stock",
        direction,
        path.name,
        alpha,
        layers_spec,
        report["edited_layers"],
        report["mtp_edited"],
        len(report["skipped"]),
    )
    return report
