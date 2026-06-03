from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def _load_mask(path: Path, size_hw):
    from PIL import Image

    h, w = size_hw
    arr = Image.open(path).convert("L").resize((w, h), Image.Resampling.NEAREST)
    return (np.array(arr) > 127).reshape(-1)


def run(cfg: Dict[str, Any], progress, *, dry_run: bool = False) -> Dict[str, Any]:
    from ..config import deep_get, output_dir, resolve_path
    from ..io_utils import save_json
    from .common import dry_run_response, list_activation_paths, maybe_disabled

    disabled = maybe_disabled(cfg, progress)
    if disabled is not None:
        return disabled

    logic = {
        "question": "Which SAE features align with object masks?",
        "control": "Features are ranked by inside-vs-outside activation contrast, not cherry-picked images.",
        "scope": "Supports U-Net spatial maps and SD3 image-token grids when the activation shape is square.",
    }
    if dry_run:
        return dry_run_response(cfg, progress, logic_check=logic)

    import pandas as pd
    import torch

    from ..sae import load_sae_checkpoint

    out_dir = output_dir(cfg)
    capture_dir = resolve_path(cfg, "data.capture_dir")
    annotations_dir = resolve_path(cfg, "data.annotations_dir")
    sae_checkpoint = resolve_path(cfg, "data.sae_checkpoint")
    if capture_dir is None or annotations_dir is None or sae_checkpoint is None or not capture_dir.exists() or not annotations_dir.exists() or not sae_checkpoint.exists():
        report = {
            "status": "skipped",
            "reason": "missing_inputs",
            "capture_dir": str(capture_dir),
            "annotations_dir": str(annotations_dir),
            "sae_checkpoint": str(sae_checkpoint),
        }
        save_json(out_dir / "report.json", report)
        progress.skip(message="missing concept dictionary inputs", **report)
        return report

    dic_cfg = cfg.get("dictionary", {})
    layer = str(dic_cfg["layer"])
    step = int(dic_cfg["step"])
    paths = list_activation_paths(capture_dir, layer=layer, step=step, max_files=dic_cfg.get("max_activation_files"))
    if not paths:
        report = {"status": "skipped", "reason": "no_activation_files", "layer": layer, "step": step}
        save_json(out_dir / "report.json", report)
        progress.skip(message="no activation files", layer=layer, step=step)
        return report

    device = str(deep_get(cfg, "model.device", "cuda"))
    sae = load_sae_checkpoint(sae_checkpoint, device=device)
    label_scores: Dict[str, torch.Tensor] = {}
    label_counts: Dict[str, int] = {}
    progress.start(total=len(paths), message="concept dictionary started")
    for i, path in enumerate(paths, 1):
        payload = torch.load(path, map_location="cpu")
        sample_id = path.parents[2].name
        sample_ann_dir = annotations_dir / sample_id
        if not sample_ann_dir.exists():
            continue
        x = payload["features"].float().to(device)
        feature_shape = payload.get("feature_shape", [])
        if len(feature_shape) == 4:
            b, h, w, c = [int(v) for v in feature_shape]
            branch = max(0, b - 1)
            start = branch * h * w
            end = start + h * w
            x_spatial = x[start:end]
            size_hw = (h, w)
        elif len(feature_shape) == 3:
            b, n, c = [int(v) for v in feature_shape]
            side = int(n ** 0.5)
            if side * side != n:
                continue
            branch = max(0, b - 1)
            start = branch * n
            end = start + n
            x_spatial = x[start:end]
            size_hw = (side, side)
        else:
            continue
        with torch.no_grad():
            z = sae.encode(x_spatial).detach().cpu()
        for mask_path in sorted(sample_ann_dir.glob("*.png")):
            label = mask_path.stem.lower()
            mask = torch.tensor(_load_mask(mask_path, size_hw), dtype=torch.bool)
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue
            inside = z[mask].mean(dim=0)
            outside = z[~mask].mean(dim=0)
            contrast = inside - outside
            label_scores[label] = label_scores.get(label, torch.zeros_like(contrast)) + contrast
            label_counts[label] = label_counts.get(label, 0) + 1
        if i % 32 == 0 or i == len(paths):
            progress.update(current=i, message="concept files processed")

    top_k = int(dic_cfg.get("top_k_features_per_label", 20))
    rows = []
    dictionary = {}
    for label, score_sum in label_scores.items():
        score = score_sum / max(label_counts[label], 1)
        values, indices = torch.topk(score, k=min(top_k, score.numel()))
        feats = [{"feature_id": int(idx), "score": float(val)} for val, idx in zip(values, indices)]
        dictionary[label] = {"count": label_counts[label], "top_features": feats}
        for item in feats:
            rows.append({"label": label, "count": label_counts[label], **item})
    pd.DataFrame(rows).to_csv(out_dir / "concept_dictionary.csv", index=False)
    save_json(out_dir / "concept_dictionary.json", dictionary)
    report = {"status": "ok" if rows else "skipped", "labels": len(dictionary), "rows": len(rows), "layer": layer, "step": step}
    save_json(out_dir / "report.json", report)
    progress.end(message="concept dictionary finished", labels=len(dictionary))
    return report
