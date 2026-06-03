from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def _infer_hw(num_queries: int, meta: Dict[str, Any]) -> Tuple[int, int]:
    h = meta.get("height")
    w = meta.get("width")
    if h and w and int(h) * int(w) == int(num_queries):
        return int(h), int(w)
    side = int(int(num_queries) ** 0.5)
    if side * side == int(num_queries):
        return side, side
    return 1, int(num_queries)


def _load_mask(path: Path, size_hw):
    import numpy as np
    from PIL import Image

    h, w = size_hw
    arr = Image.open(path).convert("L").resize((w, h), Image.Resampling.NEAREST)
    return np.array(arr) > 127


def _transformer_diagnostics(cfg: Dict[str, Any], progress, *, dry_run: bool = False) -> Dict[str, Any]:
    from ..config import output_dir, resolve_path
    from ..io_utils import save_json
    from .common import dry_run_response

    logic = {
        "question": "Do selected transformer blocks show stable activation-scale patterns across denoising time?",
        "claim_scope": "diagnostic only; not an attention-localization score",
    }
    if dry_run:
        return dry_run_response(cfg, progress, logic_check=logic)

    import pandas as pd
    import torch

    out_dir = output_dir(cfg)
    capture_dir = resolve_path(cfg, "data.capture_dir")
    if capture_dir is None or not capture_dir.exists():
        report = {"status": "skipped", "reason": "missing_capture_dir", "capture_dir": str(capture_dir)}
        save_json(out_dir / "report.json", report)
        progress.skip(message="missing capture dir", capture_dir=str(capture_dir))
        return report

    max_files = cfg.get("metrics", {}).get("max_activation_files")
    paths = sorted(capture_dir.glob("*/activations/step_*/*.pt"))
    if max_files is not None:
        paths = paths[: int(max_files)]
    if not paths:
        report = {"status": "skipped", "reason": "no_activation_files"}
        save_json(out_dir / "report.json", report)
        progress.skip(message="no activation files found")
        return report

    rows = []
    progress.start(total=len(paths), message="transformer diagnostics started")
    for i, path in enumerate(paths, 1):
        payload = torch.load(path, map_location="cpu")
        x = payload["features"].float()
        rows.append({
            "sample_id": path.parents[2].name,
            "step": int(payload.get("step_index", -1)),
            "module": payload.get("module_name", path.stem.replace("__", ".")),
            "tokens": int(x.shape[0]),
            "d_model": int(x.shape[-1]),
            "mean_abs": float(x.abs().mean()),
            "std": float(x.std()),
            "l2_mean": float(x.norm(dim=-1).mean()),
        })
        if i % 32 == 0 or i == len(paths):
            progress.update(current=i, message="diagnostics processed")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "transformer_activation_rows.csv", index=False)
    summary = df.groupby(["module", "step"], as_index=False)[["mean_abs", "std", "l2_mean"]].mean()
    summary.to_csv(out_dir / "transformer_activation_summary.csv", index=False)
    report = {"status": "ok", "rows": len(rows), "summary_rows": int(len(summary)), "metric_suite": "transformer_short_diagnostics"}
    save_json(out_dir / "report.json", report)
    progress.end(message="transformer diagnostics finished", rows=len(rows))
    return report



def _sd3_joint_attention_localization(cfg: Dict[str, Any], progress, *, dry_run: bool = False) -> Dict[str, Any]:
    from ..config import deep_get, output_dir, resolve_path
    from ..io_utils import load_json, save_json
    from .common import dry_run_response, list_attention_paths

    logic = {
        "question": "Do SD3 full joint-attention probability maps localize prompt entities across denoising time?",
        "artifact": "Uses saved full q-by-k joint-attention maps, then evaluates the image-query -> text-key slice.",
        "guardrail": "Residual-stream SAE features are evaluated separately; this metric is only for attention probability maps.",
    }
    if dry_run:
        return dry_run_response(cfg, progress, logic_check=logic)

    import pandas as pd
    import torch

    from ..metrics import binary_iou, centroid_distance, normalized_entropy, summarize_numeric

    out_dir = output_dir(cfg)
    capture_dir = resolve_path(cfg, "data.capture_dir")
    annotations_dir = resolve_path(cfg, "data.annotations_dir")
    if capture_dir is None or not capture_dir.exists():
        report = {"status": "skipped", "reason": "missing_capture_dir", "capture_dir": str(capture_dir)}
        save_json(out_dir / "report.json", report)
        progress.skip(message="missing capture dir", capture_dir=str(capture_dir))
        return report
    if annotations_dir is None or not annotations_dir.exists():
        report = {"status": "skipped", "reason": "missing_annotations_dir", "annotations_dir": str(annotations_dir)}
        save_json(out_dir / "report.json", report)
        progress.skip(message="missing annotations", annotations_dir=str(annotations_dir))
        return report

    steps = [int(x) for x in deep_get(cfg, "metrics.steps", [])]
    layer_files = deep_get(cfg, "metrics.layer_files", [])
    threshold = float(deep_get(cfg, "metrics.threshold", 0.3))
    branch_mode = str(deep_get(cfg, "metrics.branch_mode", "cond_last"))
    text_token_offset_mode = str(deep_get(cfg, "metrics.text_token_offset", "after_image_tokens"))

    attn_paths = []
    for step in steps:
        attn_paths.extend(list_attention_paths(capture_dir, step=step, layer_files=layer_files))
    if not attn_paths:
        report = {"status": "skipped", "reason": "no_attention_files"}
        save_json(out_dir / "report.json", report)
        progress.skip(message="no attention files found")
        return report

    rows: List[Dict[str, Any]] = []
    skipped = {"missing_metadata": 0, "missing_annotation": 0, "missing_token": 0, "bad_shape": 0, "not_sd3_joint": 0}
    progress.start(total=len(attn_paths), message="SD3 joint-attention localization started")
    for i, path in enumerate(attn_paths, 1):
        payload = torch.load(path, map_location="cpu")
        meta = payload.get("meta", {})
        if not str(meta.get("attention_kind", "")).startswith("sd3_joint"):
            skipped["not_sd3_joint"] += 1
            continue
        sample_id = path.parents[2].name
        metadata_path = capture_dir / sample_id / "metadata.json"
        if not metadata_path.exists():
            skipped["missing_metadata"] += 1
            continue
        metadata = load_json(metadata_path)
        token_map = metadata.get("token_indices_by_word", {})
        sample_ann_dir = annotations_dir / sample_id
        if not sample_ann_dir.exists():
            skipped["missing_annotation"] += 1
            continue

        attention = payload["attention"].float()
        if attention.ndim == 4:  # [batch, heads, q, k]
            attention = attention.mean(dim=1)
        if attention.ndim != 3:
            skipped["bad_shape"] += 1
            continue
        if branch_mode == "cond_last":
            branch = attention.shape[0] - 1
        elif branch_mode == "uncond_first":
            branch = 0
        else:
            branch = min(max(int(branch_mode), 0), attention.shape[0] - 1)
        attn = attention[branch]
        n_img = int(meta.get("query_image_tokens") or meta.get("key_image_tokens") or 0)
        n_txt = int(meta.get("key_text_tokens") or 0)
        if n_img <= 0 or n_img > attn.shape[0] or n_img > attn.shape[1]:
            skipped["bad_shape"] += 1
            continue
        side = int(n_img ** 0.5)
        if side * side != n_img:
            skipped["bad_shape"] += 1
            continue
        text_offset = n_img if text_token_offset_mode == "after_image_tokens" else 0
        max_text_key = min(attn.shape[-1], text_offset + max(n_txt, 0))
        img_to_text = attn[:n_img, text_offset:max_text_key]
        if img_to_text.numel() == 0:
            skipped["bad_shape"] += 1
            continue

        for mask_path in sorted(sample_ann_dir.glob("*.png")):
            label = mask_path.stem.lower()
            token_indices = token_map.get(label, [])
            if not token_indices:
                skipped["missing_token"] += 1
                continue
            token_indices = [int(t) for t in token_indices if 0 <= int(t) < img_to_text.shape[-1]]
            if not token_indices:
                skipped["missing_token"] += 1
                continue
            score = img_to_text[:, token_indices].mean(dim=-1).numpy().reshape(side, side)
            mask = _load_mask(mask_path, (side, side))
            rows.append({
                "sample_id": sample_id,
                "label": label,
                "step": int(meta.get("step_index", -1)),
                "layer": meta.get("layer_name", path.stem),
                "attention_kind": meta.get("attention_kind", "sd3_joint_full"),
                "stored": meta.get("stored", "head_mean"),
                "branch": int(branch),
                "image_tokens": int(n_img),
                "text_tokens": int(n_txt),
                "iou": binary_iou(score, mask, threshold=threshold),
                "centroid_distance": centroid_distance(score, mask),
                "entropy": normalized_entropy(score),
                "num_token_indices": len(token_indices),
            })
        if i % 16 == 0 or i == len(attn_paths):
            progress.update(current=i, message="SD3 joint-attention files processed")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "sd3_joint_attention_localization_rows.csv", index=False)
        by_step = df.groupby("step", as_index=False)[["iou", "centroid_distance", "entropy"]].mean()
        by_step.to_csv(out_dir / "sd3_joint_attention_localization_by_step.csv", index=False)
        by_layer_step = df.groupby(["layer", "step"], as_index=False)[["iou", "centroid_distance", "entropy"]].mean()
        by_layer_step.to_csv(out_dir / "sd3_joint_attention_localization_by_layer_step.csv", index=False)
        summary = summarize_numeric(rows, ["iou", "centroid_distance", "entropy"])
    else:
        summary = {}
        pd.DataFrame().to_csv(out_dir / "sd3_joint_attention_localization_rows.csv", index=False)
    report = {
        "status": "ok" if rows else "skipped",
        "rows": len(rows),
        "attention_files": len(attn_paths),
        "skipped": skipped,
        "summary": summary,
        "metric_suite": "sd3_joint_attention_localization",
    }
    save_json(out_dir / "report.json", report)
    progress.end(message="SD3 joint-attention localization finished", rows=len(rows), skipped=skipped)
    return report

def run(cfg: Dict[str, Any], progress, *, dry_run: bool = False) -> Dict[str, Any]:
    from ..config import deep_get, output_dir, resolve_path
    from ..io_utils import load_json, save_json
    from .common import dry_run_response, list_attention_paths, maybe_disabled

    disabled = maybe_disabled(cfg, progress)
    if disabled is not None:
        return disabled

    metric_suite = deep_get(cfg, "metrics.metric_suite", "unet_attention_localization")
    if metric_suite == "sd3_joint_attention_localization":
        return _sd3_joint_attention_localization(cfg, progress, dry_run=dry_run)
    if metric_suite == "transformer_short_diagnostics" or deep_get(cfg, "model.family") == "transformer":
        return _transformer_diagnostics(cfg, progress, dry_run=dry_run)

    logic = {
        "question": "Do cross-attention maps align with object masks?",
        "guardrail": "Missing annotations/token matches are reported separately, not silently counted as failures.",
    }
    if dry_run:
        return dry_run_response(cfg, progress, logic_check=logic)

    import numpy as np
    import pandas as pd
    import torch

    from ..metrics import binary_iou, centroid_distance, normalized_entropy, summarize_numeric

    out_dir = output_dir(cfg)
    capture_dir = resolve_path(cfg, "data.capture_dir")
    annotations_dir = resolve_path(cfg, "data.annotations_dir")
    if capture_dir is None or not capture_dir.exists():
        report = {"status": "skipped", "reason": "missing_capture_dir", "capture_dir": str(capture_dir)}
        save_json(out_dir / "report.json", report)
        progress.skip(message="missing capture dir", capture_dir=str(capture_dir))
        return report
    if annotations_dir is None or not annotations_dir.exists():
        report = {"status": "skipped", "reason": "missing_annotations_dir", "annotations_dir": str(annotations_dir)}
        save_json(out_dir / "report.json", report)
        progress.skip(message="missing annotations", annotations_dir=str(annotations_dir))
        return report

    steps = [int(x) for x in deep_get(cfg, "metrics.steps", [])]
    layer_files = deep_get(cfg, "metrics.layer_files", [])
    threshold = float(deep_get(cfg, "metrics.threshold", 0.3))
    attn_paths = []
    for step in steps:
        attn_paths.extend(list_attention_paths(capture_dir, step=step, layer_files=layer_files))
    if not attn_paths:
        report = {"status": "skipped", "reason": "no_attention_files"}
        save_json(out_dir / "report.json", report)
        progress.skip(message="no attention files found")
        return report

    rows: List[Dict[str, Any]] = []
    skipped = {"missing_metadata": 0, "missing_annotation": 0, "missing_token": 0, "non_cross": 0}
    progress.start(total=len(attn_paths), message="attention localization started")
    for i, path in enumerate(attn_paths, 1):
        payload = torch.load(path, map_location="cpu")
        meta = payload.get("meta", {})
        if not meta.get("is_cross", True):
            skipped["non_cross"] += 1
            continue
        sample_id = path.parents[2].name
        metadata_path = capture_dir / sample_id / "metadata.json"
        if not metadata_path.exists():
            skipped["missing_metadata"] += 1
            continue
        metadata = load_json(metadata_path)
        token_map = metadata.get("token_indices_by_word", {})
        sample_ann_dir = annotations_dir / sample_id
        if not sample_ann_dir.exists():
            skipped["missing_annotation"] += 1
            continue
        attention = payload["attention"].float()
        if attention.ndim == 4:  # [batch, heads, q, k]
            attention = attention.mean(dim=1)
        branch = attention.shape[0] - 1  # cond branch for CFG; safe for batch=1 too
        attn = attention[branch]
        h, w = _infer_hw(attn.shape[0], meta)
        for mask_path in sorted(sample_ann_dir.glob("*.png")):
            label = mask_path.stem.lower()
            token_indices = token_map.get(label, [])
            if not token_indices:
                skipped["missing_token"] += 1
                continue
            token_indices = [int(t) for t in token_indices if 0 <= int(t) < attn.shape[-1]]
            if not token_indices:
                skipped["missing_token"] += 1
                continue
            score = attn[:, token_indices].mean(dim=-1).numpy().reshape(h, w)
            mask = _load_mask(mask_path, (h, w))
            rows.append({
                "sample_id": sample_id,
                "label": label,
                "step": int(meta.get("step_index", -1)),
                "layer": meta.get("layer_name", path.stem),
                "iou": binary_iou(score, mask, threshold=threshold),
                "centroid_distance": centroid_distance(score, mask),
                "entropy": normalized_entropy(score),
                "num_token_indices": len(token_indices),
            })
        if i % 32 == 0 or i == len(attn_paths):
            progress.update(current=i, message="attention files processed")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "attention_localization_rows.csv", index=False)
        by_step = df.groupby("step", as_index=False)[["iou", "centroid_distance", "entropy"]].mean()
        by_step.to_csv(out_dir / "attention_localization_by_step.csv", index=False)
        summary = summarize_numeric(rows, ["iou", "centroid_distance", "entropy"])
    else:
        summary = {}
        pd.DataFrame().to_csv(out_dir / "attention_localization_rows.csv", index=False)
    report = {"status": "ok" if rows else "skipped", "rows": len(rows), "skipped": skipped, "summary": summary}
    save_json(out_dir / "report.json", report)
    progress.end(message="attention localization finished", rows=len(rows), skipped=skipped)
    return report
