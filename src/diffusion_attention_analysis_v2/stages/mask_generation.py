from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_mask_stem(text: str) -> str:
    from ..io_utils import safe_name

    return safe_name(str(text).strip().lower())


def _target_records(metadata: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return segmentation targets whose token keys match Stage-2 token maps."""
    masking = cfg.get("masking", {}) or {}
    include = {str(x).lower() for x in masking.get("include_labels", []) or []}
    exclude = {str(x).lower() for x in masking.get("exclude_labels", []) or []}
    text_template = str(masking.get("text_template", "{label}"))

    row = metadata.get("prompt_record", {}) or {}
    records: List[Dict[str, str]] = []

    def add(label: str | None, token: str | None = None, mask_prompt: str | None = None) -> None:
        if not label and not token:
            return
        label = str(label or token).strip().lower()
        token = str(token or label).strip().lower()
        if not label or not token:
            return
        if include and token not in include and label not in include:
            return
        if token in exclude or label in exclude:
            return
        query = str(mask_prompt or text_template.format(label=label, token=token)).strip()
        records.append({"label": label, "token": token, "query": query})

    for ent in row.get("entities", []) or []:
        if isinstance(ent, dict):
            add(
                label=ent.get("label") or ent.get("name") or ent.get("token"),
                token=ent.get("token") or ent.get("label") or ent.get("name"),
                mask_prompt=ent.get("mask_prompt") or ent.get("query"),
            )
        else:
            add(label=str(ent), token=str(ent))

    for key in ("target_word", "target", "label"):
        if row.get(key):
            add(label=str(row[key]), token=str(row[key]))

    out: List[Dict[str, str]] = []
    seen = set()
    for rec in records:
        if rec["token"] in seen:
            continue
        seen.add(rec["token"])
        out.append(rec)
    return out


def _iter_samples(samples_dir: Path, max_samples: int | None = None) -> List[Tuple[str, Path, Path]]:
    candidates = []
    for sample_dir in sorted(samples_dir.glob("*")):
        if not sample_dir.is_dir():
            continue
        image_path = sample_dir / "image.png"
        metadata_path = sample_dir / "metadata.json"
        if image_path.exists() and metadata_path.exists():
            candidates.append((sample_dir.name, image_path, metadata_path))
    if max_samples is not None:
        candidates = candidates[: int(max_samples)]
    return candidates


def _to_binary_mask(score, *, relative_threshold: float, min_area_fraction: float, max_area_fraction: float, fallback_top_fraction: float):
    import numpy as np

    score = np.asarray(score, dtype=np.float32)
    lo = float(np.nanmin(score))
    hi = float(np.nanmax(score))
    if hi > lo:
        norm = (score - lo) / (hi - lo)
    else:
        norm = np.zeros_like(score, dtype=np.float32)

    mask = norm >= float(relative_threshold)
    coverage = float(mask.mean())
    if coverage < float(min_area_fraction) or coverage > float(max_area_fraction):
        q = max(0.0, min(1.0, 1.0 - float(fallback_top_fraction)))
        cutoff = float(np.quantile(norm, q))
        mask = norm >= cutoff
        coverage = float(mask.mean())
    return norm, mask, coverage


def _save_preview(image, mask, out_path: Path) -> None:
    from PIL import Image
    import numpy as np

    arr = np.array(image.convert("RGB")).astype(np.float32)
    m = np.asarray(mask).astype(bool)
    overlay = arr.copy()
    overlay[m, 0] = 255
    overlay[m, 1] = overlay[m, 1] * 0.35
    overlay[m, 2] = overlay[m, 2] * 0.35
    blended = (0.65 * arr + 0.35 * overlay).clip(0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended).save(out_path)


def _load_clipseg(masking_cfg: Dict[str, Any], device: str):
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    model_id = str(masking_cfg.get("model_id", "CIDAS/clipseg-rd64-refined"))
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    return processor, model


def _run_clipseg_batch(processor, model, image, queries: List[str], *, device: str):
    import torch
    import torch.nn.functional as F

    images = [image] * len(queries)
    inputs = processor(text=queries, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    width, height = image.size
    logits = F.interpolate(logits.unsqueeze(1).float(), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)
    probs = logits.sigmoid().detach().cpu().numpy()
    if probs.ndim == 2:
        probs = probs[None, :, :]
    return probs


def run(cfg: Dict[str, Any], progress, *, dry_run: bool = False) -> Dict[str, Any]:
    from ..config import deep_get, output_dir, resolve_path
    from ..io_utils import ensure_dir, load_json, save_json
    from .common import dry_run_response, maybe_disabled

    disabled = maybe_disabled(cfg, progress)
    if disabled is not None:
        return disabled

    logic = {
        "question": "Create binary pseudo-masks aligned to generated images for Stage-2 attention localization.",
        "method": "CLIPSeg text-conditioned segmentation per prompt entity; masks are saved under annotations/<model>/<sample_id>/<token>.png.",
        "guardrail": "A per-sample mask_manifest.json preserves token aliases when filenames are sanitized.",
    }
    if dry_run:
        return dry_run_response(cfg, progress, logic_check=logic)

    import numpy as np
    import pandas as pd
    from PIL import Image

    masking_cfg = cfg.get("masking", {}) or {}
    samples_dir = resolve_path(cfg, "data.samples_dir") or resolve_path(cfg, "data.sample_dir")
    out_dir = output_dir(cfg)
    if samples_dir is None or not samples_dir.exists():
        report = {"status": "skipped", "reason": "missing_samples_dir", "samples_dir": str(samples_dir)}
        save_json(out_dir / "report.json", report)
        progress.skip(message="missing samples dir", samples_dir=str(samples_dir))
        return report

    max_samples = masking_cfg.get("max_samples")
    max_samples = None if max_samples in (None, "", "null") else int(max_samples)
    samples = _iter_samples(samples_dir, max_samples=max_samples)
    if not samples:
        report = {"status": "skipped", "reason": "no_samples", "samples_dir": str(samples_dir)}
        save_json(out_dir / "report.json", report)
        progress.skip(message="no generated samples found", samples_dir=str(samples_dir))
        return report

    device = str(deep_get(cfg, "model.device", "cuda"))
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    processor, model = _load_clipseg(masking_cfg, device)
    relative_threshold = float(masking_cfg.get("relative_threshold", 0.55))
    min_area_fraction = float(masking_cfg.get("min_area_fraction", 0.002))
    max_area_fraction = float(masking_cfg.get("max_area_fraction", 0.80))
    fallback_top_fraction = float(masking_cfg.get("fallback_top_fraction", 0.12))
    force = bool(masking_cfg.get("force", False))
    save_soft_masks = bool(masking_cfg.get("save_soft_masks", False))
    save_previews = bool(masking_cfg.get("save_previews", True))
    preview_max_samples = int(masking_cfg.get("preview_max_samples", 24))

    rows: List[Dict[str, Any]] = []
    skipped = {"no_targets": 0, "existing": 0, "errors": 0}
    progress.start(total=len(samples), message="mask generation started", samples=len(samples))
    for i, (sample_id, image_path, metadata_path) in enumerate(samples, 1):
        try:
            metadata = load_json(metadata_path)
            targets = _target_records(metadata, cfg)
            if not targets:
                skipped["no_targets"] += 1
                continue
            sample_out = ensure_dir(out_dir / sample_id)
            image = Image.open(image_path).convert("RGB")
            queries = [t["query"] for t in targets]
            probs = _run_clipseg_batch(processor, model, image, queries, device=device)
            sample_manifest = {"sample_id": sample_id, "image_path": str(image_path), "metadata_path": str(metadata_path), "masks": []}
            for target, prob in zip(targets, probs):
                stem = _safe_mask_stem(target["token"])
                mask_path = sample_out / f"{stem}.png"
                if mask_path.exists() and not force:
                    skipped["existing"] += 1
                    existing_mask = np.array(Image.open(mask_path).convert("L")) > 127
                    item = {
                        "sample_id": sample_id,
                        "file": mask_path.name,
                        "label": target["label"],
                        "token": target["token"],
                        "query": target["query"],
                        "coverage": float(existing_mask.mean()),
                        "width": int(image.size[0]),
                        "height": int(image.size[1]),
                        "existing": True,
                    }
                    rows.append(item)
                    sample_manifest["masks"].append(item)
                    continue
                score, mask, coverage = _to_binary_mask(
                    prob,
                    relative_threshold=relative_threshold,
                    min_area_fraction=min_area_fraction,
                    max_area_fraction=max_area_fraction,
                    fallback_top_fraction=fallback_top_fraction,
                )
                Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
                if save_soft_masks:
                    soft_dir = ensure_dir(sample_out / "soft")
                    Image.fromarray((score.clip(0, 1) * 255).astype(np.uint8)).save(soft_dir / f"{stem}.png")
                if save_previews and i <= preview_max_samples:
                    _save_preview(image, mask, sample_out / "preview" / f"{stem}_overlay.png")
                item = {
                    "sample_id": sample_id,
                    "file": mask_path.name,
                    "label": target["label"],
                    "token": target["token"],
                    "query": target["query"],
                    "coverage": float(coverage),
                    "width": int(image.size[0]),
                    "height": int(image.size[1]),
                }
                rows.append(item)
                sample_manifest["masks"].append(item)
            save_json(sample_out / "mask_manifest.json", sample_manifest)
        except Exception as exc:
            skipped["errors"] += 1
            rows.append({"sample_id": sample_id, "error": repr(exc)})
        if i % 8 == 0 or i == len(samples):
            progress.update(current=i, message="samples masked")

    pd.DataFrame(rows).to_csv(out_dir / "mask_manifest.csv", index=False)
    ok_rows = [r for r in rows if "file" in r]
    coverage_values = [float(r["coverage"]) for r in ok_rows if "coverage" in r]
    report = {
        "status": "ok" if ok_rows else "skipped",
        "samples_seen": len(samples),
        "masks": len(ok_rows),
        "annotations_dir": str(out_dir),
        "skipped": skipped,
        "coverage_mean": float(np.mean(coverage_values)) if coverage_values else None,
        "coverage_min": float(np.min(coverage_values)) if coverage_values else None,
        "coverage_max": float(np.max(coverage_values)) if coverage_values else None,
        "model_id": str(masking_cfg.get("model_id", "CIDAS/clipseg-rd64-refined")),
    }
    save_json(out_dir / "report.json", report)
    progress.end(message="mask generation finished", masks=len(ok_rows), skipped=skipped)
    return report
