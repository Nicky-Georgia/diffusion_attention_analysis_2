from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


def looks_like_repo(path: Path) -> bool:
    return (path / "src" / "diffusion_attention_analysis_v2").exists() and (path / "configs").exists()


def find_project_root() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    candidates.extend([
        Path("/home/jupyter/project/diffusion_attention_analysis_2"),
        Path("/home/jupyter/datasphere/project/diffusion_attention_analysis_2"),
        Path("/mnt/data/diffzip/diffusion_attention_analysis_2"),
        Path("/mnt/data/diffzip/diffusion_attention_analysis_v2"),
    ])
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if looks_like_repo(candidate):
            return candidate
    raise FileNotFoundError("Cannot locate diffusion_attention_analysis_2 repository. Set PROJECT_ROOT manually.")


def first_existing(paths: Iterable[str | Path | None]) -> Path | None:
    for raw in paths:
        if not raw:
            continue
        p = Path(raw).expanduser()
        if p.exists():
            return p.resolve()
    return None


def find_container(name: str = "container4") -> Path | None:
    upper = name.upper()
    return first_existing([
        os.environ.get(upper),
        os.environ.get(f"{upper}_ROOT"),
        os.environ.get("SD3_FILESTORE") if name == "container4" else None,
        f"/home/jupyter/datasphere/filestore/{name}",
        f"/home/jupyter/filestore/{name}",
        f"/filestore/{name}",
    ])


def _safe_backup_path(path: Path) -> Path:
    base = path.with_name(path.name + ".project_backup")
    if not base.exists():
        return base
    return path.with_name(path.name + f".project_backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def _ensure_directory_or_backup(path: Path) -> None:
    """Ensure path is a directory; move legacy placeholder files away safely.

    Some DataSphere project archives contain small files named outputs or
    annotations with an old filestore path inside. pathlib.mkdir(...,
    exist_ok=True) still raises FileExistsError for such files, so we back
    them up before creating the directory used as a symlink parent.
    """

    if path.is_dir():
        return
    if path.exists() or path.is_symlink():
        backup_dir = path.parent / ".path_placeholders_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = backup_dir / f"{path.name}.{stamp}"
        path.rename(backup)
    path.mkdir(parents=True, exist_ok=True)


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    _ensure_directory_or_backup(link_path.parent)
    target_path.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        try:
            same_target = link_path.resolve(strict=True) == target_path.resolve(strict=True)
        except FileNotFoundError:
            same_target = False
        if not same_target:
            link_path.unlink()
            link_path.symlink_to(target_path, target_is_directory=True)
        return
    if link_path.exists():
        if not link_path.is_dir():
            backup_dir = link_path.parent / ".path_placeholders_backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / f"{link_path.name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            link_path.rename(backup)
        else:
            backup = _safe_backup_path(link_path)
            if any(link_path.iterdir()):
                if shutil.which("rsync"):
                    subprocess.run(["rsync", "-a", f"{link_path}/", f"{target_path}/"], check=True)
                else:
                    shutil.copytree(link_path, target_path, dirs_exist_ok=True)
            link_path.rename(backup)
    link_path.symlink_to(target_path, target_is_directory=True)


def prepare_datashere_storage(
    *,
    project_root: Path | None = None,
    run_tag: str = "sd3_medium_sae_circuit_150",
    container_name: str = "container4",
    require_container: bool = True,
    min_free_gb: float = 80.0,
) -> Dict[str, Any]:
    """Prepare Yandex DataSphere filestore paths for SD3 Medium experiments.

    Large artifacts are linked from project outputs/annotations into the selected
    filestore. HuggingFace, torch, tmp, and pip caches are also redirected there.
    """

    project_root = project_root or find_project_root()
    container = find_container(container_name)
    if container is None:
        msg = f"{container_name} was not found. Mount it or set {container_name.upper()}=/path/to/{container_name}."
        if require_container:
            raise FileNotFoundError(msg)
        container = project_root

    storage_root = container / project_root.name
    output_run_root = storage_root / "outputs" / run_tag
    annotation_run_root = storage_root / "annotations" / run_tag
    tmp_root = container / "tmp" / run_tag
    runs_root = storage_root / "runs" / run_tag

    for p in [output_run_root, annotation_run_root, tmp_root, runs_root]:
        p.mkdir(parents=True, exist_ok=True)

    hf_home = container / "hf_home"
    env_paths = {
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_home / "hub",
        "HUGGINGFACE_HUB_CACHE": hf_home / "hub",
        "HF_DATASETS_CACHE": hf_home / "datasets",
        "HF_XET_CACHE": hf_home / "xet",
        "TORCH_HOME": container / "torch_home",
        "TMPDIR": tmp_root,
        "TEMP": tmp_root,
        "TMP": tmp_root,
        "PIP_CACHE_DIR": container / "pip_cache",
    }
    for p in set(env_paths.values()):
        p.mkdir(parents=True, exist_ok=True)
    os.environ.update({k: str(v) for k, v in env_paths.items()})
    os.environ.update({
        "HF_HUB_DISABLE_XET": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PROJECT_ROOT": str(project_root),
        "RUN_TAG": run_tag,
        container_name.upper(): str(container),
    })
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.pop("DIFFUSERS_CACHE", None)

    ensure_symlink(project_root / "outputs" / run_tag, output_run_root)
    ensure_symlink(project_root / "annotations" / run_tag, annotation_run_root)

    free_gb = shutil.disk_usage(container).free / 1024**3
    return {
        "project_root": str(project_root),
        "container": str(container),
        "run_tag": run_tag,
        "output_run_root": str(output_run_root),
        "annotation_run_root": str(annotation_run_root),
        "tmp_root": str(tmp_root),
        "pip_cache_dir": os.environ["PIP_CACHE_DIR"],
        "hf_home": os.environ["HF_HOME"],
        "free_gb": free_gb,
        "min_free_gb": float(min_free_gb),
        "free_space_ok": free_gb >= float(min_free_gb),
    }


def directory_size_bytes(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def gib(x: float | int) -> float:
    return float(x) / 1024**3


def estimate_sd3_storage(
    *,
    prompt_count: int = 150,
    seeds: int = 1,
    capture_steps: Sequence[int] = (0, 13, 27),
    activation_modules: Sequence[str] = ("transformer_blocks.0", "transformer_blocks.11", "transformer_blocks.23"),
    image_tokens: int = 4096,
    text_tokens: int = 333,
    d_model: int = 1536,
    branches: int = 2,
    fp_bytes: int = 2,
    attention_heads: int = 24,
    generated_images: int | None = None,
    intervention_images: int = 288,
    model_cache_gb: float = 32.0,
    pip_cache_gb: float = 8.0,
    safety_margin_gb: float = 45.0,
) -> Dict[str, Any]:
    """Estimate container volume for the SD3 Medium SAE-circuit experiment.

    Primary artifacts are: (1) full head-averaged SD3 joint-attention probability maps for localization/circuit-edge diagnostics, and (2) image-token residual updates for SAE training and steering. Full maps scale quadratically in the joint token count, so headwise storage is estimated but disabled by default.

    Defaults are intentionally conservative for Yandex DataSphere: SD3 Medium,
    three text encoders, CLIPSeg, transient HuggingFace snapshots, pip wheels,
    and rerun buffers can coexist in the same filestore.
    """

    generated_images = generated_images if generated_images is not None else prompt_count * seeds
    n_activation_files = prompt_count * seeds * len(capture_steps) * len(activation_modules)
    residual_bytes_per_file = branches * image_tokens * d_model * fp_bytes
    residual_total = n_activation_files * residual_bytes_per_file

    joint_tokens = image_tokens + text_tokens
    image_to_all_probs_head_averaged_per_file = branches * image_tokens * joint_tokens * fp_bytes
    image_to_all_probs_headwise_per_file = branches * attention_heads * image_tokens * joint_tokens * fp_bytes
    full_joint_probs_head_averaged_per_file = branches * joint_tokens * joint_tokens * fp_bytes
    full_joint_probs_headwise_per_file = branches * attention_heads * joint_tokens * joint_tokens * fp_bytes
    attention_total = full_joint_probs_head_averaged_per_file * n_activation_files

    images_gb = generated_images * 1.8 / 1024  # empirical 1024px PNG/JPEG ballpark in GiB
    intervention_gb = intervention_images * 1.8 / 1024
    masks_gb = (prompt_count * 2) * 0.15 / 1024
    sae_ckpts_gb = 0.4
    reports_gb = 0.7
    artifacts_no_model_or_pip_gb = gib(residual_total) + gib(attention_total) + images_gb + intervention_gb + masks_gb + sae_ckpts_gb + reports_gb
    total_recommended_gb = artifacts_no_model_or_pip_gb + model_cache_gb + pip_cache_gb + safety_margin_gb

    return {
        "prompt_count": int(prompt_count),
        "seeds": int(seeds),
        "capture_steps": list(capture_steps),
        "activation_modules": list(activation_modules),
        "expected_activation_files": int(n_activation_files),
        "residual_stream_per_file_gib": gib(residual_bytes_per_file),
        "residual_stream_total_gib": gib(residual_total),
        "image_to_all_probability_maps_head_averaged_per_file_gib": gib(image_to_all_probs_head_averaged_per_file),
        "image_to_all_probability_maps_head_averaged_total_gib": gib(image_to_all_probs_head_averaged_per_file * n_activation_files),
        "image_to_all_probability_maps_headwise_per_file_gib": gib(image_to_all_probs_headwise_per_file),
        "image_to_all_probability_maps_headwise_total_gib": gib(image_to_all_probs_headwise_per_file * n_activation_files),
        "full_joint_probability_maps_head_averaged_per_file_gib": gib(full_joint_probs_head_averaged_per_file),
        "full_joint_probability_maps_head_averaged_total_gib": gib(full_joint_probs_head_averaged_per_file * n_activation_files),
        "full_joint_probability_maps_headwise_per_file_gib": gib(full_joint_probs_headwise_per_file),
        "full_joint_probability_maps_headwise_total_gib": gib(full_joint_probs_headwise_per_file * n_activation_files),
        "generated_images_gib_estimate": images_gb,
        "intervention_images_gib_estimate": intervention_gb,
        "masks_gib_estimate": masks_gb,
        "sae_checkpoints_gib_estimate": sae_ckpts_gb,
        "reports_gib_estimate": reports_gb,
        "artifacts_without_model_or_pip_gib": artifacts_no_model_or_pip_gb,
        "model_cache_gib_budget": model_cache_gb,
        "pip_cache_gib_budget": pip_cache_gb,
        "safety_margin_gib": safety_margin_gb,
        "recommended_container4_gib": total_recommended_gb,
        "recommended_container4_gib_rounded": math.ceil(total_recommended_gb / 10) * 10,
        "primary_artifact_choice": "full SD3 joint-attention probability maps + image-token residual updates",
        "attention_probability_maps_default": "enabled as head-averaged full joint maps; headwise disabled",
    }


SD3_LABELS = [
    "dog", "cat", "car", "book", "cup", "phone", "bicycle", "tree", "bird", "flower",
    "chair", "table", "sofa", "bench", "clock", "apple", "banana", "vase", "umbrella", "boat",
    "bus", "horse", "person", "window", "bed", "lamp", "backpack", "skateboard", "bridge", "newspaper",
]


def write_sd3_circuit_prompts(path: str | Path, *, n: int = 150, labels: Sequence[str] = SD3_LABELS, force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    relations = ["next to", "near", "beside", "under", "in front of", "behind"]
    places = ["inside a bright room", "on a city street", "in a small park", "near a window", "on a wooden table"]
    styles = ["realistic photo", "documentary photo", "studio photo", "natural light photo", "clear detailed photo"]
    rows = []
    m = len(labels)
    for i in range(n):
        a = labels[i % m]
        b = labels[(i * 7 + 11) % m]
        if a == b:
            b = labels[(i + 1) % m]
        prompt = f"{styles[i % len(styles)]} of a {a} {relations[i % len(relations)]} a {b} {places[(i // len(styles)) % len(places)]}"
        rows.append({
            "id": f"sd3_circuit_{i:03d}_{a}_{b}",
            "prompt": prompt,
            "target_word": a,
            "entities": [{"label": a, "token": a}, {"label": b, "token": b}],
        })
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return path


def safe_file_part(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "item"


def write_label_probe_prompts(source_prompts: str | Path, out_dir: str | Path, labels: Sequence[str], *, prompts_per_label: int = 2) -> Dict[str, str]:
    rows = read_jsonl(source_prompts)
    out_dir = Path(out_dir)
    out: Dict[str, str] = {}
    for label in labels:
        selected = []
        for row in rows:
            entities = row.get("entities", []) or []
            tokens = {str(e.get("token", e.get("label", ""))).lower() for e in entities}
            labels_in_row = {str(e.get("label", e.get("token", ""))).lower() for e in entities}
            if label.lower() in tokens or label.lower() in labels_in_row:
                r = dict(row)
                r["target_word"] = label
                selected.append(r)
            if len(selected) >= prompts_per_label:
                break
        if selected:
            path = out_dir / f"probe_{safe_file_part(label)}.jsonl"
            write_jsonl(path, selected)
            out[label] = str(path)
    return out


def activation_inventory(capture_dir: str | Path, modules: Sequence[str], steps: Sequence[int]) -> Dict[str, Any]:
    from .io_utils import module_to_filename

    capture_dir = Path(capture_dir)
    sample_dirs = sorted([p for p in capture_dir.glob("*") if p.is_dir()])
    rows = []
    missing = []
    for sample_dir in sample_dirs:
        for step in steps:
            step_dir = sample_dir / "activations" / f"step_{int(step):03d}"
            for module in modules:
                path = step_dir / module_to_filename(module)
                item = {"sample_id": sample_dir.name, "step": int(step), "module": module, "path": str(path), "exists": path.exists()}
                rows.append(item)
                if not path.exists():
                    missing.append(item)
    return {"samples": len(sample_dirs), "rows": rows, "expected": len(rows), "actual": sum(r["exists"] for r in rows), "missing": missing}



def attention_inventory(capture_dir: str | Path, layers: Sequence[str], steps: Sequence[int]) -> Dict[str, Any]:
    from .io_utils import module_to_filename

    capture_dir = Path(capture_dir)
    sample_dirs = sorted([p for p in capture_dir.glob("*") if p.is_dir()])
    rows = []
    missing = []
    for sample_dir in sample_dirs:
        for step in steps:
            step_dir = sample_dir / "attention" / f"step_{int(step):03d}"
            for layer in layers:
                path = step_dir / module_to_filename(layer)
                item = {"sample_id": sample_dir.name, "step": int(step), "layer": layer, "path": str(path), "exists": path.exists()}
                rows.append(item)
                if not path.exists():
                    missing.append(item)
    return {"samples": len(sample_dirs), "rows": rows, "expected": len(rows), "actual": sum(r["exists"] for r in rows), "missing": missing}

def feature_specificity(concept_csv: str | Path):
    import pandas as pd

    df = pd.read_csv(concept_csv)
    if df.empty:
        return df
    df = df.copy()
    df["label"] = df["label"].astype(str)
    reuse = df.groupby("feature_id")["label"].nunique().rename("reuse_count")
    df = df.merge(reuse, on="feature_id", how="left")
    df["label_rank"] = df.groupby("label")["score"].rank(method="first", ascending=False).astype(int)
    df["specificity_score"] = df["score"] / np.log2(df["reuse_count"].astype(float) + 1.0)
    return df.sort_values(["label", "specificity_score"], ascending=[True, False])


def select_features_for_labels(concept_csv: str | Path, labels: Sequence[str], *, per_label: int = 4, max_reuse: int = 12, min_score: float = 0.0) -> Dict[str, List[int]]:
    df = feature_specificity(concept_csv)
    out: Dict[str, List[int]] = {}
    for label in labels:
        sub = df[(df["label"].str.lower() == label.lower()) & (df["score"] >= min_score)].copy()
        strict = sub[sub["reuse_count"] <= max_reuse]
        if strict.empty:
            strict = sub
        out[label] = [int(x) for x in strict.sort_values("specificity_score", ascending=False)["feature_id"].head(per_label)]
    return out


def _base_sample_id(sample_id: str) -> str | None:
    if "_seed_" not in str(sample_id):
        return None
    prefix, rest = str(sample_id).split("_seed_", 1)
    seed = rest.split("_", 1)[0]
    return f"{prefix}_seed_{seed}"


def image_diff_metrics(a_path: str | Path, b_path: str | Path, *, size=(256, 256)) -> Dict[str, float]:
    from PIL import Image

    a = Image.open(a_path).convert("RGB").resize(size)
    b = Image.open(b_path).convert("RGB").resize(size)
    x = np.asarray(a).astype(np.float32) / 255.0
    y = np.asarray(b).astype(np.float32) / 255.0
    diff = np.abs(y - x)
    diff_gray = diff.mean(axis=2)
    return {
        "l1_mean": float(diff.mean()),
        "rmse": float(np.sqrt(((y - x) ** 2).mean())),
        "changed_px_005": float((diff_gray > 0.05).mean()),
        "changed_px_010": float((diff_gray > 0.10).mean()),
        "changed_px_020": float((diff_gray > 0.20).mean()),
    }


def collect_intervention_diffs(stage_dir: str | Path, baseline_samples_dir: str | Path, *, image_name: str, kind: str, extra: Mapping[str, Any] | None = None):
    import pandas as pd

    stage_dir = Path(stage_dir)
    baseline_samples_dir = Path(baseline_samples_dir)
    rows_path = stage_dir / "intervention_rows.json"
    if not rows_path.exists():
        return pd.DataFrame()
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        sample_id = row.get("sample_id")
        base_id = _base_sample_id(sample_id or "")
        if not base_id:
            continue
        baseline = baseline_samples_dir / base_id / "image.png"
        edited = stage_dir / "samples" / sample_id / image_name
        if not baseline.exists() or not edited.exists():
            continue
        item = {
            "kind": kind,
            "sample_id": sample_id,
            "base_id": base_id,
            "prompt": row.get("prompt", ""),
            "baseline_path": str(baseline),
            "edited_path": str(edited),
            "beta": row.get("beta", np.nan),
            "scale": row.get("scale", np.nan),
            "concept_ids": row.get("concept_ids", []),
            "module": (row.get("requested_modules") or [row.get("module") or None])[0],
        }
        if extra:
            item.update(extra)
        item.update(image_diff_metrics(baseline, edited))
        out.append(item)
    return pd.DataFrame(out)


def make_image_grid(image_paths: Sequence[str | Path], titles: Sequence[str] | None = None, *, ncols: int = 4, thumb=(256, 256)):
    from PIL import Image, ImageDraw

    paths = [Path(p) for p in image_paths if Path(p).exists()]
    if not paths:
        return None
    titles = list(titles or [p.name for p in paths])
    ncols = max(1, int(ncols))
    title_h = 38
    nrows = math.ceil(len(paths) / ncols)
    canvas = Image.new("RGB", (ncols * thumb[0], nrows * (thumb[1] + title_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(paths):
        r, c = divmod(idx, ncols)
        img = Image.open(path).convert("RGB")
        img.thumbnail(thumb)
        x = c * thumb[0]
        y = r * (thumb[1] + title_h)
        draw.text((x + 6, y + 4), str(titles[idx])[:42], fill=(0, 0, 0))
        canvas.paste(img, (x + (thumb[0] - img.width) // 2, y + title_h + (thumb[1] - img.height) // 2))
    return canvas
