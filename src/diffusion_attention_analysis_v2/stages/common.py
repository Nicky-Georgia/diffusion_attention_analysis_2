from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..config import describe_plan, output_dir, resolve_path, stage_disabled
from ..io_utils import module_to_filename, save_json


def save_plan(cfg: Dict[str, Any], plan: Dict[str, Any]) -> None:
    save_json(output_dir(cfg) / "plan.json", plan)


def dry_run_response(cfg: Dict[str, Any], progress, *, logic_check: Dict[str, Any] | None = None) -> Dict[str, Any]:
    plan = describe_plan(cfg)
    if logic_check:
        plan["logic_check"] = logic_check
    save_plan(cfg, plan)
    progress.start(total=1, message=f"dry-run {plan['stage']['name']} plan")
    progress.end(message="dry-run complete", plan=plan)
    return {"status": "dry_run", "plan": plan}


def skip_disabled(cfg: Dict[str, Any], progress) -> Dict[str, Any]:
    out = output_dir(cfg)
    report = {"status": "disabled", "reason": cfg.get("stage", {}).get("skip_reason", "disabled by config")}
    save_json(out / "report.json", report)
    progress.skip(message="stage disabled", reason=report["reason"])
    return report


def maybe_disabled(cfg: Dict[str, Any], progress):
    if stage_disabled(cfg):
        return skip_disabled(cfg, progress)
    return None


def list_activation_paths(capture_dir: Path, *, layer: str, step: int, max_files: int | None = None) -> List[Path]:
    filename = module_to_filename(layer)
    paths = sorted(capture_dir.glob(f"*/activations/step_{int(step):03d}/{filename}"))
    if not paths:
        # fallback for substring matches if actual Diffusers module names differ slightly
        safe_fragment = filename.replace(".pt", "")
        paths = sorted(capture_dir.glob(f"*/activations/step_{int(step):03d}/*.pt"))
        paths = [p for p in paths if safe_fragment in p.stem or layer in p.stem.replace("__", ".")]
    if max_files is not None:
        paths = paths[: int(max_files)]
    return paths


def list_attention_paths(capture_dir: Path, *, step: int, layer_files: Iterable[str] | None = None) -> List[Path]:
    if layer_files:
        paths: List[Path] = []
        for layer_file in layer_files:
            filename = layer_file if str(layer_file).endswith(".pt") else module_to_filename(str(layer_file))
            paths.extend(sorted(capture_dir.glob(f"*/attention/step_{int(step):03d}/{filename}")))
        return paths
    return sorted(capture_dir.glob(f"*/attention/step_{int(step):03d}/*.pt"))
