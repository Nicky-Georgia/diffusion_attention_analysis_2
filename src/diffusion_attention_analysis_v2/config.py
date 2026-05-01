from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping

import yaml


PROJECT_MARKERS = ("configs", "src", "notebooks")


def project_root(start: str | Path | None = None) -> Path:
    """Return the nearest parent that looks like the repository root."""
    p = Path(start or Path.cwd()).resolve()
    candidates = [p] + list(p.parents)
    for candidate in candidates:
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    return p


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = project_root() / path
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(project_root(path.parent))
    return cfg


def save_config(path: str | Path, cfg: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def deep_get(obj: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") or value.startswith("{"):
        try:
            return yaml.safe_load(value)
        except Exception:
            return value
    return value


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str] | None) -> Dict[str, Any]:
    """Apply CLI overrides like data.max_prompts=8 to a config dict in-place."""
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must have form key=value, got: {item}")
        key, raw = item.split("=", 1)
        value = _parse_scalar(raw)
        cur: MutableMapping[str, Any] = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})  # type: ignore[assignment]
            if not isinstance(cur, dict):
                raise ValueError(f"Cannot set override {key}: {part} is not a mapping")
        cur[parts[-1]] = value
    return cfg


def resolve_path(cfg: Dict[str, Any], dotted: str, *, required: bool = False) -> Path | None:
    value = deep_get(cfg, dotted)
    if value in (None, ""):
        if required:
            raise ValueError(f"Missing required path config: {dotted}")
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(cfg.get("_project_root", project_root())) / path
    return path


def output_dir(cfg: Dict[str, Any]) -> Path:
    out = resolve_path(cfg, "data.output_dir", required=True)
    assert out is not None
    out.mkdir(parents=True, exist_ok=True)
    return out


def stage_disabled(cfg: Dict[str, Any]) -> bool:
    return deep_get(cfg, "stage.enabled", True) is False


def describe_plan(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stage": {
            "id": deep_get(cfg, "stage.id", "unknown"),
            "name": deep_get(cfg, "stage.name", "unknown"),
            "description": deep_get(cfg, "stage.description", ""),
            "claim_level": deep_get(cfg, "stage.claim_level", "unknown"),
            "enabled": deep_get(cfg, "stage.enabled", True),
        },
        "model": {
            "name": deep_get(cfg, "model.name", "unknown"),
            "family": deep_get(cfg, "model.family", "unknown"),
            "pipeline_class": deep_get(cfg, "model.pipeline_class"),
            "model_id": deep_get(cfg, "model.model_id"),
        },
        "data": {
            "prompts_path": deep_get(cfg, "data.prompts_path"),
            "max_prompts": deep_get(cfg, "data.max_prompts"),
            "output_dir": deep_get(cfg, "data.output_dir"),
        },
        "expected_artifacts": deep_get(cfg, "stage.artifacts", []),
    }
