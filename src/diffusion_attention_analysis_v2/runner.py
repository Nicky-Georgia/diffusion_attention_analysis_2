from __future__ import annotations

from typing import Any, Dict, Iterable

from .config import apply_overrides, deep_get, load_config
from .progress import JsonProgress


def _stage_module(stage_name: str):
    if stage_name == "capture":
        from .stages import capture as module
    elif stage_name == "attention_localization":
        from .stages import evaluate_attention as module
    elif stage_name == "sae_training":
        from .stages import sae_train as module
    elif stage_name == "concept_dictionary":
        from .stages import concept_dictionary as module
    elif stage_name == "intervention":
        from .stages import interventions as module
    elif stage_name == "comparison":
        from .stages import comparison as module
    else:
        raise ValueError(f"Unknown stage name: {stage_name}")
    return module


def run_config(config_path: str, *, dry_run: bool = False, overrides: Iterable[str] | None = None, echo: bool = True) -> Dict[str, Any]:
    cfg = load_config(config_path)
    apply_overrides(cfg, overrides)
    stage_id = deep_get(cfg, "stage.id", config_path)
    progress = JsonProgress(stage_id=stage_id)
    module = _stage_module(str(deep_get(cfg, "stage.name")))
    result = module.run(cfg, progress, dry_run=dry_run)
    if echo:
        progress.emit("result", result=result)
    return result
