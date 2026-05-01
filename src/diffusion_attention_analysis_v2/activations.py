from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import torch

from .io_utils import ensure_dir, module_to_filename


def extract_first_tensor(obj: Any):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            found = extract_first_tensor(item)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for value in obj.values():
            found = extract_first_tensor(value)
            if found is not None:
                return found
    return None


def replace_first_tensor(obj: Any, new_tensor: torch.Tensor, old_shape) -> Any:
    if isinstance(obj, torch.Tensor) and tuple(obj.shape) == tuple(old_shape):
        return new_tensor
    if isinstance(obj, tuple):
        changed = False
        out = []
        for item in obj:
            if not changed and isinstance(item, torch.Tensor) and tuple(item.shape) == tuple(old_shape):
                out.append(new_tensor)
                changed = True
            else:
                out.append(item)
        return tuple(out)
    if isinstance(obj, list):
        out = list(obj)
        for i, item in enumerate(out):
            if isinstance(item, torch.Tensor) and tuple(item.shape) == tuple(old_shape):
                out[i] = new_tensor
                break
        return out
    return obj


def flatten_feature_map(x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    if x.ndim == 4:
        b, c, h, w = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        return flat, [b, h, w, c]
    if x.ndim == 3:
        b, n, c = x.shape
        return x.reshape(b * n, c), [b, n, c]
    if x.ndim == 2:
        return x, list(x.shape)
    raise ValueError(f"Unsupported activation shape: {tuple(x.shape)}")


def unflatten_feature_map(flat: torch.Tensor, feature_shape: Sequence[int], *, like: torch.Tensor) -> torch.Tensor:
    if len(feature_shape) == 4:
        b, h, w, c = [int(x) for x in feature_shape]
        return flat.reshape(b, h, w, c).permute(0, 3, 1, 2).to(dtype=like.dtype, device=like.device)
    if len(feature_shape) == 3:
        b, n, c = [int(x) for x in feature_shape]
        return flat.reshape(b, n, c).to(dtype=like.dtype, device=like.device)
    if len(feature_shape) == 2:
        return flat.reshape(*feature_shape).to(dtype=like.dtype, device=like.device)
    raise ValueError(f"Unsupported feature_shape: {feature_shape}")


class ActivationStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = ensure_dir(root_dir)
        self.current_sample_id: str | None = None

    def begin_sample(self, sample_id: str) -> None:
        self.current_sample_id = sample_id

    def record(self, module_name: str, step_index: int, features: torch.Tensor, feature_shape: List[int]) -> None:
        if self.current_sample_id is None:
            raise RuntimeError("Call begin_sample before recording activations.")
        out_dir = ensure_dir(self.root_dir / self.current_sample_id / "activations" / f"step_{step_index:03d}")
        payload = {
            "module_name": module_name,
            "step_index": int(step_index),
            "feature_shape": [int(x) for x in feature_shape],
            "features": features.detach().cpu().half(),
        }
        torch.save(payload, out_dir / module_to_filename(module_name))


def resolve_module_names(model, requested: Iterable[str]) -> List[str]:
    available = dict(model.named_modules())
    resolved: List[str] = []
    for item in requested:
        if item in available:
            resolved.append(item)
        else:
            matches = [name for name in available if item in name]
            # Keep the match count bounded; configs should stay explicit.
            resolved.extend(matches[:8])
    seen = set()
    unique = []
    for name in resolved:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def register_activation_hooks(
    model,
    module_names: Iterable[str],
    store: ActivationStore,
    step_getter: Callable[[], int],
    capture_steps: Iterable[int],
):
    capture_steps = set(int(x) for x in capture_steps)
    resolved = resolve_module_names(model, module_names)
    modules = dict(model.named_modules())
    handles = []

    def make_hook(module_name: str):
        def hook(module, inputs, output):
            step = int(step_getter())
            if step not in capture_steps:
                return output
            x_in = extract_first_tensor(inputs)
            x_out = extract_first_tensor(output)
            if x_out is None:
                return output
            if x_in is not None and tuple(x_in.shape) == tuple(x_out.shape):
                update = x_out - x_in
            else:
                update = x_out
            try:
                flat, shape = flatten_feature_map(update.detach().float())
            except ValueError:
                return output
            store.record(module_name, step, flat, shape)
            return output
        return hook

    for name in resolved:
        handles.append(modules[name].register_forward_hook(make_hook(name)))
    return handles, resolved


def remove_hooks(handles: Iterable[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass
