from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence

import torch
import torch.nn.functional as F

from .activations import extract_first_tensor, flatten_feature_map, replace_first_tensor, unflatten_feature_map


def _resize_flat_mask(mask: torch.Tensor, target_n: int) -> torch.Tensor:
    mask = mask.float()
    if mask.numel() == target_n:
        return mask.reshape(target_n)
    src_n = mask.numel()
    src_side = int(src_n ** 0.5)
    tgt_side = int(target_n ** 0.5)
    if src_side * src_side == src_n and tgt_side * tgt_side == target_n:
        m = mask.reshape(1, 1, src_side, src_side)
        m = F.interpolate(m, size=(tgt_side, tgt_side), mode="nearest")
        return m.reshape(target_n)
    # fallback: nearest index interpolation over the flattened dimension
    idx = torch.linspace(0, src_n - 1, target_n).long()
    return mask.reshape(-1)[idx]


@dataclass
class SpatialTokenReweighter:
    token_indices: Sequence[int]
    query_mask: torch.Tensor
    multiplier: float = 4.0
    outside_multiplier: float = 1.0
    renormalize: bool = True
    target_layers: Sequence[str] | None = None
    target_steps: Sequence[int] | None = None

    def __call__(self, *, layer_name: str, step_index: int, attention_probs: torch.Tensor, batch_size: int, num_heads: int, height: int | None, width: int | None) -> torch.Tensor:
        if self.target_layers and not any(fragment in layer_name for fragment in self.target_layers):
            return attention_probs
        if self.target_steps and int(step_index) not in {int(x) for x in self.target_steps}:
            return attention_probs
        token_indices = [int(i) for i in self.token_indices if 0 <= int(i) < attention_probs.shape[-1]]
        if not token_indices:
            return attention_probs
        probs = attention_probs.reshape(batch_size, num_heads, attention_probs.shape[-2], attention_probs.shape[-1])
        q = _resize_flat_mask(self.query_mask, probs.shape[-2]).to(device=probs.device, dtype=probs.dtype).reshape(1, 1, -1, 1)
        t = torch.zeros((1, 1, 1, probs.shape[-1]), device=probs.device, dtype=probs.dtype)
        t[..., token_indices] = 1.0
        boost = 1.0 + q * t * (float(self.multiplier) - 1.0)
        if self.outside_multiplier != 1.0:
            boost = boost + (1.0 - q) * t * (float(self.outside_multiplier) - 1.0)
        probs = probs * boost
        if self.renormalize:
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs.reshape_as(attention_probs)


def quadrant_mask(height: int, width: int, quadrant: str) -> torch.Tensor:
    mask = torch.zeros((height, width), dtype=torch.float32)
    h_mid = height // 2
    w_mid = width // 2
    if quadrant == "top_left":
        mask[:h_mid, :w_mid] = 1.0
    elif quadrant == "top_right":
        mask[:h_mid, w_mid:] = 1.0
    elif quadrant == "bottom_left":
        mask[h_mid:, :w_mid] = 1.0
    elif quadrant == "bottom_right":
        mask[h_mid:, w_mid:] = 1.0
    elif quadrant in {"all", "full"}:
        mask[:, :] = 1.0
    else:
        raise ValueError(f"Unknown quadrant: {quadrant}")
    return mask.reshape(-1)


class ResidualConceptSteerer:
    def __init__(self, sae, concept_ids: Sequence[int], beta: float, *, step_getter: Callable[[], int], active_steps: Sequence[int] | None = None, device: str = "cuda") -> None:
        self.sae = sae.to(device)
        self.concept_ids = [int(x) for x in concept_ids]
        self.beta = float(beta)
        self.step_getter = step_getter
        self.active_steps = {int(x) for x in active_steps or []}
        self.device = device

    def __call__(self, module, inputs, output):
        step = int(self.step_getter())
        if self.active_steps and step not in self.active_steps:
            return output
        x_in = extract_first_tensor(inputs)
        x_out = extract_first_tensor(output)
        if x_out is None:
            return output
        residual_mode = x_in is not None and tuple(x_in.shape) == tuple(x_out.shape)
        update = x_out - x_in if residual_mode else x_out
        try:
            flat, shape = flatten_feature_map(update.detach().float())
        except ValueError:
            return output
        flat = flat.to(self.device)
        with torch.no_grad():
            z = self.sae.encode(flat)
            valid = [i for i in self.concept_ids if 0 <= i < z.shape[-1]]
            if valid:
                z[:, valid] = z[:, valid] + self.beta
            steered = self.sae.decode(z)
        steered = unflatten_feature_map(steered, shape, like=x_out)
        new_tensor = x_in + steered if residual_mode else steered
        return replace_first_tensor(output, new_tensor, x_out.shape)


class BlockOutputScaler:
    def __init__(self, scale: float, *, step_getter: Callable[[], int], active_steps: Sequence[int] | None = None) -> None:
        self.scale = float(scale)
        self.step_getter = step_getter
        self.active_steps = {int(x) for x in active_steps or []}

    def __call__(self, module, inputs, output):
        step = int(self.step_getter())
        if self.active_steps and step not in self.active_steps:
            return output
        x_in = extract_first_tensor(inputs)
        x_out = extract_first_tensor(output)
        if x_out is None:
            return output
        if x_in is not None and tuple(x_in.shape) == tuple(x_out.shape):
            new_tensor = x_in + self.scale * (x_out - x_in)
        else:
            new_tensor = self.scale * x_out
        return replace_first_tensor(output, new_tensor, x_out.shape)
