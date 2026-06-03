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
    # SD3_ACTIVE_TOKEN_STEERING_V3:
    # Add SAE decoder directions to the original residual update.
    # Optional active-token mode localizes the edit to spatial tokens where the selected SAE features already activate.
    def __init__(self, sae, concept_ids: Sequence[int], beta: float, *, step_getter: Callable[[], int], active_steps: Sequence[int] | None = None, device: str = "cuda") -> None:
        self.sae = sae.to(device)
        self.concept_ids = [int(x) for x in concept_ids]
        self.beta = float(beta)
        self.step_getter = step_getter
        self.active_steps = {int(x) for x in active_steps or []}
        self.device = device
        import os as _os
        self.branch_mode = _os.environ.get("SD3_SAE_BRANCH_MODE", "cond").strip().lower()
        self.direction_mode = _os.environ.get("SD3_SAE_DIRECTION_MODE", "sum_decoder_direction").strip().lower()
        self.spatial_mode = _os.environ.get("SD3_SAE_SPATIAL_MODE", "all").strip().lower()
        self.top_frac = float(_os.environ.get("SD3_SAE_TOP_FRAC", "0.15"))
        self.active_quantile = float(_os.environ.get("SD3_SAE_ACTIVE_QUANTILE", "0.85"))
        self.outside_scale = float(_os.environ.get("SD3_SAE_OUTSIDE_SCALE", "0.0"))
        self.normalize_direction = _os.environ.get("SD3_SAE_NORMALIZE_DIRECTION", "none").strip().lower()
        self.beta_scale_mode = _os.environ.get("SD3_SAE_BETA_SCALE_MODE", "constant").strip().lower()

    def _valid_ids(self, max_dim: int) -> list[int]:
        return [i for i in self.concept_ids if 0 <= int(i) < int(max_dim)]

    def _direction(self, flat: torch.Tensor) -> torch.Tensor | None:
        with torch.no_grad():
            W = self.sae.decoder.weight.detach().to(device=flat.device, dtype=flat.dtype)
            valid = self._valid_ids(W.shape[1])
            if not valid:
                return None
            dirs = W[:, valid]
            if self.direction_mode in {"mean", "mean_decoder_direction"}:
                direction = dirs.mean(dim=1)
            else:
                direction = dirs.sum(dim=1)
            if self.normalize_direction in {"rms", "unit_rms"}:
                direction = direction / direction.pow(2).mean().sqrt().clamp_min(1e-6)
            elif self.normalize_direction in {"l2", "unit_l2"}:
                direction = direction / direction.norm().clamp_min(1e-6)
            return direction.reshape(1, -1)

    def _branch_mask(self, feature_shape: Sequence[int], flat: torch.Tensor) -> torch.Tensor:
        mask = torch.ones((flat.shape[0], 1), device=flat.device, dtype=flat.dtype)
        if len(feature_shape) != 3:
            return mask
        b, n, c = [int(x) for x in feature_shape]
        if b < 2 or b % 2 != 0:
            return mask
        branch_mode = self.branch_mode if self.branch_mode in {"cond", "conditional", "uncond", "unconditional", "all"} else "cond"
        if branch_mode == "all":
            return mask
        m = torch.zeros((b, n, 1), device=flat.device, dtype=flat.dtype)
        half = b // 2
        if branch_mode in {"cond", "conditional"}:
            m[half:] = 1.0
        else:
            m[:half] = 1.0
        return m.reshape(-1, 1)

    def _active_token_mask(self, flat: torch.Tensor, feature_shape: Sequence[int]) -> torch.Tensor:
        if self.spatial_mode in {"all", "global", "none"} or len(feature_shape) != 3:
            return torch.ones((flat.shape[0], 1), device=flat.device, dtype=flat.dtype)
        b, n, c = [int(x) for x in feature_shape]
        with torch.no_grad():
            z = self.sae.encode(flat)
            valid = self._valid_ids(z.shape[-1])
            if not valid:
                return torch.ones((flat.shape[0], 1), device=flat.device, dtype=flat.dtype)
            signal = z[:, valid].float().amax(dim=-1).reshape(b, n)
            mask = torch.full((b, n), float(self.outside_scale), device=flat.device, dtype=flat.dtype)
            if self.spatial_mode in {"top_frac", "active_top"}:
                k = max(1, min(n, int(round(float(self.top_frac) * n))))
                idx = torch.topk(signal, k=k, dim=1).indices
                mask.scatter_(1, idx, 1.0)
            else:
                q = float(self.active_quantile)
                q = min(max(q, 0.0), 0.999)
                thresh = torch.quantile(signal, q, dim=1, keepdim=True)
                active = signal >= thresh
                # Fallback if activations are nearly flat.
                need_fallback = active.sum(dim=1) < 1
                if need_fallback.any():
                    k = max(1, min(n, int(round(float(self.top_frac) * n))))
                    idx = torch.topk(signal[need_fallback], k=k, dim=1).indices
                    rows = torch.where(need_fallback)[0]
                    active[rows] = False
                    active[rows.unsqueeze(1), idx] = True
                mask[active] = 1.0
            return mask.reshape(-1, 1).to(dtype=flat.dtype)

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
        direction = self._direction(flat)
        if direction is None:
            return output
        with torch.no_grad():
            if self.beta_scale_mode in {"activation_rms", "token_rms"}:
                local_scale = flat.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
                delta = self.beta * local_scale * direction.expand_as(flat)
            else:
                delta = self.beta * direction.expand_as(flat)
            delta = delta * self._active_token_mask(flat, shape)
            delta = delta * self._branch_mask(shape, flat)
            steered_flat = flat + delta
        steered = unflatten_feature_map(steered_flat, shape, like=x_out)
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
