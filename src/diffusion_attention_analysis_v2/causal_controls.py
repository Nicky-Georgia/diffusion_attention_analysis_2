"""Auditable negative controls for SD3 SAE counterfactual interchange experiments."""
from __future__ import annotations

from typing import Sequence

import torch


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().square().mean().sqrt()


def require_nonzero_dose(
    applied: torch.Tensor,
    reference: torch.Tensor,
    *,
    min_ratio: float = 0.80,
    max_ratio: float = 1.20,
    epsilon: float = 1e-7,
) -> float:
    reference_rms = float(rms(reference))
    applied_rms = float(rms(applied))
    if reference_rms <= epsilon:
        raise ValueError("The semantic reference intervention has zero dose")
    ratio = applied_rms / reference_rms
    if not (min_ratio <= ratio <= max_ratio):
        raise ValueError(f"Unmatched negative control: dose ratio={ratio:.6g}")
    return ratio


def signflip_spatial_control(
    semantic_delta: torch.Tensor,
    *,
    seed: int,
    step: int,
) -> torch.Tensor:
    """Spatial sign-falsification control with identical per-token intervention L2 norms.

    This tests directional semantic consistency, NOT whether another SAE feature
    subset is semantically neutral. The caller must report this distinction.
    """
    if semantic_delta.ndim != 2:
        raise ValueError("Expected [image_tokens, hidden_dimension]")
    if float(rms(semantic_delta)) <= 1e-7:
        raise ValueError("Cannot form control from a zero semantic intervention")
    rng = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + int(step))
    signs = torch.randint(0, 2, (semantic_delta.shape[0], 1), generator=rng).mul(2).sub(1)
    out = semantic_delta * signs.to(device=semantic_delta.device, dtype=semantic_delta.dtype)
    require_nonzero_dose(out, semantic_delta, min_ratio=0.999, max_ratio=1.001)
    return out


def active_nonsemantic_features(
    source_latents: torch.Tensor,
    destination_latents: torch.Tensor,
    decoder_weight: torch.Tensor,
    *,
    forbidden: Sequence[int],
    count: int,
    epsilon: float = 1e-7,
) -> list[int]:
    """Choose non-target features with observed nonzero natural patch energy.

    Selection uses source/destination activations only, never evaluator scores.
    """
    if source_latents.shape != destination_latents.shape or source_latents.ndim != 2:
        raise ValueError("Expected matching [tokens, SAE_features] latents")
    if decoder_weight.shape[1] != source_latents.shape[1]:
        raise ValueError("Decoder feature dimension mismatch")
    delta = (source_latents - destination_latents).float()
    energy = delta.square().mean(dim=0).sqrt() * decoder_weight.float().norm(dim=0)
    for feature_id in forbidden:
        energy[int(feature_id)] = 0
    candidates = torch.nonzero(energy > epsilon, as_tuple=False).flatten()
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} non-target features have nonzero natural patch energy; requested {count}")
    chosen = torch.topk(energy, int(count)).indices.tolist()
    if any(x in set(forbidden) for x in chosen):
        raise AssertionError("Target feature leakage into active negative control")
    return chosen


def match_dose(delta: torch.Tensor, reference: torch.Tensor, *, epsilon: float = 1e-7) -> torch.Tensor:
    """Reject zero-dose feature sets; never claim that 0 times a scale is matched."""
    base = float(rms(delta))
    target = float(rms(reference))
    if base <= epsilon or target <= epsilon:
        raise ValueError(f"Cannot dose-match zero intervention: candidate={base}, semantic={target}")
    scaled = delta * (target / base)
    require_nonzero_dose(scaled, reference, min_ratio=0.999, max_ratio=1.001)
    return scaled


def weighted_latent_coverage(
    source_latents: torch.Tensor,
    destination_latents: torch.Tensor,
    decoder_weight: torch.Tensor,
    ranked_features: Sequence[int],
    k_values: Sequence[int],
) -> dict[int, float]:
    """Diagonal decoder-energy proxy, not an exact causal-mediation fraction."""
    if source_latents.shape != destination_latents.shape:
        raise ValueError("Latent shapes must match")
    delta = (source_latents - destination_latents).float()
    mass = delta.square() * decoder_weight.float().norm(dim=0).square()[None, :]
    total = float(mass.sum())
    if total <= 1e-12:
        raise ValueError("No counterfactual SAE signal in supplied tokens")
    return {int(k): float(mass[:, list(ranked_features[:int(k)])].sum() / total) for k in k_values}
