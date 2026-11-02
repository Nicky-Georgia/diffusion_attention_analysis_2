from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sae import TopKSparseAutoencoder


class TemporalTopKSAE(TopKSparseAutoencoder):
    """Continuous timestep-conditioned encoder. Decoder remains in original residual coordinates."""

    def __init__(self, d_model=1536, expansion_factor=4, k=32):
        super().__init__(d_model=d_model, expansion_factor=expansion_factor, k=k)
        self.time_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 2 * d_model))
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def encode(self, x, step=13):
        t = torch.full((1, 1), float(step) / 27.0, device=x.device, dtype=x.dtype)
        scale, shift = self.time_mlp(t).chunk(2, dim=-1)
        return super().encode(x * (1 + scale) + shift)

    def forward(self, x, step=13):
        z = self.encode(x, step=step)
        return self.decode(z), z


def encode_at(model, x, step):
    if isinstance(model, TemporalTopKSAE):
        return model.encode(x, step=step)
    return model.encode(x)


def load_research_sae(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]
    p = meta["sae"]
    cls = TemporalTopKSAE if p.get("architecture") == "temporal" else TopKSparseAutoencoder
    sae = cls(d_model=int(meta["d_model"]), expansion_factor=int(p["expansion_factor"]), k=int(p["k"]))
    sae.load_state_dict(ckpt["state_dict"], strict=True)
    return sae.to(device).eval(), meta


def capture_files(capture_dir, layer="transformer_blocks.11", steps=(13,)):
    name = layer.replace(".", "__") + ".pt"
    root = Path(capture_dir)
    found = []
    for p in sorted(root.rglob(name)):
        if re.search(r"step_(\d+)", str(p.parent)) and int(p.parent.name.split("_")[-1]) in set(map(int, steps)):
            found.append(p)
    if not found:
        raise FileNotFoundError(f"No capture activation tensors for {name}, steps={steps}, root={root}")
    return found


def group_file_split(paths, val_fraction=0.20, seed=2601):
    """Hold out whole capture samples, never individual image tokens."""
    groups = {}
    for path in paths:
        key = path.parent.parent.parent.name
        groups.setdefault(key, []).append(path)
    keys = sorted(groups)
    if len(keys) < 10:
        raise ValueError("At least 10 independently captured sample groups are required")
    rng = random.Random(seed)
    rng.shuffle(keys)
    nval = max(2, int(round(len(keys) * val_fraction)))
    valkeys = set(keys[:nval])
    train = [x for k, v in groups.items() if k not in valkeys for x in v]
    val = [x for k, v in groups.items() if k in valkeys for x in v]
    if set(train) & set(val):
        raise AssertionError("Train/validation leakage")
    return train, val


def load_tokens(path, max_tokens=512, sample_seed=2601):
    d = torch.load(path, map_location="cpu", weights_only=False)
    x = d["features"].detach().float().reshape(-1, d["features"].shape[-1])
    if x.shape[1] != 1536:
        raise ValueError(f"Not SD3 image-token residual features: {path} {x.shape}")
    if max_tokens and x.shape[0] > max_tokens:
        h = int(hashlib.sha256(str(path).encode()).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed((sample_seed + h) % (2**31))
        x = x[torch.randperm(x.shape[0], generator=g)[:max_tokens]]
    if not torch.isfinite(x).all():
        raise ValueError(f"NaN/Inf activation in {path}")
    return x


def step_for_file(path):
    return int(Path(path).parent.name.split("_")[-1])


@torch.no_grad()
def reconstruction_metrics(model, paths, batch_size=512, max_tokens=512):
    model.eval()
    device = next(model.parameters()).device
    totals = dict(mse=0., variance=0., cosine=0., nrmse=0., active=0., count=0)
    counts = torch.zeros(model.d_hidden, dtype=torch.int64)
    for path in paths:
        x = load_tokens(path, max_tokens=max_tokens)
        step = step_for_file(path)
        for xb in x.split(batch_size):
            xb = xb.to(device)
            z = encode_at(model, xb, step)
            y = model.decode(z)
            n = len(xb)
            err = (xb - y).square().mean()
            var = (xb - xb.mean(dim=0, keepdim=True)).square().mean()
            cos = F.cosine_similarity(xb.float(), y.float(), dim=-1).mean()
            nrmse = torch.sqrt(err / xb.square().mean().clamp_min(1e-12))
            totals["mse"] += err.item() * n
            totals["variance"] += var.item() * n
            totals["cosine"] += cos.item() * n
            totals["nrmse"] += nrmse.item() * n
            totals["active"] += (z != 0).sum().item()
            totals["count"] += n
            counts += (z != 0).sum(dim=0).cpu()
    n = totals["count"]
    if not n:
        raise ValueError("Empty evaluation set")
    return {"n_tokens": n, "mse": totals["mse"] / n,
            "cosine": totals["cosine"] / n, "nrmse": totals["nrmse"] / n,
            "explained_variance_proxy": 1 - totals["mse"] / max(totals["variance"], 1e-12),
            "active_per_token": totals["active"] / n,
            "firing_fraction": float((counts > 0).float().mean()),
            "dead_features": int((counts == 0).sum())}


def train_candidate(train_paths, val_paths, *, expansion_factor, k, architecture="topk",
                    epochs=4, max_tokens=512, batch_size=512, lr=1e-4, device="cuda", seed=2601):
    torch.manual_seed(seed)
    cls = TemporalTopKSAE if architecture == "temporal" else TopKSparseAutoencoder
    model = cls(d_model=1536, expansion_factor=expansion_factor, k=k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
    hist = []
    for epoch in range(epochs):
        model.train()
        rng = random.Random(seed + epoch)
        ordered = list(train_paths)
        rng.shuffle(ordered)
        running, samples = 0., 0
        for path in ordered:
            x = load_tokens(path, max_tokens=max_tokens, sample_seed=seed + epoch)
            for xb in x.split(batch_size):
                xb = xb.to(device)
                opt.zero_grad(set_to_none=True)
                out, _ = model(xb, step=step_for_file(path)) if architecture == "temporal" else model(xb)
                loss = F.mse_loss(out, xb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                running += float(loss.item()) * len(xb)
                samples += len(xb)
        metrics = reconstruction_metrics(model, val_paths, max_tokens=max_tokens)
        hist.append({"epoch": epoch + 1, "train_mse": running / max(samples, 1), **metrics})
    return model, hist


def save_candidate(path, model, *, expansion_factor, k, architecture, val_metrics, split_hash):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"d_model": 1536,
            "sae": {"expansion_factor": int(expansion_factor), "k": int(k), "architecture": architecture},
            "validation": val_metrics, "split_sha256": split_hash}
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "metadata": meta}, path)
    return meta


def stable_split_hash(paths):
    return hashlib.sha256("\n".join(sorted(map(str, paths))).encode()).hexdigest()
