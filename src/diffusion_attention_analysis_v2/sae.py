from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKActivation(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = int(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k >= x.shape[-1]:
            return x
        values, indices = torch.topk(x, k=self.k, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(-1, indices, values)
        return out


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, expansion_factor: int = 4, k: int = 20) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_hidden = int(d_model) * int(expansion_factor)
        self.k = int(k)
        self.encoder = nn.Linear(self.d_model, self.d_hidden)
        self.decoder = nn.Linear(self.d_hidden, self.d_model)
        self.topk = TopKActivation(k=self.k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.topk(F.relu(self.encoder(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z


@dataclass
class SAETrainStats:
    epoch: int
    mean_loss: float
    mean_reconstruction_loss: float
    active_fraction: float
    batches: int


def infer_d_model(paths: Sequence[Path]) -> int:
    if not paths:
        raise ValueError("No activation files supplied.")
    payload = torch.load(paths[0], map_location="cpu")
    x = payload["features"]
    return int(x.shape[-1])


def iter_activation_batches(paths: Sequence[Path], batch_size: int, *, shuffle: bool = True, max_tokens_per_file: int | None = None):
    generator = torch.Generator().manual_seed(0)
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        x = payload["features"].float()
        if max_tokens_per_file is not None and x.shape[0] > int(max_tokens_per_file):
            idx = torch.randperm(x.shape[0], generator=generator)[: int(max_tokens_per_file)]
            x = x[idx]
        if shuffle:
            x = x[torch.randperm(x.shape[0], generator=generator)]
        for start in range(0, x.shape[0], int(batch_size)):
            yield x[start : start + int(batch_size)]


def train_sae(sae: TopKSparseAutoencoder, activation_paths: Sequence[Path], *, batch_size: int, learning_rate: float, num_epochs: int, weight_decay: float, device: str, max_tokens_per_file: int | None = None, progress=None) -> List[SAETrainStats]:
    sae = sae.to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    history: List[SAETrainStats] = []
    for epoch in range(int(num_epochs)):
        sae.train()
        total_loss = total_recon = total_active = total_elements = 0.0
        batches = 0
        for batch in iter_activation_batches(activation_paths, int(batch_size), shuffle=True, max_tokens_per_file=max_tokens_per_file):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            x_hat, z = sae(batch)
            recon = F.mse_loss(x_hat, batch)
            loss = recon
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            total_recon += float(recon.item())
            total_active += float((z > 0).float().sum().item())
            total_elements += float(z.numel())
            batches += 1
        row = SAETrainStats(
            epoch=epoch,
            mean_loss=total_loss / max(batches, 1),
            mean_reconstruction_loss=total_recon / max(batches, 1),
            active_fraction=total_active / max(total_elements, 1.0),
            batches=batches,
        )
        history.append(row)
        if progress is not None:
            progress.update(current=epoch + 1, message="SAE epoch completed", **asdict(row))
    return history


def save_sae_checkpoint(path: str | Path, sae: TopKSparseAutoencoder, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sae.state_dict(), "metadata": metadata}, path)


def load_sae_checkpoint(path: str | Path, device: str = "cuda") -> TopKSparseAutoencoder:
    ckpt = torch.load(path, map_location=device)
    meta = ckpt.get("metadata", {})
    d_model = int(meta.get("d_model"))
    sae_cfg = meta.get("sae", {})
    sae = TopKSparseAutoencoder(d_model=d_model, expansion_factor=int(sae_cfg.get("expansion_factor", 4)), k=int(sae_cfg.get("k", 20)))
    sae.load_state_dict(ckpt["state_dict"])
    return sae.to(device).eval()
