from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.nanmin(x))
    denom = float(np.nanmax(x))
    if denom > 0:
        x = x / denom
    return x


def binary_iou(score_map: np.ndarray, mask: np.ndarray, threshold: float = 0.3) -> float:
    score = normalize_map(score_map)
    pred = score >= float(threshold)
    mask = np.asarray(mask).astype(bool)
    inter = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    if union == 0:
        return float("nan")
    return float(inter / union)


def centroid(arr: np.ndarray) -> Tuple[float, float] | None:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.sum() <= 0:
        return None
    ys, xs = np.indices(arr.shape)
    total = arr.sum()
    return float((ys * arr).sum() / total), float((xs * arr).sum() / total)


def centroid_distance(score_map: np.ndarray, mask: np.ndarray) -> float:
    c1 = centroid(normalize_map(score_map))
    c2 = centroid(np.asarray(mask).astype(np.float32))
    if c1 is None or c2 is None:
        return float("nan")
    return float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))


def normalized_entropy(score_map: np.ndarray) -> float:
    x = np.asarray(score_map, dtype=np.float64).reshape(-1)
    x = x - np.nanmin(x)
    s = x.sum()
    if s <= 0:
        return float("nan")
    p = x / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(x)))


def summarize_numeric(rows, fields):
    out: Dict[str, float] = {}
    for field in fields:
        vals = [float(r[field]) for r in rows if field in r and r[field] == r[field]]
        if vals:
            out[f"{field}_mean"] = float(np.mean(vals))
            out[f"{field}_std"] = float(np.std(vals))
            out[f"{field}_n"] = int(len(vals))
    return out
