from __future__ import annotations

import numpy as np


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Expected a non-empty 2D feature matrix")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def standardize(x: np.ndarray, scaler: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = scaler
    return (x - mean) / std


def mean_shift_score(x: np.ndarray, context_windows: int = 10) -> tuple[float, int]:
    """Maximum cosine distance between means before/after each candidate split.

    Returns (score, split_index). split_index is the first window on the right.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    w = int(context_windows)
    if w < 1 or n < 2 * w:
        return 0.0, 0
    prefix = np.vstack([np.zeros((1, d)), np.cumsum(x, axis=0)])
    splits = np.arange(w, n - w + 1)
    left = (prefix[splits] - prefix[splits - w]) / w
    right = (prefix[splits + w] - prefix[splits]) / w
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-12)
    distance = 1.0 - np.clip(cosine, -1.0, 1.0)
    best = int(np.argmax(distance))
    return float(distance[best]), int(splits[best])


def split_to_verse(split_index: int, *, stride: int, window_len: int) -> int:
    """Approximate verse coordinate represented by a window split."""
    return int(split_index * stride + window_len // 2)
