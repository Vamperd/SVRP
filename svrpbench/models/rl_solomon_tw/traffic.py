"""Static and perturbed travel-time matrices for TWCVRP experiments."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def euclidean_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    matrix = np.linalg.norm(diff, axis=-1).astype(np.float32)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def base_time_matrix(instance) -> np.ndarray:
    """Use benchmark matrix when present, otherwise Euclidean travel time."""
    if getattr(instance, "static_time_matrix", None) is not None:
        matrix = np.asarray(instance.static_time_matrix, dtype=np.float32)
        np.fill_diagonal(matrix, 0.0)
        return matrix
    return euclidean_matrix(instance.coords)


def planning_matrix(
    instance,
    *,
    mode: str,
    traffic_sigma: float = 0.20,
    traffic_buffer: float = 0.50,
) -> np.ndarray:
    """Return the matrix used by the solver while constructing routes."""
    base = base_time_matrix(instance)
    if mode == "static":
        return base
    if mode == "traffic":
        factor = 1.0 + max(0.0, traffic_sigma) * max(0.0, traffic_buffer)
        matrix = (base * factor).astype(np.float32)
        np.fill_diagonal(matrix, 0.0)
        return matrix
    raise ValueError("mode must be 'static' or 'traffic'.")


def sample_traffic_matrix(
    instance,
    *,
    seed: int,
    traffic_sigma: float = 0.20,
    min_factor: float = 0.65,
    max_factor: float = 1.80,
    asymmetric: bool = False,
) -> np.ndarray:
    """Sample a reproducible traffic-perturbed matrix for evaluation."""
    base = base_time_matrix(instance)
    rng = np.random.default_rng(seed)
    sigma = max(0.0, float(traffic_sigma))
    if sigma == 0:
        return base.copy()

    factors = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=base.shape)
    factors = np.clip(factors, min_factor, max_factor).astype(np.float32)
    if not asymmetric:
        factors = ((factors + factors.T) / 2.0).astype(np.float32)
    np.fill_diagonal(factors, 0.0)
    matrix = (base * factors).astype(np.float32)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def stable_seed(*parts: object, base_seed: int = 42) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + int(base_seed)) % (2**32 - 1)


def traffic_cache_path(output_dir: str | Path, *, size: int | str, instance_name: str) -> Path:
    return Path(output_dir) / str(size) / f"{Path(instance_name).stem}_traffic.npz"

