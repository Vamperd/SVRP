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
    traffic_profile: str = "additive",
    traffic_strength: float = 1.0,
) -> np.ndarray:
    """Return the matrix used by the solver while constructing routes."""
    base = base_time_matrix(instance)
    if mode == "static":
        return base
    if mode == "traffic":
        sigma = max(0.0, float(traffic_sigma))
        buffer = max(0.0, float(traffic_buffer))
        strength = max(0.0, float(traffic_strength))
        if traffic_profile == "additive":
            factor = 1.0 + strength * sigma * buffer
        elif traffic_profile == "proportional":
            distance_factor = 1.0 - np.exp(-np.maximum(base, 0.0) / 50.0)
            factor = 1.0 + strength * sigma * buffer * distance_factor
        else:
            raise ValueError("traffic_profile must be 'additive' or 'proportional'.")
        matrix = (base * factor).astype(np.float32)
        np.fill_diagonal(matrix, 0.0)
        return matrix
    raise ValueError("mode must be 'static' or 'traffic'.")


def sample_traffic_matrix(
    instance,
    *,
    seed: int,
    traffic_sigma: float = 0.20,
    traffic_profile: str = "additive",
    traffic_strength: float = 1.0,
    min_factor: float = 0.65,
    max_factor: float = 1.80,
    asymmetric: bool = False,
) -> np.ndarray:
    """Sample a reproducible traffic-perturbed matrix for traffic-aware training."""
    base = base_time_matrix(instance)
    rng = np.random.default_rng(seed)
    sigma = max(0.0, float(traffic_sigma))
    strength = max(0.0, float(traffic_strength))
    if sigma == 0:
        return base.copy()

    noise = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=base.shape)
    if traffic_profile == "additive":
        factors = 1.0 + strength * (noise - 1.0)
        factors = np.clip(factors, min_factor, max_factor).astype(np.float32)
    elif traffic_profile == "proportional":
        distance_factor = 1.0 - np.exp(-np.maximum(base, 0.0) / 50.0)
        factors = 1.0 + strength * sigma * distance_factor * noise
        factors = np.clip(factors, 1.0, max_factor).astype(np.float32)
    else:
        raise ValueError("traffic_profile must be 'additive' or 'proportional'.")
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
