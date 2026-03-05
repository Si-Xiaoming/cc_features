

import numpy as np

import cc_features
from cc_features import FeatureType, TaskConfig, FeatureExtractor, compute_features

def make_plane(n: int = 600, z_noise: float = 1e-4, seed: int = 42) -> np.ndarray:
    """Dense horizontal plane (Z ≈ 0).  Planarity ≈ 1,  Verticality ≈ 0."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 3)).astype(np.float64)
    pts[:, 2] = z_noise * rng.standard_normal(n)
    return pts


def make_vertical_wall(n: int = 600, y_noise: float = 1e-4, seed: int = 7) -> np.ndarray:
    """Dense vertical wall (Y ≈ 0, normal ≈ ŷ).  Verticality ≈ 1."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 3)).astype(np.float64)
    pts[:, 1] = y_noise * rng.standard_normal(n)
    return pts


def make_line(n: int = 200, seed: int = 3) -> np.ndarray:
    """Points along the X-axis with tiny transverse noise."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 3), dtype=np.float64)
    pts[:, 0] = np.linspace(0, 1, n)
    pts[:, 1] = 1e-5 * rng.standard_normal(n)
    pts[:, 2] = 1e-5 * rng.standard_normal(n)
    return pts

pts = make_plane(n=100)
res = compute_features(pts, [("Verticality", 0.5)])
print(res)