import numpy as np
from numpy.typing import NDArray
from .geometry import CrackGeometry, _R_EPSILON

# Denominator guard for the partition-of-unity normalization.
# Prevents division by zero when enrichment_inner == enrichment_radius.
_POU_DENOM_EPSILON = 1e-14


def heaviside_enrichment(x: NDArray, geom: CrackGeometry) -> NDArray:
    _, y_local = geom.to_local(x)
    H = np.sign(y_local).astype(np.float64)
    H[H == 0] = 1.0
    return H  # (N, 1)


def partition_of_unity_cutoff(x: NDArray, geom: CrackGeometry) -> NDArray:
    r = geom.dist_to_nearest_tip(x)
    t = np.clip((r - geom.enrichment_inner) / (geom.enrichment_radius - geom.enrichment_inner + _POU_DENOM_EPSILON), 0.0, 1.0)
    return (1.0 - 3.0 * t ** 2 + 2.0 * t ** 3).astype(np.float64)  # C2 Hermite smoothstep, (N, 1)


def tip_enrichment_functions(x: NDArray, tip: NDArray, geom: CrackGeometry) -> NDArray:
    """4 XFEM asymptotic crack-tip enrichment functions for one tip. Returns (N, 4)."""
    r, theta = geom.polar_from_tip(x, tip)
    sqrt_r = np.sqrt(r)
    return np.concatenate([
        sqrt_r * np.sin(theta / 2),
        sqrt_r * np.cos(theta / 2),
        sqrt_r * np.sin(theta / 2) * np.sin(theta),
        sqrt_r * np.cos(theta / 2) * np.sin(theta),
    ], axis=1).astype(np.float64)


def all_tip_enrichment_features(x: NDArray, geom: CrackGeometry) -> NDArray:
    """Enrichment features for both crack tips. Returns (N, 8)."""
    return np.concatenate([
        tip_enrichment_functions(x, geom.tip1, geom),
        tip_enrichment_functions(x, geom.tip2, geom),
    ], axis=1).astype(np.float64)
