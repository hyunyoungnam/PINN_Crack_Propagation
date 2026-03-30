import numpy as np
from numpy.typing import NDArray
from .geometry import CrackGeometry
from .config import MaterialProperties

# Guard against division by zero in L2 relative error and K_I relative error.
_L2_EPSILON = 1e-14

# Theta is sampled in (-pi, pi) exclusive to avoid the branch cut at ±pi
# where arctan2 is discontinuous. This offset keeps points off the cut.
_THETA_EPSILON = 1e-6


def westergaard_ki_analytical(material: MaterialProperties, geom: CrackGeometry) -> float:
    """
    K_I for a central crack in a finite-width plate under remote tension.
    Feddersen finite-width correction: F = sqrt(sec(pi*a/W)).
    Far-field stress estimated from applied displacement: sigma = E * (2*traction / H).
    """
    a = geom.half_length
    W, H = geom.domain_size
    sigma_inf = material.E * (2 * material.traction / H)
    F = np.sqrt(1.0 / np.cos(np.pi * a / W))
    return float(sigma_inf * np.sqrt(np.pi * a) * F)


def williams_displacement_field(x: NDArray, tip: NDArray, K_I: float,
                                  material: MaterialProperties, geom: CrackGeometry) -> NDArray:
    """Leading-term Williams series displacement field near a crack tip. Returns (N, 2)."""
    r, theta = geom.polar_from_tip(x, tip)
    factor = K_I / (2 * material.mu) * np.sqrt(r / (2 * np.pi))
    u_x = factor * np.cos(theta / 2) * (material.kappa - 1 + 2 * np.sin(theta / 2) ** 2)
    u_y = factor * np.sin(theta / 2) * (material.kappa + 1 - 2 * np.cos(theta / 2) ** 2)
    return np.concatenate([u_x, u_y], axis=1)


def validate_against_williams(predict_fn, geom: CrackGeometry, material: MaterialProperties,
                                K_I_analytical: float, K_I_pinn: float,
                                n_pts: int = 500, seed: int = 42) -> dict:
    """
    Compare PINN displacement to Williams series in the near-tip region.
    Seed is fixed for reproducibility across runs.
    """
    results = {}
    rng = np.random.default_rng(seed)
    for tip_name, tip in [("tip1", geom.tip1), ("tip2", geom.tip2)]:
        r = rng.uniform(0.01, geom.enrichment_inner, n_pts)
        theta = rng.uniform(-np.pi + _THETA_EPSILON, np.pi - _THETA_EPSILON, n_pts)
        cos_a, sin_a = np.cos(geom.angle), np.sin(geom.angle)
        xl, yl = r * np.cos(theta), r * np.sin(theta)
        xg = tip[0] + xl * cos_a - yl * sin_a
        yg = tip[1] + xl * sin_a + yl * cos_a
        pts = np.column_stack([xg, yg])
        mask = ((pts[:, 0] >= 0) & (pts[:, 0] <= geom.domain_size[0]) &
                (pts[:, 1] >= 0) & (pts[:, 1] <= geom.domain_size[1]))
        pts = pts[mask]
        if len(pts) == 0:
            continue
        u_pinn = predict_fn(pts)
        u_ref = williams_displacement_field(pts, tip, K_I_analytical, material, geom)
        l2 = np.linalg.norm(u_pinn - u_ref) / (np.linalg.norm(u_ref) + _L2_EPSILON)
        results[tip_name] = {"l2_error_vs_williams": float(l2), "n_pts": len(pts)}

    ki_err = abs(K_I_pinn - K_I_analytical) / (abs(K_I_analytical) + _L2_EPSILON)
    results.update({
        "K_I_analytical": K_I_analytical,
        "K_I_pinn": K_I_pinn,
        "K_I_relative_error_pct": float(ki_err * 100),
    })
    return results
