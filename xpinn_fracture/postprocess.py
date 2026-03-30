import numpy as np
import torch
from numpy.typing import NDArray
from .geometry import CrackGeometry
from .config import MaterialProperties

# Finite-difference step as a fraction of the contour radius.
# Small enough for accuracy, large enough to avoid float64 cancellation errors.
_FD_STEP_RATIO = 1e-3

# Vertical offset for crack-opening-displacement sampling points.
# Must match geometry._CRACK_FACE_OFFSET to sample just above/below the crack.
_COD_FACE_OFFSET = 1e-5

# Guard against division by zero in K_I path-independence percentage.
_KI_EPSILON = 1e-14

# Guard for von Mises sqrt to avoid sqrt(0) at stress-free points.
_VM_EPSILON = 1e-14


def _central_difference(predict_fn, pts: np.ndarray, h: float) -> tuple:
    """Central-difference displacement gradients at pts. Returns (du_x_dx, du_x_dy, du_y_dx, du_y_dy)."""
    u_px = predict_fn(pts + np.array([[h, 0]]))
    u_mx = predict_fn(pts - np.array([[h, 0]]))
    u_py = predict_fn(pts + np.array([[0, h]]))
    u_my = predict_fn(pts - np.array([[0, h]]))
    return (
        (u_px[:, 0] - u_mx[:, 0]) / (2 * h),
        (u_py[:, 0] - u_my[:, 0]) / (2 * h),
        (u_px[:, 1] - u_mx[:, 1]) / (2 * h),
        (u_py[:, 1] - u_my[:, 1]) / (2 * h),
    )


def compute_j_integral(predict_fn, tip: NDArray, geom: CrackGeometry,
                        material: MaterialProperties, rho: float = 0.04,
                        n_pts: int = 500) -> float:
    phi = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    pts = np.column_stack([tip[0] + rho * np.cos(phi), tip[1] + rho * np.sin(phi)])
    nx, ny = np.cos(phi), np.sin(phi)

    h = rho * _FD_STEP_RATIO
    du_x_dx, du_x_dy, du_y_dx, du_y_dy = _central_difference(predict_fn, pts, h)

    lam, mu = material.lam, material.mu
    eps_xy = 0.5 * (du_x_dy + du_y_dx)
    tr = du_x_dx + du_y_dy
    s_xx = lam * tr + 2 * mu * du_x_dx
    s_yy = lam * tr + 2 * mu * du_y_dy
    s_xy = 2 * mu * eps_xy

    W = 0.5 * (s_xx * du_x_dx + s_yy * du_y_dy + 2 * s_xy * eps_xy)
    t_x = s_xx * nx + s_xy * ny
    t_y = s_xy * nx + s_yy * ny
    integrand = W * nx - (t_x * du_x_dx + t_y * du_y_dx)
    return float(np.trapezoid(integrand, dx=2 * np.pi / n_pts) * rho)


def compute_ki_j_integral(predict_fn, geom: CrackGeometry, material: MaterialProperties,
                           radii=None) -> dict:
    if radii is None:
        radii = [0.03, 0.05, 0.07]
    results = {}
    for tip_name, tip in [("tip1", geom.tip1), ("tip2", geom.tip2)]:
        ki_vals = []
        for rho in radii:
            J = compute_j_integral(predict_fn, tip, geom, material, rho=rho)
            K_I = np.sqrt(abs(J) * material.E) * np.sign(J) if J != 0 else 0.0
            ki_vals.append(float(K_I))
        results[tip_name] = {
            "radii": radii,
            "K_I_values": ki_vals,
            "K_I_mean": float(np.mean(ki_vals)),
            "K_I_std": float(np.std(ki_vals)),
            "path_independence_pct": float(np.std(ki_vals) / (abs(np.mean(ki_vals)) + _KI_EPSILON) * 100),
        }
    return results


def compute_ki_displacement_extrapolation(predict_fn, tip: NDArray, geom: CrackGeometry,
                                           material: MaterialProperties, n_pts: int = 50) -> dict:
    r_vals = np.linspace(0.005, 0.06, n_pts)
    pts_upper = np.column_stack([tip[0] - r_vals, np.full(n_pts, tip[1] + _COD_FACE_OFFSET)])
    pts_lower = np.column_stack([tip[0] - r_vals, np.full(n_pts, tip[1] - _COD_FACE_OFFSET)])
    cod = predict_fn(pts_upper)[:, 1] - predict_fn(pts_lower)[:, 1]

    K_I_vals = (material.mu * (material.kappa + 1) / 4) * np.sqrt(2 * np.pi / r_vals) * cod
    coeffs = np.polyfit(np.sqrt(r_vals), K_I_vals, 1)
    return {
        "r_values": r_vals.tolist(),
        "K_I_values": K_I_vals.tolist(),
        "K_I_extrapolated": float(coeffs[1]),
        "K_I_near_tip": float(K_I_vals[0]),
    }


def compute_stress_components(x_test: np.ndarray, model_net, material: MaterialProperties) -> dict:
    """
    Compute all stress components and von Mises stress via autograd.
    Returns dict with keys: sigma_xx, sigma_yy, sigma_xy, von_mises — each (N,) array.
    """
    x_t = torch.tensor(x_test, dtype=torch.float64, requires_grad=True)
    with torch.enable_grad():
        y_t = model_net(x_t)
        u_x, u_y = y_t[:, 0:1], y_t[:, 1:2]
        g_ux = torch.autograd.grad(u_x, x_t, torch.ones_like(u_x), create_graph=False, retain_graph=True)[0]
        g_uy = torch.autograd.grad(u_y, x_t, torch.ones_like(u_y), create_graph=False)[0]
        eps_xx = g_ux[:, 0:1]
        eps_xy = 0.5 * (g_ux[:, 1:2] + g_uy[:, 0:1])
        eps_yy = g_uy[:, 1:2]
        lam, mu = material.lam, material.mu
        tr = eps_xx + eps_yy
        s_xx = lam * tr + 2 * mu * eps_xx
        s_yy = lam * tr + 2 * mu * eps_yy
        s_xy = 2 * mu * eps_xy
        vm = torch.sqrt(s_xx ** 2 - s_xx * s_yy + s_yy ** 2 + 3 * s_xy ** 2 + _VM_EPSILON)
    return {
        "sigma_xx": s_xx.detach().numpy().flatten(),
        "sigma_yy": s_yy.detach().numpy().flatten(),
        "sigma_xy": s_xy.detach().numpy().flatten(),
        "von_mises": vm.detach().numpy().flatten(),
    }


def compute_von_mises_stress(x_test: np.ndarray, model_net, material: MaterialProperties) -> np.ndarray:
    return compute_stress_components(x_test, model_net, material)["von_mises"]
