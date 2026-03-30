import numpy as np
import deepxde as dde
from .config import MaterialProperties
from .geometry import CrackGeometry

try:
    from deepxde.icbc import PointSetOperatorBC
except ImportError:
    raise ImportError(
        "deepxde.icbc.PointSetOperatorBC is required. "
        "Upgrade deepxde: pip install 'deepxde>=1.10.0'"
    )


class LinearElasticityPDE:
    def __init__(self, material: MaterialProperties):
        self.mat = material

    def residual(self, x, y):
        u_x, u_y = y[:, 0:1], y[:, 1:2]
        lam, mu = self.mat.lam, self.mat.mu

        u_xx = dde.grad.jacobian(u_x, x, i=0, j=0)
        u_xy = dde.grad.jacobian(u_x, x, i=0, j=1)
        u_yx = dde.grad.jacobian(u_y, x, i=0, j=0)
        u_yy = dde.grad.jacobian(u_y, x, i=0, j=1)

        tr = u_xx + u_yy
        s_xx = lam * tr + 2 * mu * u_xx
        s_yy = lam * tr + 2 * mu * u_yy
        s_xy = mu * (u_xy + u_yx)

        eq_x = dde.grad.jacobian(s_xx, x, i=0, j=0) + dde.grad.jacobian(s_xy, x, i=0, j=1)
        eq_y = dde.grad.jacobian(s_xy, x, i=0, j=0) + dde.grad.jacobian(s_yy, x, i=0, j=1)
        return [eq_x, eq_y]

    def _stresses(self, x, y):
        u_x, u_y = y[:, 0:1], y[:, 1:2]
        lam, mu = self.mat.lam, self.mat.mu
        u_xx = dde.grad.jacobian(u_x, x, i=0, j=0)
        u_xy = dde.grad.jacobian(u_x, x, i=0, j=1)
        u_yx = dde.grad.jacobian(u_y, x, i=0, j=0)
        u_yy = dde.grad.jacobian(u_y, x, i=0, j=1)
        tr = u_xx + u_yy
        s_xx = lam * tr + 2 * mu * u_xx
        s_yy = lam * tr + 2 * mu * u_yy
        s_xy = mu * (u_xy + u_yx)
        return s_xx, s_yy, s_xy

    def sigma_yy_residual(self, x, y, X):
        _, s_yy, _ = self._stresses(x, y)
        return s_yy

    def sigma_xy_residual(self, x, y, X):
        _, _, s_xy = self._stresses(x, y)
        return s_xy


def build_boundary_conditions(geom_dde, geom_crack: CrackGeometry,
                               material: MaterialProperties,
                               pde_obj: LinearElasticityPDE,
                               n_crack_pts: int = 200):
    """
    Build all BCs for Mode I loading.
    Returns (bcs, crack_pts, n_crack_bcs) where n_crack_bcs is the number
    of crack-surface BC loss terms (used to size loss_weights in solver).
    """
    _, H = geom_crack.domain_size
    t = material.traction

    def on_bottom(x, on_b): return on_b and np.isclose(x[1], 0.0)
    def on_top(x, on_b):    return on_b and np.isclose(x[1], H)
    def on_left(x, on_b):   return on_b and np.isclose(x[0], 0.0)

    bc_bottom = dde.icbc.DirichletBC(geom_dde, lambda x: np.full((len(x), 1), -t), on_bottom, component=1)
    bc_top    = dde.icbc.DirichletBC(geom_dde, lambda x: np.full((len(x), 1),  t), on_top,    component=1)
    bc_left   = dde.icbc.DirichletBC(geom_dde, lambda x: np.zeros((len(x), 1)),    on_left,   component=0)

    crack_pts = geom_crack.sample_crack_surface_points(n_per_face=n_crack_pts // 2)

    zero_values = np.zeros((len(crack_pts), 1))
    bc_crack_syy = PointSetOperatorBC(crack_pts, zero_values, pde_obj.sigma_yy_residual)
    bc_crack_sxy = PointSetOperatorBC(crack_pts, zero_values, pde_obj.sigma_xy_residual)

    bcs = [bc_bottom, bc_top, bc_left, bc_crack_syy, bc_crack_sxy]
    return bcs, crack_pts, 2
