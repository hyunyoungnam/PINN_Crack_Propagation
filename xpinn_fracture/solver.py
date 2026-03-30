import os
import numpy as np
import deepxde as dde
import torch
from time import perf_counter
from typing import Optional

from .config import MaterialProperties, TrainingConfig
from .geometry import CrackGeometry
from .networks import XPINNCompositeNet
from .physics import LinearElasticityPDE, build_boundary_conditions

os.environ.setdefault("DDE_BACKEND", "pytorch")
dde.config.set_default_float("float64")
torch.set_default_dtype(torch.float64)

# Small constant for numerical stability in RAD weight normalization
_RAD_EPSILON = 1e-14


class XFEMCrackSolver:
    """
    X-PINN solver for 2D Mode I fracture mechanics.
    Uses 3-network XFEM decomposition with float64 precision.
    """

    def __init__(self, geom: CrackGeometry, material: MaterialProperties, config: TrainingConfig):
        self.geom = geom
        self.material = material
        self.config = config
        self.pde = LinearElasticityPDE(material)
        self.net: Optional[XPINNCompositeNet] = None
        self.model: Optional[dde.Model] = None
        self.data = None
        self._loss_history = None
        self._geom_dde = None

    def setup(self):
        self._geom_dde = dde.geometry.Rectangle([0.0, 0.0], list(self.geom.domain_size))

        bcs, crack_pts, n_crack_bcs = build_boundary_conditions(
            self._geom_dde, self.geom, self.material, self.pde,
            n_crack_pts=self.config.num_crack_surface
        )
        # 2 PDE equations + 3 domain BCs + 2 crack surface BCs
        self._n_loss_terms = 2 + 3 + n_crack_bcs

        self.data = dde.data.PDE(
            self._geom_dde,
            self.pde.residual,
            bcs,
            num_domain=self.config.num_domain,
            num_boundary=self.config.num_boundary,
            num_test=self.config.num_test,
            anchors=crack_pts,
        )

        self.net = XPINNCompositeNet(self.geom)
        self.model = dde.Model(self.data, self.net)

    def _loss_weights(self) -> list:
        cfg = self.config
        return (
            [cfg.loss_weight_pde] * 2
            + [cfg.loss_weight_bc] * 3
            + [cfg.loss_weight_crack] * (self._n_loss_terms - 5)
        )

    def _pde_residual_magnitude(self, X: np.ndarray) -> np.ndarray:
        x_t = torch.tensor(X, dtype=torch.float64, requires_grad=True)
        with torch.enable_grad():
            y_t = self.net(x_t)
            u_x, u_y = y_t[:, 0:1], y_t[:, 1:2]
            lam, mu = self.material.lam, self.material.mu

            def jac(u, j):
                return torch.autograd.grad(u.sum(), x_t, create_graph=True)[0][:, j:j + 1]

            u_xx, u_xy = jac(u_x, 0), jac(u_x, 1)
            u_yx, u_yy = jac(u_y, 0), jac(u_y, 1)
            tr = u_xx + u_yy
            s_xx = lam * tr + 2 * mu * u_xx
            s_yy = lam * tr + 2 * mu * u_yy
            s_xy = mu * (u_xy + u_yx)
            eq_x = jac(s_xx, 0) + jac(s_xy, 1)
            eq_y = jac(s_xy, 0) + jac(s_yy, 1)
            res = (eq_x ** 2 + eq_y ** 2).detach().numpy().flatten()
        return res

    def _rad_resample(self, n_select: int, n_cand: int = 10000) -> np.ndarray:
        X_cand = self._geom_dde.random_points(n_cand)
        res = self._pde_residual_magnitude(X_cand)
        k1, k2 = self.config.rad_k1, self.config.rad_k2
        w = np.power(np.abs(res) + _RAD_EPSILON, k1)
        w = w / w.mean() + k2
        w = w / w.sum()
        idx = np.random.choice(len(X_cand), size=min(n_select, len(X_cand)), replace=False, p=w)
        return X_cand[idx]

    def train(self):
        if self.model is None:
            raise RuntimeError("Call setup() first.")

        cfg = self.config
        weights = self._loss_weights()

        print("--- Stage 1: Adam ---")
        self.model.compile("adam", lr=cfg.adam_lr, loss_weights=weights)
        t0 = perf_counter()
        all_loss = []

        if cfg.adaptive_sampling and cfg.resample_every > 0:
            for epoch in range(0, cfg.adam_iterations, cfg.resample_every):
                n_iters = min(cfg.resample_every, cfg.adam_iterations - epoch)
                lh, ts = self.model.train(iterations=n_iters, display_every=200)
                raw = np.asarray(lh.loss_train)
                all_loss.extend((raw.sum(axis=1) if raw.ndim > 1 else raw).tolist())

                if epoch + n_iters < cfg.adam_iterations:
                    X_new = self._rad_resample(cfg.num_domain)
                    self.data.train_x_all = X_new
                    self.data.train_x = X_new
                    print(f"  Resampled {len(X_new)} pts at iter {epoch + n_iters}")
        else:
            lh, ts = self.model.train(iterations=cfg.adam_iterations, display_every=500)
            raw = np.asarray(lh.loss_train)
            all_loss = (raw.sum(axis=1) if raw.ndim > 1 else raw).tolist()

        print(f"Adam done in {perf_counter() - t0:.1f}s")

        if cfg.use_lbfgs:
            print("\n--- Stage 2: L-BFGS ---")
            t1 = perf_counter()
            self.model.compile("L-BFGS", loss_weights=weights)
            lh2, ts = self.model.train(iterations=cfg.lbfgs_iterations)
            raw2 = np.asarray(lh2.loss_train)
            all_loss.extend((raw2.sum(axis=1) if raw2.ndim > 1 else raw2).tolist())
            print(f"L-BFGS done in {perf_counter() - t1:.1f}s")
            lh2.loss_train = all_loss
            self._loss_history = lh2
            return lh2, ts

        lh.loss_train = all_loss
        self._loss_history = lh
        return lh, ts

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call setup() and train() first.")
        return np.asarray(self.model.predict(x), dtype=np.float64)
