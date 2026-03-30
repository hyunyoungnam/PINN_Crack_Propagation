import numpy as np
import torch
import torch.nn as nn
from .geometry import CrackGeometry

# Small constant to avoid sqrt(0) singularity at crack tips
_R_EPSILON = 1e-14


def _build_fnn(layer_sizes: list, activation: str = "tanh") -> nn.Sequential:
    act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "swish": nn.SiLU}
    act_cls = act_map.get(activation, nn.Tanh)
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(act_cls())
    return nn.Sequential(*layers)


class XPINNCompositeNet(nn.Module):
    """
    Three-network X-PINN decomposition (all enrichment computed natively in torch):
        u(x) = u_C(x) + H(x)*u_D(x) + psi(x)*u_S(x, Phi(x))

    Network sizes are based on the X-PINN paper (Jagtap & Karniadakis, 2020):
      u_C: smooth background,    input dim=2,  6 hidden layers x 40 neurons
      u_D: discontinuous jump,   input dim=3,  4 hidden layers x 20 neurons (x augmented with H)
      u_S: singular near-tip,    input dim=10, 4 hidden layers x 20 neurons (x augmented with 8 tip functions)
    """

    def __init__(self, geom: CrackGeometry, activation: str = "tanh"):
        super().__init__()
        self.geom = geom
        # Cache geometry scalars as floats so they work inside torch ops
        self._cos_a = float(np.cos(geom.angle))
        self._sin_a = float(np.sin(geom.angle))
        self._tip1 = geom.tip1.tolist()
        self._tip2 = geom.tip2.tolist()
        self._r_s = geom.enrichment_inner
        self._r_b = geom.enrichment_radius

        self.net_C = _build_fnn([2, 40, 40, 40, 40, 40, 40, 2], activation)
        self.net_D = _build_fnn([3, 20, 20, 20, 20, 2], activation)
        self.net_S = _build_fnn([10, 20, 20, 20, 20, 2], activation)
        self._init_weights()
        # Required by DeepXDE's model compilation
        self.regularizer = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _tip_features(self, x: torch.Tensor, tip: list) -> torch.Tensor:
        """4 XFEM asymptotic crack-tip enrichment functions for one tip. Returns (N, 4)."""
        dx = x[:, 0:1] - tip[0]
        dy = x[:, 1:2] - tip[1]
        xl = dx * self._cos_a + dy * self._sin_a
        yl = -dx * self._sin_a + dy * self._cos_a
        r = torch.sqrt(xl ** 2 + yl ** 2 + _R_EPSILON)
        theta = torch.atan2(yl, xl)
        sqrt_r = torch.sqrt(r)
        return torch.cat([
            sqrt_r * torch.sin(theta / 2),
            sqrt_r * torch.cos(theta / 2),
            sqrt_r * torch.sin(theta / 2) * torch.sin(theta),
            sqrt_r * torch.cos(theta / 2) * torch.sin(theta),
        ], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Heaviside enrichment: sign of perpendicular distance from crack line
        dx = x[:, 0:1] - self.geom.center[0]
        dy = x[:, 1:2] - self.geom.center[1]
        y_local = -dx * self._sin_a + dy * self._cos_a
        H = torch.sign(y_local)
        H = torch.where(H == 0, torch.ones_like(H), H)

        # Partition-of-unity cutoff: C2 Hermite smoothstep from enrichment_inner to enrichment_radius
        d1 = torch.sqrt((x[:, 0:1] - self._tip1[0]) ** 2 + (x[:, 1:2] - self._tip1[1]) ** 2)
        d2 = torch.sqrt((x[:, 0:1] - self._tip2[0]) ** 2 + (x[:, 1:2] - self._tip2[1]) ** 2)
        r_min = torch.minimum(d1, d2)
        t = torch.clamp((r_min - self._r_s) / (self._r_b - self._r_s), 0.0, 1.0)
        psi = 1.0 - 3.0 * t ** 2 + 2.0 * t ** 3

        # 8 XFEM tip enrichment features (4 per tip)
        Phi = torch.cat([self._tip_features(x, self._tip1),
                         self._tip_features(x, self._tip2)], dim=1)

        u_C = self.net_C(x)
        u_D = self.net_D(torch.cat([x, H], dim=1))
        u_S = self.net_S(torch.cat([x, Phi], dim=1))

        return u_C + H * u_D + psi * u_S
