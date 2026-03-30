from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

# Offset used when sampling crack face points: small enough to be on the face,
# large enough to avoid numerical issues in stress evaluation at the discontinuity.
_CRACK_FACE_OFFSET = 1e-5

# Added under the square root to prevent sqrt(0) at crack tips.
_R_EPSILON = 1e-14


@dataclass
class CrackGeometry:
    center: NDArray
    half_length: float
    angle: float = 0.0
    domain_size: tuple = (1.0, 1.0)
    enrichment_radius: float = 0.12
    enrichment_inner: float = 0.05

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64)
        self.tip1, self.tip2 = self._compute_tips()

    def _compute_tips(self):
        dx = self.half_length * np.cos(self.angle)
        dy = self.half_length * np.sin(self.angle)
        return self.center + np.array([dx, dy]), self.center - np.array([dx, dy])

    def to_local(self, x: NDArray):
        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        dx = x[:, 0:1] - self.center[0]
        dy = x[:, 1:2] - self.center[1]
        return dx * cos_a + dy * sin_a, -dx * sin_a + dy * cos_a

    def polar_from_tip(self, x: NDArray, tip: NDArray):
        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        dx = x[:, 0:1] - tip[0]
        dy = x[:, 1:2] - tip[1]
        xl = dx * cos_a + dy * sin_a
        yl = -dx * sin_a + dy * cos_a
        r = np.sqrt(xl ** 2 + yl ** 2 + _R_EPSILON)
        theta = np.arctan2(yl, xl)
        return r, theta

    def dist_to_nearest_tip(self, x: NDArray) -> NDArray:
        d1 = np.sqrt((x[:, 0:1] - self.tip1[0]) ** 2 + (x[:, 1:2] - self.tip1[1]) ** 2)
        d2 = np.sqrt((x[:, 0:1] - self.tip2[0]) ** 2 + (x[:, 1:2] - self.tip2[1]) ** 2)
        return np.minimum(d1, d2)

    def sample_crack_surface_points(self, n_per_face: int = 100) -> NDArray:
        x_line = np.linspace(self.tip2[0], self.tip1[0], n_per_face)
        y0 = self.center[1]
        upper = np.column_stack([x_line, np.full(n_per_face, y0 + _CRACK_FACE_OFFSET)])
        lower = np.column_stack([x_line, np.full(n_per_face, y0 - _CRACK_FACE_OFFSET)])
        return np.vstack([upper, lower])
