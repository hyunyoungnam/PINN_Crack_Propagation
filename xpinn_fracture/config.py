from dataclasses import dataclass, field


@dataclass
class MaterialProperties:
    E: float = 1.0
    nu: float = 0.3
    traction: float = 0.1
    formulation: str = "plane_stress"

    @property
    def mu(self) -> float:
        return self.E / (2 * (1 + self.nu))

    @property
    def lam(self) -> float:
        if self.formulation == "plane_stress":
            return self.E * self.nu / (1 - self.nu ** 2)
        else:
            return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def kappa(self) -> float:
        if self.formulation == "plane_stress":
            return (3 - self.nu) / (1 + self.nu)
        else:
            return 3 - 4 * self.nu


@dataclass
class TrainingConfig:
    num_domain: int = 5000
    num_boundary: int = 200
    num_crack_surface: int = 200
    num_test: int = 200
    adam_iterations: int = 10000
    adam_lr: float = 1e-3
    use_lbfgs: bool = True
    lbfgs_iterations: int = 2000
    adaptive_sampling: bool = True
    resample_every: int = 500
    rad_k1: float = 1.0
    rad_k2: float = 1.0
    loss_weight_pde: float = 1.0
    loss_weight_bc: float = 10.0
    loss_weight_crack: float = 10.0
