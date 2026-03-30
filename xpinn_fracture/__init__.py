from .config import MaterialProperties, TrainingConfig
from .geometry import CrackGeometry
from .solver import XFEMCrackSolver
from .postprocess import (compute_ki_j_integral, compute_ki_displacement_extrapolation,
                           compute_von_mises_stress, compute_stress_components)
from .validation import westergaard_ki_analytical, validate_against_williams
from .visualization import (plot_stress_field, plot_loss_history,
                             plot_ki_path_independence, plot_displacement_extrapolation)

__all__ = [
    "MaterialProperties", "TrainingConfig", "CrackGeometry", "XFEMCrackSolver",
    "compute_ki_j_integral", "compute_ki_displacement_extrapolation",
    "compute_von_mises_stress", "compute_stress_components",
    "westergaard_ki_analytical", "validate_against_williams",
    "plot_stress_field", "plot_loss_history",
    "plot_ki_path_independence", "plot_displacement_extrapolation",
]
