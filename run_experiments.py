#!/usr/bin/env python3
"""
X-PINN Fracture Mechanics -- Full Experiment Runner

Usage:
    python run_experiments.py                             # uses configs/mode1_quick.yaml
    python run_experiments.py --config configs/mode1_production.yaml
"""

import argparse
import yaml
import numpy as np
from xpinn_fracture import (
    MaterialProperties, TrainingConfig, CrackGeometry, XFEMCrackSolver,
    compute_ki_j_integral, compute_ki_displacement_extrapolation,
    compute_stress_components, westergaard_ki_analytical, validate_against_williams,
    plot_stress_field, plot_loss_history,
    plot_ki_path_independence, plot_displacement_extrapolation,
)

_DEFAULT_CONFIG = "configs/mode1_quick.yaml"


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_objects(cfg: dict):
    g = cfg["geometry"]
    m = cfg["material"]
    t = cfg["training"]

    geom = CrackGeometry(
        center=np.array(g["center"]),
        half_length=g["half_length"],
        angle=g.get("angle", 0.0),
        domain_size=tuple(g["domain_size"]),
        enrichment_radius=g["enrichment_radius"],
        enrichment_inner=g["enrichment_inner"],
    )
    material = MaterialProperties(
        E=m["E"], nu=m["nu"], traction=m["traction"], formulation=m["formulation"]
    )
    config = TrainingConfig(
        num_domain=t["num_domain"],
        num_boundary=t["num_boundary"],
        num_crack_surface=t.get("num_crack_surface", 200),
        num_test=t.get("num_test", 200),
        adam_iterations=t["adam_iterations"],
        adam_lr=t["adam_lr"],
        use_lbfgs=t.get("use_lbfgs", False),
        lbfgs_iterations=t.get("lbfgs_iterations", 1000),
        adaptive_sampling=t.get("adaptive_sampling", True),
        resample_every=t.get("resample_every", 500),
        loss_weight_pde=t.get("loss_weight_pde", 1.0),
        loss_weight_bc=t.get("loss_weight_bc", 10.0),
        loss_weight_crack=t.get("loss_weight_crack", 10.0),
    )
    return geom, material, config


def main(config_path: str = _DEFAULT_CONFIG):
    cfg = _load_config(config_path)
    geom, material, config = _build_objects(cfg)

    K_I_analytical = westergaard_ki_analytical(material, geom)
    print(f"\nAnalytical K_I (Westergaard + Feddersen): {K_I_analytical:.6f}\n")

    solver = XFEMCrackSolver(geom, material, config)
    solver.setup()
    solver.train()

    n_grid = 100
    xs, ys = np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid)
    x_test = np.column_stack([v.ravel() for v in np.meshgrid(xs, ys)])
    u_pred = solver.predict(x_test)

    print("\nComputing stress fields...")
    stress = compute_stress_components(x_test, solver.net, material)

    print("\nComputing K_I via J-integral...")
    ki_j = compute_ki_j_integral(solver.predict, geom, material, radii=[0.03, 0.05, 0.07])
    print(f"  tip1: {ki_j['tip1']['K_I_mean']:.6f} +/- {ki_j['tip1']['K_I_std']:.6f}")
    print(f"  tip2: {ki_j['tip2']['K_I_mean']:.6f} +/- {ki_j['tip2']['K_I_std']:.6f}")

    print("\nComputing K_I via displacement extrapolation...")
    ki_disp = compute_ki_displacement_extrapolation(solver.predict, geom.tip1, geom, material)
    print(f"  Extrapolated K_I: {ki_disp['K_I_extrapolated']:.6f}")

    K_I_pinn = (ki_j["tip1"]["K_I_mean"] + ki_j["tip2"]["K_I_mean"]) / 2
    von_mises = stress["von_mises"]

    print("\nValidating against Williams series...")
    val = validate_against_williams(solver.predict, geom, material, K_I_analytical, K_I_pinn)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  K_I analytical:                {K_I_analytical:.6f}")
    print(f"  K_I PINN (J-integral tip1):    {ki_j['tip1']['K_I_mean']:.6f}")
    print(f"  K_I PINN (J-integral tip2):    {ki_j['tip2']['K_I_mean']:.6f}")
    print(f"  K_I PINN (disp. extrap.):      {ki_disp['K_I_extrapolated']:.6f}")
    print(f"  K_I relative error:            {val['K_I_relative_error_pct']:.2f}%")
    print(f"  Path-independence (tip1):      {ki_j['tip1']['path_independence_pct']:.2f}%")
    if "tip1" in val:
        print(f"  L2 vs Williams (tip1):         {val['tip1']['l2_error_vs_williams']:.4f}")
    print("=" * 60)

    print("\nGenerating figures...")
    plot_stress_field(x_test, stress, geom, "stress_field.png")
    plot_loss_history(solver._loss_history, "loss_history.png")
    plot_ki_path_independence(ki_j, K_I_analytical, "ki_path_independence.png")
    plot_displacement_extrapolation(ki_disp, K_I_analytical, "ki_extrapolation.png")

    print("\nDone.")
    return solver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
