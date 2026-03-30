import numpy as np
import matplotlib.pyplot as plt
from .geometry import CrackGeometry

# Clips extreme stress values for cleaner contour plots.
# 99th percentile preserves the tip concentration while hiding solver artifacts.
_VM_CLIP_PERCENTILE = 99

# Smoothing window as a fraction of total training epochs.
# 1/50 gives ~2% window — wide enough to smooth noise, narrow enough to show trends.
_LOSS_WINDOW_FRACTION = 50


def plot_stress_field(x_test, stress: dict, geom: CrackGeometry,
                       save_path="stress_field.png"):
    """
    Plot all four stress fields: sigma_xx, sigma_yy, sigma_xy, and von Mises.
    stress: dict returned by compute_stress_components().
    x_test must be a square (n x n) grid.
    """
    n = int(np.sqrt(len(x_test)))
    assert n * n == len(x_test), "x_test must be a square grid (n x n points)"
    X = x_test[:, 0].reshape(n, n)
    Y = x_test[:, 1].reshape(n, n)

    crack_x = [geom.tip2[0], geom.tip1[0]]
    crack_y = [geom.center[1], geom.center[1]]
    tip_x = [geom.tip1[0], geom.tip2[0]]
    tip_y = [geom.tip1[1], geom.tip2[1]]

    fields = [
        (stress["sigma_xx"], r"$\sigma_{xx}$", "RdBu_r"),
        (stress["sigma_yy"], r"$\sigma_{yy}$", "RdBu_r"),
        (stress["sigma_xy"], r"$\sigma_{xy}$", "RdBu_r"),
        (stress["von_mises"], r"Von Mises $\sigma_{vm}$", "jet"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (raw, title, cmap) in zip(axes, fields):
        # Clip at percentile to suppress tip singularity artifacts in contour fill
        vmax = np.percentile(np.abs(raw), _VM_CLIP_PERCENTILE)
        data = raw.reshape(n, n)
        if cmap == "RdBu_r":
            im = ax.contourf(X, Y, data, levels=50, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.contourf(X, Y, np.clip(data, 0, vmax), levels=50, cmap=cmap)
        ax.plot(crack_x, crack_y, "k-", lw=3, label="Crack")
        ax.plot(tip_x, tip_y, "ko", ms=7, markerfacecolor="white", markeredgewidth=2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_loss_history(losshistory, save_path="loss_history.png"):
    if losshistory is None:
        return
    raw = np.asarray(losshistory.loss_train)
    total = raw.sum(axis=1) if raw.ndim > 1 else raw
    epochs = np.arange(len(total))
    win = max(1, len(total) // _LOSS_WINDOW_FRACTION)
    mean_e, std_e = [], []
    for i in range(len(total)):
        s, e = max(0, i - win // 2), min(len(total), i + win // 2 + 1)
        w = total[s:e]
        mean_e.append(np.mean(w)); std_e.append(np.std(w))
    mean_e, std_e = np.array(mean_e), np.array(std_e)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, mean_e, "b-", lw=2, label="Mean loss")
    plt.fill_between(epochs, np.maximum(mean_e - std_e, 1e-16), mean_e + std_e,
                     alpha=0.25, color="lightblue", label="+-1 Std Dev")
    plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Total Loss")
    plt.title("Training Loss Convergence"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {save_path}"); plt.show()


def plot_ki_path_independence(ki_results: dict, ki_analytical: float,
                               save_path="ki_path_independence.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"tip1": "steelblue", "tip2": "tomato"}
    for tip_name, res in ki_results.items():
        if "radii" not in res:
            continue
        ax.plot(res["radii"], res["K_I_values"], "o-", color=colors.get(tip_name, "gray"),
                lw=2, ms=7, label=f"{tip_name}  mean={res['K_I_mean']:.4f}")
    ax.axhline(ki_analytical, color="k", ls="--", lw=2, label=f"Analytical K_I={ki_analytical:.4f}")
    ax.set_xlabel("Contour radius rho"); ax.set_ylabel("K_I")
    ax.set_title("K_I Path-Independence (J-Integral Contours)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {save_path}"); plt.show()


def plot_displacement_extrapolation(extrap: dict, ki_analytical: float,
                                     save_path="ki_extrapolation.png"):
    r = np.array(extrap["r_values"])
    ki = np.array(extrap["K_I_values"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.sqrt(r), ki, "b.-", lw=1.5, ms=4, label="K_I from COD")
    ax.axhline(ki_analytical, color="k", ls="--", lw=2, label=f"Analytical={ki_analytical:.4f}")
    ax.axhline(extrap["K_I_extrapolated"], color="tomato", ls="-.", lw=2,
               label=f"Extrapolated={extrap['K_I_extrapolated']:.4f}")
    ax.set_xlabel("sqrt(r)"); ax.set_ylabel("K_I")
    ax.set_title("K_I via Displacement Extrapolation (COD)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {save_path}"); plt.show()
