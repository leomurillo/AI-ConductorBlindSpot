"""
E8 figure -- built locally from the JSON (no torch, no GPU): the gradient-trained
encoder's recovered chart against the target slow eigenfunction phi_1, per world.

The training (e8_trained_encoder.py) runs on a GPU and writes the JSON with the
recovered f on the grid; this script recomputes the (numpy-only) targets from
apex_world and draws the comparison. Run with the figure-capable interpreter:
  py empirical/apex_recovery/e8_plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import apex_world as aw  # noqa: E402

HERE = Path(__file__).resolve().parent


def half_width_for(world):
    return 3.0 if world == "uniform" else 6.0


def main():
    d = json.loads((HERE / "reports" / "e8_trained_encoder.json").read_text())
    rows = d["partA"]
    fig, axes = plt.subplots(1, len(rows), figsize=(3.1 * len(rows), 3.2))
    for ax, r in zip(axes, rows):
        w = r["world"]
        z, pi, lam, phi = aw.transition_eigh(w, n_points=401, half_width=half_width_for(w))
        phi1 = phi[:, 1]
        f = np.array(r["f_grid"])
        coef = np.polyfit(f, phi1, 1)                       # sign+scale align for display
        ax.plot(z, phi1, color="C0", lw=3, alpha=0.5, label="target $\\varphi_1$")
        ax.plot(z, np.polyval(coef, f), color="C3", lw=1.6, ls="--", label="trained encoder")
        ax.set_title(f"{w}\ncorr={r['corr']:.3f}   $\\nu$={r['nu_trained']:.2f}", fontsize=9)
        ax.set_xlabel("latent $z$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("chart value")
    axes[0].legend(fontsize=7)
    pb = d["partB"]
    fig.suptitle(
        "E8: a gradient-trained encoder reaches the slow-eigenfunction chart "
        f"(recovery corr 0.96–1.00; 2-D rotation partial, Procrustes {pb['procrustes_err']:.2f})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = HERE / "reports" / "e8_trained_encoder.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
