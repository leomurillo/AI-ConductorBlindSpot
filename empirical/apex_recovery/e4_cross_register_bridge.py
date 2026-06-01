"""
E4 — The cross-register bridge, measured  (turns Section 3.4 from argued to measured)
================================================================================

WHAT THIS MEASURES
------------------
Section 3.4 claims the two towers are "two registers of one boundary":

  (D) DYNAMICAL  — how curved the recovery map phi_1 is (the transition-operator
                   slow eigenfunction). Quantified by  nu_D = 1 - R^2 of the best
                   affine fit of phi_1.  nu_D = 0  iff  Gaussian.
  (S) DISTRIBUTIONAL — the cumulant tower of the stationary law. Quantified by the
                   normalized cumulants: skewness  k3 = kappa_3/sigma^3  (order 3)
                   and excess kurtosis  k4 = kappa_4/sigma^4  (order 4).

The honest, measurable statement (and the subtlety a careless bridge would miss):
the recovery-map curvature tracks the LEADING NONZERO cumulant, order by order, and

        nu_D  is proportional to  (leading normalized cumulant)^2   near the Gaussian.

  * For a SKEWED world the leading rung is order 3:  nu_D ∝ k3^2.   <-- the CBS-cubic order
  * For a SYMMETRIC non-Gaussian world  k3 = 0  exactly, and the leading rung is
    order 4:  nu_D ∝ k4^2.

Both registers vanish together at the Gaussian; off it they rise together, locked
by a square law. That is the bridge, measured.

HOW (exact, no chain to diagonalise)
------------------------------------
We use the intrinsic-ness verified in e1: observe a Gaussian world g ~ N(0,1)
through a fixed monotone warp  z = T_alpha(g).  In the continuous (OU) limit the
recovery map in the observed coordinate is EXACTLY the inverse warp,
phi_1(z) = T_alpha^{-1}(z) = g, so the recovery-map nonlinearity is just the
nonlinearity of the warp, nu_D = 1 - corr(g, z)^2. All expectations over g~N(0,1)
are computed by Gauss-Hermite quadrature (machine accuracy). Two warp families:

    skew family:       z = (exp(alpha g) - 1)/alpha        (skewed; k3 != 0)
    symmetric family:  z = sinh(alpha g)/alpha             (symmetric; k3 = 0, k4 != 0)

both reduce to z = g (Gaussian) as alpha -> 0.

A short finite coda (Part C) records the one place the scalar and tensor order-3
objects part company: on the cyclic ring index the Amari-Chentsov *tensor* cubic
survives at the symmetric uniform point (cross-packet components, the CBS counts),
where the scalar skewness of the label law is zero -- the ring keeps order 3 alive.

Run:  python empirical/apex_recovery/e4_cross_register_bridge.py
Deterministic (quadrature); < 1 second.
"""

from __future__ import annotations

import json
import sys
from math import gcd, sqrt
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252 and choke on math glyphs; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)

# Gauss-Hermite (probabilist's) nodes/weights for E[f(g)], g ~ N(0,1):
#   E[f] = (1/sqrt(2*pi)) * sum_i w_i f(x_i),  with hermegauss the physicists'
# variant; we use the e^{-x^2} convention and rescale by sqrt(2).
_NODES, _WTS = np.polynomial.hermite.hermgauss(96)
_X = _NODES * np.sqrt(2.0)        # nodes for weight e^{-x^2/2}
_W = _WTS / np.sqrt(np.pi)        # normalized so sum _W = 1  (= E over N(0,1))


def E(f_vals):
    """Expectation over g ~ N(0,1) by Gauss-Hermite quadrature."""
    return float(np.sum(_W * f_vals))


def warp(family: str, alpha: float, g: np.ndarray) -> np.ndarray:
    """Monotone warp z = T_alpha(g); -> g as alpha -> 0."""
    if family == "skew":
        return np.expm1(alpha * g) / alpha           # (e^{ag}-1)/a
    if family == "symmetric":
        return np.sinh(alpha * g) / alpha            # sinh(ag)/a
    raise ValueError(family)


def register_numbers(family: str, alpha: float) -> dict:
    """Compute nu_D (dynamical) and k3, k4 (distributional) for one world."""
    g = _X
    z = warp(family, alpha, g)
    mz = E(z)
    zc = z - mz
    m2 = E(zc**2)
    m3 = E(zc**3)
    m4 = E(zc**4)
    sigma = sqrt(m2)
    k3 = m3 / sigma**3                 # skewness
    k4 = m4 / sigma**4 - 3.0           # excess kurtosis
    # recovery-map nonlinearity: nu_D = 1 - corr(g, z)^2  (phi_1 = g exactly)
    cov_gz = E(g * z) - E(g) * mz
    corr2 = cov_gz**2 / (1.0 * m2)
    nu_D = 1.0 - corr2
    return dict(alpha=alpha, k3=k3, k4=k4, nu_D=nu_D, sigma=sigma)


def sweep(family: str, alphas):
    return [register_numbers(family, a) for a in alphas]


def leading_square_law(rows, key, n_near=3):
    """
    The measured statement is a LEADING-ORDER square law: nu_D / (cumulant)^2
    converges to a positive constant as the world approaches Gaussian, with
    higher-order corrections at large departure. We report the limiting ratio
    (smallest perturbation) and its stability over the near-Gaussian range.
    """
    ratios = [r["nu_D"] / r[key] ** 2 for r in rows]
    c0 = ratios[0]                         # smallest alpha = closest to Gaussian
    near = ratios[:n_near]
    spread = (max(near) - min(near)) / c0  # relative drift over near-Gaussian pts
    return c0, spread, ratios


def cbs_cross_packet_count(n: int) -> int:
    cnt = 0
    for k in range(1, n):
        for l in range(1, n):
            m = (-k - l) % n
            if m == 0:
                continue
            if len({n // gcd(k, n), n // gcd(l, n), n // gcd(m, n)}) > 1:
                cnt += 1
    return cnt


def main():
    print("=" * 78)
    print("E4  The cross-register bridge, measured  (Section 3.4)")
    print("=" * 78)
    print(
        "\nClaim: the recovery-map curvature nu_D (dynamical register) tracks the\n"
        "LEADING nonzero cumulant of the stationary law (distributional register),\n"
        "order by order, as  nu_D ~ (leading cumulant)^2.  Both vanish at Gaussian.\n"
    )

    # log-spaced sweep: dense near the Gaussian, filling the log-log axis evenly.
    alphas = [round(a, 4) for a in np.geomspace(0.015, 0.85, 16)]
    report = {"experiment": "E4_cross_register_bridge", "families": {}}

    # ---- Part A: skew family -> order 3 lights up -----------------------
    print("Part A — SKEW family  z = (e^{a g} - 1)/a   (asymmetric: order-3 rung)")
    rows = sweep("skew", alphas)
    print(f"  {'alpha':>6} {'skew k3':>10} {'exkurt k4':>10} {'nu_D':>12} {'nu_D/k3^2':>11}")
    for r in rows:
        ratio = r["nu_D"] / r["k3"] ** 2 if r["k3"] != 0 else float("nan")
        print(f"  {r['alpha']:6.2f} {r['k3']:10.4f} {r['k4']:10.4f} {r['nu_D']:12.3e} {ratio:11.4f}")
    c3, spread3, _ = leading_square_law(rows, "k3")
    print(f"  leading square law:  nu_D/k3^2 -> {c3:.4f} as alpha->0 "
          f"(stable to {100*spread3:.1f}% over the near-Gaussian range)")
    print("  -> nu_D and the skewness k3 BOTH vanish at the Gaussian and rise")
    print("     together, locked near-Gaussian by  nu_D ~ c*k3^2  (c the limit above).")
    print("     This is the CBS-cubic order. (The ratio drifts at large skew: the")
    print("     square law is the leading-order, near-Gaussian statement.)\n")
    report["families"]["skew"] = dict(rows=rows, leading_ratio=c3,
                                      near_gaussian_spread=spread3, order=3)

    # ---- Part B: symmetric family -> order 4 lights up ------------------
    print("Part B — SYMMETRIC family  z = sinh(a g)/a   (k3 = 0: order-4 rung)")
    rows = sweep("symmetric", alphas)
    print(f"  {'alpha':>6} {'skew k3':>10} {'exkurt k4':>10} {'nu_D':>12} {'nu_D/k4^2':>11}")
    for r in rows:
        ratio = r["nu_D"] / r["k4"] ** 2 if r["k4"] != 0 else float("nan")
        print(f"  {r['alpha']:6.2f} {r['k3']:10.3e} {r['k4']:10.4f} {r['nu_D']:12.3e} {ratio:11.4f}")
    c4, spread4, _ = leading_square_law(rows, "k4")
    print(f"  leading square law:  nu_D/k4^2 -> {c4:.4f} as alpha->0 "
          f"(stable to {100*spread4:.1f}% over the near-Gaussian range)")
    print("  -> here k3 = 0 to machine precision (symmetry); the recovery map still")
    print("     bends, and nu_D ~ c*k4^2 instead. The boundary is order-2 vs the")
    print("     FIRST nonzero rung, whichever order that is.\n")
    report["families"]["symmetric"] = dict(rows=rows, leading_ratio=c4,
                                           near_gaussian_spread=spread4, order=4)

    # ---- Part C: scalar vs tensor on the ring (why CBS is order-3) ------
    print("Part C — the ring keeps order-3 alive (scalar vs tensor cubic).")
    print("  The cyclic label law (uniform on Z/n) is SYMMETRIC, so its scalar")
    print("  skewness is 0 -- by Part B logic an unstructured symmetric world's")
    print("  first rung would be order 4. But CBS's order-3 object is the Amari-")
    print("  Chentsov TENSOR, whose cross-packet components survive at the uniform")
    print("  point via the ring's arithmetic. We contrast the two:")
    print(f"  {'n':>4} {'label skewness':>16} {'AC cross-packet cubic':>24}")
    ring = {}
    for n in (6, 8, 12, 18, 30):
        # scalar skewness of the uniform law on {0..n-1}: 0 by symmetry (exact).
        labels = np.arange(n, dtype=float)
        lc = labels - labels.mean()
        skew_label = float(np.sum(lc**3) / n) / (float(np.sum(lc**2) / n) ** 1.5)
        cnt = cbs_cross_packet_count(n)
        ring[n] = dict(label_skewness=skew_label, cross_packet_cubic=cnt)
        print(f"  {n:>4} {skew_label:16.2e} {cnt:24d}")
    print("  -> scalar skewness 0 (symmetric labels) while the tensor cubic is")
    print("     nonzero (the CBS counts). On a STRUCTURED index, order-3 survives")
    print("     the symmetry that silences the scalar; the ring is what keeps it")
    print("     alive. Same boundary (order-2 completeness), read on a richer index.")
    report["ring_scalar_vs_tensor"] = {str(k): v for k, v in ring.items()}

    # ---- optional figure -----------------------------------------------
    fig_path = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        for ax, fam, kk, lab in [
            (axes[0], "skew", "k3", r"leading cumulant  $|k_3|$  (skewness)"),
            (axes[1], "symmetric", "k4", r"leading cumulant  $|k_4|$  (excess kurtosis)"),
        ]:
            rws = report["families"][fam]["rows"]
            cmag = np.array([abs(r[kk]) for r in rws])
            y = np.array([r["nu_D"] for r in rws])
            o = np.argsort(cmag)
            cmag, y = cmag[o], y[o]
            # near-Gaussian log-log slope (smallest-cumulant half): a falsifiable
            # exponent — the square law predicts 2, linear would be 1, cubic 3.
            half = max(3, len(cmag) // 2)
            slope, _ = np.polyfit(np.log(cmag[:half]), np.log(y[:half]), 1)
            # reference power laws anchored at a near-Gaussian point
            x0, y0 = cmag[1], y[1]
            xs = np.geomspace(cmag.min() * 0.8, cmag.max() * 1.2, 60)
            for p, sty, col, lw, a in [(1, ":", "grey", 1.2, 0.6),
                                        (2, "-", "red", 2.2, 0.9),
                                        (3, "-.", "grey", 1.2, 0.6)]:
                ax.plot(xs, y0 * (xs / x0) ** p, sty, color=col, lw=lw, alpha=a,
                        label={1: "slope 1 (linear)", 2: "slope 2 (square law)",
                               3: "slope 3 (cubic)"}[p])
            ax.scatter(cmag, y, s=44, color="C0", edgecolor="k", linewidth=0.4, zorder=4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(lab)
            ax.set_ylabel(r"recovery-map nonlinearity  $\nu_D$")
            ax.set_title(f"{fam} family (order {report['families'][fam]['order']}):"
                         f"  measured slope $={slope:.2f}$")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.3, which="both")
        fig.suptitle("Log--log: the data pick slope 2 (the square law), not 1 (linear) or 3 (cubic)\n"
                     r"— $\nu_D \propto (\text{leading cumulant})^2$, the two registers locked")
        fig_path = REPORTS / "e4_bridge.png"
        fig.tight_layout()
        fig.savefig(fig_path, dpi=130)
        print(f"\nfigure -> {fig_path}")
    except ImportError:
        print("\n(matplotlib not installed; skipping figure)")

    report["takeaway"] = (
        "nu_D ∝ (leading nonzero cumulant)^2; order 3 for skewed (CBS-cubic order), "
        "order 4 for symmetric; ring structure keeps order-3 alive at the symmetric "
        "point via the Amari-Chentsov tensor's cross-packet components."
    )
    report["figure"] = str(fig_path) if fig_path else None
    rpt = REPORTS / "e4_cross_register_bridge.json"
    rpt.write_text(json.dumps(report, indent=2))
    print(f"report -> {rpt}")
    print("\nMeasured bridge: the dynamical and distributional registers vanish")
    print("together at the Gaussian and rise together off it, locked order-by-order")
    print("by a square law. Section 3.4 is now measured, not only argued.")


if __name__ == "__main__":
    main()
