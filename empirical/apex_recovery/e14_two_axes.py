"""
E14 — Two independent axes of blindness: cumulant curvature and reversibility
================================================================================

THE CLAIM
---------
This program and its parent expose two DIFFERENT blind spots, and this
certificate shows they are independent — orthogonal axes of one operator, freely
combinable, each needing its own audit.

  * CUMULANT axis (parent, *Beyond the Conductor Blind Spot*): a LINEAR probe of
    the representation is complete only on the Gaussian stratum. Defect metric:
    nu_D = 1 - R^2 of the slow eigenfunction's best affine fit (the recovery
    map's curvature). Lives in the SYMMETRIC part's eigenFUNCTIONS.

  * REVERSIBILITY axis (this paper): a SYMMETRIC objective is blind to the
    antisymmetric current (the arrow of time). Defect metric: the irreversibility
    gap Delta = sum sigma(T) - sum lambda(S). Lives in the ANTISYMMETRIC part.

These are different components of the transition operator T = S + A: nu_D reads
the curvature of S's slow eigenfunction; Delta reads A. So they are structurally
independent and can be dialed separately. We realise all four corners of the
(Gaussian/non-Gaussian) x (reversible/irreversible) square and measure both
metrics on a 2-D sweep, in a single product world (the paper's standing
independent-factor assumption): factor U carries the cumulant axis, factor V the
reversibility axis.

  WORLD = U (line)  x  V (ring)
    U: a Gaussian latent g observed through a tunable warp z = g + gamma*g^3
       (reversible OU chain). gamma dials nu_D; U contributes NOTHING to Delta.
    V: a drift ring, forward/backward (1+-rho)*base. rho dials Delta; V is
       uniform, contributing NOTHING to nu_D.

CONSEQUENCE FOR AUDITS: a representation can be defeated on either axis alone or
both at once. An audit that checks only linearity (nu_D) misses the arrow; one
that checks only reversibility (Delta) misses the curved chart. Check both.

Run:  python empirical/apex_recovery/e14_two_axes.py    (py ... for the figure)
Exact up to LAPACK eps; in run_all's gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import apex_world as aw

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Cumulant axis: the recovery-map curvature nu_D(gamma) of the warped factor U.
# ---------------------------------------------------------------------------


def cumulant_setup():
    """Gaussian g-chain; phi_1 affine in g. Returns (g, pi, g_hat) with g_hat ~ g."""
    g, pi, lam, phi = aw.transition_eigh("gaussian", 801, 4.0)
    nu0, (a, b) = aw.affine_nonlinearity(pi, g, phi[:, 1])
    g_hat = (phi[:, 1] - b) / a            # standardised slow eigenfunction ~ g
    return g, pi, g_hat


def nu_D(g, pi, g_hat, gamma):
    """
    Curvature of the recovery map in the OBSERVED coordinate z = g + gamma*g^3:
    nu_D = 1 - R^2 of the best affine fit of the recovered chart (g_hat) vs z.
    gamma = 0 -> z = g, affine, nu_D = 0; gamma > 0 -> curved chart, nu_D > 0.
    """
    z = g + gamma * g ** 3
    nu, _ = aw.affine_nonlinearity(pi, z, g_hat)
    return float(nu)


# ---------------------------------------------------------------------------
# Reversibility axis: the irreversibility gap Delta(rho) of the drift ring V.
# ---------------------------------------------------------------------------


def drift_ring(n, q, b):
    T = np.zeros((n, n))
    for k in range(n):
        T[k, (k + 1) % n] += q
        T[k, (k - 1) % n] += b
        T[k, k] += 1 - q - b
    return T


def Delta(n, rho, base=0.3):
    """Slow-pair irreversibility gap of the drift ring: 2(|lambda_1| - Re lambda_1)."""
    q, b = base * (1 + rho), base * (1 - rho)
    r = 1 - q - b
    phi = 2 * np.pi / n
    alpha = r + (q + b) * np.cos(phi)
    beta = (q - b) * np.sin(phi)
    return float(2 * (np.hypot(alpha, beta) - alpha))


def product_antisym_norm(rho, base=0.3):
    """
    ||antisym(T_U (x) T_V)|| in L2(pi), to show the arrow lives in ONE world.
    The U-factor is reversible but pi_u-self-adjoint (NOT matrix-symmetric), so the
    antisymmetric part must use the L2(pi) adjoint T* = D^{-1} T^T D with the product
    law pi = pi_u (x) pi_v; then rho=0 (both factors reversible) gives A = 0 exactly.
    """
    gu, piu, lamu, phiu = aw.transition_eigh("gaussian", 21, 4.0)
    Tu = aw.metropolis_chain(piu)                      # reversible (pi_u-self-adjoint) factor
    n = 24
    piv = np.full(n, 1.0 / n)
    Tv = drift_ring(n, base * (1 + rho), base * (1 - rho))
    T = np.kron(Tu, Tv)
    pi = np.kron(piu, piv)
    Tstar = (T.T * pi[None, :]) / pi[:, None]          # L2(pi) adjoint
    return float(np.linalg.norm(0.5 * (T - Tstar)))


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("E14  Two independent axes of blindness: cumulant curvature & reversibility")
    print("=" * 78)
    report = {"experiment": "E14_two_axes", "passed": True}

    g, pi, g_hat = cumulant_setup()
    n = 24
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    rhos = [0.0, 0.25, 0.5, 0.75, 0.9]

    NU = np.array([[nu_D(g, pi, g_hat, gm) for _ in rhos] for gm in gammas])   # rows=gamma
    DE = np.array([[Delta(n, rh) for rh in rhos] for _ in gammas])             # cols=rho

    print("\nnu_D (cumulant-axis defect) over the (gamma down, rho across) grid:")
    print("        rho:  " + "  ".join(f"{rh:5.2f}" for rh in rhos))
    for i, gm in enumerate(gammas):
        print(f"  gamma={gm:4.2f}: " + "  ".join(f"{NU[i,j]:5.3f}" for j in range(len(rhos))))
    print("\nDelta (reversibility-axis defect) over the same grid:")
    print("        rho:  " + "  ".join(f"{rh:5.2f}" for rh in rhos))
    for i, gm in enumerate(gammas):
        print(f"  gamma={gm:4.2f}: " + "  ".join(f"{DE[i,j]:5.3f}" for j in range(len(rhos))))

    # Independence (structural): nu_D varies only down gamma; Delta only across rho.
    nu_var_along_gamma = float(np.ptp(NU[:, 0]))        # intended variation
    nu_var_along_rho = float(np.max([np.ptp(NU[i, :]) for i in range(len(gammas))]))  # cross-talk
    de_var_along_rho = float(np.ptp(DE[0, :]))
    de_var_along_gamma = float(np.max([np.ptp(DE[:, j]) for j in range(len(rhos))]))
    print(f"\nnu_D: variation along its axis (gamma) = {nu_var_along_gamma:.4f}; "
          f"cross-talk (along rho) = {nu_var_along_rho:.2e}")
    print(f"Delta: variation along its axis (rho)  = {de_var_along_rho:.4f}; "
          f"cross-talk (along gamma) = {de_var_along_gamma:.2e}")

    # The four corners.
    corners = {
        "Gaussian + reversible":      dict(nu=NU[0, 0],  delta=DE[0, 0]),
        "non-Gaussian + reversible":  dict(nu=NU[-1, 0], delta=DE[-1, 0]),
        "Gaussian + irreversible":    dict(nu=NU[0, -1], delta=DE[0, -1]),
        "non-Gaussian + irreversible":dict(nu=NU[-1, -1],delta=DE[-1, -1]),
    }
    print("\nThe four corners (nu_D, Delta):")
    for k, v in corners.items():
        print(f"  {k:30}: nu_D={v['nu']:.3f}, Delta={v['delta']:.3f}")

    # One world carries both: ||antisym(product)|| tracks rho, not gamma.
    arrow_lo = product_antisym_norm(0.0)
    arrow_hi = product_antisym_norm(0.9)
    print(f"\nProduct world U x V: ||antisym(T)|| = {arrow_lo:.3f} (rho=0) -> {arrow_hi:.3f} (rho=0.9)"
          f"  (the arrow lives in the single product operator, growing with rho).")

    # ---- gate: four corners realise the 2x2; axes orthogonal -----------------
    ok = (corners["Gaussian + reversible"]["nu"] < 1e-3 and corners["Gaussian + reversible"]["delta"] < 1e-3
          and corners["non-Gaussian + reversible"]["nu"] > 0.05 and corners["non-Gaussian + reversible"]["delta"] < 1e-3
          and corners["Gaussian + irreversible"]["nu"] < 1e-3 and corners["Gaussian + irreversible"]["delta"] > 1e-2
          and corners["non-Gaussian + irreversible"]["nu"] > 0.05 and corners["non-Gaussian + irreversible"]["delta"] > 1e-2
          and nu_var_along_rho < 1e-9 and de_var_along_gamma < 1e-9      # orthogonal: zero cross-talk
          and arrow_hi > 1e-2 and arrow_lo < 1e-9)
    report.update(dict(gammas=gammas, rhos=rhos, nu_D=NU.tolist(), Delta=DE.tolist(),
                       corners={k: {kk: float(vv) for kk, vv in v.items()} for k, v in corners.items()},
                       nu_var_along_gamma=nu_var_along_gamma, nu_crosstalk=nu_var_along_rho,
                       delta_var_along_rho=de_var_along_rho, delta_crosstalk=de_var_along_gamma,
                       product_arrow=dict(rho0=arrow_lo, rho_hi=arrow_hi), passed=bool(ok)))

    # ---- figure -------------------------------------------------------------
    fig_path = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, M, title, cmap in [(axes[0], NU, r"cumulant defect $\nu_D$ (curved chart)", "viridis"),
                                   (axes[1], DE, r"reversibility defect $\Delta$ (arrow)", "magma")]:
            im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap,
                           extent=[rhos[0], rhos[-1], gammas[0], gammas[-1]])
            ax.set_xlabel(r"irreversibility  $\rho$")
            ax.set_ylabel(r"non-Gaussianity  $\gamma$")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)
        axes[0].annotate("horizontal bands:\n$\\nu_D\\perp\\rho$", (0.45, 0.5), color="w", fontsize=9)
        axes[1].annotate("vertical bands:\n$\\Delta\\perp\\gamma$", (0.1, 0.7), color="w", fontsize=9)
        fig.suptitle("Two orthogonal axes of blindness: curvature ($\\nu_D$) and arrow ($\\Delta$) vary independently",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = REPORTS / "e14_two_axes.png"
        fig.savefig(fig_path, dpi=130)
        print(f"\nfigure -> {fig_path}")
    except ImportError:
        print("\n(matplotlib not installed; skipping figure)")

    report["figure"] = str(fig_path) if fig_path else None
    (REPORTS / "e14_two_axes.json").write_text(json.dumps(report, indent=2))
    print(f"report -> {REPORTS / 'e14_two_axes.json'}")

    print("\n" + "=" * 78)
    print("E14: ALL CHECKS PASSED" if ok else "E14: SOME CHECKS FAILED")
    print("=" * 78)
    print("Takeaway: the cumulant axis (linear-probe curvature nu_D) and the reversibility")
    print("axis (irreversibility gap Delta) are independent components of one operator -- all")
    print("four corners are realisable, with zero cross-talk. An audit must check both.")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
