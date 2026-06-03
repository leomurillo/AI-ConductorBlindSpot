"""
E9 — The Irreversible Blind Spot: symmetric SSL is blind to the arrow of time
================================================================================

CLAIM UNDER TEST  (a SECOND blind spot, of the same family as CBS)
-----------------------------------------------------------------
The whole paper, and the gauge program behind it, lives on ONE axis of
blindness: an order-2 instrument cannot read an order-3 fact (the cumulant
axis). This certificate exhibits a DIFFERENT axis on the SAME ring object: an
instrument built from a SYMMETRIC quadratic form cannot read an ANTISYMMETRIC
fact (the reversibility axis). The fact it cannot read is the arrow of time.

Setup. Take the conductor blind spot's own object — the cycle Z/n — and put a
NET DRIFT on it: a Markov walk that steps forward with probability q and
backward with probability b, with q != b. Detailed balance is now broken; the
walk has a probability CURRENT (a preferred direction). Its transition operator

        (T psi)(k) = E[psi(X_1) | X_0 = k] = q psi(k+1) + b psi(k-1) + r psi(k)

splits, in L2 of the uniform stationary law, into a symmetric and an
antisymmetric part,

        T = S + A,   S = (T + T*)/2,   A = (T - T*)/2,

where T* is the time-REVERSED operator. S is a plain lazy random walk on the
ring (no arrow). A is the discrete current  A psi = (q-b)/2 [psi(k+1)-psi(k-1)],
nonzero EXACTLY when detailed balance fails. A *is* the arrow of time.

THE THEOREM, made arithmetic (Irreversible Blind Spot)
------------------------------------------------------
For ANY real encoder h, A is skew-adjoint so the quadratic form  <h, A h> = 0,
hence

        <h, T h>  =  <h, S h>      (EXACTLY, for every h).

The single-encoder alignment objective is  E||h(z')-h(z)||^2 = sum_i 2(1-<h_i,T h_i>),
so it is a function of S ALONE. Consequence, certified below:

  (1) The ENTIRE loss landscape is identical on a world and its time-reverse
      (q,b) <-> (b,q): swapping the arrow leaves S, hence every encoder's loss,
      bit-for-bit unchanged — while T itself changes by 2A != 0. The arrow is in
      the KERNEL of the objective. No single-encoder Siamese recipe can recover it.

  (2) The arrow lives in the IMAGINARY part of the spectrum. The slow eigenvalue
      is  lambda_1 = alpha + i*beta  with  alpha = r+(q+b)cos(2pi/n)  (what the
      objective sees) and  beta = (q-b) sin(2pi/n)  (the arrow, sign = drift).
      The single-encoder objective uses Re(lambda_1) only and discards beta.

  (3) The RESOLUTION is a predictor. A two-encoder / predictive objective learns
      the operator T itself (not just its quadratic form): on the slow subspace
      its optimum is the 2x2 block  B = [[alpha, beta],[-beta, alpha]], whose
      ANTISYMMETRIC part beta*J is the arrow. Under reversal B -> B^T (the flow
      spins the other way), so the predictor SEPARATES the arrow exactly where
      the single encoder is blind. Corollary: the predictor network of
      BYOL/JEPA/SimSiam is not merely an anti-collapse trick — it is the minimal
      mechanism that lifts the objective out of the irreversible blind spot.

  (4) CONTROL: at detailed balance (q=b) we have A=0, beta=0, B symmetric — the
      predictor and the single encoder agree and the blind spot vanishes. The
      blindness is switched on by, and only by, irreversibility.

This is the CBS ring with a current added; the blindness is on the
symmetric/antisymmetric (reversibility) axis instead of the order-2/order-3
(cumulant) axis. Same object, sibling theorem.

Everything is closed-form circulant linear algebra: exact up to LAPACK eps, no
randomness in the operators (the only seed is the random-encoder basket that
witnesses the identity for arbitrary h). Self-checking; exits nonzero on failure.

Run:  python empirical/apex_recovery/e9_irreversible_ring.py    (py ... for the figure)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)

TOL = 1e-12  # machine-precision identities (LAPACK eps headroom)


# ---------------------------------------------------------------------------
# 1. The drift ring on Z/n and its symmetric / antisymmetric split.
# ---------------------------------------------------------------------------


def drift_ring(n: int, q: float, b: float) -> np.ndarray:
    """
    Row-stochastic transition matrix T (T[i,j]=P(i->j)) of the drift walk on
    Z/n: forward prob q, backward prob b, stay r = 1-q-b. The uniform law is
    stationary (T is doubly stochastic), and detailed balance holds iff q == b.
    """
    r = 1.0 - q - b
    if q < 0 or b < 0 or r < -1e-15:
        raise ValueError("need q,b >= 0 and q+b <= 1")
    T = np.zeros((n, n))
    for k in range(n):
        T[k, (k + 1) % n] += q
        T[k, (k - 1) % n] += b
        T[k, k] += r
    return T


def sym_anti(T: np.ndarray):
    """
    Split T into symmetric S and antisymmetric A in L2(uniform pi). Because pi
    is uniform, the L2(pi) adjoint is the matrix transpose, so
        S = (T + T^T)/2   (the reversible part: a lazy random walk),
        A = (T - T^T)/2   (the irreversible current; A^T = -A).
    """
    return 0.5 * (T + T.T), 0.5 * (T - T.T)


def pi_uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def quad(pi: np.ndarray, h: np.ndarray, M: np.ndarray) -> float:
    """pi-weighted quadratic form <h, M h>_pi = sum_i pi_i h_i (M h)_i."""
    return float(np.sum(pi * h * (M @ h)))


def op_matrix(pi: np.ndarray, M: np.ndarray, basis) -> np.ndarray:
    """
    Matrix of the operator M in an L2(pi)-orthonormal basis {e_a}:
    B[a,b] = <e_a, M e_b>_pi, so that M e_b = sum_a B[a,b] e_a. This is exactly
    the optimal linear predictor of e(z') from e(z) restricted to span{e_a}
    (with the e_a whitened): P* = E[e(z') e(z)^T] has these entries.
    """
    d = len(basis)
    B = np.zeros((d, d))
    for a in range(d):
        for c in range(d):
            B[a, c] = float(np.sum(pi * basis[a] * (M @ basis[c])))
    return B


def slow_pair(n: int):
    """
    The m=1 Fourier modes, L2(uniform-pi)-orthonormal: the slow eigenspace of S.
    <cos,cos>_pi = <sin,sin>_pi = 1/2 on the ring, so scale by sqrt(2).
    """
    k = np.arange(n)
    c = np.sqrt(2.0) * np.cos(2 * np.pi * k / n)
    s = np.sqrt(2.0) * np.sin(2 * np.pi * k / n)
    return c, s


def slow_eigenvalue(n: int, q: float, b: float):
    """Closed-form slow eigenvalue lambda_1 = alpha + i*beta of the drift ring."""
    r = 1.0 - q - b
    phi = 2 * np.pi / n
    alpha = r + (q + b) * np.cos(phi)          # = Re lambda_1  (what S sees)
    beta = (q - b) * np.sin(phi)               # = Im lambda_1  (the arrow)
    return alpha, beta


# ---------------------------------------------------------------------------
# 2. The certificate.
# ---------------------------------------------------------------------------


def main():
    print("=" * 78)
    print("E9  The Irreversible Blind Spot — symmetric SSL is blind to the arrow")
    print("=" * 78)
    print(
        "\nThe CBS ring Z/n with a NET DRIFT (forward q != backward b). The single-\n"
        "encoder alignment objective is a function of the SYMMETRIC part S of the\n"
        "transition operator only; the antisymmetric part A — the probability\n"
        "current, the arrow of time — is in its kernel. A predictor recovers it.\n"
    )

    report = {"experiment": "E9_irreversible_blind_spot", "checks": {}, "cases": []}
    checks = report["checks"]

    # Canonical world: a CBS modulus with a forward drift.
    n, q, b = 12, 0.50, 0.20
    pi = pi_uniform(n)
    T = drift_ring(n, q, b)
    S, A = sym_anti(T)
    normA = float(np.linalg.norm(A))
    print(f"World: ring Z/{n}, forward q={q}, backward b={b}, stay r={1-q-b:.2f}.")
    print(f"  ||A||_F = {normA:.6f}  (= 0 iff detailed balance; here the current is on).")

    # --- Test 1: the exact blind-spot identity <h,Th> = <h,Sh>, <h,Ah> = 0 ----
    print("\n[1] Blind-spot identity (for ARBITRARY real encoders h):")
    rng = np.random.default_rng(0)
    basket = [rng.standard_normal(n) for _ in range(200)]
    basket += [slow_pair(n)[0], slow_pair(n)[1]]  # include the actual slow modes
    id_resid = max(abs(quad(pi, h, T) - quad(pi, h, S)) for h in basket)
    anti_resid = max(abs(quad(pi, h, A)) for h in basket)
    print(f"    max | <h,Th> - <h,Sh> |  over 202 encoders = {id_resid:.2e}")
    print(f"    max | <h,Ah> |            over 202 encoders = {anti_resid:.2e}")
    print("    => the objective sees S only; A contributes nothing to any h.")
    checks["identity_resid"] = id_resid
    checks["antisym_quadform_resid"] = anti_resid

    # --- Test 2: the WHOLE loss landscape is reversal-invariant ---------------
    print("\n[2] Time-reversal degeneracy of the single-encoder objective:")
    T_rev = drift_ring(n, b, q)  # swap drift = reverse the arrow
    S_rev, A_rev = sym_anti(T_rev)
    dS = float(np.linalg.norm(S - S_rev))      # must be 0: S is reversal-invariant
    dT = float(np.linalg.norm(T - T_rev))      # must be > 0: the worlds differ
    # every encoder's alignment loss is identical on the two worlds:
    loss = lambda H, M: sum(2.0 * (1.0 - quad(pi, H[:, i], M)) for i in range(H.shape[1]))
    Hbasket = [np.stack([rng.standard_normal(n), rng.standard_normal(n)], axis=1) for _ in range(50)]
    loss_gap = max(abs(loss(H, T) - loss(H, T_rev)) for H in Hbasket)
    print(f"    ||S_fwd - S_rev||_F = {dS:.2e}   (the instrument's input is identical)")
    print(f"    ||T_fwd - T_rev||_F = {dT:.4f}    (= 2||A||; the WORLDS are different)")
    print(f"    max | loss_fwd(h) - loss_rev(h) | over 50 encoders = {loss_gap:.2e}")
    print("    => a world and its time-reverse give bit-for-bit the same landscape:")
    print("       no single-encoder representation can encode the arrow.")
    checks["S_reversal_diff"] = dS
    checks["T_reversal_diff"] = dT
    checks["loss_reversal_gap"] = loss_gap

    # --- Test 3: the arrow is the imaginary part of the slow eigenvalue -------
    print("\n[3] The arrow lives in Im(lambda_1):")
    alpha, beta = slow_eigenvalue(n, q, b)
    evals = np.linalg.eigvals(T)               # complex spectrum of the real, non-normal-looking T
    # nearest spectrum point to the closed-form alpha+i*beta:
    spec_match = float(np.min(np.abs(evals - (alpha + 1j * beta))))
    print(f"    closed form : lambda_1 = {alpha:.6f} + i*({beta:+.6f})")
    print(f"    Re = alpha = {alpha:.6f}  (what the single-encoder objective uses)")
    print(f"    Im = beta  = {beta:+.6f}  (the arrow; sign = drift direction = sign(q-b))")
    print(f"    distance to numerical spectrum of T = {spec_match:.2e}")
    checks["slow_eig_closedform_vs_numeric"] = spec_match
    checks["arrow_sign_matches_drift"] = bool(np.sign(beta) == np.sign(q - b))

    # --- Test 4: the predictor recovers the arrow (2x2 block B) ---------------
    print("\n[4] The predictor block B (two-encoder / predictive optimum):")
    c, s = slow_pair(n)
    B = op_matrix(pi, T, [c, s])               # numeric predictor optimum on the slow subspace
    B_analytic = np.array([[alpha, beta], [-beta, alpha]])
    B_err = float(np.linalg.norm(B - B_analytic))
    B_sym = 0.5 * (B + B.T)
    B_anti = 0.5 * (B - B.T)
    print(f"    recovered B =\n{np.array2string(B, prefix='        ', precision=6)}")
    print(f"    || B - [[a,beta],[-beta,a]] ||_F = {B_err:.2e}")
    print(f"    symmetric part  = alpha*I   (||B_sym - alpha I|| = {np.linalg.norm(B_sym - alpha*np.eye(2)):.2e})"
          "   <- all the single encoder can see")
    print(f"    antisym part    = beta*J,  beta = {B_anti[0,1]:+.6f}   <- THE ARROW")
    # predictor under reversal: B -> B^T; it separates fwd from rev:
    B_rev = op_matrix(pi, T_rev, [c, s])
    B_reversal_diff = float(np.linalg.norm(B - B_rev))
    print(f"    under reversal B -> B^T:  ||B_fwd - B_rev||_F = {B_reversal_diff:.4f}  (> 0!)")
    print(f"    => predictor reads the arrow (sign = {int(np.sign(B_anti[0,1]))}) where the")
    print(f"       single encoder cannot (||S_fwd - S_rev|| = {dS:.0e}).")
    checks["predictor_block_err"] = B_err
    checks["predictor_sym_is_alpha_I"] = float(np.linalg.norm(B_sym - alpha * np.eye(2)))
    checks["predictor_reversal_diff"] = B_reversal_diff
    report["cases"].append(dict(name="canonical", n=n, q=q, b=b,
                                alpha=alpha, beta=beta, normA=normA,
                                B=B.tolist(), B_reversal_diff=B_reversal_diff))

    # --- Test 5: detailed-balance control — the blind spot switches OFF -------
    print("\n[5] Detailed-balance control (q = b): the blind spot vanishes.")
    q0 = b0 = 0.35
    T0 = drift_ring(n, q0, b0)
    _, A0 = sym_anti(T0)
    a0, beta0 = slow_eigenvalue(n, q0, b0)
    B0 = op_matrix(pi, T0, [c, s])
    B0_anti = float(np.linalg.norm(0.5 * (B0 - B0.T)))
    print(f"    ||A||_F = {np.linalg.norm(A0):.2e},  beta = {beta0:.2e},  "
          f"||antisym(B)||_F = {B0_anti:.2e}")
    print("    => reversible world: predictor = single encoder, no arrow to miss.")
    checks["reversible_normA"] = float(np.linalg.norm(A0))
    checks["reversible_predictor_antisym"] = B0_anti

    # --- Test 6: the discarded current — objective flat, arrow grows ----------
    print("\n[6] Sweep at fixed total rate q+b=0.7: objective flat, arrow linear.")
    deltas = np.linspace(-0.7, 0.7, 29)         # asymmetry q-b
    objective, arrow = [], []
    for d_ in deltas:
        qq, bb = (0.7 + d_) / 2, (0.7 - d_) / 2
        aa, bbeta = slow_eigenvalue(n, qq, bb)
        objective.append(2.0 * (1.0 - aa))      # single-encoder objective on the slow mode
        arrow.append(abs(bbeta))
    obj_spread = float(np.ptp(objective))       # peak-to-peak; must be ~0
    arrow_range = float(np.ptp(arrow))          # must be > 0
    print(f"    single-encoder objective spread across the sweep = {obj_spread:.2e}  (flat)")
    print(f"    arrow |beta| range across the sweep              = {arrow_range:.4f}  (rises)")
    checks["sweep_objective_spread"] = obj_spread
    checks["sweep_arrow_range"] = arrow_range

    # --- Test 7: it is the CBS ring — same moduli ----------------------------
    print("\n[7] Same statement on the CBS moduli n in {6,8,12,18,30}:")
    ring_rows = []
    for nn in (6, 8, 12, 18, 30):
        pin = pi_uniform(nn)
        Tn = drift_ring(nn, 0.5, 0.2)
        Sn, An = sym_anti(Tn)
        cn, sn = slow_pair(nn)
        # identity on a random basket:
        bk = [np.random.default_rng(nn).standard_normal(nn) for _ in range(20)]
        idn = max(abs(quad(pin, h, Tn) - quad(pin, h, Sn)) for h in bk)
        an, bn = slow_eigenvalue(nn, 0.5, 0.2)
        Bn = op_matrix(pin, Tn, [cn, sn])
        bn_num = 0.5 * (Bn - Bn.T)[0, 1]
        ring_rows.append(dict(n=nn, identity_resid=idn, beta=bn,
                              beta_from_predictor=float(bn_num)))
        print(f"    n={nn:2d}:  identity resid={idn:.1e},  arrow beta={bn:+.5f} "
              f"(predictor reads {bn_num:+.5f})")
    report["cases"].append(dict(name="cbs_moduli", rows=ring_rows))

    # ---- self-check gate -----------------------------------------------------
    ok = (
        checks["identity_resid"] < TOL
        and checks["antisym_quadform_resid"] < TOL
        and checks["S_reversal_diff"] < TOL
        and checks["T_reversal_diff"] > 1e-6
        and checks["loss_reversal_gap"] < TOL
        and checks["slow_eig_closedform_vs_numeric"] < 1e-9
        and checks["arrow_sign_matches_drift"]
        and checks["predictor_block_err"] < 1e-9
        and checks["predictor_sym_is_alpha_I"] < 1e-9
        and checks["predictor_reversal_diff"] > 1e-6
        and checks["reversible_normA"] < 1e-15
        and checks["reversible_predictor_antisym"] < TOL
        and checks["sweep_objective_spread"] < TOL
        and checks["sweep_arrow_range"] > 1e-3
        and all(r["identity_resid"] < TOL for r in ring_rows)
    )
    report["passed"] = bool(ok)

    # ---- optional figure -----------------------------------------------------
    fig_path = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

        # Left: the discarded current — flat objective, rising arrow.
        axL.plot(deltas, objective, "o-", color="#444", label="single-encoder objective  2(1-Re $\\lambda_1$)")
        axL.plot(deltas, arrow, "s-", color="#c0392b", label="arrow magnitude  |Im $\\lambda_1$| = |$\\beta$|")
        axL.axvline(0, color="#888", ls=":", lw=1)
        axL.set_xlabel("drift asymmetry  q - b   (fixed q+b = 0.7)")
        axL.set_title("The instrument is flat; the arrow is not")
        axL.legend(loc="upper center", fontsize=8)
        axL.grid(alpha=0.3)

        # Right: the recovered ring embedding with the predictor's rotational flow.
        k = np.arange(n)
        px, py = np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n)
        axR.plot(np.r_[px, px[0]], np.r_[py, py[0]], "-", color="#bbb", lw=1, zorder=1)
        axR.scatter(px, py, c="#2c3e50", s=30, zorder=3)
        # predictor maps embedding (cos,sin) -> (alpha cos - beta sin, beta cos + alpha sin)
        for (ex, ey), col, lab in [((alpha, beta), "#c0392b", "forward (q>b)"),
                                   ((alpha, -beta), "#2980b9", "reversed (q<b)")]:
            nx = ex * px - ey * py
            ny = ey * px + ex * py
            axR.quiver(px, py, nx - px, ny - py, angles="xy", scale_units="xy",
                       scale=1, color=col, width=0.005, zorder=2, label=lab)
        axR.set_aspect("equal")
        axR.set_title("Predictor flow on the ring embedding = the arrow")
        axR.legend(loc="upper right", fontsize=8)
        axR.set_xticks([]); axR.set_yticks([])

        fig.suptitle(f"The Irreversible Blind Spot on Z/{n}:  symmetric objective blind, predictor sees the current",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = REPORTS / "e9_irreversible_ring.png"
        fig.savefig(fig_path, dpi=130)
        print(f"\nfigure -> {fig_path}")
    except ImportError:
        print("\n(matplotlib not installed; skipping figure)")

    report["figure"] = str(fig_path) if fig_path else None
    rpt = REPORTS / "e9_irreversible_ring.json"
    rpt.write_text(json.dumps(report, indent=2))
    print(f"report -> {rpt}")

    print("\n" + "=" * 78)
    print("E9: ALL CHECKS PASSED" if ok else "E9: SOME CHECKS FAILED")
    print("=" * 78)
    print("Takeaway: a symmetric (single-encoder) objective is a function of S = the")
    print("reversible part of the dynamics ONLY; the antisymmetric current A — the")
    print("arrow of time — is in its kernel and is recovered by a predictor. This is")
    print("the CBS ring with a current, and the blind spot is on the reversibility")
    print("axis instead of the cumulant axis. Same object, sibling theorem.")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
