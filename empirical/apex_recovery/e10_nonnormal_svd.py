"""
E10 — The general (non-normal) Irreversible Blind Spot: input chart != output chart
================================================================================

WHAT THIS ADDS TO E9
--------------------
E9 lives on the symmetric ring, where the drift operator T is NORMAL
(T T* = T* T): there the arrow of time is the IMAGINARY part of the spectrum,
and the predictor block is a clean rotation in a SINGLE recovered chart. This
certificate removes the normality crutch. A genuinely non-normal transition —
here a "conveyor" on Z/n with position-dependent forward rates, the maximally
irreversible chain — separates the LEFT and RIGHT singular functions of T:

    T v_k = sigma_k u_k     (right chart v = "context/input" directions of z;
                             left chart  u = "target/output" directions of z').

For a normal operator u_k and v_k span the same subspace (E9's rotation lives
inside it). For a non-normal one they DO NOT: the coordinate you should encode
z in is different from the coordinate you read z' in. That split is the arrow of
time made geometric, and it is exactly what a single (shared) encoder cannot
represent.

THE THREE CLAIMS (all general — no normality, no reversibility assumed)
----------------------------------------------------------------------
  (1) Theorem 1 (E9) is UNCONDITIONAL. <h,Th> = <h,Sh> for every real h, normal
      or not: the single-encoder objective still sees only the symmetric part S.
      Its optimum is the top eigenspace of S (a reversible diffusion's chart).

  (2) The two-encoder / predictive optimum is the SVD of T. Maximising
      sum_i <g_i, T f_i> over L2(pi)-orthonormal {f_i},{g_i} equals sum_i sigma_i
      (top singular values), attained at f_i = v_i (right), g_i = u_i (left) —
      the asymmetric, two-chart recovery. For a NON-NORMAL T the recovered input
      chart span{v} and output chart span{u} differ (principal angle > 0); for
      the normal ring they coincide.

  (3) The IRREVERSIBILITY GAP unifies E9 and E10. The single encoder captures
      sum lambda_i(S); the two encoder captures sum sigma_i(T); and by
      Fan–Hoffman lambda_i(S) <= sigma_i(T), so

          Delta := sum_i sigma_i(T) - sum_i lambda_i(S)  >=  0,

      is the predictable correlation the single encoder LEAVES ON THE TABLE. It
      is 0 iff detailed balance holds, and positive whenever there is a current —
      whether that current shows up as E9's imaginary spectrum (normal ring,
      sigma = |lambda| > Re lambda) or as E10's chart split (non-normal). Same
      gap, two mechanisms.

  (4) Time reversal T -> T* swaps the charts: right singular functions of T* are
      the left singular functions of T. The single encoder, seeing only S, is
      invariant. The asymmetry u != v IS the arrow.

Exact up to LAPACK eps (numpy SVD / eigh in the pi-weighted inner product); no
randomness in the operators (the only seed is the random whitened-pair basket
that witnesses the variational optima). Self-checking; in run_all's gate.

Run:  python empirical/apex_recovery/e10_nonnormal_svd.py    (py ... for the figure)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)

TOL = 1e-12
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1. Two worlds on Z/n: the non-normal conveyor and (for contrast) the drift ring.
# ---------------------------------------------------------------------------


def conveyor(n: int, a: np.ndarray):
    """
    The maximally irreversible chain on Z/n: from k step FORWARD with rate a_k,
    else stay. Pure forward => no reversible part except the symmetrisation.
    Position-dependent a_k makes T NON-NORMAL. Stationary law is closed form:
    pi_k a_k = const (the current) => pi_k ∝ 1/a_k.
    """
    T = np.zeros((n, n))
    for k in range(n):
        T[k, (k + 1) % n] = a[k]
        T[k, k] = 1.0 - a[k]
    pi = (1.0 / a) / np.sum(1.0 / a)
    return T, pi


def drift_ring(n: int, q: float, b: float):
    """E9's homogeneous drift ring: circulant => NORMAL. Uniform stationary law."""
    r = 1.0 - q - b
    T = np.zeros((n, n))
    for k in range(n):
        T[k, (k + 1) % n] += q
        T[k, (k - 1) % n] += b
        T[k, k] += r
    return T, np.full(n, 1.0 / n)


# ---------------------------------------------------------------------------
# 2. Operator algebra in the pi-weighted inner product L2(pi).
# ---------------------------------------------------------------------------


def adjoint(T, pi):
    """L2(pi) adjoint T* = D^{-1} T^T D  (time-reversed conditional expectation)."""
    return (T.T * pi[None, :]) / pi[:, None]


def sym_anti(T, pi):
    Ts = adjoint(T, pi)
    return 0.5 * (T + Ts), 0.5 * (T - Ts)


def nonnormality(T, pi):
    """|| T T* - T* T ||  measured in L2(pi) (via the symmetrised matrix M)."""
    d = np.sqrt(pi)
    M = (d[:, None]) * T * (1.0 / d[None, :])
    return float(np.linalg.norm(M @ M.T - M.T @ M))


def _meanzero_projector(pi):
    """Projector (in M-space) killing the trivial constant mode sqrt(pi)."""
    w = np.sqrt(pi)
    return np.eye(len(pi)) - np.outer(w, w), w


def svd_Lpi(T, pi):
    """
    SVD of T in L2(pi), restricted to the mean-zero subspace:
        T v_k = sigma_k u_k,   {u_k},{v_k} L2(pi)-orthonormal, mean-zero.
    Returns sigma (desc), U (left, columns are functions on the grid), V (right).
    """
    d = np.sqrt(pi)
    M = (d[:, None]) * T * (1.0 / d[None, :])
    Q, _ = _meanzero_projector(pi)
    M0 = Q @ M @ Q
    Ut, sig, Vt = np.linalg.svd(M0)
    U = Ut / d[:, None]           # back to L2(pi) functions
    V = Vt.T / d[:, None]
    return sig, U, V


def eig_sym_Lpi(S, pi):
    """
    Eigen-decomposition of the self-adjoint S in L2(pi), mean-zero:
    eigenvalues (desc) and eigenfunctions (columns), L2(pi)-orthonormal.
    """
    d = np.sqrt(pi)
    Sm = (d[:, None]) * S * (1.0 / d[None, :])
    Sm = 0.5 * (Sm + Sm.T)
    Q, _ = _meanzero_projector(pi)
    Sm0 = Q @ Sm @ Q
    w, Vv = np.linalg.eigh(Sm0)
    order = np.argsort(w)[::-1]
    lam = w[order]
    phi = Vv[:, order] / d[:, None]
    return lam, phi


def subspace_angles_deg(pi, A_cols, B_cols, d):
    """Principal angles (degrees) between the top-d columns of A and B in L2(pi)."""
    cross = np.array([[float(np.sum(pi * A_cols[:, i] * B_cols[:, j])) for j in range(d)]
                      for i in range(d)])
    s = np.linalg.svd(cross, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, 0.0, 1.0)))


def random_whitened(pi, d, k=40):
    """k random L2(pi)-orthonormal, mean-zero d-tuples (for variational checks)."""
    n = len(pi)
    sd = np.sqrt(pi)
    Q, _ = _meanzero_projector(pi)
    out = []
    for _ in range(k):
        R = Q @ (sd[:, None] * RNG.standard_normal((n, d)))  # mean-zero in M-space
        Qm, _ = np.linalg.qr(R)
        out.append(Qm[:, :d] / sd[:, None])                   # L2(pi)-orthonormal
    return out


def bilinear(pi, G, F, T):
    """sum_i <g_i, T f_i>_pi  — the two-encoder objective's correlation term."""
    return float(sum(np.sum(pi * G[:, i] * (T @ F[:, i])) for i in range(F.shape[1])))


# ---------------------------------------------------------------------------
# 3. The certificate.
# ---------------------------------------------------------------------------


def main():
    print("=" * 78)
    print("E10  The general (non-normal) Irreversible Blind Spot")
    print("=" * 78)
    print(
        "\nA non-normal transition splits the LEFT and RIGHT singular functions of T:\n"
        "the chart you encode z in is not the chart you read z' in. That split is the\n"
        "arrow of time made geometric. The single encoder, seeing only S, cannot\n"
        "represent it; the two-encoder/predictive optimum (the SVD) does.\n"
    )

    report = {"experiment": "E10_nonnormal_svd", "checks": {}, "cases": []}
    ck = report["checks"]
    DTOP = 3  # number of non-trivial modes compared

    # --- The non-normal conveyor --------------------------------------------
    n = 8
    a = np.array([0.20, 0.50, 0.80, 0.30, 0.60, 0.90, 0.40, 0.70])
    T, pi = conveyor(n, a)
    assert np.allclose(pi @ T, pi), "pi must be stationary"
    S, A = sym_anti(T, pi)
    nn = nonnormality(T, pi)
    print(f"Conveyor on Z/{n}: forward rates a={list(a)}, pi ~ 1/a (slow states pool mass).")
    print(f"  non-normality || T T* - T* T ||_pi = {nn:.4f}   (> 0: genuinely non-normal)")
    ck["conveyor_nonnormality"] = nn

    # (1) Theorem 1 is unconditional: <h,Th> = <h,Sh> for arbitrary h.
    basket = [RNG.standard_normal(n) for _ in range(200)]
    id_resid = max(abs(float(np.sum(pi * h * (T @ h))) - float(np.sum(pi * h * (S @ h)))) for h in basket)
    print(f"\n[1] Single-encoder still sees only S (Thm 1, no normality used):")
    print(f"    max | <h,Th> - <h,Sh> |_pi over 200 encoders = {id_resid:.2e}")
    ck["identity_resid"] = id_resid

    # (2) SVD: left != right charts.
    sig, U, V = svd_Lpi(T, pi)
    lam, phi = eig_sym_Lpi(S, pi)
    ang_uv = subspace_angles_deg(pi, U, V, DTOP)
    print(f"\n[2] Two-encoder optimum = SVD; input chart (right v) != output chart (left u):")
    print(f"    top-{DTOP} singular values sigma   = {np.round(sig[:DTOP], 4)}")
    print(f"    principal angles(span u, span v)  = {np.round(ang_uv, 2)} deg  (> 0: charts differ)")
    ck["leftright_angles_deg"] = ang_uv.tolist()
    ck["leftright_split_max_deg"] = float(ang_uv.max())

    # Variational confirmation: SVD achieves sum sigma; nothing whitened beats it.
    svd_value = bilinear(pi, U[:, :DTOP], V[:, :DTOP], T)        # = sum sigma_i
    rand_pairs = list(zip(random_whitened(pi, DTOP), random_whitened(pi, DTOP)))
    rand_max = max(bilinear(pi, G, F, T) for G, F in rand_pairs)
    print(f"    bilinear value at (u,v) = {svd_value:.4f} = sum sigma = {sig[:DTOP].sum():.4f};"
          f"  best of 40 random whitened pairs = {rand_max:.4f} (<=)")
    ck["svd_achieves_sumsigma"] = abs(svd_value - sig[:DTOP].sum())
    ck["svd_is_optimum"] = bool(rand_max <= sig[:DTOP].sum() + 1e-9)

    # (3) The irreversibility gap Delta = sum sigma - sum lambda(S) > 0.
    Delta = float(sig[:DTOP].sum() - lam[:DTOP].sum())
    ang_sv = subspace_angles_deg(pi, phi, V, DTOP)              # single-encoder chart vs right chart
    print(f"\n[3] Irreversibility gap (predictable correlation the single encoder misses):")
    print(f"    sum sigma(T) = {sig[:DTOP].sum():.4f},  sum lambda(S) = {lam[:DTOP].sum():.4f}")
    print(f"    Delta = sum sigma - sum lambda(S) = {Delta:.4f}  (> 0)")
    print(f"    single-encoder chart (eig S) vs input chart (right v): angles = {np.round(ang_sv,2)} deg")
    ck["irreversibility_gap"] = Delta
    ck["fan_hoffman_holds"] = bool(np.all(lam[:DTOP] <= sig[:DTOP] + 1e-12))

    # (4) Time reversal swaps the charts: right-sing(T*) == left-sing(T).
    Tstar = adjoint(T, pi)
    sig_r, U_r, V_r = svd_Lpi(Tstar, pi)
    swap_angles = subspace_angles_deg(pi, V_r, U, DTOP)         # right(T*) vs left(T): should be ~0
    print(f"\n[4] Time reversal T -> T* swaps charts (right-sing(T*) = left-sing(T)):")
    print(f"    principal angles(right v of T*, left u of T) = {np.round(swap_angles, 4)} deg (~0)")
    ck["reversal_swaps_charts_maxdeg"] = float(swap_angles.max())
    report["cases"].append(dict(name="conveyor", n=n, a=a.tolist(),
                                sigma=sig[:DTOP].tolist(), lambda_S=lam[:DTOP].tolist(),
                                Delta=Delta, leftright_angles=ang_uv.tolist()))

    # --- The normal drift ring: same gap, different mechanism ----------------
    print("\n[5] Normal contrast — E9's drift ring (circulant => NORMAL):")
    nr, q, b = 12, 0.5, 0.2
    Tn, pin = drift_ring(nr, q, b)
    Sn, An = sym_anti(Tn, pin)
    nn_ring = nonnormality(Tn, pin)
    sign, Un, Vn = svd_Lpi(Tn, pin)
    lamn, phin = eig_sym_Lpi(Sn, pin)
    ang_uv_ring = subspace_angles_deg(pin, Un, Vn, 2)           # slow pair
    Delta_ring = float(sign[:2].sum() - lamn[:2].sum())
    print(f"    non-normality = {nn_ring:.2e} (~0: normal).  left-vs-right chart angle "
          f"= {np.round(ang_uv_ring,3)} deg (~0: SAME chart)")
    print(f"    but the gap persists: Delta = sum sigma - sum lambda(S) = {Delta_ring:.4f} (> 0)")
    print(f"    => normal+current: one chart, a rotation inside it (E9). non-normal: two charts (E10).")
    ck["ring_nonnormality"] = nn_ring
    ck["ring_leftright_angle_maxdeg"] = float(ang_uv_ring.max())
    ck["ring_gap"] = Delta_ring

    # --- Reversible control: the gap and the split both vanish ---------------
    print("\n[6] Reversible control (detailed balance) — everything collapses:")
    Tr, pir = drift_ring(nr, 0.35, 0.35)                        # q=b: reversible
    Sr, Ar = sym_anti(Tr, pir)
    sigr, Ur, Vr = svd_Lpi(Tr, pir)
    lamr, phir = eig_sym_Lpi(Sr, pir)
    gap_r = float(sigr[:2].sum() - lamr[:2].sum())
    ang_r = subspace_angles_deg(pir, Ur, Vr, 2)
    print(f"    ||A|| = {np.linalg.norm(Ar):.2e},  Delta = {gap_r:.2e},  "
          f"left-vs-right angle = {np.round(ang_r,3)} deg  (all ~0)")
    print("    => reversible: single encoder = two encoder, no arrow, no chart split.")
    ck["reversible_gap"] = abs(gap_r)
    ck["reversible_split_maxdeg"] = float(ang_r.max())

    # --- Conveyor on CBS moduli ---------------------------------------------
    print("\n[7] Same statement on CBS moduli n in {6,8,12,18}:")
    rows = []
    for nn_ in (6, 8, 12, 18):
        aa = 0.3 + 0.6 * (np.arange(nn_) % 3) / 2.0 + 0.05 * (np.arange(nn_) % 2)  # varying rates
        aa = np.clip(aa, 0.1, 0.95)
        Tc, pic = conveyor(nn_, aa)
        Sc, Ac = sym_anti(Tc, pic)
        sg, Uc, Vc = svd_Lpi(Tc, pic)
        lm, _ = eig_sym_Lpi(Sc, pic)
        d = min(3, nn_ - 2)
        rows.append(dict(n=nn_, nonnormality=nonnormality(Tc, pic),
                         Delta=float(sg[:d].sum() - lm[:d].sum()),
                         leftright_maxdeg=float(subspace_angles_deg(pic, Uc, Vc, d).max())))
        print(f"    n={nn_:2d}: non-normal={rows[-1]['nonnormality']:.3f}, "
              f"Delta={rows[-1]['Delta']:.4f}, chart-split={rows[-1]['leftright_maxdeg']:.1f} deg")
    report["cases"].append(dict(name="cbs_moduli", rows=rows))

    # ---- self-check gate ----------------------------------------------------
    ok = (
        ck["conveyor_nonnormality"] > 1e-3
        and ck["identity_resid"] < TOL
        and ck["leftright_split_max_deg"] > 1.0
        and ck["svd_achieves_sumsigma"] < 1e-9
        and ck["svd_is_optimum"]
        and ck["irreversibility_gap"] > 1e-3
        and ck["fan_hoffman_holds"]
        and ck["reversal_swaps_charts_maxdeg"] < 1e-3
        and ck["ring_nonnormality"] < 1e-9
        and ck["ring_leftright_angle_maxdeg"] < 1e-3
        and ck["ring_gap"] > 1e-3
        and ck["reversible_gap"] < 1e-9
        and ck["reversible_split_maxdeg"] < 1e-3
        and all(r["Delta"] > 1e-4 for r in rows)
    )
    report["passed"] = bool(ok)

    # ---- optional figure ----------------------------------------------------
    fig_path = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

        # Left: input (right v) vs output (left u) charts on the conveyor differ.
        kk = np.arange(n)
        u1 = U[:, 0] / np.sqrt(np.sum(pi * U[:, 0] ** 2))
        v1 = V[:, 0] / np.sqrt(np.sum(pi * V[:, 0] ** 2))
        if np.sum(pi * u1 * v1) < 0:
            v1 = -v1
        axL.plot(kk, v1, "o-", color="#2980b9", label="input chart  $v_1$ (encode $z$)")
        axL.plot(kk, u1, "s-", color="#c0392b", label="output chart $u_1$ (read $z'$)")
        axL.set_title(f"Non-normal conveyor: charts differ ({ang_uv[0]:.0f}°)")
        axL.set_xlabel("state $k$ on the ring")
        axL.legend(fontsize=8)
        axL.grid(alpha=0.3)

        # Right: the irreversibility gap across regimes.
        labels = ["conveyor\n(non-normal)", "drift ring\n(normal+current)", "reversible\n(detailed bal.)"]
        sums_sig = [sig[:DTOP].sum(), sign[:2].sum(), sigr[:2].sum()]
        sums_lam = [lam[:DTOP].sum(), lamn[:2].sum(), lamr[:2].sum()]
        x = np.arange(3)
        axR.bar(x - 0.18, sums_sig, 0.36, color="#27ae60", label=r"two-encoder $\sum\sigma_i$")
        axR.bar(x + 0.18, sums_lam, 0.36, color="#7f8c8d", label=r"single-encoder $\sum\lambda_i(S)$")
        for i in range(3):
            axR.annotate(f"$\\Delta$={sums_sig[i]-sums_lam[i]:.2f}", (x[i], max(sums_sig[i], sums_lam[i]) + 0.03),
                         ha="center", fontsize=8)
        axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=8)
        axR.set_title("Irreversibility gap $\\Delta=\\sum\\sigma-\\sum\\lambda(S)$")
        axR.legend(fontsize=8, loc="lower left")
        axR.grid(alpha=0.3, axis="y")

        fig.suptitle("The general Irreversible Blind Spot: a current => a gap; non-normality => two charts",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = REPORTS / "e10_nonnormal_svd.png"
        fig.savefig(fig_path, dpi=130)
        print(f"\nfigure -> {fig_path}")
    except ImportError:
        print("\n(matplotlib not installed; skipping figure)")

    report["figure"] = str(fig_path) if fig_path else None
    rpt = REPORTS / "e10_nonnormal_svd.json"
    rpt.write_text(json.dumps(report, indent=2))
    print(f"report -> {rpt}")

    print("\n" + "=" * 78)
    print("E10: ALL CHECKS PASSED" if ok else "E10: SOME CHECKS FAILED")
    print("=" * 78)
    print("Takeaway: the single-encoder objective captures sum lambda_i(S); the two-")
    print("encoder/predictive optimum captures sum sigma_i(T). The gap Delta >= 0 is the")
    print("irreversible content, zero iff detailed balance. A current makes Delta>0; non-")
    print("normality additionally splits the input and output charts (left != right SVD).")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
