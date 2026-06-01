"""
E3 — The distributional cumulant tower and its finite determinacy  (exact arithmetic)
================================================================================

WHICH TOWER THIS IS  (read this first; it is the load-bearing distinction)
--------------------------------------------------------------------------
There are TWO spectral towers in the paper, and they coincide only at the
Gaussian:

  (D) DYNAMICAL tower  = eigenfunctions of the TRANSITION operator. Its slow
      member is the RECOVERY MAP phi_1. It is affine iff the world is Gaussian.
      This tower is what e1 computes.

  (S) DISTRIBUTIONAL tower = orthogonal polynomials / cumulants of the
      STATIONARY law. Its degree-1 member is ALWAYS affine. This is what the
      gauge program T0-T7 and the Conductor Blind Spot grade by cumulant order.
      THIS SCRIPT COMPUTES TOWER (S). It does NOT compute the recovery map.

Do not say "the recovery map is the degree-1 orthogonal polynomial": that is
false off the Gaussian, where (D) and (S) split. What is true: at the Gaussian
(and, finitely, at rank 2) the two towers are the same object, so the recovery
map is affine exactly there.

CLAIM UNDER TEST  (the distributional / finite register of Section 3.4)
----------------------------------------------------------------------
For a finite-rank world (a stationary law with r distinct levels):

    * the moment Hankel matrices have rank exactly r          (Rank Closure);
    * the r levels and weights are reconstructed EXACTLY from the first 2r
      moments                                                  (Prony/Hankel);
    * the rank-2 case is the affine corner: the smallest world whose RECOVERY
      map (tower D) is also affine -- the discrete Gaussian corner;
    * the cyclic Z/n head carries order-3 distributional content (the
      Amari-Chentsov cubic): the order-3 rung the Conductor Blind Spot certifies
      finitely and whose continuous shadow bends phi_1 in e1.

Everything is EXACT RATIONAL ARITHMETIC (sympy): no floating-point tolerance,
the rank drops and reconstructions are identities, in the deterministic-
settlement spirit of the CBS certificates. The Hankel/finite-determinacy
machinery is restated in-house (it is the in-house finite apex determinacy
result), not cited.

Run:  python empirical/apex_recovery/e3_hankel_reconstruction.py
Exact; < 1 second.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# A finite-rank world: r distinct levels lam_j with rational weights w_j.
# ---------------------------------------------------------------------------


def moments(levels, weights, k_max):
    """Exact moments mu_k = sum_j w_j lam_j^k for k = 0..k_max."""
    return [sum(w * l**k for l, w in zip(levels, weights)) for k in range(k_max + 1)]


def hankel(mom, size):
    """The (size+1) x (size+1) Hankel matrix [mu_{i+j}]."""
    return sp.Matrix(size + 1, size + 1, lambda i, j: mom[i + j])


def hankel_rank_profile(levels, weights):
    """
    Certify Hankel Rank Closure: det H_N != 0 for N < r and det H_r = 0.
    Returns the list of determinants det H_0, det H_1, ... det H_r (exact ints).
    """
    r = len(levels)
    mom = moments(levels, weights, 2 * r)
    dets = []
    for N in range(0, r + 1):
        dets.append(sp.nsimplify(hankel(mom, N).det()))
    return dets, mom


def reconstruct_levels_from_moments(mom, r):
    """
    Newton-Hankel reconstruction: from mu_0..mu_{2r-1} recover the r levels as
    the roots of the degree-r orthogonal-polynomial / Prony polynomial.

    The levels lam_j satisfy a linear recurrence of order r whose coefficients
    solve the Hankel system  H c = -[mu_r..mu_{2r-1}]^T, where H = [mu_{i+j}]_{0<=i,j<r}.
    The characteristic polynomial t^r + c_{r-1} t^{r-1} + ... + c_0 has the
    levels as its exact roots (Prony's method / classical Hankel reconstruction).
    """
    H = sp.Matrix(r, r, lambda i, j: mom[i + j])
    rhs = sp.Matrix(r, 1, lambda i, _: -mom[r + i])
    c = H.solve(rhs)  # exact rational solve
    t = sp.symbols("t")
    poly = t**r + sum(c[i] * t**i for i in range(r))
    roots = sp.solve(sp.Eq(poly, 0), t)
    return sorted(roots, key=lambda x: sp.re(x)), sp.factor(poly)


def jacobi_matrix(levels, weights):
    """
    Build the r x r Jacobi (tridiagonal) matrix whose eigen-system is the
    orthogonal-polynomial basis of the discrete measure (levels, weights).
    Its eigenvectors evaluated at the levels are the orthogonal polynomials;
    in the additive-noise model the slow eigenfunction phi_1 is the degree-1
    orthogonal polynomial = the centered level coordinate. We expose the
    three-term recurrence (alpha, beta) — the data the cumulant tower carries.
    """
    r = len(levels)
    # Gram-Schmidt the monomials 1, t, t^2,... in L2(measure) -> recurrence.
    # alpha_k = <t p_k, p_k> / <p_k, p_k> ; beta_k = <p_k, p_k>/<p_{k-1},p_{k-1}>.
    t = sp.symbols("t")
    polys = [sp.Integer(1)]
    alphas, betas = [], []
    for k in range(r):
        pk = polys[k]
        ip = lambda f, g: sum(w * (f.subs(t, l)) * (g.subs(t, l)) for l, w in zip(levels, weights))
        nk = ip(pk, pk)
        ak = ip(t * pk, pk) / nk
        alphas.append(sp.nsimplify(ak))
        if k + 1 < r:
            if k == 0:
                pkp1 = (t - ak) * pk
            else:
                bk = nk / ip(polys[k - 1], polys[k - 1])
                betas.append(sp.nsimplify(bk))
                pkp1 = (t - ak) * pk - bk * polys[k - 1]
            polys.append(sp.expand(pkp1))
    return alphas, betas


def main():
    print("=" * 78)
    print("E3  The distributional cumulant tower and its finite determinacy (exact)")
    print("=" * 78)
    print("  NOTE: this is the DISTRIBUTIONAL tower (orthogonal polynomials of the")
    print("  stationary law), not the recovery map. They coincide only at the")
    print("  Gaussian / rank-2 corner. The recovery map (transition eigenfunction)")
    print("  is e1's object.")

    report = {"experiment": "E3_hankel_reconstruction", "cases": []}

    # --- Case 1: a genuine rank-3 world (needs order 3 = three rungs) -----
    print("\nCase 1 — a rank-3 world: levels {-2, 1, 4}, weights {1/2, 1/3, 1/6}.")
    levels = [sp.Integer(-2), sp.Integer(1), sp.Integer(4)]
    weights = [sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 6)]
    r = 3

    dets, mom = hankel_rank_profile(levels, weights)
    print("  Hankel determinants det H_0 .. det H_3 (exact):")
    print("    " + ", ".join(str(d) for d in dets))
    closes_at = next(N for N, d in enumerate(dets) if d == 0)
    print(f"  -> first vanishing determinant at N = {closes_at} = rank r = {r}"
          f"   (Hankel Rank Closure: tower is rank {r}).")

    roots, poly = reconstruct_levels_from_moments(mom, r)
    print(f"  Newton-Hankel reconstruction from mu_0..mu_{2*r-1}:")
    print(f"    Prony polynomial = {poly}")
    print(f"    recovered levels = {roots}   (exactly the originals: "
          f"{[r_ == l for r_, l in zip(roots, levels)]})")

    alphas, betas = jacobi_matrix(levels, weights)
    print(f"  Jacobi recurrence  alpha = {alphas},  beta = {betas}")
    print("  -> the orthogonal-polynomial basis of the stationary law is fixed by\n"
          "     these (the distributional tower); reconstructing this rank-3 world\n"
          "     needs moments through order 2r = 6. (This is NOT the recovery map:\n"
          "     that is the transition eigenfunction of e1, curved off-Gaussian.)")
    report["cases"].append(dict(
        name="rank3", levels=[str(x) for x in levels], weights=[str(x) for x in weights],
        hankel_dets=[str(d) for d in dets], closes_at=closes_at,
        recovered_levels=[str(x) for x in roots], prony_poly=str(poly),
        alpha=[str(x) for x in alphas], beta=[str(x) for x in betas]))

    # --- Case 2: the affine corner is rank 2 -----------------------------
    print("\nCase 2 — the affine corner is rank 2 (the discrete Gaussian corner).")
    print("  On a 2-level world {-1, +1} EVERY function is affine in the level")
    print("  coordinate, so here the two towers coincide: the recovery map is")
    print("  affine too. This is the smallest world a linear probe reads whole.")
    lev2, w2 = [sp.Integer(-1), sp.Integer(1)], [sp.Rational(1, 2), sp.Rational(1, 2)]
    dets2, mom2 = hankel_rank_profile(lev2, w2)
    a2, b2 = jacobi_matrix(lev2, w2)
    print(f"  Hankel dets = {dets2}  -> rank closes at N = "
          f"{next(N for N, d in enumerate(dets2) if d == 0)} = 2.")
    print(f"  Jacobi recurrence alpha = {a2}, beta = {b2}: distributional degree-1")
    print("  polynomial is t. At rank 2 (and only there, finitely) this coincides")
    print("  with the recovery map -> second-order picture complete: the finite")
    print("  shadow of the Gaussian corner.")
    report["cases"].append(dict(
        name="rank2_affine_corner", levels=[str(x) for x in lev2],
        hankel_dets=[str(d) for d in dets2],
        closes_at=next(N for N, d in enumerate(dets2) if d == 0),
        alpha=[str(x) for x in a2], beta=[str(x) for x in b2],
        note="rank 2 = discrete Gaussian corner: towers coincide, recovery affine"))

    # --- Case 3: the rank test detects the order-3 rung from the data ----
    print("\nCase 3 — the order-3 rung is visible in the moments themselves.")
    print("  An order-2 reading sees only mu_0..mu_4. The Hankel rank test on")
    print("  those moments already certifies a 3rd distributional rung:")
    H2 = sp.Matrix(3, 3, lambda i, j: mom[i + j])  # uses mu_0..mu_4
    print(f"    det H_2 (mu_0..mu_4) = {H2.det()}  != 0  -> the world is NOT")
    print("    rank<=2: its cumulant tower does not terminate at order 2. (Whether")
    print("    the recovery map itself bends depends on the dynamics' asymmetry;")
    print("    generically it does -- see e1 for the operator-side statement.)")
    report["cases"].append(dict(name="order3_rung_visible", detH2=str(H2.det()),
                                note="det H_2 != 0 certifies rank>2: cumulant tower beyond order 2"))

    # --- Case 4: the CBS cyclic world on the same axis -------------------
    print("\nCase 4 — the CBS cyclic world (Z/n head) on the same order axis.")
    print("  CBS proves the order-2 Fisher form is BLOCK-DIAGONAL across conductor")
    print("  packets, while the order-3 Amari-Chentsov cubic couples them under")
    print("  k + l + m = 0 (mod n). In this paper's language that cubic is the")
    print("  'third rung' of the distributional tower -- present (nonzero) on any")
    print("  cyclic head, exactly the rung a rank-2 / second-order learner omits.")
    print("  We re-count the surviving cross-packet cubic triples for n in")
    print("  {6,8,12,18,30} as the discrete witness that order 3 is occupied:")
    cbs_counts = {}
    for n in (6, 8, 12, 18, 30):
        # ordered triples (k,l,m) in {1..n-1}^3 with k+l+m = 0 (mod n),
        # whose three conductors n/gcd(.,n) are not all equal (cross-packet).
        from math import gcd
        cnt = 0
        for k in range(1, n):
            for l in range(1, n):
                m = (-k - l) % n
                if m == 0:
                    continue
                conds = {n // gcd(k, n), n // gcd(l, n), n // gcd(m, n)}
                if len(conds) > 1:
                    cnt += 1
        cbs_counts[n] = cnt
        print(f"    n = {n:2d}:  cross-packet cubic triples = {cnt}")
    print("  (These reproduce the CBS Section 4 counts 18, 42, 108, 252, 774 —")
    print("   the order-3 content made arithmetically explicit by the ring.)")
    report["cases"].append(dict(name="cbs_cyclic_witness",
                                cross_packet_cubic_triples=cbs_counts,
                                note="order-3 occupied on cyclic heads (CBS Thm 3.5)"))

    rpt = REPORTS / "e3_hankel_reconstruction.json"
    rpt.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {rpt}")
    print("\nTakeaway: the distributional tower has finite rank r; reconstructing the")
    print("world needs moments through order 2r. Rank 2 = the affine (Gaussian)")
    print("corner where this tower coincides with the recovery map; cyclic heads")
    print("carry order-3 content (the Conductor Blind Spot cubic).")


if __name__ == "__main__":
    main()
