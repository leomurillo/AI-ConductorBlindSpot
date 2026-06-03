"""
P4 — Theorem D: the current / blind-spot audit for Poole transition data.
================================================================================

CLAIM UNDER TEST
----------------
For a finite Markov transition matrix K on a state set Y with positive stationary
law pi, the L2(pi) adjoint is  K*(x,y) = pi_y K(y,x) / pi_x, and
        S = (K + K*)/2,   A = (K - K*)/2,
        J(x,y) = pi_x K(x,y) - pi_y K(y,x)   (the stationary edge current).
Then (1) S is pi-self-adjoint (the time-symmetric part), (2) A is pi-skew (the
arrow / current part), (3) detailed balance holds iff A = 0 iff J = 0, and
(4) J is antisymmetric with zero stationary divergence (Kirchhoff).

Corollary D1. A predictor trained only to recover next-state probabilities sees
S; the antisymmetric current A is in its kernel — the Conductor / Irreversible
Blind Spot, here on Poole transition data. A world and its time-reverse share S
(and the whole single-encoder landscape) while J flips sign.

This certificate is the GENERAL-pi companion to apex_recovery/E9 and E10 (which
fix pi uniform on the ring / conveyor). It runs three ways:

  Part A — the drift ring (uniform pi): the minimal current, cross-checked
           against the E9 closed form. Detailed-balance control switches A off.
  Part B — an explicit NON-uniform-pi irreducible chain: validates the L2(pi)
           adjoint identity <Kf,g>_pi = <f,K*g>_pi and the S/A/J split at
           general pi; a conductance (reversible) chain is the A = 0 control.
  Part C — actual Poole data: the density-coarsened representative rule with a
           small noise floor (the note's 'empirical or noisy Poole transition
           data'), irreducible with pi > 0. It carries a genuine current (an
           arrow in the coarse Poole dynamics); S is reversal-invariant while J
           flips; J is divergence-free; and the dimension of the current is the
           cycle rank b_1 = E - V + C of its support graph (ties to E12).

Exact to LAPACK epsilon. No machine learning.

Run:  python empirical/poole_bridge/p4_current_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from poole_world import (
    TOL,
    PooleRule,
    coarse_markov,
    density_sector,
    edge_current,
    is_pi_reversible,
    l2pi_adjoint,
    stationary,
    sym_anti_pi,
)

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


def ip(pi, a, b):
    """L2(pi) inner product <a,b>_pi = sum_i pi_i a_i b_i."""
    return float(np.sum(pi * a * b))


def theorem_d_residuals(K: np.ndarray, pi: np.ndarray, seed: int = 0) -> dict:
    """All Theorem-D identities for (K, pi), returned as residuals (want ~0)."""
    n = len(pi)
    Kadj = l2pi_adjoint(K, pi)
    S, A = sym_anti_pi(K, pi)
    J = edge_current(K, pi)
    rng = np.random.default_rng(seed)
    fs = [rng.standard_normal(n) for _ in range(40)]
    gs = [rng.standard_normal(n) for _ in range(40)]

    adjoint_resid = max(abs(ip(pi, K @ f, g) - ip(pi, f, Kadj @ g)) for f, g in zip(fs, gs))
    S_selfadj = max(abs(ip(pi, S @ f, g) - ip(pi, f, S @ g)) for f, g in zip(fs, gs))
    A_skew = max(abs(ip(pi, A @ f, g) + ip(pi, f, A @ g)) for f, g in zip(fs, gs))
    return {
        "stationary_resid": float(np.max(np.abs(pi @ K - pi))),  # pi K = pi
        "adjoint_identity_resid": adjoint_resid,
        "S_selfadjoint_resid": S_selfadj,
        "A_skew_resid": A_skew,
        "J_antisymmetry_resid": float(np.max(np.abs(J + J.T))),
        "J_divergence_resid": float(np.max(np.abs(J.sum(axis=1)))),
        "normA": float(np.linalg.norm(A)),
        "normJ": float(np.linalg.norm(J)),
        "detailed_balance": bool(is_pi_reversible(K, pi)),
    }


def reversal_residuals(K: np.ndarray, pi: np.ndarray) -> dict:
    """Time reversal K -> K* leaves S fixed and flips J (Corollary D1 / E9)."""
    Kstar = l2pi_adjoint(K, pi)
    S, _ = sym_anti_pi(K, pi)
    Sr, _ = sym_anti_pi(Kstar, pi)
    J = edge_current(K, pi)
    Jr = edge_current(Kstar, pi)
    return {
        "S_reversal_diff": float(np.linalg.norm(S - Sr)),   # must be 0
        "J_reversal_sum": float(np.linalg.norm(J + Jr)),    # J_rev = -J, so sum ~ 0
        "J_reversal_diff": float(np.linalg.norm(J - Jr)),   # must be > 0 if a current exists
    }


def conductance_chain(w: np.ndarray):
    """Reversible chain from symmetric conductances w: K[x,y]=w[x,y]/W[x],
    pi[x]=W[x]/sum W. Detailed balance holds exactly (the A=0 control)."""
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, np.maximum(np.diag(w), 0.0))
    W = w.sum(axis=1)
    K = w / W[:, None]
    pi = W / W.sum()
    return K, pi


def cycle_rank(J: np.ndarray, tol: float = 1e-9) -> dict:
    """b_1 = E - V + C of the current's support graph (undirected, |J|>tol)."""
    n = len(J)
    adj = [[] for _ in range(n)]
    E = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > tol:
                adj[i].append(j)
                adj[j].append(i)
                E += 1
    seen = [False] * n
    C = 0
    Vnz = 0
    for s in range(n):
        if any(abs(J[s, k]) > tol for k in range(n)):
            Vnz += 1
        if not seen[s] and adj[s]:
            C += 1
            stack = [s]
            seen[s] = True
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
    V = sum(1 for s in range(n) if adj[s])
    return {"E": E, "V": V, "C": C, "b1": E - V + C}


def drift_ring(n: int, q: float, b: float) -> np.ndarray:
    r = 1.0 - q - b
    K = np.zeros((n, n))
    for k in range(n):
        K[k, (k + 1) % n] += q
        K[k, (k - 1) % n] += b
        K[k, k] += r
    return K


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("P4  Theorem D - current / blind-spot audit for Poole transition data")
    print("=" * 78)
    print(
        "\nK = S + A in L2(pi):  S symmetric (predictable, what a symmetric objective\n"
        "sees), A antisymmetric (the current = the arrow, in its kernel). Detailed\n"
        "balance <=> A = 0 <=> J = 0; J is antisymmetric and divergence-free.\n"
    )
    report = {"experiment": "P4_current_audit", "parts": {}}

    # ---- Part A: the drift ring (uniform pi), cross-check vs E9 -------------
    print("-" * 78)
    print("[Part A] drift ring Z/12 (uniform pi), forward q=0.5 backward b=0.2:")
    n = 12
    K = drift_ring(n, 0.5, 0.2)
    pi = np.full(n, 1.0 / n)
    rA = theorem_d_residuals(K, pi)
    revA = reversal_residuals(K, pi)
    # E9 closed form for the slow eigenvalue arrow:
    beta = (0.5 - 0.2) * np.sin(2 * np.pi / n)
    print(f"    identities (adjoint/S/A/J/div) max resid = "
          f"{max(rA['adjoint_identity_resid'], rA['S_selfadjoint_resid'], rA['A_skew_resid'], rA['J_antisymmetry_resid'], rA['J_divergence_resid']):.2e}")
    print(f"    ||A||={rA['normA']:.4f}  ||J||={rA['normJ']:.4f}  arrow beta(E9)={beta:+.5f}  "
          f"detailed_balance={rA['detailed_balance']}")
    print(f"    reversal: ||S-S_rev||={revA['S_reversal_diff']:.1e} (=0), "
          f"||J-J_rev||={revA['J_reversal_diff']:.4f} (>0, arrow flips)")
    # detailed-balance control:
    K0 = drift_ring(n, 0.35, 0.35)
    r0 = theorem_d_residuals(K0, pi)
    print(f"    DB control q=b: ||A||={r0['normA']:.1e}, ||J||={r0['normJ']:.1e}, "
          f"detailed_balance={r0['detailed_balance']}")
    report["parts"]["A_drift_ring"] = {"resid": rA, "reversal": revA,
                                       "db_control": r0, "beta_E9": float(beta)}

    # ---- Part B: explicit NON-uniform-pi chain -----------------------------
    print("-" * 78)
    print("[Part B] explicit non-uniform-pi irreducible chain (general L2(pi) adjoint):")
    Kb = np.array([[0.5, 0.4, 0.1],
                   [0.2, 0.5, 0.3],
                   [0.3, 0.2, 0.5]])
    pib = stationary(Kb)
    rB = theorem_d_residuals(Kb, pib)
    revB = reversal_residuals(Kb, pib)
    print(f"    pi = {np.array2string(pib, precision=4)}  (non-uniform: "
          f"{not np.allclose(pib, pib[0])})")
    print(f"    adjoint identity <Kf,g>_pi=<f,K*g>_pi resid = {rB['adjoint_identity_resid']:.2e}")
    print(f"    S self-adjoint resid={rB['S_selfadjoint_resid']:.1e}, A skew resid={rB['A_skew_resid']:.1e}")
    print(f"    J antisym resid={rB['J_antisymmetry_resid']:.1e}, div J resid={rB['J_divergence_resid']:.1e}")
    print(f"    ||A||={rB['normA']:.4f} (current on), detailed_balance={rB['detailed_balance']}")
    # conductance (reversible) control at non-uniform pi:
    Kc, pic = conductance_chain(np.array([[0., 2., 1.], [2., 0., 3.], [1., 3., 0.]]))
    rC = theorem_d_residuals(Kc, pic)
    print(f"    conductance control: pi={np.array2string(pic, precision=3)}, "
          f"||A||={rC['normA']:.1e}, detailed_balance={rC['detailed_balance']}")
    report["parts"]["B_nonuniform"] = {"pi": pib.tolist(), "resid": rB, "reversal": revB,
                                       "conductance_control": rC}

    # ---- Part C: actual Poole data (noisy density-coarsened chain) ----------
    print("-" * 78)
    print("[Part C] Poole data: density-coarsened 3x3 rule + 2% noise floor:")
    rule = PooleRule(shape=(3, 3, 1))
    F = rule.F_array()
    labels, Kraw = coarse_markov(F, density_sector(rule.n_states))
    eps = 0.02
    m = len(labels)
    Kp = (1 - eps) * Kraw + eps * np.full((m, m), 1.0 / m)  # noisy => irreducible, pi>0
    pip = stationary(Kp)
    rP = theorem_d_residuals(Kp, pip)
    revP = reversal_residuals(Kp, pip)
    Jp = edge_current(Kp, pip)
    topo = cycle_rank(Jp)
    print(f"    sector values (densities) = {labels}")
    print(f"    pi>0 everywhere: {bool(np.all(pip > 1e-9))}  (min pi = {pip.min():.2e})")
    print(f"    identities max resid = "
          f"{max(rP['adjoint_identity_resid'], rP['S_selfadjoint_resid'], rP['A_skew_resid'], rP['J_antisymmetry_resid'], rP['J_divergence_resid']):.2e}")
    print(f"    ||A|| (arrow in coarse Poole dynamics) = {rP['normA']:.4f}  "
          f"detailed_balance={rP['detailed_balance']}")
    print(f"    reversal: ||S-S_rev||={revP['S_reversal_diff']:.1e} (S blind to arrow), "
          f"||J-J_rev||={revP['J_reversal_diff']:.4f} (>0)")
    print(f"    current support cycle rank b_1 = E-V+C = {topo['E']}-{topo['V']}+{topo['C']} "
          f"= {topo['b1']}  (dim of the arrow; cf. E12)")
    report["parts"]["C_poole_data"] = {"labels": labels, "pi": pip.tolist(),
                                       "resid": rP, "reversal": revP, "topology": topo}

    # ---- gate ---------------------------------------------------------------
    def identities_ok(r):
        return (r["adjoint_identity_resid"] < TOL and r["S_selfadjoint_resid"] < TOL
                and r["A_skew_resid"] < TOL and r["J_antisymmetry_resid"] < TOL
                and r["J_divergence_resid"] < TOL and r["stationary_resid"] < 1e-9)

    ok = (
        identities_ok(rA) and rA["normA"] > 1e-3 and not rA["detailed_balance"]
        and r0["normA"] < 1e-12 and r0["detailed_balance"]          # ring DB control
        and revA["S_reversal_diff"] < TOL and revA["J_reversal_diff"] > 1e-3
        and identities_ok(rB) and rB["normA"] > 1e-3 and not np.allclose(pib, pib[0])
        and rC["normA"] < 1e-9 and rC["detailed_balance"]           # conductance control
        and identities_ok(rP) and bool(np.all(pip > 1e-9))
        and rP["normA"] > 1e-3 and not rP["detailed_balance"]
        and revP["S_reversal_diff"] < TOL and revP["J_reversal_diff"] > 1e-3
        and topo["b1"] >= 1
    )
    report["passed"] = bool(ok)
    (REPORTS / "p4_current_audit.json").write_text(json.dumps(report, indent=2))
    print("\nreport -> " + str(REPORTS / "p4_current_audit.json"))

    print("\n" + "=" * 78)
    print("P4: ALL CHECKS PASSED" if ok else "P4: SOME CHECKS FAILED")
    print("=" * 78)
    print("Takeaway: the S/A/J split is exact at GENERAL pi, not just on the uniform")
    print("ring; detailed balance is exactly A=0=J; and Poole-coarsened data carries a")
    print("genuine current (arrow) of dimension b_1, invisible to a symmetric audit and")
    print("recovered by the current. Prediction and arrow are different axes (Cor D1).")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
