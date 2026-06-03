"""
P1 — Theorem A: the exact decoherent quantum-channel lift of a Poole rule.
================================================================================

CLAIM UNDER TEST
----------------
For ANY deterministic finite update F : X -> X (in particular a Poole cellular
rule), the Kraus operators  A_x = |F(x)><x|  define a channel

        E_F(rho) = sum_x A_x rho A_x^†

with three exact properties:

  (A1)  E_F is completely positive and trace preserving (CPTP):
          - trace preservation  sum_x A_x^† A_x = I   (computed, not asserted);
          - complete positivity  Choi(E_F) ⪰ 0        (Choi's theorem).
  (A2)  On diagonal (classical) states it IS the Poole update: the diagonal face
        E_F(diag p) = diag(F_# p) is the classical pushforward, and on a point
        mass E_F(|x><x|) = |F(x)><F(x)| is the deterministic rule itself.
  (A3)  It annihilates every coherence: E_F(|u><v|) = 0 for u != v.

(A2)+(A3) together say: the channel's classical face is exactly the Poole rule,
and the price of the minimal lift is total decoherence — the warning of
Corollary A1. Any *nontrivial* OTG quantum claim must therefore specify a
coherent dynamics whose dephased face is the rule; this certificate fixes the
floor it must sit above.

All checks are exact to LAPACK epsilon. Three witnesses: the representative Poole
rule (2x2 torus, 16 states), a hand non-injective map, and a permutation. The
theorem is rule-agnostic; the channel exists for all three.

Run:  python empirical/poole_bridge/p1_channel_lift.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from poole_world import (  # noqa: E402  (same-dir import; see run note)
    TOL,
    PooleRule,
    basis_vec,
    channel_apply_kraus,
    choi,
    cptp_defect,
    kraus_ops,
    pushforward,
)

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


def diag_state(p: np.ndarray) -> np.ndarray:
    return np.diag(p)


def max_offdiag(M: np.ndarray) -> float:
    return float(np.max(np.abs(M - np.diag(np.diag(M)))))


def witness_checks(name: str, F: np.ndarray) -> dict:
    F = np.asarray(F, dtype=int)
    d = len(F)
    out = {"name": name, "d": int(d)}

    # (A1) trace preservation, computed as max | sum A^†A - I |
    out["cptp_defect"] = cptp_defect(F)

    # (A1) complete positivity: Choi PSD (min eigenvalue >= 0)
    Jmin = float(np.min(np.linalg.eigvalsh(choi(F))))
    out["choi_min_eig"] = Jmin

    # (A2) diagonal face = pushforward, on a generic rational law
    rng = np.random.default_rng(7)
    p = rng.integers(1, 9, size=d).astype(float)
    p = p / p.sum()
    face = channel_apply_kraus(F, diag_state(p))
    out["diag_face_offdiag"] = max_offdiag(face)  # output stays diagonal
    out["diag_face_vs_pushforward"] = float(
        np.max(np.abs(np.diag(face) - pushforward(F, p)))
    )

    # (A2) on point masses it is the deterministic rule: E(|x><x|) = |F(x)><F(x)|
    pt_err = 0.0
    for x in range(d):
        ex = channel_apply_kraus(F, np.outer(basis_vec(d, x), basis_vec(d, x)))
        pt_err = max(pt_err, float(np.max(np.abs(ex - np.outer(basis_vec(d, F[x]), basis_vec(d, F[x]))))))
    out["pointmass_is_rule"] = pt_err

    # (A3) every coherence annihilated: E(|u><v|) = 0 for u != v
    coh = 0.0
    for u in range(d):
        for v in range(d):
            if u == v:
                continue
            coh = max(coh, float(np.max(np.abs(
                channel_apply_kraus(F, np.outer(basis_vec(d, u), basis_vec(d, v)))))))
    out["coherence_annihilation"] = coh

    # a vivid instance: an equal superposition over two states decoheres to a
    # classical 50/50 mixture whose face is the pushforward of that mixture.
    if d >= 2:
        psi = (basis_vec(d, 0) + basis_vec(d, 1)) / np.sqrt(2)
        rho = np.outer(psi, psi)  # pure, with a real off-diagonal coherence
        erho = channel_apply_kraus(F, rho)
        out["superposition_offdiag_before"] = max_offdiag(rho)
        out["superposition_offdiag_after"] = max_offdiag(erho)
        out["superposition_face_vs_pushforward"] = float(
            np.max(np.abs(np.diag(erho) - pushforward(F, np.diag(rho)))))
    return out


def main():
    try:  # Windows consoles default to cp1252; keep prints crash-proof.
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 78)
    print("P1  Theorem A - the exact decoherent quantum-channel lift of a Poole rule")
    print("=" * 78)
    print(
        "\nKraus  A_x = |F(x)><x|,  E_F(rho) = sum_x A_x rho A_x^dag.  The channel is\n"
        "CPTP; its classical (diagonal) face is the Poole update; and it annihilates\n"
        "every coherence. The minimal lift exists for ANY rule - and it decoheres.\n"
    )

    rule = PooleRule(shape=(2, 2, 1))  # 4 cells, 16 states, power of two
    F_poole = rule.F_array()
    F_noninj = np.array([1, 1, 3, 0, 2, 2, 7, 6])          # 8 states, 2 collisions
    F_perm = np.array([3, 0, 1, 2, 5, 6, 7, 4])            # a permutation (injective)

    witnesses = [
        witness_checks(f"poole_rule_2x2x1_B{sorted(rule.birth)}_S{sorted(rule.survive)}", F_poole),
        witness_checks("hand_noninjective", F_noninj),
        witness_checks("permutation", F_perm),
    ]

    for w in witnesses:
        print(f"[{w['name']}]  (d = {w['d']})")
        print(f"    (A1) trace-preservation  max|sum A^dag A - I| = {w['cptp_defect']:.2e}")
        print(f"    (A1) complete positivity min eig Choi      = {w['choi_min_eig']:.2e}  (>= 0)")
        print(f"    (A2) diagonal face stays diagonal          = {w['diag_face_offdiag']:.2e}")
        print(f"    (A2) diagonal face == pushforward F_# p    = {w['diag_face_vs_pushforward']:.2e}")
        print(f"    (A2) point mass |x><x| -> |F(x)><F(x)|     = {w['pointmass_is_rule']:.2e}")
        print(f"    (A3) coherence annihilation  max|E(|u><v|)| = {w['coherence_annihilation']:.2e}")
        print(f"         superposition off-diag  {w['superposition_offdiag_before']:.3f} -> "
              f"{w['superposition_offdiag_after']:.2e}  (decohered)")
        print()

    report = {"experiment": "P1_decoherent_channel_lift", "witnesses": witnesses}

    ok = all(
        w["cptp_defect"] < TOL
        and w["choi_min_eig"] > -TOL
        and w["diag_face_offdiag"] < TOL
        and w["diag_face_vs_pushforward"] < TOL
        and w["pointmass_is_rule"] < TOL
        and w["coherence_annihilation"] < TOL
        and w["superposition_offdiag_after"] < TOL
        and w["superposition_face_vs_pushforward"] < TOL
        for w in witnesses
    )
    report["passed"] = bool(ok)
    (REPORTS / "p1_channel_lift.json").write_text(json.dumps(report, indent=2))
    print(f"report -> {REPORTS / 'p1_channel_lift.json'}")

    print("\n" + "=" * 78)
    print("P1: ALL CHECKS PASSED" if ok else "P1: SOME CHECKS FAILED")
    print("=" * 78)
    print("Takeaway: every Poole rule has an exact CPTP lift whose classical face is")
    print("the rule itself — and the minimal such lift kills all coherence. The T7")
    print("classical->quantum bridge is real; a NONTRIVIAL OTG claim must add a")
    print("coherent dynamics above this decoherent floor (Corollary A1).")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
