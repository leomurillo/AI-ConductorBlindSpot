"""
P5 — Theorem E: the dimensionless-physics firewall.
================================================================================

CLAIM UNDER TEST
----------------
The finite Poole bridge can output only dimensionless structural invariants until
an external Archimedean / empirical unit map is supplied. The proof is a symmetry
statement: the input data are finite sets, Boolean states, maps, distributions
and transition operators; any quantity built from them that does not depend on an
arbitrary relabelling of the state set is, by definition, a function of the
isomorphism class alone — a pure number. A dimensionful SI quantity would require
an extra map from the combinatorial object to a measured scale, which is not a
function of the finite data.

We certify the POSITIVE half operationally: relabelling the state set by an
arbitrary permutation sigma (so F -> sigma F sigma^-1, Q -> Q sigma^-1,
K -> P K P^T, pi -> P pi) leaves every invariant the suite reports unchanged:

    * Choi spectrum of the channel E_F        (Theorem A)
    * Shannon entropy of the pushforward       (a dimensionless info quantity)
    * closure verdict + leaking-class count    (Theorem C)
    * spectral gap, ||A||, ||J||, cycle rank   (Theorem D)

and we include a NEGATIVE CONTROL — a deliberately labelling-DEPENDENT quantity
(the integer index F(0), and an index-weighted sum) — that DOES change under
relabelling, proving the invariance test has teeth and is not vacuous.

The firewall is therefore not a disclaimer but a checked property: the suite's
outputs are exactly the relabelling-invariant pure numbers, and nothing with an
SI unit can be manufactured from them without the T15/P0/T30/T32 completion.

Exact: permutations and integer/real invariants, matched to LAPACK epsilon.

Run:  python empirical/poole_bridge/p5_dimensionless_firewall.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from poole_world import (
    TOL,
    PooleRule,
    choi,
    closure_test,
    coarse_markov,
    density_sector,
    edge_current,
    l2pi_adjoint,
    pushforward,
    stationary,
    sym_anti_pi,
)

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


def relabel_F(F: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """F' = sigma o F o sigma^-1  (relabel the state set by permutation sigma)."""
    F = np.asarray(F, dtype=int)
    inv = np.argsort(sigma)
    return sigma[F[inv]]


def relabel_Q(Q: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Q' = Q o sigma^-1  (the sector reads the relabelled state)."""
    Q = np.asarray(Q)
    inv = np.argsort(sigma)
    return Q[inv]


def perm_matrix(tau: np.ndarray) -> np.ndarray:
    n = len(tau)
    P = np.zeros((n, n))
    P[tau, np.arange(n)] = 1.0
    return P


def shannon(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def spectral_gap(K: np.ndarray) -> float:
    ev = np.sort(np.abs(np.linalg.eigvals(K)))[::-1]
    return float(ev[0] - ev[1]) if len(ev) > 1 else float(ev[0])


def channel_invariants(F: np.ndarray) -> dict:
    p = np.full(len(F), 1.0 / len(F))
    return {
        "choi_spectrum": np.sort(np.linalg.eigvalsh(choi(F))).round(10).tolist(),
        "pushforward_entropy": round(shannon(pushforward(F, p)), 10),
    }


def closure_invariants(F: np.ndarray, Q: np.ndarray) -> dict:
    res = closure_test(F, Q)
    n_leak = 0
    if not res["closed"]:
        seen, leaking = {}, set()
        for x in range(len(F)):
            q, img = int(Q[x]), int(Q[F[x]])
            if q in seen and seen[q] != img:
                leaking.add(q)
            else:
                seen.setdefault(q, img)
        n_leak = len(leaking)
    return {"closed": bool(res["closed"]), "n_leaking": int(n_leak)}


def current_invariants(K: np.ndarray, pi: np.ndarray) -> dict:
    _, A = sym_anti_pi(K, pi)
    J = edge_current(K, pi)
    return {
        "spectral_gap": round(spectral_gap(K), 10),
        "normA": round(float(np.linalg.norm(A)), 10),
        "normJ": round(float(np.linalg.norm(J)), 10),
        "K_abs_spectrum": np.sort(np.abs(np.linalg.eigvals(K))).round(10).tolist(),
    }


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("P5  Theorem E - the dimensionless-physics firewall")
    print("=" * 78)
    print(
        "\nEvery output of the finite bridge is invariant under relabelling the state\n"
        "set: it is a pure number, a function of the isomorphism class alone. No SI\n"
        "unit can be made without an external Archimedean anchor (T15/P0/T30/T32).\n"
    )

    rng = np.random.default_rng(20260602)
    rule = PooleRule(shape=(2, 2, 1))           # 16 states (channel: Choi 256x256)
    F = rule.F_array()
    d = len(F)
    Q = density_sector(d)

    # coarse chain for the current invariants
    rule2 = PooleRule(shape=(3, 3, 1))
    F2 = rule2.F_array()
    labels, Kraw = coarse_markov(F2, density_sector(rule2.n_states))
    eps, m = 0.02, len(labels)
    K = (1 - eps) * Kraw + eps * np.full((m, m), 1.0 / m)
    pi = stationary(K)

    base = {
        "channel": channel_invariants(F),
        "closure": closure_invariants(F, Q),
        "current": current_invariants(K, pi),
    }

    # ---- relabel many times; every invariant must be identical -------------
    print("[positive] invariants under 200 random relabellings of the state set:")
    max_drift = {"channel": 0.0, "closure": 0, "current": 0.0}
    for _ in range(200):
        sigma = rng.permutation(d)              # relabel the 16-state channel space
        Fr, Qr = relabel_F(F, sigma), relabel_Q(Q, sigma)
        chan_r = channel_invariants(Fr)
        clos_r = closure_invariants(Fr, Qr)
        max_drift["channel"] = max(
            max_drift["channel"],
            float(np.max(np.abs(np.array(chan_r["choi_spectrum"]) - np.array(base["channel"]["choi_spectrum"])))),
            abs(chan_r["pushforward_entropy"] - base["channel"]["pushforward_entropy"]),
        )
        max_drift["closure"] = max(
            max_drift["closure"],
            int(clos_r["closed"] != base["closure"]["closed"]) + abs(clos_r["n_leaking"] - base["closure"]["n_leaking"]),
        )

        tau = rng.permutation(m)                # relabel the coarse-chain state set
        P = perm_matrix(tau)
        Kr, pir = P @ K @ P.T, P @ pi
        cur_r = current_invariants(Kr, pir)
        max_drift["current"] = max(
            max_drift["current"],
            abs(cur_r["spectral_gap"] - base["current"]["spectral_gap"]),
            abs(cur_r["normA"] - base["current"]["normA"]),
            abs(cur_r["normJ"] - base["current"]["normJ"]),
            float(np.max(np.abs(np.array(cur_r["K_abs_spectrum"]) - np.array(base["current"]["K_abs_spectrum"])))),
        )

    print(f"    channel  (Choi spectrum + pushforward entropy) max drift = {max_drift['channel']:.2e}")
    print(f"    closure  (verdict + leaking-class count)        max drift = {max_drift['closure']}")
    print(f"    current  (gap, ||A||, ||J||, |spec K|)          max drift = {max_drift['current']:.2e}")

    # ---- negative control: a labelling-DEPENDENT number DOES move -----------
    print("\n[negative control] labelling-dependent quantities (must MOVE):")
    base_F0 = int(F[0])
    base_wsum = float(np.sum(np.arange(d) * pushforward(F, np.full(d, 1.0 / d))))
    moved_F0, moved_wsum = set(), set()
    for _ in range(200):
        sigma = rng.permutation(d)
        Fr = relabel_F(F, sigma)
        moved_F0.add(int(Fr[0]))
        moved_wsum.add(round(float(np.sum(np.arange(d) * pushforward(Fr, np.full(d, 1.0 / d)))), 6))
    print(f"    F(0) index took {len(moved_F0)} distinct values across relabellings (was {base_F0})")
    print(f"    index-weighted pushforward sum took {len(moved_wsum)} distinct values (was {base_wsum:.3f})")
    print("    => these are NOT invariants; the firewall test correctly rejects them.")

    report = {
        "experiment": "P5_dimensionless_firewall",
        "base_invariants": base,
        "max_drift_under_relabelling": {k: (float(v) if isinstance(v, float) else int(v))
                                        for k, v in max_drift.items()},
        "negative_control": {"F0_distinct_values": len(moved_F0),
                             "weighted_sum_distinct_values": len(moved_wsum)},
    }

    ok = (
        max_drift["channel"] < 1e-9
        and max_drift["closure"] == 0
        and max_drift["current"] < 1e-9
        # the negative control MUST move (else the test is vacuous):
        and len(moved_F0) > 1 and len(moved_wsum) > 1
    )
    report["passed"] = bool(ok)
    (REPORTS / "p5_dimensionless_firewall.json").write_text(json.dumps(report, indent=2))
    print("\nreport -> " + str(REPORTS / "p5_dimensionless_firewall.json"))

    print("\n" + "=" * 78)
    print("P5: ALL CHECKS PASSED" if ok else "P5: SOME CHECKS FAILED")
    print("=" * 78)
    print("Takeaway: every reported quantity is relabelling-invariant - a pure number,")
    print("a function of the isomorphism class. Labelling-dependent quantities are")
    print("correctly rejected. The bridge outputs dimensionless structure only; SI")
    print("units require the external completion, exactly as Theorem E states.")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
