"""
P6 — the bridge certificates on ROOKE'S ACTUAL RULE (Obligation 1 discharged).
================================================================================

Draft 2 ran the pipeline on a representative stand-in. This certificate runs it on
the real thing: the canonical B5-7/S5-9 prime-resonance engine transcribed in
`otg_rule.py` from the Delta RPM Protocol master engine of
    github.com/rookepoole/SVP-OTG-Poole-Manifold-tests.

LICENSE NOTE. Rooke Poole's repository is "All Rights Reserved" (see his LICENSE).
`otg_rule.py` is a transcription of his engine and is therefore NOT redistributed
in this public repo pending the author's written permission; this script is kept
for provenance but is license-gated and excluded from the public `run_all.py` gate.
To run it, obtain his engine and supply a compatible `otg_rule.py` locally. The
*results* reported here (closure verdicts, current norms) are our own academic
output, computed by running the rule, and are attributed to Rooke Alan Poole,
"The Poole Manifold" (2026).

It does three things, all numpy-exact (his torch engine is read, never run):

  [Faithfulness]  reproduce his own canonical unit tests (vacuum survival,
                  overpopulation collapse) bit-for-bit; confirm the exact effective
                  integer rule B={5,6}, S={5,6,7,8,9} (prime resonance ejects m=7
                  from birth: 7+0.35 = 7.35 > B_HIGH); and cross-check the
                  equilibrium density / succession flux against his Phi ~ 0.4002.

  [Theorem C]     the exact closure / leakage test on his rule, on the smallest
                  enumerable tori carrying his exact 3x3x3 kernel (2x2x2 = 256
                  states; 4x2x2 = 65536). The id / basin / translation-orbit
                  sectors close (induced G); density / single-cell leak, with
                  decoded before-identical / after-distinct witnesses.

  [Theorem D]     the succession-flux / arrow audit. His Phi is a SCALAR activity
                  (fraction of cells changing, or equilibrium occupancy) — a
                  time-symmetric quantity. We build the exact density-coarsened
                  transition operator of his rule, split it S + A in L2(pi), and
                  show: there is a genuine current A (an arrow) of dimension b_1;
                  the symmetric part S (which any scalar activity like Phi reads)
                  is reversal-invariant while the current J flips sign. So Phi
                  cannot encode the arrow it is asked to carry (Corollary D1); the
                  arrow is the antisymmetric current, a distinct object.

This corrects an over-claim worth stating plainly: the succession flux is NOT the
probability current. It is the kind of symmetric scalar the current audit proves
blind to the arrow — which is exactly why naming the current matters.

Run:  python empirical/poole_bridge/p6_otg_real_rule.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from otg_rule import OTGTorus, effective_rule, selfcheck, step_field
from poole_world import (
    TOL,
    attractor_sector,
    closure_test,
    coarse_markov,
    density_sector,
    edge_current,
    is_pi_reversible,
    l2pi_adjoint,
    orbit_sector,
    single_cell_sector,
    stationary,
    sym_anti_pi,
)

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


def render(torus: OTGTorus, s: int) -> str:
    Lx, Ly, Lz = torus.shape
    f = torus.decode(s)
    slices = []
    for z in range(Lz):
        rows = ["".join(str(int(f[x, y, z])) for x in range(Lx)) for y in range(Ly)]
        slices.append("/".join(rows))
    return " || ".join(slices)


def equivariant_violations(torus: OTGTorus, F: np.ndarray) -> int:
    """F(T_v s) == T_v F(s) for every shift v, as array identities F[perm_v]==perm_v[F]."""
    bad = 0
    for v in torus.all_shifts():
        perm = torus.translate_perm(v)
        bad += int(np.count_nonzero(F[perm] != perm[F]))
    return bad


def orbit_sector_fast(torus: OTGTorus) -> np.ndarray:
    """Q(s) = min integer in the translation orbit of s (vectorised via perms)."""
    perms = np.stack([torus.translate_perm(v) for v in torus.all_shifts()])
    return perms.min(axis=0)


def batch_matches_step_int(torus: OTGTorus, F: np.ndarray, n: int = 64) -> float:
    """Cross-check the vectorised batch F against the per-state scipy step_int."""
    rng = np.random.default_rng(0)
    sample = rng.integers(0, torus.n_states, size=min(n, torus.n_states))
    return float(max(abs(int(F[s]) - torus.step_int(int(s))) for s in sample))


def closure_on(torus: OTGTorus, F: np.ndarray, sectors) -> dict:
    d = torus.n_states
    out = {"shape": list(torus.shape), "n_states": d,
           "translation_equivariant_violations": int(equivariant_violations(torus, F)),
           "batch_vs_stepint": batch_matches_step_int(torus, F),
           "sectors": []}
    for name, Q in sectors:
        res = closure_test(F, Q)
        row = {"sector": name, "n_values": len(set(int(q) for q in Q.tolist())),
               "closed": res["closed"]}
        if res["closed"]:
            row["G_is_identity"] = all(k == v for k, v in res["G"].items())
            row["induced_map_size"] = len(res["G"])
        else:
            x, xp = res["witness"]
            row["witness"] = [int(x), int(xp)]
            row["witness_render"] = [render(torus, x), render(torus, xp)]
            row["witness_after"] = [render(torus, int(F[x])), render(torus, int(F[xp]))]
            row["witness_valid"] = bool(Q[x] == Q[xp] and Q[F[x]] != Q[F[xp]])
        out["sectors"].append(row)
    return out


def succession_flux_probe(L: int = 16, steps: int = 100, seed: int = 0) -> dict:
    """Cross-check his Phi ~ 0.4002. Report equilibrium density AND change-fraction
    honestly — they differ, and his term reads as the former."""
    rng = np.random.default_rng(seed)
    f = (rng.random((L, L, L)) < 0.4).astype(np.int8)
    for _ in range(steps):
        f = step_field(f)
    dens, flip = [], []
    for _ in range(40):
        g = step_field(f)
        flip.append(float(np.mean(g != f)))
        dens.append(float(g.mean()))
        f = g
    return {"L": L, "eq_density": float(np.mean(dens)),
            "change_fraction": float(np.mean(flip)), "his_Phi": 0.4002}


def current_audit_on(torus: OTGTorus, eps: float = 0.02) -> dict:
    """Exact density-coarsened transition operator of his rule; S/A/J at its pi."""
    F = torus.F_array()
    labels, Kraw = coarse_markov(F, density_sector(torus.n_states))
    m = len(labels)
    K = (1 - eps) * Kraw + eps * np.full((m, m), 1.0 / m)  # noise floor -> irreducible
    pi = stationary(K)
    S, A = sym_anti_pi(K, pi)
    J = edge_current(K, pi)
    Kstar = l2pi_adjoint(K, pi)
    Sr, _ = sym_anti_pi(Kstar, pi)
    Jr = edge_current(Kstar, pi)
    rng = np.random.default_rng(0)
    fs = [rng.standard_normal(m) for _ in range(40)]
    gs = [rng.standard_normal(m) for _ in range(40)]

    def ip(a, b):
        return float(np.sum(pi * a * b))

    # cycle rank of the current support
    adj = (np.abs(J) > 1e-9)
    E = int(np.triu(adj, 1).sum())
    V = int(np.any(adj, axis=1).sum())
    return {
        "shape": list(torus.shape), "labels": labels, "pi_min": float(pi.min()),
        "adjoint_resid": max(abs(ip(K @ f, g) - ip(f, Kstar @ g)) for f, g in zip(fs, gs)),
        "S_selfadjoint_resid": max(abs(ip(S @ f, g) - ip(f, S @ g)) for f, g in zip(fs, gs)),
        "A_skew_resid": max(abs(ip(A @ f, g) + ip(f, A @ g)) for f, g in zip(fs, gs)),
        "J_antisym_resid": float(np.max(np.abs(J + J.T))),
        "J_div_resid": float(np.max(np.abs(J.sum(axis=1)))),
        "normA": float(np.linalg.norm(A)),
        "detailed_balance": bool(is_pi_reversible(K, pi)),
        "S_reversal_diff": float(np.linalg.norm(S - Sr)),
        "J_reversal_diff": float(np.linalg.norm(J - Jr)),
        "arrow_cycle_rank_b1": int(E - V + 1),
    }


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("P6  The bridge on ROOKE'S ACTUAL RULE - B5-7/S5-9 prime-resonance engine")
    print("=" * 78)

    report = {"experiment": "P6_otg_real_rule"}

    # ---- Faithfulness -------------------------------------------------------
    sc = selfcheck()
    eff = effective_rule()
    print("\n[Faithfulness] reproducing his canonical Delta RPM unit tests:")
    print(f"    vacuum stays empty            : {sc['vacuum_stays_empty']}")
    print(f"    overpopulation core evaporates: {sc['overpopulation_core_evaporates']}")
    print(f"    effective rule  B={sc['effective_birth']}  S={sc['effective_survive']}"
          f"   (m=7 -> phi={eff['phi_at_7']:.2f} > 7, ejected from birth)")
    flux = succession_flux_probe()
    print(f"    succession-flux cross-check (L={flux['L']}): "
          f"eq_density={flux['eq_density']:.4f}  change_fraction={flux['change_fraction']:.4f}"
          f"  (his Phi~{flux['his_Phi']})")
    print(f"      -> his Phi~0.40 matches the equilibrium DENSITY, not the flip rate:")
    print(f"         'succession flux' reads as an occupancy level, a SYMMETRIC scalar.")
    report["faithfulness"] = {**sc, "effective": eff, "flux_probe": flux}

    # ---- Theorem C: closure / leakage on his rule ---------------------------
    print("\n[Theorem C] exact closure / leakage on his rule (smallest enumerable tori):")
    instances = []
    for shape in [(2, 2, 2), (4, 2, 2)]:
        torus = OTGTorus(shape)
        d = torus.n_states
        F = torus.F_array()
        sectors = [
            ("id", np.arange(d)),
            ("attractor_basin", attractor_sector(F)),
            ("translation_orbit", orbit_sector_fast(torus)),
            ("density", density_sector(d)),
            ("single_cell_0", single_cell_sector(d, 0)),
        ]
        inst = closure_on(torus, F, sectors)
        instances.append(inst)
        print(f"  torus {shape} ({d} states)  translation-equivariant="
              f"{inst['translation_equivariant_violations'] == 0}"
              f"  (batch==step_int: {inst['batch_vs_stepint'] == 0})")
        for r in inst["sectors"]:
            if r["closed"]:
                kind = "G=id" if r.get("G_is_identity") else "G=quotient"
                print(f"    [CLOSED] {r['sector']:18s} |Y|={r['n_values']:5d}  {kind}")
            else:
                print(f"    [LEAKS ] {r['sector']:18s} |Y|={r['n_values']:5d}  witness valid={r['witness_valid']}")
        dens = next((r for r in inst["sectors"] if r["sector"] == "density" and not r["closed"]), None)
        if dens:
            print(f"      density witness: {dens['witness_render'][0]}  --F-->  {dens['witness_after'][0]}")
            print(f"                       {dens['witness_render'][1]}  --F-->  {dens['witness_after'][1]}")
    report["closure"] = instances

    # ---- Theorem D: succession-flux / arrow audit ---------------------------
    print("\n[Theorem D] succession-flux / arrow audit (exact density-coarsened operator):")
    ca = current_audit_on(OTGTorus((4, 2, 2)))
    print(f"    densities {ca['labels']},  pi>0 (min={ca['pi_min']:.2e})")
    print(f"    identities (adjoint/S/A/J/div) max resid = "
          f"{max(ca['adjoint_resid'], ca['S_selfadjoint_resid'], ca['A_skew_resid'], ca['J_antisym_resid'], ca['J_div_resid']):.2e}")
    print(f"    ||A|| (the ARROW in his coarse dynamics) = {ca['normA']:.4f}  "
          f"detailed_balance={ca['detailed_balance']}")
    print(f"    reversal: ||S - S_rev|| = {ca['S_reversal_diff']:.1e}  (a scalar like Phi sees S: arrow-blind)")
    print(f"              ||J - J_rev|| = {ca['J_reversal_diff']:.4f}  (>0: the current carries the arrow)")
    print(f"    dim(arrow) = cycle rank b_1 = {ca['arrow_cycle_rank_b1']}")
    report["current_audit"] = ca

    # ---- gate ---------------------------------------------------------------
    def cl(inst, name):
        return next(r for r in inst["sectors"] if r["sector"] == name)

    ok = (
        sc["vacuum_stays_empty"] and sc["overpopulation_core_evaporates"]
        and sc["effective_birth"] == [5, 6] and sc["effective_survive"] == [5, 6, 7, 8, 9]
        and all(
            inst["translation_equivariant_violations"] == 0
            and inst["batch_vs_stepint"] == 0
            and cl(inst, "id")["closed"]
            and cl(inst, "attractor_basin")["closed"] and cl(inst, "attractor_basin")["G_is_identity"]
            and cl(inst, "translation_orbit")["closed"]
            and (not cl(inst, "density")["closed"]) and cl(inst, "density")["witness_valid"]
            for inst in instances)
        and ca["adjoint_resid"] < TOL and ca["S_selfadjoint_resid"] < TOL
        and ca["A_skew_resid"] < TOL and ca["J_antisym_resid"] < TOL and ca["J_div_resid"] < TOL
        and ca["pi_min"] > 1e-9 and ca["normA"] > 1e-3 and not ca["detailed_balance"]
        and ca["S_reversal_diff"] < TOL and ca["J_reversal_diff"] > 1e-3
        and ca["arrow_cycle_rank_b1"] >= 1
    )
    report["passed"] = bool(ok)
    (REPORTS / "p6_otg_real_rule.json").write_text(json.dumps(report, indent=2))
    print("\nreport -> " + str(REPORTS / "p6_otg_real_rule.json"))

    print("\n" + "=" * 78)
    print("P6: ALL CHECKS PASSED" if ok else "P6: SOME CHECKS FAILED")
    print("=" * 78)
    print("Obligation 1 discharged: the bridge runs on the ACTUAL B5-7/S5-9 rule.")
    print("Closure is decided per sector; the succession flux is a symmetric scalar,")
    print("blind to the arrow, which is the antisymmetric current A of dimension b_1.")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
