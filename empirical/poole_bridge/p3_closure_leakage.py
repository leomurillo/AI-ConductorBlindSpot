"""
P3 — Theorem C: the exact closure / leakage test on a Poole rule.
================================================================================

CLAIM UNDER TEST  (the centerpiece — the only one with real Poole-specific output)
--------------------------------------------------------------------------------
A proposed retained sector  Q : X -> Y  is CLOSED under the global update F iff
there is an induced map  G : Y -> Y  with  Q∘F = G∘Q, equivalently iff
        Q(x) = Q(x')  =>  Q(F(x)) = Q(F(x'))    for all x, x'.
When it fails, the failure is not a metaphor: it is witnessed by a pair (x, x')
that is OBSERVATIONALLY IDENTICAL before evolution (Q(x)=Q(x')) and
OBSERVATIONALLY DISTINCT after (Q(F(x)) != Q(F(x'))). That pair is the finite
Noether-defect / boundary-leakage certificate of Corollary C1.

We run the EXACT enumeration test on the representative Poole rule, over ALL
states, for a battery of candidate sectors, and sort them into the dictionary
of Corollary C1:

  CLOSED  (G is produced):
    * Q = id                       — trivially closed, G = F (sanity).
    * Q = attractor / basin id     — F-closed by construction, G = id.
    * Q = translation-orbit rep    — closed because the rule is translation-
                                     EQUIVARIANT (verified exactly here); G is a
                                     genuine symmetry-renormalised quotient map.
  LEAKS   (a witness pair is produced):
    * Q = density (live-cell count)         — coarse-graining leaks.
    * Q = a single fixed cell's value       — local observable leaks.
    * Q = a local window (cell + neighbours) — overlapping neighbourhoods leak.

This is exactly the program's projection-closure (T11) / renormalization (T13) /
curvature-as-leakage (T14) language made operational: a Poole sector is closed
only when the witness test has no counterexample, and leakage is a concrete pair.

Exact enumeration; no randomness. The closed/leaks verdicts and the witnesses
below belong to THIS representative rule (a stand-in for Rooke's OTG rule); the
TEST is rule-agnostic and re-runs unchanged on the real rule.

Run:  python empirical/poole_bridge/p3_closure_leakage.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from poole_world import (
    PooleRule,
    attractor_sector,
    closure_test,
    density_sector,
    orbit_sector,
    single_cell_sector,
    window_sector,
)

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


def render(rule: PooleRule, s: int) -> str:
    """Render a configuration as Lz slices of Lx-by-Ly grids of 0/1."""
    Lx, Ly, Lz = rule.shape
    index = {c: i for i, c in enumerate(rule.cells)}
    slices = []
    for z in range(Lz):
        rows = []
        for y in range(Ly):
            rows.append("".join(str((s >> index[(x, y, z)]) & 1) for x in range(Lx)))
        slices.append(" / ".join(rows))
    return "  ||  ".join(slices)


def translation_equivariant(rule: PooleRule) -> float:
    """max over states s and shifts v of [ F(T_v s) != T_v F(s) ] — must be 0."""
    d = rule.n_states
    bad = 0
    for s in range(d):
        Fs = rule.step_int(s)
        for v in rule.all_shifts():
            if rule.step_int(rule.translate_int(s, v)) != rule.translate_int(Fs, v):
                bad += 1
    return bad


def count_leak_witnesses(F: np.ndarray, Q: np.ndarray) -> int:
    """Number of Q-classes that leak (contain two states with different Q(F(.)))."""
    F = np.asarray(F, dtype=int)
    Q = np.asarray(Q)
    seen = {}
    leaking = set()
    for x in range(len(F)):
        q = int(Q[x])
        img = int(Q[F[x]])
        if q in seen:
            if seen[q] != img:
                leaking.add(q)
        else:
            seen[q] = img
    return len(leaking)


def run_instance(rule: PooleRule) -> dict:
    F = rule.F_array()
    d = rule.n_states
    inst = {"shape": list(rule.shape), "n_cells": rule.n_cells, "n_states": d,
            "birth": sorted(rule.birth), "survive": sorted(rule.survive),
            "resonance": rule.resonance, "sectors": []}

    teq = translation_equivariant(rule)
    inst["translation_equivariant_violations"] = int(teq)

    sectors = [
        ("id",               np.arange(d)),
        ("attractor_basin",  attractor_sector(F)),
        ("translation_orbit", orbit_sector(rule)),
        ("density",          density_sector(d)),
        ("single_cell_0",    single_cell_sector(d, 0)),
        ("window_cell_0",    window_sector(rule, 0)),
    ]

    for name, Q in sectors:
        res = closure_test(F, Q)
        n_values = len(set(int(q) for q in Q.tolist()))
        row = {"sector": name, "n_values": n_values, "closed": res["closed"]}
        if res["closed"]:
            row["induced_map_size"] = len(res["G"])
            # is G the identity (a conserved/invariant sector) or a genuine quotient?
            row["G_is_identity"] = all(k == v for k, v in res["G"].items())
        else:
            x, xp = res["witness"]
            row["witness"] = [int(x), int(xp)]
            row["witness_render"] = [render(rule, x), render(rule, xp)]
            row["witness_after_render"] = [render(rule, int(F[x])), render(rule, int(F[xp]))]
            row["witness_Q_before"] = [int(Q[x]), int(Q[xp])]
            row["witness_Q_after"] = [int(Q[F[x]]), int(Q[F[xp]])]
            # verify the witness really is observationally-identical-then-distinct
            row["witness_valid"] = bool(Q[x] == Q[xp] and Q[F[x]] != Q[F[xp]])
            row["n_leaking_classes"] = count_leak_witnesses(F, Q)
        inst["sectors"].append(row)
    return inst


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("P3  Theorem C - exact closure / leakage test on a Poole rule")
    print("=" * 78)
    print(
        "\nQ is closed under F iff  Q(x)=Q(x') => Q(F(x))=Q(F(x')).  Failure is\n"
        "witnessed by a pair identical before evolution and distinct after - the\n"
        "finite boundary-leakage certificate (Cor C1). Exact, over all states.\n"
    )

    instances = [
        run_instance(PooleRule(shape=(3, 3, 1))),   # 9 cells, 512 states (2D torus)
        run_instance(PooleRule(shape=(2, 2, 2))),   # 8 cells, 256 states (3D box)
    ]

    for inst in instances:
        print("-" * 78)
        print(f"Poole rule {tuple(inst['shape'])}  ({inst['n_cells']} cells, "
              f"{inst['n_states']} states)  B={inst['birth']} S={inst['survive']} "
              f"resonance={inst['resonance']}")
        print(f"  translation-equivariant: {inst['translation_equivariant_violations'] == 0} "
              f"(violations = {inst['translation_equivariant_violations']})")
        for row in inst["sectors"]:
            if row["closed"]:
                kind = "G = id (invariant sector)" if row.get("G_is_identity") else "G = induced quotient map"
                print(f"  [CLOSED] {row['sector']:18s} |Y|={row['n_values']:4d}  "
                      f"-> {kind}, |G|={row['induced_map_size']}")
            else:
                print(f"  [LEAKS ] {row['sector']:18s} |Y|={row['n_values']:4d}  "
                      f"-> {row['n_leaking_classes']} leaking class(es); witness valid={row['witness_valid']}")
        # show one fully-decoded witness (density) for the primary instance
        dens = next((r for r in inst["sectors"] if r["sector"] == "density" and not r["closed"]), None)
        if dens:
            x, xp = dens["witness"]
            print(f"    density leakage witness (cells live before = {dens['witness_Q_before'][0]}):")
            print(f"      x  = [{dens['witness_render'][0]}]  --F-->  [{dens['witness_after_render'][0]}]"
                  f"  ({dens['witness_Q_after'][0]} live)")
            print(f"      x' = [{dens['witness_render'][1]}]  --F-->  [{dens['witness_after_render'][1]}]"
                  f"  ({dens['witness_Q_after'][1]} live)")
            print(f"      same density before, different density after => the coarse sector leaks.")

    report = {"experiment": "P3_closure_leakage", "instances": instances}

    # ---- gate ---------------------------------------------------------------
    def sector(inst, name):
        return next(r for r in inst["sectors"] if r["sector"] == name)

    ok = True
    for inst in instances:
        ok &= inst["translation_equivariant_violations"] == 0
        # theorem-guaranteed closures:
        ok &= sector(inst, "id")["closed"]
        ok &= sector(inst, "attractor_basin")["closed"] and sector(inst, "attractor_basin")["G_is_identity"]
        ok &= sector(inst, "translation_orbit")["closed"]
        # leakage with a VALID witness on at least the density coarse-graining:
        dens = sector(inst, "density")
        ok &= (not dens["closed"]) and dens["witness_valid"]
        # every leaking row that exists must carry a valid witness
        for r in inst["sectors"]:
            if not r["closed"]:
                ok &= r["witness_valid"]
    report["passed"] = bool(ok)
    (REPORTS / "p3_closure_leakage.json").write_text(json.dumps(report, indent=2))
    print("\nreport -> " + str(REPORTS / "p3_closure_leakage.json"))

    print("\n" + "=" * 78)
    print("P3: ALL CHECKS PASSED" if ok else "P3: SOME CHECKS FAILED")
    print("=" * 78)
    print("Takeaway: closure is decided, not declared. The id / basin / symmetry-orbit")
    print("sectors close (an induced G exists); density / single-cell / window sectors")
    print("LEAK, each with an explicit before-identical / after-distinct witness pair.")
    print("That witness is the finite Noether-defect certificate (T11/T13/T14).")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
