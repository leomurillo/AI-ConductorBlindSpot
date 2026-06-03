"""
run_all.py — reproduce every Poole–OTG bridge certificate (P1–P5).

    python empirical/poole_bridge/run_all.py

Runs P1–P5 in order, each self-contained, writing JSON into ./reports/. Runtime a
few seconds; no GPU, no network, no machine learning, deterministic. Exit code is
nonzero if any certificate's self-checks fail. Each certificate settles one theorem
of the Poole–OTG Bridge note in exact arithmetic:

  P1  Theorem A   decoherent CPTP channel lift   (Choi PSD; classical face = rule;
                                                   coherence annihilated)
  P2  Theorem B   reversible XOR dilation + Cor B1 (unitary; marginal = pushforward;
                                                   non-injective => no same-space unitary)
  P3  Theorem C   closure / leakage test          (id/basin/orbit close with induced G;
                                                   density/cell/window leak with witnesses)
  P4  Theorem D   current / blind-spot audit       (S/A/J at GENERAL pi; DB <=> A=0=J;
                                                   arrow in Poole data; cycle rank)
  P5  Theorem E   dimensionless firewall           (every output relabelling-invariant;
                                                   labelling-dependent control moves)

The representative Poole rule is a STAND-IN for Rooke's OTG rule (see poole_world.py
and the note's Obligation 1). The theorems and the tests are rule-agnostic; swapping
in the real rule re-runs the suite unchanged.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "p1_channel_lift.py",
    "p2_reversible_dilation.py",
    "p3_closure_leakage.py",
    "p4_current_audit.py",
    "p5_dimensionless_firewall.py",
    "p6_otg_real_rule.py",
]


def main() -> int:
    skipped = []
    for s in SCRIPTS:
        print("\n" + "#" * 78)
        print(f"# {s}")
        print("#" * 78)
        try:
            runpy.run_path(str(HERE / s), run_name="__main__")
        except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
            # P6 runs on Rooke Poole's actual engine, `otg_rule.py`, which is a
            # transcription of his All-Rights-Reserved code and is therefore NOT
            # redistributed in this public repo (pending the author's permission).
            # On a clone without it, P6 is skipped and P1-P5 constitute the gate.
            if s == "p6_otg_real_rule.py":
                print(f"SKIPPED {s}: requires Rooke's engine `otg_rule.py`, which is "
                      f"license-gated and not shipped here. See README. ({type(e).__name__})")
                skipped.append(s)
            else:
                raise

    # ---- post-hoc gate over the JSON outputs -------------------------------
    ok = True

    p1 = json.loads((HERE / "reports" / "p1_channel_lift.json").read_text())
    if not (p1["passed"] and all(
        w["cptp_defect"] < 1e-12 and w["choi_min_eig"] > -1e-12
        and w["coherence_annihilation"] < 1e-12
        and w["diag_face_vs_pushforward"] < 1e-12 for w in p1["witnesses"])):
        print("FAIL: P1 channel-lift checks not met")
        ok = False

    p2 = json.loads((HERE / "reports" / "p2_reversible_dilation.json").read_text())
    if not (p2["passed"]
            and all(w["unitary"] and w["init_state_index_err"] == 0
                    and w["marginal_vs_pushforward"] < 1e-12 for w in p2["dilation"])
            and all((w["injective"] == (w["rank_deficiency"] == 0)) for w in p2["obstruction"])
            and any((not w["injective"]) and w["rank_deficiency"] > 0 for w in p2["obstruction"])):
        print("FAIL: P2 dilation / obstruction checks not met")
        ok = False

    p3 = json.loads((HERE / "reports" / "p3_closure_leakage.json").read_text())
    for inst in p3["instances"]:
        sect = {r["sector"]: r for r in inst["sectors"]}
        if not (inst["translation_equivariant_violations"] == 0
                and sect["id"]["closed"]
                and sect["attractor_basin"]["closed"] and sect["attractor_basin"]["G_is_identity"]
                and sect["translation_orbit"]["closed"]
                and (not sect["density"]["closed"]) and sect["density"]["witness_valid"]
                and all(r["witness_valid"] for r in inst["sectors"] if not r["closed"])):
            print(f"FAIL: P3 closure/leakage checks not met on shape {inst['shape']}")
            ok = False

    p4 = json.loads((HERE / "reports" / "p4_current_audit.json").read_text())
    pa, pb, pc = p4["parts"]["A_drift_ring"], p4["parts"]["B_nonuniform"], p4["parts"]["C_poole_data"]

    def ident_ok(r):
        return (r["adjoint_identity_resid"] < 1e-12 and r["S_selfadjoint_resid"] < 1e-12
                and r["A_skew_resid"] < 1e-12 and r["J_antisymmetry_resid"] < 1e-12
                and r["J_divergence_resid"] < 1e-12)

    if not (p4["passed"]
            and ident_ok(pa["resid"]) and pa["resid"]["normA"] > 1e-3
            and pa["db_control"]["normA"] < 1e-12 and pa["db_control"]["detailed_balance"]
            and pa["reversal"]["S_reversal_diff"] < 1e-12 and pa["reversal"]["J_reversal_diff"] > 1e-3
            and ident_ok(pb["resid"]) and pb["resid"]["normA"] > 1e-3
            and pb["conductance_control"]["normA"] < 1e-9 and pb["conductance_control"]["detailed_balance"]
            and ident_ok(pc["resid"]) and pc["resid"]["normA"] > 1e-3
            and (not pc["resid"]["detailed_balance"]) and pc["topology"]["b1"] >= 1):
        print("FAIL: P4 current-audit checks not met")
        ok = False

    p5 = json.loads((HERE / "reports" / "p5_dimensionless_firewall.json").read_text())
    md = p5["max_drift_under_relabelling"]
    nc = p5["negative_control"]
    if not (p5["passed"] and md["channel"] < 1e-9 and md["closure"] == 0 and md["current"] < 1e-9
            and nc["F0_distinct_values"] > 1 and nc["weighted_sum_distinct_values"] > 1):
        print("FAIL: P5 firewall checks not met")
        ok = False

    # P6 — the same bridge on ROOKE'S ACTUAL B5-7/S5-9 rule (Obligation 1 discharged):
    # his unit tests reproduced, effective rule B{5,6}/S{5,6,7,8,9}, closure decided per
    # sector with valid leakage witnesses, and the succession flux shown to be a symmetric
    # scalar (S reversal-invariant) blind to the genuine current A (the arrow). This runs
    # only where the license-gated engine `otg_rule.py` is present (see README); on the
    # public repo it is skipped and P1-P5 are the gate.
    p6_json = HERE / "reports" / "p6_otg_real_rule.json"
    if "p6_otg_real_rule.py" not in skipped and p6_json.exists():
        p6 = json.loads(p6_json.read_text())
        f6, ca6 = p6["faithfulness"], p6["current_audit"]
        if not (p6["passed"]
                and f6["vacuum_stays_empty"] and f6["overpopulation_core_evaporates"]
                and f6["effective_birth"] == [5, 6] and f6["effective_survive"] == [5, 6, 7, 8, 9]
                and all(inst["translation_equivariant_violations"] == 0
                        and inst["batch_vs_stepint"] == 0
                        and next(r for r in inst["sectors"] if r["sector"] == "density")["witness_valid"]
                        for inst in p6["closure"])
                and ca6["adjoint_resid"] < 1e-12 and ca6["J_div_resid"] < 1e-12
                and ca6["normA"] > 1e-3 and not ca6["detailed_balance"]
                and ca6["S_reversal_diff"] < 1e-12 and ca6["J_reversal_diff"] > 1e-3):
            print("FAIL: P6 real-rule (OTG) checks not met")
            ok = False
    else:
        print("NOTE: P6 (real-rule) skipped — license-gated engine not present; "
              "P1-P5 constitute the public gate.")

    print("\n" + "=" * 78)
    print("ALL POOLE-BRIDGE CERTIFICATES PASSED" if ok else "SOME CHECKS FAILED")
    print("=" * 78)
    print("Five theorems of the Poole–OTG Bridge note, settled in exact arithmetic:")
    print("  A channel lift · B reversible dilation · C closure/leakage ·")
    print("  D current audit · E dimensionless firewall.")
    print("The representative rule is a stand-in; the proofs are rule-agnostic.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
