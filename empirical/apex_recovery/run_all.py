"""
run_all.py — reproduce every Apex-Matched Eigenfunction Recovery certificate.

    python empirical/apex_recovery/run_all.py        # (or: py ... for figures)

Runs E1-E4, E7, E9, E10, E12, E13, E14, E15 in order, each self-contained, writing
JSON (and PNG if matplotlib is available) into ./reports/. Runtime a few seconds;
no GPU, no network, deterministic. Exit code is nonzero if any experiment's
self-checks fail (E2's bound holds in every trial; E3 is exact; E7's chart agent
is exact and its Theorem-4 regret bound holds; E9's irreversible blind-spot
identity is exact and its predictor recovers the arrow; E10's non-normal SVD
splits the input/output charts and the gap is 0 iff reversible; E12's recovered
current dimension equals the transition graph's cycle rank beta_1 = E-V+1, refining
to the Betti number b_1 when 2-cells are filled; E13's strength Delta is estimable
from finite samples; E14's two axes (curvature nu_D, arrow Delta) are orthogonal;
E15's sparse-Hodge b_1 engine is exact vs ground truth and scales sub-second).
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "e1_eigenfunction_recovery.py",
    "e2_approximate_bound.py",
    "e3_hankel_reconstruction.py",
    "e4_cross_register_bridge.py",
    "e7_planning_certificate.py",
    "e9_irreversible_ring.py",
    "e10_nonnormal_svd.py",
    "e12_topological_blindspot.py",
    "e13_estimating_the_arrow.py",
    "e14_two_axes.py",
    "e15_efficient_betti.py",
]


def main() -> int:
    for s in SCRIPTS:
        print("\n" + "#" * 78)
        print(f"# {s}")
        print("#" * 78)
        runpy.run_path(str(HERE / s), run_name="__main__")

    # Post-hoc gate: assert the headline self-checks from the JSON outputs.
    ok = True
    e2 = json.loads((HERE / "reports" / "e2_approximate_bound.json").read_text())
    if e2["partB_failures"] != 0:
        print(f"FAIL: E2 had {e2['partB_failures']} bound violations")
        ok = False
    e3 = json.loads((HERE / "reports" / "e3_hankel_reconstruction.json").read_text())
    cbs = next(c for c in e3["cases"] if c["name"] == "cbs_cyclic_witness")
    expected = {"6": 18, "8": 42, "12": 108, "18": 252, "30": 774}
    if cbs["cross_packet_cubic_triples"] != expected:
        print("FAIL: E3 CBS cross-packet counts do not match CBS Section 4")
        ok = False
    e4 = json.loads((HERE / "reports" / "e4_cross_register_bridge.json").read_text())
    sk, sy = e4["families"]["skew"], e4["families"]["symmetric"]
    # the measured bridge: a stable, positive near-Gaussian square-law constant
    # in each register; the symmetric family has exactly-vanishing skewness.
    if not (sk["order"] == 3 and 0 < sk["leading_ratio"] < 1 and sk["near_gaussian_spread"] < 0.05):
        print("FAIL: E4 skew-family square law not stable/positive")
        ok = False
    if not (sy["order"] == 4 and 0 < sy["leading_ratio"] < 1 and sy["near_gaussian_spread"] < 0.05):
        print("FAIL: E4 symmetric-family square law not stable/positive")
        ok = False
    if max(abs(r["k3"]) for r in sy["rows"]) > 1e-10:
        print("FAIL: E4 symmetric family has nonzero skewness (should be exact 0)")
        ok = False

    e7 = json.loads((HERE / "reports" / "e7_planning_certificate.json").read_text())
    if abs(e7["proposition1"]["cube"]["regret"]) > 1e-6:
        print("FAIL: E7 chart agent not exact on the nonlinear warp (Prop 1)")
        ok = False
    lk = e7["linear_kicker"]
    if not (abs(lk[0]["regret_lin"]) < 1e-6 and lk[-1]["regret_lin"] > 1e-2):
        print("FAIL: E7 linear agent not exact-at-Gaussian / sub-optimal-off-Gaussian")
        ok = False
    if not (abs(e7["theorem4"][0]["regret"]) < 1e-6 and e7["C_fit"] < 1.0):
        print("FAIL: E7 Theorem-4 bound (regret<=C*L*T*eta, vanishing at eta=0) not met")
        ok = False

    e9 = json.loads((HERE / "reports" / "e9_irreversible_ring.json").read_text())
    c9 = e9["checks"]
    # The Irreversible Blind Spot: the single-encoder objective is a function of
    # the symmetric part S only (identity exact, loss reversal-invariant), while
    # the predictor recovers the antisymmetric arrow (and the blind spot switches
    # off exactly at detailed balance).
    if not (e9["passed"]
            and c9["identity_resid"] < 1e-12
            and c9["loss_reversal_gap"] < 1e-12
            and c9["S_reversal_diff"] < 1e-12 and c9["T_reversal_diff"] > 1e-6
            and c9["predictor_reversal_diff"] > 1e-6
            and c9["arrow_sign_matches_drift"]
            and c9["reversible_normA"] < 1e-15):
        print("FAIL: E9 Irreversible Blind Spot checks not met")
        ok = False

    e10 = json.loads((HERE / "reports" / "e10_nonnormal_svd.json").read_text())
    c10 = e10["checks"]
    # The general (non-normal) form: the single-encoder objective sees only S
    # (identity exact); a non-normal current splits the input/output SVD charts;
    # the irreversibility gap Delta = sum sigma - sum lambda(S) >= 0 is positive
    # with a current (normal ring or non-normal conveyor) and exactly 0 at
    # detailed balance; and time reversal swaps the left/right charts.
    if not (e10["passed"]
            and c10["identity_resid"] < 1e-12
            and c10["conveyor_nonnormality"] > 1e-3
            and c10["leftright_split_max_deg"] > 1.0
            and c10["svd_is_optimum"] and c10["fan_hoffman_holds"]
            and c10["irreversibility_gap"] > 1e-3 and c10["ring_gap"] > 1e-3
            and c10["reversible_gap"] < 1e-9
            and c10["reversal_swaps_charts_maxdeg"] < 1e-3):
        print("FAIL: E10 non-normal SVD / irreversibility-gap checks not met")
        ok = False

    e12 = json.loads((HERE / "reports" / "e12_topological_blindspot.json").read_text())
    # The topology: for every graph the recovered current dimension equals the cycle
    # rank beta_1 = E-V+1; the current is divergence-free (Kirchhoff); the single
    # encoder is invariant across the beta_1-family while A lives entirely in the
    # cycle space; and the gap vanishes exactly at detailed balance.
    if not (e12["passed"]
            and all(g["recovered_dim"] == g["beta1"] and g["beta1_agree"]
                    and g["single_encoder_invariance"] < 1e-10
                    and g["A_in_cycle_space_residual"] < 1e-10
                    and g["kirchhoff_residual"] < 1e-10
                    and g["irreversibility_gap"] > 1e-4
                    for g in e12["graphs"])
            and e12["reversible_control_gap"] < 1e-10
            # Theorem 5 (continuum): filling 2-cells gives b_1 = dim H^1 (torus 10->2, S^2 3->0).
            and e12["betti_refinement"]["torus T^2"]["b1_filled"] == 2
            and e12["betti_refinement"]["sphere S^2 (K4)"]["b1_filled"] == 0):
        print("FAIL: E12 topological blind-spot checks not met (dim(arrow) != cycle rank / Betti)")
        ok = False

    e13 = json.loads((HERE / "reports" / "e13_estimating_the_arrow.json").read_text())
    # The arrow's STRENGTH Delta is cheaply, robustly estimable from finite samples
    # (converges to the exact operator value; variance shrinks). Part B (topology is
    # not loop-cheap) is an honest report, not gated.
    if not (e13["passed"] and e13["partA"]["passed"]
            and e13["partA"]["rows"][-1]["rel_err"] < 0.03
            and e13["partA"]["rows"][-1]["std"] < e13["partA"]["rows"][0]["std"]):
        print("FAIL: E13 Delta-estimation convergence checks not met")
        ok = False

    e14 = json.loads((HERE / "reports" / "e14_two_axes.json").read_text())
    # The two axes are independent components of one operator: all four corners of
    # (Gaussian/non-Gaussian) x (reversible/irreversible) are realised, with zero
    # cross-talk (nu_D depends only on gamma, Delta only on rho).
    c14 = e14["corners"]
    if not (e14["passed"]
            and c14["non-Gaussian + reversible"]["nu"] > 0.05 and c14["non-Gaussian + reversible"]["delta"] < 1e-3
            and c14["Gaussian + irreversible"]["nu"] < 1e-3 and c14["Gaussian + irreversible"]["delta"] > 1e-2
            and c14["non-Gaussian + irreversible"]["nu"] > 0.05 and c14["non-Gaussian + irreversible"]["delta"] > 1e-2
            and e14["nu_crosstalk"] < 1e-9 and e14["delta_crosstalk"] < 1e-9):
        print("FAIL: E14 two-axes independence checks not met")
        ok = False

    e15 = json.loads((HERE / "reports" / "e15_efficient_betti.json").read_text())
    # The offline b_1 audit made efficient: the sparse-Hodge engine computes b_1 as
    # the nullity of L_1 exactly (matches ground truth on triangulated tori/sphere/
    # two-tori) and scales to thousands of simplices sub-second (Part A is gated;
    # Part B's point-cloud-complex frontier is an honest report).
    if not (e15["passed"]
            and all(a["b1"] == a["true"] for a in e15["partA"])
            and any(a["name"].startswith("tri-torus") and a["E"] >= 4000 and a["t"] < 5.0
                    for a in e15["partA"])):
        print("FAIL: E15 sparse-Hodge b_1 engine checks not met")
        ok = False

    print("\n" + "=" * 78)
    print("ALL CERTIFICATES PASSED" if ok else "SOME CHECKS FAILED")
    print("=" * 78)
    # E5 is the optional on-a-real-model certificate: it needs torch+transformers
    # and a cached Pythia checkpoint, is observational (not pass/fail), and is run
    # on its own. We point at it rather than gating the deterministic suite on it.
    try:
        import importlib.util as _u
        have = _u.find_spec("transformers") is not None and _u.find_spec("torch") is not None
    except Exception:
        have = False
    print("\nOptional real-model certificate (not gated):")
    print("  py empirical/apex_recovery/e5_real_model_bridge.py"
          + ("   [torch+transformers detected]" if have else "   [needs: py -m pip install transformers]"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
