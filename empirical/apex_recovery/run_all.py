"""
run_all.py — reproduce every Apex-Matched Eigenfunction Recovery certificate.

    python empirical/apex_recovery/run_all.py        # (or: py ... for figures)

Runs E1, E2, E3, E4, E7 in order, each self-contained, writing JSON (and PNG if
matplotlib is available) into ./reports/. Total runtime a few seconds; no GPU,
no network, deterministic. Exit code is nonzero if any experiment's built-in
self-checks fail (E2's bound must hold in every trial; E3 is exact; E7's chart
agent is exact and its Theorem-4 regret bound holds).
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
