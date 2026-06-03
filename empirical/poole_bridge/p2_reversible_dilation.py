"""
P2 — Theorem B + Corollary B1: the exact reversible (unitary) dilation.
================================================================================

CLAIM UNDER TEST
----------------
Theorem B. On X = {0,1}^N, with an environment register of the same size,
        U_F |x>|y> = |x>|y XOR F(x)>
permutes the computational basis, hence is unitary, and  U_F|x>|0> = |x>|F(x)>.
If the input is a diagonal (classical) mixture and register 1 is traced out, the
second register carries exactly the classical pushforward F_# p — the same face
as Theorem A.

Corollary B1 (irreversibility is an environment statement). A SAME-SPACE unitary
V with V|x> = |F(x)> for all x exists if and only if F is injective. When F is
not injective the obstruction is exact and witnessed: a collision F(x)=F(x'),
x != x', would force a unitary to send two orthonormal vectors to one, so the
target Gram matrix  G[i,j] = <F(i)|F(j)>  has rank = |image F| < |X| and cannot
equal the identity. Non-injective Poole dynamics is therefore liftable exactly
only as an OPEN system (Thm A) or a reversible system WITH an environment /
history register (Thm B) — never as a same-space unitary on configurations alone.

Everything is a 0/1 permutation or an integer rank count: exact. The dilation's
traced marginal is built numerically (permute the joint law, sum out register 1)
and matched against the independent pushforward, so the two faces are checked to
agree rather than asserted equal.

Run:  python empirical/poole_bridge/p2_reversible_dilation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from poole_world import (
    TOL,
    PooleRule,
    injectivity,
    pushforward,
    xor_dilation_perm,
    perm_is_unitary,
    perm_unitarity_defect,
)

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


def traced_marginal_numeric(F: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Build U_F as a basis permutation, apply it to the JOINT diagonal law
    (mass p_x at (x, 0)), reshape (d,d) and sum out register 1 -> register-2 law.
    Independent of the analytic helper: a genuine 'apply the unitary, trace out'.
    """
    F = np.asarray(F, dtype=int)
    d = len(F)
    perm = xor_dilation_perm(F)            # flat index x*d+y -> x*d+(y^F(x))
    joint = np.zeros(d * d)
    for x in range(d):
        joint[x * d + 0] = p[x]            # (x, y=0)
    out = np.zeros(d * d)
    out[perm] = joint                      # P e_j = e_{perm[j]}  =>  out[perm[j]] = joint[j]
    reg2 = out.reshape(d, d).sum(axis=0)   # trace out register 1
    return reg2


def dilation_checks(name: str, F: np.ndarray) -> dict:
    F = np.asarray(F, dtype=int)
    d = len(F)
    perm = xor_dilation_perm(F)
    out = {"name": name, "d": int(d)}

    # (B1) U_F is unitary
    out["unitary"] = bool(perm_is_unitary(perm))
    out["unitarity_defect"] = perm_unitarity_defect(perm)

    # (B2) U_F|x>|0> = |x>|F(x)>  (exact integer index identity, all x)
    init_err = max(abs(int(perm[x * d + 0]) - (x * d + int(F[x]))) for x in range(d))
    out["init_state_index_err"] = int(init_err)

    # (B3) traced marginal == pushforward, on a generic rational law
    rng = np.random.default_rng(3)
    p = rng.integers(1, 9, size=d).astype(float)
    p = p / p.sum()
    out["marginal_vs_pushforward"] = float(
        np.max(np.abs(traced_marginal_numeric(F, p) - pushforward(F, p))))
    return out


def obstruction_checks(name: str, F: np.ndarray) -> dict:
    """Corollary B1: same-space unitary exists iff F injective; witness if not."""
    F = np.asarray(F, dtype=int)
    d = len(F)
    info = injectivity(F)
    out = {"name": name, "d": int(d), **{k: info[k] for k in ("injective", "image_size", "domain_size")}}
    out["collisions"] = info["collisions"][:5]

    # target Gram G[i,j] = <F(i)|F(j)> = [F(i)==F(j)]; rank = image size.
    G = (F[:, None] == F[None, :]).astype(float)
    out["target_gram_rank"] = int(np.linalg.matrix_rank(G))
    out["rank_deficiency"] = int(d - out["target_gram_rank"])

    if info["injective"]:
        # the same-space unitary EXISTS: it is the permutation matrix P|x> = |F(x)>.
        P = np.zeros((d, d))
        P[F, np.arange(d)] = 1.0
        out["same_space_unitary_exists"] = True
        out["same_space_unitary_defect"] = float(np.max(np.abs(P.T @ P - np.eye(d))))
        out["same_space_maps_F"] = int(max(abs(int(np.argmax(P[:, x])) - int(F[x])) for x in range(d)))
    else:
        # NO same-space unitary: a collision sends |x>,|x'> (orthonormal) to one vector.
        out["same_space_unitary_exists"] = False
        out["same_space_unitary_defect"] = None
        out["same_space_maps_F"] = None
    return out


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("P2  Theorem B + Cor B1 - exact reversible dilation; irreversibility = environment")
    print("=" * 78)
    print(
        "\nU_F|x>|y> = |x>|y XOR F(x)> is a basis permutation, hence unitary, and\n"
        "U_F|x>|0> = |x>|F(x)>. Tracing out register 1 gives the same classical face\n"
        "as Theorem A. A same-space unitary on configurations exists iff F is\n"
        "injective; a non-injective Poole rule needs the environment register.\n"
    )

    rule = PooleRule(shape=(2, 2, 1))                 # 16 states = 2^4
    F_poole = rule.F_array()
    F_perm = np.array([3, 0, 1, 2, 5, 6, 7, 4])       # injective (a permutation)
    F_noninj = np.array([1, 1, 3, 0, 2, 2, 7, 6])     # 2 collisions

    print("[Theorem B] reversible dilation:")
    dil = [
        dilation_checks(f"poole_rule_2x2x1", F_poole),
        dilation_checks("permutation", F_perm),
        dilation_checks("hand_noninjective", F_noninj),
    ]
    for w in dil:
        print(f"  {w['name']:18s} (d={w['d']:2d}):  unitary={w['unitary']}  "
              f"defect={w['unitarity_defect']:.0e}  "
              f"U|x,0>=|x,F(x)| err={w['init_state_index_err']}  "
              f"marginal==pushforward={w['marginal_vs_pushforward']:.0e}")

    print("\n[Corollary B1] same-space unitary <=> injective:")
    poole_info = injectivity(F_poole)
    obs = [
        obstruction_checks("poole_rule_2x2x1", F_poole),
        obstruction_checks("permutation", F_perm),
        obstruction_checks("hand_noninjective", F_noninj),
    ]
    for w in obs:
        kind = "INJECTIVE  -> same-space unitary EXISTS" if w["injective"] \
            else f"NON-INJ    -> NO same-space unitary (rank {w['target_gram_rank']} < {w['d']})"
        print(f"  {w['name']:18s} (d={w['d']:2d}):  {kind}")
        if not w["injective"]:
            cx = w["collisions"][0]
            print(f"        witness: F({cx[0]}) = F({cx[1]}) = {int(F_for(w['name'], F_poole, F_perm, F_noninj)[cx[0]])}"
                  f"  (two orthonormal states -> one image; rank deficiency {w['rank_deficiency']})")

    report = {
        "experiment": "P2_reversible_dilation",
        "dilation": dil,
        "obstruction": obs,
        "poole_is_injective": bool(poole_info["injective"]),
    }

    ok = (
        all(w["unitary"] and w["unitarity_defect"] < TOL and w["init_state_index_err"] == 0
            and w["marginal_vs_pushforward"] < TOL for w in dil)
        # the obstruction: injective F admits a same-space unitary, non-injective does not
        and all((w["injective"] == (w["rank_deficiency"] == 0)) for w in obs)
        and any((not w["injective"]) and w["rank_deficiency"] > 0 for w in obs)
        and any(w["injective"] and w["same_space_unitary_defect"] is not None
                and w["same_space_unitary_defect"] < TOL and w["same_space_maps_F"] == 0 for w in obs)
    )
    report["passed"] = bool(ok)
    (REPORTS / "p2_reversible_dilation.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {REPORTS / 'p2_reversible_dilation.json'}")

    print("\n" + "=" * 78)
    print("P2: ALL CHECKS PASSED" if ok else "P2: SOME CHECKS FAILED")
    print("=" * 78)
    print("Takeaway: every Poole rule reversibilises exactly with one environment")
    print("register (Thm B); the only obstruction to a same-space unitary is")
    print("non-injectivity (Cor B1), and it is witnessed by a collision. Irreversible")
    print("classical dynamics <-> open/with-history quantum dynamics, exactly.")
    if not ok:
        raise SystemExit(1)


def F_for(name, F_poole, F_perm, F_noninj):
    return {"poole_rule_2x2x1": F_poole, "permutation": F_perm, "hand_noninjective": F_noninj}[name]


if __name__ == "__main__":
    main()
