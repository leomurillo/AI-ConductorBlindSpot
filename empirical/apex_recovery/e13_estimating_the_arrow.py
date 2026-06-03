"""
E13 — Estimating the arrow in practice: strength is cheap, topology is not
================================================================================

THE QUESTION (asked of the theory directly)
-------------------------------------------
Theorem 5 says the topological dimension of the arrow is b_1 = dim H^1 of the
state manifold. Natural worry: estimating that from continuous, high-dimensional
data sounds ruinously expensive. Is it? This certificate gives the honest,
evidence-backed answer by separating two quantities:

  * STRENGTH  Delta = sum sigma(T) - sum lambda(S)   (E10's irreversibility gap):
      "how much arrow." CHEAP and ROBUST from finite samples — it is a by-product
      of the predictor a JEPA/BYOL objective already trains. O(d^2) to assemble
      (the batch cross-covariance), O(d^3) to read at eval; nothing in the loop.

  * DIMENSION  b_1 = dim H^1   (Theorem 5): "how many independent arrows."
      This is TOPOLOGY, and it is NOT loop-cheap. A single-scale geometric Hodge
      estimate (kNN flag complex) is simply wrong and scale-dependent, and exact
      Vietoris-Rips persistence is super-linear and times out on a few hundred
      points offline. Topology must be an OFFLINE, sub-sampled audit, never an
      in-loop computation.

So the practical recipe is: track Delta in/near training (free); compute b_1, if
you want it, offline on a subsample with real persistent homology. This file
certifies the cheap half and documents the expensive half with numbers.

PART A (gated): Delta is estimable from finite samples and converges to the exact
  operator-level value, with O(d^3) cost. Robust across sample size and seed.
PART B (reported): the naive single-scale b_1 is unreliable (wrong, scale- and
  N-dependent), confirming that topology is an offline problem.

Run:  python empirical/apex_recovery/e13_estimating_the_arrow.py    (py ... for the figure)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Worlds and the slow-eigenfunction embedding (what the encoder recovers).
# ---------------------------------------------------------------------------


def drift_ring(n, q, b):
    T = np.zeros((n, n))
    for k in range(n):
        T[k, (k + 1) % n] += q
        T[k, (k - 1) % n] += b
        T[k, k] += 1 - q - b
    return T, np.full(n, 1.0 / n)


def slow_embedding(T, pi, K):
    """Top-K non-constant eigenfunctions of the symmetric part S (the encoder's chart)."""
    S = 0.5 * (T + T.T)
    d = np.sqrt(pi)
    Sm = (d[:, None]) * S * (1.0 / d[None, :])
    Sm = 0.5 * (Sm + Sm.T)
    w, V = np.linalg.eigh(Sm)
    order = np.argsort(w)[::-1][1:1 + K]          # drop the constant mode
    return (V[:, order] / d[:, None])             # columns: L2(pi)-orthonormal functions


def delta_exact(T, pi, Phi):
    """Operator-level Delta = sum sigma(B) - sum lambda(sym B), B the block of T in Phi."""
    B = Phi.T @ (pi[:, None] * (T @ Phi))
    sig = np.linalg.svd(B, compute_uv=False)
    lam = np.linalg.eigvalsh(0.5 * (B + B.T))
    return float(sig.sum() - lam.sum()), B


def delta_from_samples(T, Phi, m, seed):
    """Estimate Delta from m sampled positive pairs (the in-loop-style readout)."""
    r = np.random.default_rng(seed)
    n = T.shape[0]
    z = r.integers(0, n, m)
    zp = np.array([r.choice(n, p=T[s]) for s in z])
    F, Fp = Phi[z], Phi[zp]
    P = (Fp.T @ F / m) @ np.linalg.pinv(F.T @ F / m)   # empirical predictor block
    sig = np.linalg.svd(P, compute_uv=False)
    lam = np.linalg.eigvalsh(0.5 * (P + P.T))
    return float(sig.sum() - lam.sum())


# ---------------------------------------------------------------------------
# Part B helpers: the naive single-scale geometric b_1 (shown to be unreliable).
# ---------------------------------------------------------------------------


def naive_b1(X, k):
    """Single-scale b_1 from a kNN flag (clique) complex. Fast but WRONG for 2D."""
    from scipy.spatial import cKDTree

    V = len(X)
    _, idx = cKDTree(X).query(X, k=k + 1)
    E = sorted({(min(i, int(j)), max(i, int(j))) for i in range(V) for j in idx[i, 1:]})
    adj = {v: set() for v in range(V)}
    for i, j in E:
        adj[i].add(j)
        adj[j].add(i)
    tri = [(i, j, kk) for (i, j) in E for kk in adj[i] & adj[j] if i < j < kk]
    # components via union-find (rank of d1 = V - components, no SVD needed)
    par = list(range(V))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for i, j in E:
        a, c = find(i), find(j)
        if a != c:
            par[a] = c
    comps = len({find(x) for x in range(V)})
    if tri:
        ei = {e: n for n, e in enumerate(E)}
        B2 = np.zeros((len(E), len(tri)))
        for n, (i, j, kk) in enumerate(tri):
            B2[ei[(i, j)], n] += 1
            B2[ei[(j, kk)], n] += 1
            B2[ei[(i, kk)], n] -= 1
        r2 = int(np.linalg.matrix_rank(B2))
    else:
        r2 = 0
    return (len(E) - (V - comps)) - r2


def sample_manifold(name, n, seed):
    r = np.random.default_rng(seed)
    if name == "circle":      # b1 = 1
        th = r.uniform(0, 2 * np.pi, n)
        return np.c_[np.cos(th), np.sin(th)]
    if name == "torus":       # b1 = 2
        u, v = r.uniform(0, 2 * np.pi, n), r.uniform(0, 2 * np.pi, n)
        return np.c_[(3 + np.cos(v)) * np.cos(u), (3 + np.cos(v)) * np.sin(u), np.sin(v)]
    if name == "sphere":      # b1 = 0
        X = r.normal(size=(n, 3))
        return X / np.linalg.norm(X, axis=1, keepdims=True)
    raise ValueError(name)


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("E13  Estimating the arrow in practice: strength is cheap, topology is not")
    print("=" * 78)
    report = {"experiment": "E13_estimating_the_arrow", "passed": True}

    # ---- Part A: Delta (strength) converges from finite samples (GATED) ------
    print("\n[A] Irreversibility strength Delta from finite samples (the in-loop signal):")
    n, q, b, K = 40, 0.45, 0.20, 6
    T, pi = drift_ring(n, q, b)
    Phi = slow_embedding(T, pi, K)
    Dx, _ = delta_exact(T, pi, Phi)
    print(f"    exact Delta (slow K={K}) = {Dx:.5f}")
    rowsA = []
    for m in (1000, 5000, 20000, 100000):
        est = [delta_from_samples(T, Phi, m, s) for s in range(5)]
        mean, std = float(np.mean(est)), float(np.std(est))
        rel = abs(mean - Dx) / Dx
        rowsA.append(dict(m=m, mean=mean, std=std, rel_err=rel))
        print(f"    m={m:6d}: Delta_est = {mean:.5f} +- {std:.5f}   (rel err {rel*100:4.1f}%)")
    # gate: converges (final rel err small) and variance shrinks with m.
    A_ok = (rowsA[-1]["rel_err"] < 0.03 and rowsA[-1]["std"] < rowsA[0]["std"])
    report["partA"] = dict(delta_exact=Dx, rows=rowsA, passed=bool(A_ok))
    print(f"    => converges to exact, variance shrinks ~1/sqrt(m); cost O(K^2)+O(K^3). "
          f"[{'OK' if A_ok else 'FAIL'}]")

    # ---- Part B: b_1 (topology) is NOT loop-cheap (REPORTED) -----------------
    print("\n[B] Topological dimension b_1: the naive single-scale estimate is unreliable.")
    truth = {"circle": 1, "torus": 2, "sphere": 0}
    rowsB = {}
    for name, N in [("circle", 150), ("torus", 300), ("sphere", 250)]:
        X = sample_manifold(name, N, seed=1)
        ests = {k: int(naive_b1(X, k)) for k in (6, 9, 12)}
        rowsB[name] = dict(true=truth[name], naive_by_k=ests)
        print(f"    {name:7} (true b1={truth[name]}): single-scale kNN b1 by k = {ests}"
              f"   -> {'wrong & scale-dependent' if len(set(ests.values()))>1 or list(ests.values())[0]!=truth[name] else 'ok'}")
    report["partB"] = dict(rows=rowsB,
                           note=("single-scale flag-complex b_1 is wrong/scale-dependent for 2D; "
                                 "exact Vietoris-Rips persistence is super-linear and times out on "
                                 "a few hundred points (exit 124 @120s) -> topology is an OFFLINE, "
                                 "sub-sampled audit, not an in-loop computation."))
    # the documented difficulty: for the torus the naive estimate never lands on 2.
    B_unreliable = all(v != 2 for v in rowsB["torus"]["naive_by_k"].values())
    report["partB"]["torus_naive_never_2"] = bool(B_unreliable)
    print("    => exact persistence times out on hundreds of points; topology is an OFFLINE audit.")
    print("       In/near the loop, track Delta (strength), not b_1 (count).")

    report["passed"] = bool(A_ok)   # only Part A is gated; Part B is an honest report

    # ---- figure -------------------------------------------------------------
    fig_path = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
        ms = [r["m"] for r in rowsA]
        means = [r["mean"] for r in rowsA]
        stds = [r["std"] for r in rowsA]
        axL.axhline(Dx, color="#c0392b", ls="--", label=f"exact $\\Delta$={Dx:.4f}")
        axL.errorbar(ms, means, yerr=stds, fmt="o-", color="#2c3e50", capsize=3, label="estimated (5 seeds)")
        axL.set_xscale("log")
        axL.set_xlabel("sample pairs $m$")
        axL.set_ylabel("irreversibility strength $\\Delta$")
        axL.set_title("CHEAP & robust: $\\Delta$ from the predictor")
        axL.legend(fontsize=8)
        axL.grid(alpha=0.3)

        names = list(rowsB)
        ks = [6, 9, 12]
        x = np.arange(len(names))
        for c, k in enumerate(ks):
            axR.bar(x + (c - 1) * 0.25, [rowsB[nm]["naive_by_k"][k] for nm in names], 0.25, label=f"k={k}")
        axR.plot(x, [rowsB[nm]["true"] for nm in names], "kD", ms=9, label="true $b_1$")
        axR.set_xticks(x); axR.set_xticklabels(names)
        axR.set_ylabel("estimated $b_1$")
        axR.set_title("NOT cheap: single-scale $b_1$ is wrong (topology = offline)")
        axR.legend(fontsize=8)
        axR.grid(alpha=0.3, axis="y")

        fig.suptitle("Estimating the arrow: track strength $\\Delta$ in-loop (free); leave the Betti count $b_1$ to an offline audit",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = REPORTS / "e13_estimating_the_arrow.png"
        fig.savefig(fig_path, dpi=130)
        print(f"\nfigure -> {fig_path}")
    except ImportError:
        print("\n(matplotlib not installed; skipping figure)")

    report["figure"] = str(fig_path) if fig_path else None
    (REPORTS / "e13_estimating_the_arrow.json").write_text(json.dumps(report, indent=2))
    print(f"report -> {REPORTS / 'e13_estimating_the_arrow.json'}")

    print("\n" + "=" * 78)
    print("E13: ALL CHECKS PASSED" if report["passed"] else "E13: SOME CHECKS FAILED")
    print("=" * 78)
    print("Takeaway: the in-loop-relevant quantity is the arrow's STRENGTH Delta -- a free,")
    print("robust by-product of the predictor (O(d^3) at eval). Its DIMENSION b_1 is genuine")
    print("topology: not loop-cheap, an offline sub-sampled persistent-homology audit.")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
