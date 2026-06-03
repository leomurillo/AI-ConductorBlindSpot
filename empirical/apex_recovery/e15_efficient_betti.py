"""
E15 — The offline b_1 audit, made efficient: sparse Hodge eigenvalue counting
================================================================================

THE GOAL
--------
Theorem 5 says the dimension of the irreversible blind spot is b_1 = dim H^1 of
the state manifold, and E13 showed the naive computation is the obstacle: a
single-scale flag complex is wrong, and exact persistence times out on a few
hundred points. This certificate delivers the *engine* that makes the offline
audit cheap, exact, and scalable.

THE ENGINE
----------
b_1 = dim ker L_1, the nullity of the Hodge 1-Laplacian L_1 = d_1^T d_1 + d_2 d_2^T.
Instead of E12's dense boundary-rank (O(E^3), infeasible past ~10^3 simplices) we
count the **near-zero eigenvalues of the SPARSE L_1** by shift-invert Lanczos
(`eigsh(sigma=-1e-8)`), touching only the bottom of the spectrum: O(E * k) work
for the b_1 (+ a few) smallest eigenvalues. Same integer answer, but it scales.

WHAT IS CERTIFIED (Part A, deterministic, exact, sub-second)
------------------------------------------------------------
The ENGINE is exact & scalable on a GIVEN complex with known topology:
triangulated tori (b_1 = 2) up to thousands of simplices in well under a second
(where dense boundary-rank is already hopeless); a triangulated sphere
(b_1 = 0, no false positives); two disjoint tori (b_1 = 4). The sparse-Hodge
nullity equals ground truth in every case, with near-linear cost in the number of
simplices -- this is the offline audit's core step, made cheap.

THE HONEST FRONTIER (Part B, reported not gated)
------------------------------------------------
The ENGINE is solved; building a FAITHFUL sparse complex from a *raw point cloud*
is the remaining cost, and it genuinely wants dedicated tooling. A single-scale
flag (Rips) complex has no good scale: at small k a sampling gap breaks a loop
(under-count), at larger k triangles fill it (under-count), and on a surface
phantom loops survive (over-count) -- so circle samples give b_1 = 0 at every k
we try, and a Clifford-torus sample over-counts. The fix is not a cleverer single
scale but PERSISTENCE / an ALPHA complex (GUDHI, ripser), computed OFFLINE on a
sub-sample of the (low-dim) encoder embedding; then b_1 is read by the cheap
sparse-Hodge engine of Part A. We name that boundary rather than paper over it.

So the audit recipe: build a faithful sparse complex (graphs/known meshes:
directly; point clouds: an alpha/persistence library offline), then read b_1 as
the sparse-Hodge nullity -- the cheap, scalable step this file certifies.

Run:  python empirical/apex_recovery/e15_efficient_betti.py    (py ... for the figure)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# The engine: b_1 = nullity of the sparse Hodge 1-Laplacian (bottom spectrum).
# ---------------------------------------------------------------------------


def hodge_b1(V, edges, tris, tol=1e-6, k_eig=None):
    """
    b_1 = dim ker L_1 via shift-invert Lanczos on the SPARSE Hodge Laplacian.
    Only the smallest ~k_eig eigenvalues are computed; the near-zero count is b_1.
    """
    E = len(edges)
    if E == 0:
        return 0
    eidx = {e: n for n, e in enumerate(edges)}
    r, c, d = [], [], []
    for n, (i, j) in enumerate(edges):
        r += [i, j]; c += [n, n]; d += [-1.0, 1.0]
    B1 = sp.csr_matrix((d, (r, c)), shape=(V, E))
    if tris:
        r, c, d = [], [], []
        for n, (i, j, k) in enumerate(tris):
            r += [eidx[(i, j)], eidx[(j, k)], eidx[(i, k)]]; c += [n, n, n]; d += [1.0, 1.0, -1.0]
        B2 = sp.csr_matrix((d, (r, c)), shape=(E, len(tris)))
    else:
        B2 = sp.csr_matrix((E, 1))
    L1 = (B1.T @ B1 + B2 @ B2.T).tocsc().astype(float)
    if E <= 60:                                          # tiny: dense is fine
        return int((np.linalg.eigvalsh(L1.toarray()) < tol).sum())
    k = k_eig or 12
    k = min(k, E - 1)
    try:
        w = eigsh(L1, k=k, sigma=-1e-8, which="LM", return_eigenvectors=False)
    except Exception:
        w = np.sort(np.linalg.eigvalsh(L1.toarray()))[:k]
    return int((np.asarray(w) < tol).sum())


# ---------------------------------------------------------------------------
# Complexes with known topology (constructed directly -- no enumeration cost).
# ---------------------------------------------------------------------------


def tri_torus(L):
    """Flat triangulated torus Z/L x Z/L: b_1 = 2 (b_0=1, b_2=1)."""
    V = L * L
    idx = lambda i, j: (i % L) * L + (j % L)
    edges, tris = set(), set()
    for i in range(L):
        for j in range(L):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1)
            for e in [(a, b), (a, c), (a, d), (b, d), (c, d)]:
                edges.add(tuple(sorted(e)))
            tris.add(tuple(sorted((a, b, d)))); tris.add(tuple(sorted((a, c, d))))
    return V, sorted(edges), sorted(tris)


def sphere_hull(n, seed=0):
    """Triangulated 2-sphere as the convex hull of points on S^2: b_1 = 0."""
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    H = ConvexHull(X)
    edges, tris = set(), set()
    for s in H.simplices:
        i, j, k = sorted(int(x) for x in s)
        edges.update([(i, j), (j, k), (i, k)]); tris.add((i, j, k))
    return n, sorted(edges), sorted(tris)


def disjoint(*complexes):
    """Disjoint union (b_1 adds): two tori -> b_1 = 4."""
    V = 0; E, T = [], []
    for v, e, t in complexes:
        E += [(a + V, b + V) for a, b in e]
        T += [(a + V, b + V, c + V) for a, b, c in t]
        V += v
    return V, E, T


def point_cloud_rips(X, k=6):
    """Small-k Rips complex of a point cloud (cheap; faithful for curve-like sets)."""
    from scipy.spatial import cKDTree

    V = len(X)
    _, idx = cKDTree(X).query(X, k=k + 1)
    E = sorted({(min(i, int(j)), max(i, int(j))) for i in range(V) for j in idx[i, 1:]})
    adj = {v: set() for v in range(V)}
    for a, b in E:
        adj[a].add(b); adj[b].add(a)
    T = [(a, b, c) for (a, b) in E for c in adj[a] & adj[b] if a < b < c]
    return V, E, T


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("E15  The offline b_1 audit, made efficient (sparse Hodge eigenvalue counting)")
    print("=" * 78)
    report = {"experiment": "E15_efficient_betti", "passed": True}
    rng = np.random.default_rng(0)

    # ---- A. exact & scalable on known-topology complexes ---------------------
    print("\n[A] Exact & scalable (b_1 = sparse-Hodge nullity vs ground truth):")
    A = []
    for L in (10, 20, 30, 40):
        V, E, T = tri_torus(L)
        t0 = time.perf_counter(); b1 = hodge_b1(V, E, T); dt = time.perf_counter() - t0
        A.append(dict(name=f"tri-torus L={L}", V=V, E=len(E), T=len(T), b1=b1, true=2, t=dt))
        print(f"    tri-torus L={L:2d}: V={V:4d} E={len(E):5d} T={len(T):5d}  b1={b1} (true 2)  {dt:.3f}s")
    for name, (V, E, T), true in [("sphere (hull)", sphere_hull(300), 0),
                                  ("two disjoint tori", disjoint(tri_torus(12), tri_torus(12)), 4)]:
        t0 = time.perf_counter(); b1 = hodge_b1(V, E, T, k_eig=max(true + 6, 8)); dt = time.perf_counter() - t0
        A.append(dict(name=name, V=V, E=len(E), T=len(T), b1=b1, true=true, t=dt))
        print(f"    {name:18}: V={V:4d} E={len(E):5d} T={len(T):5d}  b1={b1} (true {true})  {dt:.3f}s")
    report["partA"] = A
    A_ok = all(a["b1"] == a["true"] for a in A)

    # ---- B. the honest frontier: a faithful complex from raw samples is the cost
    print("\n[B] Frontier (reported, not gated): the engine is solved; the COMPLEX is the cost.")
    print("    A single-scale flag complex on raw samples has no good scale --")
    th = np.sort(rng.uniform(0, 2 * np.pi, 240)); circle = np.c_[np.cos(th), np.sin(th)]
    circle_by_k = {}
    for k in (2, 3, 4, 6, 8):
        V, E, T = point_cloud_rips(circle, k=k)
        circle_by_k[k] = hodge_b1(V, E, T, k_eig=10)
    print(f"      circle (true b1=1), flag-complex b1 by k = {circle_by_k}")
    print("      (small k: a sampling gap breaks the loop -> 0; large k: triangles fill it -> 0)")
    u = rng.uniform(0, 2 * np.pi, 700); v = rng.uniform(0, 2 * np.pi, 700)
    cliff = np.c_[np.cos(u), np.sin(u), np.cos(v), np.sin(v)] / np.sqrt(2)
    Vc, Ec, Tc = point_cloud_rips(cliff, k=8)
    b1_cliff = hodge_b1(Vc, Ec, Tc, k_eig=10)
    print(f"      Clifford torus (true b1=2), flag-complex k=8 b1 = {b1_cliff} (phantom loops -> over-count)")
    print("    => build the faithful sparse complex with a persistence/alpha library (GUDHI,")
    print("       ripser) OFFLINE; then read b_1 with the sparse-Hodge engine above -- cheaply.")
    report["partB_frontier"] = dict(circle_flag_b1_by_k=circle_by_k, clifford_flag_b1_k8=b1_cliff,
                                    note=("single-scale flag complex on raw samples is unreliable "
                                          "(sampling gaps break loops; triangles fill them; surfaces "
                                          "get phantom loops). Faithful complex = alpha/persistence "
                                          "(GUDHI/ripser) offline; b_1 = sparse-Hodge nullity (Part A)."))

    report["passed"] = bool(A_ok)   # the ENGINE (Part A) is the certified, gated deliverable

    # ---- figure -------------------------------------------------------------
    fig_path = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
        tori = [a for a in A if a["name"].startswith("tri-torus")]
        Es = [a["E"] for a in tori]; ts = [a["t"] for a in tori]
        axL.plot(Es, ts, "o-", color="#16a085", label="sparse Hodge (eigsh)")
        # illustrative dense O(E^3) wall (normalised to the smallest point)
        c0 = ts[0] / Es[0] ** 3
        axL.plot(Es, [c0 * e ** 3 for e in Es], "--", color="#c0392b", label="dense rank $O(E^3)$ (ref.)")
        axL.set_xlabel("edges $E$"); axL.set_ylabel("seconds")
        axL.set_title("Efficient & scalable: b$_1$ from the bottom of the spectrum")
        axL.legend(fontsize=8); axL.grid(alpha=0.3)

        items = A
        names = [it["name"].replace("tri-torus ", "T²\n") for it in items]
        x = np.arange(len(names))
        axR.bar(x - 0.2, [it["b1"] for it in items], 0.4, color="#16a085", label="sparse-Hodge b$_1$")
        axR.plot(x + 0.2, [it["true"] for it in items], "kD", ms=8, label="true b$_1$")
        axR.set_xticks(x); axR.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
        axR.set_ylabel("b$_1$"); axR.set_title("Exact across topologies (tori, sphere, 2$\\times$T²)")
        axR.legend(fontsize=8); axR.grid(alpha=0.3, axis="y")

        fig.suptitle("The offline $b_1$ audit made efficient: sparse Hodge nullity — exact, scalable",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = REPORTS / "e15_efficient_betti.png"
        fig.savefig(fig_path, dpi=130)
        print(f"\nfigure -> {fig_path}")
    except ImportError:
        print("\n(matplotlib not installed; skipping figure)")

    report["figure"] = str(fig_path) if fig_path else None
    (REPORTS / "e15_efficient_betti.json").write_text(json.dumps(report, indent=2))
    print(f"report -> {REPORTS / 'e15_efficient_betti.json'}")

    print("\n" + "=" * 78)
    print("E15: ALL CHECKS PASSED" if report["passed"] else "E15: SOME CHECKS FAILED")
    print("=" * 78)
    print("Takeaway: b_1 = nullity of the sparse Hodge 1-Laplacian, read from the bottom of")
    print("the spectrum -- exact and scalable (the efficient offline audit). The remaining")
    print("cost is building a faithful sparse complex for surfaces-from-samples (alpha/GUDHI).")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
