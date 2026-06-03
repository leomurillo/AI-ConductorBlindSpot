"""
E12 — The topology of the irreversible blind spot: dim(arrow) = cycle rank
================================================================================

THE STATEMENT  (the deep generalisation of E9/E10)
--------------------------------------------------
E9's drift ring has ONE current (one beta). A general transition graph has a
whole SPACE of independent currents, and the single-encoder objective is blind to
ALL of them. That space is a topological invariant.

A stationary probability current J_ij = pi_i T_ij - pi_j T_ji is antisymmetric and
**divergence-free** (Kirchhoff: sum_j J_ij = 0, because pi is stationary). The
divergence-free edge flows on a connected graph G form its **cycle space**, of
dimension the first Betti number

        beta_1(G) = E - V + 1            (edges - vertices + 1).

Detailed balance is J = 0 (the origin of that space). Irreversibility is any
nonzero point in it. Since the single-encoder objective is a function of the
SYMMETRIC part S alone (Theorem 1), it cannot see ANY current; a predictor, which
learns the operator T itself, recovers the whole beta_1-dimensional current space.
Hence:

    *** dim(irreversible blind spot) = beta_1(G) = cycle rank of the graph. ***

    1 ring  -> beta_1 = 1  (E9's single arrow);  richer connectivity -> a
    higher-dimensional arrow no symmetric objective can represent.

This ties the blind spot to Hodge/Helmholtz decomposition and graph homology.

HOW IT IS CERTIFIED  (exact, numpy only)
----------------------------------------
For each of several graphs (a ring; a theta graph; the complete graph K4; a 3x3
torus) we:
  * compute beta_1 three independent ways (E-V+1; nullity of the incidence/boundary
    operator; number of non-tree edges) and check they agree;
  * build beta_1 fundamental CIRCULATIONS C_k (skew, zero row/col sums), so each
    T = T0 + sum_k c_k C_k is a valid stochastic chain with the SAME symmetric part
    T0 and the same (uniform) stationary law — a beta_1-parameter family of
    distinct irreversible worlds sharing one reversible backbone;
  * verify the single-encoder objective (top eigenvalues of S) is INVARIANT across
    the whole family (Theorem 1), while the worlds genuinely differ (||A|| > 0);
  * verify the predictor recovers the current coefficients c_k exactly, that the
    antisymmetric part A lives ENTIRELY in the beta_1-dim cycle space (zero
    residual), and that the recovered dimension equals beta_1;
  * report the irreversibility gap Delta = sum sigma(T) - sum lambda(S) >= 0 (E10),
    here growing with the total current and zero at detailed balance.

Run:  python empirical/apex_recovery/e12_topological_blindspot.py    (py ... for the figure)
Exact up to LAPACK eps; in run_all's gate.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)

TOL = 1e-10


# ---------------------------------------------------------------------------
# 1. Graph topology: incidence, spanning tree, fundamental cycles.
# ---------------------------------------------------------------------------


def incidence(V, edges):
    """Oriented boundary operator d_1 : R^E -> R^V  (tail -1, head +1)."""
    B = np.zeros((V, len(edges)))
    for e, (i, j) in enumerate(edges):
        B[i, e] = -1.0
        B[j, e] = +1.0
    return B


def spanning_tree(V, edges):
    """Union-find spanning tree; returns (tree_edge_indices, non_tree_edge_indices)."""
    par = list(range(V))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    tree, non = [], []
    for e, (i, j) in enumerate(edges):
        ri, rj = find(i), find(j)
        if ri != rj:
            par[ri] = rj
            tree.append(e)
        else:
            non.append(e)
    return tree, non


def cycle_vertices(V, edges, tree, ne):
    """Vertex sequence of the fundamental cycle of non-tree edge ne (closed s->t)."""
    s, t = edges[ne]
    adj = collections.defaultdict(list)
    for e in tree:
        i, j = edges[e]
        adj[i].append(j)
        adj[j].append(i)
    prev = {s: None}
    q = [s]
    while q:
        u = q.pop(0)
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
    seq = [t]
    u = t
    while prev[u] is not None:
        u = prev[u]
        seq.append(u)
    return seq  # t -> ... -> s along the tree; close with s -> t (the non-tree edge)


def circulation(V, seq, delta=1.0):
    """Skew circulation around the closed cycle `seq` (zero row & column sums)."""
    C = np.zeros((V, V))
    full = seq + [seq[0]]
    for a, b in zip(full[:-1], full[1:]):
        C[a, b] += delta
        C[b, a] -= delta
    return C


def cycle_rank(V, edges):
    """beta_1 three ways; returns the dict so the certificate can check agreement."""
    E = len(edges)
    tree, non = spanning_tree(V, edges)
    return dict(combinatorial=E - V + 1,
                nullity=E - int(np.linalg.matrix_rank(incidence(V, edges))),
                non_tree_edges=len(non)), tree, non


# ---------------------------------------------------------------------------
# 1b. The continuum refinement: fill the 2-cells, count Betti numbers (Thm 5).
# ---------------------------------------------------------------------------


def boundary_2(edges, faces):
    """Face -> edge boundary operator d_2, in the stored edge orientations."""
    eidx = {edges[e]: e for e in range(len(edges))}
    B2 = np.zeros((len(edges), len(faces)))
    for f, face in enumerate(faces):
        for (i, j) in face:
            if (i, j) in eidx:
                B2[eidx[(i, j)], f] += 1.0
            else:
                B2[eidx[(j, i)], f] -= 1.0
    return B2


def torus_complex(L=3):
    """L x L torus as a CW 2-complex: vertices, edges, square faces."""
    vid = lambda r, c: (r % L) * L + (c % L)
    edges, faces = [], []
    for r in range(L):
        for c in range(L):
            edges.append((vid(r, c), vid(r, c + 1)))
            edges.append((vid(r, c), vid(r + 1, c)))
    for r in range(L):
        for c in range(L):
            faces.append([(vid(r, c), vid(r, c + 1)), (vid(r, c + 1), vid(r + 1, c + 1)),
                          (vid(r + 1, c + 1), vid(r + 1, c)), (vid(r + 1, c), vid(r, c))])
    return L * L, edges, faces


def tetra_complex():
    """K4 filled with its 4 triangles = boundary of a tetrahedron = the 2-sphere."""
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    faces = [[(0, 1), (1, 2), (2, 0)], [(0, 1), (1, 3), (3, 0)],
             [(0, 2), (2, 3), (3, 0)], [(1, 2), (2, 3), (3, 1)]]
    return 4, edges, faces


def betti1_filled(V, edges, faces):
    """
    First Betti number b_1 = dim H^1 of the 2-complex, via the Hodge 1-Laplacian
    L_1 = d_1^T d_1 + d_2 d_2^T (down + up). dim ker L_1 = b_1 (Hodge theorem). The
    graph cycle rank beta_1 = E - V + 1 is the no-faces case; filling the 2-cells
    removes the contractible cycles, leaving only the harmonic (topological) ones.
    """
    B1 = incidence(V, edges)
    B2 = boundary_2(edges, faces)
    d1d2 = float(np.linalg.norm(B1 @ B2))                 # must be 0: a valid complex
    beta1_graph = len(edges) - int(np.linalg.matrix_rank(B1))
    L1 = B1.T @ B1 + B2 @ B2.T
    b1 = len(edges) - int(np.linalg.matrix_rank(L1))
    return beta1_graph, b1, d1d2


# ---------------------------------------------------------------------------
# 2. Reversible backbone + the beta_1-parameter family of irreversible worlds.
# ---------------------------------------------------------------------------


def reversible_base(V, edges, p=0.15):
    """Symmetric nearest-neighbour walk (uniform stationary law). T0 = T0^T."""
    T0 = np.zeros((V, V))
    for (i, j) in edges:
        T0[i, j] = p
        T0[j, i] = p
    for i in range(V):
        T0[i, i] = 1.0 - T0[i].sum()
    return T0


def graphs():
    """A ring, a theta graph, K4, and a 3x3 torus — increasing cycle rank."""
    g = {}
    n = 6
    g["ring C6"] = (n, [(i, (i + 1) % n) for i in range(n)])
    g["theta"] = (4, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 0)])
    g["K4"] = (4, [(i, j) for i in range(4) for j in range(i + 1, 4)])
    # 3x3 torus: V=9, each node degree 4, E=18, beta_1 = 10.
    L = 3
    tedges = []
    for r in range(L):
        for c in range(L):
            v = r * L + c
            tedges.append((v, r * L + (c + 1) % L))
            tedges.append((v, ((r + 1) % L) * L + c))
    g["torus 3x3"] = (L * L, tedges)
    return g


def sym_anti_uniform(T):
    """S, A in L2(uniform): adjoint = transpose."""
    return 0.5 * (T + T.T), 0.5 * (T - T.T)


def spectrum_sums(M, V):
    """Sum of the V-1 non-trivial eigenvalues of symmetric M (drop the constant mode)."""
    w = np.sort(np.linalg.eigvalsh(0.5 * (M + M.T)))[::-1]
    return float(w[1:].sum())


def singular_sums(T, V):
    """Sum of the V-1 non-trivial singular values of T in L2(uniform), mean-zero."""
    w = np.ones(V) / np.sqrt(V)
    Q = np.eye(V) - np.outer(w, w)
    s = np.linalg.svd(Q @ T @ Q, compute_uv=False)
    return float(np.sort(s)[::-1][:V - 1].sum())


# ---------------------------------------------------------------------------
# 3. The certificate.
# ---------------------------------------------------------------------------


def main():
    print("=" * 78)
    print("E12  The topology of the irreversible blind spot: dim(arrow) = cycle rank")
    print("=" * 78)
    print(
        "\nA stationary current is a divergence-free edge flow (Kirchhoff), so it lives\n"
        "in the graph's cycle space, of dimension beta_1 = E - V + 1. The single encoder\n"
        "(sees S only) is blind to ALL of it; the predictor recovers the whole beta_1-\n"
        "dimensional current space. dim(blind spot) = cycle rank — a topological invariant.\n"
    )

    report = {"experiment": "E12_topological_blindspot", "graphs": [], "passed": True}
    rng = np.random.default_rng(0)

    print(f"{'graph':12} {'V':>3} {'E':>3} {'beta_1':>7} {'recov.dim':>10} "
          f"{'single-enc inv':>15} {'A in cyc.space':>15} {'pred recovers J':>16} {'Delta':>8}")
    print("-" * 100)

    for name, (V, edges) in graphs().items():
        E = len(edges)
        b1, tree, non = cycle_rank(V, edges)
        beta1 = b1["combinatorial"]
        agree = (b1["combinatorial"] == b1["nullity"] == b1["non_tree_edges"])

        # beta_1 fundamental circulations.
        Cs = [circulation(V, cycle_vertices(V, edges, tree, ne)) for ne in non]
        skew_ok = all(np.allclose(C, -C.T) and np.allclose(C.sum(1), 0) for C in Cs)

        T0 = reversible_base(V, edges)
        # two distinct random members of the beta_1-parameter family (same S=T0, same pi).
        scale = 0.25 * 0.15
        c1 = rng.uniform(-scale, scale, beta1)
        c2 = rng.uniform(-scale, scale, beta1)
        T = T0 + sum(c * C for c, C in zip(c1, Cs))
        Talt = T0 + sum(c * C for c, C in zip(c2, Cs))
        valid = (np.allclose(T.sum(1), 1) and (T >= -1e-12).all()
                 and np.allclose(np.full(V, 1 / V) @ T, np.full(V, 1 / V)))
        S, A = sym_anti_uniform(T)
        s_unchanged = np.allclose(S, T0)

        # Kirchhoff: current is divergence-free.
        pi = np.full(V, 1 / V)
        J = pi[:, None] * T - pi[None, :] * T.T  # J_ij = pi_i T_ij - pi_j T_ji
        kirchhoff = float(np.abs(J.sum(1)).max())

        # single-encoder objective invariant across the family (Theorem 1).
        single_inv = abs(spectrum_sums(T, V) - spectrum_sums(Talt, V))

        # predictor recovers the current coefficients; A lives in the cycle space.
        Gmat = np.array([[np.sum(Ci * Cj) for Cj in Cs] for Ci in Cs])
        rhs = np.array([np.sum(A * Ci) for Ci in Cs])
        c_rec = np.linalg.solve(Gmat, rhs)
        recover_err = float(np.linalg.norm(c_rec - c1))
        A_residual = float(np.linalg.norm(A - sum(c * C for c, C in zip(c_rec, Cs))))
        recov_dim = int(np.linalg.matrix_rank(Gmat))

        # irreversibility gap (E10) over the full non-trivial spectrum.
        Delta = singular_sums(T, V) - spectrum_sums(T, V)

        ok = (agree and skew_ok and valid and s_unchanged
              and kirchhoff < TOL and single_inv < TOL
              and recover_err < TOL and A_residual < TOL
              and recov_dim == beta1 and Delta > 1e-4)
        report["passed"] &= ok
        report["graphs"].append(dict(
            name=name, V=V, E=E, beta1=beta1, beta1_three_ways=b1, beta1_agree=bool(agree),
            recovered_dim=recov_dim, single_encoder_invariance=single_inv,
            A_in_cycle_space_residual=A_residual, current_recover_err=recover_err,
            kirchhoff_residual=kirchhoff, S_unchanged=bool(s_unchanged),
            irreversibility_gap=Delta, passed=bool(ok)))

        print(f"{name:12} {V:>3} {E:>3} {beta1:>7} {recov_dim:>10} "
              f"{single_inv:>15.2e} {A_residual:>15.2e} {recover_err:>16.2e} {Delta:>8.4f}")

    # detailed-balance control: zero current => Delta = 0, nothing to recover.
    V, edges = graphs()["K4"]
    T0 = reversible_base(V, edges)
    Delta0 = singular_sums(T0, V) - spectrum_sums(T0, V)
    print(f"\nDetailed-balance control (K4, current = 0):  Delta = {Delta0:.2e}  (~0)")
    report["reversible_control_gap"] = float(abs(Delta0))

    # the continuum refinement (Theorem 5): fill the 2-cells, count Betti numbers.
    print("\n[8] From cycle rank to Betti number (Theorem 5, the continuum shadow):")
    print("    filling a graph's 2-cells turns the cycle rank beta_1 = E-V+1 into the")
    print("    manifold's first Betti number b_1 = dim H^1; only the topologically")
    print("    protected (harmonic) currents survive the continuum limit.")
    betti = {}
    for nm, (Vc, ec, fc) in [("torus T^2", torus_complex(3)), ("sphere S^2 (K4)", tetra_complex())]:
        bg, b1, chk = betti1_filled(Vc, ec, fc)
        betti[nm] = dict(beta1_graph=bg, b1_filled=b1, d1d2=chk)
        print(f"    {nm:16}: beta_1(graph) = {bg:2d}  ->  b_1(filled) = {b1}   "
              f"(d1.d2 = {chk:.0e}; the other {bg-b1} are contractible eddies)")
    report["betti_refinement"] = betti
    betti_ok = (betti["torus T^2"]["beta1_graph"] == 10 and betti["torus T^2"]["b1_filled"] == 2
                and betti["sphere S^2 (K4)"]["beta1_graph"] == 3 and betti["sphere S^2 (K4)"]["b1_filled"] == 0
                and all(b["d1d2"] < 1e-9 for b in betti.values()))
    report["passed"] = bool(report["passed"] and abs(Delta0) < TOL and betti_ok)

    # ---- figure -------------------------------------------------------------
    fig_path = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
        names = [g["name"] for g in report["graphs"]]
        b1s = [g["beta1"] for g in report["graphs"]]
        rec = [g["recovered_dim"] for g in report["graphs"]]
        x = np.arange(len(names))
        axL.bar(x - 0.18, b1s, 0.36, color="#34495e", label=r"cycle rank $\beta_1=E-V+1$")
        axL.bar(x + 0.18, rec, 0.36, color="#16a085", label="recovered current dim")
        axL.set_xticks(x); axL.set_xticklabels(names, fontsize=8)
        axL.set_ylabel("dimension of the irreversible blind spot")
        axL.set_title(r"dim(arrow) $=\beta_1$, measured")
        axL.legend(fontsize=8)
        axL.grid(alpha=0.3, axis="y")

        # right: K4 with its recovered current as directed arrows on a circle layout.
        V, edges = graphs()["K4"]
        T0 = reversible_base(V, edges); _, ntree = spanning_tree(V, edges)[0:2] if False else (None, None)
        b1, tree, non = cycle_rank(V, edges)
        Cs = [circulation(V, cycle_vertices(V, edges, tree, ne)) for ne in non]
        cc = np.array([0.03, -0.025, 0.02])
        T = T0 + sum(c * C for c, C in zip(cc, Cs))
        pi = np.full(V, 1 / V)
        J = pi[:, None] * T - pi[None, :] * T.T
        pos = np.array([[np.cos(2 * np.pi * k / V), np.sin(2 * np.pi * k / V)] for k in range(V)])
        for (i, j) in edges:
            axR.plot(pos[[i, j], 0], pos[[i, j], 1], color="#ddd", lw=1, zorder=1)
        for i in range(V):
            for j in range(V):
                if J[i, j] > 1e-9:  # net current i->j
                    axR.annotate("", xy=pos[j], xytext=pos[i],
                                 arrowprops=dict(arrowstyle="-|>", color="#c0392b",
                                                 lw=1.5 + 40 * J[i, j], shrinkA=12, shrinkB=12), zorder=2)
        axR.scatter(pos[:, 0], pos[:, 1], s=240, c="#2c3e50", zorder=3)
        axR.set_aspect("equal"); axR.axis("off")
        axR.set_title(r"$K_4$: a $\beta_1=3$ current the single encoder cannot see")

        fig.suptitle("The irreversible blind spot is topological: its dimension is the graph's cycle rank",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = REPORTS / "e12_topological_blindspot.png"
        fig.savefig(fig_path, dpi=130)
        print(f"\nfigure -> {fig_path}")
    except ImportError:
        print("\n(matplotlib not installed; skipping figure)")

    report["figure"] = str(fig_path) if fig_path else None
    (REPORTS / "e12_topological_blindspot.json").write_text(json.dumps(report, indent=2))
    print(f"report -> {REPORTS / 'e12_topological_blindspot.json'}")

    print("\n" + "=" * 78)
    print("E12: ALL CHECKS PASSED" if report["passed"] else "E12: SOME CHECKS FAILED")
    print("=" * 78)
    print("Takeaway: the stationary current is a divergence-free edge flow, so it lives in")
    print("the cycle space of dimension beta_1 = E - V + 1. The single encoder is blind to")
    print("all of it; the predictor recovers the whole space. dim(blind spot) = cycle rank.")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
