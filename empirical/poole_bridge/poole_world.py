"""
poole_world.py — shared toolkit for the Poole–OTG bridge certificates (P1–P5).
================================================================================

This module is the finite, exact spine of the *Poole–OTG Bridge Theorem*. It has
two layers, kept rigorously apart, because they carry different epistemic weight.

  LAYER A  — rule-agnostic finite-dynamics primitives. These implement the bridge
             THEOREMS, which hold for ANY deterministic finite update F : X -> X:
               * kraus_ops / channel_apply_kraus / choi   (Thm A: decoherent lift)
               * xor_dilation_perm / dilation_marginal     (Thm B: reversible dilation)
               * injectivity                               (Cor B1: the obstruction)
               * closure_test                              (Thm C: closure / leakage)
               * l2pi_adjoint / sym_anti_pi / edge_current (Thm D: current audit)
             Everything here is exact (0/1 permutations, rational pushforwards),
             checked to LAPACK epsilon. The theorems are [P] — proved, certified.

  LAYER B  — a CONCRETE, REPRESENTATIVE Poole-style rule, so the whole pipeline
             runs end to end on a real cellular automaton instead of an abstract F.
             It is a deterministic outer-totalistic Boolean update on a finite
             torus (Z/Lx)x(Z/Ly)x(Z/Lz), with a "prime resonance" convention that
             shifts the threshold input when the live-neighbour count is prime.

             *** FIREWALL (read this). *** This rule is a STAND-IN, not Rooke's
             actual OTG rule. The note's Obligation 1 ("freeze the actual rule")
             is the team's to discharge. The rule lives behind a single dataclass
             `PooleRule`; swapping in the real neighbourhood / birth-survival /
             resonance convention is a one-object edit, after which every
             certificate re-runs unchanged. The theorems do not depend on the
             choice; the specific witness pairs in P3 do, and are labelled as
             belonging to THIS representative rule.

No randomness in Layer A's operators. No machine learning anywhere. CPU-millisecond.

Run nothing here directly; it is imported by p1_*.py … p5_*.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product

import numpy as np

TOL = 1e-12  # machine-precision identity headroom (matches apex_recovery)


# ===========================================================================
# LAYER A.1 — Theorem A: the exact decoherent quantum-channel lift.
#   For F : X -> X on |X| = d states, the Kraus operators are A_x = |F(x)><x|,
#   and  E_F(rho) = sum_x A_x rho A_x^†.  (real entries, so A_x^† = A_x^T.)
# ===========================================================================


def basis_vec(d: int, i: int) -> np.ndarray:
    e = np.zeros(d)
    e[i] = 1.0
    return e


def kraus_ops(F) -> list[np.ndarray]:
    """A_x = |F(x)><x|, one d x d matrix per state x. (Small d: O(d^3) storage.)"""
    F = np.asarray(F, dtype=int)
    d = len(F)
    ops = []
    for x in range(d):
        A = np.zeros((d, d))
        A[F[x], x] = 1.0
        ops.append(A)
    return ops


def cptp_defect(F) -> float:
    """max | (sum_x A_x^† A_x) - I |.  Theorem A trace-preservation, computed (not asserted)."""
    ops = kraus_ops(F)
    M = sum(A.T @ A for A in ops)
    return float(np.max(np.abs(M - np.eye(len(ops)))))


def channel_apply_kraus(F, rho: np.ndarray) -> np.ndarray:
    """E_F(rho) = sum_x A_x rho A_x^†, the genuine Kraus sum (reads all of rho)."""
    return sum(A @ rho @ A.T for A in kraus_ops(F))


def pushforward(F, p: np.ndarray) -> np.ndarray:
    """The classical pushforward (F_# p)(y) = sum_{x: F(x)=y} p_x."""
    F = np.asarray(F, dtype=int)
    out = np.zeros(len(F), dtype=float)
    for x in range(len(F)):
        out[F[x]] += p[x]
    return out


def choi(F) -> np.ndarray:
    """
    Choi matrix  J(E) = sum_{i,j} E(|i><j|) ⊗ |i><j|.  Complete positivity of E_F
    is exactly J ⪰ 0 (Choi's theorem). For E_F the off-diagonals die, so
    J = sum_i |F(i)><F(i)| ⊗ |i><i| is a sum of orthogonal rank-1 projectors —
    PSD with eigenvalues in {0,1}, an exact CP witness. (d^2 x d^2; small d only.)
    """
    F = np.asarray(F, dtype=int)
    d = len(F)
    J = np.zeros((d * d, d * d))
    for i in range(d):
        for j in range(d):
            Eij = channel_apply_kraus(F, np.outer(basis_vec(d, i), basis_vec(d, j)))
            J += np.kron(Eij, np.outer(basis_vec(d, i), basis_vec(d, j)))
    return J


# ===========================================================================
# LAYER A.2 — Theorem B: the exact reversible (unitary) dilation.
#   On X = {0,1}^N, index states by integers 0..2^N-1. With an environment
#   register of the same size, U_F |x>|y> = |x>|y XOR F(x)> permutes the
#   computational basis, hence is unitary; U_F|x>|0> = |x>|F(x)>.
# ===========================================================================


def xor_dilation_perm(F) -> np.ndarray:
    """
    U_F as a permutation of {0..d^2-1}, flat index = x*d + y -> x*d + (y XOR F(x)).
    Requires d = 2^N (XOR acts on the N-bit register). Returns the permutation array.
    """
    F = np.asarray(F, dtype=int)
    d = len(F)
    N = d.bit_length() - 1
    if (1 << N) != d:
        raise ValueError("XOR dilation needs d a power of two")
    perm = np.empty(d * d, dtype=int)
    for x in range(d):
        fx = int(F[x])
        for y in range(d):
            perm[x * d + y] = x * d + (y ^ fx)
    return perm


def perm_is_unitary(perm: np.ndarray) -> bool:
    """A basis permutation is unitary iff it is a genuine bijection of indices."""
    perm = np.asarray(perm, dtype=int)
    return np.array_equal(np.sort(perm), np.arange(len(perm)))


def perm_unitarity_defect(perm: np.ndarray) -> float:
    """|| P^T P - I ||_max for the permutation matrix P — 0 iff unitary."""
    n = len(perm)
    P = np.zeros((n, n))
    P[perm, np.arange(n)] = 1.0  # P e_j = e_{perm[j]}
    return float(np.max(np.abs(P.T @ P - np.eye(n))))


def dilation_marginal(F, p: np.ndarray) -> np.ndarray:
    """
    Apply U_F to (sum_x p_x |x><x|) ⊗ |0><0|, trace out register 1, return the
    register-2 distribution. Since |x,0> -> |x,F(x)>, this is the pushforward.
    """
    F = np.asarray(F, dtype=int)
    out = np.zeros(len(F), dtype=float)
    for x in range(len(F)):
        out[F[x]] += p[x]
    return out


def injectivity(F) -> dict:
    """
    Corollary B1's obstruction. A same-space unitary U with U|x>=|F(x)> for all x
    exists iff F is injective (a unitary preserves orthonormality). Returns the
    collisions that witness non-injectivity when F is not injective.
    """
    F = np.asarray(F, dtype=int)
    seen: dict[int, int] = {}
    collisions = []
    for x in range(len(F)):
        y = int(F[x])
        if y in seen:
            collisions.append((seen[y], x))  # F(x_a) = F(x_b), x_a != x_b
        else:
            seen[y] = x
    return dict(
        injective=(len(collisions) == 0),
        collisions=collisions,
        image_size=len(set(int(v) for v in F.tolist())),
        domain_size=len(F),
    )


# ===========================================================================
# LAYER A.3 — Theorem C: the exact closure / leakage test.
#   Q : X -> Y a retained sector. Q is closed under F iff there is G : Y -> Y with
#   Q∘F = G∘Q, iff [ Q(x)=Q(x') => Q(F(x))=Q(F(x')) ]. Failure is witnessed by a
#   pair (x,x') identical before evolution and distinct after.
# ===========================================================================


def closure_test(F, Q) -> dict:
    """
    Returns {closed, G, witness}. G is the induced map (as a dict label->label)
    when closed; witness is the first found pair (x, x') with Q(x)=Q(x') but
    Q(F(x))!=Q(F(x')) when not. Exact enumeration over all of X.
    """
    F = np.asarray(F, dtype=int)
    Q = np.asarray(Q)
    d = len(F)
    rep: dict = {}  # Q-label -> (representative x, its Q(F(x)) image label)
    for x in range(d):
        qx = Q[x].item() if hasattr(Q[x], "item") else Q[x]
        qfx = Q[F[x]].item() if hasattr(Q[F[x]], "item") else Q[F[x]]
        if qx in rep:
            if rep[qx][1] != qfx:
                return dict(closed=False, G=None, witness=(int(rep[qx][0]), int(x)),
                            witness_label=qx)
        else:
            rep[qx] = (x, qfx)
    G = {qx: img for qx, (x0, img) in rep.items()}
    return dict(closed=True, G=G, witness=None, witness_label=None)


# ===========================================================================
# LAYER A.4 — Theorem D: the symmetric / current decomposition of transition data.
#   For a finite Markov K with positive stationary pi, the L2(pi) adjoint is
#   K*(x,y) = pi_y K(y,x) / pi_x; S = (K+K*)/2, A = (K-K*)/2; the stationary edge
#   current is J(x,y) = pi_x K(x,y) - pi_y K(y,x). (This is the E9/E10 machinery,
#   generalised from uniform pi to arbitrary positive pi.)
# ===========================================================================


def stationary(K: np.ndarray) -> np.ndarray:
    """Stationary law of a row-stochastic K (left 1-eigenvector, normalised, >0)."""
    w, V = np.linalg.eig(K.T)
    i = int(np.argmin(np.abs(w - 1.0)))
    v = np.real(V[:, i])
    v = v / v.sum()
    return v


def l2pi_adjoint(K: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """K*(x,y) = pi_y K(y,x) / pi_x — the adjoint of K in L2(pi)."""
    pi = np.asarray(pi, dtype=float)
    return (pi[None, :] * K.T) / pi[:, None]


def sym_anti_pi(K: np.ndarray, pi: np.ndarray):
    """S = (K + K*)/2 (reversible part), A = (K - K*)/2 (the current)."""
    Kadj = l2pi_adjoint(K, pi)
    return 0.5 * (K + Kadj), 0.5 * (K - Kadj)


def edge_current(K: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """J(x,y) = pi_x K(x,y) - pi_y K(y,x). Antisymmetric; divergence-free at pi."""
    pi = np.asarray(pi, dtype=float)
    return pi[:, None] * K - (pi[:, None] * K).T


def is_pi_reversible(K: np.ndarray, pi: np.ndarray, tol: float = TOL) -> bool:
    """Detailed balance pi_x K(x,y) = pi_y K(y,x), i.e. A = 0."""
    return float(np.max(np.abs(edge_current(K, pi)))) < tol


# ===========================================================================
# LAYER B — a concrete, swappable representative Poole-style rule.
# ===========================================================================


@dataclass(frozen=True)
class PooleRule:
    """
    A deterministic outer-totalistic Boolean cellular automaton on the finite
    torus (Z/Lx) x (Z/Ly) x (Z/Lz), with a prime-resonance threshold convention.

    A cell with current value `center in {0,1}` and `m` live neighbours updates to:
        center == 0 (dead) :  alive next  <=>  m_eff in birth
        center == 1 (live) :  alive next  <=>  m_eff in survive
    where m_eff applies the resonance convention:
        resonance == "none"        :  m_eff = m
        resonance == "input_shift" :  m_eff = m + 1   if m is prime,  else m
        resonance == "band_shift"  :  m_eff = m, but birth/survive are tested
                                      against {t, t-1} (the band widened down by 1)
                                      whenever m is prime.
    The neighbourhood is the von Neumann (axis ±1) shell on the active axes,
    de-duplicated on the torus (so an L=2 axis contributes one neighbour, not two).

    *** Representative stand-in for Rooke's OTG rule. Swap freely. ***
    The defaults below (B={3}, S={2,3}, input_shift) are a Life-flavoured rule with
    a prime kick; they are illustrative, not claimed to be the OTG dynamics.
    """

    shape: tuple = (3, 3, 1)
    birth: frozenset = frozenset({3})
    survive: frozenset = frozenset({2, 3})
    primes: frozenset = frozenset({2, 3, 5, 7, 11, 13, 17, 19, 23})
    resonance: str = "input_shift"  # "none" | "input_shift" | "band_shift"

    # precomputed, derived fields
    cells: tuple = field(default=(), compare=False, repr=False)
    neighbours: tuple = field(default=(), compare=False, repr=False)

    def __post_init__(self):
        Lx, Ly, Lz = self.shape
        cells = tuple(product(range(Lx), range(Ly), range(Lz)))
        index = {c: i for i, c in enumerate(cells)}
        nbrs = []
        for (x, y, z) in cells:
            shell = set()
            for axis, L, coord in ((0, Lx, x), (1, Ly, y), (2, Lz, z)):
                if L <= 1:
                    continue
                for step in (+1, -1):
                    nc = [x, y, z]
                    nc[axis] = (coord + step) % L
                    shell.add(tuple(nc))
            shell.discard((x, y, z))
            nbrs.append(tuple(index[c] for c in sorted(shell)))
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "neighbours", tuple(nbrs))

    # ---- the local update --------------------------------------------------
    def _alive_next(self, center: int, m: int) -> int:
        prime = m in self.primes
        if self.resonance == "input_shift":
            m_eff = m + 1 if prime else m
            band = self.survive if center else self.birth
            return int(m_eff in band)
        if self.resonance == "band_shift":
            base = self.survive if center else self.birth
            band = base | {t - 1 for t in base} if prime else base
            return int(m in band)
        band = self.survive if center else self.birth  # "none"
        return int(m in band)

    # ---- the global map ----------------------------------------------------
    def step_int(self, s: int) -> int:
        """Synchronous update of the whole torus, state encoded as an integer
        (bit c = value of cell c). Works at any size."""
        out = 0
        for c, nbr in enumerate(self.neighbours):
            center = (s >> c) & 1
            m = 0
            for nc in nbr:
                m += (s >> nc) & 1
            if self._alive_next(center, m):
                out |= (1 << c)
        return out

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_states(self) -> int:
        return 1 << self.n_cells

    def F_array(self, cap: int = 1 << 16) -> np.ndarray:
        """The global map as an array F[s] over ALL states (enumerable instances)."""
        d = self.n_states
        if d > cap:
            raise ValueError(f"{d} states exceeds enumeration cap {cap}; use step_int")
        return np.fromiter((self.step_int(s) for s in range(d)), dtype=np.int64, count=d)

    # ---- torus translations (for the symmetry-quotient sector in P3) -------
    def translate_int(self, s: int, shift: tuple) -> int:
        """Translate a configuration by `shift` = (dx,dy,dz) on the torus."""
        Lx, Ly, Lz = self.shape
        dx, dy, dz = shift
        index = {c: i for i, c in enumerate(self.cells)}
        out = 0
        for c, (x, y, z) in enumerate(self.cells):
            if (s >> c) & 1:
                tc = ((x + dx) % Lx, (y + dy) % Ly, (z + dz) % Lz)
                out |= (1 << index[tc])
        return out

    def all_shifts(self) -> list[tuple]:
        Lx, Ly, Lz = self.shape
        return list(product(range(Lx), range(Ly), range(Lz)))


# ---- sector helpers used by P3 / P4 ---------------------------------------


def popcount(s: int) -> int:
    return int(s).bit_count()


def density_sector(d: int) -> np.ndarray:
    """Q(s) = number of live cells (translation-invariant coarse observable)."""
    return np.fromiter((popcount(s) for s in range(d)), dtype=np.int64, count=d)


def single_cell_sector(d: int, cell: int) -> np.ndarray:
    """Q(s) = value of one fixed cell."""
    return np.fromiter(((s >> cell) & 1 for s in range(d)), dtype=np.int64, count=d)


def window_sector(rule: PooleRule, cell: int) -> np.ndarray:
    """Q(s) = the local window (cell value + its neighbours' values) around `cell`."""
    d = rule.n_states
    nbr = rule.neighbours[cell]
    bits = (cell,) + nbr
    out = np.empty(d, dtype=np.int64)
    for s in range(d):
        w = 0
        for k, b in enumerate(bits):
            w |= ((s >> b) & 1) << k
        out[s] = w
    return out


def attractor_sector(F: np.ndarray) -> np.ndarray:
    """
    Q(s) = id of the eventual cycle (attractor) of s under F. F-closed by
    construction: once the attractor is known, F keeps you in its basin, so
    Q(F(s)) = Q(s) and G = identity. A sanity instance of closure.
    """
    F = np.asarray(F, dtype=int)
    d = len(F)
    label = np.full(d, -1, dtype=np.int64)
    next_id = 0
    for start in range(d):
        if label[start] != -1:
            continue
        path = []
        x = start
        while label[x] == -1 and x not in path:
            path.append(x)
            x = int(F[x])
        if label[x] != -1:
            lab = label[x]
        else:
            # x is the entry into a fresh cycle: walk the cycle, give it an id
            lab = next_id
            next_id += 1
            cyc = x
            while True:
                label[cyc] = lab
                cyc = int(F[cyc])
                if cyc == x:
                    break
        for y in path:
            if label[y] == -1:
                label[y] = lab
    return label


def orbit_sector(rule: PooleRule) -> np.ndarray:
    """
    Q(s) = canonical representative (min integer) of the torus-translation orbit
    of s. Because the rule is translation-equivariant, F descends to orbits:
    Q∘F = G∘Q with G the induced map. A genuine symmetry-renormalised sector.
    """
    d = rule.n_states
    shifts = rule.all_shifts()
    rep = np.empty(d, dtype=np.int64)
    for s in range(d):
        orbit_min = s
        for v in shifts:
            t = rule.translate_int(s, v)
            if t < orbit_min:
                orbit_min = t
        rep[s] = orbit_min
    return rep


def coarse_markov(F: np.ndarray, Q: np.ndarray, mu: np.ndarray | None = None):
    """
    The coarse transition matrix K of a deterministic F observed through a sector
    Q, under a base measure mu on states (uniform by default): for sector values
    a,b,  K[a,b] = mu( Q(s)=a and Q(F(s))=b ) / mu( Q(s)=a ). Row-stochastic.
    Returns (labels, K). This is genuine 'noisy Poole transition data' (Thm D).
    """
    F = np.asarray(F, dtype=int)
    Q = np.asarray(Q)
    d = len(F)
    if mu is None:
        mu = np.full(d, 1.0 / d)
    labels = sorted(set(int(q) for q in Q.tolist()))
    idx = {a: i for i, a in enumerate(labels)}
    m = len(labels)
    K = np.zeros((m, m))
    mass = np.zeros(m)
    for s in range(d):
        a = idx[int(Q[s])]
        b = idx[int(Q[F[s]])]
        K[a, b] += mu[s]
        mass[a] += mu[s]
    K = K / mass[:, None]
    return labels, K
