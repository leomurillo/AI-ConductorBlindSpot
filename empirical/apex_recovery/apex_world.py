"""
apex_world.py — shared toolkit for the Apex-Matched Eigenfunction Recovery experiments.
================================================================================

WHAT THIS FILE IS FOR
---------------------
The companion paper proves a *non-Gaussian* generalisation of LeJEPA's
identifiability theorem (Klindt, LeCun & Balestriero 2026, arXiv:2605.26379).
This module is the small, exact, dependency-light backbone every experiment
in this folder shares. It has no machine learning in it: the whole point of
the theory is that the learning problem *collapses to a linear-algebra problem*
about a single self-adjoint operator, so the experiments are linear algebra.

THE ONE PICTURE TO HOLD IN YOUR HEAD
------------------------------------
A "world" is a latent variable z with a stationary law p(z), observed through
positive pairs (z, z') that evolve by a *reversible, additive-noise* Markov
transition. Self-supervised learning à la LeJEPA pulls the embeddings of a
pair together (alignment) while keeping the embedding whitened (anti-collapse).

    LeJEPA's mechanism, stated operator-theoretically:
    minimising alignment subject to whitening  ==  finding the SLOWEST
    non-constant functions of z, i.e. the top eigenfunctions of the
    transition operator  T f (z) = E[ f(z') | z ].

    * For a GAUSSIAN world, the slowest function is *linear*: phi_1(z) = z.
      => recovering it = recovering z up to rotation = LINEAR identifiability.
      This is the only world where that happens. (LeJEPA Thm 1 & 2.)

    * For ANY OTHER world, the slowest function phi_1 is a *monotone but
      nonlinear* reparametrisation of z. You still recover z perfectly —
      but THROUGH phi_1, not linearly. A linear probe necessarily fails.
      This is our theorem; LeJEPA is its rank-2 / Gaussian corner.

WHY A FINITE MARKOV CHAIN
-------------------------
We realise each world as a reversible nearest-neighbour Markov chain on a grid
(a "Metropolis random walk that prefers high-probability states"). This is the
exact, fully diagonalisable, discrete cousin of LeJEPA's continuous
Ornstein–Uhlenbeck transition. It has three virtues:

  1. It is *exactly* reversible, so the transition operator T is *self-adjoint*
     in L2(pi) and has a real eigenbasis we can compute to machine precision.
  2. The Gaussian target reproduces (a discretisation of) Ornstein–Uhlenbeck,
     whose slow eigenfunction is affine — LeJEPA's world, recovered.
  3. The finite chain makes the world's distributional structure finite
     spectral data, reconstructible from finitely many moments (see e3_*) —
     the finite-determinacy view restated in-house in the paper's appendix.

Everything below is plain numpy + a symmetric eigensolver. No randomness in the
operators; the only seeds are for the small contamination demos in e2_*.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Worlds: stationary laws on a grid.
# ---------------------------------------------------------------------------
#
# A world is specified by its log-density (up to a constant) on a grid of z
# values. We provide the three that matter for the LeJEPA comparison:
#
#   * "gaussian"  — V(z) = z^2 / 2.  Score (log p)' = -z is LINEAR.
#                   This is the unique world with an affine slow eigenfunction.
#   * "laplace"   — V(z) = |z|/b.    Score is the SIGN function: not linear.
#   * "bimodal"   — a two-well mixture. Score is cubic-ish: strongly nonlinear.
#
# The "affine iff Gaussian" boundary (LeJEPA Thm 2) is, at bottom, the
# statement that  (log p)'  is linear iff p is Gaussian — visible directly in
# these scores, and then again spectrally in phi_1.


def grid(n_points: int = 401, half_width: float = 6.0) -> np.ndarray:
    """A symmetric, evenly spaced grid of latent values z."""
    return np.linspace(-half_width, half_width, n_points)


def log_density(world: str, z: np.ndarray) -> np.ndarray:
    """Unnormalised log p(z) for the named world (constants drop out below)."""
    if world == "gaussian":
        return -0.5 * z**2
    if world == "laplace":
        b = 1.0
        return -np.abs(z) / b
    if world == "bimodal":
        # Symmetric double well: density concentrates near z = +/- mu.
        mu, s = 2.4, 1.1
        a = np.exp(-0.5 * ((z - mu) / s) ** 2)
        c = np.exp(-0.5 * ((z + mu) / s) ** 2)
        return np.log(a + c)
    if world == "uniform":
        # Flat box on [-a, a]: a bounded world whose slow eigenfunction is a
        # half-cosine — manifestly nonlinear in z near the walls.
        a = 3.0
        return np.where(np.abs(z) <= a, 0.0, -60.0)
    raise ValueError(f"unknown world: {world!r}")


def score(world: str, z: np.ndarray) -> np.ndarray:
    """
    The score (log p)'(z), shown explicitly because LeJEPA's converse is
    'phi_1 affine  <=>  score linear  <=>  Gaussian'. You can read the whole
    boundary off this one function:
        gaussian -> -z          (linear:    YES  -> affine phi_1)
        laplace  -> -sign(z)/b  (linear:    no)
        bimodal  -> nonlinear   (linear:    no)
    """
    if world == "gaussian":
        return -z
    if world == "laplace":
        b = 1.0
        return -np.sign(z) / b
    if world == "bimodal":
        mu, s = 2.4, 1.1
        a = np.exp(-0.5 * ((z - mu) / s) ** 2)
        c = np.exp(-0.5 * ((z + mu) / s) ** 2)
        # d/dz log(a + c)
        da = a * (-(z - mu) / s**2)
        dc = c * (-(z + mu) / s**2)
        return (da + dc) / (a + c)
    if world == "uniform":
        return np.zeros_like(z)  # flat inside the box
    raise ValueError(f"unknown world: {world!r}")


# ---------------------------------------------------------------------------
# 2. The reversible transition operator and its eigenbasis.
# ---------------------------------------------------------------------------


def stationary(world: str, z: np.ndarray) -> np.ndarray:
    """Normalised stationary law pi on the grid."""
    lp = log_density(world, z)
    pi = np.exp(lp - lp.max())
    return pi / pi.sum()


def metropolis_chain(pi: np.ndarray) -> np.ndarray:
    """
    Reversible nearest-neighbour Metropolis chain with stationary law pi.

    Proposal: step to a neighbour (left/right) with probability 1/2 each.
    Acceptance: min(1, pi_j / pi_i)  — the Metropolis rule.
    Rejected mass stays put (the diagonal). This satisfies detailed balance
        pi_i P_ij = pi_j P_ji
    exactly, so the transition operator is self-adjoint in L2(pi). It is the
    discrete, exactly-diagonalisable cousin of LeJEPA's OU transition.

    Returns the row-stochastic transition matrix P (P[i, j] = P(i -> j)).
    """
    m = len(pi)
    P = np.zeros((m, m))
    for i in range(m):
        for j in (i - 1, i + 1):
            if 0 <= j < m:
                P[i, j] = 0.5 * min(1.0, pi[j] / pi[i])
        P[i, i] = 1.0 - P[i].sum()
    return P


def eigenbasis(P: np.ndarray, pi: np.ndarray):
    """
    Eigen-decompose the reversible operator T = P in L2(pi).

    Trick: T is self-adjoint in the pi-weighted inner product, not the plain
    one. Symmetrise with S = D^{1/2} P D^{-1/2}, D = diag(pi). S is a genuine
    symmetric matrix; eigh gives an orthonormal eigenbasis of S, and
    phi = D^{-1/2} (eigvecs of S) are the eigenfunctions of T, orthonormal in
    L2(pi):  sum_i pi_i phi_k(i) phi_l(i) = delta_kl.

    Returns
    -------
    lam : eigenvalues sorted DESCENDING (lam[0] = 1, the constant mode).
    phi : phi[:, k] is the k-th eigenfunction on the grid, L2(pi)-orthonormal,
          sign-fixed so the first nonzero bulk slope is positive.
    """
    d = np.sqrt(pi)
    S = (d[:, None]) * P * (1.0 / d[None, :])
    S = 0.5 * (S + S.T)  # kill asymmetry from finite precision
    w, V = np.linalg.eigh(S)  # ascending
    order = np.argsort(w)[::-1]  # descending: lam[0] ~ 1
    lam = w[order]
    phi = V[:, order] / d[:, None]
    # Normalise to L2(pi) = 1 (eigh already orthonormal in S; division by d
    # restores the pi-weighting exactly) and fix signs for readability.
    for k in range(phi.shape[1]):
        nrm = np.sqrt(np.sum(pi * phi[:, k] ** 2))
        phi[:, k] /= nrm
        # sign convention: positive correlation with the grid coordinate
        if np.sum(pi * phi[:, k] * np.arange(len(pi))) < 0:
            phi[:, k] *= -1.0
    return lam, phi


def transition_eigh(world: str, n_points: int = 401, half_width: float = 6.0):
    """Convenience: build a world and return (z, pi, lam, phi)."""
    z = grid(n_points, half_width)
    pi = stationary(world, z)
    P = metropolis_chain(pi)
    lam, phi = eigenbasis(P, pi)
    return z, pi, lam, phi


# ---------------------------------------------------------------------------
# 3. Measuring recovery.
# ---------------------------------------------------------------------------


def pi_inner(pi, f, g):
    """pi-weighted inner product <f, g> = sum_i pi_i f_i g_i."""
    return float(np.sum(pi * f * g))


def affine_nonlinearity(pi, z, phi1):
    """
    How far is the slow eigenfunction phi_1 from being an affine function of z?

    Returns nu = 1 - R^2 of the best pi-weighted affine fit  phi_1 ~ a*z + b.
        nu ~ 0  : phi_1 is linear  -> LINEAR identifiability holds (Gaussian).
        nu  > 0 : phi_1 is nonlinear -> linear probe necessarily loses content.
    This single scalar is LeJEPA's order-2/order-3 boundary, measured.
    """
    # weighted least squares of phi1 on [z, 1]
    w = pi
    Z = np.vstack([z, np.ones_like(z)]).T
    WZ = Z * w[:, None]
    coef = np.linalg.solve(Z.T @ WZ, Z.T @ (w * phi1))
    fit = Z @ coef
    ss_res = float(np.sum(w * (phi1 - fit) ** 2))
    ss_tot = float(np.sum(w * (phi1 - np.sum(w * phi1)) ** 2))
    # nu = 1 - R^2 = residual variance fraction = the nonlinearity.
    # A constant function (ss_tot = 0) is trivially affine: nu = 0.
    if ss_tot <= 1e-300:
        return 0.0, coef
    return ss_res / ss_tot, coef


def is_monotone(phi1, tol=1e-9):
    """LeJEPA / Sturm–Liouville: the first eigenfunction has no interior sign
    change in its increments. We check phi_1 is (weakly) monotone on the grid."""
    d = np.diff(phi1)
    d = d[np.abs(d) > tol]
    if len(d) == 0:
        return True
    return bool(np.all(d > 0) or np.all(d < 0))


def procrustes_recovery_error(pi, H, Phi):
    """
    The quantity the approximate theorem bounds:
        min_{U orthogonal}  || H - U Phi ||^2_{L2(pi)}     (averaged per coord)

    H   : (m, n) learned embedding coordinates on the grid.
    Phi : (m, n) target recovery coordinates (the top-n eigenfunctions).
    Both are columns of functions on the grid; inner products are pi-weighted.

    Solved exactly by orthogonal Procrustes on the n x n cross-Gram
        C_{ab} = <H_a, Phi_b>_pi ,   min_U ||...|| uses U = (C C^T)^{-1/2} C ...
    We return the achieved squared error per coordinate, plus the principal-
    angle leakage theta^2 = sum sin^2 (sum of squared sines).
    """
    m, n = H.shape
    # Gram of H (should be ~ I if whitened) and cross-Gram with Phi.
    GH = np.array([[pi_inner(pi, H[:, a], H[:, b]) for b in range(n)] for a in range(n)])
    C = np.array([[pi_inner(pi, H[:, a], Phi[:, b]) for b in range(n)] for a in range(n)])
    # Best orthogonal U aligning Phi to H: from SVD of C.
    Uu, s, Vt = np.linalg.svd(C)
    Uopt = Uu @ Vt
    # || H - U Phi ||^2 = tr(GH) - 2 tr(U C^T) + tr(GPhi);  GPhi = I (Phi orthonormal)
    err = float(np.trace(GH) - 2.0 * np.sum(s) + n)
    # principal-angle leakage of span(H) out of span(Phi):
    # sum sin^2 = n - sum sigma^2(C-after-whitening). Use whitened cross-corr.
    # For (near-)orthonormal H, singular values of C are cosines of angles.
    theta2 = float(n - np.sum(np.clip(s, 0, 1) ** 2))
    return err, theta2, GH


def gap_from_spectrum(lams_per_coord, n):
    """
    The spectral gap gamma that controls the approximate theorem, for a product
    world whose per-coordinate transition spectra are given.

    lams_per_coord : list of 1-D arrays, each the DESCENDING spectrum of one
                     coordinate's transition operator (lam[0] = 1 = const mode).
    n              : number of latent coordinates (= number retained).

    Retained band  = the n single-coordinate first modes {lam_1^(i)}.
    First discarded = max over (any second mode lam_2^(i)) and
                      (any two-coordinate product lam_1^(i) lam_1^(j)).
    gamma = min_i lam_1^(i)  -  first_discarded.
    Assumption (G) is exactly gamma > 0.
    """
    first = np.array([l[1] for l in lams_per_coord[:n]])  # lam_1 per coord
    second = np.array([l[2] for l in lams_per_coord[:n]])  # lam_2 per coord
    retained_min = float(first.min())
    # competitors among the discarded non-trivial modes:
    comp_second = float(second.max())
    comp_pair = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                comp_pair = max(comp_pair, float(first[i] * first[j]))
    first_discarded = max(comp_second, comp_pair)
    return retained_min - first_discarded, retained_min, first_discarded


__all__ = [
    "grid",
    "log_density",
    "score",
    "stationary",
    "metropolis_chain",
    "eigenbasis",
    "transition_eigh",
    "pi_inner",
    "affine_nonlinearity",
    "is_monotone",
    "procrustes_recovery_error",
    "gap_from_spectrum",
]
