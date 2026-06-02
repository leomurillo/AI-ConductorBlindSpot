"""
E6 diagnostics  --  torch-free, numpy-only, unit-testable on a laptop.
================================================================================

Every quantity E6 logs is computed here, from plain numpy arrays, so the heavy
GPU script (`e6_grokking_bridge.py`) only has to *produce* the arrays (an
embedding matrix, a logit tensor, a score vector) and hand them over. That split
buys three things:

  1. these diagnostics can be tested with no GPU and no torch (run this file);
  2. the cross-packet mass below is CBS's Definition 6.1 *verbatim* -- the
     `__main__` self-test re-derives the §4.2 ladder counts 18,42,108,252,774,
     so we know the ring arithmetic matches the published paper;
  3. if a diagnostic is mis-specified we can fix it OFFLINE from the logged raw
     vectors, without paying for another GPU run.

THE THREE REGISTERS
-------------------
  D  (dynamical)      embedding_fourier_concentration(W, n)
        the number-token embedding aligning to the ring's slow eigenfunctions
        (its cos_k / sin_k character chart). Rises as the curved chart is built.

  S  (distributional, cheap proxy)   logit_additivity(L, a, b, n)
        share of the logit tensor explained by the sum-class s=(a+b) mod n.
        -> 1 when the head depends only on a+b (the cubic constraint a+b-c=0).

  rho_x (distributional, CBS Def 6.1)   RhoX(n)(u)
        cross-packet cubic mass of the batch-averaged centered score u on Z/nZ.
        This is the ACTUAL Conductor-Blind-Spot diagnostic, not a proxy. It is
        the object CBS Conjecture 5.8 is about (does the cross-packet cubic
        emerge along a training trajectory, and stay null on a control?).

        IMPORTANT -- rho_x needs a COMPOSITE modulus. For prime p every nonzero
        character has conductor p, so there are no cross-packet triples and
        rho_x == 0 identically. Run prime p for the D/S co-emergence; run a
        composite n (12, 30) for the rho_x / Conjecture-5.8 test. RhoX reports
        n_triples_cross == 0 in the degenerate prime case.
"""

from __future__ import annotations

from math import gcd

import numpy as np


# ---------------------------------------------------------------------------
# CBS Definition 6.1 -- cross-packet cubic mass on the ring Z/nZ.
# Vendored verbatim from empirical/pythia_rho_x_sweep/rho_x.py (pure arithmetic,
# no dependency) so the GPU script stays self-contained on a fresh clone.
# ---------------------------------------------------------------------------


def conductor(k: int, n: int) -> int:
    """Conductor of the additive character chi_k on Z/nZ: n / gcd(k, n)."""
    if k % n == 0:
        raise ValueError("trivial character has no conductor packet")
    return n // gcd(k, n)


def selection_rule_triples(n: int) -> list[tuple[int, int, int]]:
    """All (k, l, m) in {1,...,n-1}^3 with k + l + m == 0 (mod n).  O(n^2)."""
    out = []
    for k in range(1, n):
        for l in range(1, n):
            m = (-(k + l)) % n
            if m != 0:
                out.append((k, l, m))
    return out


def cross_packet_triples(n: int) -> list[tuple[int, int, int]]:
    """Selection-rule triples whose three conductors are NOT all equal."""
    out = []
    for k, l, m in selection_rule_triples(n):
        c = (conductor(k, n), conductor(l, n), conductor(m, n))
        if not (c[0] == c[1] == c[2]):
            out.append((k, l, m))
    return out


def packet_summary(n: int) -> dict:
    out: dict = {}
    for k in range(1, n):
        out.setdefault(conductor(k, n), []).append(k)
    return dict(sorted(out.items()))


class RhoX:
    """Cross-packet cubic mass of a centered tangent vector u on Z/nZ.

    Triples are precomputed once for the modulus n; __call__ is then a single
    FFT plus a sum over triples. np.fft.fft uses exp(-2*pi*i*k*y/n), matching
    CBS's DFT convention exactly, so |u_hat[k] u_hat[l] u_hat[m]| is the same
    weight Definition 6.1 sums.
    """

    def __init__(self, n: int):
        self.n = n
        self.triples = selection_rule_triples(n)
        cross = set(cross_packet_triples(n))
        # store as index arrays for a vectorized weight sum
        if self.triples:
            ks, ls, ms = zip(*self.triples)
            self.k = np.asarray(ks)
            self.l = np.asarray(ls)
            self.m = np.asarray(ms)
            self.is_cross = np.asarray([t in cross for t in self.triples], dtype=bool)
        else:  # n < 3, no nontrivial triples
            self.k = self.l = self.m = np.zeros(0, dtype=int)
            self.is_cross = np.zeros(0, dtype=bool)
        self.n_cross = int(self.is_cross.sum())

    def __call__(self, u) -> dict:
        n_total = len(self.triples)
        if self.n_cross == 0:
            # prime / degenerate: no cross-packet structure to measure
            return dict(rho_x=0.0, total_mass=0.0, cross_mass=0.0,
                        n_triples_total=n_total, n_triples_cross=0)
        u = np.asarray(u, dtype=float)
        mag = np.abs(np.fft.fft(u))                       # |u_hat[k]|, length n
        w = mag[self.k] * mag[self.l] * mag[self.m]        # |u_hat_k u_hat_l u_hat_m|
        total = float(w.sum())
        cross = float(w[self.is_cross].sum())
        return dict(rho_x=(cross / total if total > 0 else float("nan")),
                    total_mass=total, cross_mass=cross,
                    n_triples_total=n_total, n_triples_cross=self.n_cross)


# ---------------------------------------------------------------------------
# D-register -- embedding Fourier concentration.
# ---------------------------------------------------------------------------


def embedding_fourier_concentration(W, n: int):
    """Max / top-5 share of the number-token embedding variance held by single
    ring frequencies (cos_k, sin_k). W: [n, d] numpy (rows = token embeddings).

    Returns (max_share, top5_share, shares) where shares[k-1] is the variance
    share of frequency k for k=1..n//2. Uniform-ish pre-grok; a few frequencies
    dominate post-grok (the curved character chart has been acquired).
    """
    W = np.asarray(W, dtype=float)
    W = W - W.mean(0, keepdims=True)
    total = float((W ** 2).sum()) + 1e-12
    idx = np.arange(n)
    shares = []
    for k in range(1, n // 2 + 1):
        c = np.cos(2 * np.pi * k * idx / n)
        s = np.sin(2 * np.pi * k * idx / n)
        c /= np.linalg.norm(c) + 1e-12
        s /= np.linalg.norm(s) + 1e-12
        proj_c = W.T @ c
        proj_s = W.T @ s
        ek = float(proj_c @ proj_c + proj_s @ proj_s)
        shares.append(ek / total)
    shares = np.asarray(shares)
    order = np.sort(shares)[::-1]
    return float(order[0]), float(order[:5].sum()), shares.tolist()


# ---------------------------------------------------------------------------
# S-register -- logit additivity (cheap exact proxy for the ring cubic).
# ---------------------------------------------------------------------------


def logit_additivity(L, a, b, n: int) -> float:
    """Share of the logit tensor explained by the sum-class s=(a+b) mod n alone
    (R^2 of grouping rows of L by s). L: [N, n] logits, a,b: [N] int arrays.
    Low pre-grok (lookup over (a,b)); -> 1 post-grok (head depends only on a+b)."""
    L = np.asarray(L, dtype=float)
    a = np.asarray(a)
    b = np.asarray(b)
    s = (a + b) % n
    Lc = L - L.mean(0, keepdims=True)
    total = float((Lc ** 2).sum()) + 1e-12
    within = 0.0
    for sv in np.unique(s):
        rows = L[s == sv]
        if len(rows):
            r = rows - rows.mean(0, keepdims=True)
            within += float((r ** 2).sum())
    return 1.0 - within / total


# ---------------------------------------------------------------------------
# Persistent logit-cubic input -- the offset profile on the ring.
# ---------------------------------------------------------------------------


def offset_profile(L, a, b, n: int):
    """h[t] = mean over examples of the logit assigned to the class t *below* the
    correct answer (a+b):  h[t] = mean_i L[i, (a_i + b_i - t) mod n].

    h is the average logit as a function of ring offset-from-correct. It is
    PEAKED at t=0 and carries the model's key ring frequencies once it
    generalises; it is roughly FLAT for a structureless (memorising) model. Unlike
    the centered score u, h does NOT vanish as the model groks -- it grows -- so
    rho_x(h - mean h) is a persistent cross-packet cubic that stays well-posed
    through and after the grok. (For a prime modulus rho_x is still 0 by
    construction; the offset profile only rescues the COMPOSITE case.)
    """
    L = np.asarray(L, dtype=float)
    a = np.asarray(a)
    b = np.asarray(b)
    N = L.shape[0]
    correct = (a + b) % n
    rows = np.arange(N)
    h = np.empty(n)
    for t in range(n):
        h[t] = L[rows, (correct - t) % n].mean()
    return h


# ---------------------------------------------------------------------------
# Self-test: arithmetic only, no torch, no GPU. Confirms the ring machinery
# matches CBS, and that the register diagnostics behave at the two extremes.
# ---------------------------------------------------------------------------


def _selftest():
    print("CBS ladder -- cross-packet selection-rule triple counts (Def 6.1):")
    expected = {6: 18, 8: 42, 12: 108, 18: 252, 30: 774}
    ok = True
    for n in (6, 8, 12, 18, 30):
        nt = len(selection_rule_triples(n))
        nx = len(cross_packet_triples(n))
        flag = "OK" if nx == expected[n] else f"!! expected {expected[n]}"
        ok = ok and (nx == expected[n])
        print(f"  n={n:3d}  total_triples={nt:5d}  cross={nx:4d}  {flag}")
    assert ok, "cross-packet counts do not match CBS -- ring arithmetic is wrong"

    print("\nrho_x is degenerate on a prime modulus (expected):")
    rprime = RhoX(13)
    print(f"  n=13  n_triples_cross={rprime.n_cross}  -> rho_x forced 0.0")
    assert rprime.n_cross == 0

    print("\nrho_x on n=12, broadband vs within-packet score:")
    rng = np.random.default_rng(0)
    r12 = RhoX(12)
    # broadband random score: cubic mass spreads over all 110 triples, 108 of
    # which are cross-packet -> rho_x near 1. A score concentrated on the
    # conductor-3 packet {4,8} (the only n=12 packet with internal selection-rule
    # triples (4,4,4),(8,8,8)) puts its mass WITHIN a packet -> rho_x near 0.
    u_broad = rng.standard_normal(12); u_broad -= u_broad.mean()
    u_within = np.cos(2 * np.pi * 4 * np.arange(12) / 12)   # FFT support {4,8}
    u_within -= u_within.mean()
    rho_broad = r12(u_broad)["rho_x"]
    rho_within = r12(u_within)["rho_x"]
    print(f"  broadband score    rho_x={rho_broad:.4f}  (mass spreads -> mostly cross)")
    print(f"  within-packet {{4,8}} rho_x={rho_within:.4f}  (mass on (4,4,4),(8,8,8))")
    assert rho_within < 0.5 < rho_broad, "rho_x fails to separate within/cross"

    print("\nD-register sanity (n=12): random vs single-frequency embedding:")
    d = 16
    W_rand = rng.standard_normal((12, d))
    ang = 2 * np.pi * 3 * np.arange(12) / 12
    W_freq = np.outer(np.cos(ang), rng.standard_normal(d)) \
        + np.outer(np.sin(ang), rng.standard_normal(d))
    mr, _, _ = embedding_fourier_concentration(W_rand, 12)
    mf, _, _ = embedding_fourier_concentration(W_freq, 12)
    print(f"  random embedding   max-share={mr:.3f}")
    print(f"  single-freq embed  max-share={mf:.3f}  (should be ~1)")
    assert mf > 0.9 and mr < 0.6

    print("\nS-register sanity (n=12): lookup vs additive logits:")
    a = np.repeat(np.arange(12), 12)
    b = np.tile(np.arange(12), 12)
    L_lookup = rng.standard_normal((144, 12))                       # depends on (a,b)
    sclass = (a + b) % 12
    L_add = np.stack([np.cos(2 * np.pi * 2 * (sclass - c) / 12) for c in range(12)], 1)
    print(f"  lookup logits   R2_add={logit_additivity(L_lookup, a, b, 12):.3f}  (low)")
    print(f"  additive logits R2_add={logit_additivity(L_add, a, b, 12):.3f}  (~1)")
    assert logit_additivity(L_add, a, b, 12) > 0.95

    print("\noffset-profile rho_x (persistent logit-cubic), n=12:")
    # grokked-like logits built from key freqs spanning conductor packets:
    # w=2 (cond 6), w=3 (cond 4), w=4 (cond 3). h[t] should be sum_w cos(2pi w t/12).
    W = [2, 3, 4]
    Lg = np.array([[sum(np.cos(2 * np.pi * w * ((av + bv - c) % 12) / 12) for w in W)
                    for c in range(12)] for av, bv in zip(a, b)])
    hg = offset_profile(Lg, a, b, 12)
    hg_exp = np.array([sum(np.cos(2 * np.pi * w * t / 12) for w in W) for t in range(12)])
    print(f"  offset profile == sum-of-cos(key freqs): {np.allclose(hg, hg_exp)}")
    assert np.allclose(hg, hg_exp)
    rg = r12(hg - hg.mean())
    Lm = np.random.default_rng(1).standard_normal((144, 12))   # memorised-like = noise
    hm = offset_profile(Lm, a, b, 12)
    rm = r12(hm - hm.mean())
    print(f"  grokked-like:   rho_x={rg['rho_x']:.4f}  total_mass={rg['total_mass']:.4f}  (persistent)")
    print(f"  memorised-like: rho_x={rm['rho_x']:.4f}  total_mass={rm['total_mass']:.4g}  (flat -> tiny)")
    assert rg["total_mass"] > 10 * rm["total_mass"], "offset-profile mass must separate structure from noise"

    print("\nALL SELF-TESTS PASSED.")


if __name__ == "__main__":
    _selftest()
