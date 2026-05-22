"""Cross-packet cubic mass diagnostic for ring-structured categorical heads.

Implements Definition 6.1 of ConductorBlindSpot.md with the T_{p_*} weighting
(uniform n^3 on selection-rule triples). For batches whose predicted points
are near the maximum-entropy point this matches Def 6.1 to leading order;
the off-centroid correction (T at p_bar) can be added later if needed.

Input convention: u is the batch-averaged centered score on the ring,
    u_y = mean over batch of (1[y == label_i] - p_i(y)),  y in Z/nZ,
i.e. u lives in the simplex tangent space at p_bar (sum_y u_y = 0).
"""

from __future__ import annotations
from math import gcd
from typing import Sequence
import cmath


def conductor(k: int, n: int) -> int:
    """Conductor of additive character chi_k on Z/nZ: n / gcd(k, n)."""
    if k % n == 0:
        raise ValueError("trivial character has no conductor packet")
    return n // gcd(k, n)


def selection_rule_triples(n: int) -> list[tuple[int, int, int]]:
    """All (k, l, m) in {1,...,n-1}^3 with k + l + m == 0 mod n."""
    return [
        (k, l, m)
        for k in range(1, n)
        for l in range(1, n)
        for m in range(1, n)
        if (k + l + m) % n == 0
    ]


def cross_packet_triples(n: int) -> list[tuple[int, int, int]]:
    """Selection-rule triples whose three conductors are not all equal."""
    out = []
    for k, l, m in selection_rule_triples(n):
        c = (conductor(k, n), conductor(l, n), conductor(m, n))
        if not (c[0] == c[1] == c[2]):
            out.append((k, l, m))
    return out


def dft(u: Sequence[float], n: int) -> list[complex]:
    """Standard DFT: u_hat[k] = sum_y u[y] * exp(-2*pi*i*k*y/n).

    Returns the full length-n spectrum (u_hat[0] should be ~0 since
    sum_y u_y = 0 for a tangent vector).
    """
    assert len(u) == n
    out = []
    for k in range(n):
        s = 0 + 0j
        for y in range(n):
            s += u[y] * cmath.exp(-2j * cmath.pi * k * y / n)
        out.append(s)
    return out


def rho_x(u: Sequence[float], n: int) -> dict:
    """Cross-packet cubic mass of u on the ring Z/nZ.

    Returns a dict with:
      rho_x:        |cross-packet selection-rule cubic mass| / |total|
      total_mass:   sum_{selection-rule triples} |u_hat[k] u_hat[l] u_hat[m]|
      cross_mass:   sum_{cross-packet selection-rule triples} |...|
      n_triples_total: count of selection-rule triples
      n_triples_cross: count of cross-packet selection-rule triples
    """
    u_hat = dft(u, n)
    all_triples = selection_rule_triples(n)
    cross = set(cross_packet_triples(n))

    total = 0.0
    cross_mass = 0.0
    for k, l, m in all_triples:
        weight = abs(u_hat[k] * u_hat[l] * u_hat[m])
        total += weight
        if (k, l, m) in cross:
            cross_mass += weight

    return {
        "rho_x": cross_mass / total if total > 0 else float("nan"),
        "total_mass": total,
        "cross_mass": cross_mass,
        "n_triples_total": len(all_triples),
        "n_triples_cross": len(cross),
    }


def packet_summary(n: int) -> dict[int, list[int]]:
    """For each divisor d > 1 of n, list the k in {1,...,n-1} with cond(k) == d."""
    out: dict[int, list[int]] = {}
    for k in range(1, n):
        d = conductor(k, n)
        out.setdefault(d, []).append(k)
    return dict(sorted(out.items()))


if __name__ == "__main__":
    # Self-check: replicate the §4.2 ladder triple counts.
    for n in (6, 8, 12, 18, 30):
        triples = selection_rule_triples(n)
        cross = cross_packet_triples(n)
        same = len(triples) - len(cross)
        print(f"n={n:3d}  packets={packet_summary(n)}  "
              f"total_triples={len(triples):4d}  cross={len(cross):4d}  same={same:3d}")
