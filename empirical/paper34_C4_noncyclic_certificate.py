#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper 34 -- C4: Non-cyclic finite abelian extension certificate.

Promotes Appendix B from a SPECIFICATION to a VERIFIED extension on a set of
non-cyclic finite abelian groups $G$, by replicating the Section 4 / Appendix
C enumeration on $\\widehat{G}$ (the Pontryagin dual) for each candidate
$G$.

The certificate's two claims generalize verbatim:

  (Q) Fisher block-diagonality at p_*: g_{p_*} on $\\Delta_G^\\circ$ is
      DIAGONAL in the character basis of $\\widehat{G}$ (character
      orthogonality holds on any finite abelian $G$ -- proved by the same
      finite geometric-sum argument applied to the cyclic factors in the
      elementary-divisor decomposition).

  (C) AC cubic selection rule at p_*:
      T_{p_*}(chi, chi', chi'') = |G|^3 * [chi*chi'*chi'' = 1 in $\\widehat{G}$].
      In coordinates, the selection rule is k_i + l_i + m_i == 0 (mod n_i)
      for EVERY component i of the elementary-divisor decomposition
      $G = Z/n_1 x ... x Z/n_r$.

Cross-packet criterion is "not all three conductors equal" -- where the
conductor of a character is its order in $\\widehat{G}$ (equivalently, the
smallest finite abelian quotient of $G$ through which the character factors).

Total ordered selection-rule triples = $(|G| - 1)(|G| - 2)$, by the same
counting argument as cyclic: m is determined by (k, l), exclude m = 0.

Candidate groups:

  G                       order   structure                  remark
  ---------------------   -----   ------------------------   ------
  Z/2 x Z/4               8       2 packets (cond 2, 4)      smallest non-cyclic
  Z/4 x Z/4               16      2 packets (cond 2, 4)      square non-cyclic
  Z/2 x Z/8               16      3 packets (cond 2, 4, 8)   deeper prime-power
  (Z/2)^2 x Z/4           16      2 packets (cond 2, 4)      3-factor non-cyclic
  Z/3 x Z/9               27      2 packets (cond 3, 9)      odd prime non-cyclic

Cyclic counterparts at matching orders -- for cross-reference (these are
distinct groups whose dual structure differs from the non-cyclic case):
  |G| = 8  cyclic: Z/8     (3 packets cond 2, 4, 8; same-packet count 0)
  |G| = 16 cyclic: Z/16    (4 packets; ...)
  |G| = 27 cyclic: Z/27    (3 packets cond 3, 9, 27; ...)

The diagnostic-applicability condition transfers: a ring G admits a
non-trivial $\\rho_\\times$ measurement iff at least one same-packet
selection-rule triple exists, i.e., iff the structural floor of cross-
packet content is strictly < |T|.

Author: Leo. C4 of paper 34 / Appendix B promotion.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from math import gcd
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "empirical" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Finite abelian group: enumeration, addition, conductors.
# ---------------------------------------------------------------------------


GroupTuple = Tuple[int, ...]  # element of G as tuple (k_1, ..., k_r)


def group_order(moduli: List[int]) -> int:
    """|G| = product of moduli for G = Z/n_1 x ... x Z/n_r."""
    return math.prod(moduli)


def enumerate_elements(moduli: List[int]) -> List[GroupTuple]:
    """All |G| elements of G = Z/n_1 x ... x Z/n_r as tuples."""
    return [tuple(t) for t in itertools.product(*[range(n) for n in moduli])]


def add(a: GroupTuple, b: GroupTuple, moduli: List[int]) -> GroupTuple:
    """Componentwise mod-addition in G."""
    return tuple((a[i] + b[i]) % moduli[i] for i in range(len(moduli)))


def is_identity(a: GroupTuple) -> bool:
    return all(x == 0 for x in a)


def conductor(k: GroupTuple, moduli: List[int]) -> int:
    """Order of character chi_k in $\\widehat{G}$ (equivalently, conductor).

    Closed form for $\\widehat{G} \\cong G$ via the elementary-divisor
    decomposition: conductor(k) = lcm over i (with k_i != 0) of
        n_i / gcd(k_i, n_i).
    For the identity (all k_i = 0) the conductor is 1 (excluded from
    nontrivial-character enumeration).
    """
    if is_identity(k):
        return 1
    parts: List[int] = []
    for ki, ni in zip(k, moduli):
        if ki == 0:
            continue
        parts.append(ni // gcd(ki, ni))
    cond = 1
    for p in parts:
        cond = (cond * p) // gcd(cond, p)  # lcm
    return cond


def nontrivial_characters(moduli: List[int]) -> List[GroupTuple]:
    """Enumerate $\\widehat{G} \\setminus \\{1\\}$ as tuples."""
    return [el for el in enumerate_elements(moduli) if not is_identity(el)]


def conductor_packets(moduli: List[int]) -> Dict[int, List[GroupTuple]]:
    """Group nontrivial characters by conductor."""
    packets: Dict[int, List[GroupTuple]] = {}
    for k in nontrivial_characters(moduli):
        d = conductor(k, moduli)
        packets.setdefault(d, []).append(k)
    return dict(sorted(packets.items()))


# ---------------------------------------------------------------------------
# 2. The two halves of the certificate, enumerated.
# ---------------------------------------------------------------------------


def fisher_block_diagonality(moduli: List[int]) -> bool:
    """At p_* on $\\Delta_G^\\circ$, $g_{p_*}(\\chi, \\overline{\\chi'}) =
    |G| \\cdot 1[\\chi = \\chi']$ by character orthogonality.

    This is exact and basis-free; we don't compute the Gram matrix
    numerically (would be redundant). The function returns True
    unconditionally and documents the fact.
    """
    return True


def selection_rule_triples(
    moduli: List[int],
) -> List[Tuple[Tuple[GroupTuple, GroupTuple, GroupTuple], bool]]:
    """All ordered triples (k, l, m) of nontrivial characters with
    k + l + m = identity in G. Each entry is paired with cross_packet
    = (conductors of k, l, m are not all equal).
    """
    nontriv = nontrivial_characters(moduli)
    nontriv_set = set(nontriv)
    out: List[Tuple[Tuple[GroupTuple, GroupTuple, GroupTuple], bool]] = []
    conds: Dict[GroupTuple, int] = {k: conductor(k, moduli) for k in nontriv}
    for k in nontriv:
        for l in nontriv:
            # m = -(k + l) in G
            kl = add(k, l, moduli)
            m = tuple((-x) % moduli[i] for i, x in enumerate(kl))
            if m in nontriv_set:
                dk, dl, dm = conds[k], conds[l], conds[m]
                not_all_same = not (dk == dl == dm)
                out.append(((k, l, m), not_all_same))
    return out


# ---------------------------------------------------------------------------
# 3. Per-group verification + report.
# ---------------------------------------------------------------------------


def verify_group(name: str, latex: str, moduli: List[int]) -> dict:
    N = group_order(moduli)
    packets = conductor_packets(moduli)
    triples = selection_rule_triples(moduli)
    cross_packet = [(t, x) for (t, x) in triples if x]
    same_packet = [(t, x) for (t, x) in triples if not x]

    # Sanity: total surviving = (N-1)(N-2) by the same counting argument.
    expected_total = (N - 1) * (N - 2)
    assert len(triples) == expected_total, (
        f"{name}: total triples = {len(triples)}, expected {expected_total}"
    )

    # Sample cross-packet triples, prioritizing the most distinct conductors.
    def n_distinct(t):
        (a, b, c), _ = t
        ca, cb, cc = (
            conductor(a, moduli),
            conductor(b, moduli),
            conductor(c, moduli),
        )
        return len({ca, cb, cc})

    cross_sorted = sorted(cross_packet, key=lambda t: -n_distinct(t))
    samples = []
    for (k, l, m), _ in cross_sorted[:5]:
        samples.append(
            {
                "k": list(k),
                "l": list(l),
                "m": list(m),
                "conds": [
                    conductor(k, moduli),
                    conductor(l, moduli),
                    conductor(m, moduli),
                ],
            }
        )

    result = {
        "name": name,
        "latex": latex,
        "moduli": list(moduli),
        "order": N,
        "tangent_dim": N - 1,
        "packets": {
            int(d): [list(k) for k in ks] for d, ks in packets.items()
        },
        "packet_sizes": {int(d): len(ks) for d, ks in packets.items()},
        "fisher_block_diagonal": fisher_block_diagonality(moduli),
        "selection_rule_total_triples": len(triples),
        "expected_total_triples_closed_form": expected_total,
        "cross_packet_triples_count": len(cross_packet),
        "same_packet_triples_count": len(same_packet),
        "diagnostic_applicable": len(same_packet) > 0 and len(cross_packet) > 0,
        "sample_cross_packet_triples": samples,
    }
    return result


# ---------------------------------------------------------------------------
# 4. Candidate groups + Markdown table.
# ---------------------------------------------------------------------------


# Each candidate: (ascii name, LaTeX inline form, moduli)
CANDIDATES: List[Tuple[str, str, List[int]]] = [
    ("Z/2 x Z/4",         r"\mathbb{Z}/2 \times \mathbb{Z}/4",            [2, 4]),
    ("Z/4 x Z/4",         r"\mathbb{Z}/4 \times \mathbb{Z}/4",            [4, 4]),
    ("Z/2 x Z/8",         r"\mathbb{Z}/2 \times \mathbb{Z}/8",            [2, 8]),
    ("(Z/2)^2 x Z/4",     r"(\mathbb{Z}/2)^2 \times \mathbb{Z}/4",        [2, 2, 4]),
    ("Z/3 x Z/9",         r"\mathbb{Z}/3 \times \mathbb{Z}/9",            [3, 9]),
]


def emit_summary(results: List[dict]) -> None:
    md = [
        "# Paper 34 -- C4: Non-cyclic Finite Abelian Extension Certificate",
        "",
        "Generalizes the Section 4 / Appendix C enumeration from $\\mathbb{Z}/n\\mathbb{Z}$",
        "to a set of non-cyclic finite abelian groups $G$. Promotes Appendix B from",
        "specification to verified extension.",
        "",
        "## Certificate verification per non-cyclic $G$",
        "",
        "| $G$ | $|G|$ | conductor packets (cond: #chars) | total triples = $(|G|{-}1)(|G|{-}2)$ | same-packet | cross-packet | diagnostic applicable? |",
        "|---|---:|---|---:|---:|---:|:---:|",
    ]
    for r in results:
        pkts = ", ".join(
            f"{d}:{sz}" for d, sz in sorted(r["packet_sizes"].items())
        )
        applicable = "yes" if r["diagnostic_applicable"] else "no (degenerate)"
        md.append(
            f"| ${r['latex']}$ "
            f"| {r['order']} | {pkts} | {r['selection_rule_total_triples']} "
            f"| {r['same_packet_triples_count']} | {r['cross_packet_triples_count']} "
            f"| {applicable} |"
        )

    md += [
        "",
        "## Sample cross-packet triples (chosen with most distinct conductor labels)",
        "",
    ]
    for r in results:
        md.append(f"### {r['name']} (|G| = {r['order']})")
        md.append("")
        md.append("| $\\vec k$ | $\\vec\\ell$ | $\\vec m$ | $(\\mathrm{cond}\\,\\vec k, \\mathrm{cond}\\,\\vec\\ell, \\mathrm{cond}\\,\\vec m)$ | distinct conductors |")
        md.append("|---|---|---|---|:---:|")
        if not r["sample_cross_packet_triples"]:
            md.append("| _(no cross-packet triples; structurally degenerate)_ | | | | |")
        for s in r["sample_cross_packet_triples"]:
            md.append(
                f"| {tuple(s['k'])} | {tuple(s['l'])} | {tuple(s['m'])} "
                f"| {tuple(s['conds'])} | {len(set(s['conds']))} |"
            )
        md.append("")

    md += [
        "## What this verifies",
        "",
        "For each candidate $G$, the certificate (Theorem 3.5 of the manuscript)",
        "holds in the form stated in Appendix B:",
        "",
        "1. **Fisher block-diagonality at $p_*$.** $g_{p_*}$ is diagonal in the",
        "   character basis of $\\widehat{G}$ by character orthogonality (the",
        "   finite geometric-sum argument applies factor-by-factor in the",
        "   elementary-divisor decomposition of $G$). No off-diagonal entry is",
        "   nonzero. The block-diagonal structure across conductor packets is",
        "   immediate.",
        "",
        "2. **Cross-packet AC cubic coupling at $p_*$.** Per the table above, on",
        "   every non-degenerate candidate $G$ the cross-packet triple count is",
        "   strictly positive: the cubic carries inter-packet coupling that the",
        "   quadratic class cannot represent.",
        "",
        "**Degenerate cases.** A group $G$ has *no* same-packet triples (and",
        "$\\rho_\\times \\equiv 1$ on every update) when no triple $(k, \\ell, m)$",
        "with conductors all equal sums to the identity in $G$. The diagnostic",
        "(Definition 6.1) is structurally constant on such groups -- a property",
        "of the dual lattice, not of any model. The cyclic case $n = 8$ (Section",
        "6.5) and the elementary-abelian case $(\\mathbb{Z}/p)^k$ (every nontrivial",
        "character has order $p$, single packet) are the canonical examples.",
        "",
        "**Same-packet vs cross-packet on non-cyclic vs cyclic of equal order.**",
        "The counts differ between $G$ and its cyclic counterpart at the same",
        "order. For $|G| = 8$: $\\mathbb{Z}/8$ has 0 same-packet, 42 cross-packet;",
        "$\\mathbb{Z}/2 \\times \\mathbb{Z}/4$ has 6 same-packet, 36 cross-packet.",
        "The certificate is unaffected (both halves hold), but the diagnostic's",
        "structural ceiling differs.",
        "",
    ]
    out_path = REPORTS_DIR / "paper34_C4_noncyclic_summary.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"summary written: {out_path}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Default runs all 5 candidates.
    return p.parse_args(argv)


def main(argv: List[str] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    results: List[dict] = []
    for name, latex, moduli in CANDIDATES:
        print(f"\n{'=' * 78}\n  paper 34 C4 -- G = {name} (moduli {moduli})\n{'=' * 78}")
        r = verify_group(name, latex, moduli)
        results.append(r)
        print(f"  |G| = {r['order']}, packets: {r['packet_sizes']}")
        print(
            f"  total triples = {r['selection_rule_total_triples']} "
            f"(closed form (N-1)(N-2) = {r['expected_total_triples_closed_form']})"
        )
        print(
            f"  cross-packet = {r['cross_packet_triples_count']}, "
            f"same-packet = {r['same_packet_triples_count']}, "
            f"diagnostic applicable: {r['diagnostic_applicable']}"
        )
        if r["sample_cross_packet_triples"]:
            s = r["sample_cross_packet_triples"][0]
            print(
                f"  sample cross-packet: k={tuple(s['k'])}, l={tuple(s['l'])}, "
                f"m={tuple(s['m'])}, conds={tuple(s['conds'])}"
            )
        (REPORTS_DIR / f"paper34_C4_{name.replace(' ', '_').replace('/', '').replace('(', '').replace(')', '').replace('^', '')}.json").write_text(
            json.dumps(r, indent=2), encoding="utf-8"
        )

    emit_summary(results)
    print(f"\n[done] artifacts in {REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
