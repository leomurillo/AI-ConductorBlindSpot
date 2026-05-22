# Paper 34 -- C4: Non-cyclic Finite Abelian Extension Certificate

Generalizes the Section 4 / Appendix C enumeration from $\mathbb{Z}/n\mathbb{Z}$
to a set of non-cyclic finite abelian groups $G$. Promotes Appendix B from
specification to verified extension.

## Certificate verification per non-cyclic $G$

| $G$ | $|G|$ | conductor packets (cond: #chars) | total triples = $(|G|{-}1)(|G|{-}2)$ | same-packet | cross-packet | diagnostic applicable? |
|---|---:|---|---:|---:|---:|:---:|
| $\mathbb{Z}/2 \times \mathbb{Z}/4$ | 8 | 2:3, 4:4 | 42 | 6 | 36 | yes |
| $\mathbb{Z}/4 \times \mathbb{Z}/4$ | 16 | 2:3, 4:12 | 210 | 102 | 108 | yes |
| $\mathbb{Z}/2 \times \mathbb{Z}/8$ | 16 | 2:3, 4:4, 8:8 | 210 | 6 | 204 | yes |
| $(\mathbb{Z}/2)^2 \times \mathbb{Z}/4$ | 16 | 2:7, 4:8 | 210 | 42 | 168 | yes |
| $\mathbb{Z}/3 \times \mathbb{Z}/9$ | 27 | 3:8, 9:18 | 650 | 218 | 432 | yes |

## Sample cross-packet triples (chosen with most distinct conductor labels)

### Z/2 x Z/4 (|G| = 8)

| $\vec k$ | $\vec\ell$ | $\vec m$ | $(\mathrm{cond}\,\vec k, \mathrm{cond}\,\vec\ell, \mathrm{cond}\,\vec m)$ | distinct conductors |
|---|---|---|---|:---:|
| (0, 1) | (0, 1) | (0, 2) | (4, 4, 2) | 2 |
| (0, 1) | (0, 2) | (0, 1) | (4, 2, 4) | 2 |
| (0, 1) | (1, 0) | (1, 3) | (4, 2, 4) | 2 |
| (0, 1) | (1, 1) | (1, 2) | (4, 4, 2) | 2 |
| (0, 1) | (1, 2) | (1, 1) | (4, 2, 4) | 2 |

### Z/4 x Z/4 (|G| = 16)

| $\vec k$ | $\vec\ell$ | $\vec m$ | $(\mathrm{cond}\,\vec k, \mathrm{cond}\,\vec\ell, \mathrm{cond}\,\vec m)$ | distinct conductors |
|---|---|---|---|:---:|
| (0, 1) | (0, 1) | (0, 2) | (4, 4, 2) | 2 |
| (0, 1) | (0, 2) | (0, 1) | (4, 2, 4) | 2 |
| (0, 1) | (2, 0) | (2, 3) | (4, 2, 4) | 2 |
| (0, 1) | (2, 1) | (2, 2) | (4, 4, 2) | 2 |
| (0, 1) | (2, 2) | (2, 1) | (4, 2, 4) | 2 |

### Z/2 x Z/8 (|G| = 16)

| $\vec k$ | $\vec\ell$ | $\vec m$ | $(\mathrm{cond}\,\vec k, \mathrm{cond}\,\vec\ell, \mathrm{cond}\,\vec m)$ | distinct conductors |
|---|---|---|---|:---:|
| (0, 1) | (0, 1) | (0, 6) | (8, 8, 4) | 2 |
| (0, 1) | (0, 2) | (0, 5) | (8, 4, 8) | 2 |
| (0, 1) | (0, 3) | (0, 4) | (8, 8, 2) | 2 |
| (0, 1) | (0, 4) | (0, 3) | (8, 2, 8) | 2 |
| (0, 1) | (0, 5) | (0, 2) | (8, 8, 4) | 2 |

### (Z/2)^2 x Z/4 (|G| = 16)

| $\vec k$ | $\vec\ell$ | $\vec m$ | $(\mathrm{cond}\,\vec k, \mathrm{cond}\,\vec\ell, \mathrm{cond}\,\vec m)$ | distinct conductors |
|---|---|---|---|:---:|
| (0, 0, 1) | (0, 0, 1) | (0, 0, 2) | (4, 4, 2) | 2 |
| (0, 0, 1) | (0, 0, 2) | (0, 0, 1) | (4, 2, 4) | 2 |
| (0, 0, 1) | (0, 1, 0) | (0, 1, 3) | (4, 2, 4) | 2 |
| (0, 0, 1) | (0, 1, 1) | (0, 1, 2) | (4, 4, 2) | 2 |
| (0, 0, 1) | (0, 1, 2) | (0, 1, 1) | (4, 2, 4) | 2 |

### Z/3 x Z/9 (|G| = 27)

| $\vec k$ | $\vec\ell$ | $\vec m$ | $(\mathrm{cond}\,\vec k, \mathrm{cond}\,\vec\ell, \mathrm{cond}\,\vec m)$ | distinct conductors |
|---|---|---|---|:---:|
| (0, 1) | (0, 2) | (0, 6) | (9, 9, 3) | 2 |
| (0, 1) | (0, 3) | (0, 5) | (9, 3, 9) | 2 |
| (0, 1) | (0, 5) | (0, 3) | (9, 9, 3) | 2 |
| (0, 1) | (0, 6) | (0, 2) | (9, 3, 9) | 2 |
| (0, 1) | (1, 0) | (2, 8) | (9, 3, 9) | 2 |

## What this verifies

For each candidate $G$, the certificate (Theorem 3.5 of the manuscript)
holds in the form stated in Appendix B:

1. **Fisher block-diagonality at $p_*$.** $g_{p_*}$ is diagonal in the
   character basis of $\widehat{G}$ by character orthogonality (the
   finite geometric-sum argument applies factor-by-factor in the
   elementary-divisor decomposition of $G$). No off-diagonal entry is
   nonzero. The block-diagonal structure across conductor packets is
   immediate.

2. **Cross-packet AC cubic coupling at $p_*$.** Per the table above, on
   every non-degenerate candidate $G$ the cross-packet triple count is
   strictly positive: the cubic carries inter-packet coupling that the
   quadratic class cannot represent.

**Degenerate cases.** A group $G$ has *no* same-packet triples (and
$\rho_\times \equiv 1$ on every update) when no triple $(k, \ell, m)$
with conductors all equal sums to the identity in $G$. The diagnostic
(Definition 6.1) is structurally constant on such groups -- a property
of the dual lattice, not of any model. The cyclic case $n = 8$ (Section
6.5) and the elementary-abelian case $(\mathbb{Z}/p)^k$ (every nontrivial
character has order $p$, single packet) are the canonical examples.

**Same-packet vs cross-packet on non-cyclic vs cyclic of equal order.**
The counts differ between $G$ and its cyclic counterpart at the same
order. For $|G| = 8$: $\mathbb{Z}/8$ has 0 same-packet, 42 cross-packet;
$\mathbb{Z}/2 \times \mathbb{Z}/4$ has 6 same-packet, 36 cross-packet.
The certificate is unaffected (both halves hold), but the diagnostic's
structural ceiling differs.
