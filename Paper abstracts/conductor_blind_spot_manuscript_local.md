---
title: "The Conductor Blind Spot of the Quadratic Curvature Class on Ring-Structured Categorical Heads"
authors: "Leonardo Murillo Montero"
year: 2026
venue: "Local manuscript (preprint, ConductorBlindSpot project)"
url: "file:./ConductorBlindSpot.md"
type: local
---

# The Conductor Blind Spot of the Quadratic Curvature Class on Ring-Structured Categorical Heads

**Subtitle:** *From Curvature Preconditioning to Gradient-Based Mechanism Attribution*
**Authors:** Leonardo Murillo Montero (leonardo.murillo@gmail.com)
**Year:** 2026 (dated May 19, 2026)
**Venue:** Local manuscript — ConductorBlindSpot project (this repository). Target venue per internal log: arXiv preprint (stat.ML / cs.LG primary; math.PR secondary).
**Local path:** `ConductorBlindSpot/ConductorBlindSpot.md`

## Bibliographic citation

Murillo Montero, L. (2026). *The Conductor Blind Spot of the Quadratic Curvature Class on Ring-Structured Categorical Heads: From Curvature Preconditioning to Gradient-Based Mechanism Attribution.* Manuscript, ConductorBlindSpot project (May 19, 2026).

## Abstract

On a categorical model with cyclic ring index $R = \mathbb{Z}/n\mathbb{Z}$, two finite identities hold exactly at the maximum-entropy point $p_*$: the Fisher information form $g_{p_*}$ is diagonal in the additive-character basis, hence block-diagonal across the conductor-packet decomposition of the tangent space; and the Amari–Chentsov cubic form $T_{p_*}$ couples distinct conductor packets exactly under the integer selection rule $k + \ell + m \equiv 0 \pmod n$. The cross-packet cubic coupling is therefore outside the representational capacity of $g_{p_*}$ — the underlying object that the curvature surrogates of natural gradient, Gauss–Newton, K-FAC, Adam's empirical Fisher, and the neural-tangent-kernel Gram reduce to, pull back from, or approximate; the same identity lifts at $p_*$ to the head-side input of linear gradient-based edge-attribution scores in mechanistic circuit discovery (EAP, EAP-IG, E-ACT). Section 2.3 organizes the method-class lift in four tiers and attaches the strict claim to $g_{p_*}$ itself rather than uniformly to each named method. We verify the certificate computationally for $n \in \{6, 8, 12, 18, 30\}$ in exact integer arithmetic.

From the certificate we derive a retraining-free diagnostic $\rho_\times$ (Definition 6.1, an L1 spectral-mass ratio on the $p_*$-anchored selection-rule triples) and pre-specify a single experiment with a built-in negative control. A synthetic-MLP demo separates ring task from structureless control by $+0.15$ time-mean at $n = 12$ in the pre-grokking representation-learning phase (§6.5); a retraining-free Pythia checkpoint sweep on a calendar-months head identifies the diagnostic's off-trajectory boundary (§6.6). Three head-only instances of the pre-specified intervention at $\alpha = 0.1$ on $n = 30$ with an AdamW baseline — Newton-style cross-packet, Neumann-style cross-packet, and sign-reversed within-packet amplification — all return Branch B (§8.6; mean interactions $+0.003$, $-0.011$, $-0.004$). On this evidence the certificate's role is *measurement*, not *steering*; an $\alpha$-sweep across forms (B) and (C), a parameter-side K-FAC variant, and longer-budget runs on other rings remain open.

**Keywords:** second-order optimization, natural gradient, Fisher information, Amari–Chentsov tensor, categorical models, additive characters, conductor decomposition, ring-structured labels, preconditioning, information geometry, grokking.

**MSC 2020:** 53B12 (information geometry); 62B10 (statistical aspects of information theory); 68T07 (artificial neural networks and deep learning); 11L03 (elementary character sums); 65K10 (numerical optimization).
