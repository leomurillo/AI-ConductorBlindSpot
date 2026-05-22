---
title: "The Geometric Inductive Bias of Grokking: Bypassing Phase Transitions via Architectural Topology"
authors: "Alper Yıldırım"
year: 2026
venue: "arXiv:2603.05228 [cs.LG]"
url: "https://arxiv.org/abs/2603.05228"
type: paper
---

# The Geometric Inductive Bias of Grokking

**Subtitle:** Bypassing Phase Transitions via Architectural Topology
**Authors:** Alper Yıldırım (Independent Researcher)
**Year:** 2026
**Venue:** arXiv:2603.05228 [cs.LG], version v3
**URLs:**
- https://arxiv.org/abs/2603.05228
- https://arxiv.org/html/2603.05228v3
- https://arxiv.org/html/2603.05228v1
- https://arxiv.org/pdf/2603.05228

## Bibliographic citation
Yıldırım, A. (2026). *The geometric inductive bias of grokking: Bypassing phase transitions via architectural topology*. arXiv:2603.05228 [cs.LG] (v3, May 4, 2026; v1 submitted March 5, 2026). https://doi.org/10.48550/arXiv.2603.05228

## Abstract
Prior work on grokking—delayed generalization after memorization—has sought to shorten this delay through data augmentation or optimizer design. We take an architectural approach: rather than analyzing trained networks post-hoc, we modify architectural topology *before* training to test whether specific degrees of freedom prolong the memorization phase. We identify two independent factors in standard Transformers—unbounded residual magnitude and data-dependent attention routing—and introduce targeted interventions for each. A fully bounded spherical topology, enforcing $L_{2}$ normalization throughout the residual stream, eliminates the memorization phase entirely on modular addition and multiplication over $\mathbb{Z}_{p}$: training and test accuracy rise together from initialization. Independently, replacing learned attention with uniform aggregation achieves the same effect, consistent with theoretical results showing that commutative operations require only a bag-of-tokens representation. However, the same spherical constraint *fails completely* on non-commutative $S_{5}$ permutation composition, where successful solutions rely on discrete coset structures rather than continuous Fourier features. This contrast demonstrates that bypassing the generalization delay is possible—but strictly depends on alignment between architectural priors and task symmetry. Since even the simplest non-commutative task resists the same constraint that eliminates delay on cyclic arithmetic, these results suggest that no single geometric prior can universally accelerate learning across the heterogeneous structures present in general-purpose domains.
