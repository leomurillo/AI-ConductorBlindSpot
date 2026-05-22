---
title: "A Systematic Empirical Study of Grokking: Depth, Architecture, Activation, and Regularization"
authors: "Shalima Binta Manir, Anamika Paul Rupa"
year: 2026
venue: "arXiv:2603.25009"
url: "https://arxiv.org/html/2603.25009v1"
type: paper
---

# A Systematic Empirical Study of Grokking: Depth, Architecture, Activation, and Regularization

**Subtitle:** None
**Authors:** Shalima Binta Manir (University of Maryland, Baltimore County), Anamika Paul Rupa (Howard University)
**Year:** 2026
**Venue:** arXiv:2603.25009v1 [cs.LG]
**URLs:**
- https://arxiv.org/html/2603.25009v1
- https://arxiv.org/pdf/2603.25009
- https://arxiv.org/abs/2603.25009

## Bibliographic citation
Manir, S. B., & Rupa, A. P. (2026). *A Systematic Empirical Study of Grokking: Depth, Architecture, Activation, and Regularization*. arXiv preprint arXiv:2603.25009.

## Abstract
Grokking the delayed transition from memorization to generalization in neural networks remains poorly understood, in part because prior empirical studies confound the roles of architecture, optimization, and regularization. We present a controlled study that systematically disentangles these factors on modular addition (mod 97), with matched and carefully tuned training regimes across models. Our central finding is that grokking dynamics are not primarily determined by architecture, but by interactions between optimization stability and regularization. Specifically, we show: (1) depth has a non-monotonic effect, with depth-4 MLPs consistently failing to grok while depth-8 residual networks recover generalization, demonstrating that depth requires architectural stabilization; (2) the apparent gap between Transformers and MLPs largely disappears under matched hyperparameters, indicating that previously reported differences are largely due to optimizer and regularization confounds; (3) activation function effects are regime-dependent, with GELU up to 4.3x faster than ReLU only when regularization permits memorization; and (4) weight decay is the dominant control parameter, exhibiting a narrow "Goldilocks" regime in which grokking occurs, while too little or too much prevents generalization.
