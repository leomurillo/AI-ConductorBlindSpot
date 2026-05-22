---
title: "Why Do You Grok? A Theoretical Analysis on Grokking Modular Addition"
authors: "Mohamad Amin Mohamadi, Zhiyuan Li, Lei Wu, Danica J. Sutherland"
year: 2024
venue: "Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR Volume 235, pp. 35934-35967"
url: "https://proceedings.mlr.press/v235/mohamadi24a.html"
type: paper
---

# Why Do You Grok? A Theoretical Analysis on Grokking Modular Addition

**Subtitle:** None
**Authors:** Mohamad Amin Mohamadi, Zhiyuan Li, Lei Wu, Danica J. Sutherland
**Year:** 2024
**Venue:** Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR Volume 235, pp. 35934-35967
**URLs:**
- https://proceedings.mlr.press/v235/mohamadi24a.html
- https://raw.githubusercontent.com/mlresearch/v235/main/assets/mohamadi24a/mohamadi24a.pdf

## Bibliographic citation
Mohamadi, M. A., Li, Z., Wu, L., & Sutherland, D. J. (2024). Why Do You Grok? A Theoretical Analysis on Grokking Modular Addition. In *Proceedings of the 41st International Conference on Machine Learning* (Vol. 235, pp. 35934-35967). PMLR.

## Abstract
We present a theoretical explanation of the "grokking" phenomenon (Power et al., 2022), where a model generalizes long after overfitting, for the originally-studied problem of modular addition. First, we show that early in gradient descent, so that the "kernel regime" approximately holds, no permutation-equivariant model can achieve small population error on modular addition unless it sees at least a constant fraction of all possible data points. Eventually, however, models escape the kernel regime. We show that one-hidden-layer quadratic networks that achieve zero training loss with bounded ℓ∞ norm generalize well with substantially fewer training points, and further show such networks exist and can be found by gradient descent with small ℓ∞ regularization. We further provide empirical evidence that these networks leave the kernel regime only after initially overfitting. Taken together, our results strongly support the case for grokking as a consequence of the transition from kernel-like behavior to limiting behavior of gradient descent on deep networks.
