---
title: "Grokking Group Multiplication with Cosets"
authors: "Dashiell Stander, Qinan Yu, Honglu Fan, Stella Biderman"
year: 2023
venue: "arXiv:2312.06581 (ICML 2024)"
url: "https://arxiv.org/html/2312.06581v2"
type: paper
---

# Grokking Group Multiplication with Cosets

**Subtitle:** None
**Authors:** Dashiell Stander, Qinan Yu, Honglu Fan, Stella Biderman
**Year:** 2023 (v1 submitted 11 Dec 2023; v2 revised 17 Jun 2024)
**Venue:** arXiv:2312.06581 [cs.LG]; published at ICML 2024
**URLs:**
- https://arxiv.org/html/2312.06581v2
- https://arxiv.org/abs/2312.06581
- https://github.com/dashstander/sn-grok

## Bibliographic citation
Stander, D., Yu, Q., Fan, H., & Biderman, S. (2024). *Grokking Group Multiplication with Cosets*. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024). arXiv:2312.06581.

## Abstract
The complex and unpredictable nature of deep neural networks prevents their safe use in many high-stakes applications. There have been many techniques developed to interpret deep neural networks, but all have substantial limitations. Algorithmic tasks have proven to be a fruitful test ground for interpreting a neural network end-to-end. Building on previous work, we completely reverse engineer fully connected one-hidden layer networks that have "grokked" the arithmetic of the permutation groups $S_5$ and $S_6$. The models discover the true subgroup structure of the full group and converge on neural circuits that decompose the group arithmetic using the permutation group's subgroups. We relate how we reverse engineered the model's mechanisms and confirmed our theory was a faithful description of the circuit's functionality. We also draw attention to current challenges in conducting interpretability research by comparing our work to Chughtai et al. [4] which alleges to find a different algorithm for this same problem.
