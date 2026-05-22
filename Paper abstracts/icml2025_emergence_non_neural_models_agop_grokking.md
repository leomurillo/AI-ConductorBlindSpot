---
title: "Emergence in non-neural models: grokking modular arithmetic via average gradient outer product"
authors: "Neil Mallinar, Daniel Beaglehole, Libin Zhu, Adityanarayanan Radhakrishnan, Parthe Pandit, Mikhail Belkin"
year: 2025
venue: "ICML 2025 (Spotlight Poster #46553); arXiv:2407.20199"
url: "https://icml.cc/virtual/2025/poster/46553"
type: paper
---

# Emergence in non-neural models: grokking modular arithmetic via average gradient outer product

**Subtitle:** None
**Authors:** Neil Mallinar, Daniel Beaglehole, Libin Zhu, Adityanarayanan Radhakrishnan, Parthe Pandit, Mikhail Belkin
**Year:** 2025 (arXiv v1 29 Jul 2024; v3 9 Jul 2025; ICML 2025 spotlight poster)
**Venue:** ICML 2025, Spotlight Poster #46553; arXiv:2407.20199
**URLs:**
- https://icml.cc/virtual/2025/poster/46553
- https://arxiv.org/abs/2407.20199

## Bibliographic citation
Mallinar, N., Beaglehole, D., Zhu, L., Radhakrishnan, A., Pandit, P., & Belkin, M. (2025). *Emergence in non-neural models: grokking modular arithmetic via average gradient outer product.* In *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025).* arXiv:2407.20199.

## Abstract
Neural networks trained to solve modular arithmetic tasks exhibit grokking, a phenomenon where the test accuracy starts improving long after the model achieves 100% training accuracy in the training process. It is often taken as an example of "emergence", where model ability manifests sharply through a phase transition. In this work, we show that the phenomenon of grokking is not specific to neural networks nor to gradient descent-based optimization. Specifically, we show that this phenomenon occurs when learning modular arithmetic with Recursive Feature Machines (RFM), an iterative algorithm that uses the Average Gradient Outer Product (AGOP) to enable task-specific feature learning with general machine learning models. When used in conjunction with kernel machines, iterating RFM results in a fast transition from random, near zero, test accuracy to perfect test accuracy. This transition cannot be predicted from the training loss, which is identically zero, nor from the test loss, which remains constant in initial iterations. Instead, as we show, the transition is completely determined by feature learning: RFM gradually learns block-circulant features to solve modular arithmetic. Paralleling the results for RFM, we show that neural networks that solve modular arithmetic also learn block-circulant features. Furthermore, we present theoretical evidence that RFM uses such block-circulant features to implement the Fourier Multiplication Algorithm, which prior work posited as the generalizing solution neural networks learn on these tasks. Our results demonstrate that emergence can result purely from learning task-relevant features and is not specific to neural architectures nor gradient descent-based optimization methods. Furthermore, our work provides more evidence for AGOP as a key mechanism for feature learning in neural networks.
