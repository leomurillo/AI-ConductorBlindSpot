---
title: "Progress Measures for Grokking via Mechanistic Interpretability"
authors: "Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt"
year: 2023
venue: "arXiv:2301.05217 [cs.LG]; ICLR 2023"
url: "https://arxiv.org/abs/2301.05217"
type: paper
---

# Progress Measures for Grokking via Mechanistic Interpretability

**Subtitle:** None
**Authors:** Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt
**Year:** 2023
**Venue:** arXiv:2301.05217 [cs.LG]; published at ICLR 2023
**URLs:**
- https://arxiv.org/abs/2301.05217
- https://neelnanda.io/grokking
- https://www.neelnanda.io/grokking-paper
- https://arxiv.org/pdf/2301.05217.pdf

## Bibliographic citation
Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). *Progress measures for grokking via mechanistic interpretability*. arXiv:2301.05217 [cs.LG]. https://doi.org/10.48550/arXiv.2301.05217 (Submitted January 12, 2023; last revised October 19, 2023.)

## Abstract
Neural networks often exhibit emergent behavior, where qualitatively new capabilities arise from scaling up the amount of parameters, training data, or training steps. One approach to understanding emergence is to find continuous progress measures that underlie the seemingly discontinuous qualitative changes. We argue that progress measures can be found via mechanistic interpretability: reverse-engineering learned behaviors into their individual components. As a case study, we investigate the recently-discovered phenomenon of "grokking" exhibited by small transformers trained on modular addition tasks. We fully reverse engineer the algorithm learned by these networks, which uses discrete Fourier transforms and trigonometric identities to convert addition to rotation about a circle. We confirm the algorithm by analyzing the activations and weights and by performing ablations in Fourier space. Based on this understanding, we define progress measures that allow us to study the dynamics of training and split training into three continuous phases: memorization, circuit formation, and cleanup. Our results show that grokking, rather than being a sudden shift, arises from the gradual amplification of structured mechanisms encoded in the weights, followed by the later removal of memorizing components.
