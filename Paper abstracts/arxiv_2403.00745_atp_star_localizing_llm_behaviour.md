---
title: "AtP*: An efficient and scalable method for localizing LLM behaviour to components"
authors: "János Kramár, Tom Lieberum, Rohin Shah, Neel Nanda"
year: 2024
venue: "arXiv:2403.00745"
url: "https://arxiv.org/html/2403.00745v1"
type: paper
---

# AtP*: An efficient and scalable method for localizing LLM behaviour to components

**Subtitle:** None
**Authors:** János Kramár, Tom Lieberum, Rohin Shah, Neel Nanda
**Year:** 2024
**Venue:** arXiv:2403.00745v1 [cs.LG]
**URLs:**
- https://arxiv.org/html/2403.00745v1
- https://arxiv.org/abs/2403.00745

## Bibliographic citation
Kramár, J., Lieberum, T., Shah, R., & Nanda, N. (2024). *AtP*: An efficient and scalable method for localizing LLM behaviour to components*. arXiv preprint arXiv:2403.00745.

## Abstract
Activation Patching is a method of directly computing causal attributions of behavior to model components. However, applying it exhaustively requires a sweep with cost scaling linearly in the number of model components, which can be prohibitively expensive for SoTA Large Language Models (LLMs). We investigate Attribution Patching (AtP) (Nanda, 2022), a fast gradient-based approximation to Activation Patching and find two classes of failure modes of AtP which lead to significant false negatives. We propose a variant of AtP called AtP*, with two changes to address these failure modes while retaining scalability. We present the first systematic study of AtP and alternative methods for faster activation patching and show that AtP significantly outperforms all other investigated methods, with AtP* providing further significant improvement. Finally, we provide a method to bound the probability of remaining false negatives of AtP* estimates.
