---
title: "RelP: Faithful and Efficient Circuit Discovery via Relevance Patching"
authors: "Farnoush Rezaei Jafari, Oliver Eberle, Ashkan Khakzar, Neel Nanda"
year: 2025
venue: "arXiv:2508.21258"
url: "https://arxiv.org/html/2508.21258v1"
type: paper
---

# RelP: Faithful and Efficient Circuit Discovery via Relevance Patching

**Subtitle:** None
**Authors:** Farnoush Rezaei Jafari, Oliver Eberle, Ashkan Khakzar, Neel Nanda
**Year:** 2025
**Venue:** arXiv:2508.21258v1 [cs.LG]
**URLs:**
- https://arxiv.org/html/2508.21258v1
- https://arxiv.org/abs/2508.21258
- https://github.com/FarnoushRJ/RelP.git (code)

## Bibliographic citation
Rezaei Jafari, F., Eberle, O., Khakzar, A., & Nanda, N. (2025). *RelP: Faithful and Efficient Circuit Discovery via Relevance Patching*. arXiv preprint arXiv:2508.21258.

## Abstract
Activation patching is a standard method in mechanistic interpretability for localizing the components of a model responsible for specific behaviors, but it is computationally expensive to apply at scale. Attribution patching offers a faster, gradient-based approximation, yet suffers from noise and reduced reliability in deep, highly non-linear networks. In this work, we introduce Relevance Patching (RelP), which replaces the local gradients in attribution patching with propagation coefficients derived from Layer-wise Relevance Propagation (LRP). LRP propagates the network's output backward through the layers, redistributing relevance to lower-level components according to local propagation rules that ensure properties such as relevance conservation or improved signal-to-noise ratio. Like attribution patching, RelP requires only two forward passes and one backward pass, maintaining computational efficiency while improving faithfulness. We validate RelP across a range of models and tasks, showing that it more accurately approximates activation patching than standard attribution patching, particularly when analyzing residual stream and MLP outputs in the Indirect Object Identification (IOI) task.
