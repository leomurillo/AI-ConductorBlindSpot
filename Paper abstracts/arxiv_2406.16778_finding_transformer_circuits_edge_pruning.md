---
title: "Finding Transformer Circuits with Edge Pruning"
authors: "Adithya Bhaskar, Alexander Wettig, Dan Friedman, Danqi Chen"
year: 2024
venue: "arXiv:2406.16778"
url: "https://arxiv.org/html/2406.16778v1"
type: paper
---

# Finding Transformer Circuits with Edge Pruning

**Subtitle:** None
**Authors:** Adithya Bhaskar, Alexander Wettig, Dan Friedman, Danqi Chen
**Year:** 2024
**Venue:** arXiv:2406.16778v1 [cs.LG]
**URLs:**
- https://arxiv.org/html/2406.16778v1
- https://arxiv.org/abs/2406.16778
- https://github.com/princeton-nlp/Edge-Pruning (code and data)

## Bibliographic citation
Bhaskar, A., Wettig, A., Friedman, D., & Chen, D. (2024). *Finding Transformer Circuits with Edge Pruning*. arXiv preprint arXiv:2406.16778.

## Abstract
The path to interpreting a language model often proceeds via analysis of circuits—sparse computational subgraphs of the model that capture specific aspects of its behavior. Recent work has automated the task of discovering circuits. Yet, these methods have practical limitations, as they rely either on inefficient search algorithms or inaccurate approximations. In this paper, we frame automated circuit discovery as an optimization problem and propose Edge Pruning as an effective and scalable solution. Edge Pruning leverages gradient-based pruning techniques, but instead of removing neurons or components, it prunes the edges between components. Our method finds circuits in GPT-2 that use less than half the number of edges compared to circuits found by previous methods while being equally faithful to the full model predictions on standard circuit-finding tasks. Edge Pruning is efficient even with as many as 100K examples, outperforming previous methods in speed and producing substantially better circuits. It also perfectly recovers the ground-truth circuits in two models compiled with Tracr. Thanks to its efficiency, we scale Edge Pruning to CodeLlama-13B, a model over 100x the scale that prior methods operate on.
