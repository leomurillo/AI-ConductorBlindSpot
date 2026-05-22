---
title: "Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking"
authors: "Ali Saheb Pasand, Elvis Dohmatob"
year: 2025
venue: "ICLR 2026 (poster); arXiv:2510.04930"
url: "https://arxiv.org/abs/2510.04930"
type: paper
---

# Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking

**Subtitle:** None
**Authors:** Ali Saheb Pasand (McGill University & Mila Institute), Elvis Dohmatob (Concordia University & Mila Institute)
**Year:** 2025 (v1 submitted 6 Oct 2025; v2 14 May 2026; v3 18 May 2026)
**Venue:** ICLR 2026 poster (Saturday, April 25, 2026, 6:30 AM - 9:00 AM PDT, Pavilion 3, Poster P3-#2011); arXiv:2510.04930
**URLs:**
- https://arxiv.org/abs/2510.04930
- https://arxiv.org/html/2510.04930v2
- https://iclr.cc/virtual/2026/poster/10006751

## Bibliographic citation
Saheb Pasand, A., & Dohmatob, E. (2026). *Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking.* In *Proceedings of the International Conference on Learning Representations (ICLR 2026).* arXiv:2510.04930.

## Abstract
Grokking is the phenomenon whereby, unlike the training performance which peaks very early on during training, the test/generalization performance of a model stagnates over arbitrarily many epochs and then suddenly jumps to usually close to perfect levels. In practice, it is desirable to reduce the length of such plateaus, that is to make the learning process "grok" faster. In this work, we provide new insights into grokking. First, we show both empirically and theoretically that grokking can be induced by asymmetric speeds of (stochastic) gradient descent, along different principal (i.e singular directions) of the gradients. We then propose a simple modification that normalizes the gradients so that dynamics along all the principal directions evolves at exactly the same speed. Then, we establish that this modified method, which we call egalitarian gradient descent (EGD) and can be seen as a carefully modified form of natural gradient descent, groks much faster. In fact, in some cases the stagnation is completely removed. Finally, we empirically show that on classical arithmetic problems like modular addition and sparse parity problem which this stagnation has been widely observed and intensively studied, that our proposed method removes the plateaus.
