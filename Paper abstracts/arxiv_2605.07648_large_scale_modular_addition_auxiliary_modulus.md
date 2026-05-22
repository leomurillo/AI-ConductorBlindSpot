---
title: "Learning Large-Scale Modular Addition with an Auxiliary Modulus"
authors: "Hanato Kikuchi, Ryosuke Masuya, Kazuhiko Kawamoto, Hiroshi Kera"
year: 2026
venue: "arXiv:2605.07648 [cs.LG]"
url: "https://arxiv.org/html/2605.07648v1"
type: paper
---

# Learning Large-Scale Modular Addition with an Auxiliary Modulus

**Subtitle:** None
**Authors:** Hanato Kikuchi, Ryosuke Masuya, Kazuhiko Kawamoto, Hiroshi Kera
**Year:** 2026
**Venue:** arXiv:2605.07648v1 [cs.LG], May 8, 2026
**URLs:**
- https://arxiv.org/html/2605.07648v1
- https://arxiv.org/pdf/2605.07648
- https://arxiv.org/abs/2605.07648

## Bibliographic citation
Kikuchi, H., Masuya, R., Kawamoto, K., & Kera, H. (2026). *Learning Large-Scale Modular Addition with an Auxiliary Modulus*. arXiv preprint arXiv:2605.07648v1 [cs.LG].

## Abstract
Learning parity functions, more general modular addition, is a challenging machine learning task due to its input sensitivity. A recent study substantially scaled modular addition learning in both the number of summands and the modulus. Its key idea is to increase zeros in training sequences, reducing the effective number of summands and thus controlling training difficulty; however, this induces covariate shift between training and test input distributions. This study theoretically and empirically analyzes this side effect and proposes a covariate-shift-free method for modular addition. Specifically, we introduce an auxiliary modulus Kq during training, which reduces wrap-around frequency and problem difficulty while preserving the same input distribution across training and testing. Experiments show strong scalability and sample efficiency: even for large input length N, large modulus q, and small datasets—where the sparse method fails to learn—our method achieves equal or better match accuracy and relaxed τ-accuracy. For example, at N=64 and q=974269, our method trained on 100K samples achieves 97.0% τ-accuracy at τ=0.05, while the sparse method achieves only 9.5% with the same data size and 93.9% even when extended to 1M samples.
