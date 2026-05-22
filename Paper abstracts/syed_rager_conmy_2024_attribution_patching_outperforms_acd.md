---
title: "Attribution Patching Outperforms Automated Circuit Discovery"
authors: "Aaquib Syed, Can Rager, Arthur Conmy"
year: 2024
venue: "Proceedings of the 7th BlackboxNLP Workshop (EMNLP 2024); also NeurIPS 2023 ATTRIB Workshop; arXiv:2310.10348"
url: "https://aclanthology.org/2024.blackboxnlp-1.25/"
type: paper
---

# Attribution Patching Outperforms Automated Circuit Discovery

**Subtitle:** None
**Authors:** Aaquib Syed, Can Rager, Arthur Conmy
**Year:** 2024 (ACL Anthology / BlackboxNLP version); arXiv preprint first posted 16 October 2023, revised 20 November 2023
**Venue:** Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, co-located with EMNLP 2024 (Miami, Florida, November 2024), pages 407–416. Editors: Yonatan Belinkov, Najoung Kim, Jaap Jumelet, Hosein Mohebbi, Aaron Mueller, Hanjie Chen. An earlier version appeared at the NeurIPS 2023 ATTRIB Workshop.
**URLs:**
- https://aclanthology.org/2024.blackboxnlp-1.25/
- https://aclanthology.org/2024.blackboxnlp-1.25.pdf
- https://openreview.net/pdf?id=tiLbFR4bJW
- https://arxiv.org/abs/2310.10348

## Bibliographic citation
Syed, A., Rager, C., & Conmy, A. (2024). Attribution Patching Outperforms Automated Circuit Discovery. In Y. Belinkov, N. Kim, J. Jumelet, H. Mohebbi, A. Mueller, & H. Chen (Eds.), *Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP* (pp. 407–416). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.blackboxnlp-1.25 (arXiv:2310.10348)

BibTeX:
```
@inproceedings{syed-etal-2024-attribution,
    title     = "Attribution Patching Outperforms Automated Circuit Discovery",
    author    = "Syed, Aaquib and Rager, Can and Conmy, Arthur",
    editor    = "Belinkov, Yonatan and Kim, Najoung and Jumelet, Jaap and Mohebbi, Hosein and Mueller, Aaron and Chen, Hanjie",
    booktitle = "Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP",
    year      = "2024",
    pages     = "407--416",
    doi       = "10.18653/v1/2024.blackboxnlp-1.25",
    url       = "https://aclanthology.org/2024.blackboxnlp-1.25/"
}
```

## Abstract
Automated interpretability research has recently attracted attention as a potential research direction that could scale explanations of neural network behavior to large models. Existing automated circuit discovery work applies activation patching to identify subnetworks responsible for solving specific tasks (circuits). In this work, we show that a simple method based on attribution patching outperforms all existing methods while requiring just two forward passes and a backward pass. We apply a linear approximation to activation patching to estimate the importance of each edge in the computational subgraph. Using this approximation, we prune the least important edges of the network. We survey the performance and limitations of this method, finding that averaged over all tasks our method has greater AUC from circuit recovery than other methods.
