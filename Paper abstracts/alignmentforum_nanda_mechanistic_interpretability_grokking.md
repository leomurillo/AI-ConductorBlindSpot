---
title: "A Mechanistic Interpretability Analysis of Grokking"
authors: "Neel Nanda; Tom Lieberum"
year: 2022
venue: "AI Alignment Forum (also LessWrong)"
url: "https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking"
type: post
---

# A Mechanistic Interpretability Analysis of Grokking

**Subtitle:** None
**Authors:** Neel Nanda, Tom Lieberum
**Year:** 2022 (posted 15 August 2022)
**Venue:** AI Alignment Forum (cross-posted to LessWrong); accompanying blog post on neelnanda.io ("Interlude: A Mechanistic Interpretability Analysis of Grokking", 18 Aug 2022)
**URLs:**
- https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking
- https://www.lesswrong.com/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking
- https://www.neelnanda.io/blog/interlude-a-mechanistic-interpretability-analysis-of-grokking
- Companion Colab notebook: https://bit.ly/neelgrokking
- Summary thread: https://twitter.com/NeelNanda5/status/1559060507524403200

## Bibliographic citation
Nanda, N., & Lieberum, T. (2022, August 15). *A mechanistic interpretability analysis of grokking* [Online post]. AI Alignment Forum. https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking

## Abstract
Summary (no formal abstract; the post opens with an informal TL;DR — paraphrased and partially quoted below):

The post reports on independent research conducted after Nanda left Anthropic, applying mechanistic-interpretability tools to investigate the grokking phenomenon. As Nanda writes in the companion blog post: "I used mechanistic interpretability tools to investigate what's up with the ML phenomena of grokking — where models trained on simple mathematical operations like addition mod 113 and given 30% of the data will initially memorise the data, but then if trained for a long time will abruptly generalise." The Alignment Forum write-up presents the technical findings: that the grokked one-layer transformer trained on modular addition implements a Fourier-basis / discrete-Fourier-transform plus trigonometric-identity algorithm, that this algorithm can be reverse-engineered from the trained weights, and that "progress measures" (excluded loss, restricted loss, Gini coefficient on the Fourier spectrum) reveal grokking is not a sudden phase transition but the smooth completion of a circuit whose generalisation only manifests externally once the memorising component is cleaned up. The work served as the basis for the later ICLR 2023 paper "Progress measures for grokking via mechanistic interpretability" (Nanda, Chan, Lieberum, Smith, Steinhardt, arXiv:2301.05217).
