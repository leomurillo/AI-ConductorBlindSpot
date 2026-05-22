---
title: "Attribution Patching: Activation Patching At Industrial Scale"
authors: "Neel Nanda"
year: 2023
venue: "neelnanda.io (personal blog)"
url: "https://www.neelnanda.io/mechanistic-interpretability/attribution-patching"
type: blog
---

# Attribution Patching: Activation Patching At Industrial Scale

**Subtitle:** None
**Authors:** Neel Nanda
**Year:** 2023 (post dated February 4; work credited as completed during the author's time at Anthropic, with Chris Olah, Catherine Olsson, Nelson Elhage, and Tristan Hume)
**Venue:** neelnanda.io — Mechanistic Interpretability section of Neel Nanda's personal blog
**URLs:**
- https://www.neelnanda.io/mechanistic-interpretability/attribution-patching
- Companion notebook: https://neelnanda.io/attribution-patching-demo
- Related: https://neelnanda.io/exploratory-analysis-demo

## Bibliographic citation
Nanda, N. (2023). *Attribution Patching: Activation Patching At Industrial Scale*. neelnanda.io. https://www.neelnanda.io/mechanistic-interpretability/attribution-patching

## Abstract
**Summary (no formal abstract — TL;DR captured verbatim from the post):**

- Activation patching is an existing technique for identifying which model activations are most important for determining model behaviour between two similar prompts that differ in a key detail.
- Attribution patching uses gradients to create a linear (first-order Taylor) approximation to activation patching. It is far faster because activation patching requires a separate forward pass per activation patched, whereas every attribution patch can be computed simultaneously in two forward passes and one backward pass.
- In practice the approximation works decently well for "small" activations such as individual head outputs, but poorly for "large" activations such as residual streams.
- The technique is presented as an exploratory scaling tool for mechanistic interpretability, making activation patching tractable on large models, rather than a strict replacement for activation patching.

The post credits prior work and collaboration with Chris Olah, Catherine Olsson, Nelson Elhage, and Tristan Hume at Anthropic, and is aimed at mechanistic-interpretability researchers already familiar with activation patching and causal tracing.
