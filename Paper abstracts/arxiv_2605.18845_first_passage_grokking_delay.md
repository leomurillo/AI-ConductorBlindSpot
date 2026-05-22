---
title: "First-Passage Prediction of Grokking Delay: A Calibrated Law under AdamW with Causal Validation"
authors: "Truong Xuan Khanh, Truong Quynh Hoa, Luu Duc Trung, Phan Thanh Duc"
year: 2026
venue: "arXiv:2605.18845v1"
url: "https://arxiv.org/html/2605.18845v1"
type: paper
---

# First-Passage Prediction of Grokking Delay: A Calibrated Law under AdamW with Causal Validation

**Subtitle:** A Calibrated Law under AdamW with Causal Validation
**Authors:** Truong Xuan Khanh, Truong Quynh Hoa, Luu Duc Trung, Phan Thanh Duc
**Year:** 2026
**Venue:** arXiv preprint, arXiv:2605.18845v1
**URLs:**
- https://arxiv.org/html/2605.18845v1
- https://arxiv.org/pdf/2605.18845
- https://arxiv.org/abs/2605.18845

## Bibliographic citation
Truong, X. K., Truong, Q. H., Luu, D. T., & Phan, T. D. (2026). *First-Passage Prediction of Grokking Delay: A Calibrated Law under AdamW with Causal Validation* (arXiv:2605.18845v1) [Preprint]. arXiv. https://arxiv.org/abs/2605.18845

## Abstract
We give the first quantitative prediction of grokking delay under AdamW: a closed-form first-passage law with MAPE 17.7% on 26 held-out runs spanning a 41× delay range, with mechanism causally validated. The result fits a unified picture of delayed generalisation as a crossing problem in the joint (V_t, α_t) space of parameter norm and angular displacement: V_t = ‖θ_t‖² contracts exponentially toward an architecture-dependent threshold V*, while α_t must reach an angular threshold α*, and grokking occurs only when both are reached. The closed-form law,

T_grok − T_mem ≈ 1/(2κ_LL η λ) log(V_mem / V*),

is calibrated on a single cell by an architecture-level constant κ_LL that absorbs the AdamW correction to the clean-SGD rate 2ηλ; MAPE rises to 18.0% across architectures and 23.3% across full cross-task scope (46 runs, 43.5× range), with structured residual identifying V*/V_mem as the comparatively stable invariant within architecture (CV ≈ 14% on 1L).

Mechanistic foundation. The joint-crossing picture is supported by (i) a quantile-margin theorem: positive delay requires both norm separation V_mem > V_post and angular reachability of α* = arcsin(C/V^{1/2}_{T_mem}) with C := M_q Δ / G_eff; calibrating C on p = 89 predicts α* = 47.2° for p = 97 (observed 47.8°, error 1.3%); (ii) Block F causal interventions (0/6 grok vs. 3/3 baseline when V_t is frozen or λ is removed at T_mem), which trap α near 12° ≪ α*. Concurrent theoretical work (Boursier et al., 2025; Muşat et al., 2025) characterises the post-memorisation dynamics; our contribution is the corresponding prediction theory for AdamW with explicit observables, calibration, and causal tests.
