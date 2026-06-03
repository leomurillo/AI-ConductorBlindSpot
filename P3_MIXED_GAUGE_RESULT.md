# P3 Result: T5/T6 mixed-gauge 4th §8.6 arm

*Result note, 2026-05-24.* Closes P3 of T0_T6_HARVEST.md. Adds **Form
D — harmonic-dual gauge correction** to the §8.6 experimental matrix
as the first observer-change intervention (vs the additive cubic
corrections of forms A/B/C).

## The intervention (Form D, T5-derived)

**Motivation.** All three §8.6 forms (A Newton, B Neumann, C
within-packet) act in the RMS gauge — AdamW's natural observer — by
adding a cross-packet correction to the batch-mean logit gradient.
T6 Theorem 1.1 proves the RMS gauge is a closed L²-normalization
geometry; T5 Theorem 4.2 proves that the **odd cumulant tower** is
witnessed by the gap between arithmetic and harmonic means
(log(A·H/G²)). Forms A/B/C target ODD-cumulant content (cubic) but
INJECT in the RMS observer — a gauge mismatch, the peer reading.

**Form D.** For per-example head logit gradients $u_{\text{per}}$ of
shape $(B, n)$:

$$g_R[y] = \frac{1}{B}\sum_i u_{\text{per}}[i, y], \qquad H[y] = \frac{B}{\sum_i 1/(|u_{\text{per}}[i, y]| + \epsilon)}$$

$$g_H[y] = \text{sign}(g_R[y]) \cdot H[y], \qquad w[y] = g_H[y] - g_R[y]$$

The correction $w$ (centered) is added at strength $\alpha \cdot
\|g_R\|/\|w\|$ to the dlogits, matching the §8.6 protocol. By T5
Theorem 4.2, $w$ is a per-class scalar witness of the **odd-cumulant
content of the per-example gradient distribution** that the RMS-gauge
preconditioner $v_t = \text{EMA}(g^2)$ cannot absorb.

This is the first §8.6 arm motivated by *observer change* rather than
*additive cubic correction*.

## Pre-registered branches

**Branch A (T5 + T6 + T0 mixed-gauge confirmed):** interaction > 0,
Δ_T1 > 0 with Δ_T2s ≈ 0. Reads as: the harmonic observer
recovers ring-aligned odd-cumulant signal the four §8.6 RMS-gauge
forms miss.

**Branch B (universal head-side null):** interaction ≤ 0 within noise.
Reads as: §8.6's null is *observer-independent*; NO head-side
intervention — additive cubic OR observer-change harmonic —
preferentially helps the ring task. Consistent with P1's analytical
finding that the head-only β-flow has ρ_×(t) = const and the 10%+
depth deviation is the only meaningful gap.

## Headline result

*(Filled in by `cbs_p3_mixed_gauge_experiment.py` on completion)*

See `empirical/reports/cbs_p3_mixed_gauge_summary.md` for the actual
numerical result and which branch it falls into.

## What this resolves for the §8.6 story

Whichever branch lands:

- **Branch A** would establish that the operational intervention
  space at the head is **not** structurally closed under observer
  change — only under additive corrections. The T0 gauge corridor
  becomes the natural design space for cubic-aware optimizers, with
  T5's (M_r, M_{−r}) parity pairs as the structural primitive.

- **Branch B** would establish the operational intervention space at
  the head IS closed under both moves: any head-side intervention is
  null on this substrate. The depth-feature contribution (P1's
  12.5% asymmetry on n = 12) becomes the only meaningful remaining
  intervention surface, and the program shifts decisively to
  EGD/PGD-style within-mode methods (§7.8) and parameter-side
  K-FAC variants (§8.6 follow-up list).

## Files

- Script: [`empirical/cbs_p3_mixed_gauge_experiment.py`](empirical/cbs_p3_mixed_gauge_experiment.py)
- Per-seed JSONs: `empirical/reports/cbs_p3_form_d_n30_seed*.json`
- Summary: [`empirical/reports/cbs_p3_mixed_gauge_summary.md`](empirical/reports/cbs_p3_mixed_gauge_summary.md)
- Harvest doc: [`T0_T6_HARVEST.md`](T0_T6_HARVEST.md)
- Precursors:
  [`P1_BETA_FLOW_RESULT.md`](P1_BETA_FLOW_RESULT.md),
  [`P2_HELMERT_PULLBACK_RESULT.md`](P2_HELMERT_PULLBACK_RESULT.md)
