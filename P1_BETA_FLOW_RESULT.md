# P1 Result: β-flow closure of CBS Conjecture 5.8

*Result note, 2026-05-24.* Closes the analytical half of CBS
Conjecture 5.8 via the T0 β-flow and quantifies the depth-driven
contribution on the existing §6.5 substrate.

## Theorem (head-only β-flow time-invariance)

Consider a categorical head on $R = \mathbb{Z}/n\mathbb{Z}$ trained
by cross-entropy with per-example target $y^* = (a+b) \bmod n$. The
per-example softmax + cross-entropy trajectory is exactly the T0
β-flow (T0 Theorem 2.4) with constant fitness $y = e_{y^*}$:

$$P_\beta^{(y^*)}(y^*) = \frac{e^\beta}{e^\beta + n - 1},
  \qquad
  P_\beta^{(y^*)}(c) = \frac{1}{e^\beta + n - 1} \text{ for } c \neq y^*.$$

The per-example centered score $u^{(a,b)}(\beta) = e_{y^*} -
P_\beta^{(y^*)}$ has additive-character Fourier coefficients

$$\boxed{\;\hat u^{(a,b)}(\beta)_k = e^{-2\pi i k y^*/n} \cdot
   \frac{n}{e^\beta + n - 1}, \qquad k \in \{1, \ldots, n-1\}.\;}$$

In particular $|\hat u^{(a,b)}(\beta)_k|^2$ is **independent of k**:
the per-example update has equal Plancherel mass on every nonzero
frequency. The batch-mean spectrum factorizes:

$$\bar u(\beta)_k = \frac{n}{e^\beta + n - 1} \cdot
   E_B\!\left[ e^{-2\pi i k y^*/n} \right].$$

**Corollary.** Both ρ_×(t) (CBS Definition 6.1) and σ_PI(t) (the
T0/T5 scalar surrogate) are **β-invariant** under the head-only
constant-fitness β-flow. The head's intrinsic dynamic produces no
trajectory variation in either diagnostic.

## Numerical verification

Script `empirical/cbs_t0_beta_flow_prediction.py` confirms the
prediction at machine ε across $n \in \{6, 12, 30\}$ and modes
D1 / D2-weak / D2-strong: per-seed standard deviation of σ_PI across
β ∈ {0, 0.5, 1, 2, 4, 8, 16} is $\le 3 \times 10^{-15}$ (12 seeds,
batch size 256).

## Empirical comparison on the CBS §6.5 substrate

Script `empirical/cbs_p1_empirical_vs_analytical.py`, against the
existing demo curves:

### n = 12 (the discriminative ring)

| mode | empirical mean ρ_× | analytical head-only ρ_× | depth deviation |
|---|---:|---:|---:|
| D1 (ring) | 0.847 | 0.951 ± 0.063 | **−0.104** |
| D2-weak | 0.995 | 0.981 ± 0.031 | +0.014 |
| D2-strong | 0.983 | 0.962 ± 0.046 | +0.021 |

**Depth-driven asymmetry D1 − D2-strong = −0.125** (12.5%).

### Cross-ring depth asymmetry

| n | depth asymmetry D1 − D2-strong | §6.5 regime |
|---|---:|---|
| 8  | +0.000 | structural-degeneracy (ρ_× ≡ 1) |
| 12 | **−0.125** | **pre-grokking representation-learning** |
| 18 | +0.025 | insufficient budget |
| 30 | +0.010 | post-grokking saturation |

The 12.5% asymmetry exists **only** in the pre-grokking discriminative
phase. The β-flow analytical baseline turns CBS §6.5's "three diagnostic
regimes" qualitative observation into a quantitative gap measurable
against a closed-form analytical prediction.

## What this closes for the manuscript

**(i) Analytical half of Conjecture 5.8.** The head's intrinsic
β-flow predicts $\rho_\times(t) = \text{const}$, $\sigma_{PI}(t) =
\text{const}$. The empirically observed trajectory variation in §6.5
is therefore the depth-driven deviation from this constant, NOT a
head-geometric effect. Conjecture 5.8's "rate asymmetry" is the
asymmetry of this deviation between ring and structureless tasks,
now expressed as a scalar gap between two computable curves rather
than an open dynamical question.

**(ii) The §8.6 Branch B null is consistent with the certificate.**
The three Branch B forms (cubic injection at the head) target the
head-geometric content that the analytical-empirical comparison
shows contributes ≤2% to the cross-packet activation gap. The
~10%+ depth contribution that drives the asymmetry is upstream of
the head, where head-side cubic injection cannot reach.

**(iii) Right intervention surface.** The operational target is the
upstream MLP's feature dynamics, not the head's curvature. Three
concrete candidates:
- **(P3)** Mixed-gauge optimizer (T5 + T6 + T0 fusion):
  alternating RMS-flow (β = 2) and harmonic-flow (β = −1) steps
  change the observer per step and probe the gauge structure.
- **(P2)** Helmert-coordinate readout of the MLP's pre-head feature
  evolution, via the T_{c,k} transfer matrix from T4. The
  Helmert basis exposes the prime-rung radical content that the
  Fourier basis cannot.
- **(EGD / PGD from §7.8)** Uniform within-mode rescaling, which
  acts on the depth-side spectrum and is known to compress
  grokking delay on the same testbed.

**(iv) Diagnostic boundaries re-derived.** The cross-ring sweep
reproduces CBS §6.5's three regime boundaries (degeneracy at n = 8;
insufficient-budget at n = 18; saturation at n = 30) with a
quantitative analytical baseline rather than only a qualitative
comparison across arms.

## Suggested manuscript update

Suggested CBS §5.4 addition immediately after Conjecture 5.8:

> **Theorem 5.10 (head-only β-flow time-invariance).** Let
> $P_\beta(c) = e^{\beta \delta_{c,y^*}} / Z_\beta$ be the
> constant-fitness β-flow at the head with per-example target $y^*$.
> For $k \neq 0$, the per-example centered score has
> $|\hat u^{(a,b)}(\beta)_k|^2 = (n/(e^\beta + n - 1))^2$,
> independent of $k$. Consequently the batch-mean Fourier spectrum
> $\bar u(\beta)_k$ factorizes into a β-dependent scalar and a
> batch-determined shape, and both $\rho_\times(\bar u(\beta))$ and
> $\sigma_{PI}(\bar u(\beta))$ are independent of $\beta$.
>
> **Remark 5.11 (depth-driven deviation as Conjecture 5.8's signal).**
> Empirically, on the §6.5 substrate at n = 12, the AdamW + MLP
> trajectory's mean $\rho_\times$ deviates from the analytical
> head-only β-flow value by −10.4% on the ring task vs +2.1% on the
> structureless control (asymmetry −12.5%). On the §6.5 ladder, this
> asymmetry is measurable only in the pre-grokking representation-
> learning phase (n = 12), and collapses to noise in the structural-
> degeneracy (n = 8), insufficient-budget (n = 18), and post-grokking
> saturation (n = 30) regimes. Conjecture 5.8's rate asymmetry is
> the depth-driven deviation from Theorem 5.10's analytical
> baseline, and the operational target for any intervention that
> aims to compress it must act on the upstream-feature dynamics that
> drive the deviation, not on the head's curvature (which Theorem
> 5.10 shows to be time-invariant).

## Files

- [`empirical/cbs_t0_beta_flow_prediction.py`](empirical/cbs_t0_beta_flow_prediction.py)
  — derivation + verification.
- [`empirical/cbs_p1_empirical_vs_analytical.py`](empirical/cbs_p1_empirical_vs_analytical.py)
  — empirical comparison + cross-ring sweep.
- [`empirical/reports/cbs_t0_beta_flow_prediction.md`](empirical/reports/cbs_t0_beta_flow_prediction.md)
  — verification report.
- [`empirical/reports/cbs_p1_empirical_vs_analytical.md`](empirical/reports/cbs_p1_empirical_vs_analytical.md)
  — empirical comparison report.
- [`T0_T6_HARVEST.md`](T0_T6_HARVEST.md) §4 — closure summary in the
  harvest doc.
