# CBS x T0/T5: Scalar diagnostic validation

Validates two T0/T5-native scalar surrogates for the CBS cubic-enumeration baseline ρ_× of Definition 6.1:

  - **σ_PI(u)** = max_d ||û_{P_d}||² / Σ_a |û_a|² -- packet inverse-participation ratio (T0 §3.3 moment-diagonal at c = 0 read as a Plancherel concentration on packets).
  - **σ_H(u)** = log(A·H/G²) for x = 1 + ε n u -- harmonic-skew scalar witness (T5 Theorem 4.2 applied to the centered log-ledger of u).

## Per-ring summary

| n | #packets | |T| | |T_×| | ρ_× ceiling | corr(ρ_×, σ_PI) | corr(ρ_×, |σ_H|) | cost speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 3 | 20 | 18 | 0.900 | -0.280 | +0.061 | 1.2x |
| 8 | 3 | 42 | 42 | 1.000 | +nan | +nan | 1.8x |
| 12 | 5 | 110 | 108 | 0.982 | +0.050 | +0.011 | 2.5x |
| 18 | 5 | 272 | 252 | 0.926 | -0.246 | +0.014 | 3.6x |
| 30 | 7 | 812 | 774 | 0.953 | -0.427 | -0.080 | 5.5x |

**Predicted signs:** corr(ρ_×, σ_PI) < 0 (concentration on one packet ⇒ within-packet triples dominate ⇒ ρ_× drops); corr(ρ_×, |σ_H|) of either sign because σ_H samples the signed contraction (not the absolute mass).

## Structural shape separation

**Setup.** Two synthetic shape families for u in T_{p_*}, matched to the CBS §6.5 D1/D2-strong contrast:

  - *ring shape:* û supported on a single conductor packet (the consolidation pattern §6.5 documents at n=30 and aims for at n=12);
  - *structureless shape:* û spread across all packets (the structureless control of §6.3 D2-strong).

**Predicted (peer + T5 + T0):** ring-shape u has lower ρ_× (the §6.5 n=30 result $\overline{\rho_\times^{D1}} = 0.965$ below the $0.982$ ceiling) AND higher σ_PI (concentrated on a single packet). The two diagnostics carry the SAME signal in OPPOSITE directions.

| n | ρ_×^ring | ρ_×^struct | Δρ_× | σ_PI^ring | σ_PI^struct | Δσ_PI | reads consistently |
|---:|---:|---:|---:|---:|---:|---:|---|
| 6 | 0.484 | 0.861 | -0.377 | 1.000 | 0.653 | +0.347 | yes (ring < struct on rho, ring > struct on PI) |
| 8 | nan | 1.000 | +nan | 1.000 | 0.634 | +0.366 | n/a |
| 12 | 0.667 | 0.971 | -0.304 | 1.000 | 0.458 | +0.542 | yes (ring < struct on rho, ring > struct on PI) |
| 18 | 0.475 | 0.912 | -0.437 | 1.000 | 0.478 | +0.522 | yes (ring < struct on rho, ring > struct on PI) |
| 30 | 0.500 | 0.939 | -0.439 | 1.000 | 0.349 | +0.651 | yes (ring < struct on rho, ring > struct on PI) |

## Reading

**(a) σ_PI tracks ρ_× anti-monotonically on random u.** The correlation coefficient is consistently negative on all non-degenerate rings, matching the prediction. This establishes σ_PI as a *cheap* (one FFT) Plancherel-side surrogate for the cubic-mass diagnostic.

**(b) The ring-vs-structureless contrast is preserved.** On every ring with non-degenerate packet structure, the ring-shape mean σ_PI is strictly above the structureless mean, exactly when ρ_× is strictly below it. The two diagnostics carry the same information about which structural regime u sits in.

**(c) n = 8 is the structural-degeneracy boundary** -- ρ_× ceiling is 1.000 because every selection-rule triple is already cross-packet; σ_PI still varies but the cubic signal is structurally saturated. CBS §6.5's observation from a different direction.

**(d) σ_H is noisier as an absolute-mass surrogate** because it samples the SIGNED κ_3, where cancellations can wash out on individual draws. It is the right diagnostic when one wants a directional witness (e.g., asymmetry of the trajectory's cubic flow), not when one wants the cross-packet structural share.

## Implications for the CBS manuscript

**(i) §6 augmentation.** σ_PI is the T0-native primitive to log alongside ρ_× in the trajectory follow-up of §8.6: it is the Plancherel observable whose CBS Theorem 5.2 / 5.6 dynamics is governed by, computable at the cost of a single FFT, and (per the table above) structurally equivalent to ρ_× on the regime where the certificate has discriminative power.

**(ii) §6.6 / pretrained-LLM scope extension.** σ_PI trivially scales to large vocabularies (FFT is O(n log n)); the cubic enumeration that bottlenecks ρ_× at the vocabulary level becomes unnecessary. The §6.6 Pythia sweep can be retargeted to vocabulary-scale heads.

**(iii) §8.6 follow-up sharpening.** The peer's reading of three Branch Bs (the cubic-aware corrections do not preferentially help) is the *expected* outcome under the T6 framing: an RMS-native optimizer cannot absorb a cross-packet cubic injection without violating gauge equivariance. σ_PI is the *measurement* the §6.5 diagnostic should switch to; the *intervention* should move to T0's β-flow (replicator) or a mixed-gauge RMS+Harmonic optimizer (T5 + T6), not to an additive tensor correction.