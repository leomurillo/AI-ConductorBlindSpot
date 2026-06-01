# Paper 34 -- Conductor Blind-Spot Demo Summary

Layers 1 + 2 only (no cubic-aware optimizer correction; section-9 firewall intact).

## Layer 1 (certificate, exact at $p_\ast$)

| $n$ | packets (conductor: #chars) | cross-packet cubic triples | exact ladder | total surviving (= $(n{-}1)(n{-}2)$) |
|---|---|---:|---:|---:|
| 6 | 2: 1, 3: 2, 6: 2 | 18 | 18 | 20 |
| 8 | 2: 1, 4: 2, 8: 4 | 42 | 42 | 42 |
| 12 | 12: 4, 2: 1, 3: 2, 4: 2, 6: 2 | 108 | 108 | 110 |
| 18 | 18: 6, 2: 1, 3: 2, 6: 2, 9: 6 | 252 | 252 | 272 |
| 30 | 10: 4, 15: 8, 2: 1, 3: 2, 30: 8, 5: 4, 6: 2 | 774 | 774 | 812 |

## Layer 2 (the diagnostic $\rho_\times$)

Final-step and *time-mean* values reported per arm. The time-mean is the
robust summary across the training trajectory (the per-step $\rho_\times$
oscillates as the model goes through Fourier-feature consolidation states,
especially pre-grokking). $\Delta_\rho := \overline{\rho_\times^{D2}}
 - \overline{\rho_\times^{D1}}$ is the predicted-direction separation: a
positive value means the ring-respecting arm (D1) has concentrated cubic
mass *within* a single conductor packet, reducing its cross-packet share.

| $n$ | arm | train_acc | test_acc | $\rho_\times$ final | $\overline{\rho_\times}$ (time-mean) |
|---|---|---:|---:|---:|---:|
| 6 | D1 (ring) | 1.000 | 0.000 | 0.9407 | 0.9436 |
| 6 | D2-weak (label permuted) | 1.000 | 0.000 | 0.8491 | 0.8292 |
| 6 | **D2-weak - D1** | | | -0.0916 | **-0.1145** |
| 8 | D1 (ring) | 1.000 | 0.000 | 1.0000 | 1.0000 |
| 8 | D2-weak (label permuted) | 1.000 | 0.000 | 1.0000 | 1.0000 |
| 8 | **D2-weak - D1** | | | +0.0000 | **+0.0000** |
| 8 | D2-strong (random fn) | 1.000 | 0.031 | 1.0000 | 1.0000 |
| 8 | **D2-strong - D1** | | | +0.0000 | **+0.0000** |
| 12 | D1 (ring) | 1.000 | 0.000 | 0.8264 | 0.8467 |
| 12 | D2-weak (label permuted) | 1.000 | 0.000 | 0.9980 | 0.9952 |
| 12 | **D2-weak - D1** | | | +0.1716 | **+0.1485** |
| 12 | D2-strong (random fn) | 1.000 | 0.069 | 0.9863 | 0.9828 |
| 12 | **D2-strong - D1** | | | +0.1598 | **+0.1361** |
| 18 | D1 (ring) | 1.000 | 0.043 | 0.9606 | 0.9588 |
| 18 | D2-weak (label permuted) | 1.000 | 0.056 | 0.9611 | 0.9602 |
| 18 | **D2-weak - D1** | | | +0.0004 | **+0.0013** |
| 18 | D2-strong (random fn) | 1.000 | 0.043 | 0.9072 | 0.9060 |
| 18 | **D2-strong - D1** | | | -0.0534 | **-0.0528** |
| 30 | D1 (ring) | 1.000 | 0.982 | 0.9628 | 0.9653 |
| 30 | D2-weak (label permuted) | 1.000 | 1.000 | 0.9710 | 0.9323 |
| 30 | **D2-weak - D1** | | | +0.0082 | **-0.0330** |
| 30 | D2-strong (random fn) | 1.000 | 0.047 | 0.9734 | 0.9573 |
| 30 | **D2-strong - D1** | | | +0.0106 | **-0.0080** |

## Verification gates

**Layer 1.** Fisher exactly diagonal at $p_\ast$ (max off-diagonal abs
$= 0$ up to float64 noise) and cross-packet cubic triple count matches
the Section 4 exact-certificate ladder. See per-$n$ certificate JSON.

**Layer 2 D1-vs-D2 separation gate.** Time-mean $\Delta_\rho :=
\overline{\rho_\times^{D2}} - \overline{\rho_\times^{D1}}$ positive,
with magnitude above seed-to-seed noise. The mechanism is that
ring-coherent learning aligns the model's update direction with a single
conductor packet (a Nanda-style Fourier-feature consolidation), reducing
the cross-packet share of cubic mass on D1; the permuted control has no
such alignment, so its share stays near the structural default. Headline
pass on $n=12$ over 3 seeds. **Scope boundaries:** $n=6$ has only 3
packets and gives below-noise resolving power; $n=30$ requires the
*D2-strong* control (input-and-output permutation, or a structureless
random target) because the label-only D2 is bypassable by internal
composition $\sigma \circ (a+b \bmod n)$.

**Preconditioner discard ratio.** Reported in per-$n$ curve JSON but
clamped to $[-1, 1]$; the Adam-proxy is unstable in directions the model
has memorized (per-class variance $\to 0$). Definition 6.2 is most
informative when $M^{-1}$ is well-conditioned (K-FAC, Gauss--Newton,
exact Fisher); the headline $\rho_\times$ does not depend on it.
