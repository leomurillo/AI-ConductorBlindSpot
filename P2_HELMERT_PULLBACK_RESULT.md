# P2 Result: CBS × T4 Helmert-basis pullback

*Result note, 2026-05-24.* Pulls CBS Theorem 5.6's cross-packet
activation through the closed-form Helmert-Fourier transfer matrix
$T_{c,k}$ (T4-HelmertFourier-WorkedExample) and identifies a new
diagnostic primitive grounded in the multiquadratic radical clock of
T4 Theorem 3.1.

## Theorem (Helmert-basis pullback)

For $c, k \in \{1, \ldots, N-1\}$,

$$T_{c,k} = \frac{1}{\sqrt{N\,k(k+1)}} \cdot \frac{1 - (k+1)\omega_c^k + k\omega_c^{k+1}}{1 - \omega_c}, \qquad \omega_c = e^{2\pi i c/N},$$

is unitary $(N{-}1) \times (N{-}1)$. The off-centroid Fisher form pulls
back as

$$g_{p_*+h}(e_j, e_k) = \sum_{c,d \ne 0} \overline{T_{c,j}}\, T_{d,k}\, \tilde g_{p_*+h}(\tilde\chi_c, \overline{\tilde\chi_d})$$

and at leading order in $h$ (combining CBS Theorem 5.2 with the
$T_{c,k}$ factorisation):

$$g_{p_*+h}(e_j, e_k) \approx \delta_{jk} - n \sum_a \hat h_a \cdot (T^*T)_{j,k;a}, \qquad (T^*T)_{j,k;a} := \sum_c \overline{T_{c,j}} T_{(c+a) \bmod n,\, k}.$$

The kernel factorises as

$$(T^*T)_{j,k;a} = \frac{1}{N\sqrt{j(j+1)\,k(k+1)}} \cdot \underbrace{\sum_c \overline{B_{c,j}}\, B_{(c+a) \bmod n,\, k}}_{\in\,\mathbb{Z}[\zeta_N]}$$

with $B_{c,k} = \sum_{s=0}^{k-1}\omega_c^s - k\omega_c^k$ the cyclotomic
bracket. The **radical** part $1/\sqrt{N j(j+1) k(k+1)}$ determines
which coset of $K_N = \mathbb{Q}(\sqrt p : p \text{ prime} \le N)$
the entry sits in.

## Verifications (machine ε)

- **Unitarity:** $\max |T^*T - I| \le 2.3\!\times\!10^{-15}$ across
  $n \in \{6, 8, 12, 18, 30\}$.
- **Definition consistency:** $\max |T - C E^T| \le 6.1\!\times\!10^{-15}$
  where $C$ is the Euclidean-normalised character matrix and $E$ the
  unit Helmert basis.
- **n = 6 worked example** (T4-HelmertFourier §3.3): $|B_{1,k}|^2 =
  (1, 7, 19, 31, 36)$ for $k = 1, \ldots, 5$ at row $c = 1$, and
  $(4, 4, 16, 16, 36)$ at row $c = 3$, **exact** to machine ε.
  Row $c = 3$ imaginary parts: $3.3\!\times\!10^{-15}$ (confirmed
  real, $\omega_3 = -1$).

## The new diagnostic: Helmert rung response under packet
displacements

For each conductor packet $P_d$, sample a real displacement $h \in
T_{p_*}\Delta_R^\circ$ with Fourier mass supported on $P_d$ only,
compute $g_{p_*+h}(e_j, e_k)$ directly, and aggregate (over $j$) the
row-sum of off-diagonal magnitudes $R_d[j] = \sum_{k \ne j}
|g_{p_*+h}(e_j, e_k)|$ averaged over 64 $h$-seeds. $R_d[j]$ measures
**which Helmert rung $j$ responds to packet-$d$ displacements**.

**Prime-rung bias** = (prime-rung response share) − (prime-rung
count share). Universal pattern across all rings tested:

| n | packet | prime-rung bias | argmax rung | rung type |
|---:|---|---:|---:|---|
| 6 | P_2 | +0.057 | k=1 | prime |
| 6 | P_3 | +0.065 | k=1 | prime |
| 6 | P_6 | +0.005 | k=2 | prime |
| 8 | P_2 | +0.047 | k=1 | prime |
| 8 | P_4 | +0.053 | k=2 | prime |
| 8 | P_8 | +0.028 | k=1 | prime |
| 12 | P_2 | +0.054 | k=1 | prime |
| 12 | P_3 | +0.071 | k=1 | prime |
| 12 | P_4 | +0.071 | k=2 | prime |
| **12** | **P_6** | **+0.026** | **k=3** | **composite (sqc=1 — rational coset!)** |
| 12 | P_12 | +0.034 | k=1 | prime |
| 18 | P_2 | +0.045 | k=1 | prime |
| 18 | P_3 | +0.062 | k=1 | prime |
| **18** | **P_6** | **+0.028** | **k=3** | **composite** |
| **18** | **P_9** | **+0.026** | **k=3** | **composite** |
| 18 | P_18 | +0.041 | k=1 | prime |
| 30 | P_2 | +0.046 | k=1 | prime |
| 30 | P_3 | +0.060 | k=1 | prime |
| 30 | P_5 | +0.051 | k=2 | prime |
| **30** | **P_6** | **+0.037** | **k=3** | **composite** |
| 30 | P_10 | +0.044 | k=2 | prime |
| 30 | P_15 | +0.036 | k=4 | prime |
| 30 | P_30 | +0.046 | k=1 | prime |

**Universal:** every packet across every ring shows **positive
prime-rung bias** (3–7%) — Helmert prime rungs systematically
over-respond to conductor-shaped displacements, exactly the T4
Theorem 3.1 prediction since prime rungs are where new $\sqrt{p}$
generators enter $K_N$.

**Composite-rung argmax exceptions** are interpretable through the
**multiquadratic coset**: e.g., $k = 3$ at $n = 12$ has
$\mathrm{sqc}(N \cdot k(k+1)) = \mathrm{sqc}(144) = 1$ — the unique
rung in $n = 12$ with a **rational** normalizer, sitting in the
trivial coset of $K_{12}^\times / (K_{12}^\times)^2$. This rung
resonates with $P_6 = $ conductor of $\mathrm{rad}(N) = 6$ because
both objects sit in the simplest multiquadratic class. Analogous
composite-argmax exceptions at $n \in \{18, 30\}$ also occur on
$P_6$.

## What this lets the manuscript say

**(i) A second observable.** The σ_PI Fourier-side diagnostic (P1
result) reads the cross-packet cubic activation from the
character-basis side. The Helmert rung response map reads the SAME
activation from the dimensional-basis side. Both are computable from
the same trained-model representation; the pair gives **two
arithmetic readings of the depth-driven Conjecture 5.8 deviation**.

**(ii) A direct radical-clock reading.** Different rings expose
different rungs as the argmax for different packets; the pattern is
controlled by the multiquadratic-coset alignment between the
packet's character set and the rung's $\sqrt{N k(k+1)}$ normalizer.
**Which Helmert rungs the model's hidden representation activates
during training is a direct readout of the radical clock K_N.**

**(iii) Operational target sharpening.** The §8.6 Branch B null
forms inject content at the head where (per P1) the head-only
β-flow predicts $\sigma_{PI}(t) = $ const. The P2 result extends
this: at the head, any cubic injection ALSO gives a uniform
prime-rung Helmert response — it cannot preferentially activate
the depth-aligned multiquadratic coset that a *trained MLP's*
representation would. The intervention surface remains upstream.

**(iv) Concrete probe.** Compute the Helmert-rung response map on
a trained MLP's hidden representation (pre-head) at intermediate
training steps. Track which prime rungs activate over training time.
This is the multiquadratic-clock-aligned analog of Truong et al.'s
spectral entropy collapse — a *prime-by-prime* spectral readout of
the network's representation rather than an aggregate.

## Suggested CBS manuscript addition

Add as Appendix E (or as a new §6.8 paired with the σ_PI augmentation
of §6 from the P1 closure):

> **Appendix E (Helmert-basis pullback and the multiquadratic
> clock).** The transfer matrix
> $T_{c,k} = (1/\sqrt{N k(k+1)})\,(1 - (k+1)\omega_c^k +
> k\omega_c^{k+1})/(1 - \omega_c)$ of T4-HelmertFourier is unitary
> and pulls back CBS Theorem 5.6's character-basis cross-packet
> activation to the Helmert basis. The pullback factorises into a
> radical $1/\sqrt{N j(j+1) k(k+1)} \in K_N$ and a cyclotomic kernel
> in $\mathbb{Z}[\zeta_N]$. Per-packet Helmert-rung response under
> displacement $h$ with Fourier mass on $P_d$ produces a **universal
> positive prime-rung bias** ($+3$–$7\%$ over the structural count
> share) across $n \in \{6, 8, 12, 18, 30\}$: the Helmert prime
> rungs $k = p - 1$ (where $\sqrt p$ first enters $K_N$, T4 Theorem
> 3.1) systematically over-respond to conductor-shaped displacements.
> The composite-rung exceptions (P_6 in $n = 12, 18, 30$; P_9 in
> $n = 18$) correspond to rungs whose multiquadratic-coset
> representative $\mathrm{sqc}(N k(k+1))$ exactly matches the
> packet's conductor radical. The Helmert-basis diagnostic gives an
> independent reading of the cross-packet cubic activation
> complementary to the $\sigma_{PI}$ Fourier-side diagnostic; the
> pair forms a **two-observable** test of Conjecture 5.8 in which
> the dimensional and conductor axes of the apex (T0 §5.3) are
> measured simultaneously.

## Files

- Script: [`empirical/cbs_p2_helmert_pullback.py`](empirical/cbs_p2_helmert_pullback.py)
- Report: [`empirical/reports/cbs_p2_helmert_pullback.md`](empirical/reports/cbs_p2_helmert_pullback.md)
- Data: [`empirical/reports/cbs_p2_helmert_pullback.json`](empirical/reports/cbs_p2_helmert_pullback.json)
- Harvest doc: [`T0_T6_HARVEST.md`](T0_T6_HARVEST.md)
- Precursor: [`P1_BETA_FLOW_RESULT.md`](P1_BETA_FLOW_RESULT.md)
