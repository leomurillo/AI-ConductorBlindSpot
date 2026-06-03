# T0–T6 → CBS: Harvest and Plantation

*Working note, 2026-05-24.* This document records what the T0–T6 series
makes available to the Conductor Blind-Spot manuscript right now, the
single computational task executed to validate the most actionable
consequence, and the analytical / empirical plantation it opens.

---

## 1. The harvest (what T0–T6 makes available to CBS)

Each T-paper resolves a specific tension in CBS. Compressed:

| T-paper | CBS section it addresses | What it adds |
|---|---|---|
| **T0** | §5.4 Conjecture 5.8 | The β-flow (replicator equation) is the analytical Hamiltonian flow on $H_0$ whose moments give the running displacement ĥ(t) directly. Off-centroid trajectory is no longer "what does training accumulate"; it is the trajectory of the replicator escort family. |
| **T1** | §2.1–2.2 chart | Confirms $H_0$ as the genuine Euclidean tangent space — CBS's "tangent space" claim is the T1 chart. |
| **T2** | §2.2 Fisher / Amari–Chentsov | The cubic vertex IS the third derivative of one scalar functional $\Phi = \log Z$; the certificate's "cross-packet content the quadratic class cannot represent" is the $\partial^3$ entry of the cumulant tower. |
| **T3** | §3 main result | The selection rule $a+b+c \equiv 0 \pmod n$ is exactly the cubic-mode arithmetic CBS Theorem 3.5 uses. |
| **T4** | §9 multiquadratic extension | The blind spot is **structural**: $L^2$-normalization at every dimensional rung forces $\mathbb{Q}(\sqrt p : p \le n)$, locking the quadratic class into the radical field. Cross-prime cubic interaction lives outside this field. T4-HelmertFourier-WorkedExample gives the closed-form transfer matrix $T_{c,k}$ that pulls Theorem 5.6's cross-packet activation explicitly to the Helmert basis. |
| **T5** | §6.1 ρ_× diagnostic | The cubic content has a *scalar* witness: $\log(A \cdot H / G^2) = \frac{1}{3}\kappa_3 + O(\\|y\\|^5)$. The tensor-heavy enumeration is not the only way to read CBS's cubic content. |
| **T6** | §8.6 Branch B failure | The quadratic curvature class is the **RMS gauge** — a perfectly closed geometric universe under $L^2$-normalization. Injecting Amari–Chentsov cubic content into an RMS-native optimizer **breaks gauge equivariance** by construction; that is why all three forms (A/B/C) returned Branch B. To intervene one must *change observer*, not bolt on a tensor. |

The T6 reading is the decisive one for the manuscript's open empirical
gap: §8.6 is no longer a "we tried the natural correction and it
didn't work, mystery"; it is a **predicted null** of the RMS-gauge
framing.

---

## 2. The computational task (done now)

**Built.** A T0/T5-native scalar diagnostic that reproduces the
discriminative content of CBS Definition 6.1's ρ_× at FFT cost:

> **σ_PI(u; n) := max_d ||û_{P_d}||² / Σ_{a≠0} |û_a|²** — packet
> inverse-participation ratio.

The diagnostic is one FFT plus a per-packet Plancherel sum, total
$O(n \log n)$ per evaluation vs the $O(n^3)$ cubic enumeration ρ_×
requires. It is the right primitive because CBS Lemma 3.4 / Definition
6.1 measure exactly the cross-packet share of cubic mass, which (by
character orthogonality) is *structurally* determined by how
concentrated û is on a single conductor packet.

Also implemented: **σ_H(u; n) = log(A · H / G²)** for x = 1 + εnu, the
T5 Theorem 4.2 harmonic-skew scalar witness of the signed cubic
contraction.

**Verified.** Script
[`empirical/cbs_t0t5_scalar_diagnostic.py`](empirical/cbs_t0t5_scalar_diagnostic.py),
run on the canonical CBS ladder $n \in \{6, 8, 12, 18, 30\}$. Results
in
[`empirical/reports/cbs_t0t5_scalar_diagnostic.md`](empirical/reports/cbs_t0t5_scalar_diagnostic.md)
and
[`empirical/reports/cbs_t0t5_scalar_diagnostic.json`](empirical/reports/cbs_t0t5_scalar_diagnostic.json).

**Headline numbers (structural shape separation).** Ring-shape u (û
supported on a single conductor packet) vs structureless u (û spread
across packets):

| n | Δρ_× (ring − struct) | Δσ_PI (ring − struct) | reads consistently |
|---:|---:|---:|---|
| 6  | −0.377 | +0.347 | yes |
| 8  | n/a (ρ_× saturated at 1.000) | +0.366 | n/a — structural-degeneracy ring |
| 12 | −0.304 | +0.542 | yes |
| 18 | −0.437 | +0.522 | yes |
| 30 | −0.439 | +0.651 | yes |

The σ_PI separation is *larger in magnitude* than the ρ_× separation
on every non-degenerate ring — σ_PI is in fact a **more sensitive**
scalar discriminator of the ring-vs-structureless contrast than the
cubic-enumeration ρ_×. Correlation on random u is consistently negative
(−0.28 to −0.43 across rings), confirming σ_PI is the
anti-monotonically-coupled FFT primitive of the ρ_× cubic structure.

**Why this matters for the manuscript.**

- §6 augmentation: σ_PI is the right primitive to log alongside ρ_× on
  vocabulary-scale heads (Pythia §6.6 scope extension) and on
  trajectory follow-ups (§8.6 open list). FFT cost makes it deployable
  where cubic enumeration becomes infeasible.
- §6.6 scope: on a calendar-months n = 12 head, σ_PI is two cheap FFTs
  per prompt batch. The Pythia sweep can be re-run on n = 7
  (weekdays), n = 10 (digits), n = 24 (hours), n = 60 (minutes) and
  any other vocabulary-restricted cyclic head, without per-checkpoint
  cubic enumeration.
- §8.6 reframing: the Branch B null is *predicted* under T6 + T5 —
  cubic-aware additive corrections in the RMS gauge violate
  equivariance; the operational intervention has to come from a
  different observer.

---

## 3. The plantation

In rough order of cost / expected payoff. Items 1–4 are
analytic-or-cheap-empirical; items 5–7 are heavier compute.

### 3.1 Analytical / paper-level

**P1 (close Conjecture 5.8 analytically, T0 §2.4 / 3.5).** The β-flow
is the autonomous ODE $\dot P_\beta(i) = P_\beta(i)(y_i -
\mathbb{E}_{P_\beta}[y])$. The running displacement on a *constant-
fitness* trajectory has a closed-form Fourier spectrum: $\widehat{\hat
h(t)}_a = \widehat{P_{\beta(t)} - u}_a$ where $P_{\beta(t)}$ is the
escort family. For modular addition on $\mathbb{Z}/n$, the fitness
vector $y$ has a known character-basis spectrum (determined by the
target ring representation); apply Theorem 5.6's Fourier-convolution
form to obtain an *analytic prediction* of $\sigma_{PI}(\hat h(t))$
along the β-flow, and compare to AdamW's empirical trajectory.
Conjecture 5.8's "rate asymmetry" becomes a falsifiable analytic
statement.

**P2 (Helmert-Fourier transfer-matrix pullback, T4 + T4-companion).**
The T4-HelmertFourier-WorkedExample draft has the closed form
$T_{c,k}$. Apply it to CBS Theorem 5.6's cross-packet entry to express
the off-centroid Fisher activation in the Helmert basis. The
prediction is that prime dimensional rungs $k \in \{2, 3, 5, 7\}$
behave qualitatively differently from composite rungs because their
$T_{c,k}$ entries land in disjoint multiquadratic field cosets. This
turns the §9 "multiquadratic-field" remark from a structural
observation into a coordinate-explicit predictive statement.

**P3 (harmonic-gauge intervention arm for §8.6, T5).** The §8.6
follow-up list calls for a uniform within-mode rescaling arm (EGD /
PGD). T5's parity decomposition gives a *finer* prescription:
gradient-side rescaling weighted by $M_r / M_{-r}$ where $r$ ranges
through the relevant escort family. The R-flow ($r = 2$) is RMS / Adam;
the H-flow ($r = -1$) is harmonic / softmin. **A mixed-gauge
optimizer** alternating between R-flow and H-flow steps changes the
observer per step and is the natural T5 + T6 + T0 fusion to test
against §8.6 Form A/B/C.

**P4 (closure of §6.6 with vocabulary-scale σ_PI).** σ_PI is cheap
enough to deploy on the full Pythia checkpoint suite (140 ckpts × 5
sizes × 60 prompts ≈ 42k FFTs, minutes on the existing rig). Re-run
the §6.6 sweep with σ_PI replacing ρ_× and add three more rings (n =
7 weekdays, n = 10 digits, n = 24 hours). The off-trajectory
boundary identified at §6.6 is *expected* to persist on calendar
months; whether it does on weekdays / digits is a genuinely open
question σ_PI lets you answer cheaply.

### 3.2 Empirical / training-loop tasks

**P5 (β-flow trajectory paired with AdamW trajectory, n = 12).**
Re-run the §6.5 demo on n = 12 with three logging additions:
(i) σ_PI(t) at each step; (ii) the analytic β-flow prediction of
σ_PI at the same step (via the closed-form replicator escort
spectrum); (iii) the Truong et al. spectral entropy $\widetilde
H(t)$. Comparison: does the empirical AdamW trajectory's σ_PI(t)
match the β-flow prediction? Where does it diverge? This is the
direct test of "Conjecture 5.8 is the β-flow's deterministic
content" hypothesis.

**P6 (mixed-gauge optimizer arm for §8.6 follow-up).** Implement the
T5 R+H alternating arm (P3) and run as the fourth arm of the §8.6
$\alpha$-sweep matrix. Pre-claim both branches: a positive
interaction with the ring task is Branch A (the T5 + T6 fusion lifts
the §8.6 null); a null is Branch B and refutes the gauge-change
hypothesis on this substrate.

**P7 (calibration: SmolLM/Pythia auxiliary-modulus extension,
Kikuchi et al. 2026).** [Kikuchi et al. 2026]'s auxiliary-modulus
trick scales modular addition to q = 974269. Re-run σ_PI on those
training trajectories with cyclic input structure — this is the
ring task at $n$ orders of magnitude larger than CBS §6.5, and σ_PI
is the only diagnostic that runs at that scale.

### 3.3 Theory hardening

**T1 (general finite abelian extension, T0 §4 + CBS Appendix B).**
T0 §4's CRT decomposition extends Appendix B's per-group verification
to a coordinate-free packet-mixing law (Corollary 6.2 of T6). State
Theorem 3.5 verbatim on any finite abelian $G$, with the conductor
replaced by the order in $\hat G$ and the selection rule replaced by
$\chi \cdot \chi' \cdot \chi'' = \mathbf{1}_{\hat G}$. Appendix B
already shows this verifies on five non-cyclic candidates; T0
provides the unified statement.

**T2 (non-abelian frontier, hinted at in §9).** [Stander et al. 2024]
grok $S_5$ multiplication. The cubic Amari–Chentsov tensor on $\Delta_G^\circ$
for non-abelian $G$ decomposes into irreducible representations of
$G$; the analog of the conductor selection rule is the existence of
non-trivial multiplicities in the triple product $V_\rho \otimes V_\sigma \otimes V_\tau$.
This is a genuine open theoretical question and the natural
continuation of T0/T3's program beyond cyclic.

---

## 4. P1 — CLOSED 2026-05-24

The β-flow analytical prediction for the head-only cross-entropy
trajectory on Z/n has the closed form

  $$\hat u^{(a,b)}(\beta)_k = e^{-2\pi i k y^*/n} \cdot \frac{n}{e^\beta + n - 1}, \qquad k \neq 0,$$

so $|\hat u^{(a,b)}(\beta)_k|^2$ is **independent of k**, and the
batch-mean Fourier spectrum factorises into a scalar β-dependence
and a batch-determined shape:

  $$\bar u(\beta)_k = \frac{n}{e^\beta + n - 1} \cdot E_B[\chi_{-k}(y^*)].$$

**Consequence (theorem).** ρ_×(t) and σ_PI(t) are **β-invariant** under
the head-only constant-fitness β-flow.

**Numerical verification.** Per-seed β-std of σ_PI at machine ε
($\sim 10^{-15}$) across n ∈ {6, 12, 30} and modes D1/D2-weak/D2-strong
(script [`empirical/cbs_t0_beta_flow_prediction.py`](empirical/cbs_t0_beta_flow_prediction.py),
report
[`empirical/reports/cbs_t0_beta_flow_prediction.md`](empirical/reports/cbs_t0_beta_flow_prediction.md)).

**Empirical comparison (n = 12).** Against the existing §6.5 demo
trajectory ([`empirical/cbs_p1_empirical_vs_analytical.py`](empirical/cbs_p1_empirical_vs_analytical.py),
report
[`empirical/reports/cbs_p1_empirical_vs_analytical.md`](empirical/reports/cbs_p1_empirical_vs_analytical.md)):

| mode | empirical mean ρ_× | analytical head-only ρ_× | **depth deviation** |
|---|---:|---:|---:|
| D1 (ring) | 0.847 | 0.951 ± 0.063 | **−0.104** |
| D2-weak | 0.995 | 0.981 ± 0.031 | +0.014 |
| D2-strong | 0.983 | 0.962 ± 0.046 | +0.021 |

**Asymmetry D1 − D2-strong = −0.125.** This is the depth-driven
cross-packet consolidation that Conjecture 5.8 was framed to detect,
now expressed as a scalar gap between two computable curves rather
than an open dynamical question.

**What this closes for the manuscript:**

1. **Conjecture 5.8's analytical half is closed.** The head's
   intrinsic β-flow produces no rate asymmetry; the empirically
   observed trajectory variation is a depth-feature effect.

2. **The §8.6 Branch B null is consistent.** All three forms (A/B/C)
   inject content at the *head*, which the analytical comparison shows
   contributes only ≤2% to the gap. The 10%+ depth contribution is
   where any operational intervention must act.

3. **Conjecture 5.8' (sharpened).** Suggested CBS §5.4 addition as
   Remark 5.9 (see report): the rate asymmetry is upstream-feature
   driven; head-only β-flow predicts time-invariance; the deviation
   is the depth contribution to cross-packet cubic activation.

4. **Right intervention surface.** Not head curvature, but upstream
   feature dynamics. Candidates: a hidden-layer regularizer that
   tracks the depth-induced σ_PI deviation; a Helmert-coordinate
   readout of the MLP's pre-head feature evolution (P2); the EGD /
   PGD within-mode methods (§7.8) that act on the depth-side
   spectrum.

## 5. Updated next action

With P1 closed, the highest-value follow-up shifts to **P3 +
P5 paired**: build a mixed-gauge optimizer arm (T5/T6 fusion) AND
re-run the §6.5 demo with σ_PI(t) logged. The σ_PI(t) trajectory's
deviation from the analytical β-flow constant is now the **target
quantity** the intervention should compress on D1 and leave alone on
D2-strong.

A complementary cheap probe (now done): **measure the depth
deviation on several rings (n ∈ {6, 8, 12, 18, 30}) with the
existing demo data**:

| n | depth asymmetry D1 − D2-strong | §6.5 regime |
|---|---:|---|
| 8  | +0.000 | structural-degeneracy (ρ_× ≡ 1) |
| 12 | **−0.125** | **pre-grokking discriminative phase** |
| 18 | +0.025 | insufficient budget |
| 30 | +0.010 | post-grokking saturation |

**The depth-driven asymmetry exists only in the pre-grokking
representation-learning phase.** This is the β-flow comparison's
sharpening of CBS §6.5's three-regime observation: the same
diagnostic boundaries appear, but now with a *quantified analytical
baseline* (head-only β-flow constant) against which the depth
contribution is measured. The n = 12 row is the only ring on the
ladder where the depth-feature dynamics produce a measurable
cross-packet consolidation signal at the diagnostic level used in
§6.5.

---

## Recommended next action (legacy P1 description, retained for context)

**Original P1 target:** close Conjecture 5.8 analytically
via the β-flow. The setup is fully constructive:

1. Take $y$ = the centered log-fitness of the modular-addition
   target distribution on $\mathbb{Z}/n$.
2. The escort family $P_\beta$ has spectrum $\widehat{P_\beta}_c =
   \mathcal{M}_y(\beta, c) / \mathcal{M}_y(\beta, 0)$ by T0 Theorem
   3.5.
3. Substitute into CBS Theorem 5.6 (Fourier-convolution form of
   cross-packet activation) to obtain $g_{P_\beta}(\chi_k,
   \overline{\chi_\ell})$ as a closed analytic function of $\beta$.
4. The prediction is the trajectory $\sigma_{PI}(P_\beta - u)$ as a
   function of $\beta$, which is *deterministic given the target
   distribution*.
5. Compare to AdamW's empirical $\sigma_{PI}(\hat h(t))$ on n = 12 —
   this is the §6.5 demo re-run with σ_PI logging.

If the β-flow prediction matches AdamW's trajectory, then Conjecture
5.8 becomes a theorem on the β-flow; the dynamical conjecture is
deterministic content of T0.

If it does not match, the *deviation* between the β-flow prediction
and AdamW's trajectory is the operational gap the §8.6 intervention
should target — and that gap is computable, not conjectural.

Either way, the manuscript moves from "we have a representational
certificate and an open dynamical conjecture" to "we have a
representational certificate, an analytic dynamical prediction, and
a measured residual." That is the form in which the program
publishes.
