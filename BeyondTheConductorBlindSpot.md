<!--
================================================================================
INTERNAL PROJECT LOG — NOT FOR PUBLICATION
HTML comments; Pandoc drops them on the way to LaTeX, so they never reach the
PDF. Editorial state for future-us and the team.

  Filename:     BeyondTheConductorBlindSpot.md  (pairs with ConductorBlindSpot.md)
  Build:        ./build.sh BeyondTheConductorBlindSpot.md   (same toolchain as CBS;
                title/author/date parsed from the H1 block below)
  Target venue: arXiv preprint (cs.LG / stat.ML primary; math.PR, math.ST secondary)
  Status:       DRAFT 1 — structure complete, prose to polish. Eulerian house style.

--------------------------------------------------------------------------------
FRAMING DECISIONS (locked 2026-05-31)

  * Build DIRECTLY on CBS: this is the continuous generalisation of the cyclic
    conductor blind spot. Assume reader familiarity with CBS at the level of its
    abstract + Section 3 (order-2 Fisher blind to order-3 Amari-Chentsov cubic).
  * Citable internal papers: ONLY CBS and T0-T7 (the gauge program). T12 (finite
    apex determinacy / Hankel) is NOT yet public — its needed lemmas are RESTATED
    and PROVED in-house in Appendix A.4 / B, never cited by number.
  * LeJEPA (Klindt, LeCun & Balestriero 2026) is cited MINIMALLY and
    professionally: it is the Gaussian/linear special case, and a clear statement
    of the world-model stakes. We do NOT frame the paper as "generalising LeJEPA"
    and we do not trade on author names. The paper stands on CBS + T0-T7 + its own
    theorems. (Reviewer-safety: an outsider over-leaning on a famous name reads as
    leverage; we avoid that. The maths is ours; the corner is theirs; we say so once.)

--------------------------------------------------------------------------------
THE LOAD-BEARING DISTINCTION (do not let any later edit blur this)

  TWO TOWERS, coinciding only at the Gaussian:
    (D) DYNAMICAL tower  = eigenfunctions of the transition operator T.
                           Recovery map = its slow eigenfunction phi_1.
                           phi_1 affine  <=>  Gaussian (Sturm-Liouville).  [E1]
    (S) DISTRIBUTIONAL tower = orthogonal polynomials / cumulants of the
                           stationary law. Degree-1 member ALWAYS affine.
                           This is what T0-T7 and CBS grade.            [E3]
  They coincide for the Gaussian (Hermite = both). CBS's order-2/order-3 boundary
  (S-register, finite, exact) and this paper's linear/nonlinear recovery boundary
  (D-register, continuous, learned) are the SAME boundary in two registers.
  Earlier internal notes (and E3's first narration) loosely called the degree-1
  orthogonal polynomial "the recovery map" — that is FALSE off-Gaussian. Fixed in
  e3 script and never asserted here. If a future pass reintroduces "recovery order
  = Hankel rank of the recovery map", STOP: that conflates (D) and (S).

--------------------------------------------------------------------------------
OPEN TODOs / OPPORTUNITIES (team)
  [ ] CBS DOI: fill exact Zenodo/arXiv id in refs [C1] (placeholder now).
  [ ] Figures: e1_eigenfunctions.png, e2_bound_scatter.png live in
      empirical/apex_recovery/reports/ — embed in build or keep as repo pointers.
  [ ] Cor D (approximate planning): exact simulation-lemma constant is sketched;
      a tight constant is a clean appendix addition for v2.
  [x] DONE (E4): the "two registers of one boundary" is now MEASURED — nu_D ∝
      (leading cumulant)^2 near the Gaussian, order 3 skewed / order 4 symmetric,
      constants 0.0555 / 0.0104; plus the scalar-skewness-0 vs tensor-cubic-nonzero
      contrast on the ring. See §3.4 "the bridge, measured", §6.4, App B, e4 script.
  [x] DONE (E5, §6.5): the SAME diagnostic end-to-end on a pretrained model
      (Pythia-70m residual stream). Honest result: the WEAK bridge holds at every
      estimable layer (Gaussian dirs affine); the STRONG cross-direction bridge is
      a MIDDLE-LAYER phenomenon (Spearman -> +0.41 at layer 4, band 3-5; ~0 at
      embedding/early/final), exactly as the two-tower distinction predicts. NOT a
      universal real-model square law -- the depth profile IS the result. E5 is
      optional (torch+transformers), observational, not in the pass/fail gate.
  [ ] OPPORTUNITY (next): more models/sizes; augmentation-pairs vs adjacent-token
      pairs; a matched linear-vs-eigenfunction probe head-to-head on the
      middle-layer rogue dims (the §7 interpretability claim, made quantitative).
  [ ] OPPORTUNITY: anisotropy -> signed-permutation identifiability is a genuine
      practical lever (disentanglement without rotation ambiguity). Section 7
      mentions it; could be its own short paper.
================================================================================
-->

# Beyond the Conductor Blind Spot: Eigenfunction Identifiability and Planning in Non-Gaussian Worlds

*What alignment-and-whitening recovers, when a linear probe suffices, and when it provably does not*

**Leonardo Murillo Montero**

*leonardo.murillo@gmail.com*

*May 31, 2026*

---

## Abstract

The companion certificate *The Conductor Blind Spot* proved a sharp, finite fact: on a categorical head whose outcomes carry a cyclic ring structure, the second-order curvature class — the Fisher form shared by natural gradient, K-FAC, Adam's empirical Fisher, and gradient-based attribution — is constitutionally unable to represent an order-three coupling (the Amari–Chentsov cubic) that the head's own information geometry carries. That statement is finite, exact, and lives on a discrete index. This paper asks, and answers, the continuous counterpart: **when a representation is learned by the generic self-supervised recipe — pull positive pairs together, keep the embedding whitened — what does it recover, and when is the second-order (linear) picture of that representation complete?**

We show that the optimum recovers, in each latent coordinate, the *slowest eigenfunction* of the pair-generating transition operator, up to a rotation: a clean, model-free consequence of the Ky Fan maximum principle. That eigenfunction is a **straight line exactly when the latent law is Gaussian**, and a known, monotone, *curved* coordinate otherwise. Three theorems follow. (i) **Exact recovery**: the learned chart equals the slow-eigenfunction chart up to a block rotation; a linear probe is complete if and only if the world is Gaussian, and is provably lossy otherwise. (ii) **Approximate recovery**: under inexact whitening ($\varepsilon$) and inexact alignment ($\delta$), the chart is recovered within $(\varepsilon+\sqrt{2\delta/\gamma})^2$, where $\gamma$ is the slow-feature spectral gap; the guarantee degrades gracefully and is governed by a checkable condition, not a tunable. (iii) **Planning**: because the recovered chart is a diffeomorphism, every optimal-control problem on the true latent transports *exactly* into the learned coordinates — planning in the representation equals planning in the world, with the warp applied when it is not the identity.

The unifying statement is that the conductor blind spot and the failure of linear probing share a single **trigger** — non-Gaussianity ($\kappa_{\ge 3}\neq 0$) — and are **quantitatively locked** near it; they are two registers of one boundary, offset by one in their order index and coinciding at the Gaussian corner. This is a correspondence of model classes that we *measure* — the finite order-three object regenerated exactly, the lock measured as a square law — not a single diagnostic run end-to-end across both. We give deterministic, exact-arithmetic and numerical certificates for every claim; the finite-categorical certificate independently regenerates the conductor blind spot's published counts, and a cross-register certificate measures the two registers locking together — the recovery-map curvature scales as the square of the leading distributional cumulant, vanishing with it at the Gaussian. The same boundary is visible, unforced, in a pretrained language model, where it appears as a middle-layer phenomenon: the heavy non-Gaussian directions of the residual stream are exactly those whose recovered slow feature is curved. The consequences for practice — interpretability (linear probes are provably partial off-Gaussian), optimization (the same order-two ceiling as the blind spot), and the auditing of learned world models (identifiability holds, but only up to a *known nonlinear* chart) — are drawn out explicitly.

**Keywords:** self-supervised learning, identifiability, transition operator, slow feature analysis, eigenfunctions, Sturm–Liouville, information geometry, Amari–Chentsov cubic, Ky Fan, optimal control, world models.

---

## 1. Introduction {#sec:intro}

### 1.1 A finite fact, and its continuous shadow

The conductor blind spot is a statement one can check by hand on a clock. Index the outcomes of a categorical head by $\mathbb{Z}/n\mathbb{Z}$ — the hours of an $n$-hour clock, the months of a calendar, the residues of a modular-arithmetic task. At the uniform (maximum-entropy) point the Fisher information is diagonal in the additive-character basis, hence *block-diagonal across conductor packets*; but the next form in the cumulant tower, the Amari–Chentsov cubic, couples those packets exactly when three frequencies sum to zero modulo $n$. The cubic coupling is therefore real local structure that no second-order curvature surrogate can represent. That is the blind spot: an order-three fact invisible to an order-two instrument, certified in exact integer arithmetic.

<!-- team: keep the clock image; it is the single most teachable entry point and it is literally true. -->

It is natural to suspect that the same ceiling governs *continuous* representation learning, where there is no ring and no exact integer certificate — only a network trained to make the embeddings of two related views agree. This paper makes that suspicion precise. We do not transport the cyclic combinatorics; we ask the question the continuous setting actually poses and follow the mathematics to its own answer.

### 1.2 The question, and the shape of the answer

A *world* is a hidden variable $z$ with a stationary law $p$, observed through *positive pairs* $(z,z')$ — two augmentations of an image, two nearby frames of a video, two consecutive states of a process — generated by a stationary, reversible, additive-noise transition. A learner sees only a scrambled observation $x=g(z)$ and trains an encoder so that paired embeddings align while the embedding distribution stays whitened (the standard anti-collapse constraint). The question is what the trained encoder recovers of $z$.

The answer has one moving part. Minimising alignment subject to whitening is, exactly, a Rayleigh quotient on the transition operator $T$ that carries one view to the next; by the Ky Fan maximum principle its solution is the span of the *slowest* non-constant eigenfunctions of $T$. So the encoder recovers the slow-eigenfunction chart of the world — nothing more, nothing less. Whether that chart is the latent itself or a curved version of it is then decided by a single classical dichotomy:

> the slowest eigenfunction of an additive-noise transition is an **affine** function of the latent **if and only if** the latent law is **Gaussian** (Sturm–Liouville);
> for every other law it is a **monotone but curved** coordinate.

The Gaussian is thus the one world where the recovered chart is the latent up to rotation — where a *linear* probe of the representation is complete. Everywhere else, recovery still succeeds, but through a known nonlinear coordinate, and a linear probe necessarily discards content.

We fix here a convention that recurs throughout, because it is the one place a careful reader can trip. A representation is read by a *linear probe* — a degree-one map — and that probe is complete exactly when the representation's **second-order statistics** (its covariance) determine the latent, which by Theorem 2 holds if and only if the world is Gaussian. So whenever we say "second-order" we mean *the sufficiency of second-order statistics*, never the degree of the probe map (which is one) nor the order of the cumulant whose appearance defeats it (which is three). With that reading fixed, the linear-probe boundary is the continuous counterpart of the blind spot: a linear reading of the representation is complete exactly on the Gaussian (second-order-sufficient) stratum, and provably partial off it.

### 1.3 Contributions

1. **An exact recovery theorem** (§3, Appendix A.1–A.3). The whitened-alignment optimum equals the slow-eigenfunction chart $\Phi_1$ up to a block rotation, under an explicit and checkable slow-feature gap. Linear identifiability holds iff the world is Gaussian; we give the Sturm–Liouville criterion and its finite-state counterpart.

2. **An approximate recovery theorem** (§4, Appendix A.5). Under whitening error $\varepsilon$ and alignment excess $\delta$, $\min_{U}\|h-U\Phi_1\|^2 \le (\varepsilon+\sqrt{2\delta/\gamma})^2$ with $\gamma$ the spectral gap — a graceful, gap-controlled degradation proved by a variational ("trace-gap") argument.

3. **A planning theorem** (§5, Appendix A.6). Because $\Phi_1$ is a diffeomorphism, every finite-horizon control problem transports exactly into the learned chart: value and optimal plan are preserved, with the warp applied when costs are stated on the latent and are not invariant to it. The Gaussian corner is the case where the warp is the identity.

4. **The two-registers bridge** (§3.4, Appendix B). We make precise the sense in which the conductor blind spot (finite, distributional, exact) and the linear-probe failure (continuous, dynamical, learned) are one boundary, coinciding at the Gaussian corner — and *not* the same in the naive sense that would conflate two distinct spectral towers.

5. **Certificates** (§6, Appendix C). Four self-contained, CPU-second, exact-where-possible experiments check every claim — one regenerates the conductor blind spot's published cross-packet counts, one measures the two registers locking together as a square law — and a fifth, observational certificate exhibits the same boundary on a pretrained language model as a middle-layer phenomenon.

6. **Consequences for practice** (§7). Interpretability, optimization, and the auditing of learned world models each inherit a precise statement from the dichotomy.

### 1.4 Relation to prior and concurrent work, stated once

The recovered object — the slow eigenfunctions of a transition operator — is the classical content of *slow feature analysis* and of *diffusion-map* geometry; what is new here is the identifiability reading and its sharp Gaussian boundary. The information-geometric grading by cumulant order, the conductor-packet decomposition, and the cubic selection rule are the gauge program \heb{ל}0–\heb{ל}7 and the conductor blind spot, on which this paper is built and which it cites directly. A concurrent result in self-supervised learning establishes the Gaussian/linear case — linear identifiability holds precisely for Gaussian latents — which is exactly the corner of the present theory; we use it as a clean statement of the world-model stakes and as the boundary case our continuous theorem contains, and claim no more of it than that.

<!-- team: this is the ONLY substantive LeJEPA mention before §7. Keep it one paragraph, professional, no names in the body prose (names live in the reference entry). -->

---

## 2. Setup: worlds, transitions, and what alignment optimises {#sec:setup}

We proceed in the Eulerian manner: a smallest concrete computation first, the general object named only once it is forced.

### 2.1 The smallest interesting world

Let a latent $g$ be standard Gaussian and let it drift by the simplest stationary, reversible rule — the Ornstein–Uhlenbeck step
$$g' = \rho\, g + \sqrt{1-\rho^2}\;\eta, \qquad \eta\sim\mathcal N(0,1),\quad \rho\in(0,1).$$
Now *observe* this world through a fixed, invertible, nonlinear lens: $z = g^3$. The pair $(z,z')=(g^3,(g')^3)$ is all the learner ever sees (after a further unknown scrambling $x=g(z)$, which we will show is irrelevant). What function of $z$ does an aligned, whitened encoder recover?

The answer can be written before any theory. In the $g$-coordinate the slow direction is $g$ itself (the OU step contracts every nonlinear power of $g$ faster than $g$). In the observed $z$-coordinate that same direction is
$$\boxed{\;\phi_1(z) = g = \sqrt[3]{z}\;,}$$
a manifestly *curved* coordinate. A linear probe of the representation regresses $z$ on $\phi_1=\sqrt[3]{z}$; it captures only $R^2\approx 0.61$ of the latent and is blind to the rest. The eigenfunction probe captures it exactly. We have not yet proved that the encoder recovers $\sqrt[3]{z}$ — but §3 will, and §6 verifies it numerically to discretisation accuracy. Hold the picture: *the learner recovers the cube root, and a straight line cannot see it.*

<!-- team: this cube-root example is the spine of the whole paper. It is computed for real in e1 (known_ground_truth_warp). Lead every talk with it. -->

### 2.2 Worlds

A **world** is a triple $(\,p,\;T,\;g\,)$:

- a **stationary law** $p$ on a latent space $\mathcal Z\subseteq\mathbb R^n$, which we take to be a product $p=\prod_{i=1}^n p_i$ over independent coordinates (the standard disentanglement premise);
- a **transition** generating positive pairs $(z,z')$, acting coordinatewise as a stationary, reversible, additive-noise step $z_i' = m_i(z_i)+\eta_i$ with $\eta_i\perp z_i$ and $p_i$ preserved;
- an unknown measurable **observation map** $x=g(z)$, an injection (a diffeomorphism onto its image).

Reversibility (detailed balance) is the one structural assumption that earns us a real spectral theorem; the Ornstein–Uhlenbeck step of §2.1 satisfies it, as does any Langevin diffusion with a confining potential. Per coordinate, define the **transition operator** on $L^2(p_i)$,
$$(T_i\psi)(z_i) = \mathbb E[\psi(z_i')\mid z_i].$$
By reversibility $T_i$ is self-adjoint; we assume it is compact (discrete spectrum), as holds for Ornstein–Uhlenbeck and confining Langevin generators. Write its eigenpairs
$$1=\lambda_0^{(i)}>\lambda_1^{(i)}\ge\lambda_2^{(i)}\ge\cdots\ge 0,\qquad \varphi_0^{(i)}\equiv 1,\ \varphi_1^{(i)},\ \varphi_2^{(i)},\dots$$
orthonormal in $L^2(p_i)$. The eigenfunction $\varphi_1^{(i)}$ — the **slowest non-constant mode** of coordinate $i$ — is the protagonist. Collect the leading modes into the **slow-feature chart**
$$\Phi_1(z) := \big(\varphi_1^{(1)}(z_1),\dots,\varphi_1^{(n)}(z_n)\big).$$

### 2.3 The learner, and what it optimises

The encoder is the composite $h=f\circ g:\mathcal Z\to\mathbb R^n$. Training minimises the **alignment** loss under a **whitening** constraint:
$$\min_h\ \mathbb E\big\|h(z')-h(z)\big\|^2 \quad\text{s.t.}\quad \mathbb E[h_i]=0,\ \ \mathbb E[h_i h_j]=\delta_{ij}.$$
A two-line computation (Appendix A.1) using stationarity and the definition of $T=\bigotimes_i T_i$ turns this into a Rayleigh problem:
$$\mathbb E\|h(z')-h(z)\|^2 = \sum_{i=1}^n 2\big(1-\langle h_i, T h_i\rangle\big),$$
so **minimising alignment is maximising $\sum_i\langle h_i, Th_i\rangle$ over whitened $\{h_i\}$** — a search for the slowest functions of the world. This is the only optimisation in the paper; everything else is its consequence. It is, verbatim, the variational principle of slow feature analysis, here read as an identifiability statement.

<!-- team: the whitening constraint is doing ALL the anti-collapse work. We deliberately do NOT require the embedding to be Gaussian (only whitened); that is what lets the theorem be non-Gaussian. Note this explicitly in §7 vs the Gaussian-regulariser recipe. -->

---

## 3. Exact recovery {#sec:exact}

### 3.1 The recovery theorem

**Theorem 1 (Slow-feature recovery).** *Suppose the per-coordinate operators $T_i$ are self-adjoint and compact, and the* **slow-feature gap** *holds:*
$$\gamma := \min_i \lambda_1^{(i)} - \max\Big(\max_i\lambda_2^{(i)},\ \max_{i\ne j}\lambda_1^{(i)}\lambda_1^{(j)}\Big) > 0. \tag{G}$$
*Then every minimiser of the whitened-alignment problem has the form*
$$h(z) = U\,\Phi_1(z),\qquad U\in O(n),$$
*with $U$ free only within blocks of equal $\lambda_1^{(i)}$. The minimal alignment value is $2\sum_i(1-\lambda_1^{(i)})$.*

The proof (Appendix A.2–A.3) is two classical facts. First, Ky Fan's maximum principle: the maximum of $\sum_{i}\langle h_i,Th_i\rangle$ over orthonormal $\{h_i\}$ is the sum of the top $n$ non-trivial eigenvalues of $T$, attained exactly on their eigenspace. Second, the product structure: the eigenfunctions of $T=\bigotimes_iT_i$ are products of single-coordinate modes, and (G) is precisely the statement that the $n$ slowest non-trivial modes are the $n$ single-coordinate leaders $\varphi_1^{(i)}$ — so the top eigenspace is the span of $\Phi_1$.

Two readings of the conclusion deserve names. The chart $\Phi_1$ is recovered **up to an orthogonal mixing $U$**; when the $\lambda_1^{(i)}$ are all equal (an *isotropic* world) $U$ is a full rotation — the familiar rotation ambiguity — but when they are *distinct* (an *anisotropic* world) $U$ collapses to a signed permutation, and the coordinates are individually identified. Anisotropy buys disentanglement.

### 3.2 When is the chart a straight line? The Gaussian boundary

Theorem 1 recovers $\Phi_1$; whether that is the latent itself or a curved version is decided coordinatewise by the shape of $\varphi_1$. Here the Eulerian payoff of §2.1: the score of the law decides everything.

**Theorem 2 (Affine recovery $\iff$ Gaussian).** *For a reversible additive-noise coordinate with stationary density $p_i$, the slow eigenfunction $\varphi_1^{(i)}$ is an affine function of $z_i$ if and only if $p_i$ is Gaussian. Equivalently, writing the Langevin generator $L_i\psi = D(\psi'' + (\log p_i)'\psi')$ that shares $T_i$'s eigenfunctions, an affine eigenfunction forces the score $(\log p_i)'$ to be affine, whose only solution is the Gaussian.*

The proof is a one-line ODE (Appendix A.3): if $\varphi_1(z)=az+b$ then $L_i\varphi_1 = aD(\log p_i)'$, and the eigen-equation $L_i\varphi_1=-\mu\varphi_1$ gives $(\log p_i)'(z) = -\tfrac{\mu}{aD}(z+b/a)$, i.e. $\log p_i$ is quadratic. For the Gaussian, $(\log p_i)' = -z$ is linear and $\varphi_1(z)=z$; for the Laplace law the score is $-\operatorname{sign}(z)$, for a bimodal law it is cubic-shaped, and in each case $\varphi_1$ bends. The cube-root world of §2.1 is the extreme illustration: its score is that of $g^3$, far from linear, and $\varphi_1=\sqrt[3]{z}$.

Combining Theorems 1 and 2:

> **Linear identifiability holds if and only if the world is Gaussian.** On the Gaussian stratum $\Phi_1=\mathrm{id}$ and $h(z)=Uz$ — recovery up to rotation, and a linear probe is complete. Off it, recovery is exact but through the curved chart $\Phi_1$, and any linear probe is provably lossy by exactly the amount its coordinates $\varphi_1$ depart from a line.

<!-- team: Theorem 2 is the Sturm-Liouville criterion. It is the same statement the concurrent SSL result reaches for the Gaussian case; we prove the general dichotomy and name the Gaussian as its boundary. Do not over-cite here. -->

### 3.3 The finite world, and the smallest curved chart

The blind spot lives on a *finite* index, so we record the finite-state counterpart explicitly. A coordinate with $r$ distinct latent levels has an $r\times r$ reversible transition matrix with exactly $r$ eigenfunctions. Two facts mirror the continuous dichotomy:

- **Two levels are always straight.** On $r=2$ states every function is affine in the level coordinate, so $\varphi_1$ is affine: a two-state world is the *discrete Gaussian corner*. (This is the smallest world, and a linear probe is complete on it.)
- **Three asymmetric levels already bend.** On $r\ge 3$ states a non-symmetric stationary law gives a $\varphi_1$ that is not affine in the levels — the smallest world with a genuinely curved chart. The cyclic head of the blind spot, with $n\ge 3$ outcomes, is of this kind.

The number of levels $r$ is the **effective rank** of the world, and it is itself recoverable from observed statistics: the moment Hankel matrices of the stationary law close at rank $r$, and the levels and weights are reconstructed from the first $2r$ moments (a classical Prony/Hankel reconstruction; restated and proved in Appendix A.4 so the paper is self-contained). This is the constructive face — one can *estimate the rank of a world*, and hence how far below it a linear probe is reading.

### 3.4 Two registers of one boundary {#sec:registers}

It is tempting, and wrong, to say "the recovery map is the degree-one member of the cumulant tower." We are careful here, because two distinct spectral towers are in play and they coincide only at the Gaussian.

- The **dynamical tower** is the eigenbasis of the transition operator $T$. Its slow member $\varphi_1$ is the recovery map. Affine $\iff$ Gaussian.
- The **distributional tower** is the orthogonal-polynomial / cumulant grading of the stationary law $p$ — the object the gauge program \heb{ל}0–\heb{ל}7 builds and on which the conductor blind spot's Fisher and Amari–Chentsov forms are read. Its degree-one member is *always* affine; its degree-three member is the cubic.

For the Gaussian the two towers are the same object (the Hermite polynomials are simultaneously the transition eigenfunctions and the orthogonal polynomials, with the degree-$d$ mode carrying eigenvalue $\rho^d$). Off the Gaussian they split, and it is exactly that splitting which makes $\varphi_1$ a *mixture* of distributional orders — hence curved, hence invisible to a degree-one probe.

This yields the precise bridge to the blind spot. The conductor blind spot is a statement in the **distributional** register: the order-two Fisher form cannot carry the order-three cubic on a cyclic head. The linear-probe failure of §3.2 is the **dynamical** register: a degree-one reading cannot carry the curved slow eigenfunction off the Gaussian. These two boundaries carry *different* order indices — the distributional one is order two versus three, the dynamical one degree one versus $\ge 2$ — and the offset between them is structural, not loose: the recovery map's leading nonlinear degree is exactly one below the leading nonzero cumulant order. A skew (order-three) world bends $\varphi_1$ quadratically; a symmetric heavy-tailed (order-four) world bends it cubically (§6.4). So the linear-probe boundary (map degree one versus $\ge 2$) and the cumulant boundary (order two versus $\ge 3$) are one boundary read a single step apart. What they genuinely share is a single **trigger** — non-Gaussianity, $\kappa_{\ge 3}\neq 0$ — and a single quantitative law locking them near it (§6.4); calling them "two registers of one boundary" is shorthand for exactly that, not a claim that the two order-counts coincide. They agree precisely where the towers agree: the Gaussian / two-level corner.

**The bridge, measured.** The two registers are not merely concordant in sign; near the Gaussian they are locked by a square law. Tuning a world away from the Gaussian along a one-parameter family, the dynamical curvature $\nu_D = 1-R^2$ of the recovery map scales as the *square of the leading nonzero distributional cumulant*: $\nu_D \approx c\,\tilde\kappa_3^2$ for a skewed world — the order-three, Amari–Chentsov-cubic instance — and $\nu_D \approx c'\,\tilde\kappa_4^2$ for a symmetric one, where $\kappa_3$ vanishes identically and the first occupied rung is order four. Both registers vanish together exactly at the Gaussian; §6.4 measures the constants ($c\approx 0.0555$, $c'\approx 0.0104$) to machine accuracy. This also disciplines the order-three emphasis honestly: on an *unstructured* world order three is the relevant rung only when the world is skewed, but on a *structured* (cyclic) index the order-three object is the Amari–Chentsov *tensor*, whose cross-packet components survive even at the symmetric uniform point — the ring is what keeps order three alive where a scalar skewness would vanish. That is the structural reason the conductor blind spot is an order-three statement at the maximum-entropy point, and the reason the two registers meet there.

<!-- team: §3.4 is the intellectual core and the thing most likely to be garbled by a careless edit. The phrase "two registers of one boundary" is the claim; "recovery order = Hankel rank of the recovery map" is the forbidden conflation. Guard it. The measured-bridge paragraph (E4) is what upgrades this from argued to measured; the order-3-vs-order-4 nuance and the scalar-vs-tensor distinction are load-bearing honesty, not hedging. -->
<!-- team OPPORTUNITY realised: the cross-register bridge experiment flagged in the front-matter log is now E4; the remaining open one (same diagnostic end-to-end on a pretrained head) stays for follow-up. -->

---

## 4. Approximate recovery {#sec:approx}

Exactness is a limit; training reaches its neighbourhood. The next theorem says the neighbourhood is benign and gap-controlled.

**Theorem 3 (Approximate recovery).** *Let $\tilde h = G^{-1/2}h$ be the whitened encoder ($G_{ij}=\mathbb E[h_ih_j]$, $\varepsilon<1$). Suppose* **approximate whitening** *$\|G-I\|_F\le\varepsilon$ and* **approximate alignment** *(the whitened encoder's total Dirichlet energy exceeds the optimum by at most $\delta$). Then, under (G),*
$$\min_{U\in O(n)} \mathbb E\big\|h(z)-U\,\Phi_1(z)\big\|^2 \ \le\ \Big(\varepsilon+\sqrt{\tfrac{2\delta}{\gamma}}\Big)^2.$$

Both terms vanish at $\delta=\varepsilon=0$, recovering Theorem 1; the bound has the shape of a **nonlinearity defect** $2\delta/\gamma$ (how far the learned span leaks out of the slow eigenspace — how curved-beyond-$\Phi_1$ the encoder is) plus a **whitening distortion** $\varepsilon$. The proof (Appendix A.5) is a variational Davis–Kahan ("trace-gap") argument: excess Dirichlet energy $\delta$ forces leakage $\theta^2\le\delta/\gamma$ out of the eigenspace, an orthogonal-Procrustes step converts leakage into a rotation error of at most $2\theta^2$, and a triangle inequality pays the whitening term.

The constant that matters is the **gap $\gamma$ in the denominator**. Recovery is well-posed precisely when the slowest *retained* mode is strictly slower than the fastest *discarded* mode — condition (G) — and the guarantee weakens as that margin closes. The condition is *checkable*, not cosmetic: §6.2 exhibits a perfectly ordinary pair of worlds whose mixing rates differ enough that (G) fails, and the certificate correctly reports the guarantee as undefined there. Two practical corollaries: anisotropic worlds (distinct $\lambda_1^{(i)}$) enjoy *larger* $\gamma$ and signed-permutation identifiability at once; and the recovery map itself, once $\Phi_1$ is to be estimated from finite data rather than known, inherits a second Davis–Kahan bound on the finite reconstruction of §3.3 (Appendix A.4).

---

## 5. Planning: the chart is a faithful coordinate for control {#sec:planning}

Identifiability is worth little if the recovered coordinate is useless for acting. It is not: because $\Phi_1$ is a diffeomorphism, *any* control problem on the true latent transports through it without loss.

Fix a finite-horizon Markov decision problem on the latent: controlled kernels $P_t(dz'\mid z,a)$, stage costs $c_t(z,a)$, terminal cost $c_T$, value $V^\star_t$, optimal policy $a^\star_t$. The agent, however, only ever sees the embedding $\widehat z=\psi(z)$ where $\psi:=U\Phi_1$ is the recovered chart (a diffeomorphism: each $\varphi_1^{(i)}$ is strictly monotone where $p_i>0$, and $U$ is invertible). Define the **pushforward problem** on the embedding by carrying kernels and costs through $\psi$: $\widehat P_t = \psi_\# P_t$, $\widehat c_t = c_t\circ\psi^{-1}$.

**Proposition 1 (Exact planning equivalence).** *For all $t$ and $z$,*
$$\widehat V^\star_t\big(\psi(z)\big) = V^\star_t(z), \qquad \widehat a^\star_t\big(\psi(z)\big) = a^\star_t(z).$$

This is a change of variables, not a deep fact: it is the covariance of dynamic programming under the state-space diffeomorphism $\psi$ — one line of the Bellman recursion (Appendix A.6), exact for *any* diffeomorphism, Gaussian or not. Its weight is in what it licenses — **planning in the only space the agent has, its representation, is exactly planning in the world it cannot see** — and in two corollaries that place the regimes. *Native-cost:* if the objective is stated on the representation itself, no knowledge of $\psi$ is needed and planning is exact. *Invariant-cost (the Gaussian corner):* if the objective is stated on the latent and is $\psi$-invariant, it is read directly off the embedding; when $\Phi_1=\mathrm{id}$ this is invariance under the recovered rotation, the familiar zero-correction case.

The substantive result is what survives when recovery is only *approximate* — the regime training actually reaches, and the one where the recovery bound of §4 enters.

**Theorem 4 (Approximate planning).** *Suppose recovery is within the bound of Theorem 3 — $\mathbb E\|h-\psi\|^2 \le \eta^2$ with $\eta = \varepsilon+\sqrt{2\delta/\gamma}$ — the latent value functions $V^\star_t$ are $L$-Lipschitz, and the chart $\psi$ is bi-Lipschitz. Then the policy $\hat\pi$ obtained by planning in the learned chart and acting via $\widehat a^\star_t(h(z))$ incurs true-world regret*
$$\mathbb E\big[V^{\hat\pi}_0(z) - V^\star_0(z)\big] \;\le\; C\,L\,T\,\eta,$$
*for an absolute constant $C$ (absorbing the chart's bi-Lipschitz constant). The plan degrades linearly in the recovery RMS error $\eta$ and the horizon $T$, and vanishes in the exact limit $\eta\to 0$, recovering Proposition 1.*

The proof (Appendix A.6) is a performance-difference argument: the recovery error $\eta$ is a state-misidentification of size $O(\eta)$, which an $L$-Lipschitz value propagates across $T$ steps. The honest content, again: off the Gaussian the warp $\Phi_1$ is *not* the identity and must be applied — but it is *known* (recoverable, §3.3), so planning degrades from "free" to "one known change of variables away," never to "lost."

---

## 6. Certificates {#sec:experiments}

Every claim above is checked by a self-contained, deterministic, CPU-second program in [`empirical/apex_recovery/`](https://github.com/leomurillo/AI-ConductorBlindSpot/tree/main/empirical/apex_recovery) (shared toolkit [`apex_world.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/apex_world.py); one-command [`run_all.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/run_all.py) with a self-check gate). There is no machine learning in them: the theory collapses the learning problem to one self-adjoint operator's eigenproblem, so the certificates are exact linear algebra. We summarise; full protocol in Appendix C.

### 6.1 Recovery and the Gaussian boundary (E1)

For four worlds we build the reversible transition operator, diagonalise it exactly, and measure the nonlinearity $\nu = 1-R^2$ of the best affine fit of the slow eigenfunction $\varphi_1$:

| world | $\lambda_1$ | $\varphi_1$ monotone | nonlinearity $\nu$ | linear-probe $R^2$ |
|---|---|---|---|---|
| gaussian | 0.9998 | yes | **0.000** | 1.000 |
| uniform | 1.0000 | yes | 0.014 | 0.986 |
| bimodal | 1.0000 | yes | 0.068 | 0.932 |
| laplace | 0.9999 | yes | 0.131 | 0.869 |

The Gaussian alone has $\nu=0$ (affine $\varphi_1$, linear-probe $R^2=1$); every other world has $\nu>0$, and every $\varphi_1$ is monotone, as Sturm–Liouville requires. The known-ground-truth check confirms the spine: in the cube-root world $z=g^3$ the recovered $\varphi_1$ matches $\sqrt[3]{z}$ to discretisation accuracy ($\approx 2.5\times 10^{-3}$ in $L^2(p)$), while a linear probe of $z$ recovers $\sqrt[3]{z}$ at only $R^2=0.61$. A mixing-invariance check confirms recovery is intrinsic to the operator: an order-reversing relabelling of states returns the same $\varphi_1$ to machine precision.

![The recovered slow eigenfunction $\varphi_1(z)$ is a straight line **only** for the Gaussian world; for every non-Gaussian law it is monotone but curved, by the nonlinearity $\nu = 1-R^2$ of its best affine fit (legend). This is the linear-versus-nonlinear identifiability boundary made visible: a linear probe of the representation is complete exactly where this curve is straight, and provably lossy in proportion to its bend.](empirical/apex_recovery/reports/e1_eigenfunctions.png){width=78%}

### 6.2 The bound, and the reality of the gap (E2)

On a product world we verify Theorem 3 directly. Across a sweep of leakage and whitening perturbations, the measured recovery error sits under $(\varepsilon+\sqrt{2\delta/\gamma})^2$ in **15 of 15** trials, with slack that stays of the order of the error (the bound is not vacuous) and vanishes at $\delta=\varepsilon=0$. The product-spectrum check confirms (G): the two slowest joint modes are exactly the two single-coordinate slow modes (leakage $0$ to machine precision), with $\gamma=0.0224$. A deliberately heterogeneous pair (a fast and a slow coordinate) is shown to *violate* (G) — the slow coordinate's second mode outranks the fast coordinate's first — and the certificate reports the guarantee as undefined there, exactly as it should. The gap is a checkable hypothesis, not a decoration.

![Approximate recovery (Theorem 3): across a sweep of whitening ($\varepsilon$) and alignment ($\delta$) perturbations, every measured recovery error $\min_U\|h-U\Phi_1\|^2$ lies on or under the guarantee $(\varepsilon+\sqrt{2\delta/\gamma})^2$ (dashed line; colour encodes $\varepsilon$). The bound vanishes with the perturbations and is never vacuous.](empirical/apex_recovery/reports/e2_bound_scatter.png){width=52%}

### 6.3 The finite register, and the blind-spot counts (E3)

In exact rational arithmetic we exhibit the distributional tower of §3.4: a rank-three world's moment Hankel determinants are $1,5,81,0$, closing at rank $3$; the Prony/Hankel reconstruction recovers its levels $\{-2,1,4\}$ exactly from the first six moments; and a two-level world is shown to be the affine (Gaussian-corner) case. The certificate then re-counts, from the distributional side, the cross-packet cubic triples of the cyclic head for $n\in\{6,8,12,18,30\}$:
$$18,\quad 42,\quad 108,\quad 252,\quad 774,$$
**reproducing exactly** the conductor blind spot's published Section-4 counts. The same order-three object that the blind spot certifies finitely is the order-three rung whose continuous shadow bends $\varphi_1$ in §6.1 — the two registers, meeting on one set of integers.

<!-- team: §6.3 is the hinge. The fact that an independent script in THIS paper regenerates CBS's counts is the single most convincing cross-paper artifact we have. Lead the experiments section with it in talks. -->

### 6.4 The bridge, measured (E4)

This certificate turns §3.4 from an argued correspondence into a measured one. Along a skew family interpolating from the Gaussian, the dynamical curvature $\nu_D$ and the distributional skewness $\tilde\kappa_3$ vanish together and rise locked by
$$\nu_D/\tilde\kappa_3^2 \;\longrightarrow\; 0.0555 \quad(\text{stable to }1.3\%\text{ across the near-Gaussian range}),$$
computed by Gauss–Hermite quadrature in the continuous (Ornstein–Uhlenbeck) limit, where the recovery map is exactly the inverse warp. Along a symmetric family the skewness is zero to machine precision ($\sim 10^{-17}$) while $\nu_D/\tilde\kappa_4^2 \to 0.0104$ — the recovery map still bends, but the first occupied rung is order four. The finite coda contrasts the two faces of order three: the cyclic label law has scalar skewness exactly zero, yet its Amari–Chentsov cross-packet cubic is nonzero (the counts of §6.3), so the structured index keeps order three alive where the scalar vanishes. The dynamical and distributional registers are one boundary — now with the constant of proportionality on the table.

![The cross-register bridge, measured. The recovery-map curvature $\nu_D$ scales as the square of the leading distributional cumulant — skewness$^2$ for the skew family (left, order three) and excess-kurtosis$^2$ for the symmetric family (right, order four) — both vanishing with it at the Gaussian. The dashed line is the near-Gaussian slope; points fall below it at large departure (the square law is the leading-order statement).](empirical/apex_recovery/reports/e4_bridge.png){width=92%}

<!-- team: E4 is the genuinely NEW empirical content of this revision. It both strengthens §3.4 (square law, measured constants) and surfaces the honest order-3/order-4 + scalar/tensor nuance. If a reviewer asks "is the CBS link real or rhetorical?", §6.4 is the answer. -->

### 6.5 On a real model: the bridge is a middle-layer phenomenon (E5)

E4 is the continuous-limit twin of a question we can ask of an actual pretrained model: take the residual stream of a small language model (Pythia-70m), treat each coordinate of a layer as a "world", form positive pairs from adjacent token positions, estimate the slow eigenfunction of that token-to-token transition, and ask whether its curvature $\nu_D$ tracks the coordinate's distributional cumulants. Two readings emerge, and we report both.

First, the **weak bridge** holds at every layer whose transition operator is estimable: the least non-Gaussian directions have near-floor curvature — a Gaussian direction's recovery map is affine regardless of depth, the real-model image of §3.4's "agree at the Gaussian corner." Second, the **strong** cross-direction link — marginal non-Gaussianity predicting dynamical curvature across the 512 directions — is *not* uniform: it is a **middle-layer phenomenon**. The rank correlation between $\nu_D$ and the leading-cumulant$^2$ rises to $+0.41$ at the centre of the network (layers 3–5) and is absent ($\approx 0$) at the embedding, the earliest, and the final layer. At the peak layer the two ends of the boundary are concrete: the most-Gaussian direction has an affine slow eigenfunction ($\nu_D=0.03$), while a heavy outlier dimension (skewness $11.8$, excess kurtosis $142$) has a sharply curved one ($\nu_D=0.51$) — a linear probe reads the first whole and the second only in part.

This depth dependence is exactly what the two-register picture predicts. The registers are *distinct* objects off the Gaussian (§3.4); a coordinate can be skewed in its marginal yet affine in its slow dynamics if the non-Gaussianity lives in fast modes. They couple — marginal shape showing up in the slow eigenfunction — only where the slow dynamics carry the structure, which is the middle of the network, where representation is built, and not at the token-bound embedding or the output-shaped final layer. We do not force the clean E4 constant onto real activations; the depth profile, with its estimable-operator floor and its middle-layer peak, is the result. It is an observational certificate — estimated operators, one model — and it shows the boundary the theory predicts is visible, unforced, in a pretrained network.

![The bridge on a pretrained model (Pythia-70m). *Left:* the rank correlation between the slow-eigenfunction curvature $\nu_D$ and the leading-cumulant$^2$ across the 512 residual-stream directions is a **middle-layer phenomenon**, rising to $+0.41$ at the centre and vanishing at the embedding, earliest, and final layers. *Centre:* at the peak layer, $\nu_D$ rises above the noise floor with $|$skewness$|$. *Right:* the two ends of the boundary — a near-Gaussian direction has an affine recovery map (straight), a heavy outlier dimension a sharply curved one.](empirical/apex_recovery/reports/e5_real_model_bridge.png){width=100%}

<!-- team: E5 is the on-a-real-model win, and its honesty is the point. We do NOT claim a universal real-model square law; we claim (a) the weak bridge everywhere, (b) a structured middle-layer strong bridge that the two-tower theory predicts. The inverted-V depth profile is more convincing than a flat correlation precisely because it has interpretable structure and vanishes where the operator is not estimable (embedding floor 0.18). If a reviewer pushes on generality: one model, observational, depth-resolved; the continuous-limit exact twin is E4. Follow-ups: more models/sizes, augmentation-pairs instead of adjacent-token pairs, and a matched linear-vs-eigenfunction probe head-to-head on the middle-layer rogue dims. -->

---

## 7. Consequences for practice {#sec:consequences}

The dichotomy is not only structural; it has operational teeth for how learned representations are read, trained, and trusted.

**Interpretability and explainability.** Linear probing is the default instrument for asking what a representation encodes. Theorem 2 says that instrument is *complete only on the Gaussian stratum*: wherever the underlying factors are non-Gaussian — which is to say, wherever the world has genuine higher-order structure — a linear probe reads a strict shadow of what the representation has identified, and the missing part is precisely the curvature of $\varphi_1$. The remedy is not a bigger probe but the *right* one: the recovered chart is nonlinear and known, so a probe matched to $\Phi_1$ recovers what a linear probe cannot. This is visible directly in a pretrained model (§6.5): at the middle layers, the most non-Gaussian residual-stream directions — the heavy outlier dimensions known to dominate transformer activations — are exactly the ones whose recovered slow feature is curved, so a linear read of them is provably partial. This also explains, without appeal to optimization pathology, why linear-probe audits can under-report capability on structured tasks.

**Optimization.** The order-two ceiling is the same one the conductor blind spot draws for curvature preconditioners. A second-order (covariance) model of a representation — the Fisher/Gauss–Newton class — is matched to the Gaussian corner; off it, the curvature it cannot see is the same distributional order-three content. The constructive corollary is the gap $\gamma$: it is an estimable diagnostic of *how well-posed* slow-feature recovery is for a given encoder and task, and a small $\gamma$ is an early warning that the representation's slow structure is nearly degenerate.

**Auditing and the governance of world models.** Identifiability is the property a regulator or safety auditor most wants from a learned world model: that the representation corresponds to the world's real degrees of freedom rather than scrambling them. This paper sharpens what can be promised. Identifiability *does* hold for non-Gaussian worlds — the representation recovers the true factors — but only **up to a known nonlinear chart and a block rotation**, and only when the slow-feature gap (G) holds. An audit that assumes *linear* correspondence will mis-certify a perfectly identified non-Gaussian representation as entangled; an audit that checks the *gap* and reconstructs the *chart* will not. The planning theorem makes the stakes concrete: a controller that plans in the representation acts in the world exactly, provided the known warp is applied — and silently sub-optimally if it is assumed away.

**Anisotropy as a design lever.** Theorem 1's fine print is an opportunity: distinct mixing rates across factors turn the rotation ambiguity into per-coordinate identification. A learner or augmentation scheme that *induces* anisotropy buys disentanglement without an auxiliary objective — a direction we flag for separate study.

**Grokking, as a prediction (to be tested).** The sharpest forward use of the two-register picture is dynamical. On the canonical delayed-generalisation task — modular addition $a+b\equiv c \pmod p$ — the generalising "Fourier circuit" is an order-three object: it computes via the character selection rule $k+\ell+m\equiv 0\pmod p$, the same Amari–Chentsov cubic the conductor blind spot names, while the memorising solution that precedes it is order-two-sufficient (a lookup with no ring structure). Grokking would then be the order-two $\to$ order-three boundary of this paper *crossed during training*: we predict that at the generalisation step the distributional register (the head becoming ring-additive, depending on $a+b$ alone) and the dynamical register (the token embedding aligning to its slow eigenfunctions — circularising onto the character chart) rise **together**, both tracking validation accuracy and both flat on a structure-free control. A partial run reaches only the pre-grokking phase — memorisation complete, validation at chance, both registers resting at the order-two baseline, exactly as the picture requires — but does not yet reach the transition; instrumenting the in-training co-emergence is the experiment a follow-up will run. The prediction is falsifiable in a strong sense: were the two registers to rise at *different* times, the single-trigger reading would be wrong.

<!-- team: §7 is where the AI-development value lives and where the paper earns attention without leaning on any name. Expand the auditing/regulation paragraph if a policy-adjacent venue is targeted. -->

---

## 8. Scope and limitations {#sec:scope}

The theorems are conditional and we state the conditions plainly. **Reversibility** (detailed balance) buys the self-adjoint spectral theorem; genuinely non-reversible additive noise requires the symmetric part and is left to follow-up. **Discrete spectrum** (compact $T_i$) holds for Ornstein–Uhlenbeck and confining Langevin generators; heavy-tailed or unconfined laws may add continuous spectrum. The **slow-feature gap (G)** is a real hypothesis, checkable and sometimes false (§6.2). Recovery is **up to a nonlinear chart**: this is the correct and honest statement, not a weakness — claiming linear recovery off the Gaussian would be false. Theorem 4 (approximate planning) assumes **Lipschitz value functions and a bi-Lipschitz chart**; its constant, given in Appendix A.6, carries the chart's bi-Lipschitz modulus and is finite away from the support boundary. Finally, the bridge of §3.4 is a statement about *model classes and registers*, not a claim that a single diagnostic has been run end-to-end across both; §6 checks each register on its own ground, and the cross-register experiment flagged in the internal log is the natural next step.

---

## 9. Conclusion {#sec:conclusion}

The conductor blind spot showed, on a clock, that an order-two instrument cannot read an order-three fact. This paper showed the same boundary in the continuous, learned setting: alignment-and-whitening recovers the slow-eigenfunction chart of a world; that chart is a straight line exactly for the Gaussian and a known curve otherwise; recovery is exact up to a block rotation, robust by a gap-controlled bound, and faithful for planning under a known change of variables. The Gaussian corner — where a linear probe is complete and the warp is the identity — is one stratum of a theory whose general statement is nonlinear, and whose limits and licenses we have tried to state without flinching. The finite and the continuous, the distributional and the dynamical, meet on the same integers; the practical lesson is to read a representation through the chart it actually learned, and to certify the gap that makes that chart well-posed.

---

## Appendix A. Proofs {#sec:appA}

### A.1 Alignment as a Rayleigh form {#sec:A1}

With $\{h_i\}$ whitened and $T$ self-adjoint, stationarity gives $\mathbb E[h_i(z)^2]=\mathbb E[h_i(z')^2]=1$ and, by the definition of $T$ as conditional expectation, $\mathbb E[h_i(z')h_i(z)] = \mathbb E[h_i(z)\,\mathbb E[h_i(z')\mid z]] = \langle h_i, Th_i\rangle$. Hence
$$\mathbb E\|h(z')-h(z)\|^2 = \sum_i\big(\mathbb E[h_i(z')^2]+\mathbb E[h_i(z)^2]-2\mathbb E[h_i(z')h_i(z)]\big) = \sum_i 2\big(1-\langle h_i,Th_i\rangle\big).\qquad\square$$

### A.2 Ky Fan and the product spectrum {#sec:A2}

*Ky Fan maximum principle.* For self-adjoint compact $T$ and orthonormal $\{h_i\}_{i=1}^n$ in the mean-zero subspace, $\max\sum_i\langle h_i,Th_i\rangle$ equals the sum of the $n$ largest non-trivial eigenvalues of $T$, attained iff $\mathrm{span}\{h_i\}$ is the corresponding top eigenspace; the maximiser is unique up to orthogonal mixing within degenerate blocks (Fan 1949).

*Product spectrum.* The spectrum of $T=\bigotimes_iT_i$ is $\{\prod_i\lambda_{k_i}^{(i)}\}$ with eigenfunctions $\bigotimes_i\varphi_{k_i}^{(i)}$. Under (G), the $n$ largest non-trivial eigenvalues are the single-excitations $\{\lambda_1^{(i)}\}$ with eigenfunctions $\varphi_1^{(i)}(z_i)$: any competitor is a higher single mode ($\le\lambda_2^{(i)}$) or a multi-excitation ($\le\lambda_1^{(i)}\lambda_1^{(j)}$), both strictly dominated by (G). Combining with Ky Fan gives Theorem 1, with the equal-$\lambda_1$ degeneracy as the only freedom in $U$. $\square$

### A.3 Affine recovery iff Gaussian {#sec:A3}

In the reversible diffusion realisation $T_i=e^{\tau L_i}$ with $L_i\psi = D(\psi''+(\log p_i)'\psi')$, sharing eigenfunctions with $T_i$. If $\varphi_1(z)=az+b$, then $L_i\varphi_1 = aD(\log p_i)'$, and $L_i\varphi_1=-\mu\varphi_1$ forces $(\log p_i)'(z) = -\tfrac{\mu}{aD}(z+b/a)$, so $\log p_i$ is quadratic and $p_i$ Gaussian. Conversely the Gaussian gives Ornstein–Uhlenbeck, whose first eigenfunction is $\varphi_1(z)=z$. $\square$

### A.4 Finite rank and Hankel reconstruction (restated in-house) {#sec:A4}

<!-- team: this is the T12 material RESTATED so we never cite the unpublished paper. Keep self-contained. -->

Let a coordinate take $r$ distinct values $\lambda_1,\dots,\lambda_r$ with weights $w_j>0$, $\sum w_j=1$, and write $\mu_k=\sum_j w_j\lambda_j^k$.

**(Rank closure.)** The Hankel matrices $H_N=[\mu_{a+b}]_{0\le a,b\le N}$ satisfy $\det H_N>0$ for $N<r$ and $\det H_r=0$: the moment sequence has exact rank $r$. (Standard for an $r$-atomic measure; the positive-definiteness is Hamburger's condition.)

**(Reconstruction.)** The levels are the roots of the degree-$r$ Prony polynomial whose coefficients solve the Hankel system $[\mu_{a+b}]_{0\le a,b<r}\,c = -[\mu_r,\dots,\mu_{2r-1}]^\top$; the weights then solve a Vandermonde system. Hence $r$ levels and weights are determined by $\mu_0,\dots,\mu_{2r-1}$ — the first $2r$ moments. $\square$

This is the *distributional* tower (§3.4): the orthogonal polynomials of $p_i$ are fixed by the same Hankel data and form a tridiagonal (Jacobi) recurrence. It reconstructs the **world's law and effective rank**, not the transition eigenfunction; the two coincide only at the Gaussian.

### A.5 The approximate bound (trace-gap Davis–Kahan) {#sec:A5}

Work with $A=I-T$ on the mean-zero subspace, eigenvalues $a_k=1-\lambda_k$; let $V_n$ be the bottom-$n$ eigenspace (top-$n$ of $T$), $P$ its projector, and $\theta^2=\|P^\perp\Pi_W\|_{HS}^2$ the leakage of $W=\mathrm{span}(\tilde h)$ out of $V_n$. The gap is $\gamma=a_{n+1}-a_n$.

**Lemma (trace-gap).** $\mathrm{tr}(A\Pi_W)\ge \sum_{i\le n}a_i + \gamma\theta^2$.

*Proof.* Split $\tilde h_i=u_i+v_i$, $u_i=P\tilde h_i$. The discarded part gives $\sum_i\langle v_i,Av_i\rangle\ge a_{n+1}\theta^2$. For the retained part, $\Sigma=P\Pi_WP$ satisfies $0\preceq\Sigma\preceq I_{V_n}$, $\mathrm{tr}\Sigma=n-\theta^2$, and minimising $\mathrm{tr}(A|_{V_n}\Sigma)$ fills the smallest eigenvalues: $\sum_i\langle u_i,Au_i\rangle\ge\sum_{i\le n}a_i - a_n\theta^2$. Add and use $a_{n+1}-a_n\ge\gamma$. $\square$

Approximate alignment ($\mathrm{tr}(A\Pi_W)\le\sum_{i\le n}a_i+\delta$) then gives $\theta^2\le\delta/\gamma$. An orthogonal-Procrustes step (Appendix, with $C_{ij}=\langle\tilde h_i,\varphi_1^{(j)}\rangle$, singular values $\le 1$, $\sum\sigma_k^2=n-\theta^2$) yields $\min_U\|\tilde h-U\Phi_1\|^2\le 2\theta^2\le 2\delta/\gamma$. Finally $\|h-\tilde h\|^2=\sum_k(1-\sqrt{g_k})^2\le\|G-I\|_F^2\le\varepsilon^2$, and the triangle inequality gives $\min_U\|h-U\Phi_1\|\le\varepsilon+\sqrt{2\delta/\gamma}$. Square. $\square$

### A.6 Planning equivalence {#sec:A6}

**Proposition 1 (exact).** Backward induction on $t$. Base: $\widehat V^\star_T(\psi(z))=\widehat c_T(\psi(z))=c_T(z)=V^\star_T(z)$. Step: assume $\widehat V^\star_{t+1}(\psi(z'))=V^\star_{t+1}(z')$. By the pushforward identity $\int g(\widehat z')\widehat P_t(d\widehat z'\mid\psi(z),a)=\int g(\psi(z'))P_t(dz'\mid z,a)$ with $g=\widehat V^\star_{t+1}$, and $\widehat c_t(\psi(z),a)=c_t(z,a)$,
$$\widehat V^\star_t(\psi(z)) = \min_a\Big\{c_t(z,a)+\int V^\star_{t+1}(z')P_t(dz'\mid z,a)\Big\} = V^\star_t(z),$$
with identical minimands in $a$, so the optimal actions coincide. The native- and invariant-cost corollaries are immediate: if the objective is given on $\widehat z$ no $\psi$ is needed; if $c\circ\psi^{-1}=c$ it is evaluated on $\widehat z$ directly, reducing at $\Phi_1=\mathrm{id}$ to rotation-invariance. $\square$

**Theorem 4 (approximate).** The agent decodes its observation to a believed latent $\tilde z := \psi^{-1}(h(z))$ and executes the action $a^\star_t(\tilde z)$ optimal at $\tilde z$. With $\psi$ bi-Lipschitz, $\|\psi^{-1}(u)-\psi^{-1}(v)\|\le\kappa\|u-v\|$, so the state-decoding error obeys, by Cauchy–Schwarz and the Theorem 3 bound,
$$\mathbb E\|\tilde z - z\| = \mathbb E\big\|\psi^{-1}(h(z))-\psi^{-1}(\psi(z))\big\| \le \kappa\,\mathbb E\|h(z)-\psi(z)\| \le \kappa\big(\mathbb E\|h-\psi\|^2\big)^{1/2} \le \kappa\,\eta.$$
By the performance-difference lemma (Kakade–Langford 2002), the regret of executing, at each step, the action optimal for a state estimate that is off by $\Delta_t$ is at most $2\sum_{t} \mathrm{Lip}(V^\star_t)\,\mathbb E\|\tilde z_t - z_t\|$; with each $V^\star_t$ $L$-Lipschitz and the per-step decoding error $\le\kappa\eta$ over horizon $T$,
$$\mathbb E\big[V^{\hat\pi}_0(z)-V^\star_0(z)\big] \le 2\kappa\,L\,T\,\eta =: C\,L\,T\,\eta .$$
At $\eta=0$ the bound is zero and $\tilde z=z$, recovering Proposition 1. $\square$

*Remark.* The constant $C=2\kappa$ carries the chart's bi-Lipschitz modulus, which is finite wherever the stationary density is bounded away from $0$ on the planning region (so $\varphi_1^{(i)}$ has positive, bounded derivative there); the bound is vacuous only as that region approaches a support boundary, exactly where the chart itself degenerates.

---

## Appendix B. The conductor blind spot as the finite, distributional register {#sec:appB}

We make §3.4 explicit. The conductor blind spot studies the categorical model at the uniform point $p_*$ on $\mathbb Z/n\mathbb Z$: the Fisher form is diagonal in the additive-character basis, block-diagonal across conductor packets, while the Amari–Chentsov cubic couples packets under $k+\ell+m\equiv 0\pmod n$. These are statements about the **stationary law's** local geometry — the distributional tower — read at order two and order three.

The present paper's recovery map is the **dynamical** slow eigenfunction. At the Gaussian/uniform corner the two towers coincide (Hermite simultaneously diagonalises the Ornstein–Uhlenbeck transition and orthogonalises the Gaussian), and the degree-$d$ rung carries transition eigenvalue $\rho^d$; the order-three cubic is then the degree-three eigenspace, eigenvalue $\rho^3$, the first rung below the recovered degree-one mode. Off the corner the towers split: the cubic remains a distributional order-three object, while the dynamical $\varphi_1$ becomes a mixture of distributional orders and bends.

The certificate of §6.3 lives entirely on the distributional side and regenerates the blind spot's cross-packet cubic counts $18,42,108,252,774$ for $n\in\{6,8,12,18,30\}$, confirming that the order-three object this paper's continuous theory points to is, on the finite cyclic index, exactly the one the blind spot certifies. The two papers therefore share an order-three witness: finite and exact there, continuous and learned here.

**Scalar versus tensor: why the ring is special.** One subtlety is worth stating plainly, because the measured bridge of §6.4 exposes it. The order-three content of an *unstructured* one-dimensional world is the scalar third cumulant (the skewness), and it vanishes whenever the law is symmetric — there the recovery map first bends at order four. The conductor blind spot's order-three object is not a scalar but the Amari–Chentsov *tensor* on the tangent space, and its cross-packet components are nonzero at the symmetric maximum-entropy point precisely because the cyclic index supplies the arithmetic that survives the symmetry (the selection rule $k+\ell+m\equiv 0\pmod n$ has solutions with unequal conductors). So the ring structure is exactly what keeps order three occupied where a scalar skewness would be silent. This is not a discrepancy between the registers; it is the reason the blind spot is an order-three statement at $p_*$ at all, and it is measured directly in §6.4 (scalar skewness zero, tensor cubic nonzero, on the same $n$).

---

## Appendix C. Experimental protocol and reproduction {#sec:appC}

All code lives in [`empirical/apex_recovery/`](https://github.com/leomurillo/AI-ConductorBlindSpot/tree/main/empirical/apex_recovery) of the companion repository; running [`run_all.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/run_all.py) there reproduces every number (use the figure-capable interpreter to also render the `.png` figures). The runner asserts the headline self-checks — the approximate bound holds in every trial; the finite reconstructions are exact; the blind-spot counts match; the cross-register square law is stable and positive — and exits nonzero on any failure.

**Worlds.** Each world is a reversible nearest-neighbour Metropolis chain on a grid — the exact, fully diagonalisable discrete counterpart of the Ornstein–Uhlenbeck transition. The transition operator is symmetrised in the $\pi$-weighted inner product and diagonalised by a symmetric eigensolver; eigenfunctions are $L^2(p)$-orthonormal. There is no randomness in the operators; the only seed is the contamination sweep of E2.

**E1** ([`e1_eigenfunction_recovery.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e1_eigenfunction_recovery.py)) builds four worlds, measures $\varphi_1$'s nonlinearity, monotonicity, and linear-probe $R^2$, runs the cube-root ground-truth check and the mixing-invariance check. **E2** ([`e2_approximate_bound.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e2_approximate_bound.py)) builds product worlds, verifies (G) and the product spectrum, sweeps leakage and whitening to certify the bound, and exhibits a (G)-violating pair. **E3** ([`e3_hankel_reconstruction.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e3_hankel_reconstruction.py)) does the finite, exact-rational distributional tower: Hankel rank closure, Prony reconstruction, the two-level corner, and the cyclic cross-packet cubic counts. **E4** ([`e4_cross_register_bridge.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e4_cross_register_bridge.py)) measures §3.4: by Gauss–Hermite quadrature it sweeps a skew family and a symmetric family from the Gaussian and certifies the square law $\nu_D \propto (\text{leading cumulant})^2$ (order three skewed, order four symmetric), closing with the scalar-skewness-zero versus tensor-cubic-nonzero contrast on the ring.

**E5 (optional, real model)** ([`e5_real_model_bridge.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e5_real_model_bridge.py)) is the on-a-real-model counterpart of E4 and the only experiment needing `torch`+`transformers` and a cached checkpoint. It runs Pythia-70m once over a bundled real-text corpus, collects residual activations at every layer, treats each coordinate as a world, estimates the symmetrised adjacent-token transition per direction, and reports the depth profile of the rank correlation between $\nu_D$ and the leading-cumulant$^2$, together with the per-layer noise floor and two example slow-eigenfunction curves at the peak layer. It is observational (estimated operators) and is not part of the pass/fail gate; the depth profile (a middle-layer peak; an embedding-layer floor where the operator is not estimable) is the reported result. Runs on CPU in under a minute, fully offline from the HF cache.

**Environment.** `numpy`, `scipy`, `sympy` (E3 exact arithmetic); `matplotlib` optional (figures). No GPU, no network; runtime a few seconds. Tested on Python 3.12 and 3.13.

---

## Code and data availability {#sec:code-and-data}

All code and the checked-in run artifacts (JSON certificates and figures) are in the companion repository:

* **GitHub:** <https://github.com/leomurillo/AI-ConductorBlindSpot>, folder [`empirical/apex_recovery/`](https://github.com/leomurillo/AI-ConductorBlindSpot/tree/main/empirical/apex_recovery) (see its [`README.md`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/README.md)).

The repository mirrors the names used here. The exact recovery and Gaussian-boundary certificate of §6.1 is [`e1_eigenfunction_recovery.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e1_eigenfunction_recovery.py); the approximate-bound certificate of §6.2, [`e2_approximate_bound.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e2_approximate_bound.py); the exact-arithmetic distributional tower and blind-spot counts of §6.3, [`e3_hankel_reconstruction.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e3_hankel_reconstruction.py); the measured cross-register square law of §6.4, [`e4_cross_register_bridge.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e4_cross_register_bridge.py); the real-model depth profile of §6.5, [`e5_real_model_bridge.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e5_real_model_bridge.py); the shared toolkit, [`apex_world.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/apex_world.py); and the one-command runner with its self-check gate, [`run_all.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/run_all.py). The grokking testbed of the §7 forward prediction is [`e6_grokking_bridge.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e6_grokking_bridge.py); its in-training instrumentation is left to follow-up work. Run outputs cited in the text are the checked-in files under [`empirical/apex_recovery/reports/`](https://github.com/leomurillo/AI-ConductorBlindSpot/tree/main/empirical/apex_recovery/reports).

---

## References {#sec:references}

<!-- team: T0-T7 and CBS [C1] DOIs are exact and hyperlinked (T7's reference-list format). CBS DOI 10.5281/zenodo.20330864 supplied by author. Trim/expand external refs at polish. -->

[1] Fan, K. (1949). *On a theorem of Weyl concerning eigenvalues of linear transformations.* Proceedings of the National Academy of Sciences, 35(11), 652–655.

[2] Davis, C., & Kahan, W. M. (1970). *The rotation of eigenvectors by a perturbation. III.* SIAM Journal on Numerical Analysis, 7(1), 1–46.

[3] Wiskott, L., & Sejnowski, T. J. (2002). *Slow feature analysis: Unsupervised learning of invariances.* Neural Computation, 14(4), 715–770.

[4] Coifman, R. R., & Lafon, S. (2006). *Diffusion maps.* Applied and Computational Harmonic Analysis, 21(1), 5–30.

[5] Karlin, S., & McGregor, J. (1959). *Random walks.* Illinois Journal of Mathematics, 3(1), 66–81.

[6] Hyvärinen, A., & Pajunen, P. (1999). *Nonlinear independent component analysis: Existence and uniqueness results.* Neural Networks, 12(3), 429–439.

[7] Amari, S., & Nagaoka, H. (2000). *Methods of Information Geometry.* Translations of Mathematical Monographs, Vol. 191. American Mathematical Society / Oxford University Press.

[8] Bakry, D., Gentil, I., & Ledoux, M. (2014). *Analysis and Geometry of Markov Diffusion Operators.* Grundlehren der mathematischen Wissenschaften, Vol. 348. Springer.

[9] Levin, D. A., & Peres, Y. (2017). *Markov Chains and Mixing Times* (2nd ed.). American Mathematical Society.

[10] Schmüdgen, K. (2017). *The Moment Problem.* Graduate Texts in Mathematics, Vol. 277. Springer.

[11] Rockafellar, R. T. (1970). *Convex Analysis.* Princeton University Press.

[12] Klindt, D., LeCun, Y., & Balestriero, R. (2026). *When does LeJEPA learn a world model?* arXiv preprint arXiv:2605.26379.

[13] Kakade, S., & Langford, J. (2002). *Approximately optimal approximate reinforcement learning.* In *Proceedings of the 19th International Conference on Machine Learning* (ICML 2002), pp. 267–274.

[C1] Murillo Montero, L. (2026). *The Conductor Blind Spot of the Quadratic Curvature Class on Ring-Structured Categorical Heads.* Zenodo preprint, DOI [10.5281/zenodo.20330864](https://doi.org/10.5281/zenodo.20330864).

[\heb{ל}0] Murillo Montero, L. (2026a). *\heb{ל}0 — Gauge Selection and the Fourier–Mellin Simplex.* Zenodo preprint (concept DOI, resolves to latest version), DOI [10.5281/zenodo.20423985](https://doi.org/10.5281/zenodo.20423985).

[\heb{ל}1] Murillo Montero, L. (2026b). *\heb{ל}1 — Identifying 3: The Native Simplicial Chart.* Zenodo preprint, DOI [10.5281/zenodo.20336872](https://doi.org/10.5281/zenodo.20336872).

[\heb{ל}2] Murillo Montero, L. (2026c). *\heb{ל}2 — Identifying 3: The Cumulant Tower as Effective Action.* Zenodo preprint, DOI [10.5281/zenodo.20338126](https://doi.org/10.5281/zenodo.20338126).

[\heb{ל}3] Murillo Montero, L. (2026d). *\heb{ל}3 — Identifying 3: Character Orthogonality on the Conductor Lattice.* Zenodo preprint, DOI [10.5281/zenodo.20339049](https://doi.org/10.5281/zenodo.20339049).

[\heb{ל}4] Murillo Montero, L. (2026e). *\heb{ל}4 — The Helmert–Simplicial Residue Ladder.* Zenodo preprint, DOI [10.5281/zenodo.20383504](https://doi.org/10.5281/zenodo.20383504).

[\heb{ל}5] Murillo Montero, L. (2026f). *\heb{ל}5 — The Harmonic–Simplicial Dual Gauge.* Zenodo preprint, DOI [10.5281/zenodo.20388419](https://doi.org/10.5281/zenodo.20388419).

[\heb{ל}6] Murillo Montero, L. (2026g). *\heb{ל}6 — The RMS–Simplicial Quadratic Gauge.* Zenodo preprint, DOI [10.5281/zenodo.20392428](https://doi.org/10.5281/zenodo.20392428).

[\heb{ל}7] Murillo Montero, L. (2026h). *\heb{ל}7 — Complex Amplitudes and the Quantum Sector of the Apex.* Zenodo preprint, DOI [10.5281/zenodo.20392442](https://doi.org/10.5281/zenodo.20392442).
