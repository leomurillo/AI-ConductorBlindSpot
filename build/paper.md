---
title: "The Conductor Blind Spot of the Quadratic Curvature Class on Ring-Structured Categorical Heads"
author: "Leonardo Murillo Montero"
date: "May 19, 2026"
abstract: |
  Second-order curvature surrogates — natural gradient, K-FAC, Gauss–Newton, and Adam's diagonal empirical-Fisher proxy — are widely deployed on tasks with cyclic categorical structure: modular arithmetic, calendar / clock heads, periodic phase prediction. We prove that at the maximum-entropy point $p_*$ of such a head, all of them share an exact structural *blind spot*: a third-order coupling carried by the Amari–Chentsov cubic that the quadratic curvature class cannot represent. The proof rests on two finite identities that hold exactly at $p_*$ on a categorical model with cyclic ring index $R = \mathbb{Z}/n\mathbb{Z}$: the Fisher information form $g_{p_*}$ is diagonal in the additive-character basis, hence block-diagonal across the conductor-packet decomposition of the tangent space; and the Amari–Chentsov cubic form $T_{p_*}$ couples distinct conductor packets exactly under the integer selection rule $k + \ell + m \equiv 0 \pmod n$. The cross-packet cubic coupling is therefore outside the representational capacity of $g_{p_*}$, and outside the surrogates above plus the neural-tangent-kernel Gram by §2.3's four-tier lift; the same identity lifts at $p_*$ to the head-side input of linear gradient-based edge-attribution scores in mechanistic circuit discovery (EAP-family methods). We verify the certificate computationally for $n \in \{6, 8, 12, 18, 30\}$ in exact integer arithmetic.
  
  From the certificate we derive a retraining-free diagnostic $\rho_\times$ (an L1 spectral-mass ratio on $p_*$-anchored selection-rule triples) and pre-specify a single experiment with a built-in negative control. A synthetic-MLP demo (§6.5) separates ring task from structureless control in the pre-grokking phase; a Pythia checkpoint sweep on a calendar-months head (§6.6) identifies the diagnostic's off-trajectory boundary. Three head-only cubic-aware interventions on $n = 30$ all return Branch B (§8.6): on this evidence the certificate's role is *measurement*, not *steering*. Open follow-ups — $\alpha$-sweep, parameter-side K-FAC variant, longer-budget rings, trajectory instrumentation — are listed in §8.6.
---
**Keywords:** second-order optimization, natural gradient, Fisher information, Amari–Chentsov tensor, categorical models, additive characters, conductor decomposition, ring-structured labels, preconditioning, information geometry, grokking.

**MSC 2020:** 53B12 (information geometry); 62B10 (statistical aspects of information theory); 68T07 (artificial neural networks and deep learning); 11L03 (elementary character sums); 65K10 (numerical optimization).

---

## 1. Introduction

### 1.1 Setting and the question

A central computational object in the practical training of categorical models is the *curvature matrix* used to precondition the gradient: the Fisher information matrix of natural gradient methods [Amari 1998], the Gauss–Newton matrix and its Kronecker-factored approximation K-FAC [Martens & Grosse 2015], the diagonal or block-diagonal empirical Fisher used by Adam-type adaptive optimizers [Kingma & Ba 2015; Kunstner et al. 2019], and the neural-tangent-kernel Gram matrix used as a curvature model in linearized-training analyses [Jacot et al. 2018]. Each of these is a positive-semidefinite *quadratic* form on the output or parameter tangent space. We refer to any optimization method built around such a quadratic curvature model as a *second-order preconditioner*, regardless of how the form is estimated; the certificate of this paper concerns the representational capacity of the quadratic model class itself, not the quality of any particular estimator. The same label-weighted quadratic form has recently been identified, under the name of a Hessian-like label-weighted moment operator $\widehat{C}^{(\ell)}$, as the leading primitive of a spectral theory of multilayer feature learning [Dandi, Vilucchio, Arnaboldi, Tabanelli & Krzakala 2026]; the present certificate identifies a structural null space of that primitive on ring-structured categorical heads (§7.4).

A recurring informal claim is that second-order preconditioners "miss important structure" on tasks with arithmetic regularity (modular arithmetic, calendar prediction, periodic position codes, and more generally any categorical head whose outcomes carry an additive-group structure). The claim is empirically suggestive but, as ordinarily stated, has no exact failure witness: it requires an empirical definition of *important*, an empirical measurement of *missed*, and an empirical demonstration that it *matters*, each contestable, each typically reducible to a benchmarking dispute about baseline tuning. The present paper does not make that empirical claim. We make a smaller one whose witness is finite, exact, and tunable-free: a statement about the representational capacity of the quadratic model class at a specific point in a specific basis, decided by a specific finite predicate.

### 1.2 The result

Let $R = \mathbb{Z}/n\mathbb{Z}$ be the cyclic ring of order $n$, used as the outcome index of a categorical distribution. The tangent space at the open simplex $\Delta_R^\circ$ carries two anchored information-geometric objects: the Fisher information form $g_p$, the second cumulant of the score, and the Amari–Chentsov cubic form $T_p$, the third cumulant of the score (Section 2). These are the second and third floors of a tower of cumulants whose successive forms increasingly resolve the local geometry; the Fisher form is by construction what any quadratic curvature model can carry, and the Amari–Chentsov cubic is the first cumulant it cannot.

The additive characters $\chi_k(y) = e^{2\pi i k y / n}$ for $k \in \mathbb{Z}/n\mathbb{Z}$ are an orthogonal basis of the complexified tangent space; the integer $\mathrm{cond}(k) = n / \gcd(k, n)$ is the *conductor* of the character $\chi_k$, and the *conductor packets* are the equivalence classes of nontrivial characters under equality of conductor. The maximum-entropy (uniform) point is $p_* = \tfrac{1}{n}\mathbf{1}$.

The main result (Theorem 3.5) is the following finite identity, exact by character orthogonality:

* At $p_*$, in the additive-character basis, the Fisher form $g_{p_*}$ is *diagonal*, hence in particular block-diagonal across conductor packets. Every cross-packet entry is exactly zero — the predicate is the integer condition $(k - \ell) \bmod n \ne 0$, with no floating-point comparison entering the decision.

* At $p_*$, in the same basis, the Amari–Chentsov cubic $T_{p_*}(\chi_k, \chi_\ell, \chi_m)$ is nonzero if and only if $k + \ell + m \equiv 0 \pmod{n}$; for non-prime $n$ there exist surviving triples whose three conductor labels are not all equal, so the cubic form carries genuine cross-packet coupling.

Together these establish a structural null space: there is information in the local geometry — the cross-packet content of the cubic — that is provably outside the representational capacity of $g_{p_*}$ on the output tangent space, hence outside the reach of any method built on $g_{p_*}$ as its quadratic curvature surrogate (preconditioning side) or contracting against its head-side first-order gradient (attribution side), to the extent named in the four-tier scope of §2.3 and the paired Proposition 3.11 / Corollary 3.12 in §3.4. We refer to this gap, informally, as the *conductor blind spot* of the quadratic curvature class.

We verify the result computationally on the ladder $n \in \{6, 8, 12, 18, 30\}$, chosen to span squarefree composites ($6$, $30$), a mixed-radix composite ($12$), and prime-power depths ($8 = 2^3$, $18 = 2 \cdot 3^2$). The verification uses exact integer arithmetic; no floating-point comparison enters any vanishing decision.

### 1.3 Contributions

1. **A finite certificate (Section 3).** Theorem 3.5 states the representational limit of the quadratic curvature model class as a pair of exact identities — Fisher block-diagonality and the Amari–Chentsov cross-packet selection rule — at the maximum-entropy point. The proof is a single application of character orthogonality, with no fitted quantity and no tolerance. Three immediate corollaries lift the static claim from the curvature form to the operators built on it at $p_*$: the natural-gradient direction at $p_*$ is the simplex-tangent gradient up to a uniform rescaling (Corollary 3.8); the same scalar-identity form lifts verbatim to parameter space for the single-layer softmax (Corollary 3.9); and the principal symbol of any second-order linear differential operator on the Fisher base whose leading coefficient is $g^{ab}$ — including the Laplace–Beltrami operator and any quadratic kinetic action — is conductor-packet block-diagonal at $p_*$ (Corollary 3.10). A structural number-theoretic characterization (Theorem A.2) identifies exactly when the cross-packet existence condition admits a triple with three *pairwise* distinct conductors: if and only if $n$ is not a prime power. Because there is no baseline to tune and no benchmark to dispute, none of these results has an adversary surface.

2. **A deterministic computational verification (Section 4 and Appendix C).** An exact-arithmetic enumeration confirms both halves of the certificate on the five-ring ladder $n \in \{6, 8, 12, 18, 30\}$, with every cross-packet Fisher entry verified to be exactly zero and the cross-packet cubic triples enumerated and counted ($18, 42, 108, 252, 774$ ordered triples respectively). The pre-claimed falsifier branches — Fisher *not* block-diagonal, or cubic with *no* surviving cross-packet triples, or the character basis failing to diagonalize $g$ — did not occur on any ring.

3. **An exact all-orders off-centroid expansion (Section 5).** A first-order Taylor expansion of the Fisher form along a displacement $h$ from $p_*$ identifies the leading correction with the Amari–Chentsov cubic contracted with $h$ (Lemma 5.1). Read in the character basis, this becomes an exact per-frequency activation rule: the cross-packet entry of $g_{p_* + h}$ between $\chi_k$ and $\chi_\ell$ equals $-n^3\, \hat h_{(\ell - k) \bmod n} + O(\|h\|^2)$ (Theorem 5.2). The expansion extends to all orders in $h$ as an absolutely convergent series whose $j$-th term is a contraction with the $(j+2)$-th score-moment tensor at $p_*$; the cross-packet entry then has the closed Fourier-convolution form $\sum_{j \geq 1} (-1)^j n^{j+2} (\hat h^{*j})_{(\ell - k) \bmod n}$ (Theorem 5.6). The off-centroid geometry of $g$ along any ray $p(t) = p_* + t h$ is in consequence a deterministic analytic function of the Fourier spectrum of $h$. The remaining claim — that this analytic picture persists in a measurable form along a real training trajectory, on a ring-structured task and not on a label-permuted control — is stated as a conjecture (Conjecture 5.8) rather than a theorem, since persistence depends on what a trajectory's running displacement accumulates and is no longer finite-dimensional linear algebra.

4. **A retraining-free diagnostic (Section 6).** From the certificate we derive a measurement instrument computable on any trained model with a categorical head over a ring-structured index. It produces two numbers per model per task: the *cross-packet cubic mass* $\rho_\times$, the fraction of the update direction's third-order content carried by directions the certificate proves the quadratic model class cannot couple; and the *preconditioner discard ratio* $\delta$, the relative change in $\rho_\times$ between the raw update direction and its preconditioned counterpart, measuring how much of the cross-packet cubic mass share the deployed preconditioner suppresses ($\delta > 0$), preserves ($\delta = 0$), or amplifies ($\delta < 0$). The diagnostic makes no performance claim; it converts a qualitative intuition into a measured quantity. Two empirical instances delineate where the diagnostic carries signal: a synthetic-MLP demo on $n \in \{6, 8, 12, 18, 30\}$ (§6.5) identifying three regimes — *structural degeneracy* (n = 8, $\rho_\times \equiv 1$), *insufficient budget* (n = 18, pre-representation-learning), and the discriminative *pre-grokking representation-learning phase* (n = 12); and a retraining-free Pythia checkpoint sweep across $N \in [70\text{M}, 1.4\text{B}]$ and $D \in [2.7 \times 10^{8}, 3.0 \times 10^{11}]$ tokens on a calendar-months head (§6.6), which adds a fourth regime — *off-trajectory* — in which a pretrained model's head sits at the diagnostic's structural ceiling because the training corpus does not exercise the candidate ring structure sufficiently for representation learning to consolidate.

5. **A pre-specified experiment with a negative control (Section 8).** A single sharp test of Conjecture 5.8 on a delayed-generalization (grokking) task, predicting a positive interaction between a cubic-aware preconditioner correction and a label-permutation that destroys the ring structure. Both outcome branches are pre-claimed; a null or negative interaction is reported as a result, not reframed.

6. **A lift to gradient-based mechanism attribution (Proposition 3.11, Corollary 3.12, and §7.7).** The same character-orthogonality identity that proves Theorem 3.5 lifts the static block-diagonality from the curvature object to the *head-side input* of gradient-based edge-attribution scores used in mechanistic circuit discovery — Edge Attribution Patching [Syed, Rager & Conmy 2024], its integrated-gradient refinement EAP-IG [Hanna, Pezzelle & Belinkov 2024], and activation-patching attribution E-ACT [Zhang & Nanda 2023], with the nonlinear ACDC method [Conmy et al. 2023] inheriting the claim at first order around $p_*$. Proposition 3.11 shows that at $p_*$ the back-propagated logit gradient any such method contracts against decomposes orthogonally across conductor packets in the head tangent space; Corollary 3.12 then observes that a linear functional of this first-order input cannot resolve a third-order distinction, so two examples whose local geometries differ only in the cross-packet cubic content of the Amari–Chentsov tensor produce identical attribution score vectors — independent of the downstream network Jacobian. A recently published frontier-LLM circuit-discovery study reports a single discovered circuit reaching $87\%$ faithfulness on two structurally distinct mechanisms in a mixed dataset [Rai, Geva & Yao 2026]; the qualitative signature is consistent with what the certificate's lift to attribution would predict, but the present paper does not run the diagnostic on those artifacts, so the finding is treated as motivating concordant evidence rather than as a confirmed instance of the certificate. §6.7 specifies a retraining-free application of the diagnostic of §6 to per-example attribution vectors as a third diagnostic substrate, with a pre-specified prediction for the recoverable cluster count on cyclic tasks — testable directly on the Rai/Geva/Yao 2026 release.

### 1.4 Organization

Section 2 fixes the categorical model and derives the Fisher and Amari–Chentsov forms as the second and third cumulants of the categorical score, recalls the additive characters of $\mathbb{Z}/n\mathbb{Z}$ and the conductor decomposition of the tangent space, and defines the broader *second-order method* class to include both preconditioner-side use of the curvature object and attribution-side use of the same object in mechanistic circuit discovery (§2.3). Section 3 states and proves the main result and gives four operational lifts of the static claim from the curvature form to the operators built on it at $p_*$: to natural-gradient update behavior at $p_*$ (Corollary 3.8), to parameter space for the single-layer softmax (Corollary 3.9), to the principal symbol of any $g$-built second-order linear differential operator on the Fisher base — Laplace–Beltrami, covariant Laplacian, quadratic kinetic action — at $p_*$ (Corollary 3.10), and to the head-side input of gradient-based edge-attribution scores used by hypothesis-driven circuit discovery (Proposition 3.11 and Corollary 3.12 in §3.4). Section 4 reports the computational verification. Section 5 expands the Fisher form off the maximum-entropy point, proves the leading-order Fourier selection rule (Theorem 5.2) and its extension to an all-orders convergent series with closed Fourier-convolution cross-packet form (Theorem 5.6), and states the dynamical off-centroid persistence conjecture (Conjecture 5.8). Section 6 specifies the diagnostic, reports a first-run synthetic-MLP demo on the five-ring ladder (§6.5), reports a first-run retraining-free application to Pythia pretrained checkpoints on a calendar-months head (§6.6), and lifts the diagnostic to per-example circuit-attribution vectors as a third diagnostic substrate (§6.7). Section 7 discusses the relation to information geometry, the empirical-Fisher critique, mechanistic interpretability of modular-arithmetic learning, recent spectral theories of hierarchical feature learning whose label-weighted second-order primitive is exactly the object the certificate identifies a null space of, a recent nonlinear-sigma-model construction on statistical-manifold bases [Amaral 2025] in which the same purely-second-order model class appears as a kinetic action and inherits the same conductor-packet null space on the categorical Fisher base, a recent algebraic-geometric attention proposal whose sign-dropping squared score destroys exactly the cross-packet orientation the certificate names, and a recent large-scale circuit-discovery study (§7.7) whose gradient-based edge-attribution scores are the operational surface that Proposition 3.11 and Corollary 3.12 address, with a published case in which a single discovered circuit reaches $87\%$ faithfulness on two structurally distinct mechanisms in a mixed dataset [Rai, Geva & Yao 2026] — read as motivating concordant evidence pending the §6.7 pre-specified diagnostic test on the released artifacts — and an optimisation-side counterpart (§7.8) covering two recent methods that *do* compress grokking delay on modular arithmetic by acting uniformly within modes of the gradient spectrum [Saheb Pasand & Dohmatob 2026; Jiang et al. 2026], read against the certificate's claim that within-mode action is exactly the action the cross-packet null structure leaves available, together with an architectural counterpart [Yıldırım 2026] whose spherical-topology bypass aligns with cyclic symmetry but fails on non-abelian tasks. Section 8 pre-specifies the experiment and reports three head-only instances of it (§8.6: Branch B in all three forms — Newton-style and Neumann-style cross-packet injection, and within-packet amplification — at $\alpha = 0.1$ on $n = 30$). Section 9 lists scope and limitations. Appendix A gives the existence proof for cross-packet zero-sum triples on every non-prime $n$ (Proposition A.1) and characterizes exactly when three pairwise-distinct conductors are achievable (Theorem A.2: iff $n$ is not a prime power); Appendix B specifies the non-cyclic finite abelian extension and verifies it on five candidate groups ($\mathbb{Z}/2 \times \mathbb{Z}/4$, $\mathbb{Z}/4 \times \mathbb{Z}/4$, $\mathbb{Z}/2 \times \mathbb{Z}/8$, $(\mathbb{Z}/2)^2 \times \mathbb{Z}/4$, $\mathbb{Z}/3 \times \mathbb{Z}/9$); Appendix C records the full cyclic computational artifact; Appendix D writes out the off-centroid Fisher expansion in full.

---

## 2. Setup

### 2.1 The categorical model on a ring index

Let $n \geq 2$ be an integer and $R = \mathbb{Z}/n\mathbb{Z}$, with elements $0, 1, \dots, n-1$ and addition modulo $n$. We say the outcome index of a categorical model carries a *ring structure* when $R$ is used as the label set with its additive-group operation. The model emits a probability vector $p \in \Delta_R^\circ$, the open probability simplex over $R$:

$$
\Delta_R^\circ = \left\{ p \in \mathbb{R}^R : p_y > 0 \text{ for all } y \in R,\ \textstyle\sum_{y \in R} p_y = 1 \right\}.
$$

The tangent space at $p$ is

$$
T_p \Delta_R^\circ = \left\{ u \in \mathbb{R}^R : \textstyle\sum_{y \in R} u_y = 0 \right\},
$$

a real vector space of dimension $n - 1$. The complexification $T_p \Delta_R^\circ \otimes_{\mathbb{R}} \mathbb{C}$ is the analogous space over $\mathbb{C}$, also of complex dimension $n - 1$.

The ring-structured hypothesis is minimal: we make no assumption on the model architecture or the parametrization of the head. The carrier of the result is the geometry of $\Delta_R^\circ$ as a statistical manifold and the action of the additive group $R$ on the index.

### 2.2 The Fisher information and Amari–Chentsov forms

For a smooth statistical manifold $\{p_\theta\}$, the *Fisher information form* and *Amari–Chentsov cubic form* at the parameter point $\theta$ are the second and third cumulants of the score $\partial_\theta \log p_\theta$ [Amari & Nagaoka 2000, §2.3, §3.3]:

$$
g_\theta(u, v) = \mathbb{E}_{Y \sim p_\theta}\!\left[ \partial_u \log p_\theta(Y) \,\partial_v \log p_\theta(Y) \right], \qquad
T_\theta(u, v, w) = \mathbb{E}_{Y \sim p_\theta}\!\left[ \partial_u \log p_\theta(Y) \,\partial_v \log p_\theta(Y) \,\partial_w \log p_\theta(Y) \right].
$$

Specialized to the categorical model on $R$ with $\theta = p$ a free parametrization of the open simplex, a direct calculation gives the *anchored coordinate forms*

$$
g_p(u, v) = \sum_{y \in R} \frac{u_y v_y}{p_y}, \qquad
T_p(u, v, w) = \sum_{y \in R} \frac{u_y v_y w_y}{p_y^2}, \tag{2.1}
$$

defined on tangent vectors $u, v, w \in T_p \Delta_R^\circ$. We record (2.1) as the definition we will use; the cumulant derivation is standard and we sketch it in the remainder of this subsection for self-containedness.

**Derivation.** A small perturbation $p \to p + \varepsilon u$ with $u \in T_p \Delta_R^\circ$ produces a log-likelihood increment $\log(p_y + \varepsilon u_y) - \log p_y = \varepsilon (u_y / p_y) - \tfrac{1}{2}\varepsilon^2 (u_y/p_y)^2 + \tfrac{1}{3}\varepsilon^3 (u_y/p_y)^3 - \dots$ Substituting into the cumulant definition of $g$ and using $\mathbb{E}_{Y \sim p}[f(Y)] = \sum_y p_y f(y)$ gives $g_p(u, u) = \sum_y p_y (u_y/p_y)^2 = \sum_y u_y^2 / p_y$; polarization gives the bilinear form (2.1). The same expansion at third order, together with the vanishing $\sum_y u_y = 0$ that removes the first-cumulant contribution, yields the trilinear form (2.1). The Fisher form is symmetric positive-definite on $T_p \Delta_R^\circ$ for every $p \in \Delta_R^\circ$; the Amari–Chentsov form is symmetric trilinear and is the first cumulant that is not a function of $g_p$.

### 2.3 What "second-order method" means here, and the scope of the lift

Theorem 3.5 is a statement about the Fisher form $g_{p_*}$ on the output tangent space $T_{p_*}\Delta_R^\circ$. Any operational reading that names a deployed method — natural-gradient preconditioning, K-FAC, Adam's empirical Fisher, NTK-based curvature, EAP-family attribution — is a *lift* of that single statement to a curvature surrogate the method actually uses. The strength of the lift varies across methods, and we therefore organize the broader method class by the kind of lift required and attach the strict, tunable-parameter-free claim to the underlying object $g_{p_*}$ itself rather than to each named method uniformly. We define a *second-order method* (preconditioner or attribution surrogate) by its relation to $g_{p_*}$ on $T_{p_*}\Delta_R^\circ$, in four tiers:

* **(T1) Exact at the output side at $p_*$.** Methods whose local curvature surrogate evaluated at $p_*$ on $T_{p_*}\Delta_R^\circ$ equals $g_{p_*}$ verbatim inherit Corollary 3.3's null structure exactly. This tier includes the exact natural gradient method on the simplex [Amari 1998] and the Gauss–Newton output Hessian on a cross-entropy softmax head, both of which reduce to $g_{p_*}$ at $p_*$ by direct calculation. The inheritance here is not improvable away by better estimation — even an oracle returning the exact true Fisher at $p_*$ carries the null structure.

* **(T2) Jacobian pullback.** Methods whose curvature surrogate is a pullback $J^\top g_{p_*} J$ through some Jacobian $J$ inherit the null structure *conditional on $J$ preserving the conductor-packet decomposition*. The single-layer softmax pullback is exactly packet-preserving (Corollary 3.9 in §3.4); for deeper architectures the analogous packet-preservation statement is open and is recorded as such in §9. This tier includes the parameter-side natural gradient and the Gauss–Newton matrix in parameter space.

* **(T3) Approximate, factorized, or estimator surrogates.** Methods whose curvature surrogate is a further approximation of a tier-T2 pullback object — the Kronecker-factored approximation K-FAC [Martens & Grosse 2015], the empirical Fisher used by Adam-type adaptive methods [Kingma & Ba 2015], which is known to differ structurally from the true Fisher [Kunstner, Hennig & Balles 2019], and the neural-tangent-kernel Gram matrix in linearized-training regimes [Jacot et al. 2018] — inherit the null structure of the underlying true object through the pullback chain of tier T2 plus an additional approximation step. The certificate constrains the true object $g_{p_*}$; how much of that constraint survives a given approximation is method-specific and is not asserted by this paper.

* **(T4) Linear gradient-based attribution.** Gradient-based edge-attribution methods used in mechanistic circuit discovery — Edge Attribution Patching (EAP) [Syed, Rager & Conmy 2024], its integrated-gradient refinement EAP-IG [Hanna, Pezzelle & Belinkov 2024], and activation-patching attribution (E-ACT) [Zhang & Nanda 2023] — do not have a quadratic curvature surrogate per se; they compute per-edge scores as a linear functional of the back-propagated logit gradient with respect to edge activations. The lift is at the *input* to the attribution, not at a curvature object: Proposition 3.11 in §3.4 shows that at $p_*$ the head-side input these methods contract against decomposes orthogonally across conductor packets, and Corollary 3.12 then observes that a linear functional of that first-order input cannot resolve the third-order cross-packet cubic distinction. The nonlinear method Automated Circuit Discovery (ACDC) [Conmy et al. 2023] inherits the same conclusion at first order around $p_*$ (see the ACDC remark after Corollary 3.12). Three further methods in this family inherit Corollary 3.12 verbatim because their per-edge score is also a linear (or path-integral-of-linear) functional of the head-side first-order gradient: gradient-based Edge Pruning [Bhaskar et al. 2024], the scalable AtP\* refinement of attribution patching with bounded false-negative probability [Kramár et al. 2024], and Relevance Patching (RelP), which substitutes LRP propagation coefficients for the local gradient while preserving the two-forward / one-backward attribution-patching cost structure [Rezaei Jafari et al. 2025]. The cross-layer transcoder replacement model of [Ameisen, Lindsey, Pearce, Gurnee et al. 2025] is a strictly more elaborate substrate: it introduces a *learned* approximate replacement (the transcoder) before the attribution step, and the linear-functional argument of Corollary 3.12 applies to the post-transcoder attribution scores conditional on the transcoder being approximately linear in the head-side input around $p_*$ — under that condition the certificate's lift to attribution carries over; relaxing that condition is open. The methodology survey of [Rai, Zhou, Feng, Saparov & Yao 2024] organises this family at the practical level.

We write "second-order *preconditioner*" when the application is optimizer-side, "second-order *attribution*" when it is interpretability-side, and "second-order method" when the distinction is immaterial. Theorem 3.5 is a statement about the *representational capacity* of $g_{p_*}$ at the output side; the umbrella phrase "the quadratic curvature model class" used in this paper refers, throughout, to the object $g_{p_*}$ on $T_{p_*}\Delta_R^\circ$ and its inheritance across the four tiers above, not to a uniform identification of all listed methods with a single matrix.

### 2.4 Additive characters and the conductor decomposition

The *additive characters* of $R = \mathbb{Z}/n\mathbb{Z}$ are the group homomorphisms $\chi : R \to \mathbb{C}^\times$, namely

$$
\chi_k(y) = e^{2\pi i k y / n}, \qquad k \in \mathbb{Z}/n\mathbb{Z}. \tag{2.2}
$$

The character $\chi_0 \equiv 1$ is the trivial character; the nontrivial characters $\{\chi_k\}_{k=1}^{n-1}$ span the complexified tangent space (the trivial character corresponds to the simplex normal direction $\mathbf{1}$, excluded from $T_p \Delta_R^\circ$).

**Lemma 2.1 (character orthogonality).** *For $k, \ell \in \mathbb{Z}/n\mathbb{Z}$,*
$$
\sum_{y \in R} \chi_k(y)\, \overline{\chi_\ell(y)} = n \cdot \mathbf{1}\!\left[\, k \equiv \ell \pmod{n} \,\right]. \tag{2.3}
$$

*Proof.* Set $\zeta = e^{2\pi i (k - \ell)/n}$. Then $\sum_{y=0}^{n-1} \chi_k(y)\overline{\chi_\ell(y)} = \sum_{y=0}^{n-1} \zeta^y$. If $k \equiv \ell \pmod{n}$ then $\zeta = 1$ and the sum is $n$. Otherwise $\zeta \neq 1$, $\zeta^n = 1$, and the finite geometric sum is $(\zeta^n - 1)/(\zeta - 1) = 0$. $\square$

We emphasize the arithmetic character of (2.3): the decision whether the sum is $n$ or $0$ is the integer predicate $(k - \ell) \bmod n = 0$. No floating-point comparison enters.

The *conductor* of the character $\chi_k$ for $k \in \mathbb{Z}/n\mathbb{Z}$, $k \ne 0$, is

$$
\mathrm{cond}(k) := \frac{n}{\gcd(k, n)} \in \{d : d \mid n,\ d > 1\}.
$$

Equivalently, $\mathrm{cond}(k)$ is the order of $\chi_k$ in the dual group of $R$, and is the smallest positive integer $d$ such that $\chi_k$ factors through the quotient map $\mathbb{Z}/n\mathbb{Z} \to \mathbb{Z}/d\mathbb{Z}$. The *conductor packet* with conductor $d$ is

$$
\mathcal{P}_d := \{\, k \in \mathbb{Z}/n\mathbb{Z} : \mathrm{cond}(k) = d \,\} = \{\, k : \gcd(k, n) = n/d \,\}, \qquad d \mid n,\ d > 1.
$$

The nontrivial characters partition into conductor packets: $\{1, \dots, n-1\} = \bigsqcup_{d \mid n, d > 1} \mathcal{P}_d$. The number of nontrivial divisors $d > 1$ of $n$ is the number of packets; $|\mathcal{P}_d| = \varphi(d)$, where $\varphi$ is Euler's totient. For prime $n$ there is a single packet ($\mathcal{P}_n$); for composite $n$ there are at least two.

**Examples.** For $n = 6$: $\mathcal{P}_2 = \{3\}$, $\mathcal{P}_3 = \{2, 4\}$, $\mathcal{P}_6 = \{1, 5\}$ (three packets). For $n = 8$: $\mathcal{P}_2 = \{4\}$, $\mathcal{P}_4 = \{2, 6\}$, $\mathcal{P}_8 = \{1, 3, 5, 7\}$ (three packets). For $n = 12$: $\mathcal{P}_2 = \{6\}$, $\mathcal{P}_3 = \{4, 8\}$, $\mathcal{P}_4 = \{3, 9\}$, $\mathcal{P}_6 = \{2, 10\}$, $\mathcal{P}_{12} = \{1, 5, 7, 11\}$ (five packets).

The decomposition into conductor packets is the canonical orthogonal decomposition of the complexified tangent space by the cyclic-group symmetry; in arithmetic-theoretic language it is the decomposition by the kernel of the quotient maps to smaller cyclic quotients of $R$.

### 2.5 A worked example: $n = 6$

To make the conductor decomposition and the certificate concrete before the general proofs, we work through the smallest non-degenerate cyclic case. For $R = \mathbb{Z}/6\mathbb{Z}$, the divisors $d \mid n$ with $d > 1$ are $\{2, 3, 6\}$, and the three conductor packets are

$$
\mathcal{P}_2 = \{3\}, \qquad \mathcal{P}_3 = \{2, 4\}, \qquad \mathcal{P}_6 = \{1, 5\}.
$$

The five nontrivial characters partition into these three packets of sizes $\varphi(2) = 1$, $\varphi(3) = 2$, $\varphi(6) = 2$, summing to $n - 1 = 5$. Each character $\chi_k$ is placed in $\mathcal{P}_d$ for $d = n/\gcd(k, n)$ — its order in the dual group. The divisor lattice $\{1, 2, 3, 6\}$ of $n = 6$ has three nontrivial divisors $\{2, 3, 6\}$, which index the three packets:

| Divisor $d$ | $\gcd(k, n)$ | Packet $\mathcal{P}_d$ | Character order |
|:-----------:|:------------:|:-----------------------|:---------------:|
| $2$ | $3$ | $\{3\}$ | $2$ |
| $3$ | $2$ | $\{2, 4\}$ | $3$ |
| $6$ | $1$ | $\{1, 5\}$ | $6$ |

**Fisher matrix at $p_*$ in the character basis.** By Lemma 3.1 (proved in §3.1), $g_{p_*}(\chi_k, \overline{\chi_\ell}) = n^2 \delta_{k\ell}$, so the Fisher matrix at $p_* = \tfrac{1}{6}\mathbf{1}$ on the basis $\{\chi_1, \chi_2, \chi_3, \chi_4, \chi_5\}$ is $G_{p_*}^{(n=6)} = 36 \cdot I_5$. Re-arranging in *packet order* $(\chi_3 \mid \chi_2, \chi_4 \mid \chi_1, \chi_5)$ exhibits the block-diagonal structure with packet blocks of sizes $1{\times}1$, $2{\times}2$, $2{\times}2$:

$$
G_{p_*}^{(n=6)} \;=\;
\left(
\begin{array}{c|cc|cc}
36 & 0 & 0 & 0 & 0 \\ \hline
0 & 36 & 0 & 0 & 0 \\
0 & 0 & 36 & 0 & 0 \\ \hline
0 & 0 & 0 & 36 & 0 \\
0 & 0 & 0 & 0 & 36
\end{array}
\right).
$$

Diagonality is stronger than block-diagonality at $p_*$; the block-diagonal reading is the one the certificate generalises (off $p_*$, the Fisher remains within-packet to leading order and gains cross-packet *block-off-diagonal* entries from the cubic at $O(\|h\|)$, per Theorem 5.2).

**Amari–Chentsov cubic at $p_*$.** The cubic $T_{p_*}(\chi_k, \chi_\ell, \chi_m) = n^3 \cdot \mathbf{1}[k + \ell + m \equiv 0 \pmod n]$ (Lemma 3.4, proved in §3.2) is supported on the integer predicate $k + \ell + m \equiv 0 \pmod 6$. There are exactly $(n-1)(n-2) = 20$ ordered triples $(k, \ell, m) \in \{1, \dots, 5\}^3$ satisfying this predicate; they break down by conductor pattern (the multiset $\{\mathrm{cond}(k), \mathrm{cond}(\ell), \mathrm{cond}(m)\}$) as

| Conductor pattern | Count | Examples |
|:------------------|------:|:---------|
| $\{3, 3, 3\}$ (same-packet, both in $\mathcal{P}_3$) | $2$ | $(2,2,2)$, $(4,4,4)$ |
| $\{3, 6, 6\}$ (cross-packet) | $6$ | three ordered placements each of $(1,1,4)$ and $(2,5,5)$ |
| $\{2, 3, 6\}$ (cross-packet, all three packets) | $12$ | six permutations each of $(1,2,3)$ and $(3,4,5)$ |

Total $2 + 6 + 12 = 20$, of which $18$ are cross-packet, matching Appendix C.

**The contrast in one snapshot.** Take the canonical cross-packet sample $(k, \ell, m) = (1, 2, 3)$: $k + \ell + m = 6 \equiv 0 \pmod 6$, conductors $(6, 3, 2)$ all distinct. Then

$$
g_{p_*}(\chi_1, \overline{\chi_2}) \;=\; 0 \quad\text{(packets differ)}, \qquad T_{p_*}(\chi_1, \chi_2, \chi_3) \;=\; 6^3 \;=\; 216 \;\ne\; 0.
$$

The Fisher form contains no information that couples $\chi_1 \in \mathcal{P}_6$ to $\chi_2 \in \mathcal{P}_3$ at $p_*$; the cubic, evaluated on the same pair extended by $\chi_3 \in \mathcal{P}_2$, is exactly $n^3$. That coupling is what no surrogate built from $g_{p_*}$ can carry — the certificate's content, exhibited for the smallest non-degenerate $n$ in one line.

---

## 3. The main result

### 3.1 Fisher block-diagonality at the maximum-entropy point

Let $p_* := \tfrac{1}{n}\mathbf{1} \in \Delta_R^\circ$ denote the maximum-entropy (uniform) point.

**Lemma 3.1 (Fisher diagonality at $p_*$).** *In the additive-character basis, the Fisher form at $p_*$ is exactly diagonal:*
$$
g_{p_*}(\chi_k, \overline{\chi_\ell}) = n^2 \cdot \mathbf{1}\!\left[\, k \equiv \ell \pmod{n} \,\right], \qquad k, \ell \in \mathbb{Z}/n\mathbb{Z}, \ k, \ell \ne 0. \tag{3.1}
$$

*Proof.* Since $p_{*, y} = 1/n$ for every $y$,
$$
g_{p_*}(u, v) = \sum_{y \in R} \frac{u_y v_y}{p_{*, y}} = n \sum_{y \in R} u_y v_y.
$$
Substituting $u = \chi_k$, $v = \overline{\chi_\ell}$ and applying Lemma 2.1:
$$
g_{p_*}(\chi_k, \overline{\chi_\ell}) = n \sum_{y \in R} \chi_k(y) \overline{\chi_\ell(y)} = n \cdot n \cdot \mathbf{1}\!\left[\, k \equiv \ell \pmod{n} \,\right] = n^2 \delta_{k\ell},
$$
which is (3.1). The off-diagonal entries are exactly zero, decided by the integer predicate $(k - \ell) \bmod n = 0$. $\square$

**Remark 3.2.** Lemma 3.1 implicitly extends $g_{p_*}$ Hermitian-bilinearly to the complexified tangent space. The associated real symmetric form on $T_{p_*} \Delta_R^\circ$ is its real part, and inherits block-diagonality across conductor packets in the real cosine/sine basis $\{\cos(2\pi k y / n), \sin(2\pi k y / n)\}_{k}$. The certificate below is stated in the complexified basis for cleanliness; the real-basis version is equivalent.

**Corollary 3.3 (zero inter-packet content of the quadratic model class).** *Any quadratic form equal to $g_{p_*}$ — equivalently, any second-order preconditioner whose curvature model is the Fisher form at $p_*$ — carries identically zero coupling between distinct conductor packets:*
$$
g_{p_*}(u, v) = 0 \quad \text{whenever } u \in \mathcal{P}_{d}^{\mathbb{C}},\ v \in \mathcal{P}_{d'}^{\mathbb{C}},\ d \ne d',
$$
*where $\mathcal{P}_d^{\mathbb{C}} := \mathrm{span}_{\mathbb{C}}\{\chi_k : k \in \mathcal{P}_d\}$ is the conductor-$d$ packet in the complexified tangent space.*

*Proof.* Immediate from Lemma 3.1 by linearity: $g_{p_*}$ is diagonal, so any two vectors supported on disjoint character indices are orthogonal in $g_{p_*}$. Since $\mathcal{P}_d^{\mathbb{C}}$ and $\mathcal{P}_{d'}^{\mathbb{C}}$ are supported on disjoint subsets of $\{1, \dots, n-1\}$ when $d \ne d'$, the claim follows. $\square$

Corollary 3.3 is the *zero-coupling* half of the certificate. There is no estimation error here to improve away by better curvature modelling: the exact object $g_{p_*}$ carries the null structure.

### 3.2 Amari–Chentsov inter-packet selection rule

**Lemma 3.4 (cross-packet content of the cubic at $p_*$).** *In the additive-character basis,*
$$
T_{p_*}(\chi_k, \chi_\ell, \chi_m) = n^3 \cdot \mathbf{1}\!\left[\, k + \ell + m \equiv 0 \pmod{n} \,\right], \qquad k, \ell, m \in \mathbb{Z}/n\mathbb{Z}. \tag{3.2}
$$

*Proof.* Since $p_{*, y}^2 = 1/n^2$,
$$
T_{p_*}(u, v, w) = \sum_{y \in R} \frac{u_y v_y w_y}{p_{*, y}^2} = n^2 \sum_{y \in R} u_y v_y w_y.
$$
Substituting $u = \chi_k$, $v = \chi_\ell$, $w = \chi_m$ (no conjugation; $T$ is symmetric trilinear, not Hermitian),
$$
T_{p_*}(\chi_k, \chi_\ell, \chi_m) = n^2 \sum_{y \in R} e^{2\pi i (k + \ell + m) y / n}.
$$
Let $\zeta = e^{2\pi i (k + \ell + m)/n}$. If $k + \ell + m \equiv 0 \pmod n$, then $\zeta = 1$ and the sum is $n$, giving $T_{p_*}(\chi_k, \chi_\ell, \chi_m) = n^3$. Otherwise $\zeta \ne 1$, $\zeta^n = 1$, and the finite geometric sum vanishes. $\square$

The vanishing decision in (3.2) is, again, an integer predicate: $(k + \ell + m) \bmod n = 0$ or not. The triple $(k, \ell, m)$ couples through the Amari–Chentsov cubic at $p_*$ exactly when the integer sum modulo $n$ is zero.

### 3.3 The certificate

We now combine the two identities.

**Theorem 3.5 (Representational limit of quadratic curvature on cyclic categorical heads).** *Fix $n$ composite (so that $\mathbb{Z}/n\mathbb{Z}$ has at least two conductor packets), and let $p_* = \tfrac{1}{n}\mathbf{1}$. Then:*

1. *(Block-diagonal Fisher.) The Fisher form $g_{p_*}$ is block-diagonal across conductor packets: $g_{p_*}(u, v) = 0$ whenever $u, v$ lie in distinct conductor packets of the complexified tangent space (Corollary 3.3). The second-order curvature model class therefore carries identically zero inter-packet coupling at $p_*$.*

2. *(Cross-packet cubic.) There exist triples $(k, \ell, m) \in (\mathbb{Z}/n\mathbb{Z} \setminus \{0\})^3$ with $k + \ell + m \equiv 0 \pmod{n}$ whose conductor labels $(\mathrm{cond}(k), \mathrm{cond}(\ell), \mathrm{cond}(m))$ are not all equal; for each such triple $T_{p_*}(\chi_k, \chi_\ell, \chi_m) = n^3 \ne 0$.*

*Consequently, the Amari–Chentsov cubic at $p_*$ couples directions that the Fisher form $g_{p_*}$ certifies are decoupled, and the second-order curvature model class is constitutionally unable to represent this coupling at $p_*$.*

*Proof.* Part 1 is Corollary 3.3. For Part 2, by Lemma 3.4 it suffices to exhibit a single triple $(k, \ell, m)$ with $k + \ell + m \equiv 0 \pmod n$, none of $k, \ell, m$ zero modulo $n$, and not all of $\mathrm{cond}(k), \mathrm{cond}(\ell), \mathrm{cond}(m)$ equal. Let $p$ be the smallest prime divisor of $n$ (such exists since $n$ is composite, and $p \leq \sqrt{n}$, hence $p \leq n/2$ for $n \geq 4$). Set $k = 1$, $\ell = p$, $m = n - 1 - p$. Then $k + \ell + m = n \equiv 0 \pmod n$; the bound $p \leq n/2$ gives $m = n - 1 - p \geq n/2 - 1 \geq 1$, so $m \in \{1, \dots, n-1\}$. The conductors are $\mathrm{cond}(k) = n / \gcd(1, n) = n$ and $\mathrm{cond}(\ell) = n / \gcd(p, n) = n/p < n$ (since $p \mid n$). Hence $\mathrm{cond}(k) \ne \mathrm{cond}(\ell)$ and the three conductors are not all equal. Appendix A records the construction in full; Theorem A.2 there characterizes exactly when the stronger condition of three pairwise-distinct conductors is achievable. $\square$

**Remark 3.6 (three-distinct-conductor case).** Part 2 of Theorem 3.5 asserts only that the three conductor labels of the surviving triple are *not all equal* — two distinct conductors among $(\mathrm{cond}(k), \mathrm{cond}(\ell), \mathrm{cond}(m))$ suffice to certify cross-packet content. The stronger assertion — that there exist triples with *three pairwise distinct* conductors — has a clean number-theoretic characterization, proved as Theorem A.2: it holds if and only if $n$ is not a prime power. The certificate does not require this stronger condition, and for the same reason the standard notion of *cross-packet* in our diagnostic (Section 6) is the weaker one.

**Remark 3.7 (the witness has no tunable parameter).** The certificate is a pair of exact integer-arithmetic identities: a Kronecker delta on $(k - \ell) \bmod n$ and a Kronecker delta on $(k + \ell + m) \bmod n$. There is no learning rate, no baseline optimizer, no benchmark, no tolerance, no fitted tensor, and no trained model anywhere in the statement or in the proof. The result cannot be deflected by a tuning argument because it has nothing to tune.

### 3.4 Operational consequences at the maximum-entropy point

Corollary 3.3 is a statement about the curvature *form* $g_{p_*}$. We record four immediate consequences at the level of operators built on $g_{p_*}$ at the maximum-entropy point: two at the level of *updates* — first on the simplex tangent space (the natural-gradient direction for any loss on the categorical head, Corollary 3.8), then in parameter space for the simplest concrete head architecture (a single layer of logits, Corollary 3.9) — one at the level of *principal symbols* of $g$-built second-order linear differential operators on the Fisher base (Corollary 3.10), and a paired Proposition–Corollary at the level of *gradient-based mechanism attribution*, where the head-side input to the attribution is packet-diagonal (Proposition 3.11) and a linear functional of that first-order input cannot resolve a third-order distinction (Corollary 3.12).

**Corollary 3.8 (natural-gradient packet preservation at $p_*$).** *Let $\mathcal{L} : \Delta_R^\circ \to \mathbb{R}$ be any smooth loss, and write $\pi_d : T_{p_*}\Delta_R^\circ \otimes \mathbb{C} \to \mathcal{P}_d^{\mathbb{C}}$ for the orthogonal projection onto the conductor-$d$ packet of the complexified tangent space. The natural-gradient direction at the maximum-entropy point is*
$$
-\, g_{p_*}^{-1} \nabla \mathcal{L}(p_*) \;=\; -\frac{1}{n^2}\, \nabla \mathcal{L}(p_*),
$$
*a uniform rescaling of the simplex-tangent gradient. Consequently for every $d \mid n$, $d > 1$,*
$$
\pi_d\!\left[ -\, g_{p_*}^{-1} \nabla \mathcal{L}(p_*) \right] \;=\; -\frac{1}{n^2}\, \pi_d \nabla \mathcal{L}(p_*),
$$
*and the natural-gradient operator at $p_*$ acts packet-by-packet: information about $\nabla \mathcal{L}(p_*)$ in $\mathcal{P}_{d_1}^{\mathbb{C}}$ does not reach the update component in $\mathcal{P}_{d_2}^{\mathbb{C}}$ for $d_1 \ne d_2$.*

*Proof.* By Lemma 3.1, $g_{p_*}(\chi_k, \overline{\chi_\ell}) = n^2\, \delta_{k\ell}$ for $k, \ell \in \{1, \dots, n - 1\}$. Hence in the character basis $g_{p_*}$ is the scalar $n^2$ times the identity on the complexified tangent space, and its inverse is the scalar $1/n^2$ times the identity. Applying $-g_{p_*}^{-1}$ to $\nabla \mathcal{L}(p_*)$ yields the stated direction; the projection identity follows because scalar multiples of the identity commute with every orthogonal projector, in particular each $\pi_d$. $\square$

**Remark.** Corollary 3.8 makes the static "blind spot" claim concrete at the update level: at $p_*$, the natural-gradient direction is the simplex-tangent gradient up to a uniform rescaling, so any cross-packet structure present in the *update* must have come from the *gradient*. The optimizer's curvature model introduces no cross-packet mixing. Off $p_*$, this scalar-identity form of $g$ no longer holds: by Theorem 5.6, the off-diagonal entries of $g_p$ activate as a deterministic function of the displacement's Fourier spectrum, and the natural-gradient direction begins to mix packets. The mixing rate is governed by the cubic and higher score-moment tensors (Theorem 5.6), all outside the quadratic model class — which is what the certificate identifies as the structural deficit.

**Corollary 3.9 (single-layer-softmax parameter pullback).** *Consider a single-layer softmax head $p_\theta(y) = e^{\theta_y} / \sum_z e^{\theta_z}$ with logit parameters $\theta \in \mathbb{R}^R$, taken modulo the standard additive-shift gauge $\theta \sim \theta + c\mathbf{1}$, and write $T_0 := \{\delta\theta \in \mathbb{R}^R : \sum_y \delta\theta_y = 0\}$ for the centered logit subspace. At any constant configuration $\theta = c\mathbf{1}$ — at which $p_\theta = p_*$ — the parameter Fisher matrix restricted to $T_0$ equals $(1/n)\, I$, a scalar multiple of the identity. Consequently:*

*(i) The additive-character basis of $\mathbb{Z}/n\mathbb{Z}$ is an orthogonal eigenbasis of the parameter Fisher on $T_0$ with all eigenvalues equal to $1/n$; the conductor-packet decomposition of $T_0 \otimes \mathbb{C}$ coincides with the conductor-packet decomposition of the simplex tangent space.*

*(ii) The natural-gradient parameter update at the uniform configuration is*
$$
\delta\theta_{\mathrm{NG}} \;=\; -n\, \nabla_\theta \mathcal{L}(c\mathbf{1}),
$$
*a uniform rescaling of the parameter gradient; it preserves every orthogonal decomposition of the parameter gradient, including the conductor-packet decomposition.*

*Proof.* The softmax score is $\partial_z \log p_\theta(y) = \delta_{yz} - p_\theta(z)$, so the parameter Fisher entries are $F_{zz'} = \mathbb{E}_{Y \sim p_\theta}[(\delta_{Yz} - p_\theta(z))(\delta_{Yz'} - p_\theta(z'))] = p_\theta(z) \delta_{zz'} - p_\theta(z) p_\theta(z')$. At $p_\theta = p_*$ this gives $F = (1/n)\, I - (1/n^2)\, \mathbf{1} \mathbf{1}^\top = (1/n)\, \Pi$, where $\Pi$ is the orthogonal projection onto $T_0$. Restricted to $T_0$, $F|_{T_0} = (1/n)\, I$. Inverting gives $F|_{T_0}^{-1} = n\, I$, hence $\delta\theta_{\mathrm{NG}} = -F|_{T_0}^{-1} \nabla_\theta \mathcal{L} = -n \nabla_\theta \mathcal{L}$. (i) follows because a scalar multiple of the identity has every orthogonal subspace as an eigenspace, including each $\mathcal{P}_d^{\mathbb{C}}$. (ii) follows because a scalar multiple of the identity commutes with every linear projector. $\square$

Together, Corollaries 3.8 and 3.9 lift the static blind-spot claim from the simplex tangent space to the parameter space of the simplest concrete categorical head: at the uniform configuration, no inter-packet mixing is introduced by the natural-gradient update, on either side of the softmax. For deeper architectures the lift is mediated by the network's Jacobian; we leave the general pullback as future work and discuss it briefly in Section 9.

**Corollary 3.10 (Principal-symbol blind spot for second-order differential operators on the Fisher base at $p_*$).** *Let $\Delta_R^\circ$ carry the Fisher–Rao metric $g$, and let $\Delta_{\mathcal{M}}$ denote the Laplace–Beltrami operator on the Riemannian manifold $(\Delta_R^\circ, g)$.*

*(i) In Euclidean-orthonormal coordinates $(\xi^1, \dots, \xi^{n-1})$ on the zero-sum hyperplane $H = T_{p_*}\Delta_R^\circ \subset \mathbb{R}^R$, the principal-symbol contribution to $\Delta_{\mathcal{M}}$ at $p_*$ is*
$$
\Delta_{\mathcal{M}}\big|_{p_*}^{\mathrm{symb}} \;=\; \frac{1}{n}\,\Delta_H, \qquad \Delta_H = \sum_{a=1}^{n-1} \frac{\partial^2}{\partial (\xi^a)^2}.
$$

*(ii) The principal symbol of $\Delta_{\mathcal{M}}$ at $p_*$, viewed as a quadratic form on $T^*_{p_*}\Delta_R^\circ \otimes \mathbb{C}$, is diagonal in the dual character basis $\{\chi_k^*\}_{k=1}^{n-1}$ with $g^{-1}_{p_*}(\chi_k^*, \overline{\chi_\ell^*}) = (1/n^2)\,\delta_{k\ell}$, and carries identically zero coupling between distinct conductor packets: for $d \ne d'$ and any covectors $\xi \in \mathcal{P}_d^{\mathbb{C}*},\ \eta \in \mathcal{P}_{d'}^{\mathbb{C}*}$, the principal-symbol pairing $g^{-1}_{p_*}(\xi, \eta) = 0$.*

*(iii) The same diagonalization holds for the principal symbol at $p_*$ of any second-order linear differential operator on $\Delta_R^\circ$ whose leading coefficient is $g^{ab}$ — including any covariant Laplacian on $(\Delta_R^\circ, g)$, the Hessian of any quadratic action functional with kinetic term $g^{ab}\partial_a\phi\,\partial_b\phi$, and any Bochner-type Laplacian on $g$-tensor bundles.*

*Proof.* For $u, v \in T_{p_*}\Delta_R^\circ = H$,
$$
g_{p_*}(u, v) \;=\; \sum_{y \in R} \frac{u_y v_y}{1/n} \;=\; n\, \langle u, v \rangle_{\mathrm{Euc}}.
$$
In the Euclidean-orthonormal chart on $H$, the matrix of $g_{p_*}$ is therefore $g_{ab}|_{p_*} = n\,\delta_{ab}$ and the matrix of the inverse metric is $g^{ab}|_{p_*} = (1/n)\,\delta^{ab}$. The principal-symbol contribution to $\Delta_{\mathcal{M}} = (1/\sqrt{g})\,\partial_a(\sqrt{g}\,g^{ab}\,\partial_b)$ at $p_*$ is the second-order coefficient $g^{ab}|_{p_*}\,\partial_a\partial_b = (1/n)\sum_a \partial_a^2 = (1/n)\,\Delta_H$, proving (i).

For (ii), Lemma 3.1 gives $g_{p_*}(\chi_k, \overline{\chi_\ell}) = n^2\,\delta_{k\ell}$ on the character basis $\{\chi_k\}_{k=1}^{n-1}$ of $T_{p_*}\Delta_R^\circ \otimes \mathbb{C}$. Inverting the metric on this basis gives $g^{-1}_{p_*}(\chi_k^*, \overline{\chi_\ell^*}) = (1/n^2)\,\delta_{k\ell}$ on the dual basis $\{\chi_k^*\}$. The principal symbol of $\Delta_{\mathcal{M}}$ at $p_*$ is exactly this $g^{-1}_{p_*}$ viewed as a quadratic form on $T^*_{p_*}\Delta_R^\circ \otimes \mathbb{C}$, so it is diagonal in the dual character basis. The cross-packet vanishing follows from the partition $\{1,\dots,n-1\} = \bigsqcup_{d \mid n, d > 1} \mathcal{P}_d$ and linearity.

For (iii), any second-order linear differential operator on $\Delta_R^\circ$ whose leading coefficient is $g^{ab}$ has principal symbol $g^{ab}|_p\,\xi_a\xi_b$ at every $p$; at $p = p_*$ this inherits the diagonalization of (ii). $\square$

**Remark.** Corollary 3.10 generalizes the operational blind-spot beyond the discrete preconditioner setting of Corollaries 3.8 and 3.9 to the *principal symbol* of any $g$-built second-order differential operator on the categorical Fisher base. The claim is at the symbol level — the leading kinetic coefficient at $p_*$ — because sub-principal contributions (Christoffels, divergence terms) involve first derivatives of $g$ that depend on chart choice; the principal symbol is the canonical chart-independent $g$-built invariant, and it is what enters the kinetic action of every quadratic field theory on $(\Delta_R^\circ, g)$ and the leading-order behavior of every diffusion or heat-flow process driven by $g$. Any kinetic-energy, field-theoretic, or heat-flow construction built on $(\Delta_R^\circ, g)$ — including the nonlinear sigma model on statistical-manifold bases discussed in §7.5 — inherits the same conductor-packet block-diagonality of its principal symbol at the maximum-entropy point. The Amari–Chentsov cubic appears here, as in Theorem 3.5, at one order higher: as the first base-curvature tensor that *would* couple distinct packets if admitted into the kinetic action, and as the leading off-centroid correction to the principal symbol away from $p_*$ via Lemma 5.1.

**Proposition 3.11 (Head-side packet diagonality of the attribution input at $p_*$).** *On any categorical head with ring index $R = \mathbb{Z}/n\mathbb{Z}$, the back-propagated logit gradient that any gradient-based edge-attribution method contracts against — Edge Attribution Patching [Syed, Rager & Conmy 2024], its integrated-gradient refinement EAP-IG [Hanna, Pezzelle & Belinkov 2024], and activation-patching attribution E-ACT [Zhang & Nanda 2023] — passes at the head through the softmax Jacobian $J^{\mathrm{sm}}_p = \mathrm{diag}(p) - p\,p^\top$. At $p = p_*$, this head-side input vector*
$$
\tilde{\nabla}(x) \;:=\; J^{\mathrm{sm}}_{p_*}\, \nabla_{\mathrm{logit}}\!\mathcal{L}(x) \;\in\; T_{p_*}\Delta_R^\circ
$$
*decomposes orthogonally across conductor packets: writing $\tilde{\nabla}(x) = \sum_{d \mid n,\, d > 1} \tilde{\nabla}^{(d)}(x)$ with $\tilde{\nabla}^{(d)}(x) \in \mathcal{P}_d^{\mathbb{C}}$, the cross-packet pairings $\langle \tilde{\nabla}^{(d)}(x),\ \tilde{\nabla}^{(d')}(x)\rangle$ vanish identically for $d \ne d'$.*

*Proof.* At $p = p_*$, the softmax Jacobian is
$$
J_{p_*}^{\mathrm{sm}} \;=\; \tfrac{1}{n}\!\left( I_n - \tfrac{1}{n}\,\mathbf{1}\mathbf{1}^\top \right) \;=\; \tfrac{1}{n}\,\Pi,
$$
where $\Pi$ is the orthogonal projector onto $T_{p_*}\Delta_R^\circ$, so $\tilde{\nabla}(x) \in T_{p_*}\Delta_R^\circ$. By Lemma 3.1, the additive-character basis is orthogonal in the inner product on $T_{p_*}\Delta_R^\circ$ that $g_{p_*}$ rescales to $n^2$-diagonal, with the standard Euclidean inner product satisfying $\langle \chi_k,\overline{\chi_\ell}\rangle = n\,\delta_{k\ell}$; the conductor-packet decomposition $T_{p_*}\Delta_R^\circ \otimes \mathbb{C} = \bigoplus_d \mathcal{P}_d^{\mathbb{C}}$ is therefore orthogonal, and the cross-packet pairings of any vector's packet components vanish. $\square$

**Corollary 3.12 (Linear gradient-based attribution is blind to cubic content at $p_*$).** *Let $s : x \mapsto s(x) \in \mathbb{R}^{|E|}$ be a per-example edge-attribution score for the methods of Proposition 3.11, defined as a linear functional, or a path integral of linear functionals, of the head-side input $\tilde{\nabla}(x)$ — EAP and E-ACT use a single contraction; EAP-IG integrates linear contractions over the counterfactual interpolation. Then at $p_*$ the score $s(x)$ depends on $x$ only through $\tilde{\nabla}(x)$. In particular, two examples $x_1, x_2$ with identical head-side first-order gradients $\tilde{\nabla}(x_1) = \tilde{\nabla}(x_2)$, but whose local information geometries at $p_*$ differ only in the cross-packet cubic content of the Amari–Chentsov tensor $T_{p_*}$ (Theorem 3.5 part 2), produce identical attribution score vectors: $s(x_1) = s(x_2)$, regardless of the downstream network Jacobian.*

*Proof.* By hypothesis $s(x) = \mathcal{F}(\tilde{\nabla}(x))$ for some linear $\mathcal{F}$ (or a path integral of such); linearity gives the first claim immediately. The cross-packet cubic content invoked is a property of $T_{p_*}$, the third cumulant of the score (Lemma 3.4), and not of $g_{p_*}$ (Corollary 3.3): it is third-order in the displacement from $p_*$. A linear functional of the first-order gradient cannot resolve a third-order distinction in the local information geometry. $\square$

**Remark (ACDC).** Automated Circuit Discovery [Conmy et al. 2023] computes the metric *change* under counterfactual edge ablation, a nonlinear functional of the forward pass; Corollary 3.12's linearity argument applies cleanly only to the gradient-based methods (EAP, EAP-IG, E-ACT). For ACDC the same conclusion holds at first order around $p_*$: a Taylor expansion of the patching difference in the ablation magnitude recovers a linear contraction of $\tilde{\nabla}(x)$, so two examples with identical $\tilde{\nabla}(x)$ produce ACDC scores agreeing to leading order. The full nonlinear ACDC score can in principle resolve the cubic distinction at higher order, and we do not claim otherwise.

**Remark.** Proposition 3.11 and Corollary 3.12 together lift the static blind spot from the optimizer side (where Corollaries 3.8 and 3.9 act) to the interpretability side, on which gradient-based mechanism attribution lives — but the lift is at the *input* to attribution, not at the output edge-score covariance. The operational reading: when a categorical head with ring index processes two examples whose mechanisms agree on the head-side first-order gradient and differ only in the Amari–Chentsov tensor's cross-packet directions at $p_*$, any of the gradient-based attribution methods in the cited family will return identical attribution score vectors as functions of those inputs, regardless of what the downstream network Jacobian does. Aggregating such per-example scores across a mixed dataset (the standard hypothesis-driven workflow that returns one circuit per task by averaging $s_e$ across examples) is therefore *blind by construction* to mechanism differences resolvable only at the third-order cross-packet level, regardless of how faithful the resulting circuit appears on the head-output endpoint. This is the structural reason the §7.7 production-LLM result — a single discovered circuit reaching $87\%$ faithfulness on two structurally distinct mechanisms after a $10\%$ mixture injection — is consistent with Theorem 3.5 rather than anomalous to it.

### 3.5 What the result is and is not

We close this section with a remark on the scope of the certificate, repeated in formal voice in Section 9.

**Remark 3.13 (scope).** Theorem 3.5 is a statement about the *representational capacity of a model class at a single point*. It is not:

* a claim about any trained network's optimization trajectory (cf. Conjecture 5.8 and Section 8);
* a claim that the missed coupling carries gradient signal whose neglect costs performance — the certificate proves the coupling is *missed*, not that it is *task-relevant* (cf. Section 6 and Section 8);
* a claim that a second-order method which re-estimates curvature at the current $p$ is blind off $p_*$ in the same exact form — that is the off-centroid question of Section 5, addressed there as a leading-order expansion and a conjecture.

These distinctions are restated as scope statements in Section 9 and are load-bearing.

---

## 4. Computational verification

We verify the two halves of Theorem 3.5 on a ladder of five composite moduli using exact integer arithmetic.

### 4.1 Method

For each ring $R = \mathbb{Z}/n\mathbb{Z}$ in the ladder, the verification procedure:

1. Enumerates the conductor packets $\{\mathcal{P}_d : d \mid n,\ d > 1\}$ by computing $\gcd(k, n)$ for each $k \in \{1, \dots, n-1\}$.

2. Verifies Fisher block-diagonality (Lemma 3.1) by checking, for every pair $(k, \ell) \in \{1, \dots, n-1\}^2$ with $\mathrm{cond}(k) \ne \mathrm{cond}(\ell)$, that the integer predicate $(k - \ell) \bmod n \ne 0$ holds. By Lemma 3.1 this is equivalent to $g_{p_*}(\chi_k, \overline{\chi_\ell}) = 0$. The check is exact: no floating-point comparison enters.

3. Verifies Amari–Chentsov cross-packet coupling (Lemma 3.4 and Theorem 3.5) by enumerating all ordered triples $(k, \ell, m) \in \{1, \dots, n-1\}^3$ with $(k + \ell + m) \bmod n = 0$, partitioning them into *same-packet* (all three of $\mathrm{cond}(k), \mathrm{cond}(\ell), \mathrm{cond}(m)$ equal) and *cross-packet* (not all equal), and confirming that the cross-packet set is nonempty.

The procedure operates entirely on integers; every decision is a modular-arithmetic equality. The two pre-claimed falsifier branches before the run were: (i) a nonzero cross-packet entry of the Fisher form in the character basis (would falsify Lemma 3.1), and (ii) an empty cross-packet set of zero-sum triples for some composite $n$ in the ladder (would falsify part 2 of Theorem 3.5).

### 4.2 Results

| $n$ | Conductor packets ($d : |\mathcal{P}_d|$) | Cross-packet Fisher entries | Cross-packet cubic triples | Same-packet cubic triples |
|-----:|:------------------------------------------|:---------------------------:|:--------------------------:|:-------------------------:|
| $6$ | $2{:}1,\ 3{:}2,\ 6{:}2$ | $0$ (exact) | $18$ | $2$ |
| $8$ | $2{:}1,\ 4{:}2,\ 8{:}4$ | $0$ (exact) | $42$ | $0$ |
| $12$ | $2{:}1,\ 3{:}2,\ 4{:}2,\ 6{:}2,\ 12{:}4$ | $0$ (exact) | $108$ | $2$ |
| $18$ | $2{:}1,\ 3{:}2,\ 6{:}2,\ 9{:}6,\ 18{:}6$ | $0$ (exact) | $252$ | $20$ |
| $30$ | $2{:}1,\ 3{:}2,\ 5{:}4,\ 6{:}2,\ 10{:}4,\ 15{:}8,\ 30{:}8$ | $0$ (exact) | $774$ | $38$ |

The total number of ordered selection-rule triples is $(n-1)(n-2)$ by a one-line count: for each $(k, \ell) \in \{1, \dots, n-1\}^2$, $m$ is determined modulo $n$, and the only excluded case is $k + \ell \equiv 0$, giving $(n-1)^2 - (n-1)$. The same-packet contributions come primarily from "diagonal" triples $(k, k, k)$ with $3k \equiv 0 \pmod n$ (the elements of conductor packet $d = 3$ when $3 \mid n$) and, for prime-power-depth rings such as $n = 18$, from shapes $(k, k, m)$ with $2k + m \equiv 0$ where both $k$ and $m$ lie in the same large packet of order coprime to the small prime divisor.

For each ring, every cross-packet Fisher entry is exactly zero, confirming Lemma 3.1, and the cross-packet cubic-triple set is nonempty, confirming Theorem 3.5 part 2. Neither falsifier branch was realized. Sample cross-packet triples for each ring (chosen to exhibit as many distinct conductor labels as the ring admits) are recorded in Appendix C.

### 4.3 Reproducibility

The verification is a deterministic enumeration with no random seeds, no tolerances, and no floating-point in any vanishing decision. Source code and a full run transcript are available with the manuscript. The computational scope of this paper is precisely Section 4 plus Appendix C; everything else is analytic.

---

## 5. The maximum-entropy point as a base case

The certificate of Section 3 holds exactly at $p_*$. The natural objection is that training does not sit at maximum entropy. Section 5 addresses this precisely, and does not overclaim the resolution.

### 5.1 Why $p_*$ is where the result is provable

At $p_*$ the conductor packets are *exactly* Fisher-orthogonal (Lemma 3.1), and the cross-packet content of the cubic is decided by an exact integer-arithmetic predicate (Lemma 3.4). The maximum-entropy point is the unique configuration at which the block-diagonality of $g$ is an *identity* in the character basis rather than an approximation; this is why a clean theorem is available there and only there.

By the same construction, $p_*$ is the configuration of maximum predictive uncertainty: the model has learned nothing and emits the uniform distribution. The blind spot exists at $p_*$ but is *dynamically dormant* there — there is no learned structure for the missed coupling to attach to, and so the missing capacity is observably costless. The base case is, deliberately, the configuration at which the gap is provable precisely because it is not yet load-bearing.

### 5.2 The off-centroid expansion and its Fourier selection rule

We compute what happens at first order in a displacement off $p_*$, first as a basis-free identity (Lemma 5.1), then in the additive-character basis where it becomes a per-frequency activation rule (Theorem 5.2).

**Lemma 5.1 (off-centroid Fisher expansion).** *Let $p = p_* + h$ with $h \in T_{p_*}\Delta_R^\circ$ a tangent displacement, $\|h\|_\infty < 1/n$ so that $p \in \Delta_R^\circ$. Then for all $u, v \in T_{p_*}\Delta_R^\circ$,*
$$
g_p(u, v) = g_{p_*}(u, v) \;-\; T_{p_*}(h, u, v) \;+\; O(\|h\|^2). \tag{5.1}
$$

*Proof.* Expand $1/p_y = 1/(p_{*, y} + h_y)$ as a geometric series in $h_y / p_{*, y}$, valid for $|h_y| < p_{*, y} = 1/n$:
$$
\frac{1}{p_y} = \frac{1}{p_{*, y}} \cdot \frac{1}{1 + h_y / p_{*, y}} = \frac{1}{p_{*, y}} - \frac{h_y}{p_{*, y}^2} + O(h_y^2 / p_{*, y}^3).
$$
Multiplying by $u_y v_y$ and summing over $y$:
$$
g_p(u, v) = \sum_y \frac{u_y v_y}{p_y} = \sum_y \frac{u_y v_y}{p_{*, y}} - \sum_y \frac{u_y v_y h_y}{p_{*, y}^2} + O(\|h\|^2),
$$
where the first sum is $g_{p_*}(u, v)$ by definition and the second sum is $T_{p_*}(h, u, v)$ by definition. (5.1) follows. A full residual bound is given in Appendix D. $\square$

The structural content of Lemma 5.1 is that *the leading correction to the Fisher form as one leaves the maximum-entropy point is exactly the Amari–Chentsov cubic contracted with the displacement.* The very term that turns the off-centroid Fisher metric away from block-diagonal across conductor packets is the cubic that the second-order model class cannot represent. Read in the additive-character basis, the same identity becomes a quantitative per-frequency selection rule.

**Theorem 5.2 (off-centroid Fourier selection rule).** *Let $h \in T_{p_*}\Delta_R^\circ$ have additive-character expansion*
$$
h_y \;=\; \sum_{\substack{a \in \mathbb{Z}/n\mathbb{Z} \\ a \ne 0}} \hat h_a\, \chi_a(y), \qquad \hat h_a \;=\; \frac{1}{n}\sum_{y \in R} h_y\, \overline{\chi_a(y)}.
$$
*Then for distinct $k, \ell \in \{1, \dots, n-1\}$,*
$$
g_{p_* + h}(\chi_k, \overline{\chi_\ell}) \;=\; -\, n^3\, \hat h_{(\ell - k) \bmod n} \;+\; O(\|h\|^2). \tag{5.2}
$$
*Hence the off-diagonal entry between $\chi_k$ and $\chi_\ell$ in $g_{p_* + h}$ activates at leading order in $h$ if and only if $h$ has nonzero Fourier coefficient at frequency $(\ell - k) \bmod n$.*

*Proof.* By Lemma 5.1,
$$
g_{p_* + h}(\chi_k, \overline{\chi_\ell}) = g_{p_*}(\chi_k, \overline{\chi_\ell}) - T_{p_*}(h, \chi_k, \overline{\chi_\ell}) + O(\|h\|^2).
$$
For $k \ne \ell$, Lemma 3.1 gives $g_{p_*}(\chi_k, \overline{\chi_\ell}) = 0$. For the cubic term, $\overline{\chi_\ell} = \chi_{-\ell}$, and expanding $h$ in characters and applying Lemma 3.4:
$$
T_{p_*}(h, \chi_k, \overline{\chi_\ell}) \;=\; \sum_{a \ne 0} \hat h_a\, T_{p_*}(\chi_a, \chi_k, \chi_{-\ell}) \;=\; \sum_{a \ne 0} \hat h_a\, n^3 \cdot \mathbf{1}\!\left[\, a + k - \ell \equiv 0 \pmod n \,\right].
$$
Exactly one value of $a \in \{1, \dots, n-1\}$ satisfies the integer predicate, namely $a = (\ell - k) \bmod n$ (which is nonzero since $k \ne \ell$). Hence $T_{p_*}(h, \chi_k, \overline{\chi_\ell}) = n^3\, \hat h_{(\ell - k) \bmod n}$, and (5.2) follows. $\square$

**Remark 5.3 (packet difference sets).** Theorem 5.2 admits an immediate packet-level reformulation. For divisors $d_1, d_2 \mid n$ with $d_1, d_2 > 1$, define the *packet difference set*
$$
\mathcal{P}_{d_2} - \mathcal{P}_{d_1} \;:=\; \{\, (\ell - k) \bmod n : k \in \mathcal{P}_{d_1},\ \ell \in \mathcal{P}_{d_2},\ k \ne \ell \,\} \;\subset\; \mathbb{Z}/n\mathbb{Z} \setminus \{0\}.
$$
Then a displacement $h$ activates the cross-packet block of $g_{p_* + h}$ between $\mathcal{P}_{d_1}$ and $\mathcal{P}_{d_2}$ at leading order if and only if $h$ has nonzero Fourier mass on $\mathcal{P}_{d_2} - \mathcal{P}_{d_1}$. The activation is thus controlled by a finite, purely number-theoretic invariant of the pair $(d_1, d_2)$: which displacement frequencies can break the cross-packet block-diagonality, and which cannot. Applied to an update direction $u$ averaged over an evaluation batch (Section 6), Theorem 5.2 turns the cross-packet cubic mass from a heuristic third-cumulant ratio into a per-frequency, per-packet-pair quantity directly observable on existing checkpoints.

### 5.3 The all-orders expansion

We now extend the leading-order analysis to all orders in $h$. The result is a clean analytic representation of $g_{p_* + h}$ as an absolutely convergent series whose $j$-th term is a contraction with the $(j + 2)$-th score-moment tensor at $p_*$, and a closed Fourier-convolution expression for the cross-packet entry. In consequence, the off-centroid geometry of $g$ along any ray $p(t) = p_* + t h$ becomes a deterministic analytic function of the Fourier spectrum of $h$.

**Lemma 5.5 (score-moment tensors at $p_*$).** *For $k \geq 2$, the $k$-th moment of the categorical score at $p_*$ is the symmetric $k$-tensor $M^{(k)}_{p_*} : (T_{p_*}\Delta_R^\circ)^k \to \mathbb{R}$ given by*
$$
M^{(k)}_{p_*}(w_1, w_2, \dots, w_k) \;=\; n^{k - 1} \sum_{y \in R} w_{1, y}\, w_{2, y} \cdots w_{k, y}. \tag{5.3}
$$
*In particular $M^{(2)}_{p_*} = g_{p_*}$ and $M^{(3)}_{p_*} = T_{p_*}$ recover the Fisher and Amari–Chentsov forms. In the additive-character basis,*
$$
M^{(k)}_{p_*}(\chi_{a_1}, \chi_{a_2}, \dots, \chi_{a_k}) \;=\; n^k \cdot \mathbf{1}\!\left[\, a_1 + a_2 + \cdots + a_k \equiv 0 \pmod n \,\right]. \tag{5.4}
$$
*The $k$-th score-moment tensor at $p_*$ is supported on character $k$-tuples whose indices sum to zero modulo $n$.*

*Proof.* At $p_*$, $\partial_{w} \log p(y) = w_y / p_{*, y} = n w_y$. Hence
$$
M^{(k)}_{p_*}(w_1, \dots, w_k) \;=\; \mathbb{E}_{Y \sim p_*}\!\left[\prod_{i = 1}^{k} \partial_{w_i} \log p(Y)\right] \;=\; \sum_{y \in R} \frac{1}{n} \prod_{i = 1}^{k} (n w_{i, y}) \;=\; n^{k - 1} \sum_y \prod_{i = 1}^{k} w_{i, y},
$$
which is (5.3). For (5.4), substitute $w_i = \chi_{a_i}$, so $\prod_i \chi_{a_i}(y) = \chi_{a_1 + \cdots + a_k}(y)$; the sum over $y$ is $n$ if $a_1 + \cdots + a_k \equiv 0 \pmod n$ and zero otherwise, by Lemma 2.1. $\square$

**Theorem 5.6 (all-orders off-centroid Fisher expansion).** *Let $h \in T_{p_*}\Delta_R^\circ$ with $\|h\|_\infty < 1/n$, and write $\hat h_a = \frac{1}{n}\sum_y h_y \overline{\chi_a(y)}$ for its additive-character coefficients. Then for all $u, v \in T_{p_*}\Delta_R^\circ$, the Fisher form at $p_* + h$ admits the absolutely convergent series*
$$
g_{p_* + h}(u, v) \;=\; \sum_{j = 0}^{\infty} (-1)^j\, M^{(j + 2)}_{p_*}\!\left(\underbrace{h, h, \dots, h}_{j},\, u, v\right). \tag{5.5}
$$
*For distinct $k, \ell \in \{1, \dots, n - 1\}$, the cross-packet entry has the closed Fourier-convolution form*
$$
g_{p_* + h}(\chi_k, \overline{\chi_\ell}) \;=\; \sum_{j = 1}^{\infty} (-1)^j\, n^{j + 2}\, (\hat h^{*j})_{(\ell - k) \bmod n}, \tag{5.6}
$$
*where $\hat h^{*j}$ denotes the $j$-fold cyclic convolution of $\hat h$ on $\mathbb{Z}/n\mathbb{Z}$. The $j = 1$ term recovers Theorem 5.2; the terms $j \geq 2$ resolve the residual $O(\|h\|^2)$ to all orders.*

*Proof.* From $p_{*, y} = 1/n$ and $p_y = p_{*, y} + h_y$, write $p_y = (1 + n h_y)/n$, hence $1/p_y = n / (1 + n h_y)$. For $|n h_y| < 1$ — equivalently $\|h\|_\infty < 1/n$ — the geometric series $1/(1 + n h_y) = \sum_{j \geq 0} (- n h_y)^j$ converges absolutely. Substituting and exchanging the (absolutely convergent) sums,
$$
g_p(u, v) \;=\; \sum_y \frac{u_y v_y}{p_y} \;=\; n \sum_y u_y v_y \sum_{j \geq 0} (- n h_y)^j \;=\; \sum_{j \geq 0} (-1)^j\, n^{j + 1} \sum_y u_y v_y\, h_y^{\,j}.
$$
By Lemma 5.5, the inner sum equals $n^{- (j + 1)}\, M^{(j + 2)}_{p_*}(h, \dots, h, u, v)$; substituting gives (5.5).

For (5.6), apply (5.5) with $u = \chi_k$, $v = \overline{\chi_\ell} = \chi_{-\ell}$, and unfold the $j$-th term using (5.3):
$$
(-1)^j\, n^{j + 1} \sum_y \chi_k(y)\, \chi_{-\ell}(y)\, h_y^{\,j} \;=\; (-1)^j\, n^{j + 1} \sum_y \chi_{k - \ell}(y)\, h_y^{\,j}.
$$
Expanding the pointwise $j$-fold product in the character basis, $h_y^{\,j} = \sum_c (\hat h^{*j})_c\, \chi_c(y)$ — the Fourier coefficients of a pointwise product are the cyclic convolution of the factors' Fourier coefficients — the inner sum becomes
$$
\sum_c (\hat h^{*j})_c \sum_y \chi_{k - \ell + c}(y) \;=\; n \cdot (\hat h^{*j})_{(\ell - k) \bmod n}
$$
by Lemma 2.1. The $j = 0$ term contributes $n^2 (\hat h^{*0})_{\ell - k} = n^2 \delta_{k, \ell} = 0$ for $k \ne \ell$, so the sum starts at $j = 1$; this gives (5.6). $\square$

**Remark 5.7 (closed form, frequency containment, dynamical scope).** Three structural consequences follow from Theorem 5.6.

*(i) Closed form.* For $k \ne \ell$, (5.6) is the formal Fourier-convolution expansion of the exact closed expression
$$
g_{p_* + h}(\chi_k, \overline{\chi_\ell}) \;=\; n \sum_{y \in R} \frac{\chi_{k - \ell}(y)}{1 + n h_y}, \qquad \|h\|_\infty < 1/n.
$$
The cross-packet entry is exactly a $\chi_{k - \ell}$-weighted average of $1/(1 + n h_y)$.

*(ii) Frequency containment by subgroups.* If $\hat h$ is supported on a subset $S \subseteq \mathbb{Z}/n\mathbb{Z}$, then $\hat h^{*j}$ is supported on the $j$-fold sumset $S + S + \cdots + S$. When $S$ generates a proper subgroup $H \leq \mathbb{Z}/n\mathbb{Z}$, every convolution $\hat h^{*j}$ remains in $H$, and the cross-packet entry $g_{p_* + h}(\chi_k, \overline{\chi_\ell})$ is identically zero at every order in $h$ whenever $(\ell - k) \bmod n \notin H$. A displacement with frequency support in a divisor subgroup therefore activates only the cross-packet entries whose frequency difference lies in that subgroup, and leaves the others untouched at every order.

*(iii) Off-centroid geometry as a function of $\hat h$.* Along any ray $p(t) = p_* + t h$, the cross-packet entry $g_{p(t)}(\chi_k, \overline{\chi_\ell})$ is an analytic function of $t$ on the disk $|t| < 1/(n \|h\|_\infty)$, fully determined by the sequence of convolution values $\{(\hat h^{*j})_{(\ell - k) \bmod n}\}_{j \geq 1}$. The off-centroid geometry of $g$ along any ray is therefore a closed-form, deterministic function of the Fourier spectrum of $h$; the open question is no longer "what does the geometry off $p_*$ look like" but, narrowly, "what $\hat h(t)$ does a real training trajectory accumulate".

### 5.4 Conjecture: off-centroid persistence on training trajectories

Theorems 5.2 and 5.6 give the off-centroid geometry of $g$ along any ray $p(t) = p_* + t h$ as a closed analytic function of the Fourier spectrum of $h$. A second-order method that re-estimates the exact Fisher at the current $p(t)$ absorbs this geometry, but does so as a sequence of *static quadratic snapshots* — one per training step — with no model of how the geometry evolves with $t$. The rate of change is governed by the cubic at leading order (Lemma 5.1) and by the full tower of score-moment tensors $\{M^{(j+2)}_{p_*}\}_{j \geq 1}$ to all orders (Theorem 5.6), each of which lies outside the quadratic model class.

What is therefore not finite-dimensional linear algebra — and what we state as a conjecture rather than a theorem — is what the running Fourier spectrum $\hat h(t)$ of a real training trajectory on a ring-structured task actually accumulates, and whether the corresponding cross-packet activation curves are distinguishable from those on a matched label-permuted control.

**Conjecture 5.8 (off-centroid persistence on ring-structured trajectories).** *Let $\{p(t)\}_{t \geq 0}$ be a training trajectory of a categorical head on a ring-structured task with displacement $h(t) := p(t) - p_*$. Then on the ring-structured task, the Fourier mass $\sum_{a \in \mathcal{P}_{d_2} - \mathcal{P}_{d_1}} |\hat h(t)_a|^2$ of the running displacement on each packet difference set $\mathcal{P}_{d_2} - \mathcal{P}_{d_1}$ (for distinct $d_1, d_2 \mid n$, $d_1, d_2 > 1$) grows with $t$ at a rate distinguishable from the rate on a matched label-permuted control task; equivalently, the cross-packet activation of $g_{p(t)}$ predicted by Theorem 5.6 becomes detectable on the ring-structured task and remains null on the control. Since this cross-packet activation is, by Corollary 3.3 and Theorem 5.6, outside the representational capacity of a second-order method modelling the time-evolution of $g$ as a quadratic, we conjecture that this absence manifests as a measurable acquisition-rate asymmetry between ring-structured and non-ring directions in the training dynamics, present on the ring-structured task and null on the control.*

Conjecture 5.8 is the conjecture this paper *does not prove* and pre-specifies as the hypothesis of the experiment in Section 8. Theorems 5.2 and 5.6 settle the analytic side of the off-centroid question: for any fixed displacement $h$, the cross-packet activation curve $t \mapsto g_{p_* + t h}(\chi_k, \overline{\chi_\ell})$ is a closed-form analytic function of $t$ with explicit Fourier-convolution coefficients. What remains is the *dynamical* question of what $\hat h(t)$ a real training trajectory produces. The relationship between Theorem 3.5 and Conjecture 5.8 is the relationship between a clean base case and an unproven dynamical claim; Theorems 5.2 and 5.6 close the analytic side along any ray but do not address what trajectory the dynamics produce.

It is worth stating sharply what the certificate is and is not a claim about. Theorem 3.5 is a *representational* fact about the quadratic curvature class, not a claim about where descent stalls: whether the missing cross-packet capacity is dynamically load-bearing — or, as the §7.8 within-mode acceleration line suggests, descent on these cyclic tasks lives within a single conductor packet and the cross-packet null is dynamically inert — is exactly the question Conjecture 5.8 leaves open. A reading on which the cross-packet null is itself the *bottleneck* a second-order optimizer is blocked by is stronger than anything the certificate asserts; the certificate is consistent both with that reading and with its opposite, in which descent is well-aligned to the within-packet subspace and the cross-packet null is a representational fact that no intervention need supply.

**Concordance with recent trajectory-dynamics work.** Two recent empirical lines provide candidate trajectory statistics that the analytic content of Theorems 5.2 and 5.6 maps directly onto. [Truong et al. 2026a] identify a *normalised spectral entropy* of the representation covariance $\widetilde{H}(t) = H(t)/\log d$ whose collapse below a task-specific threshold $\widetilde{H}^*$ precedes generalisation across both abelian ($\mathbb{Z}/p\mathbb{Z}$) and non-abelian ($S_5$) tasks, with a representation-mixing intervention delaying grokking by $+5\,020$ steps ($p = 0.044$); they also report a strong anti-correlation ($\overline{\rho} = -0.82$) between $\widetilde{H}$ and a Fourier-alignment observable on cyclic-group tasks, identifying spectral entropy as a basis-free signature of Fourier-structure formation. [Truong et al. 2026b] give a *first-passage law* for the grokking delay under AdamW as a crossing problem in $(V_t, \alpha_t)$-space, where $V_t = \|\theta_t\|^2$ contracts exponentially and an angular threshold $\alpha_t = \alpha^*$ must also be reached. Both are dynamical observables of the running displacement $h(t)$; the cross-packet entries of $g_{p(t)}$ predicted by Theorem 5.6 are the analytic skeleton these empirical signatures probe. The conjecture as stated above is intentionally narrower than either of these published claims — it predicts a measurable rate asymmetry on packet difference sets between a ring task and a label-permuted control — but the analytic content (Fourier-convolution structure of off-diagonal $g$ entries) is the right primitive for whatever finer dynamical statement the trajectory data ultimately supports. The §8.6 experiment instruments interaction effects on a coarser endpoint than these trajectory observables; an extension that instruments $\widetilde{H}(t)$ and $V_t$ simultaneously with $\rho_\times$ is listed in §8.6's open follow-ups.

<!--
Internal note: Theorems 5.2 and 5.6 together close the off-centroid
analytic question along ANY ray p(t) = p_* + t*h as a closed-form
function of h's Fourier spectrum. What remains genuinely open is the
DYNAMICAL question: what spectrum hat-h(t) does a training trajectory
accumulate on a real ring-structured task vs. on a label-permuted
control? That is Conjecture 5.8, pre-specified in Section 8. With both
analytic halves now proven, the claim surface is substantially smaller
than the v0.1 of this paper, and the diagnostic of Section 6 has clean
mathematical underpinnings (the Fourier coefficients computed there are
exactly the inputs to Theorem 5.6's convolution series).
-->

---

## 6. A retraining-free diagnostic

The certificate hands us a measurement instrument. We specify the diagnostic (Definitions 6.1–6.2) as a deterministic computation derived from Theorem 3.5 and demonstrate its computability and predicted behavior on the canonical modular-addition testbed (§6.5). What §6.5 does not do is the broader sweep across public checkpoints with calendar/clock heads, which remains future work. The diagnostic's empirical interpretation — whether a large value indicates a *consequential* deficit — is the question of Section 8 and is not presupposed.

### 6.1 The cross-packet cubic mass

Let $M$ be a trained model with a categorical head over $R = \mathbb{Z}/n\mathbb{Z}$, let $\mathcal{B}$ be an evaluation batch, and for each example in $\mathcal{B}$ let $u^{(i)} := e_{y^{(i)}} - p^{(i)}$ be the centered score (the logit gradient of cross-entropy at the model's prediction $p^{(i)}$ given the label $y^{(i)}$). Let $u := \frac{1}{|\mathcal{B}|}\sum_i u^{(i)}$ be the mean update direction on the simplex tangent space at the batch-averaged prediction $\bar{p}$. Compute the discrete additive-character transform

$$
\hat{u}_k := \sum_{y \in R} u_y\, \overline{\chi_k(y)}, \qquad k \in \mathbb{Z}/n\mathbb{Z},\ k \ne 0.
$$

Group the coefficients by conductor: for each $d \mid n$, $d > 1$, the conductor-$d$ component of $u$ is the projection onto $\mathcal{P}_d$ given by $\{\hat{u}_k\}_{k \in \mathcal{P}_d}$.

Let $\mathcal{T} := \{(k, \ell, m) \in \{1, \dots, n-1\}^3 : k + \ell + m \equiv 0 \pmod n\}$ be the set of *surviving* zero-sum triples (those on which the Amari–Chentsov cubic at $p_*$ does not vanish, by Lemma 3.4), and let $\mathcal{T}_\times \subseteq \mathcal{T}$ be the subset with $(\mathrm{cond}(k), \mathrm{cond}(\ell), \mathrm{cond}(m))$ not all equal.

**Definition 6.1 (cross-packet cubic mass).** *The cross-packet cubic mass of the update direction $u$ at the prediction point $\bar{p}$ is the ratio of L1 spectral masses*
$$
\rho_\times(u;\, \bar{p}) := \frac{\displaystyle\sum_{(k, \ell, m)\, \in\, \mathcal{T}_\times} \big| T_{\bar{p}}(\chi_k, \chi_\ell, \chi_m)\, \hat{u}_k \hat{u}_\ell \hat{u}_m \big|}{\displaystyle\sum_{(k, \ell, m)\, \in\, \mathcal{T}} \big| T_{\bar{p}}(\chi_k, \chi_\ell, \chi_m)\, \hat{u}_k \hat{u}_\ell \hat{u}_m \big|},
$$
*over the selection-rule-restricted triples of Lemma 3.4, with the cross-packet subset $\mathcal{T}_\times \subseteq \mathcal{T}$ in the numerator.*

$\rho_\times$ is a ratio of *absolute-value-summed* spectral masses over $\mathcal{T}$; it is not literally a fraction of the signed third-cumulant contraction $T_{\bar p}(u, u, u) = \sum_{k, \ell, m} T_{\bar p}(\chi_k, \chi_\ell, \chi_m)\, \hat u_k \hat u_\ell \hat u_m$. Three structural gaps separate the two quantities and are by design:

* *Cancellations.* The numerator and denominator discard the phases of the individual triple contributions and are therefore $\geq$ the absolute value of the corresponding signed sums, with equality only when all triple terms in the support share a common phase. The diagnostic is built on the absolute-value sum because we want to measure how much cubic structure on the selection-rule subspace the optimizer would have to address, not the net signed third-cumulant value that batch-level phase cancellations can drive to zero — a phenomenon explicitly observed in §6.6 (the batch-mean form fluctuates symmetrically about zero when prompts are cyclically balanced).

* *Basis normalization.* The character coefficients use the convention $\hat u_k = \sum_y u_y\, \overline{\chi_k(y)}$ (no $1/n$ prefactor); both numerator and denominator carry the same homogeneity in this convention, so $\rho_\times$ is invariant under the conventional $1/n$ rescaling between $u$-space and $\hat u$-space.

* *Off-centroid non-selection-rule contributions, and the $p_*$-anchored proxy.* At $\bar p = p_*$, Lemma 3.4 gives $T_{p_*}(\chi_k, \chi_\ell, \chi_m) = n^3 \cdot \mathbf{1}[k + \ell + m \equiv 0 \pmod n]$, so restricting the spectral sum to $\mathcal{T}$ captures the entire tensor support; the $T$-factor becomes the constant $n^3$ on every surviving triple and cancels from numerator and denominator, reducing $\rho_\times(u; p_*)$ to the $|\hat u_k \hat u_\ell \hat u_m|$-weighted ratio on $\mathcal{T}_\times$ vs.\ $\mathcal{T}$. Off $p_*$, the $1/p_y^2$ weighting in the coordinate expression for $T_{\bar p}$ produces nonzero $T_{\bar p}(\chi_k, \chi_\ell, \chi_m)$ even on triples with $k + \ell + m \not\equiv 0 \pmod n$ — at $\bar p \ne p_*$, $T_{\bar p}$ is *not* supported only on the zero-sum selection rule. Definition 6.1 restricts the spectral mass to $\mathcal{T}$ regardless, and is therefore a ***$p_*$-anchored proxy***: it tracks the cross-packet share of the L1 spectral mass of $T_{\bar p}$ on the selection-rule subspace fixed by Lemma 3.4 at $p_*$, not the full off-centroid tensor support. We adopt the proxy form as primary because the certificate of Theorem 3.5 part 2 is itself an identity on the selection-rule subspace, the cross-packet null structure of the quadratic curvature class lives there, and the cubic-aware correction motivated in §8.3 targets precisely those selection-rule directions; the proxy keeps the diagnostic, the certificate, and the candidate correction in the same subspace.

**Optional all-triples variant.** A reader who wants the full off-centroid cubic spectral mass — including the selection-rule-violating contributions $T_{\bar p}(\chi_k, \chi_\ell, \chi_m)$ with $k + \ell + m \not\equiv 0 \pmod n$ at $\bar p \ne p_*$ — may use the obvious extension
$$
\rho_\times^{\mathrm{full}}(u;\, \bar p) \;:=\; \frac{\displaystyle\sum_{(k, \ell, m)\, \in\, \mathcal{T}_\times^{\mathrm{full}}} \big| T_{\bar{p}}(\chi_k, \chi_\ell, \chi_m)\, \hat{u}_k \hat{u}_\ell \hat{u}_m \big|}{\displaystyle\sum_{(k, \ell, m)\, \in\, \{1, \dots, n-1\}^3} \big| T_{\bar{p}}(\chi_k, \chi_\ell, \chi_m)\, \hat{u}_k \hat{u}_\ell \hat{u}_m \big|},
$$
with $\mathcal{T}_\times^{\mathrm{full}}$ the set of all ordered triples of nontrivial-character indices whose conductor labels are not all equal (no selection-rule restriction). $\rho_\times^{\mathrm{full}}$ coincides with $\rho_\times$ exactly at $\bar p = p_*$ (the off-$\mathcal{T}$ tensor entries vanish there by Lemma 3.4) and differs in general off $p_*$ by the amount of cubic spectral mass that $T_{\bar p}$ carries on selection-rule-violating triples. The empirical measurements in §6.5–§6.7 use the $p_*$-anchored proxy $\rho_\times$ throughout, for the structural reason given above and to keep the diagnostic comparable across training steps and across rings; the all-triples variant is recorded here as the natural alternative for readers who want to track the full off-centroid spectral support directly.

These choices make $\rho_\times$ a $p_*$-anchored structural diagnostic of the conductor blind spot — the share of the selection-rule L1 spectral mass of the cubic on $u$ that the quadratic curvature model class cannot couple (Corollary 3.3, lifted across the four-tier scope of §2.3) — and not an attempt to estimate the value of $T_{\bar p}(u, u, u)$. At $\bar p = p_*$, with the $T$-factor constant on $\mathcal{T}$, the diagnostic reduces to a $|\hat u|$-weighted structural ratio and inherits the degeneracy boundaries of the divisor lattice (prime $n$, the $n = 8$-style same-packet-empty case; see §6.5).

### 6.2 The preconditioner discard ratio

Let $M^{-1}$ be the curvature model used by the deployed second-order preconditioner (the inverse of the Fisher, K-FAC factors, the diagonal empirical Fisher, etc., restricted or lifted to the output tangent space). Let $u_{\mathrm{prec}} := M^{-1} u$ be the preconditioned update.

**Definition 6.2 (preconditioner discard ratio).** *The preconditioner discard ratio of $M^{-1}$ on the update direction $u$ at $\bar p$ is the relative change in the cross-packet cubic mass share between the raw update and its preconditioned counterpart:*
$$
\delta(u, u_{\mathrm{prec}};\, \bar{p}) \;:=\; 1 \;-\; \frac{\rho_\times(u_{\mathrm{prec}};\, \bar p)}{\rho_\times(u;\, \bar p)}, \qquad \text{provided } \rho_\times(u;\, \bar p) > 0,
$$
*where $\rho_\times$ is the cross-packet cubic mass of Definition 6.1. $\delta > 0$ indicates the preconditioner attenuates the cross-packet cubic-mass share relative to within-packet (literally: $\rho_\times(u_{\mathrm{prec}}) < \rho_\times(u)$); $\delta = 0$ indicates a share-preserving preconditioner; $\delta < 0$ indicates one that amplifies the cross-packet share.*

The definition uses the L1-spectral-mass diagnostic $\rho_\times$ of Definition 6.1 evaluated on $u$ and $u_{\mathrm{prec}}$, and is therefore well-specified through the conductor-packet decomposition of the spectrum without invoking a "cross-packet subspace" of the tangent space. The latter is not a canonical object: "cross-packet" is a property of triples in $\mathcal{T}_\times$ (and of operator blocks acting between distinct packets), not of vectors — a vector $u \in T_{\bar p}\Delta_R^\circ$ decomposes canonically into packet components $u = \sum_d u^{(d)}$ with each $u^{(d)} \in \mathcal{P}_d^{\mathbb{C}}$, but no canonical "cross-packet" component lives outside any packet.

**Operational reading at $p_*$.** At $\bar p = p_*$, the Fisher form $g_{p_*}$ is a scalar multiple of the identity on $T_{p_*}\Delta_R^\circ$ (Lemma 3.1, Corollary 3.8); any tier-T1 preconditioner whose curvature surrogate equals $g_{p_*}$ on the output tangent space (§2.3) therefore acts as a scalar on $u$, so $u_{\mathrm{prec}} \propto u$, $\hat u_{\mathrm{prec},\,k} \propto \hat u_k$, the L1 spectral masses on $\mathcal{T}_\times$ and $\mathcal{T}$ scale by the same factor in numerator and denominator of $\rho_\times$, and $\rho_\times(u_{\mathrm{prec}}; p_*) = \rho_\times(u; p_*)$. Hence $\delta = 0$. The certificate's representational claim — that the quadratic class cannot couple cross-packet directions — manifests at the *operator level* as the block-diagonality of $g_{p_*}$ (Corollary 3.3); at the *update level*, the operational consequence is that a tier-T1 preconditioner has no cross-packet entries to differentially apply, and $\delta = 0$ is the correct signature, not $\delta = 1$. Off $p_*$, the Fisher form $g_{\bar p}$ acquires cross-packet entries via Theorem 5.6; the deployed $M^{-1}$ may approximate these or not, and $\delta \ne 0$ then measures how the deployed curvature surrogate interacts with the off-centroid cross-packet structure of the true Fisher (positive when the deployed preconditioner suppresses the cross-packet share beyond what scalar action would produce, negative when it amplifies).

**Remark 6.3 (scope of $\rho_\times$ vs. $\delta$ across preconditioner classes).** The headline diagnostic $\rho_\times$ of Definition 6.1 does not depend on the preconditioner $M^{-1}$ at all — it is computed from the head logit gradient $u$ and the batch-averaged prediction $\bar p$ alone — and is therefore the robust measurement on any optimizer, Adam/AdamW included; the empirical demonstrations of §6.5 and §6.6 apply $\rho_\times$ directly to Adam-trained models for exactly this reason. The secondary diagnostic $\delta$ of Definition 6.2 is well-defined for any $M^{-1}$ but is informative only when $M^{-1}$ has off-diagonal structure to act with. On full or block-structured curvature surrogates — natural gradient with the exact Fisher, K-FAC, Gauss–Newton, Shampoo — $\delta$ measures the substantive question of how the deployed preconditioner interacts with the off-centroid cross-packet structure of the true Fisher (Theorem 5.6). On *diagonal* preconditioners such as Adam's empirical-Fisher proxy, $\delta$ is structurally moot rather than merely noisy: a diagonal $M^{-1}$ has no off-diagonal cross-packet entries to differentially apply, so the quantity $\delta$ is designed to detect — differential attenuation of the cross-packet share by the preconditioner's block structure — is null by construction. The familiar pathology of Adam's $1/\sqrt{v + \varepsilon}$ blowing up in memorised directions (per-coordinate variance $v \to 0$ on held-out data) compounds the issue but is not its source. We therefore recommend that $\delta$ be reported with $M^{-1}$ disclosed and treated as informative only when $M^{-1}$ is full or block-structured; we do *not* recommend any retreat from $\rho_\times$ on Adam-trained substrates. The moot-ness of $\delta$ on Adam dovetails with the §8.6 reading that Adam's per-coordinate $1/\sqrt{v}$ scaling on the head logits *implicitly* absorbs much of what an explicit packet-aware correction would supply: under that reading, $\delta = 0$ on diagonal $M^{-1}$ is the correct signature of a preconditioner that has already addressed the relevant structure through a non-block channel, not a defect of the measurement.

### 6.3 Protocols

The diagnostic is paired with two ring-ablation controls — a weak one (label-only permutation) and a strong one (label-and-input permutation, or a structureless random target). The pairing is itself informative: a sufficiently capable network can sometimes bypass the weak control by internalizing a ring representation upstream of the permuted label, in which case the gap between the weak control and the ring-respecting task collapses; the strong control blocks this bypass and is the binding falsifier for the diagnostic.

**D1 (checkpoint sweep).** For a family of public checkpoints of a model with a ring-structured categorical head, compute $(\rho_\times, \delta)$ as a function of training step. Output: a curve. No performance claim is attached to the curve's shape in this paper; the curve *is* the measurement.

**D2-weak (label-only basis ablation).** Re-compute D1 after applying a fixed random permutation $\sigma$ of the *label set* that destroys the additive-group structure on the output while preserving dimension, entropy, and sample statistics. If the diagnostic depends on label-side ring structure, $\rho_\times$ in D2-weak should differ from $\rho_\times$ in D1. D2-weak is sensitive but bypassable: on tasks where the network can learn an internal ring representation upstream and then compose with $\sigma$, the gap can collapse — a phenomenon documented for the modular-addition testbed at $n = 30$ in the demo of Section 6.5.

**D2-strong (label-and-input basis ablation).** Replace the target by a *structureless random function* of the input — pick a fixed uniformly-random $f : R^k \to R$ (where $R^k$ is the input domain of the categorical head) at the start of training and use $y = f(\text{input})$. Equivalently, when the input has additive-group structure of its own (such as the modular-addition testbed with input $(a, b) \in R \times R$), apply fixed random bijections $\pi_a, \pi_b$ to the inputs *in addition* to the label permutation $\sigma$, so the network sees $(a', b', y') = (\pi_a(a), \pi_b(b), \sigma((a+b) \bmod n))$; this blocks internal ring composition. D2-strong is the binding ring-ablation control: a positive D1-vs-D2-strong gap is the strongest version of the diagnostic's claim, and a null gap is the strongest empirical refutation of ring-structure dependence on the model under test.

D2-weak is retained as a *sensitive but bypassable* control whose collapse on a given task is itself a finding (it identifies that the model has learned internal ring structure upstream of the head). D2-strong is the *binding* control. Both should be reported.

### 6.4 Status

Definitions 6.1 and 6.2 are deterministic functions of the model, the batch, and the prediction point; their definitions are settled by Theorem 3.5. Their empirical interpretation — that large $\rho_\times$ together with high $\delta$ indicates a consequential deficit — is the empirical hypothesis of Section 8. The diagnostic produces a quantity to be measured, not a quantity that has been certified to predict downstream loss.

### 6.5 A first-run empirical demo

To illustrate the diagnostic on a concrete categorical head, we report a first-run empirical evaluation on the canonical modular-addition testbed of [Power et al. 2022; Nanda et al. 2023]. The demo uses a multi-layer architecture (a two-embedding-table MLP with a single hidden layer of GELUs and an $n$-way softmax head) trained with AdamW on $(a + b) \bmod n$ for $n \in \{6, 8, 12, 18, 30\}$, with the matched controls D2-weak (label permutation only) and D2-strong (target $= f(a, b)$ for a fixed uniformly random function $f$) of Section 6.3. The diagnostic $\rho_\times$ of Definition 6.1 is computed on the held-out test batch every 50 training steps; the time-mean $\overline{\rho_\times}$ across training is the robust summary because the per-step $\rho_\times$ oscillates as the model passes through Fourier-feature consolidation states (especially pre-grokking). The artifact is archived at [`empirical/paper34_conductor_blindspot_demo.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/paper34_conductor_blindspot_demo.py) with run outputs in [`empirical/reports/`](https://github.com/leomurillo/AI-ConductorBlindSpot/tree/main/empirical/reports).

The demo uses a multi-layer architecture; the diagnostic $\rho_\times$ is well-defined on the output tangent space (Section 6) regardless of depth, and the parameter-side claim (Corollary 3.9) is not directly tested by this demo. The Adam preconditioner's effective $M^{-1}$ on the held-out batch is ill-conditioned in directions the model has memorized (Remark 6.3), so the preconditioner discard ratio $\delta$ is reported but is not the primary signal; the headline measurement is $\overline{\rho_\times}$.

**Headline finding (n = 12; pre-grokking regime).** On $n = 12$ the diagnostic separates D1 from both controls in the predicted direction:

| arm | $\overline{\rho_\times}$ | test_acc | $\overline{\rho_\times} - \overline{\rho_\times^{D1}}$ |
|---|---:|---:|---:|
| D1 (ring) | $0.847$ | $0.000$ | — |
| D2-weak (label permutation) | $0.995$ | $0.000$ | $+0.149$ |
| D2-strong (random function of $(a,b)$) | $0.983$ | $0.069$ | $+0.136$ |

The separation is present before any arm generalizes — test_acc remains at zero on D1 and D2-weak throughout the training budget, and at approximately chance ($1/n \approx 0.083$) on D2-strong, confirming D2-strong's structureless target blocks generalization. The mechanism is the predicted one: the ring task drives the model's update direction to concentrate cubic mass within a small subset of conductor packets, reducing the cross-packet share of the selection-rule mass; both controls — having no ring structure to align with — keep their cross-packet share near the structural default. The diagnostic distinguishes ring-respecting from ring-ablated tasks during the *representation-learning phase that precedes grokking* — a regime in which test accuracy is uninformative. That D2-strong gives a near-identical signal to D2-weak on $n = 12$ confirms that the diagnostic is reading genuine ring structure on D1, not merely an artifact of which permutation map is in place.

**Scope-boundary observation (n = 6).** At $n = 6$ there are only three conductor packets, two of which ($\mathcal{P}_2 = \{3\}$ and $\mathcal{P}_3 = \{2, 4\}$) are small enough that the cross-packet share of the selection-rule mass is structurally near the ceiling and the diagnostic has insufficient resolving power. The observed gap on a single seed runs in the wrong direction by approximately the magnitude of seed-to-seed noise. Reportable as a scope boundary of the diagnostic on the smallest ladder ring.

**Structural-degeneracy observation (n = 8).** At $n = 8$ the same-packet count is *zero*: every surviving selection-rule triple is cross-packet (the only $(k, k, k)$ candidate would need $3k \equiv 0 \pmod 8$, which has no solution since $3 \nmid 8$, and no other within-packet shapes survive — see Appendix C). By Definition 6.1, this forces $\sum_{(k,\ell,m) \in \mathcal{T}_\times} = \sum_{(k,\ell,m) \in \mathcal{T}}$ on every update direction $u$, giving $\rho_\times \equiv 1$ identically. The demo confirms this: all three arms on $n = 8$ report $\rho_\times = 1.000$ at every training step. The diagnostic is *structurally constant* on rings with no same-packet surviving triples — a property of the divisor lattice, not of any model or task. The condition for $\rho_\times$ to have nontrivial range is, in essence, $3 \mid n$ (so that $(k, k, k)$ triples with $3k \equiv 0 \pmod n$ exist as a within-packet floor), with additional contributions for rings with large packets of order coprime to small prime divisors (such as $n = 18$). Rings violating this — including prime $n$ (single packet, $\rho_\times \equiv 0$) and $3 \nmid n$ composites with no other same-packet shape ($\rho_\times \equiv 1$) — are out of scope for this version of the diagnostic.

**Insufficient-budget observation (n = 18).** At $n = 18$ the same-packet count is $20$ out of $272$ surviving triples (structural ceiling $\rho_\times = 252/272 \approx 0.926$ at the uniform $u$), so the diagnostic has nontrivial range. At our training budget (20,000 steps) and architecture, however, none of the three arms generalize on $n = 18$ (test_acc near chance $1/18 \approx 0.056$ on all arms). The three-arm time-mean values are clustered: $\overline{\rho_\times^{D1}} = 0.959$, $\overline{\rho_\times^{D2\text{-weak}}} = 0.960$, $\overline{\rho_\times^{D2\text{-strong}}} = 0.906$. D2-strong lies *below* D1 (gap $-0.05$), opposite to the predicted direction. The most defensible reading is that none of the arms have reached the consolidation phase where ring structure becomes load-bearing: with no internal ring representation forming, the diagnostic reads a near-uniform $u$ on D1 and D2-weak, and the D2-strong target — a random function — happens to bias $u$ toward modes that look more same-packet at this seed. Reportable as a *budget-insufficient* observation: the diagnostic's discriminative phase is pre-grokking representation learning, but pre-representation-learning itself (the early memorization regime before ring features form) is not where the diagnostic carries signal. A longer training budget or larger architecture would test whether $n = 18$ falls in the diagnostic's useful range like $n = 12$ or in the post-grokking saturation regime like $n = 30$.

**A paper-actionable finding (n = 30; the bypass and what D2-strong corrects).** At $n = 30$ the architecture is capable enough to *grok within the training budget on the D2-weak permuted control* (test_acc $= 1.00$ on D2-weak, $0.98$ on D1, $0.05$ on D2-strong). The network learns an internal $(a + b) \bmod n$ representation upstream of the head and composes with $\sigma$ at the output — internalizing ring structure despite the label-side ablation. In consequence:

| arm | $\overline{\rho_\times}$ | test_acc | $\overline{\rho_\times} - \overline{\rho_\times^{D1}}$ |
|---|---:|---:|---:|
| D1 (ring) | $0.965$ | $0.982$ | — |
| D2-weak (label permutation) | $0.932$ | $1.000$ | $-0.033$ |
| D2-strong (random function of $(a,b)$) | $0.957$ | $0.047$ | $-0.008$ |

D2-weak shows a *negative* time-mean gap: the permuted task transiently concentrates cubic mass *more* than the ring task, exactly the bypass-induced internal-ring-composition signature. D2-strong, which blocks the bypass (test_acc $\approx 1/30 \approx 0.033$, at chance), restores the gap to near-zero — but loses the discriminative D1-vs-control signal on this ring, because D1 itself also reaches the saturated post-grokking $\rho_\times \approx 0.965$.

The two-part lesson: first, D2-weak is bypassable on a sufficiently capable architecture, and its $\rho_\times$ inversion is itself informative — the diagnostic is registering an *internalized* ring structure on the permuted task. Second, D2-strong is the correct control for ring-structure dependence, but the diagnostic's discriminative power is concentrated in the *pre-grokking* representation-learning phase, not the post-grokking saturated state. On $n = 30$, D1 groks within the training budget and the gap against either control collapses; on $n = 12$, D1 does not grok within budget and the pre-grokking gap is clearly visible against both controls. This was identified by the demo and is what makes T2-strong, not T2-weak, the primary control in the pre-specified experiment of Section 8. It also identifies a regime — pre-grokking representation learning — where the diagnostic is most informative.

**Placement against documented grokking phase transitions.** The pre-grokking regime in which $\rho_\times$ carries discriminative signal sits at a specific point in the documented phase-transition picture. [Power et al. 2022; Nanda et al. 2023] identify grokking as a transition from a memorisation phase to a circuit-formation phase that precedes generalisation; [Liu, Michaud & Tegmark 2023] (Omnigrok) frame the transition as controlled by the weight norm, with a phase boundary beyond which generalisation is reachable. [Truong et al. 2026a, b] give sharper dynamical observables of the same transition — normalised spectral entropy $\widetilde{H}(t)$ of the representation covariance, whose collapse below a task-specific threshold $\widetilde{H}^*$ precedes generalisation, and a first-passage law for grokking delay in $(V_t, \alpha_t)$-space with $V_t = \|\theta_t\|^2$. The $\rho_\times$ signal observed on $n = 12$ sits inside this window: it is largest in the representation-learning phase that follows memorisation and precedes generalisation — the same window in which the Truong observables move sharply — and collapses to the post-grokking saturated ceiling on $n = 30$ once the circuit has formed and the cross-packet share of the selection-rule mass has consolidated below the structural default. Under the §7.8 reading, the $n = 30$ collapse is the right signature, not a failure: ring-coherent descent has concentrated cubic mass within a single conductor packet, exactly the within-mode consolidation EGD/PGD induce by uniform within-mode rescaling. The natural sharpening, listed in §8.6's open follow-ups, is the trajectory-paired measurement: co-instrumenting $\rho_\times(t)$ with $\widetilde{H}(t)$ and $V_t$ on the same $n = 12$ training runs, to test whether $\rho_\times$'s descent below the structural default precedes, coincides with, or lags the spectral-entropy collapse and the weight-norm first-passage. The pre-grokking discriminative phase is, on this reading, the conductor-decomposition counterpart of the circuit-formation phase the broader grokking literature has documented.

The demo's findings are reported within the tier of the diagnostic itself: well-defined measurements with empirical interpretation pending the experiment of Section 8. Nothing in §6.5 supports Conjecture 5.8; the demo's purpose is to confirm that the diagnostic is computable on a real network, behaves as the certificate would predict on the ring-respecting task, and exposes a control-design refinement (D2-strong over D2-weak) that the pre-specified experiment must adopt.

### 6.6 A first-run application to pretrained-LLM checkpoints

The diagnostic of §6 is designed for retraining-free deployment on existing checkpoints. We report a first-run application to the Pythia suite [Biderman et al. 2023], chosen because the suite publishes ~140 intermediate checkpoints per model size across $N \in [70\text{M}, 12\text{B}]$ on identical data — giving an $N$-axis (parameter count at the final checkpoint) and a $D$-axis (training tokens at fixed $N$) from a single training run. The candidate ring-structured head is the conditional $\mathbb{Z}/12\mathbb{Z}$-categorical defined by restricting the final-layer softmax to the twelve single-token English month abbreviations under templated cyclic-shift prompts. The artifact is archived at [`empirical/pythia_rho_x_sweep/`](https://github.com/leomurillo/AI-ConductorBlindSpot/tree/main/empirical/pythia_rho_x_sweep).

**Setup.** Single-token verification on the GPT-NeoX tokenizer: every short-form month (Jan, …, Dec) is a single token in both bare and leading-space contexts (full month names also single-token; we use short forms throughout). Prompts: five context lengths $k \in \{3, 5, 7, 9, 11\}$ times twelve cyclic shifts, $60$ prompts per checkpoint. The continuation distribution is restricted to the twelve month-token IDs in the style (bare vs. leading-space) carrying the larger mass on the average row, then renormalized to a 12-way categorical. The diagnostic computes $\rho_\times$ both in the Definition 6.1 batch-mean form and in a per-example variant — the mean over prompts of the single-prompt $\rho_\times$ — with $50$ random control permutations of the ring index assignment per measurement. All forward passes are fp16 on a single consumer GPU (RTX 3050 Laptop, 4 GB).

**Result on the $N$-axis (Pythia-{70M, 160M, 410M, 1B, 1.4B} at the final checkpoint).** Per-example $\rho_\times$ on the ring task is in $[0.960, 0.981]$ across the five sizes; permuted control $\rho_\times$ is in $[0.974, 0.980]$ with per-permutation standard deviation $0.001$–$0.005$. Per-example separation $\overline{\rho_\times^{\text{perm}}} - \overline{\rho_\times^{\text{ring}}}$ ranges from $+0.000$ to $+0.015$ without monotone $N$-dependence. The Definition 6.1 batch-mean form fluctuates between $-0.046$ and $+0.079$ across the same five points, also without monotone $N$-dependence; we attribute the fluctuation to phase cancellation of the per-prompt centered scores when prompts are cyclically balanced and the model is alternately confident-and-correct (where $u^{(i)} \to 0$) or confident-and-wrong-shift-equivariantly (where the per-prompt phase washes out in the batch mean). The per-example variant is the more robust statistic on this substrate; we report both forms throughout.

**Result on the $D$-axis (Pythia-1B at step$\in$ {128, 1000, 3000, 10000, 30000, 70000, 143000}, spanning ${\sim}2.7 \times 10^{8}$ to ${\sim}3.0 \times 10^{11}$ training tokens).** Per-example $\rho_\times$ on the ring task is in $[0.973, 0.983]$ across the seven points; permuted control in $[0.978, 0.980]$. The per-example separation oscillates in $[-0.004, +0.005]$ at the per-permutation noise floor. The top-1 accuracy of the conditional 12-way head moves non-monotonically from $0.083$ (step 128) through a ${\sim}0.03$ trough (steps 3000–10000) to $0.533$ at the final checkpoint, with no clean ordering by training step.

**Reading: a third diagnostic-boundary observation.** Both axes lie at or just below the structural ceiling $\rho_\times = 108/110 \approx 0.982$ for $n = 12$. Compared to the §6.5 synthetic-MLP measurement at $n = 12$ pre-grokking — $\overline{\rho_\times^{D1}} \approx 0.847$, ring consolidation ${\sim}0.14$ below the ceiling — the Pythia calendar-months head shows no ring consolidation at any tested $(N, D)$ point. The directionality insight of §8.6 reads this directly: ring-coherent learning *reduces* the gradient's cross-packet share as the model consolidates within a conductor packet, driving $\rho_\times$ below the structural default. Pythia's $\rho_\times$ sitting at the default on calendar months means the calendar-cyclic structure is not in Pythia's representation-learning regime at any of the tested scale points; the diagnostic is not failing, it is correctly reporting that this head is *off the ring-structure trajectory*.

This is a third diagnostic-boundary observation, complementing the *structural-degeneracy* boundary (n = 8, §6.5) and the *insufficient-budget* boundary (n = 18, §6.5). The *off-trajectory* boundary is the regime in which a pretrained model has been trained on a corpus that does not exercise the ring structure of a candidate categorical head sufficiently for representation learning to consolidate. The diagnostic's null on this regime is informative in a specific narrow sense: it identifies that a deployed retraining-free measurement on a generic LM head is not, by default, on the certificate's relevant axis. Calendar tokens in a generic LM head are vocabulary items the model has learned to predict as natural-language sequence completions; they are not a *modular-arithmetic head* in the sense the certificate addresses, and a Pythia-trajectory sweep is therefore not a test of Conjecture 5.8.

**Implication for the diagnostic's deployment.** The §6.5 demo and the §6.6 sweep together delineate where the diagnostic is informative: on heads in the pre-grokking representation-learning phase for the candidate ring structure (synthetic modular arithmetic mid-training; a clean cyclic fine-tune of a pretrained model whose head is restricted to a ring-indexed vocabulary; the consolidation phase of a calendar/clock-conditional head in a model whose pretraining corpus heavily exercises that cyclic structure). A retraining-free deployment on an arbitrary pretrained checkpoint should be paired with an *a priori* trajectory check — for instance, that the conditional head's top-1 accuracy or the diagnostic's deviation from the structural ceiling moves nontrivially with training step on the candidate ring. A null reading at the structural ceiling, in the absence of such a check, is uninformative about the certificate's representational claim and should be reported as such.

**Scope of this report.** §6.6 reports an *application* of the diagnostic to a pretrained substrate, not a test of the certificate. The N-axis is bounded above by single-GPU memory at fp16 (Pythia-1.4B); quantized inference on Pythia-2.8B / 6.9B and a larger-VRAM extension to Pythia-12B are listed in §9 (Diagnostic deployment scope) as the natural extensions if the trajectory-check question is taken up on the same substrate.

**Reading by §2.3 tier.** The §6.6 measurement reads $\rho_\times$ directly on the head logits and so probes the diagnostic at tier T3 of §2.3 — an estimator-side application on the natural-language LM head, where the cross-packet null structure of $g_{p_*}$ would have to survive the Jacobian pullback and approximation chain to produce a signal, and the head is *not* in the pre-grokking representation-learning phase for the candidate ring. The cleaner tier-T4 reading — running $\rho_\times^{\mathrm{attr}}$ of Definition 6.4 on per-example attribution vectors from a frontier checkpoint on a cyclic mixed-task dataset (§6.7) — is the substrate on which the certificate's representational claim is exact at the head-side input by Proposition 3.11, and the §6.7 prediction on the released [Rai, Geva & Yao 2026] artifacts is the next direct experimental cell.

### 6.7 Per-example circuit-attribution vectors as a third diagnostic substrate

The diagnostic of §6.1 is defined on the simplex tangent space and computed from the centered logit gradient $u$ of a head batch. Proposition 3.11 lifts the same packet decomposition to the *head-side input* of per-edge attribution scores, which opens a third substrate on which $\rho_\times$ is measurable without retraining: the *per-example edge-attribution vectors* $s(x) \in \mathbb{R}^{|E|}$ produced by EAP, EAP-IG, or E-ACT (or, at first order around $p_*$, by ACDC) as a routine output of hypothesis-driven circuit discovery on a deployed model. Where §6.5 measures $\rho_\times$ at the output tangent and §6.6 at the pretrained-LLM head logits, §6.7 measures it after one Jacobian pullback to recover the head-side first-order input the attribution contracts against — the substrate where the certificate's representational claim is exact.

**Definition 6.4 (attribution-side cross-packet cubic mass).** *Let $\{s(x_i)\}_{i=1}^{|\mathcal{B}|}$ be a batch of per-example edge-attribution vectors produced by an EAP-family method on a categorical head with ring index $R = \mathbb{Z}/n\mathbb{Z}$. Project each $s(x_i)$ onto the head tangent space through the network's adjoint Jacobian to obtain $\tilde u^{(i)} := (J^\top)^{-1} s(x_i)|_{\mathrm{head}}$; in EAP / EAP-IG this projection is supplied as the per-example head-side score by construction and no inversion is required. Compute $\hat{\tilde u}^{(i)}_k$ as the additive-character transform of $\tilde u^{(i)}$, and define the attribution-side cross-packet cubic mass as $\rho_\times^{\mathrm{attr}}(s) := \rho_\times(\frac{1}{|\mathcal{B}|}\sum_i \tilde u^{(i)};\ \bar p)$, with $\rho_\times$ as in Definition 6.1.*

By Proposition 3.11 the head-side input the attribution contracts against is at $p_*$ already packet-diagonal; $\rho_\times^{\mathrm{attr}}$ measures the off-$p_*$ deviation of this diagonality after pulling each $s(x_i)$ back to the head tangent, where Theorem 3.5's cross-packet selection rule is exact and Corollary 3.12's blindness of linear attribution to the third-order cubic content is the question the diagnostic puts to data.

**Pre-specified prediction on a cyclic mixed-task dataset.** Apply EAP-IG on a frontier checkpoint (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, or GPT-2) to a mixed dataset consisting of a $C_n$-cyclic task (months, $n = 12$; weekdays, $n = 7$; digits, $n = 10$) at mixture proportions $\{0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0\}$ against a structurally distinct second task (entity binding, arithmetic addition, sequence completion). Both branches reportable:

* *Branch A (consistent with the certificate).* Residualizing the per-example attribution vectors $s(x)$ against the cross-packet cubic subspace defined by the selection rule $k + \ell + m \equiv 0 \pmod n$ on the cyclic sub-vocabulary changes the silhouette-selected cluster count $K^*$ from $1$ on the mixed dataset to $\geq 2$. The pre-residualization $K^* = 1$ is the cubic-blind aggregation Corollary 3.12 predicts (a linear functional of the head-side first-order gradient cannot resolve the third-order cross-packet distinction between the two mechanisms); the post-residualization jump is that third-order cross-packet content the certificate proves is absent from the head-side input the attribution contracts against, recovered by an explicit cubic projection on the pulled-back head-side score. Per-cyclic-task, the divisor count $\tau(n)$ is the structural ceiling on $K^*$: $\tau(12) = 6$ for months, $\tau(7) = 2$ for weekdays, $\tau(10) = 4$ for digits.

* *Branch B (refutes the corollary's operational reach).* No $K^*$ jump under the cross-packet residualization, or $K^*$ jump that does not respect the divisor-count ceiling. The corollary as a representational statement at $p_*$ stands; the operational reach to mixed-dataset cluster recovery on a frontier checkpoint is reported as not supported on this evidence.

The prediction is testable on already-published artifacts: the Rai, Geva & Yao 2026 release at https://github.com/Ziyu-Yao-NLP-Lab/data-driven-circuit-discovery exposes the per-example EAP-IG attribution vectors for GPT-2 / Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct on all four of their tasks (IOI, entity binding, arithmetic addition, sequence completion); the cyclic-task adaptation is one substitution of the head sub-vocabulary. Total expected cost: tens of GPU-hours on already-released attribution vectors plus a discrete additive-character transform on the cyclic sub-vocabulary; no model retraining.

**Status.** §6.7 specifies the substrate and pre-specifies the prediction. The measurement is not run in the present manuscript and is recorded as a definite next step.

---

## 7. Related work

The certificate sits at the intersection of three lines of work. We name what we connect and where the present construction adds an ingredient; we do not claim primacy on the constituent objects, all of which are classical or established.

### 7.1 Information geometry and natural gradient

The Fisher information form and the Amari–Chentsov cubic have been studied as the second and third cumulant tensors of the score for several decades [Amari & Nagaoka 2000; Chentsov 1982], together with the role of the cubic as a curvature correction expressed through the $\alpha$-connections, the e/m duality, and higher-order natural-gradient schemes [Amari 2016]. The objects $g_p$ and $T_p$ used here are exactly these. The ingredient the present construction adds to this confluence is the choice of basis: the additive characters of $\mathbb{Z}/n\mathbb{Z}$, indexed by conductor, in which both forms admit exact integer-arithmetic identities at the maximum-entropy point. In that basis the selection rule $k + \ell + m \equiv 0 \pmod n$ names which directions the quadratic part of the geometry can and cannot couple, as a finite predicate.

### 7.2 The empirical-Fisher critique

A body of work documents that the *empirical Fisher* used as a curvature surrogate by Adam-type adaptive methods is a poor proxy for the *true Fisher*, and traces part of the empirical disappointment of adaptive methods relative to natural-gradient-style optimizers to this estimation error [Kunstner, Hennig & Balles 2019; Martens 2020]. That line of work concerns the *quality of the estimate* of $g$. Theorem 3.5 concerns the *representational capacity* of the exact $g_{p_*}$. The two questions live on the same object from different angles, and are compatible: even an oracle returning the exact true Fisher at $p_*$ carries identically zero cross-packet coupling (Corollary 3.3), so the certificate is consistent with — and complementary to — the empirical-Fisher critique, naming a class of misalignment that is not estimator-removable.

### 7.3 Modular arithmetic, Fourier features, and grokking

Network training on modular arithmetic exhibits *grokking*, a delayed and sudden generalisation after long apparent overfitting [Power et al. 2022]. Mechanistic interpretability work has shown that the learned representations on such tasks are organised in the additive-character (Fourier) basis of $\mathbb{Z}/n\mathbb{Z}$ [Nanda et al. 2023]. The certificate is consistent with that picture and lives in the same basis: the conductor decomposition of the tangent space is the same arithmetic structure the learned representations have been observed to align with. What the certificate contributes alongside the empirical observation is an *a priori* statement about the optimiser's curvature model class — that the quadratic class cannot couple the cross-packet directions at $p_*$ — and a corresponding mechanistic prediction: an acquisition-rate asymmetry between ring-structured and non-ring directions. We offer this as a prediction to be tested (Section 8), not as an explanation of grokking; whether it accounts for the observed delay is the empirical question of Conjecture 5.8.

The sub-literature on modular-arithmetic grokking has since expanded along several axes that the certificate sits cleanly inside. On the *theoretical* side, [Mohamadi et al. 2024] prove that in the kernel regime no permutation-equivariant model can achieve small population error on modular addition without seeing a constant fraction of the data, and that one-hidden-layer quadratic networks with bounded $\ell_\infty$ norm generalise from substantially fewer training points; their account frames grokking as a transition out of kernel-like behaviour. [Gu et al. 2024] sharpen the Fourier-feature picture: under $L_{2,k+1}$-margin maximisation, one-hidden-layer networks and one-layer Transformers learning $k$-input modular addition have hidden neurons aligned with specific Fourier spectra, with neuron-count threshold $m \geq 2^{2k-2}(p-1)$. The conductor decomposition of the present paper sharpens which Fourier modes are *packet-coupled* at the curvature level versus packet-coupled at the cubic level; the cited works characterise which modes are *learned*, on the trained-model side, in a basis the certificate identifies as canonical.

[Manir & Rupa 2026] report a systematic empirical study disentangling depth, architecture, activation, and regularisation effects on grokking, finding that the Transformer/MLP gap largely disappears under matched hyperparameters and that weight decay is the dominant control parameter. [Quirke & Barez 2024] reverse-engineer a one-layer Transformer on $n$-digit integer addition into parallel per-digit streams. [Yıldırım 2026] takes an architectural rather than optimiser-side approach: enforcing spherical residual topology eliminates the memorisation phase entirely on cyclic modular addition / multiplication, but the same constraint *fails* on $S_5$ permutation composition — the alignment between architectural prior and task symmetry is what bypasses the delay. The conductor decomposition of $\widehat{\mathbb{Z}/n\mathbb{Z}}$ is the cyclic case of the symmetry that the spherical-topology prior aligns with; the structural parallel is taken up in §7.8.

Beyond cyclic groups, [Stander, Yu, Fan & Biderman 2024] reverse-engineer one-hidden-layer networks that have grokked $S_5$ and $S_6$ multiplication, finding circuits that decompose group arithmetic via the permutation group's subgroups — the *non-abelian* analogue of the additive-character story. The certificate's non-cyclic extension in Appendix B treats *abelian* groups via the Pontryagin dual; the bridge to the genuinely non-abelian case studied by [Stander et al. 2024] remains open and is recorded as such in §9. [Mallinar et al. 2025] further demonstrate that grokking on modular arithmetic is *not* specific to neural networks or to gradient descent: Recursive Feature Machines using the Average Gradient Outer Product (AGOP) — a purely second-order gradient statistic — also exhibit a sharp transition to perfect test accuracy on modular addition, and the discovered features are block-circulant, implementing the Fourier multiplication algorithm. This is highly informative for the certificate's scope, and we take it up explicitly in §7.4. Finally, the scale frontier: [Kikuchi et al. 2026] introduce an auxiliary modulus $Kq$ during training to scale modular-addition learning to very large $q$ (e.g., $q = 974\,269$, $N = 64$) without the covariate shift induced by zero-padding methods, providing the natural large-$q$ test bed for whether the diagnostic of §6 retains discriminative range.

### 7.4 Spectral theories of hierarchical feature learning

A concurrent line of work develops a positive theory of multilayer feature learning in which the central object is a label-weighted second-order operator on the current representation: at each layer $\ell$, the moment operator $\widehat{C}^{(\ell)} = \tfrac{1}{n}\sum_\mu y_\mu z_{\ell-1}(x_\mu) z_{\ell-1}(x_\mu)^\top$ is diagonalized and its leading eigendirections selected as the next layer's features, yielding a layerwise, backpropagation-free spectral surrogate for early gradient-based feature acquisition [Dandi et al. 2026]. The operator $\widehat{C}^{(\ell)}$ is the empirical, representation-localized analogue of the Fisher form $g_p$ studied here: both are second cumulants of a label-weighted score, both admit a spectral filter as their canonical action, and both are explicitly identified in the cited work as the "first nontrivial term in the expansion" of the feature-learning dynamics.

The same paper's §3.3, under the heading *Beyond second order*, names the natural successor problem — higher-order tensor extensions in which the second-order statistic $y \varphi^2$ is replaced by higher-degree statistics — and flags it as a deferred future direction, citing tensor-PCA hardness as the principal obstacle. Theorems 5.2 and 5.6 of the present paper are precisely such a higher-order extension, specialized to a categorical head over $\mathbb{Z}/n\mathbb{Z}$: the all-orders score-moment tower $\{M^{(j+2)}_{p_*}\}_{j \geq 1}$ admits, in the additive-character basis, a closed Fourier-convolution form that bypasses the tensor-PCA difficulty by trivializing on the arithmetic structure of the ring. The A2 cubic-aware preconditioner of §8.3 is the constructive instance: a deterministic, closed-form correction adding the conductor-packet-resolved component of the Amari–Chentsov cubic to the second-order curvature model, in the regime where the cubic spectrum is decidable by a finite Kronecker delta on $k + \ell + m \pmod n$.

The two views are dual halves of a single picture. The label-weighted second-order operator is the leading feature-learning primitive [Dandi et al. 2026]; on ring-structured categorical heads it carries a structural null space exactly aligned with the cubic the cited work names as the next term (Theorem 3.5 and Corollary 3.3). The conductor decomposition is the basis in which both statements are exactly readable on the cyclic case.

**Remark 7.4.1 (non-neural grokking via AGOP — concordance check on the second-order class).** A complementary line establishes that grokking on modular arithmetic occurs in *non-neural* models built around a second-order gradient statistic: Recursive Feature Machines (RFM) iterating the Average Gradient Outer Product (AGOP) transition sharply from near-zero to perfect test accuracy on $(a + b) \bmod p$, with the discovered features *block-circulant* and theoretically argued to implement the Fourier Multiplication Algorithm; the same block-circulant features appear in neural networks trained on the same task [Mallinar et al. 2025, ICML 2025 spotlight]. AGOP is a second-order statistic of the gradient (an outer product), so the RFM + AGOP construction is an external instance of feature learning driven entirely by the second-order class identified in §7.4.

This is informative for the certificate's scope in two ways. First, the second-order class is *sufficient to discover the Fourier multiplication algorithm via dynamics*, even though Theorem 3.5 establishes that $g_{p_*}$ at the maximum-entropy point cannot represent the cross-packet cubic coupling. The conductor blind spot is therefore a statement about *what coupling the quadratic curvature model carries at a single point*, not about *what algorithm a second-order learning method can ultimately reach via trajectory*. Second, the off-centroid analytic content of Theorems 5.2 and 5.6 names how the missing cubic couplings appear in the off-diagonal entries of $g_{p(t)}$ along a real trajectory — a dynamical resource that AGOP-driven feature learning can in principle absorb as it iterates, even though the static second-order object at any given step does not carry it. The Mallinar et al. result is concordant with — and sharpens the scope of — the certificate: a second-order method that re-estimates curvature at the running point can navigate the cubic resource through the trajectory, while at any frozen point the deficit is exact.

### 7.5 Nonlinear sigma models on statistical manifolds

A complementary recent construction derives the classical equations of motion for nonlinear sigma models in which the *base* manifold is a statistical manifold equipped with its Fisher–Rao metric, with fields mapping to a flat Euclidean target [Amaral 2025]. For five prototypical distributions — Bernoulli, Gaussian, categorical $N{=}3$, shifted exponential, Gumbel — the cited work computes the Fisher–Rao metric, scalar curvature, action density, and Laplace–Beltrami equation of motion explicitly, and records the two-dimensional Weyl anomaly $\langle T^a{}_a \rangle = -(N_{\mathrm{target}}/24\pi)\, R[g]$ as a direct consequence of the base curvature. The action

$$
S[\phi] \;=\; \frac{1}{2\alpha}\!\int\! d^d\theta\, \sqrt{g}\, g^{ab}(\theta)\, \partial_a \phi^\mu\, \partial_b \phi^\nu\, \delta_{\mu\nu}
$$

and the resulting equation of motion $\Delta_{\mathcal{M}}\phi^\mu = 0$ depend on the Fisher metric $g$ alone, with no appearance of the Amari–Chentsov cubic or any higher score-moment tensor: the construction is purely second-order in the base geometry.

For the categorical case on $R = \mathbb{Z}/n\mathbb{Z}$, this construction is an externally constructed instance of the second-order model class to which Theorem 3.5 applies, expressed in the language of mathematical physics rather than optimization. By Lemma 3.1, $g_{p_*}$ is diagonal in the additive-character basis with all eigenvalues equal to $n^2$; by Corollary 3.10, the Laplace–Beltrami operator on the Fisher base at $p_*$ acts as $(1/n)\,\Delta_H$ and preserves every conductor packet $\mathcal{P}_d^{\mathbb{C}}$. The Amaral 2025 NLSM action and its equation of motion therefore carry, at the maximum-entropy point, the same structural null space across distinct conductor packets that the certificate identifies in the optimizer setting. The blind spot is a property of the second-order quadratic model class itself, not of any particular application domain.

The natural higher-order extension on the NLSM side has the same form as the cubic-aware correction the present paper motivates as a successor to second-order preconditioning (§8.3, A2 arm), namely a cubic kinetic term

$$
S_{\mathrm{cubic}}[\phi] \;=\; \frac{1}{3!\, \alpha'}\!\int\! d^d\theta\, \sqrt{g}\, T_{abc}(\theta)\, \partial^a \phi^\mu\, \partial^b \phi^\nu\, \partial^c \phi^\rho\, C_{\mu\nu\rho},
$$

with $T_{abc}$ the Amari–Chentsov cubic of the base. On the categorical $n$-base at $p_*$, the kinetic content of this term is supported, by Lemma 3.4, by the integer-arithmetic selection rule $k + \ell + m \equiv 0 \pmod n$, exactly as in Theorem 3.5 part 2. We do not develop the field-theoretic side here. We record the structural alignment because the Fisher-base NLSM is an external, peer-reviewable instance in which the second-order curvature object is the *only* dynamical input; the certificate's claim that a quadratic model class cannot couple cross-packet content at $p_*$ is independent of whether that quadratic model class is interpreted as a preconditioner, a kinetic action, or a covariant Laplacian.

### 7.6 Sign-loss in squared-dot-product attention scoring

A recent proposal replaces softmax attention with the spread-based score
$$
s(q, k) \;=\; 1 \;-\; \frac{(q \cdot k)^2}{Q(q)\, Q(k)},
$$
closed over $\mathbb{Q}$ for algebraic-geometric tractability [Thomson 2026]. The squaring discards $\mathrm{sgn}(q \cdot k)$, a choice the cited work motivates separately and whose information-geometric content is the subject of this remark.

On a categorical readout indexed by $R = \mathbb{Z}/n\mathbb{Z}$, the real-valued tangent vectors decompose in the character basis with the symmetry $\chi_{-k} = \overline{\chi_k}$; nontrivial characters come in conjugate pairs $(\chi_k, \chi_{-k})$. Any zero-sum triple $(k, \ell, m)$ with $k + \ell + m \equiv 0 \pmod n$ admits a sign-partner $(-k, -\ell, -m)$ that is also zero-sum (Lemma 3.4) and has the same packet pattern. Consequently, the *unsigned* cubic mass $|T_{p_*}(\chi_k, \chi_\ell, \chi_m)|^2$ is invariant under simultaneous character inversion, while the *signed* orientation of the Amari–Chentsov tensor against a chosen positive-frequency convention is exactly what squaring destroys. The sign information lost by spread scoring is therefore not incidental: it is the cross-packet orientation of the cubic that Theorem 3.5 already identifies as missing from any purely quadratic curvature model.

The recovery channel is structurally cheap. A $\mathbb{Z}_2$ polarity bit attached to the attention head as a typed companion channel — what Thomson 2026 §6.5 names *Janus* — exposes precisely the sign data that squaring destroys. Under that augmentation, spread scoring carries cosine magnitude through $(q \cdot k)^2$ and cross-packet orientation through the polarity flag; the unsigned-Fisher curvature class cannot use either, but an attention head exposing the polarity flag can in principle use the latter.

**Pre-specified correlation prediction (three-outcome).** On any cyclic task $R = \mathbb{Z}/n\mathbb{Z}$ for which the cross-packet cubic mass $\rho_\times$ of Definition 6.1 has nontrivial range (excluding the degenerate boundaries identified in §6.5 — prime $n$ and the $n = 8$-style structural ceiling), the per-task performance gain from adding a $\mathbb{Z}_2$ polarity bit to a spread-based attention head is predicted to correlate with $\rho_\times(R)$ across rings. Three outcome branches, all reportable:

* *positive slope:* polarity-bit gain proportional to the recovered cross-packet orientation; rings with higher cubic mass benefit more. The dictionary {sign-loss} $\leftrightarrow$ {cross-packet cubic mass} is operationally correct, and the polarity bit is the architectural channel that delivers the certificate's missing content.
* *uncorrelated:* sign information is not the dominant content the polarity bit recovers; the gain (or absence of gain) is structural-architecture noise unrelated to the certificate's representational claim.
* *mixed:* the correlation holds on one ring sub-family and inverts on another, indicating that the {sign-loss} $\leftrightarrow$ {cubic mass} dictionary is composed with a confound — for instance, the baseline degradation of squared scoring relative to softmax is itself task-dependent — which the experiment would need to identify.

The protocol is additive on the modular-arithmetic testbed of §6.5: for each ring $n$ in the diagnostic ladder, run paired training cells with and without the polarity bit on a spread-attention head, and regress the per-task gain on $\rho_\times(n)$ across rings. We list this as a falsifiable follow-on prediction rather than a result; the structural reading of the polarity bit as the architectural realization of the cross-packet orientation channel stands or falls with the predicted slope.

### 7.7 Hypothesis-driven circuit discovery as second-order attribution

A recent large-scale circuit-discovery study reports that hypothesis-driven mechanistic-interpretability methods (EAP, EAP-IG, ACDC, E-ACT) do not discover *general task circuits* but rather *dataset-specific* ones [Rai, Geva & Yao 2026]. The central empirical findings of the cited work are sharp: across four standard tasks (indirect object identification, entity binding, arithmetic addition, sequence completion) and three model scales (GPT-2 124M, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct), in-distribution faithfulness at circuit size $0.05$ of $0.73$ on the 2-person IOI variant drops to $0.42$ cross-dataset on the 3-person variant; entity-binding circuits drift smoothly across position variants ($P_2$ at $111\%$ in-distribution faithfulness on Qwen2.5 falls to $33\%$ on $P_8$); most strikingly, when the discovery dataset is constructed as a mixture of two structurally distinct tasks at proportion $\alpha \in [0, 1]$, a single discovered circuit reaches $87\%$ faithfulness on both tasks simultaneously at $\alpha = 0.1$ (an $81.3$-point jump from the $5.8\%$ baseline at $\alpha = 0.0$). The pairwise Jaccard similarity between circuits discovered at adjacent mixture ratios is high ($0.79$–$0.89$) and falls smoothly to $0.26$ at the extreme mixture endpoints.

These findings are *concordant with* — not a measurement of — the certificate's lift to attribution. The present paper does not run the diagnostic $\rho_\times^{\mathrm{attr}}$ of Definition 6.4 on the released Rai/Geva/Yao 2026 attribution artifacts, and the structural reading we offer below is therefore motivating concordant evidence pending the §6.7 pre-specified test. With that scope in place: the EAP-IG attribution score is by construction an integrated-gradient contraction of the head's linearized response (§2.3); by Proposition 3.11 the head-side input it contracts against is packet-diagonal at $p_*$, and by Corollary 3.12 a linear functional of that first-order input cannot resolve a third-order distinction — so two examples whose mechanisms agree on the head-side first-order gradient and differ only in the cross-packet cubic structure of the local geometry would be indistinguishable to the score, regardless of the downstream network Jacobian. The $81.3$-point mixed-faithfulness jump at the $10\%$ injection point is the qualitative signature this lift would predict: cross-packet cubic content distinguishing the two mechanisms would be outside the head-side first-order signature the attribution can resolve, so an aggregation across the mixed batch returns a single circuit that is faithful to both. The smooth $0.79$–$0.89$ Jaccard ridge across adjacent mixture ratios is consistent with the leading-order Fourier activation rule of Theorem 5.2 — the cross-packet entries of the off-centroid Fisher form (the quadratic content of the head-side input) activate as a deterministic function of the displacement $h$ of the centered logit from $p_*$, and a small change in $\alpha$ would produce a small displacement of $h$, hence a smooth gradient in the attribution-discovered circuit's edge support. Whether these qualitative concordances are the certificate at work or coincidental signatures of a different mechanism is settled, not by this section, but by the §6.7 pre-specified prediction run on the released artifacts.

The cited work proposes Data-driven Circuit Discovery (DCD): cluster per-example attribution vectors $s(x)$ via PCA + K-means, then run circuit discovery within each cluster. On the all-task mixed dataset DCD recovers $K^* \approx 7$ clusters with silhouette/gap-statistic selection, aligned cleanly with the seven source task subsets. On the cyclic sub-tasks the certificate addresses (months $C_{12}$, weekdays $C_{7}$, digits $C_{10}$), the present manuscript predicts a *structural* cluster count: the divisor count $\tau(n)$ on the cyclic sub-vocabulary, with per-cluster feature support bounded by $\varphi(d)$ on packet $H_d$. This is the pre-specified prediction of §6.7. The DCD pipeline supplies the empirical scaffold; the certificate supplies the *a priori* divisor table.

The certificate offers a candidate structural reading of the cited work's "hypothesis-driven methods are dataset-specific" finding: a circuit discovered by an EAP-family method on dataset $\mathcal{D}_1$ is, at $p_*$, a linear functional of a packet-diagonal head-side input averaged over $\mathcal{D}_1$'s empirical distribution. A second dataset $\mathcal{D}_2$ with the same task semantics but different surface features would displace the centered logit by a different $h$, activating the off-centroid cross-packet entries of the head-side input differently (Theorem 5.2 + Theorem 5.6), so under this reading the discovered circuit follows the dataset rather than the task. We propose this as a candidate structural reading rather than a confirmed instance: the smooth dataset-to-circuit dependence and the mechanism-mixing failure are signatures the certificate would predict, and would also be consistent with non-conductor explanations — distinguishing between the two is what the §6.7 pre-specified prediction on a cyclic mixed-task dataset is for.

A complementary mechanistic-interpretability line treats *circuit stability* — the consistency of a model's discovered circuit across input variants — as a measurable proxy for generalisation, with formal definitions of circuit stability and circuit equivalence and case studies linking lack of stability to specific generalisation failures [Sun 2025]. Under the present reading, circuit *instability* of the type Sun identifies can manifest at the head-side first-order level (a small Fourier displacement $h$ activates cross-packet entries differently and shifts the discovered edge support) without any change in the underlying task semantics; the cross-packet residualisation of Definition 6.4 is the natural intervention to test whether observed instability is reducible to the head-side cubic gap the certificate names. The broader practical-review survey of [Rai, Zhou, Feng, Saparov & Yao 2024] organises the EAP-family methodology at the task-centric level and is the standard background scaffold for situating Proposition 3.11 / Corollary 3.12's lift within the field's current toolkit.

### 7.8 Optimization-side acceleration on grokking

Two recent optimizer-side proposals reduce the grokking delay on modular arithmetic by acting *uniformly within modes* of the gradient spectrum. Egalitarian gradient descent [Saheb Pasand & Dohmatob 2026] normalises the gradient so that dynamics along all principal directions evolve at the same speed; on classical modular addition and sparse parity, the method removes the grokking plateau outright, with the framing that asymmetric convergence speed along principal directions is what produces the delay. Preconditioned gradient descent toward the rich learning regime [Jiang et al. 2026] uses a Gauss–Newton-style preconditioner to mitigate spectral bias on the NTK-to-rich transition, and reports that PGD reduces grokking delay by enabling uniform exploration of the parameter space in the NTK regime. Both methods sit in tier T2 of the §2.3 scope: they act through the Jacobian pullback of an exact or approximate output-side curvature surrogate.

Both methods are within-mode preconditioners — they rescale the gradient's principal directions, but do not introduce coupling between distinct conductor packets at $p_*$. By Corollaries 3.3 and 3.8, this within-mode action is exactly the action that the cross-packet null structure of $g_{p_*}$ leaves available: a tier-T1 preconditioner acts as a scalar on $T_{p_*}\Delta_R^\circ$, and the parameter-side analogue at the uniform configuration is a scalar multiple of the identity on the centered logit subspace (Corollary 3.9). The certificate is silent on within-mode dynamics, and the empirical success of EGD and PGD on modular arithmetic is consistent with — not in tension with — the certificate's structural reading: optimization descent on these cyclic tasks has within-packet content to address, and addressing it within-mode-uniformly is what these methods do.

This reading converges with the §8.6 closing observation. Three pre-specified head-only cubic-aware corrections — Newton-style cross-packet, Neumann-style cross-packet, and sign-reversed within-packet amplification — all returned null or directionally-reversed interactions at $\alpha = 0.1$ on $n = 30$. The within-packet amplification form (C) is the closest of the three to the EGD / PGD framing in spirit, but acts on a single dominant packet via the cubic substrate rather than uniformly across modes; the null result identifies that this specific operationalisation does not translate the certificate's representational content into an operational benefit, while leaving open whether a properly EGD-style uniform within-mode rescaling at a different $\alpha$, or a parameter-side K-FAC variant, would. The relationship between the certificate and these published optimization-side accelerations is therefore *complementary*: the certificate measures what the quadratic class cannot represent, and the EGD / PGD line addresses what the quadratic class *can* represent — uniform within-mode scaling — and reduces the grokking delay through that channel. We list a concordance-style follow-up in §8.6's "What remains open about Conjecture 5.8" — an $\alpha$-sweep paired against an EGD-style uniform within-mode normalisation arm, with trajectory instrumentation in the §5.4 sense.

A separate complementary line proceeds *architecturally* rather than optimisationally [Yıldırım 2026]: enforcing a fully bounded spherical topology ($L_2$ normalisation throughout the residual stream) eliminates the memorisation phase entirely on modular addition and multiplication over $\mathbb{Z}_p$, while the same constraint *fails completely* on non-commutative $S_5$ permutation composition. The contrast — spherical topology bypasses the cyclic-grokking plateau, but the same prior breaks on non-abelian tasks — is the architectural counterpart to the conductor decomposition: the alignment of the architectural prior with the task symmetry is what bypasses the delay, and the conductor-packet structure of $\widehat{\mathbb{Z}/n\mathbb{Z}}$ is the symmetry that the cyclic case carries. We do not claim a direct technical bridge here; we note the structural parallel and list the empirical test as a candidate follow-on: a spherical-topology baseline paired with a $\rho_\times$ measurement (Definition 6.1) on $n \in \{6, 12, 18, 30\}$, predicting that the bypass coincides with $\rho_\times$ remaining at the pre-grokking structural default rather than consolidating below it.

### 7.9 Why a finite witness was available here

The local information geometry of a categorical model has been understood for decades: the Fisher form and the Amari–Chentsov cubic at any interior point of the simplex are explicit, and the cubic is not a function of the Fisher [Amari & Nagaoka 2000]. The question of whether second-order preconditioning is structurally blind to part of this geometry is, in retrospect, a question one could have asked at any point in the last three decades. What made it decidable here is the simultaneous choice of three things: the *base point* $p_*$, at which the Fisher form on $\mathbb{Z}/n\mathbb{Z}$ is a scalar multiple of the identity; the *basis* of additive characters, in which both $g_{p_*}$ and $T_{p_*}$ admit integer-arithmetic identities by orthogonality (Lemma 2.1); and the *cross-packet predicate* on the conductor decomposition of $\widehat{\mathbb{Z}/n\mathbb{Z}}$, which names the surviving inter-block coupling. Any one of the three in isolation is classical; together they reduce the question to a finite predicate decidable without floating-point comparison and without a single trained model.

The natural places to have looked instead were less fortunate. A trained network at an interior point inherits a geometry whose closed form depends on parametrization and trajectory; a generic categorical head viewed in the standard basis carries no orthogonality identity, so the cubic and the Fisher form do not separate by inspection; and parameter space, the natural object for an optimization claim, mixes the head with the body of the network and the parametrization gauge. We came here by following the conductor decomposition of $\widehat{R}$ — the same arithmetic structure the mechanistic-interpretability literature documents in the learned representations [Nanda et al. 2023] — back to the maximum-entropy point of the head, where it makes both forms exactly decidable. The retrospective inevitability of the witness is the inevitability of arithmetic on a ring, evaluated at the one point on the simplex at which the geometry is symmetric enough to be read off.

### 7.10 Loss-landscape symmetries

The certificate of Section 3 is, structurally, a statement about how a symmetry of the loss landscape factorises the local curvature operator. The cyclic group $C_n = \mathbb{Z}/n\mathbb{Z}$ acts on the categorical head's output by index translation, and the maximum-entropy point $p_*$ is the unique configuration on the open simplex fixed by this action. The Fisher form $g_{p_*}$ is consequently $C_n$-invariant, and its spectral decomposition along the irreducible representations of $C_n$ — the additive characters $\chi_k$ — produces the eigenbasis of Lemma 3.1; the conductor decomposition is the refinement of this irrep decomposition by orbit-of-character-order, grouping characters by which divisor $d \mid n$ generates them. Read this way, the cross-packet null structure of Theorem 3.5 is the curvature-class shadow of a representation-theoretic decomposition of the head's output tangent space, in the same sense in which an equivariant feature decomposition factorises a $G$-invariant function on its representation summands.

This places the certificate inside the broader family of loss-landscape-symmetry results, though the relevant symmetry differs from the one studied in the active mode-connectivity / permutation-alignment line. Permutation-symmetry work [Entezari et al. 2022; Ainsworth et al. 2023] is about the hidden-neuron permutation group $S_h$ acting on *parameter* space, with the empirical finding that two networks trained from different seeds reach modes that become linearly connected after a permutation alignment; the basin geometry there is described "modulo $S_h$". The certificate's symmetry is different in three respects: $C_n$ acts on the *output label space* rather than on hidden parameters, the relevant decomposition is by irreducible characters rather than by permutation alignments, and the resulting structural statement is about the *block structure of curvature within a single basin* rather than about connectivity between basins. The two lines ask non-overlapping questions about the same loss surface — one about the surface's symmetry-modular shape *across* basins (gauge-fixing the $S_h$ ambiguity), the other about how an output-space symmetry restricts the curvature class *within* a single basin (block-diagonalising $g_{p_*}$ by $C_n$-irreps). They are complementary rather than overlapping; the structural family of "loss-landscape symmetries" subsumes both.

The closest existing bridges within this paper's reference set are [Yıldırım 2026] (architectural enforcement of a symmetry prior — spherical residual topology — that bypasses the cyclic-grokking plateau while failing on the non-abelian $S_5$ task, structurally parallel to the conductor decomposition as noted in §7.8) and [Stander et al. 2024] (non-abelian generalisation: $S_5/S_6$ multiplication grokking reverse-engineered as circuits factorising through subgroups, the non-abelian-irrep counterpart of the cyclic-character story). The recurring observation across these works and the present certificate is that *the symmetry of the task selects the basis in which the learned representation, the discovered circuit, and the local curvature class simultaneously diagonalise.* Whether the conductor decomposition, the Yıldırım architectural prior, the Stander coset structure, and the Entezari / Ainsworth permutation alignment all fall under a single subsuming theory of loss-landscape symmetry — distinct symmetry groups, common diagonalisation principle — is a research-programme question we record here but do not address.

---

## 8. A pre-specified experiment

We pre-specify a single experiment with a built-in negative control and pre-claim both outcome branches, so that a null or negative result is reported as a result and not reframed as a tuning argument. Subsection 8.6 reports a first-run instance of this experiment on a head-only cubic-aware variant.

### 8.1 The pre-specification commitment

By *pre-specification* we mean the following: the design below — arms, tasks, primary endpoint, success and failure branches — is fixed in writing before any data is collected; the analysis plan, including the sign of the predicted interaction, is committed to before any optimizer is run; both outcome branches are recorded as full results of the same value. The pre-specification is the discipline that makes Conjecture 5.8 falsifiable; without it, a null result is too easily reframed as a tuning problem.

We use *pre-specified* rather than *pre-registered* deliberately. In scientific publishing the latter term is reserved for protocols lodged with an external registry — OSF, AsPredicted, ClinicalTrials.gov, or an equivalent venue — with a timestamp predating the runs; that infrastructure provides the audit guarantee that the analysis plan was committed before the data. This paper does not lodge the §8.2–§8.4 protocol with such a registry, so the safeguard against post-hoc reinterpretation is internal: both branches of the §8.4 success/failure claim are stated before the §8.6 results, and §8.6 reports the realized branch under the label it fell into, not under the label convenient for the narrative. We name the discipline *pre-specification* to avoid the stronger claim *pre-registered* would carry.

### 8.2 Hypothesis

From Conjecture 5.8: a cubic-aware correction to a second-order preconditioner — even the simplest such correction, the addition to the preconditioner of the conductor-packet-resolved component of the Amari–Chentsov operator — should *accelerate the acquisition of ring structure if and only if ring structure is present*, and be *null when it is absent*. The asymmetry, not the main effect, is the hypothesis. On a delayed-generalization task the sharp form is: the cubic-aware correction reduces the generalization-delay gap on the ring-structured task and does not reduce it on the matched non-ring control.

### 8.3 Design

**Arms.** (A1) a matched second-order baseline (K-FAC, Shampoo, or a Fisher-preconditioned method tuned to best effort); (A2) identical to A1, plus a conductor-packet-resolved cubic correction added to the preconditioner update rule. The two arms share initialization, batch size, learning-rate schedule (jointly tuned over a fixed grid), and seed protocol.

**The A2 cubic-aware correction (explicit form).** Let $\bar p$ be the batch-averaged prediction and $u$ the batch-mean update direction on the output tangent space. The *conductor-packet-resolved component of the Amari–Chentsov operator* at $\bar p$ is the symmetric matrix $C(u)$ given, in the additive-character basis of $\mathbb{Z}/n\mathbb{Z}$, by the cross-packet projection of the cubic contracted with $u$:
$$
\widehat{C(u)}_{k\ell} \;:=\; T_{\bar p}(\chi_k,\, u,\, \overline{\chi_\ell}) \cdot \mathbf{1}\!\left[(\mathrm{cond}(k),\, \mathrm{cond}(\ell),\, \mathrm{cond}((\ell - k) \bmod n)) \text{ not all equal}\right],
$$
with $T_{\bar p}$ the cubic form (2.1); at $\bar p = p_*$ this reduces by Lemma 3.4 to $\widehat{C(u)}_{k\ell} = n^3\, \hat u_{(\ell - k) \bmod n}$ when the cross-packet predicate on $(k, \ell, (\ell - k) \bmod n)$ holds and zero otherwise. The A2 update is then
$$
\delta\theta_{A2} \;:=\; -\, (M + \lambda\, J^\top C(u)\, J)^{-1}\, \nabla_\theta \mathcal{L},
$$
where $M$ is the A1 preconditioner, $J$ the head's output-to-parameter Jacobian (the standard pullback used by K-FAC and natural gradient), and $\lambda$ a scalar tuned jointly with the A1 hyperparameter grid. Concretely: (i) compute $\hat u_a$, the discrete additive-character transform of $u$; (ii) assemble $\widehat{C(u)}$ in the character basis via the cross-packet predicate; (iii) transform back to the canonical basis to obtain $C(u)$; (iv) pull back through $J^\top \cdot J$ and add $\lambda$ times the result to $M$ before inversion. The construction adds curvature signal exactly on the cross-packet subspace the certificate proves $M$ alone cannot couple; on the within-packet block and at $\bar p = p_*$ along directions $h$ with packet-aligned spectrum, $C(u) \to 0$ and A2 reduces to A1. Refinements (a $\lambda$ schedule tied to the measured $\rho_\times(u; \bar p)$ of §6.1, or a damping term on the cross-packet block alone) are left as ablations; the form above is the simplest cubic-aware correction the certificate motivates. The construction is the constructive instance, on the cyclic case, of the higher-order extension to second-order spectral feature learning named as a deferred future direction in [Dandi et al. 2026, §3.3]: the additive-character basis of $\mathbb{Z}/n\mathbb{Z}$ trivializes the tensor-PCA step they identify as the principal obstacle, leaving a closed-form, conductor-packet-resolved correction.

**Tasks.** Three tasks corresponding to the three protocols of Section 6.3:

* **(T1)** modular addition or multiplication on $\mathbb{Z}/n\mathbb{Z}$ for a moderate composite $n$ in the range studied here, the canonical grokking testbed [Power et al. 2022; Nanda et al. 2023];
* **(T2-weak)** the same task under a fixed random permutation $\sigma : \mathbb{Z}/n\mathbb{Z} \to \mathbb{Z}/n\mathbb{Z}$ of the label set (D2-weak of Section 6.3, promoted to an experimental arm), destroying *label-side* ring structure while preserving dimension, entropy, sample statistics, and overall task difficulty;
* **(T2-strong)** the same task with fixed random bijections $\pi_a, \pi_b : \mathbb{Z}/n\mathbb{Z} \to \mathbb{Z}/n\mathbb{Z}$ applied to the *inputs* in addition to the label permutation, i.e., target $= \sigma\!\left((\pi_a^{-1}(a) + \pi_b^{-1}(b)) \bmod n\right)$ as a function of the network-observed $(a, b)$ — equivalent to a structureless random function $f : (\mathbb{Z}/n\mathbb{Z})^2 \to \mathbb{Z}/n\mathbb{Z}$ with the same sample statistics and task difficulty. T2-strong blocks internal ring composition and is the binding control (D2-strong of Section 6.3).

The dual control is essential because, as observed in the demo of Section 6.5, T2-weak can be bypassed on a sufficiently capable architecture: the network learns an internal $(a + b) \bmod n$ representation and composes with $\sigma$ at the head, recovering both the task and a ring-aligned internal geometry. T2-strong forbids this bypass.

**Primary endpoint.** The *primary interaction* between optimizer arm and task, computed against T2-strong:
$$
\text{interaction} \;:=\; \big(\text{delay-gap reduction on T1 under A2 vs A1}\big) - \big(\text{delay-gap reduction on T2-strong under A2 vs A1}\big).
$$
Conjecture 5.8 predicts a strictly positive primary interaction. **Secondary endpoint:** the same interaction against T2-weak; its informative value lies in the *difference* T2-weak $-$ T2-strong, which is a quantitative indicator of how much internal ring structure the architecture learns upstream of the head. The main effects on T1 alone are *not* the endpoint, because a main effect without the asymmetry is consistent with a generic preconditioning improvement and does not bear on the conductor mechanism.

### 8.4 Pre-claimed outcomes

**Branch A (confirms Conjecture 5.8).** Positive interaction: the cubic-aware correction specifically accelerates ring-structure acquisition and is null on the control. Reported as confirmation; the experiment in this form does not address persistence beyond the specific task family tested, which remains a follow-on question.

**Branch B (refutes or fails to confirm Conjecture 5.8).** Zero or negative interaction: the correction helps both tasks or neither, or helps the control as much as the ring-structured task. Reported as refutation on the tested instance. Theorem 3.5, Corollary 3.3, the diagnostic of Section 6, and the off-centroid expansion of Section 5 stand regardless — none of them depends on the trajectory or task-relevance claim. Conjecture 5.8 is reported as not supported by the experiment; the certificate is not.

### 8.5 Scope of ring structure

The practical reach of the experiment is bounded by where ring structure genuinely occurs. Honest candidate categorical heads:

* modular and cyclic synthetic tasks (cleanest; the canonical grokking testbed);
* calendar and clock token prediction (weekday, month, hour heads with genuine $\mathbb{Z}/7$, $\mathbb{Z}/12$, $\mathbb{Z}/24$ structure);
* cyclic or rotational classification labels (orientation prediction, periodic phase, angular discretization);
* positional encoding schemes with explicit modular structure.

We do *not* assert that general next-token prediction over a natural-language vocabulary has exploitable ring structure; whether any latent periodic subword structure renders $\rho_\times$ nontrivial on such heads is itself an instance of the Section 6 diagnostic, to be measured rather than assumed.

### 8.6 Empirical results on the head-only cubic-aware variants (three forms)

In accordance with the pre-specification commitment of §8.1, we report three instances of the experiment on a head-only variant in which the cubic-aware correction is applied directly to the logit gradient before backprop, on a multi-layer MLP trained with AdamW for the remaining parameters. The substrate is the same architecture used by the demo of §6.5, on $n = 30$ where the architecture groks within budget on the ring task. The three instances differ in the *form* of the correction; the first two inject cross-packet content (the side of the gap the quadratic class cannot represent), and the third (sign-reversed) amplifies the within-packet content the §6.5 diagnostic identifies as the consolidation direction.

**Tasks (all three instances).** (T1) $(a + b) \bmod 30$. (T2-strong) target $= f(a, b)$ for a fixed uniformly-random $f$ (per §6.3 D2-strong).

**Arms (all three instances).** (A1) Plain AdamW. (A2) Plain AdamW augmented with a correction $w$ on the head logit gradient at strength $\alpha = 0.1$ relative to $\|\bar u\|$. The three forms of $w$:

* **Form (A) — Newton-style cross-packet injection.** $\hat w_m = \sum_{(k, \ell, m) \in \mathcal{T}_\times} \hat u_k \hat u_\ell$ (quadratic in $\bar u$). The cubic contraction of the gradient with itself, restricted to cross-packet character outputs; the third-order-Taylor Newton correction.
* **Form (B) — Neumann-style cross-packet injection.** $\hat w_m = \sum_{(k, \ell, m) \in \mathcal{T}_\times} \hat h_k \hat u_\ell$ (linear in $h := \bar p - p_*$, linear in $\bar u$). The certificate's natural operationalization: the first-order Neumann expansion $g_{p_*+h}^{-1} \approx g_{p_*}^{-1} + g_{p_*}^{-1} T_{p_*}(h, \cdot, \cdot) g_{p_*}^{-1}$ contributes exactly this $T_{p_*}(h, \cdot, \cdot)$ to the preconditioned update.
* **Form (C) — within-packet amplification (sign-reversed).** $w$ is the projection of $\bar u$ onto its *dominant* conductor packet — the packet with the largest Fourier-mass share — IDFT'd back to real space. Adding $\alpha \cdot w$ amplifies the within-packet direction the model is descending in. Forms (A) and (B) inject cross-packet content; form (C) amplifies within-packet content. Form (C) is the operational test of the directionality reading suggested by §6.5: that ring-coherent learning progresses by consolidating cubic mass within a single packet.

**Results over 3 seeds at 15{,}000 training steps.** Late-window mean test accuracy and interaction endpoint:

*Form (A) — Newton-style cross-packet.*

| seed | A1×T1 | A2×T1 | $\Delta_{T1}$ | A1×T2-s | A2×T2-s | $\Delta_{T2\text{-s}}$ | interaction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | $0.955$ | $0.942$ | $-0.013$ | $0.047$ | $0.023$ | $-0.024$ | $+0.011$ |
| 1 | $1.000$ | $0.998$ | $-0.002$ | $0.043$ | $0.044$ | $+0.001$ | $-0.003$ |
| 2 | $0.981$ | $0.972$ | $-0.008$ | $0.049$ | $0.041$ | $-0.009$ | $+0.000$ |
| **mean** | | | $-0.008$ | | | $-0.011$ | **$+0.003$** |

*Form (B) — Neumann-style cross-packet.*

| seed | A1×T1 | A2×T1 | $\Delta_{T1}$ | A1×T2-s | A2×T2-s | $\Delta_{T2\text{-s}}$ | interaction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | $0.955$ | $0.911$ | $-0.044$ | $0.047$ | $0.039$ | $-0.008$ | $-0.036$ |
| 1 | $1.000$ | $0.990$ | $-0.010$ | $0.043$ | $0.041$ | $-0.002$ | $-0.008$ |
| 2 | $0.981$ | $0.990$ | $+0.009$ | $0.049$ | $0.047$ | $-0.003$ | $+0.012$ |
| **mean** | | | $-0.015$ | | | $-0.004$ | **$-0.011$** (std $0.020$) |

*Form (C) — within-packet amplification.*

| seed | A1×T1 | A2×T1 | $\Delta_{T1}$ | A1×T2-s | A2×T2-s | $\Delta_{T2\text{-s}}$ | interaction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | $0.955$ | $0.936$ | $-0.019$ | $0.047$ | $0.041$ | $-0.007$ | $-0.012$ |
| 1 | $1.000$ | $1.000$ | $+0.000$ | $0.043$ | $0.047$ | $+0.004$ | $-0.004$ |
| 2 | $0.981$ | $0.988$ | $+0.008$ | $0.049$ | $0.053$ | $+0.003$ | $+0.004$ |
| **mean** | | | $-0.004$ | | | $+0.000$ | **$-0.004$** (std $0.007$) |

**Closing reading: three Branch Bs, in converging shapes.** The mean interaction across all three forms is null or negative; none preferentially helps ring-task acquisition. At $\alpha = 0.1$ on $n = 30$ with the AdamW baseline, *no form of head-side cubic-aware correction we have tested — cross-packet add (A), cross-packet linear-in-$h$ add (B), or within-packet amplification (C) — preferentially accelerates ring-structure acquisition over the structureless random control*.

Form (C) is particularly informative: it is the operational test of the directionality reading. If §6.5's observation that ring-task learning concentrates cubic mass within a single packet had identified the optimization's *actual* descent bottleneck, then amplifying that direction should accelerate it. The empirical answer is that it does not — within noise, form (C) is as null as form (A) and as marginally-negative as form (B). The directionality reading was sharper but not sufficient: it correctly identified that *adding cross-packet content opposes the descent* (form (B)'s consistent $\overline{\Delta_{T1}} < \overline{\Delta_{T2\text{-s}}}$), but the converse — that *amplifying within-packet content accelerates it* — did not materialize on this architecture and operating point.

**Why three forms can all fail.** One plausible reading: Adam's per-coordinate adaptive scaling $1/\sqrt{v}$ already differentially weights the gradient's coordinates by their per-character second moment, which on the head logits absorbs much of what an explicit packet-aware correction would supply. The cubic-aware corrections — whether they add cross-packet content or amplify within-packet content — would then be operating in directions Adam has already addressed implicitly, with the explicit additive perturbation acting mostly as small noise. This is consistent with the §6.5 diagnostic's value as a *measurement* of the geometry without implying it would help to *intervene* on. Stronger versions of this reading (Adam absorbs the certificate; the certificate is purely measurement-grade) are not directly testable on this evidence, but the three-Branch-B pattern is consistent with it.

A second, structurally sharper reading is available in light of the optimisation-side acceleration line surveyed in §7.8. [Saheb Pasand & Dohmatob 2026] and [Jiang et al. 2026] both compress the grokking delay on modular addition by *uniformly rescaling within modes* of the gradient spectrum — egalitarian gradient descent normalises the principal-direction speeds; Gauss–Newton-style PGD mitigates spectral bias on the NTK-to-rich transition. Neither method introduces cross-packet coupling at $p_*$, and by Corollaries 3.3 and 3.8 the cross-packet block of $g_{p_*}$ is zero, so within-packet uniform rescaling is exactly what the cross-packet null structure leaves available. Form (A) and form (B) of §8.6 act in the opposite direction — they inject cross-packet content into the head-side update — and their null/negative interactions are *predicted* by combining the certificate with this published optimisation-side picture: descent on these cyclic tasks lives within a single conductor packet, and adding cross-packet content opposes the descent (form (B)'s consistently negative $\overline{\Delta_{T1}}$). Form (C) amplifies a single dominant packet via the cubic substrate, which is closer to but not the same as a uniform within-mode rescaling; its null result is consistent with both the EGD/PGD picture (the right intervention is uniform within-mode, not single-packet via cubic) and the Adam-absorbs reading. Under this combined reading, the cleanest remaining experimental cell is the §8.6 follow-up that pairs the $\alpha$-sweep with an EGD-style uniform within-mode normalisation arm.

A third reading places Branch B inside the broader documented behaviour of higher-order stochastic optimization. Cubic regularization is a foundational nonconvex method [Nesterov & Polyak 2006], but its *stochastic* variants pay a sample-complexity premium: estimating a third-cumulant contraction on a mini-batch has strictly higher variance than the corresponding second-cumulant estimate, and known convergence guarantees for stochastic cubic regularization scale as $O(\epsilon^{-3.5})$ in sample complexity versus the $O(\epsilon^{-2})$ of stochastic first-order methods [Tripuraneni et al. 2018]. Two cost surfaces must be distinguished here. The conductor-packet operator $C(u)$ of §8.3 does *not* inherit the tensor-decomposition difficulty named as the obstacle in [Dandi et al. 2026, §3.3]: Theorem 5.6 reduces the cubic spectrum to a closed-form Fourier convolution on $\mathbb{Z}/n\mathbb{Z}$, trivializing the tensor-PCA step on the operator side. The *single-batch contraction* of $C(u)$ against the running update direction $\bar u$, however, does inherit the standard third-cumulant estimator-variance penalty: the per-step cubic contraction at $\alpha = 0.1$ on a head-only forward batch is exactly the regime in which the broader literature predicts that sampling noise dominates the cubic signal. The Adam-absorbs reading is the positive counterpart on this axis: Adam's per-coordinate $1/\sqrt{v + \varepsilon}$ accumulates per-coordinate gradient variance across many mini-batches and uses the running scale as an implicit preconditioner — one of the few ways to exploit third-cumulant-scale information without paying the sample-complexity cost of estimating the cubic directly. Under this combined reading, the three Branch Bs are consistent with a general structural fact about higher-order stochastic optimization — a single-batch cubic contraction at fixed $\alpha$ is the regime in which the cubic signal is dominated by sampling noise, and Adam's running per-coordinate scale absorbs the same content through a variance-controlled accumulator instead — as much as with the EGD/PGD within-mode reading above. A *batch-size sweep* on forms (A) or (B), testing whether the cubic contraction's per-step variance is the rate-limiting penalty by raising the batch at fixed $\alpha$, is the natural empirical extension of this reading and a candidate addition to the §8.6 open-follow-ups list.

**What this does and does not imply.** Theorem 3.5, Corollary 3.3, the diagnostic of Section 6 (and its first-run signal of §6.5), the leading-order Fourier selection rule of Theorem 5.2, and the all-orders expansion of Theorem 5.6 stand regardless. The certificate's representational claim is intact; what Branch B across three forms refutes is the specific *operational* claim that any of three natural cubic-aware corrections at $\alpha = 0.1$ preferentially helps ring-task acquisition over a matched non-ring control on this architecture. The relation between the certificate and the EGD/PGD line of §7.8 is *complementary*: the certificate measures what the quadratic class cannot represent (cross-packet cubic coupling at $p_*$), and the cited optimisation-side methods address what it *can* represent (within-mode dynamics) — and reduce the grokking delay through that channel. Read this way, the three Branch Bs do not undermine the form of Conjecture 5.8 that survives §7.8 — that the running displacement $\hat h(t)$ accumulates a measurable rate asymmetry on packet difference sets between a ring task and a label-permuted control, an instrumentation question pre-specified for the trajectory-paired follow-up in the open-questions list below; they refute only the stronger operational corollary that a head-side cubic-aware addition is the right *intervention* at this operating point, when an *opposite* within-mode intervention (EGD/PGD) is already known to compress the same delay.

**What remains open about Conjecture 5.8.** Four follow-ups, in order of cost:

1. **$\alpha$-sweep across forms (B) and (C), with a uniform within-mode arm.** $\alpha = 0.1$ is a single operating point; the response surface across $\alpha \in \{0.01, 0.02, 0.05, 0.2, 0.5\}$ might expose a strength regime where one of the forms helps. Pair the sweep with a fourth arm that performs the uniform within-mode rescaling of [Saheb Pasand & Dohmatob 2026] (EGD) or the Gauss–Newton-style preconditioning of [Jiang et al. 2026] (PGD), since both are published methods that compress grokking delay on modular addition by acting *within* modes — exactly the within-packet direction §8.6 (form C) attempted to amplify via the cubic substrate. Cheapest remaining instance.
2. **Parameter-side cubic correction (K-FAC variant).** The head-side correction acts only on logit gradients; the §8 protocol envisaged K-FAC on the parameter side. Different functional space, plausibly different empirical behavior.
3. **Other rings.** $n = 12$ at a longer training budget — the regime where §6.5's pre-grokking signal is strongest ($+0.15$ separation) — would test whether the operational relevance shows up where the diagnostic signal is largest.
4. **Trajectory instrumentation paired with $\rho_\times$.** The endpoint of the present experiment is a coarse late-window interaction. Pairing it with the trajectory observables of [Truong et al. 2026a, b] — normalised spectral entropy $\widetilde{H}(t)$ and the joint $(V_t, \alpha_t)$ first-passage signature — would let one test whether the off-centroid analytic content of Theorems 5.2 / 5.6 produces a *measurable rate asymmetry* on packet difference sets between T1 and T2-strong, even at operating points where the late-window interaction is null. This is the cleanest empirical surface for the conjecture as stated.

We list these as open empirical questions. The pre-specified prediction was specific — positive interaction at the operating point of these experiments — and that prediction did not survive in any of three forms. The cleanest summary of what the experimental program established to date: the certificate has exact representational content and a measurable signature (§6.5), and the simplest natural cubic-aware corrections at $\alpha = 0.1$ on this substrate do not translate that content into an operational benefit on ring-task acquisition.

---

## 9. Scope and limitations

Theorem 3.5 is exact and unconditional **at the maximum-entropy point** of a categorical model with cyclic ring index $\mathbb{Z}/n\mathbb{Z}$, for composite $n$. The principal limitations, each stated as a definite next step rather than a hope:

**Off-centroid persistence.** The certificate is exact at $p_*$; the claim that the deficit persists, in a *task-relevant* form along a training trajectory leaving $p_*$ is Conjecture 5.8. Three pre-specified instances of the experiment (§8.6) on head-only cubic-aware variants — form (A) Newton, form (B) Neumann, and form (C) within-packet amplification, all at $\alpha = 0.1$ on $n = 30$ with Adam baseline — all gave Branch B (mean interactions $+0.003$, $-0.011$, $-0.004$ respectively). The conjecture as stated is refuted on these three instances. Form (B)'s consistently negative direction on T1 is the predicted-direction-reversed signature anticipated by the §6.5 diagnostic, and form (C)'s null result establishes that the sign-reversed test of the directionality reading is also negative. The remaining open follow-ups (§8.6) are an $\alpha$-sweep across forms (B) and (C) paired with an EGD/PGD uniform within-mode arm [Saheb Pasand & Dohmatob 2026; Jiang et al. 2026], a parameter-side K-FAC variant, longer-budget runs on other rings, and trajectory instrumentation (normalised spectral entropy $\widetilde{H}(t)$ and the joint $(V_t, \alpha_t)$ first-passage signature [Truong et al. 2026a, b]) paired with $\rho_\times$ on the same runs.

**Cyclic versus general finite abelian groups, and the non-abelian frontier.** The certificate is stated for $R = \mathbb{Z}/n\mathbb{Z}$. Whether the same selection rule, with the appropriate notion of conductor on the Pontryagin dual, controls the cubic on $R = (\mathbb{Z}/2)^k$, $R = \mathbb{Z}/p \times \mathbb{Z}/q$, and more generally on finite abelian groups, is a deterministic extension question; Appendix B specifies it and verifies it on five non-cyclic candidates ($\mathbb{Z}/2 \times \mathbb{Z}/4$, $\mathbb{Z}/4 \times \mathbb{Z}/4$, $\mathbb{Z}/2 \times \mathbb{Z}/8$, $(\mathbb{Z}/2)^2 \times \mathbb{Z}/4$, $\mathbb{Z}/3 \times \mathbb{Z}/9$). The genuinely *non-abelian* extension — to $S_5$, $S_6$, and other finite non-abelian groups where the dual is not given by a Pontryagin construction but by an irreducible-representation decomposition — is open. [Stander et al. 2024] reverse-engineer one-hidden-layer networks that have grokked $S_5$ and $S_6$ multiplication into circuits that decompose the group arithmetic via the permutation group's subgroups; the empirical *non-abelian grokking* substrate exists, and whether an analogous block-diagonality + cubic-selection-rule certificate holds on the irreducible-representation decomposition of $T_{p_*}\Delta_G^\circ$ is the natural next theoretical step. [Yıldırım 2026]'s observation that the cyclic-bypass spherical-topology prior *fails* on $S_5$ further signals that the cyclic and non-abelian cases live on structurally different geometric scaffolds and should not be conflated.

**Scaling the cyclic case.** The verified ladder $n \in \{6, 8, 12, 18, 30\}$ is small. [Kikuchi et al. 2026] scale modular-addition learning to very large $q$ (e.g., $q = 974\,269$, $N = 64$) via an auxiliary modulus $Kq$ that controls wrap-around frequency without inducing covariate shift; whether $\rho_\times$ of Definition 6.1 retains discriminative range across the same scale axis is a finite enumeration we do not perform here. The auxiliary-modulus construction is an *inputs-side* analogue of the manuscript's *labels-side* ring structure, and the natural diagnostic-deployment test is to pair the two: a $\rho_\times$ sweep on heads trained under the Kikuchi et al. auxiliary-modulus regime, at $q$ values spanning several orders of magnitude, to confirm that the structural-ceiling, insufficient-budget, and off-trajectory regimes catalogued in §6.5 and §6.6 are the only boundaries between the pre-grokking representation-learning phase and post-grokking saturation.

**Prime-power depth.** Two prime-power rings ($n = 8 = 2^3$, $n = 18 = 2 \cdot 3^2$) are already in the verified ladder. Whether the cross-packet structure becomes qualitatively richer at higher prime-power depths, particularly with respect to the three-distinct-conductor refinement (Remark 3.6), is a finite enumeration we do not perform here.

**Task-relevance.** The certificate proves the missed coupling is *missed*. It does not prove the missed coupling is *task-relevant*. Whether and where the cross-packet cubic carries gradient signal whose neglect costs performance is an empirical question, taken up by the diagnostic of Section 6 and the experiment of Section 8.

**Diagnostic deployment scope.** The retraining-free diagnostic of §6 is informative on heads in the pre-grokking representation-learning phase for the candidate ring structure (§6.5). The first-run application to pretrained Pythia checkpoints on a calendar-months conditional head (§6.6) finds the per-example $\rho_\times$ at the structural ceiling ($\overline{\rho_\times^{\text{ring}}} \in [0.96, 0.98]$ against ceiling $108/110 \approx 0.982$) across $N \in [70\text{M}, 1.4\text{B}]$ and $D \in [2.7 \times 10^{8}, 3.0 \times 10^{11}]$ tokens — a third diagnostic-boundary regime (*off-trajectory*) complementing the structural-degeneracy boundary at n = 8 and the insufficient-budget boundary at n = 18. A retraining-free deployment on an arbitrary checkpoint must therefore include an a priori check that the head's training regime exercises the ring structure (e.g., that the conditional head's top-1 accuracy or the deviation $\rho_\times - 108/110$ moves nontrivially with training step on the candidate ring); without that, a null reading is uninformative. Two natural extensions of §6.6 — quantized inference on Pythia-2.8B and Pythia-6.9B, and a clean ring-structured fine-tune of a pretrained Pythia-1B (which puts the pretrained substrate onto a ring-structure-learning trajectory and is the genuine pretrained-LLM test of Conjecture 5.8) — are listed at the end of §6.6 as the follow-on questions.

**Specific architectures and parametrizations.** The certificate concerns the categorical head and the output tangent space. Corollary 3.9 records the parameter-side lift for the simplest architecture — a single-layer softmax head: at the uniform configuration, the parameter Fisher is a scalar multiple of the identity on the centered logit subspace, and the conductor decomposition pulls back identically. For deeper architectures the lift is mediated by the network's Jacobian; whether the conductor decomposition pulls back coherently through nontrivial parametrizations, and what the analogous parameter-space selection rule looks like, are open. The diagnostic of Section 6 sidesteps this by working on the output side, where the certificate is exact.

The paper makes no claim that (i) trained networks' optimization trajectories are governed by conductor geometry; (ii) the missed coupling is necessarily task-relevant; (iii) the certificate is exact off $p_*$ in the same form; (iv) the certificate explains the grokking phenomenon; (v) any specific learning rate, baseline tuning, or benchmark number is endorsed by this work. The paper emits a theorem, a measurement procedure, and a pre-specified protocol; it emits no benchmark number.

---

## 10. Conclusion

We have shown, exactly and with no tunable parameter, that on a categorical model with cyclic ring index at maximum entropy, the Fisher form $g_{p_*}$ on the output tangent space has a structural null space precisely aligned with a cross-packet cubic coupling that the local information geometry provably carries. This is a single statement about a single object; its operational reach across the broader quadratic-curvature method class — natural gradient on the simplex (which reduces to $g_{p_*}$ at the output), K-FAC / Adam's empirical Fisher / the neural-tangent-kernel Gram (which target $g_{p_*}$ through Jacobian pullback and approximation), gradient-based edge-attribution scores in mechanistic circuit discovery [EAP, EAP-IG, ACDC, E-ACT] (which contract the head-side first-order gradient at the input to their per-edge scoring), and the kinetic action of any quadratic field theory on the Fisher base — is organized as a four-tier scope in §2.3, with each named method inheriting the null structure to the extent its curvature surrogate at $p_*$ on the output tangent space reduces to, pulls back from, or estimates $g_{p_*}$, and the attribution-side lift mediated by Proposition 3.11 and Corollary 3.12 at the input to the score. The result is, deliberately, small: it is the bounded form of a claim whose unbounded forms have repeatedly failed peer review for lack of an exact witness, and the bounded form has an exact witness and therefore no baseline to dispute. From the certificate we derived a retraining-free diagnostic on three substrates — the optimizer's update direction (§6.1), pretrained-LLM head logits (§6.6), and per-example circuit-attribution vectors (§6.7) — that converts a qualitative intuition into a measurement, and pre-specified a single sharp experiment whose negative outcome is as informative as its positive one. Two recent external lines of work are concordant with the model-class breadth of the claim, pending direct tests: a frontier-LLM circuit-discovery study in which a single discovered circuit reaches $87\%$ faithfulness on two structurally distinct mechanisms in a mixed dataset [Rai, Geva & Yao 2026] — the qualitative signature §7.7's reading expects, not a measurement of the certificate's diagnostic; the latter is what §6.7 pre-specifies on the released artifacts — and a nonlinear-sigma-model construction on statistical-manifold bases [Amaral 2025] in which the same purely-quadratic kinetic action appears with the cubic-extension structure §7.5 notes, an external structural alignment rather than a test of the certificate.

The certificate is exact *at the maximum-entropy point*; whether the deficit persists along a training trajectory through the interior of the simplex is the explicit pre-specified question of the experiment, related to the certificate as an unproven inductive step is related to a clean base case. We have stated this relation precisely throughout because the reason a bounded statement of this kind is worth making is that it is the only one that survives contact with an adversary who wants it to fail.

---

## Code and data availability

All scripts that produce the empirical numbers, tables, and figures of this paper, together with the run outputs they produced, are released as a companion repository:

* **GitHub:** <https://github.com/leomurillo/AI-ConductorBlindSpot>
* **Zenodo archive (DOI):** *to be assigned on arXiv submission* — `10.5281/zenodo.XXXXXXX`

The repository mirrors the file paths used in this manuscript: the non-cyclic certificate of Appendix B is produced by [`empirical/paper34_C4_noncyclic_certificate.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/paper34_C4_noncyclic_certificate.py); the synthetic-MLP demo of §6.5 by [`empirical/paper34_conductor_blindspot_demo.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/paper34_conductor_blindspot_demo.py); the Pythia checkpoint sweep of §6.6 by the scripts in [`empirical/pythia_rho_x_sweep/`](https://github.com/leomurillo/AI-ConductorBlindSpot/tree/main/empirical/pythia_rho_x_sweep); and the three head-only cubic-aware interventions of §8.6 by [`empirical/paper34_layer3_cubic_experiment.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/paper34_layer3_cubic_experiment.py). All checked-in run artifacts in `empirical/reports/` are the exact outputs cited in the manuscript.

---

## Acknowledgments

The author thanks Andrew Ross Thomson for sharing the unpublished manuscript on algebraic-geometric attention scoring that §7.6 builds on. Any errors are the author's own.

---

## References

[1] Ainsworth, S. K., Hayase, J., & Srinivasa, S. (2023). *Git Re-Basin: Merging models modulo permutation symmetries.* In *Proceedings of the 11th International Conference on Learning Representations* (ICLR 2023). arXiv:2209.04836.

[2] Amari, S. (1998). *Natural gradient works efficiently in learning.* Neural Computation, 10(2), 251–276. doi:10.1162/089976698300017746.

[3] Amari, S. (2016). *Information Geometry and Its Applications.* Applied Mathematical Sciences, Vol. 194. Springer Tokyo. doi:10.1007/978-4-431-55978-8.

[4] Amari, S., & Nagaoka, H. (2000). *Methods of Information Geometry.* Translations of Mathematical Monographs, Vol. 191. American Mathematical Society / Oxford University Press.

[5] Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., Persic, A., Qi, Z., Thompson, T. B., Zimmerman, S., Rivoire, K., Conerly, T., Olah, C., & Batson, J. (2025). *Circuit tracing: Revealing computational graphs in language models.* Transformer Circuits Thread, March 27, 2025. https://transformer-circuits.pub/2025/attribution-graphs/methods.html.

[6] Amaral, M. M. (2025). *Nonlinear sigma models on statistical manifolds: Equations of motion.* Gauge Freedom Field Notes (posted content, not peer-reviewed). DOI:10.65323/gf-lab.2025.001.

[7] Bhaskar, A., Wettig, A., Friedman, D., & Chen, D. (2024). *Finding transformer circuits with edge pruning.* arXiv preprint arXiv:2406.16778.

[8] Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). *Pythia: A suite for analyzing large language models across training and scaling.* In *Proceedings of the 40th International Conference on Machine Learning* (ICML 2023), PMLR 202:2397–2430. arXiv:2304.01373.

[9] Chentsov, N. N. (1982). *Statistical Decision Rules and Optimal Inference.* Translations of Mathematical Monographs, Vol. 53. American Mathematical Society. (Translated from the Russian *Statisticheskie reshayushchie pravila i optimal'nye vyvody*, Nauka, 1972.)

[10] Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). *Towards automated circuit discovery for mechanistic interpretability.* In *Advances in Neural Information Processing Systems 36* (NeurIPS 2023), pp. 16318–16352.

[11] Dandi, Y., Vilucchio, M., Arnaboldi, L., Tabanelli, H., & Krzakala, F. (2026). *Deep learning as neural low-degree filtering: A spectral theory of hierarchical feature learning.* arXiv preprint arXiv:2605.13612.

[12] Entezari, R., Sedghi, H., Saukh, O., & Neyshabur, B. (2022). *The role of permutation invariance in linear mode connectivity of neural networks.* In *Proceedings of the 10th International Conference on Learning Representations* (ICLR 2022). arXiv:2110.06296.

[13] Gu, J., Li, C., Liang, Y., Shi, Z., Song, Z., & Zhou, T. (2024). *Fourier circuits in neural networks: Unlocking the potential of large language models in mathematical reasoning and modular arithmetic.* arXiv preprint arXiv:2402.09469 (subsequently AISTATS 2025).

[14] Hanna, M., Pezzelle, S., & Belinkov, Y. (2024). *Have faith in faithfulness: Going beyond circuit overlap when finding model mechanisms.* In *First Conference on Language Modeling* (CoLM 2024).

[15] Jacot, A., Gabriel, F., & Hongler, C. (2018). *Neural tangent kernel: Convergence and generalization in neural networks.* In *Advances in Neural Information Processing Systems 31* (NeurIPS 2018), pp. 8571–8580. arXiv:1806.07572.

[16] Jiang, S., Voronin, A., Cyr, E., & Southworth, B. (2026). *On the convergence behavior of preconditioned gradient descent toward the rich learning regime.* arXiv preprint arXiv:2601.03162.

[17] Kikuchi, H., Masuya, R., Kawamoto, K., & Kera, H. (2026). *Learning large-scale modular addition with an auxiliary modulus.* arXiv preprint arXiv:2605.07648.

[18] Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization.* In *Proceedings of the 3rd International Conference on Learning Representations* (ICLR 2015). arXiv:1412.6980.

[19] Kramár, J., Lieberum, T., Shah, R., & Nanda, N. (2024). *AtP\*: An efficient and scalable method for localizing LLM behaviour to components.* arXiv preprint arXiv:2403.00745.

[20] Kunstner, F., Hennig, P., & Balles, L. (2019). *Limitations of the empirical Fisher approximation for natural gradient descent.* In *Advances in Neural Information Processing Systems 32* (NeurIPS 2019), pp. 4156–4167. arXiv:1905.12558.

[21] Liu, Z., Michaud, E. J., & Tegmark, M. (2023). *Omnigrok: Grokking beyond algorithmic data.* In *Proceedings of the 11th International Conference on Learning Representations* (ICLR 2023). arXiv:2210.01117.

[22] Mallinar, N., Beaglehole, D., Zhu, L., Radhakrishnan, A., Pandit, P., & Belkin, M. (2025). *Emergence in non-neural models: Grokking modular arithmetic via average gradient outer product.* In *Proceedings of the 42nd International Conference on Machine Learning* (ICML 2025), spotlight. arXiv:2407.20199.

[23] Manir, S. B., & Rupa, A. P. (2026). *A systematic empirical study of grokking: Depth, architecture, activation, and regularization.* arXiv preprint arXiv:2603.25009.

[24] Martens, J. (2020). *New insights and perspectives on the natural gradient method.* Journal of Machine Learning Research, 21(146), 1–76. arXiv:1412.1193.

[25] Martens, J., & Grosse, R. (2015). *Optimizing neural networks with Kronecker-factored approximate curvature.* In *Proceedings of the 32nd International Conference on Machine Learning* (ICML 2015), PMLR 37, pp. 2408–2417. arXiv:1503.05671.

[26] Mohamadi, M. A., Li, Z., Wu, L., & Sutherland, D. J. (2024). *Why do you grok? A theoretical analysis on grokking modular addition.* In *Proceedings of the 41st International Conference on Machine Learning* (ICML 2024), PMLR 235:35934–35967.

[27] Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). *Progress measures for grokking via mechanistic interpretability.* In *Proceedings of the 11th International Conference on Learning Representations* (ICLR 2023). arXiv:2301.05217.

[28] Nesterov, Y., & Polyak, B. T. (2006). *Cubic regularization of Newton method and its global performance.* Mathematical Programming, 108(1), 177–205. doi:10.1007/s10107-006-0706-8.

[29] Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization beyond overfitting on small algorithmic datasets.* In *ICLR 2022 Workshop on Mathematical and Empirical Understanding of Foundation Models* (MATH-AI). arXiv:2201.02177.

[30] Quirke, P., & Barez, F. (2024). *Understanding addition in transformers.* arXiv preprint arXiv:2310.13121.

[31] Rai, D., Geva, M., & Yao, Z. (2026). *Data-driven circuit discovery for interpretability of language models.* arXiv preprint arXiv:2605.09129. Code: https://github.com/Ziyu-Yao-NLP-Lab/data-driven-circuit-discovery.

[32] Rai, D., Zhou, Y., Feng, S., Saparov, A., & Yao, Z. (2024). *A practical review of mechanistic interpretability for transformer-based language models.* arXiv preprint arXiv:2407.02646.

[33] Rezaei Jafari, F., Eberle, O., Khakzar, A., & Nanda, N. (2025). *RelP: Faithful and efficient circuit discovery via relevance patching.* arXiv preprint arXiv:2508.21258.

[34] Saheb Pasand, A., & Dohmatob, E. (2026). *Egalitarian gradient descent: A simple approach to accelerated grokking.* In *Proceedings of the 14th International Conference on Learning Representations* (ICLR 2026). arXiv:2510.04930.

[35] Stander, D., Yu, Q., Fan, H., & Biderman, S. (2024). *Grokking group multiplication with cosets.* In *Proceedings of the 41st International Conference on Machine Learning* (ICML 2024). arXiv:2312.06581.

[36] Sun, A. (2025). *Circuit stability characterizes language model generalization.* arXiv preprint arXiv:2505.24731.

[37] Syed, A., Rager, C., & Conmy, A. (2024). *Attribution patching outperforms automated circuit discovery.* In *Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP*, pp. 407–416.

[38] Thomson, A. R. (2026). *Algebraic geometry as AI accelerant.* Unpublished manuscript; personal communication.

[39] Tripuraneni, N., Stern, M., Jin, C., Regier, J., & Jordan, M. I. (2018). *Stochastic cubic regularization for fast nonconvex optimization.* In *Advances in Neural Information Processing Systems 31* (NeurIPS 2018). arXiv:1711.02838.

[40] Truong, X. K., Truong, Q. H., Luu, D. T., & Phan, T. D. (2026a). *Spectral entropy collapse as a phase transition in delayed generalisation: An interventional and predictive framework for grokking.* arXiv preprint arXiv:2604.13123.

[41] Truong, X. K., Truong, Q. H., Luu, D. T., & Phan, T. D. (2026b). *First-passage prediction of grokking delay: A calibrated law under AdamW with causal validation.* arXiv preprint arXiv:2605.18845.

[42] Yıldırım, A. (2026). *The geometric inductive bias of grokking: Bypassing phase transitions via architectural topology.* arXiv preprint arXiv:2603.05228.

[43] Zhang, F., & Nanda, N. (2023). *Towards best practices of activation patching in language models: Metrics and methods.* arXiv preprint arXiv:2309.16042.

---

## Appendix A. Existence of cross-packet zero-sum triples for every composite $n$

We prove the existence statement used in Part 2 of Theorem 3.5: for every composite $n$, there exist $(k, \ell, m) \in (\mathbb{Z}/n\mathbb{Z} \setminus \{0\})^3$ with $k + \ell + m \equiv 0 \pmod{n}$ and the three conductors $\mathrm{cond}(k), \mathrm{cond}(\ell), \mathrm{cond}(m)$ not all equal.

**Proposition A.1.** *Let $n \geq 4$ be composite. Then there exists $(k, \ell, m) \in (\mathbb{Z}/n\mathbb{Z} \setminus \{0\})^3$ with $k + \ell + m \equiv 0 \pmod n$ and at least two distinct conductors among $\mathrm{cond}(k), \mathrm{cond}(\ell), \mathrm{cond}(m)$.*

*Proof.* Since $n$ is composite, $n$ has a smallest prime divisor $p$, and $p \leq \sqrt{n} \leq n/2$ for $n \geq 4$. Set $k = 1$, $\ell = p$, $m = n - 1 - p$. Then
$$
k + \ell + m = 1 + p + (n - 1 - p) = n \equiv 0 \pmod n,
$$
and the bound $p \leq n/2$ gives $m = n - 1 - p \geq n/2 - 1 \geq 1$ for $n \geq 4$, so $m \in \{1, \dots, n-1\}$ and $m \ne 0 \pmod n$. The conductors are
$$
\mathrm{cond}(k) = n / \gcd(1, n) = n, \qquad
\mathrm{cond}(\ell) = n / \gcd(p, n) = n / p < n,
$$
where the second equality uses $p \mid n$ (so $\gcd(p, n) = p$). Hence $\mathrm{cond}(k) \ne \mathrm{cond}(\ell)$, and the three conductors $\mathrm{cond}(k), \mathrm{cond}(\ell), \mathrm{cond}(m)$ are not all equal.

The auxiliary observation that the case $n = p + 1$ would require $p \mid n = p + 1$, hence $p \mid 1$, hence is impossible, confirms that the construction above is well-defined for every composite $n \geq 4$. $\square$

Proposition A.1 establishes the existence of cross-packet zero-sum triples with conductors *not all equal* — the condition Theorem 3.5 requires. A finer question is whether the conductors can be made *pairwise distinct*. This admits a clean iff characterization in terms of the prime factorization of $n$.

**Theorem A.2 (three-distinct-conductor characterization).** *Let $n \geq 4$ be composite. The following are equivalent:*

*(a) There exists $(k, \ell, m) \in (\mathbb{Z}/n\mathbb{Z} \setminus \{0\})^3$ with $k + \ell + m \equiv 0 \pmod n$ and $\mathrm{cond}(k), \mathrm{cond}(\ell), \mathrm{cond}(m)$ pairwise distinct.*

*(b) $n$ is not a prime power.*

*Proof.* *(a) $\Rightarrow$ (b).* We prove the contrapositive: if $n = p^a$ is a prime power, no zero-sum triple has three pairwise distinct conductors. The conductors of nontrivial characters of $\mathbb{Z}/p^a\mathbb{Z}$ are $\{p^i : 1 \leq i \leq a\}$; the conductor of $k \in \{1, \dots, p^a - 1\}$ is $p^a / \gcd(k, p^a) = p^{a - v_p(k)}$, where $v_p$ is the $p$-adic valuation. So $\mathrm{cond}(k) = p^i$ iff $v_p(k) = a - i$.

Suppose $\mathrm{cond}(k), \mathrm{cond}(\ell), \mathrm{cond}(m) = p^{i_1}, p^{i_2}, p^{i_3}$ are pairwise distinct with $i_1 < i_2 < i_3$ and $i_j \geq 1$ for all $j$. The $p$-adic valuations $v_p(k), v_p(\ell), v_p(m) = a - i_1, a - i_2, a - i_3$ are then strictly decreasing, and the minimum valuation $a - i_3 \leq a - 1 < a$ is realized exactly once. The $p$-adic valuation of a sum of integers with distinct $p$-adic valuations equals the minimum of the valuations, so $v_p(k + \ell + m) = a - i_3 < a$. But $k + \ell + m \equiv 0 \pmod{p^a}$ requires $v_p(k + \ell + m) \geq a$ — contradiction. Hence no such triple exists.

*(b) $\Rightarrow$ (a).* Let $p < q$ be two distinct prime divisors of $n$, which exist since $n$ is not a prime power. Set
$$
k_1 \;:=\; n/p, \qquad k_2 \;:=\; n/q, \qquad k_3 \;:=\; -(k_1 + k_2) \bmod n.
$$
We verify the three required properties.

*Sum.* $k_1 + k_2 + k_3 \equiv 0 \pmod n$ by construction.

*Nonzero entries.* $k_1 = n/p \in \{1, \dots, n-1\}$ since $p > 1$. Similarly $k_2 \in \{1, \dots, n-1\}$. For $k_3 \not\equiv 0 \pmod n$, we need $k_1 + k_2 = n/p + n/q = n(p + q)/(pq) \not\equiv 0 \pmod n$, i.e., $pq \nmid p + q$. Since $\gcd(p, q) = 1$ and $p, q \geq 2$ with $p \ne q$, we have $pq \geq 6$ while $p + q \leq pq - 1$ (verified directly: $p = 2, q = 3$ gives $pq - (p+q) = 1$; the gap grows for larger primes). Hence $pq \nmid p + q$ and $k_3 \ne 0$.

*Conductors.* $\gcd(k_1, n) = \gcd(n/p, n) = n/p$, so $\mathrm{cond}(k_1) = n/(n/p) = p$. Similarly $\mathrm{cond}(k_2) = q$. It remains to show $\mathrm{cond}(k_3) \notin \{p, q\}$. Suppose for contradiction $\mathrm{cond}(k_3) = p$, so $\gcd(k_3, n) = n/p$ and $k_3 = (n/p) r$ for some $r \in \{1, \dots, p-1\}$ with $\gcd(r, p) = 1$. Then $(n/p) r \equiv -(n/p + n/q) \pmod n$, so $(n/p)(r + 1) \equiv -n/q \pmod n$. Multiplying both sides by $pq/n$,
$$
q (r + 1) \;\equiv\; -p \pmod{pq}.
$$
Reducing modulo $q$: $0 \equiv -p \pmod q$, i.e., $q \mid p$ — impossible since $p < q$ and both prime. Hence $\mathrm{cond}(k_3) \ne p$. By the symmetric argument with $p$ and $q$ exchanged, $\mathrm{cond}(k_3) \ne q$. Thus the three conductors $(p, q, \mathrm{cond}(k_3))$ are pairwise distinct. $\square$

*Examples confirming Theorem A.2.* For $n = 6 = 2 \cdot 3$, the construction gives $(k_1, k_2, k_3) = (3, 2, 1)$ with conductors $(2, 3, 6)$, pairwise distinct. For $n = 12 = 2^2 \cdot 3$, the construction gives $(6, 4, 2)$ with conductors $(2, 3, 6)$, pairwise distinct. For $n = 30 = 2 \cdot 3 \cdot 5$, the construction gives $(15, 10, 5)$ with conductors $(2, 3, 6)$, pairwise distinct. For $n = 8 = 2^3$ (prime power), no such triple exists, in agreement with the computational artifact of Section 4: the $42$ cross-packet triples for $n = 8$ all have exactly two distinct conductors.

---

## Appendix B. Non-cyclic finite abelian extension (verified)

The certificate generalizes verbatim from the cyclic ring $\mathbb{Z}/n\mathbb{Z}$ to any finite abelian group $G$, with the Pontryagin dual replacing the additive characters of $\mathbb{Z}/n\mathbb{Z}$. The verification is a finite enumeration over $\widehat{G}$ in the exact-arithmetic family of Section 4; we report it here on a set of non-cyclic candidates.

### B.1 Setting

Let $G$ be a finite abelian group, used as the outcome index of a categorical model. The probability simplex $\Delta_G^\circ$ and its tangent space $T_p \Delta_G^\circ = \{u \in \mathbb{R}^G : \sum_{y \in G} u_y = 0\}$ are as before. The dual group $\widehat{G} := \mathrm{Hom}(G, \mathbb{C}^\times)$ has $|\widehat{G}| = |G|$ and consists of the characters $\chi : G \to \mathbb{C}^\times$. Character orthogonality on $G$ is
$$
\sum_{y \in G} \chi(y) \overline{\chi'(y)} = |G| \cdot \mathbf{1}\!\left[\chi = \chi'\right],
$$
proved by the same finite geometric-sum argument applied to the cyclic factors in the elementary-divisor decomposition $G \cong \mathbb{Z}/n_1 \times \dots \times \mathbb{Z}/n_r$ of $G$.

**Conductor on $\widehat{G}$.** For $\chi \in \widehat{G}$, write $\chi$ via the elementary-divisor decomposition as $\chi = \chi_{k_1} \otimes \dots \otimes \chi_{k_r}$ with $k_i \in \mathbb{Z}/n_i$, and define the *conductor of $\chi$* to be the order of $\chi$ in $\widehat{G}$. Equivalently, the conductor is the smallest finite abelian quotient $G / H$ such that $\chi$ factors through $G \to G/H$. In coordinates,
$$
\mathrm{cond}(\chi_{k_1, \dots, k_r}) = \mathrm{lcm}\!\left( \frac{n_i}{\gcd(k_i, n_i)} \,:\, k_i \ne 0 \right).
$$

**Conductor packets.** The packets of $\widehat{G}$ are the equivalence classes under equality of conductor, exactly as in the cyclic case.

**Selection rule.** The Amari–Chentsov cubic at the uniform point $p_* = \tfrac{1}{|G|}\mathbf{1}$ satisfies
$$
T_{p_*}(\chi, \chi', \chi'') = |G|^3 \cdot \mathbf{1}\!\left[\chi \cdot \chi' \cdot \chi'' = \mathbf{1}_{\widehat{G}}\right],
$$
where $\mathbf{1}_{\widehat{G}}$ is the trivial character. The selection rule replaces "$k + \ell + m \equiv 0 \pmod n$" with "$\chi \cdot \chi' \cdot \chi''$ is the trivial character in $\widehat{G}$"; in coordinates, $k_i + \ell_i + m_i \equiv 0 \pmod{n_i}$ for *every* component $i$ of the elementary-divisor decomposition.

**The non-cyclic certificate.** Define cross-packet to mean "the three conductors of $(\chi, \chi', \chi'')$ are not all equal". Then:

* the Fisher form $g_{p_*}$ on $\Delta_G^\circ$ is diagonal in the character basis of $\widehat{G}$, so block-diagonal across conductor packets;
* there exist triples $(\chi, \chi', \chi'')$ of nontrivial characters with $\chi \cdot \chi' \cdot \chi'' = \mathbf{1}_{\widehat{G}}$ and conductors not all equal, provided $G$ has at least two conductor packets (i.e., $G$ is not elementary $p$-abelian).

The total ordered selection-rule triples in $(\widehat{G} \setminus \{1\})^3$ equals $(|G|-1)(|G|-2)$ by the same counting argument as the cyclic case: $\chi''$ is determined modulo $G$ by $(\chi, \chi')$, and the only excluded case is $\chi \cdot \chi' = \mathbf{1}_{\widehat{G}}$.

### B.2 Verified examples

We enumerated the certificate on five non-cyclic candidate groups using the program of Section 4 generalized to the Pontryagin dual; results below. All cases verify Fisher block-diagonality (by character orthogonality) and a strictly positive cross-packet triple count.

| $G$ | $|G|$ | conductor packets ($d : |\mathcal{P}_d|$) | total = $(|G|{-}1)(|G|{-}2)$ | same-packet | cross-packet |
|---|---:|:---|---:|---:|---:|
| $\mathbb{Z}/2 \times \mathbb{Z}/4$ | $8$ | $2{:}3,\ 4{:}4$ | $42$ | $6$ | $36$ |
| $\mathbb{Z}/4 \times \mathbb{Z}/4$ | $16$ | $2{:}3,\ 4{:}12$ | $210$ | $102$ | $108$ |
| $\mathbb{Z}/2 \times \mathbb{Z}/8$ | $16$ | $2{:}3,\ 4{:}4,\ 8{:}8$ | $210$ | $6$ | $204$ |
| $(\mathbb{Z}/2)^2 \times \mathbb{Z}/4$ | $16$ | $2{:}7,\ 4{:}8$ | $210$ | $42$ | $168$ |
| $\mathbb{Z}/3 \times \mathbb{Z}/9$ | $27$ | $3{:}8,\ 9{:}18$ | $650$ | $218$ | $432$ |

The enumeration is deterministic, exact, and reproducible; the program and full per-$G$ JSON are archived at [`empirical/paper34_C4_noncyclic_certificate.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/paper34_C4_noncyclic_certificate.py) and [`empirical/reports/paper34_C4_noncyclic_summary.md`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/reports/paper34_C4_noncyclic_summary.md). On every candidate the certificate holds: Fisher block-diagonality is exact, and the cross-packet AC cubic triple count is strictly positive.

**Distinct distribution from the cyclic counterpart at equal order.** $\mathbb{Z}/2 \times \mathbb{Z}/4$ and $\mathbb{Z}/8$ both have order $8$ and an identical structural cap on total triples ($42$), yet differ on the same-packet count ($6$ vs $0$). The certificate is preserved in both, but the structural ceiling of $\rho_\times$ (the largest fraction attainable at a uniform $u$, equal to cross/total) shifts from $1.00$ in $\mathbb{Z}/8$ to $36/42 \approx 0.857$ in $\mathbb{Z}/2 \times \mathbb{Z}/4$. This is a property of the divisor lattice of $G$ alone.

**Diagnostic-degeneracy boundary.** A group $G$ has *no* same-packet selection-rule triples (and $\rho_\times \equiv 1$ on every $u$) when no nontrivial triple of characters with conductors all equal sums to the identity in $G$. Two structural families exemplify this:

* elementary $p$-abelian groups $G \cong (\mathbb{Z}/p)^k$ for prime $p$ — every nontrivial character has order $p$, so $\widehat{G}$ has only one conductor packet and the diagnostic is identically constant: $\rho_\times \equiv 0$ (a single packet means no triples are cross-packet);
* the cyclic case $n = 8$ noted in Section 6.5 (no $(k, k, k)$ with $3k \equiv 0 \pmod 8$, no other same-packet shape, all $42$ surviving triples are cross-packet, so $\rho_\times \equiv 1$).

Determining whether a given $G$ admits a non-degenerate diagnostic reduces to checking whether the closed-form same-packet count is strictly positive — a finite, decidable arithmetic question on $G$'s elementary-divisor decomposition.

---

## Appendix C. The computational artifact

This appendix records the verification of Theorem 3.5 over the ladder $n \in \{6, 8, 12, 18, 30\}$ in self-contained form.

**Setup.** For each $n$, the procedure of Section 4 is implemented as a deterministic enumeration in exact integer arithmetic (Python's arbitrary-precision integers; no `numpy`, no floating-point comparison). For each ordered pair $(k, \ell) \in \{1, \dots, n-1\}^2$, the predicate "$(k - \ell) \bmod n = 0$" is evaluated; for each ordered triple $(k, \ell, m) \in \{1, \dots, n-1\}^3$, the predicate "$(k + \ell + m) \bmod n = 0$" is evaluated. Conductors $\mathrm{cond}(k) = n / \gcd(k, n)$ are computed once per character.

### C.1 Conductor packets per ring

| $n$ | Packets ($d : \mathcal{P}_d$) |
|----:|:------------------------------|
| $6$ | $2{:}\{3\}$, $3{:}\{2, 4\}$, $6{:}\{1, 5\}$ |
| $8$ | $2{:}\{4\}$, $4{:}\{2, 6\}$, $8{:}\{1, 3, 5, 7\}$ |
| $12$ | $2{:}\{6\}$, $3{:}\{4, 8\}$, $4{:}\{3, 9\}$, $6{:}\{2, 10\}$, $12{:}\{1, 5, 7, 11\}$ |
| $18$ | $2{:}\{9\}$, $3{:}\{6, 12\}$, $6{:}\{3, 15\}$, $9{:}\{2, 4, 8, 10, 14, 16\}$, $18{:}\{1, 5, 7, 11, 13, 17\}$ |
| $30$ | $2{:}\{15\}$, $3{:}\{10, 20\}$, $5{:}\{6, 12, 18, 24\}$, $6{:}\{5, 25\}$, $10{:}\{3, 9, 21, 27\}$, $15{:}\{2, 4, 8, 14, 16, 22, 26, 28\}$, $30{:}\{1, 7, 11, 13, 17, 19, 23, 29\}$ |

### C.2 Fisher block-diagonality check

For each $n$ in the ladder, every ordered pair $(k, \ell)$ with $\mathrm{cond}(k) \ne \mathrm{cond}(\ell)$ was checked for the predicate $(k - \ell) \bmod n \ne 0$. All checks passed in exact arithmetic; no cross-packet Fisher entry was nonzero on any ring.

### C.3 Amari–Chentsov cross-packet cubic enumeration

For each $n$, all ordered triples $(k, \ell, m)$ with $k, \ell, m \in \{1, \dots, n-1\}$ and $(k + \ell + m) \bmod n = 0$ were enumerated; the count of *same-packet* triples (all three of $\mathrm{cond}(k), \mathrm{cond}(\ell), \mathrm{cond}(m)$ equal) and *cross-packet* triples (not all equal) was recorded. The total is $(n-1)(n-2)$ in closed form (for each $(k, \ell) \in \{1, \dots, n-1\}^2$, $m$ is determined modulo $n$; the only excluded case is $k + \ell \equiv 0$, giving $(n-1)^2 - (n-1)$).

| $n$ | Surviving triples (total $= (n{-}1)(n{-}2)$) | Same-packet | Cross-packet |
|----:|---------------------------------------------:|------------:|-------------:|
| $6$ | $20$ | $2$ | $18$ |
| $8$ | $42$ | $0$ | $42$ |
| $12$ | $110$ | $2$ | $108$ |
| $18$ | $272$ | $20$ | $252$ |
| $30$ | $812$ | $38$ | $774$ |

The same-packet entries account for triples of the form $(k, k, k)$ with $3k \equiv 0 \pmod n$ (yielding 2 such triples for every $n$ divisible by $3$, namely $k \in \{n/3, 2n/3\} \subset \mathcal{P}_3$) and, additionally for $n$ with a large prime-power factor such as $n = 18, 30$, triples of shape $(k, k, m)$ with $2k + m \equiv 0$ where both $k$ and $m$ lie in the large packet $\mathcal{P}_9$ ($n = 18$) or $\mathcal{P}_{15}$ ($n = 30$). The cross-packet count is the quantity asserted in Theorem 3.5 part 2 and is the empirical signature the diagnostic of Section 6 reads.

Sample cross-packet triple per ring, chosen to exhibit as many distinct conductor labels as the ring admits:

| $n$ | $(k, \ell, m)$ | $(\mathrm{cond}(k), \mathrm{cond}(\ell), \mathrm{cond}(m))$ | Distinct conductors |
|----:|:--------------:|:------------------------------------------------------------:|:-------------------:|
| $6$ | $(1, 2, 3)$ | $(6, 3, 2)$ | $3$ |
| $8$ | $(1, 1, 6)$ | $(8, 8, 4)$ | $2$ (maximum on $n = 8$) |
| $12$ | $(1, 2, 9)$ | $(12, 6, 4)$ | $3$ |
| $18$ | $(1, 2, 15)$ | $(18, 9, 6)$ | $3$ |
| $30$ | $(1, 2, 27)$ | $(30, 15, 10)$ | $3$ |

### C.4 Pre-claimed falsifier branches

Before the run, the falsifier branches were:

(F1) A nonzero cross-packet Fisher entry in the character basis on some ring (would falsify Lemma 3.1 on that ring).

(F2) An empty cross-packet cubic-triple set on some ring (would falsify part 2 of Theorem 3.5 on that ring).

Neither falsifier was realized on any ring in the ladder. The verification is complete in the sense that both pre-claimed branches are settled; the realized branch is recorded.

---

## Appendix D. The off-centroid Fisher expansion (full derivation)

We give the full residual bound for Lemma 5.1.

**Setup.** Let $p = p_* + h$ with $h \in T_{p_*}\Delta_R^\circ$ (so $\sum_y h_y = 0$) and $\|h\|_\infty < 1/n$, ensuring $p \in \Delta_R^\circ$. For each $y \in R$, write $r_y := h_y / p_{*, y} = n h_y$, so that $|r_y| < 1$ and $1/p_y = (1/p_{*, y}) \cdot 1/(1 + r_y)$.

**Geometric series.** For $|r_y| < 1$,
$$
\frac{1}{1 + r_y} = \sum_{j=0}^{\infty} (-r_y)^j = 1 - r_y + r_y^2 - r_y^3 + \cdots, \qquad \text{hence} \qquad \frac{1}{1 + r_y} = 1 - r_y + R_y(h),
$$
where the remainder satisfies $|R_y(h)| \leq r_y^2 / (1 - |r_y|)$.

**Application to $g_p$.** For tangent vectors $u, v \in T_{p_*}\Delta_R^\circ$,
$$
g_p(u, v) = \sum_{y \in R} \frac{u_y v_y}{p_y} = \sum_{y \in R} \frac{u_y v_y}{p_{*, y}} \cdot \frac{1}{1 + r_y} = \sum_{y \in R} \frac{u_y v_y}{p_{*, y}}(1 - r_y + R_y(h)).
$$
The first two terms expand as
$$
\sum_y \frac{u_y v_y}{p_{*, y}} = g_{p_*}(u, v), \qquad
\sum_y \frac{u_y v_y r_y}{p_{*, y}} = \sum_y \frac{u_y v_y h_y}{p_{*, y}^2} = T_{p_*}(h, u, v).
$$
Therefore
$$
g_p(u, v) = g_{p_*}(u, v) - T_{p_*}(h, u, v) + \mathcal{R}(u, v, h), \tag{D.1}
$$
where the residual is
$$
\mathcal{R}(u, v, h) = \sum_y \frac{u_y v_y R_y(h)}{p_{*, y}} = n \sum_y u_y v_y R_y(h).
$$

**Residual bound.** Using $|R_y(h)| \leq r_y^2 / (1 - |r_y|) = n^2 h_y^2 / (1 - n |h_y|)$,
$$
\big|\mathcal{R}(u, v, h)\big| \leq n^3 \sum_y \frac{|u_y v_y| \, h_y^2}{1 - n|h_y|}.
$$
For $\|h\|_\infty \leq \alpha/n$ with $\alpha \in (0, 1)$,
$$
\big|\mathcal{R}(u, v, h)\big| \leq \frac{n^3}{1 - \alpha} \sum_y |u_y v_y| h_y^2 \leq \frac{n^3}{1 - \alpha} \|u\|_2 \|v\|_2 \|h\|_\infty^2 \leq \frac{n}{1 - \alpha} \|u\|_2 \|v\|_2 \cdot \alpha^2 / n^{-1},
$$
or, more cleanly, $|\mathcal{R}(u, v, h)| = O(\|u\| \|v\| \|h\|^2)$ as $\|h\| \to 0$ at any chosen norm; the constant depends on $n$ and on $\alpha$.

This proves the residual bound implicit in (5.1). The leading correction is exactly $-T_{p_*}(h, u, v)$.

**Coordinate-free reformulation.** Define the linear operator $L_{p_*}(h) : T_{p_*}\Delta_R^\circ \times T_{p_*}\Delta_R^\circ \to \mathbb{R}$ by $L_{p_*}(h)(u, v) := T_{p_*}(h, u, v)$. Then (D.1) reads
$$
g_p = g_{p_*} - L_{p_*}(h) + O(\|h\|^2),
$$
displaying $L_{p_*}(h)$ as the first-order *form-valued* derivative of the Fisher metric in the direction $h$. By Lemma 3.4, $L_{p_*}(h)$ couples conductor packets if and only if $h$'s character expansion contains modes whose triple-sum selection rule activates cross-packet entries. The cross-packet content of $g_p$ at leading order is therefore generated by the cross-packet content of the Amari–Chentsov cubic acting on the displacement $h$ — the structural identity invoked in Section 5.2.

<!--
End of manuscript. Internal note: as of this revision, the analytic
skeleton of the paper is essentially complete:

  * Items 1, 4 (Theorems 5.2, 5.6) — off-centroid Fourier selection
    rule + all-orders cumulant expansion with closed Fourier-convolution
    cross-packet form. CLOSED.

  * Item 2 (Theorem A.2) — three-distinct-conductor characterization
    (iff n is not a prime power). p-adic valuation argument. CLOSED.

  * Item 3 (Corollary 3.8) — natural-gradient packet preservation at
    p_*. CLOSED. Operational statement at the update level: the
    natural-gradient direction at p_* is the simplex-tangent gradient
    up to a uniform 1/n^2 rescaling.

  * Item 5 (Corollary 3.9) — single-layer-softmax parameter pullback.
    CLOSED. The parameter Fisher at the uniform configuration is a
    scalar multiple of the identity on the centered logit subspace;
    the conductor decomposition pulls back identically.

Remaining work is empirical:

  * C3 — diagnostic sweep on public checkpoints with ring-structured
    categorical heads (Section 6 protocols D1/D2). Not implemented in
    this paper; specified.

  * C5 — pre-specified experiment (Section 8). Run only after C3,
    with both branches reported.

The IDEAL ABSTRACT block at the top of this file remains the target;
after C3 and C5 land, the abstract above can be revised to incorporate
the empirical results in their measured form.
-->
