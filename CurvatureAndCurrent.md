<!--
================================================================================
INTERNAL PROJECT LOG — NOT FOR PUBLICATION (Pandoc drops HTML comments)

  Filename:     CurvatureAndCurrent.md
  What:         The UNIFIED submission. Merges the two companion preprints —
                  * BeyondTheConductorBlindSpot.md   (the CUMULANT axis)
                  * IrreversibleBlindSpot.md          (the REVERSIBILITY axis)
                into one journal-length paper around a single thesis: the
                aligned-whitened SSL learner has TWO independent blind spots,
                both children of the conductor blind spot, and they are the two
                halves of one operator T = S + A.
  Status:       v1 (released 2026-06-02) — unified structure + all 9 theorems +
                E1-E16 + merged consequences/scope. Proofs in the appendix are
                CONDENSED; the two source papers carry the fully expanded versions
                and stay as the detailed references.
  Build:        ./build.sh CurvatureAndCurrent.md  (title/author/date from the H1
                block). Add the affiliation line before the first real build.
                NOTE: avoid \heb here (kept build-safe) — the gauge program is
                referenced in plain text.
  Theorem map (unified -> source):
     Thm 1 slow-feature recovery     = Beyond Thm 1
     Thm 2 affine<=>Gaussian          = Beyond Thm 2
     Thm 3 approximate recovery       = Beyond Thm 3
     Prop 1 / Thm 4 planning          = Beyond Prop 1 / Thm 4
     Lemma 1 skew forms vanish        = Irreversible Lemma 1
     Thm 5 irreversible blind spot    = Irreversible Thm 1
     Thm 6 predictor recovers arrow   = Irreversible Thm 2
     Thm 7 non-normal SVD + gap       = Irreversible Thm 3
     Thm 8 topological (cycle rank)   = Irreversible Thm 4
     Thm 9 continuum (Hodge H^1)      = Irreversible Thm 5
================================================================================
-->

# Curvature and Current: The Two Blind Spots of Self-Supervised Representation Learning

*One operator, two failure modes: what a linear probe cannot read, and what a symmetric objective cannot see*

**Leonardo Murillo Montero**

*leonardo.murillo@gmail.com*

*June 2, 2026*

---

## Abstract

The generic self-supervised recipe — pull positive pairs together, keep the embedding whitened — is, at its population optimum, a spectral statement about one object: the transition operator $T$ that carries one view of the world to the next. We show that this learner has **two independent, structural blind spots**, and that both are visible in the single decomposition $T = S + A$ into its symmetric and antisymmetric parts.

The first is the **cumulant axis** (*curvature*). The optimum recovers, in each coordinate, the slowest eigenfunction of $T$ — a clean consequence of the Ky Fan principle — and that eigenfunction is a straight line *exactly when the latent law is Gaussian in the observed coordinate* (Sturm–Liouville), and a recoverable but curved coordinate otherwise. So a **linear probe** of the representation is complete only on the Gaussian stratum and is provably lossy off it, by exactly the curvature of the recovered chart; recovery itself remains exact, up to a nonlinear chart and a block rotation, and faithful for planning. The second is the **reversibility axis** (*current*). The single-encoder objective is a functional of the symmetric part $S$ *alone*; the antisymmetric part $A$ — nonzero exactly when detailed balance fails, and equal to the probability current, the arrow of time — lies in the **kernel** of the objective. A world and its time-reverse therefore produce an identical loss landscape, so no symmetric objective can encode the direction of time; the cure is structural, a **predictor** (two-encoder) objective whose optimum is the singular value decomposition of $T$, which is precisely what the predictor head of BYOL/SimSiam/I-JEPA supplies. The irreversible content has a topological dimension: it is the cycle rank $b_1$ (first Betti number) of the transition graph, the de Rham dimension of harmonic $1$-forms of the state manifold.

The two axes are **orthogonal** — they are the curvature of $S$'s eigenfunctions and the existence of $A$, independent components of one operator — and we measure them varying with zero cross-talk. We give sixteen self-contained certificates: deterministic exact-arithmetic checks of every theorem (one regenerates the conductor blind spot's published counts; one measures the two registers locking as a near-Gaussian square law; one certifies the irreversibility gap is zero iff detailed balance; one proves the arrow's dimension equals the cycle rank), a trained-encoder confirmation that SGD reaches each optimum, and an end-to-end, efficient offline audit of $b_1$ (persistent homology for the faithful complex, sparse-Hodge eigenvalue counting for the cardinal number). The practical lesson is one sentence: a representation can be defeated on either axis or both, so a learned world model must be audited for *both* — read it through the nonlinear chart it actually learned, and check whether its arrow is even in the objective's range.

**Keywords:** self-supervised learning, identifiability, transition operator, slow feature analysis, Sturm–Liouville, Amari–Chentsov cubic, detailed balance, probability current, arrow of time, predictor networks, Hodge decomposition, Betti number, optimal control, world models.

---

## 1. Introduction {#sec:intro}

### 1.1 A finite fact, and its two continuous shadows

The *conductor blind spot* is a statement one can check by hand on a clock. Index the outcomes of a categorical head by $\mathbb{Z}/n\mathbb{Z}$. At the uniform point the Fisher information is diagonal in the additive-character basis, hence block-diagonal across conductor packets; but the next form in the cumulant tower, the Amari–Chentsov cubic, couples those packets exactly when three frequencies sum to zero modulo $n$. The cubic is real local structure that **no second-order curvature surrogate** — the form shared by natural gradient, K-FAC, Adam's empirical Fisher, gradient attribution — can represent. An order-three fact, invisible to an order-two instrument, certified in exact integer arithmetic.

This paper is about the *continuous* counterpart, where there is no ring and no integer certificate, only a network trained to make the embeddings of two related views agree. We find that the same theme — a second-order/symmetric instrument failing to carry structure the world's geometry holds — has **two distinct continuous shadows**, along two independent axes, and that both live in one operator.

### 1.2 The learner, and the one operator

A *world* is a hidden variable $z$ with stationary law $p$, observed through **positive pairs** $(z,z')$ — two augmentations of an image, two nearby frames, two consecutive states — generated by a stationary transition, and seen only through an unknown scrambling $x=g(z)$. The learner trains an encoder so paired embeddings align while the embedding stays whitened (the standard anti-collapse constraint). Everything the learner can be said to recover is governed by the **transition (conditional-expectation) operator**
$$(T\psi)(z) = \mathbb{E}[\psi(z')\mid z], \qquad T = S + A,\quad S=\tfrac12(T+T^{*}),\ A=\tfrac12(T-T^{*}),$$
where $T^{*}$ is the $L^2(p)$ adjoint (the time-reversed conditional expectation). $S$ is the reversible (symmetric) part; $A$ is the antisymmetric part, the part that vanishes exactly under detailed balance. The two blind spots are read off this one split.

### 1.3 The two blind spots

> **Curvature (cumulant axis).** Minimising alignment under whitening is a Rayleigh problem on $T$; its optimum is the span of the slowest eigenfunctions. That slow chart is a **straight line iff the world is Gaussian** in the observed coordinate, and a *curved* coordinate otherwise — so a **linear probe** of the representation is complete only on the Gaussian stratum. The defect is the curvature of $S$'s slow eigenfunction.

> **Current (reversibility axis).** The single-encoder objective depends on $S$ **alone**; the antisymmetric $A$ — the probability current, the **arrow of time** — is in its kernel. No symmetric objective can recover it; a **predictor** (two-encoder / SVD) objective can. The defect is the existence of $A$.

These are independent: the first is about the *shape* of $S$'s eigenfunctions, the second about the *presence* of $A$. A world can be defeated on either, or both.

### 1.4 Contributions

1. **The cumulant axis** (§3): an exact slow-feature recovery theorem (Thm 1), the affine-iff-Gaussian dichotomy (Thm 2, Sturm–Liouville), a graceful gap-controlled approximate bound (Thm 3), the two-towers distinction with a measured near-Gaussian bridge, and faithful planning in the recovered (Koopman) chart (Prop 1, Thm 4).
2. **The reversibility axis** (§4): the Irreversible Blind Spot (Lemma 1, Thm 5), its resolution by a predictor (Thm 6) with the BYOL/JEPA reading, the general non-normal form as an SVD with two charts and the irreversibility gap (Thm 7), and the topological dimension of the arrow — cycle rank, then Hodge $H^1$ (Thms 8–9).
3. **Independence** (§5): the two axes are orthogonal components of $T=S+A$, measured varying with zero cross-talk.
4. **Practice** (§6) and **sixteen certificates** (§7): deterministic exact checks of every claim, a trained-encoder confirmation, and an efficient end-to-end offline $b_1$ audit.

### 1.5 Relation to prior and concurrent work, stated once

The recovered object — slow eigenfunctions of a transition operator — is the content of slow feature analysis and diffusion-map geometry; what is new is the identifiability reading and its two sharp boundaries. The cumulant grading, conductor-packet decomposition, and cubic selection rule are the gauge program (T0–T7) and the conductor blind spot, on which this is built. A concurrent self-supervised result establishes the Gaussian/linear case — linear identifiability holds precisely for Gaussian latents — which is exactly the corner of the cumulant axis; we contain it as a boundary case and claim no more. The reversibility axis connects to nonequilibrium cycle currents (Schnakenberg) and combinatorial Hodge theory; the predictor reading connects to BYOL, SimSiam, and I-JEPA.

---

## 2. Setup: worlds, the operator, and what alignment optimises {#sec:setup}

A **world** is a triple $(p,T,g)$: a stationary law $p=\prod_i p_i$ on $\mathcal Z\subseteq\mathbb R^n$ (independent coordinates, the disentanglement premise); a transition generating pairs $(z,z')$, coordinatewise stationary additive-noise; and an unknown injective observation $x=g(z)$. Per coordinate, $T_i$ acts on $L^2(p_i)$. Two structural assumptions are named where they are used: **reversibility** ($T_i=T_i^{*}$) buys a real spectral theorem and is the hypothesis the *reversibility axis* removes; **compactness** (discrete spectrum) holds for Ornstein–Uhlenbeck and confining Langevin generators. Write the eigenpairs $1=\lambda_0>\lambda_1\ge\lambda_2\ge\cdots\ge0$ with $\varphi_0\equiv1,\varphi_1,\dots$ and collect the leaders into the **slow-feature chart** $\Phi_1(z)=(\varphi_1^{(1)}(z_1),\dots,\varphi_1^{(n)}(z_n))$.

The encoder is $h=f\circ g$. Training minimises **alignment** under **whitening**:
$$\min_h\ \mathbb E\|h(z')-h(z)\|^2\quad\text{s.t.}\quad \mathbb E[h_i]=0,\ \mathbb E[h_ih_j]=\delta_{ij}.$$
A two-line computation (Appendix A.1) using stationarity gives, with $T=\bigotimes_iT_i$,
$$\mathbb E\|h(z')-h(z)\|^2=\sum_i 2\big(1-\langle h_i,Th_i\rangle\big)=\sum_i 2\big(1-\langle h_i,S\,h_i\rangle\big),\tag{2.1}$$
the last equality because $A$ is skew-adjoint, so $\langle h_i,Ah_i\rangle=0$ (Lemma 1). Equation (2.1) is the hinge of the whole paper: **the objective is a Rayleigh quotient, and it sees only $S$.** Reading it for the *shape* of $S$'s eigenfunctions gives the cumulant axis (§3); reading it for the *absence* of $A$ gives the reversibility axis (§4). All results are statements about this population optimum (and a sufficiently expressive encoder that attains it), not about the SGD landscape — except §4's trained certificate, which checks the landscape directly.

---

## 3. The Cumulant Axis: curvature {#sec:cumulant}

### 3.1 Slow-feature recovery

**Theorem 1 (slow-feature recovery).** *Suppose the $T_i$ are self-adjoint compact and the* **slow-feature gap** *holds,*
$$\gamma:=\min_i\lambda_1^{(i)}-\max\Big(\max_i\lambda_2^{(i)},\ \max_{i\ne j}\lambda_1^{(i)}\lambda_1^{(j)}\Big)>0.\tag{G}$$
*Then every minimiser of the whitened-alignment problem has the form $h(z)=U\,\Phi_1(z)$, $U\in O(n)$, with $U$ free only within blocks of equal $\lambda_1^{(i)}$; the minimal value is $2\sum_i(1-\lambda_1^{(i)})$.* The proof (Appendix A.2) is Ky Fan's maximum principle plus the product spectrum. When the $\lambda_1^{(i)}$ are distinct (an *anisotropic* world) $U$ collapses to a signed permutation and the coordinates are individually identified: anisotropy buys disentanglement.

### 3.2 The Gaussian boundary

**Theorem 2 (affine recovery $\iff$ Gaussian).** *For a reversible diffusion $T_i=e^{\tau L_i}$, $L_i\psi=D(\psi''+(\log p_i)'\psi')$, with full support on $\mathbb R$ and natural boundary conditions, the slow eigenfunction $\varphi_1^{(i)}$ is affine in $z_i$ if and only if $p_i$ is Gaussian.* One ODE (Appendix A.3): an affine eigenfunction forces $(\log p_i)'$ affine, whose only normalisable full-support solution is the Gaussian. Combining Theorems 1–2:

> **Linear identifiability of the latent factors holds iff the world is Gaussian in the observed coordinate.** On the Gaussian stratum $\Phi_1=\mathrm{id}$ and $h(z)=Uz$ — a linear probe reads the latent whole. Off it, recovery is exact but through the curved chart $\Phi_1$, and any linear probe of the latent factors is lossy by exactly the departure of $\varphi_1$ from a line.

The cube-root world makes this concrete: a Gaussian $g$ observed as $z=g^3$ has slow chart $\varphi_1(z)=\sqrt[3]{z}$; a linear probe captures only $R^2\approx0.61$, the eigenfunction probe captures it exactly. "Gaussian" always means *in the observed coordinate*; the coordinate-free object is the operator's slow spectrum. (On finite/bounded domains the dichotomy weakens to an *affine corner*; a symmetric kernel can keep $\varphi_1$ affine on a non-Gaussian law — §3.4, Appendix C.)

### 3.3 Approximate recovery

**Theorem 3 (approximate recovery).** *Under approximate whitening $\|G-I\|_F\le\varepsilon$ and alignment excess $\delta$ (Dirichlet energy above the optimum), and (G),*
$$\min_{U\in O(n)}\mathbb E\|h-U\Phi_1\|^2\ \le\ \Big(\varepsilon+\sqrt{\tfrac{2\delta}{\gamma}}\Big)^2.$$
A variational Davis–Kahan ("trace-gap") argument (Appendix A.5): excess energy $\delta$ forces leakage $\theta^2\le\delta/\gamma$ out of the slow eigenspace, an orthogonal-Procrustes step converts it to a rotation error $\le2\theta^2$, and a triangle inequality pays $\varepsilon$. The constant that matters is the gap $\gamma$ in the denominator — a *checkable* hypothesis (a heterogeneous pair can violate (G), and the certificate then reports the guarantee undefined), and an estimable diagnostic of how well-posed recovery is.

### 3.4 Two towers, and the bridge to the blind spot

Two distinct spectral towers are in play and coincide only at the Gaussian. The **dynamical tower** is the eigenbasis of $T$; its slow member $\varphi_1$ is the recovery map (affine $\iff$ Gaussian). The **distributional tower** is the orthogonal-polynomial / cumulant grading of $p$ — what the gauge program and the conductor blind spot grade; its degree-one member is *always* affine, its degree-three member is the cubic. Off the Gaussian they split, and that splitting is what makes $\varphi_1$ a *mixture* of distributional orders — curved, hence invisible to a degree-one probe. The two boundaries — the linear-probe boundary (map degree one vs $\ge2$) and the cumulant boundary (order two vs $\ge3$) — are one boundary read a single step apart: the recovery map's leading nonlinear degree is one below the leading nonzero cumulant order. **Measured** (§7, E4): tuning a world off the Gaussian, the dynamical curvature $\nu_D=1-R^2$ scales as the square of the leading cumulant — $\nu_D\approx c\,\tilde\kappa_3^2$ for a skewed world, $\nu_D\approx c'\,\tilde\kappa_4^2$ for a symmetric one — both vanishing at the Gaussian (constants $0.0555,\,0.0104$). The ring is special: its scalar skewness is zero, yet the Amari–Chentsov *tensor*'s cross-packet components survive at the symmetric uniform point — which is why the conductor blind spot is an order-three statement there (Appendix C).

### 3.5 Planning in the curved chart

Because $\Phi_1$ is a diffeomorphism, control transports through it without loss. Fix a finite-horizon MDP on the latent; the agent sees only $\widehat z=\psi(z)$, $\psi=U\Phi_1$ a bi-Lipschitz diffeomorphism.

**Proposition 1 (exact planning).** *$\widehat V^\star_t(\psi(z))=V^\star_t(z)$ and $\widehat a^\star_t(\psi(z))=a^\star_t(z)$* — a change of variables (Appendix A.6), exact for any diffeomorphism. **Theorem 4 (approximate planning).** *With recovery RMS error $\eta=\varepsilon+\sqrt{2\delta/\gamma}$, Lipschitz values, transition stability, and a bi-Lipschitz chart, the true-world regret of planning in the learned chart is $\le C\,L\,T\,\eta$,* vanishing as $\eta\to0$ (Appendix A.6). Off the Gaussian the warp is *not* the identity and must be applied — and it is *recoverable from the dynamics*, not read free off the encoder.

*Remark (Koopman).* The chart linearises the dynamics: $\mathbb E[\varphi_1^{(i)}(z')\mid z]=\lambda_1^{(i)}\varphi_1^{(i)}(z)$, so the encoder is a Koopman embedding and **linear control (LQR/MPC) on the slow coordinates is optimal for the nonlinear world**, the warp entering only for latent-stated costs.

---

## 4. The Reversibility Axis: current {#sec:reversibility}

Section 3 read (2.1) for the *shape* of $S$'s eigenfunctions. Now read it for what is *missing*: $A$.

### 4.1 The Irreversible Blind Spot

**Lemma 1 (skew forms vanish).** *$A$ is skew-adjoint, so $\langle h,Ah\rangle=0$ for every real $h$.* (A real number equal to its own negative is $0$.)

**Theorem 5 (Irreversible Blind Spot).** *The single-encoder objective (2.1) is a functional of the symmetric part $S$ alone. Consequently (i) a world and its time-reverse $T\leftrightarrow T^{*}$ have a bit-for-bit identical loss landscape — every encoder, hence every minimiser, coincides; and (ii) the antisymmetric part $A$ — nonzero exactly when detailed balance fails, and equal to the generator of the probability current, the arrow of time — lies in the kernel of the objective.* No single-encoder representation trained this way can encode the direction of time. **Corollary.** Under (G) the minimiser is $h=U\Psi_1$ with $\Psi_1$ the top eigenspace of $S$ — the slow chart of the *reversibilised* dynamics $\tfrac12(T+T^{*})$: the world with its arrow erased. (And $S$ can still be non-Gaussian, so the chart can still be curved — the two axes compose.)

### 4.2 The predictor resolves it

To see $A$ the objective must probe $\langle f,Tg\rangle$ with $f\ne g$. Introduce a learnable linear **predictor** $P$ trained to predict the partner embedding; its optimum is $P^{\star}=\mathbb E[h(z')h(z)^{\top}]$ (whitened $h$) — the **matrix of $T$ itself** in the embedding basis, not its quadratic form.

**Theorem 6 (the predictor recovers the arrow).** *On the slow subspace the predictive optimum equals the operator block $B_{ab}=\langle\psi_a,T\psi_b\rangle$; its symmetric part is all the single encoder can see, its antisymmetric part $\tfrac12(B-B^\top)_{ab}=\langle\psi_a,A\psi_b\rangle$ is a faithful readout of the current, and time reversal sends $B\mapsto B^\top$. A strictly asymmetric optimal predictor exists iff detailed balance fails.* **Corollary (the architecture the theorem explains).** A symmetric Siamese objective is provably arrow-blind; the **predictor head of BYOL, the stop-gradient asymmetry of SimSiam, the predictor of I-JEPA** are the minimal devices that lift the objective out of the blind spot. On temporal/causal data — where the transition is genuinely irreversible — this predicts that predictive architectures capture what symmetric contrastive ones cannot, and locates the reason in $A$.

On the drift ring $\mathbb{Z}/n$ ($q\ne b$) the block is $B=\big(\begin{smallmatrix}\alpha&\beta\\-\beta&\alpha\end{smallmatrix}\big)$ with $\beta=(q-b)\sin\tfrac{2\pi}{n}$ the arrow; the predictor turns the static ring embedding into a rotational flow whose direction *is* the arrow of time, reversing with the drift, and vanishing at detailed balance $q=b$.

### 4.3 The general non-normal form: two charts and the gap

The ring is *normal*; the arrow is then the imaginary spectrum. Dropping normality: let $T=\sum_k\sigma_k\,u_k\langle v_k,\cdot\rangle$ be the SVD on the mean-zero subspace.

**Theorem 7 (general recovery; two charts; the gap).** *The two-encoder objective $\min_{f,g}\mathbb E\|f(z')-g(z)\|^2$ over whitened $f,g$ is optimised by $g_i=u_i$ (left/output chart), $f_i=v_i$ (right/input chart). The input and output charts span the same subspace iff $T$ is normal; for non-normal $T$ they differ, and the principal angle between them is a coordinate-free measure of irreversibility (time reversal swaps them). Moreover, with $\lambda_i(S)\le\sigma_i(T)$ (Fan–Hoffman),*
$$\Delta:=\sum_i\sigma_i(T)-\sum_i\lambda_i(S)\ \ge\ 0$$
*is the predictable correlation the single encoder forfeits — zero iff detailed balance, positive with a current (two faces: a phase deficit for a normal current, a chart split for a non-normal one).* (Proof: Eckart–Young for the bilinear form; Appendix B.)

### 4.4 The topology of the arrow

The arrow is not a scalar but a homology-class count. The stationary current $J_{ij}=\pi_iT_{ij}-\pi_jT_{ji}$ is antisymmetric and **divergence-free** ($\sum_jJ_{ij}=0$, Kirchhoff, by stationarity), hence lives in the cycle space of the transition graph.

**Theorem 8 (topological dimension).** *The single-encoder objective is invariant under any change of the current $J$, while the predictor recovers it; so the dimension of the irreversible content invisible to a symmetric objective equals the cycle rank*
$$\dim(\text{irreversible blind spot})\;=\;\beta_1(G)\;=\;E-V+1.$$
*Detailed balance is $J=0$; the drift ring ($\beta_1=1$) is the minimal nonzero case.* **Theorem 9 (continuum).** *On a state manifold the count refines, via Hodge–Helmholtz, to $\dim H^1$ — the harmonic $1$-forms — the de Rham shadow of the cycle rank.* A tree carries no recoverable arrow; a $3\times3$ torus hides a ten-dimensional current behind one reversible backbone. Connectivity is a design knob for irreversible expressivity.

---

## 5. The two axes are independent {#sec:independence}

The two blind spots are different components of one operator $T=S+A$: the cumulant defect $\nu_D$ reads the **curvature of $S$'s slow eigenfunction**, the reversibility defect $\Delta$ reads the **antisymmetric part $A$**. They are therefore independently dialable. On a single product world — a warped Gaussian factor $U$ (knob $\gamma$ = non-Gaussianity) times a drift ring $V$ (knob $\rho$ = irreversibility) — we measure $\nu_D$ and $\Delta$ over the $(\gamma,\rho)$ grid (§7, E14): $\nu_D$ depends on $\gamma$ alone, $\Delta$ on $\rho$ alone, with **zero cross-talk to machine precision** (the heat-maps are orthogonal — one banded horizontally, the other vertically). All four corners of (Gaussian / non-Gaussian) $\times$ (reversible / irreversible) are realised. A representation can be defeated on either axis or both; an audit that checks only linearity misses the arrow, and one that checks only reversibility misses the curved chart.

This is the unified statement the conductor blind spot anticipated: a second-order/symmetric class is blind along **two** independent axes — the cumulant order at which second-order statistics stop sufficing (curvature), and the symmetry under which a quadratic form is invariant (current) — and a learned world model lives at a point in this $2\times2$ plane.

---

## 6. Consequences for practice {#sec:practice}

**Interpretability.** A linear probe is complete only on the Gaussian stratum (Thm 2); wherever factors are non-Gaussian it reads a strict shadow, the missing part being the curvature of $\varphi_1$. The remedy is the *right* probe — one matched to the recovered chart $\Phi_1$ (estimable from the dynamics) — not a bigger one. A pretrained-model signature is visible, unforced (§7, E5): at middle layers the heavy non-Gaussian residual-stream directions *tend* to be the ones whose recovered slow feature is curved.

**Optimization, and three "second-orders."** (a) A linear *readout* is order-one, complete only on the Gaussian stratum. (b) A second-order *curvature model* (Fisher/K-FAC) cannot carry the order-three cubic — the conductor blind spot's own statement. (c) But an *objective* using second moments is **not** so limited: our whitened-alignment objective imposes a second-moment constraint yet recovers the fully nonlinear chart. The ceiling is in the linear readout and the quadratic curvature *model*, not in second-order *training*.

**Auditing world models — both axes.** Identifiability holds for non-Gaussian worlds, but only up to a *recoverable nonlinear chart and a block rotation*, and only under the gap (G); an audit assuming linear correspondence mis-certifies a perfectly identified representation as entangled. And a *symmetric* audit cannot see the arrow at all. So a learned world model must be audited on **both** axes: read it through its chart, and ask whether its current is even in the objective's range.

**Grokking.** On modular addition the generalising Fourier circuit is an order-three object (the cubic selection rule); training to the grok shows the distributional register (head additivity) and the dynamical register (embedding circularisation) rising **together** to validation accuracy, flat on a structure-free control (§7, E6) — a prediction tested, not a consequence proven.

**Design levers.** *Spectral-gap engineering:* positive pairs define $T$, so the lever is the gap $\gamma$ — computable before training; to make a representation ignore a nuisance, demote it in the spectrum. *The predictor:* on temporal/causal data, use a predictive (two-encoder) objective — a symmetric one is arrow-blind by Theorem 5. *Anisotropy* buys disentanglement (Thm 1); *Koopman* buys linear control (§3.5).

**The arrow as an audit quantity, made efficient.** The arrow's *strength* $\Delta$ is a free, robust by-product of the predictor (estimable from finite samples at $O(d^3)$ — §7, E13). Its *dimension* $b_1$ is genuine topology and is an **offline** audit, now end-to-end and cheap (§7, E15–E16): persistent homology supplies the faithful complex, sparse-Hodge eigenvalue counting reads $b_1$ — exact, scalable, and zero iff the world is reversible.

---

## 7. Certificates {#sec:certificates}

Every claim is checked by a self-contained program in `empirical/apex_recovery/` (shared toolkit `apex_world.py`; one-command `run_all.py` with a pass/fail gate). There is no machine learning in the deterministic ones: the theory collapses learning to one operator's spectral problem, so they are exact linear algebra. Two certificates (E11, E16) use external libraries (torch, GUDHI) and are runnable but outside the dependency-light gate; E5/E6 are observational.

| # | axis | claim | result |
|---|---|---|---|
| E1 | curvature | slow chart + Gaussian boundary | $\nu_D=0$ Gaussian only; cube-root recovered to discretisation, linear probe $R^2=0.61$ |
| E2 | curvature | approximate bound + the gap (G) | bound holds 15/15; a heterogeneous pair violates (G), guarantee correctly undefined |
| E3 | curvature | finite/distributional tower | Hankel rank closure; **regenerates the blind-spot counts $18,42,108,252,774$** |
| E4 | curvature | the two-towers bridge | $\nu_D\propto(\text{leading cumulant})^2$; constants $0.0555$ (order 3), $0.0104$ (order 4) |
| E5 | curvature | pretrained-model signature | middle-layer phenomenon (rank corr $\to+0.41$ at layers 3–5); observational |
| E6 | curvature | grokking co-emergence | two registers rise together to val-acc 1; flat on control ($p=113$, $n=110$, $n=121$) |
| E7 | curvature | planning faithfulness | chart agent exact ($<10^{-9}$); linear planner sub-optimal off-Gaussian; Thm 4 bound holds |
| E8 | curvature | SGD reaches the chart | trained encoder recovers $\varphi_1$, corr $0.96$–$1.00$ |
| E9 | current | irreversible blind spot (normal ring) | identity $3\times10^{-16}$; loss reversal-invariant $9\times10^{-16}$; predictor exact; OFF at $q=b$ |
| E10 | current | non-normal SVD + gap | charts differ to $60^\circ$; $\Delta=0.40/0.025/0$ (conveyor/ring/reversible) |
| E11 | current | SGD reaches the predictor | **(GCP/torch)** single-encoder reversal-blind; predictor $\hat\beta$ flips $+0.17\to-0.17$ |
| E12 | current | topological dimension | recovered current dim $=\beta_1$ (ring 1, theta 2, $K_4$ 3, torus 10); Hodge $b_1$ refinement |
| E13 | both | what is cheaply estimable | $\Delta$ converges from samples ($2.6\%\to0.2\%$); exact $b_1$ is not loop-cheap |
| E14 | both | the two axes are orthogonal | $\nu_D\perp\rho$, $\Delta\perp\gamma$, zero cross-talk; all four corners realised |
| E15 | current | efficient $b_1$ engine | sparse-Hodge nullity exact (torus 2 to $L=40$ in $0.4$ s, sphere 0, two-tori 4) |
| E16 | current | faithful complex (end-to-end) | **(GUDHI)** persistent $b_1=1,2,0,3$ exact, robust; composes with E15 |

---

## 8. Scope and limitations {#sec:scope}

The theorems are conditional and stated as such. **Reversibility** is the hypothesis the cumulant axis assumes and the reversibility axis removes — §4 needs only the $L^2(p)$ adjoint and holds for every encoder. **Product (independent-factor) structure** underlies the single-excitation accounting of Theorem 1; correlated latents give back the joint slow eigenspace. **Discrete spectrum** holds for OU/confining Langevin generators. The **gap (G)** is real and checkable. Recovery is **up to a nonlinear chart** — the correct statement, not a weakness. On the reversibility axis: the **continuum** Hodge count (Thm 9) is stated where the finite cycle-rank count (Thm 8) is proved; the **trained** confirmation (E11) is one off-host run; and **efficient $b_1$ from raw samples** (E16) is solved for the cases studied — hardening it on large real encoders (sub-sampling, effective dimension, persistence-stability confidence) is the engineering follow-up. **Empirically**, the deterministic certificates are synthetic and exact by design; E5/E6 are observational; large-scale real-SSL benchmarks with known non-Gaussian *and* irreversible factors are the natural next test.

---

## 9. Conclusion {#sec:conclusion}

The conductor blind spot showed, on a clock, that an order-two instrument cannot read an order-three fact. In the continuous, learned setting that single fact opens into two: the aligned-whitened encoder **linearizes** and it **symmetrizes**, and so it is blind to *curvature* and to *current* — the shape of $S$'s slow eigenfunctions off the Gaussian, and the antisymmetric part $A$ off detailed balance. Both are read from one operator $T=S+A$; both have exact, certified boundaries; both have constructive cures — the nonlinear chart and the predictor; and they are independent, so a learned world model must be audited for both. The arrow even has a topological dimension, $b_1$, which we can now estimate end-to-end and cheaply. The practical lesson is to stop reading representations as if they were flat and reversible: read them through the chart they learned, and check whether the arrow of time was ever in the objective's range.

---

## Appendix A. Proofs — the cumulant axis (condensed) {#sec:appA}

**A.1 Alignment as a Rayleigh form.** With $\{h_i\}$ whitened and stationarity, $\mathbb E[h_i(z')h_i(z)]=\langle h_i,Th_i\rangle$, giving (2.1); $\langle h_i,Th_i\rangle=\langle h_i,Sh_i\rangle$ by Lemma 1. $\square$

**A.2 Theorem 1.** Ky Fan: for self-adjoint compact $T$ and orthonormal $\{h_i\}$ in the mean-zero subspace, $\max\sum_i\langle h_i,Th_i\rangle$ is the sum of the top $n$ non-trivial eigenvalues, attained on their eigenspace (unique up to mixing in degenerate blocks). The spectrum of $\bigotimes_iT_i$ is $\{\prod_i\lambda_{k_i}^{(i)}\}$; under (G) the top $n$ non-trivial modes are the single-excitations $\varphi_1^{(i)}$. $\square$

**A.3 Theorem 2.** In the diffusion realisation $L_i\psi=D(\psi''+(\log p_i)'\psi')$: if $\varphi_1=az+b$ then $L_i\varphi_1=aD(\log p_i)'$, and $L_i\varphi_1=-\mu\varphi_1$ forces $(\log p_i)'$ affine, so $\log p_i$ is quadratic; full support + natural BC leave only the Gaussian. $\square$

**A.5 Theorem 3 (trace-gap).** With $A_0=I-T$ on the mean-zero subspace, gap $\gamma=a_{n+1}-a_n$: excess energy $\delta$ gives leakage $\theta^2\le\delta/\gamma$; an orthogonal-Procrustes step gives $\min_U\|\tilde h-U\Phi_1\|^2\le2\theta^2\le2\delta/\gamma$; inexact whitening adds $\|h-\tilde h\|\le\varepsilon$ via $\sum_k(\sqrt{g_k}-1)^2\le\|G-I\|_F^2$; the triangle inequality, squared, is the claim. $\square$

**A.6 Planning.** Prop 1 is backward induction with the pushforward identity (a change of variables in the Bellman recursion). Thm 4 is the performance-difference lemma: a bi-Lipschitz chart turns recovery error $\eta$ into a state-decoding error $\le\kappa\eta$, transition stability keeps it from compounding, and an $L$-Lipschitz value propagates it over $T$ steps: regret $\le2\kappa LT\eta$. $\square$

## Appendix B. Proofs — the reversibility axis (condensed) {#sec:appB}

**B.1 Lemma 1 / Theorem 5.** $A^{*}=-A$ gives $\langle h,Ah\rangle=-\langle h,Ah\rangle=0$; so (2.1) depends on $S$ only. $S$ is the symmetric part of both $T$ and $T^{*}$, so $T\mapsto T^{*}$ fixes the objective; $\partial(\text{objective})/\partial A\equiv0$. $\square$

**B.2 Theorem 6.** The predictor normal equations give $P^{\star}=\mathbb E[h(z')h(z)^\top]$, entries $\langle\psi_b,T\psi_a\rangle$; restrict to the slow subspace and split $T=S+A$ to read off the symmetric (visible) and antisymmetric (arrow) parts; $T^{*}$ swaps left/right. $\square$

**B.3 Theorem 7.** Eckart–Young/Ky Fan for the bilinear form $\sum_i\langle g_i,Tf_i\rangle$ gives the optimum $\sum_{i\le d}\sigma_i$ at the singular functions; normality $\iff$ left and right singular subspaces coincide; Fan–Hoffman $\lambda_i(S)\le\sigma_i(T)$ gives $\Delta\ge0$, $=0$ iff $T$ is self-adjoint on the top subspace. $\square$

**B.4 Theorem 8.** $J$ is antisymmetric and divergence-free (stationarity $\Rightarrow$ Kirchhoff), so $J\in\ker\partial_1$, of dimension $E-(V-1)=\beta_1$ for connected $G$; a skew, divergence-free circulation changes only $A$, leaving $S,\pi$ fixed, so the whole $\beta_1$-family shares one single-encoder optimum, all of it recovered by the predictor. $\square$

## Appendix C. The conductor blind spot as the finite origin {#sec:appC}

At the uniform point on $\mathbb{Z}/n$, the Fisher form is block-diagonal across conductor packets while the Amari–Chentsov cubic couples them under $k+\ell+m\equiv0\ (\mathrm{mod}\ n)$ — order-two vs order-three in the *distributional* tower. At the Gaussian/uniform corner the dynamical and distributional towers coincide (Hermite), and the cubic is the degree-three rung one below the recovered degree-one mode; off the corner they split. The cubic's cross-packet components are nonzero at the *symmetric* uniform point precisely because the ring's arithmetic survives the symmetry (the selection rule has solutions of unequal conductor) — the scalar-vs-tensor distinction that keeps order three alive where a scalar skewness vanishes, and the reason E3 regenerates the published counts on the same integers the continuous theory points to.

---

## References {#sec:references}

[1] Fan, K. (1949). *On a theorem of Weyl concerning eigenvalues of linear transformations.* PNAS 35(11), 652–655.

[2] Davis, C., & Kahan, W. M. (1970). *The rotation of eigenvectors by a perturbation. III.* SIAM J. Numer. Anal. 7(1), 1–46.

[3] Wiskott, L., & Sejnowski, T. J. (2002). *Slow feature analysis.* Neural Computation 14(4), 715–770.

[4] Coifman, R. R., & Lafon, S. (2006). *Diffusion maps.* Appl. Comput. Harmon. Anal. 21(1), 5–30.

[5] Bakry, D., Gentil, I., & Ledoux, M. (2014). *Analysis and Geometry of Markov Diffusion Operators.* Springer.

[6] Amari, S., & Nagaoka, H. (2000). *Methods of Information Geometry.* AMS / Oxford.

[7] Rényi, A. (1959). *On measures of dependence.* Acta Math. Hungar. 10, 441–451.

[8] Grill, J.-B., et al. (2020). *Bootstrap your own latent (BYOL).* NeurIPS 2020.

[9] Chen, X., & He, K. (2021). *Exploring simple Siamese representation learning (SimSiam).* CVPR 2021.

[10] Assran, M., et al. (2023). *Self-supervised learning from images with a joint-embedding predictive architecture (I-JEPA).* CVPR 2023.

[11] Kakade, S., & Langford, J. (2002). *Approximately optimal approximate reinforcement learning.* ICML 2002.

[12] Schnakenberg, J. (1976). *Network theory of microscopic and macroscopic behavior of master equation systems.* Rev. Mod. Phys. 48(4), 571–585.

[13] Jiang, X., Lim, L.-H., Yao, Y., & Ye, Y. (2011). *Statistical ranking and combinatorial Hodge theory.* Math. Program. 127(1), 203–244.

[14] Klindt, D., LeCun, Y., & Balestriero, R. (2026). *When does LeJEPA learn a world model?* arXiv:2605.26379.

[C1] Murillo Montero, L. (2026). *The Conductor Blind Spot of the Quadratic Curvature Class on Ring-Structured Categorical Heads.* Zenodo, DOI 10.5281/zenodo.20330864.

[C2] Murillo Montero, L. (2026). *Beyond the Conductor Blind Spot: Eigenfunction Identifiability and Planning in Non-Gaussian Worlds.* (Companion; the cumulant axis, full proofs.)

[C3] Murillo Montero, L. (2026). *The Irreversible Blind Spot: Why Symmetric Self-Supervised Objectives Cannot See the Arrow of Time.* (Companion; the reversibility axis, full proofs.)
