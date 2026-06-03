<!--
================================================================================
INTERNAL PROJECT LOG — NOT FOR PUBLICATION
HTML comments; Pandoc drops them on the way to LaTeX, so they never reach the PDF.

  Filename:     IrreversibleBlindSpot.md
  Pairs with:   BeyondTheConductorBlindSpot.md (the parent) and ConductorBlindSpot.md
  Build:        ./build.sh IrreversibleBlindSpot.md  (same toolchain)
  Target venue: arXiv (cs.LG / stat.ML; math.PR secondary)
  Status:       v1 (released 2026-06-01). The reversibility axis in full: Lemma 1,
                Theorems 1-5, the predictor resolution, and the topological
                (cycle-rank -> Hodge b_1) dimension. Certificates E9, E10, E12, E13,
                E14, E15 are in the run_all pass/fail gate; E11 (trained two-encoder,
                torch) and E16 (point-cloud Betti, GUDHI) are runnable but outside it.
                Promotes the parent paper's "non-reversible case left to follow-up"
                (§8) from a deferral to a stated, proved, certified result.

FRAMING (locked 2026-06-01)
  * This is a SIBLING blind spot, not a sequel. CBS and the parent live on the
    CUMULANT axis (order-2 instrument blind to order-3 fact). This paper is the
    REVERSIBILITY axis (symmetric instrument blind to antisymmetric fact). The
    unifying claim: a second-order/symmetric class is blind along TWO independent
    axes, and the arrow of time is the second one.
  * Lead with the CBS ring + a current (concrete, same object), then the general
    operator theorem, then the predictor resolution. Eulerian: smallest worked
    instance first.
  * Cite the parent for the Rayleigh-form reduction (its App A.1) rather than
    re-deriving. Cite CBS for the ring object. Predictor corollary names
    BYOL/SimSiam/I-JEPA once, professionally, as the architectures the theorem
    explains — no leverage on names.
  * E9 (e9_irreversible_ring.py) is the exact certificate; it is in run_all's gate.
================================================================================
-->

# The Irreversible Blind Spot: Why Symmetric Self-Supervised Objectives Cannot See the Arrow of Time

*A second blind spot, on the reversibility axis: the predictor is what lifts it*

**Leonardo Murillo Montero**

*leonardo.murillo@gmail.com*

*June 1, 2026*

---

## Abstract

The companion paper *Beyond the Conductor Blind Spot* showed that the standard self-supervised recipe — align positive pairs, keep the embedding whitened — recovers, at its population optimum, the slowest eigenfunctions of the pair-generating transition operator $T$, and that a *linear* reading of that representation is complete exactly on the Gaussian stratum. That analysis, like the conductor blind spot it generalises, lives on one axis: the **cumulant** axis, where an order-two instrument cannot carry an order-three fact. This note exhibits a second, independent axis of the same phenomenon — the **reversibility** axis — and the architecture that resolves it.

We prove that the single-encoder alignment objective is a functional of the **symmetric part** $S=\tfrac12(T+T^{*})$ of the transition operator *alone*. The antisymmetric part $A=\tfrac12(T-T^{*})$ — which is nonzero exactly when detailed balance fails, and which *is* the probability current, the arrow of time — lies in the **kernel** of the objective. The consequence is exact and falsifiable: a world and its time-reverse produce a bit-for-bit identical loss landscape, so no representation trained this way can encode the direction of time, even in principle. This is a blind spot of the same family as the conductor blind spot — a symmetric (second-order) instrument blind to an antisymmetric fact — now on the cycle $\mathbb{Z}/n$ with a current rather than a cumulant. The resolution is structural, not a matter of more data or capacity: the irreversible content is recoverable only by an objective sensitive to $\langle f,Tg\rangle$ with $f\neq g$ — a **predictive (two-encoder)** objective, whose optimum is the singular value decomposition of $T$. For non-normal dynamics this sharpens into a geometric statement: the **left and right singular charts differ** — the coordinate $z$ is encoded in is not the coordinate $z'$ is read in — and a single nonnegative **irreversibility gap** $\Delta=\sum_i\sigma_i(T)-\sum_i\lambda_i(S)\ge 0$ measures the predictable structure the symmetric objective forfeits, with two faces (a phase deficit for a normal current, a chart split for a non-normal one) and vanishing exactly at detailed balance. The predictor network of BYOL/SimSiam/I-JEPA is, on this reading, not merely an anti-collapse device but the minimal mechanism that lifts the objective out of the irreversible blind spot. Finally the result is **topological**: a stationary current is a divergence-free flow, so the irreversible structure a symmetric objective discards has, on a graph, exactly $\beta_1=E-V+1$ independent modes (the *cycle rank*), and in the continuum exactly $\dim H^1(M)=b_1(M)$ — the first Betti number of the state manifold, the number of *harmonic* currents that circulate its holes. The discrete count refines to the continuum one when the mesh is filled (a $3\times3$ torus's $\beta_1=10$ collapses to $b_1(T^2)=2$); for a learned representation $b_1(M)$ counts the periodic latent factors that carry a real arrow of time. Every claim is settled by deterministic, exact-arithmetic certificates — E9 on the drift ring (the arrow as imaginary spectrum), E10 on the non-normal conveyor (the arrow as a $60^\circ$ chart split), and E12 across graphs of growing cycle rank and their Hodge Betti numbers (the arrow's dimension as $\beta_1$, refining to $b_1$) — all showing the blind spot switch off precisely at detailed balance. Two further certificates address practice: E13 finds the arrow's *strength* cheaply and robustly estimable from finite samples while its *dimension* $b_1$ is an offline topological audit (exact persistence times out; single-scale estimates are wrong), E14 finds the reversibility axis and the parent's cumulant axis **orthogonal** — independently dialable, jointly auditable; E15 makes the topological audit efficient — $b_1$ as the nullity of the sparse Hodge $1$-Laplacian, exact and scaling to thousands of simplices sub-second; and E16 closes the loop, recovering $b_1$ from raw point clouds by persistent homology and feeding E15's engine, so the audit runs end-to-end.

**Keywords:** self-supervised learning, transition operator, detailed balance, time reversal, probability current, predictor networks, slow feature analysis, arrow of time, identifiability, singular value decomposition, Hodge/Helmholtz decomposition, graph homology, Betti number.

---

## 1. Two axes of one blindness {#sec:intro}

The conductor blind spot is a statement on the **cumulant axis**: on a categorical head with cyclic structure, the order-two Fisher form is constitutionally unable to carry the order-three Amari–Chentsov cubic. The parent paper carried that boundary into continuous representation learning: the whitened-alignment optimum recovers the slow-eigenfunction chart of a world, and a *linear* probe of it is complete only on the Gaussian (second-order-sufficient) stratum. Both statements are about the *order* at which a second-order class stops sufficing.

There is a second axis, orthogonal to the first, and the parent paper named it only to defer it: its §8 lists **reversibility** (detailed balance) as the structural hypothesis that "buys the self-adjoint spectral theorem," and leaves "genuinely non-reversible additive noise" to follow-up. This note is that follow-up, and the deferral turns out to hide a blind spot exactly as sharp as the first.

The object is the same cycle. Take the conductor blind spot's $\mathbb{Z}/n$ and give it a **net drift**: a walk that steps forward with probability $q$ and backward with probability $b\neq q$. Detailed balance is broken; the walk carries a **probability current**, a preferred direction — an arrow of time. We ask the question the parent paper asks of every world: *what does the aligned, whitened encoder recover?* The answer is that it recovers everything the parent paper says it does — and is **exactly blind to the arrow**. The reason is a one-line fact about symmetric forms, and the cure is a specific, familiar piece of architecture.

### 1.1 The smallest irreversible world

On $\mathbb{Z}/n$ let the transition operator be
$$(T\psi)(k) \;=\; q\,\psi(k{+}1) + b\,\psi(k{-}1) + r\,\psi(k), \qquad r = 1-q-b,$$
with the uniform law stationary (it is doubly stochastic). In $L^2$ of the uniform law the adjoint is the matrix transpose, so $T$ splits as
$$T = S + A, \qquad S=\tfrac12(T+T^{*}) \;(\text{forward}=\text{backward}=\tfrac{q+b}{2}), \qquad A=\tfrac12(T-T^{*}) \;(\text{the current}).$$
$S$ is a plain lazy random walk on the ring — no arrow. $A\psi(k)=\tfrac{q-b}{2}\big(\psi(k{+}1)-\psi(k{-}1)\big)$ is a centred difference, the discrete derivative, **nonzero exactly when $q\neq b$**. The slow eigenvalue is complex,
$$\lambda_1 = \underbrace{r+(q+b)\cos\tfrac{2\pi}{n}}_{\alpha\,=\,\mathrm{Re}\,\lambda_1} \;+\; i\,\underbrace{(q-b)\sin\tfrac{2\pi}{n}}_{\beta\,=\,\mathrm{Im}\,\lambda_1},$$
and the entire arrow of time is the single number $\beta$, whose sign is the drift direction. Hold this picture: **the real part $\alpha$ is what the standard objective will see; the imaginary part $\beta$ is what it will throw away.**

---

## 2. The Irreversible Blind Spot {#sec:theorem}

We use the parent paper's reduction verbatim. With $\{h_i\}$ whitened and $T$ the transition operator, stationarity gives $\mathbb{E}[h_i(z')h_i(z)] = \langle h_i, T h_i\rangle$, so (parent, Appendix A.1)
$$\mathbb{E}\big\|h(z')-h(z)\big\|^2 \;=\; \sum_i 2\big(1-\langle h_i, T h_i\rangle\big). \tag{2.1}$$

**Lemma 1 (skew forms vanish).** *For the $L^2(p)$ adjoint, $A=\tfrac12(T-T^{*})$ is skew-adjoint ($A^{*}=-A$), so for every real $h\in L^2(p)$,* $\langle h, A h\rangle = 0.$
*Proof.* $\langle h,Ah\rangle = \langle A^{*}h, h\rangle = -\langle Ah,h\rangle = -\langle h,Ah\rangle$, and a real number equal to its own negative is $0$. $\square$

**Theorem 1 (Irreversible Blind Spot).** *The single-encoder whitened-alignment objective (2.1) is a functional of the symmetric part $S$ alone:*
$$\mathbb{E}\big\|h(z')-h(z)\big\|^2 \;=\; \sum_i 2\big(1-\langle h_i, S\,h_i\rangle\big). \tag{2.2}$$
*Consequently:*

1. *(**Time-reversal degeneracy**.) The map $T\mapsto T^{*}$ (reverse the pair, flip the arrow) leaves $S$ fixed, hence leaves the objective unchanged as a functional of $h$. A world and its time-reverse have a **bit-for-bit identical loss landscape**: every encoder receives the same loss, so every minimiser — and the entire recovered representation — coincides.*

2. *(**The arrow is in the kernel**.) $\partial/\partial A$ of the objective is identically zero: no perturbation of the antisymmetric part changes any value of the loss. Since $A\neq 0$ exactly when detailed balance fails, and $A$ is the generator of the probability current, the irreversible content — the arrow of time — is unrecoverable by any single-encoder representation trained this way.*

*Proof.* (2.2) is (2.1) with $\langle h_i,Th_i\rangle=\langle h_i,Sh_i\rangle$ by Lemma 1. Both consequences are immediate: $S$ is the symmetric part of both $T$ and $T^{*}$, and the right-hand side of (2.2) does not contain $A$. $\square$

**Corollary 1 (what is recovered instead).** *Under a slow-feature gap on $S$, the minimiser is $h=U\,\Psi_1$ with $U$ orthogonal and $\Psi_1$ the top eigenspace of $S$ — the slow chart of the **reversibilised** dynamics $\tfrac12(T+T^{*})$, the closest reversible world to $T$. The encoder recovers the world with its arrow erased.* (This is exactly the parent paper's Theorem 1, with $S$ in place of $T$; everything the parent proves about the chart's curvature and the Gaussian boundary applies to $S$ unchanged. The cumulant axis and the reversibility axis are independent: $S$ can still be non-Gaussian, so the chart can still be curved — the two blind spots compose.)

This is the conductor blind spot's structure on a new axis. There, an order-two instrument (the Fisher form) was blind to an order-three fact (the cubic). Here, a symmetric instrument (a quadratic form in a single encoder) is blind to an antisymmetric fact (the current). Both are second-order classes failing to carry a piece of the world's geometry that is, by a parity/symmetry argument, simply not in their range.

---

## 3. The resolution is a predictor {#sec:predictor}

The blind spot is a property of the *form* $\langle h, Th\rangle$, which symmetrises $T$ before the encoder ever sees it. To recover $A$ the objective must probe $T$ *off the diagonal* — it must evaluate $\langle f, T g\rangle$ for $f\neq g$. That is precisely what a **predictive** objective does.

Fix a whitened embedding $h$ and introduce a learnable linear **predictor** $P$ acting in embedding space, trained to predict the partner's embedding:
$$P^{\star} \;=\; \arg\min_{P}\ \mathbb{E}\big\|P\,h(z) - h(z')\big\|^2 \;=\; \mathbb{E}\big[h(z')\,h(z)^{\!\top}\big] \quad(\text{whitened }h),$$
whose entries are $P^{\star}_{ab}=\mathbb{E}[h_a(z')h_b(z)] = \langle h_b, T h_a\rangle$. So $P^{\star}$ is the **matrix of the operator $T$ itself** in the embedding basis — not its quadratic form.

**Theorem 2 (the predictor recovers the arrow).** *Restricted to the slow subspace with orthonormal basis $\{\psi_a\}$, the predictive optimum equals the operator block $B_{ab}=\langle\psi_a, T\psi_b\rangle$. Its symmetric part $\tfrac12(B+B^{\top})$ is exactly what the single-encoder objective (2.2) can see; its antisymmetric part*
$$\tfrac12\big(B-B^{\top}\big)_{ab} \;=\; \langle \psi_a, A\,\psi_b\rangle$$
*is a faithful readout of the current. Under time reversal $B\mapsto B^{\top}$. In particular a strictly asymmetric optimal predictor, $P^{\star}\neq (P^{\star})^{\top}$, exists if and only if detailed balance fails.*

*Proof.* $P^{\star}$ has entries $\langle\psi_b,T\psi_a\rangle$ by the normal-equations computation above; restricting to the slow subspace gives $B$. Splitting $T=S+A$ and using $S^{*}=S$, $A^{*}=-A$ gives $\tfrac12(B+B^{\top})_{ab}=\langle\psi_a,S\psi_b\rangle$ and $\tfrac12(B-B^{\top})_{ab}=\langle\psi_a,A\psi_b\rangle$. Time reversal sends $A\mapsto -A$, hence $B\mapsto B^{\top}$. The asymmetry is nonzero iff $A\neq 0$ iff detailed balance fails. $\square$

**Corollary 2 (the architecture the theorem explains).** *A symmetric Siamese objective — shared encoder, covariance/whitening anti-collapse, no predictor — is provably arrow-blind (Theorem 1). The predictor head of BYOL, the stop-gradient asymmetry of SimSiam, and the predictor of I-JEPA are the minimal devices that make the objective sensitive to $\langle f,Tg\rangle$ with $f\neq g$, lifting it out of the irreversible blind spot. On temporal and causal data — where the transition is genuinely irreversible — this predicts that predictive architectures capture structure that symmetric contrastive ones cannot, and locates the reason in the antisymmetric part of one operator.* We state this as the structural reading the theorem licenses; a controlled head-to-head on irreversible worlds is the natural empirical test.

### 3.1 The ring, worked

On the drift ring of §1.1, the slow subspace is spanned by the $L^2$-orthonormal pair $\psi_1=\sqrt2\cos\frac{2\pi k}{n}$, $\psi_2=\sqrt2\sin\frac{2\pi k}{n}$, and a one-line computation gives the predictor block in closed form,
$$B \;=\; \begin{pmatrix}\alpha & \beta\\[2pt] -\beta & \alpha\end{pmatrix}, \qquad \alpha=r+(q+b)\cos\tfrac{2\pi}{n}, \qquad \beta=(q-b)\sin\tfrac{2\pi}{n}.$$
The symmetric part is $\alpha I$ — an isotropic scaling, all the single encoder can read, and identical under $q\leftrightarrow b$. The antisymmetric part is $\beta J$ (with $J=\big(\begin{smallmatrix}0&1\\-1&0\end{smallmatrix}\big)$) — a rotation generator whose sign is the drift. The predictor turns the static recovered ring into a **rotational flow**, and the direction of that rotation *is* the arrow of time; reversing the drift transposes $B$ and spins the flow the other way. At detailed balance $q=b$ we have $\beta=0$, $B=\alpha I$ is symmetric, predictor and single encoder agree, and there is no arrow to miss.

### 3.2 The general (non-normal) form: two charts, and the irreversibility gap

The ring is *normal* ($TT^{*}=T^{*}T$), so its arrow is carried by the imaginary spectrum and the predictor block is a clean rotation inside a **single** recovered chart. The general statement drops normality. Let $T=\sum_k\sigma_k\,u_k\langle v_k,\cdot\rangle_p$ be the singular value decomposition of $T$ on the mean-zero subspace of $L^2(p)$: $T v_k=\sigma_k u_k$, with $\{v_k\}$ the **right** (input) chart, $\{u_k\}$ the **left** (output) chart, both $L^2(p)$-orthonormal.

**Theorem 3 (general recovery is the SVD; two charts).** *The two-encoder objective $\min_{f,g}\,\mathbb{E}\|f(z')-g(z)\|^2$ over whitened $f,g$ has optimum value $2\big(d-\sum_{i\le d}\sigma_i\big)$, attained at $g_i=u_i$, $f_i=v_i$. The input and output charts span the same subspace if and only if $T$ is normal; for a non-normal $T$ they differ, and the principal angle between $\mathrm{span}\{v_i\}$ and $\mathrm{span}\{u_i\}$ is a coordinate-free measure of irreversibility. Time reversal $T\mapsto T^{*}$ exchanges the two charts.*

*Proof.* With $f,g$ whitened, $\mathbb{E}\|f(z')-g(z)\|^2=\sum_i 2\big(1-\langle g_i,Tf_i\rangle\big)$, and maximising the bilinear form $\sum_i\langle g_i,Tf_i\rangle$ over orthonormal $\{f_i\},\{g_i\}$ gives $\sum_{i\le d}\sigma_i$ at the singular functions (the Eckart–Young / Ky Fan principle for singular values). The left and right singular subspaces coincide iff $T$ is normal; and $T^{*}=\sum_k\sigma_k\,v_k\langle u_k,\cdot\rangle_p$ swaps $u\leftrightarrow v$. $\square$

So irreversibility is not one phenomenon but a **chart split**: the coordinate in which $z$ is most predictable is not the coordinate in which $z'$ is read. A single shared encoder, forced to use $f=g$, cannot represent the split — it collapses to the symmetric part (Theorem 1). The two failures, the cumulant axis (parent) and the reversibility axis (here), are independent, and the reversibility axis itself now has two faces: a phase (normal) and a chart split (non-normal).

Both faces are measured by one nonnegative scalar. The single encoder captures $\sum_{i\le d}\lambda_i(S)$ (Theorem 1 with Ky Fan on $S$); the two encoder captures $\sum_{i\le d}\sigma_i(T)$; and by the Fan–Hoffman inequality $\lambda_i(S)\le\sigma_i(T)$, so

> **The irreversibility gap.** $\displaystyle \Delta \;:=\; \sum_{i\le d}\sigma_i(T)-\sum_{i\le d}\lambda_i(S)\;\ge\;0$ *is the predictable correlation the single encoder leaves on the table.* $\Delta=0$ *if and only if $T$ is self-adjoint on the top-$d$ subspace (detailed balance there). $\Delta>0$ arises two ways: a* **normal current** *($\sigma_i=\lvert\lambda_i\rvert>\mathrm{Re}\,\lambda_i=\lambda_i(S)$, the §2 imaginary-spectrum mechanism) or* **non-normality** *(the chart split, this section). One gap, two mechanisms.*

On the drift ring $\Delta$ is the phase deficit $\sum(\lvert\lambda_i\rvert-\mathrm{Re}\,\lambda_i)>0$; on a conveyor with position-dependent rates it is the chart-split deficit. Both, and the exact collapse $\Delta=0$ at detailed balance, are certified in E10.

### 3.3 The topology of the blind spot: $\dim(\text{arrow})=$ cycle rank

The drift ring carries a *single* current (one $\beta$). A general world carries a whole space of them, and its dimension is a topological invariant of the transition graph. This is the deepest form of the result, and it is exact.

Let $T$ act on a connected graph $G=(V,E)$ with stationary law $\pi$. The **stationary current** $J_{ij}=\pi_i T_{ij}-\pi_j T_{ji}$ is antisymmetric and, crucially, **divergence-free**: by stationarity $\sum_j J_{ij}=\pi_i\sum_j T_{ij}-\sum_j\pi_j T_{ji}=\pi_i-(\pi^{\top}T)_i=0$ (Kirchhoff's current law). The divergence-free edge flows on $G$ are exactly the kernel of the boundary operator $\partial_1:\mathbb{R}^E\to\mathbb{R}^V$ — the **cycle space** — of dimension the first Betti number
$$\beta_1(G)=\dim\ker\partial_1 = E-V+1.$$

**Theorem 4 (topological dimension of the irreversible blind spot).** *The single-encoder objective is invariant under any change of the current $J$ (Theorem 1: it is a function of $S$ alone), whereas the predictor recovers $J$ in full (Theorem 3). Hence the dimension of the irreversible content invisible to a symmetric objective — the space of currents sharing one reversible backbone $S$ and one stationary law $\pi$ — equals the cycle rank*
$$\boxed{\ \dim(\text{irreversible blind spot}) \;=\; \beta_1(G) \;=\; E-V+1.\ }$$
*Detailed balance is the origin $J=0$; the drift ring ($\beta_1=1$) is the minimal nonzero case.*

*Proof.* $J$ is antisymmetric by construction and divergence-free as shown, so $J\in\ker\partial_1$, of dimension $E-\mathrm{rank}\,\partial_1=E-(V-1)=\beta_1$ for connected $G$. Adding a **circulation** — a skew, divergence-free edge flow — to $T$ changes only $A=\tfrac12(T-T^{*})$, leaving $S$ and $\pi$ fixed, so the entire $\beta_1$-parameter family of worlds shares one single-encoder optimum (Theorem 1). The predictor recovers the operator $T$, hence $A$, hence $J$ (Theorem 3), so all $\beta_1$ directions are recoverable. $\square$

The arrow of time is therefore not a scalar but a **homology class count**: the irreversible structure a symmetric objective discards has exactly as many independent modes as the transition graph has independent cycles. A tree ($\beta_1=0$) carries no recoverable arrow; a single loop carries one (E9); a richly connected world — a $3\times3$ torus *as a graph* has $\beta_1=10$ — hides a multidimensional current behind one reversible backbone. Connectivity is thus a design knob for irreversible expressivity: capturing many independent modes of temporal/causal structure requires a transition graph of high cycle rank, and a predictor to read them.

### 3.4 The continuum: the harmonic arrow ($\dim = \dim H^1$)

Theorem 4 is combinatorial. Its continuum shadow is cleaner and is a genuine topological invariant of the *state space*, not of a chosen mesh. Let the state space be a closed oriented Riemannian manifold $M$ (or $\mathbb{R}^n$ with a confining potential and decay at infinity), with a reversible diffusion backbone of stationary density $p$ and an added stationary current. The probability current $J = bp - D\nabla p$ is a vector field, and stationarity is the continuity equation $\nabla\!\cdot\! J = 0$; via the metric, $J$ is a **co-closed** $1$-form, $\delta J = 0$. The Hodge decomposition $\Omega^1 = \mathrm{im}\,d \oplus \mathrm{im}\,\delta \oplus \mathcal H^1$ then forces

$$J \;=\; \underbrace{\delta\beta}_{\text{co-exact eddy}} \;+\; \underbrace{h}_{\text{harmonic}}, \qquad h\in\mathcal H^1(M)\cong H^1_{\mathrm{dR}}(M),$$

with no exact part (on a closed manifold $\delta J=0$ kills it).

**Theorem 5 (continuum: the harmonic arrow).** *Under the hypotheses above: (i) the single-encoder objective is blind to the entire current $J$ — Theorem 1 is metric/operator-level and uses no graph structure, only the $L^2(p)$ self-adjoint/skew splitting of the generator; (ii) the co-exact part $\delta\beta$ is locally a curl (a contractible eddy, removable by a local change of dynamics), while the harmonic part $h$ is the obstruction no local change removes — the current forced to circulate the holes of $M$; (iii) the harmonic currents form $\mathcal H^1(M)\cong H^1_{\mathrm{dR}}(M)$, of dimension the first Betti number $b_1(M)$. This $b_1(M)$ is the topologically protected, mesh-independent dimension of the arrow, and Theorem 4 is its $1$-complex degeneration: a graph has no $2$-cells, so every divergence-free flow is harmonic and the blind spot has dimension $\beta_1(G)=E-V+1$; filling the $2$-cells sends the contractible cycles into the $\delta\beta$ eddies and leaves exactly $b_1(M)$ harmonic modes.*

*Proof sketch.* (i) The identity $\langle h, Th\rangle=\langle h, Sh\rangle$ of Theorem 1 is the statement that the skew part of the generator contributes zero to every real quadratic form; it holds verbatim on $M$. (ii)–(iii) $\nabla\!\cdot\!J=0\iff\delta J=0$; a co-closed form has $J=\delta\beta+h$ (the exact part vanishes since $\delta d\alpha=0\Rightarrow d\alpha=0$ on closed $M$); the harmonic projection is the unique harmonic representative, and the Hodge theorem gives $\mathcal H^1(M)\cong H^1_{\mathrm{dR}}(M)$, $\dim=b_1(M)$. The graph case: a $1$-complex has trivial co-exact part, so $\ker\delta$ (divergence-free flows) $=\mathcal H^1$, dimension $\beta_1$; the boundary $d_2$ of added $2$-cells enlarges $\mathrm{im}\,\delta$ and shrinks $\mathcal H^1$ to $b_1$ of the complex. $\square$

The refinement is a *correction*, not a technicality: a discrete model's cycle rank $E-V+1$ **over-counts** the arrow, because most of its cycles are discretisation artifacts (the little faces of the mesh) that a finer model fills in. E12 makes this exact — the $3\times3$ torus graph's $\beta_1=10$ collapses to $b_1(T^2)=2$ once its square faces are filled, and the tetrahedral $K_4$ ($\beta_1=3$) collapses to $b_1(S^2)=0$ (a sphere has no holes; every current on it is a removable eddy). The invariant that survives *every* discretisation — the true count of independent irreversible modes — is $b_1(M)$. For a learned representation, $M$ is the low-dimensional data manifold the encoder discovers, and $b_1(M)$ counts the **periodic/cyclic latent factors** — rotation, phase, time-of-day, gait, conversational turn-taking — that carry a genuine arrow of time and that a symmetric objective is, by Theorem 1, constitutionally unable to see.

**Remark (estimating the harmonic arrow without meshing).** Theorem 5 appears to demand an expensive object — a Hodge decomposition of a high-dimensional space — but in practice none is built, for three reasons. *(1) The chart is already low-dimensional.* The decomposition is read in the learned embedding $\mathbb{R}^d$, not the ambient data space; the encoder has done the dimensionality reduction, and $b_1$ of the slow data manifold is small. One never meshes the ambient space (which would incur the curse of dimensionality and defeat any persistent-homology approach). *(2) The predictor already estimates the current.* The two-encoder optimum $P^\star=\mathbb E[f(z')f(z)^\top]$ (whitened) is the operator in the embedding basis; its antisymmetric part $\tfrac12(P-P^\top)$ is the discretised current — a $d\times d$ matrix assembled from the same batch cross-covariance a predictor/VICReg objective already forms. There is no separate Hodge solve in the gradient loop; the current is a by-product of training. *(3) The harmonic dimension is a spectral readout, not a simplicial one.* Each harmonic current appears as a conjugate pair of complex eigenvalues of $P$ whose eigenfunctions wind; the count of *persistent* such pairs — stable under coarsening the pair-lag $\tau\to2\tau$ and under encoder perturbation — estimates the active $b_1$. That is an $O(d^3)$ eigendecomposition at evaluation time and an $O(d^2)$ assembly folded into the existing covariance computation, both negligible against the forward/backward pass; harmonic-versus-eddy is decided by this coarsening-persistence test rather than by constructing a complex. The honest limitation: this is a spectral *estimator* of the active harmonic dimension — exact only in the finite-graph case (E12), and in the continuum a cheap, graceful lower bound that can under-count modes too slow or too mixed to separate.

---

## 4. Certificates (E9, E10, E12, E13, E14, E15, E16) {#sec:certificate}

Every claim is checked by a deterministic, exact, CPU-millisecond program, [`empirical/apex_recovery/e9_irreversible_ring.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e9_irreversible_ring.py), wired into the suite's pass/fail gate ([`run_all.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/run_all.py)). There is no machine learning in it: the theory collapses both objectives to operator algebra on a circulant matrix, so the certificate is exact up to LAPACK $\varepsilon$. On the drift ring $\mathbb{Z}/12$ ($q=0.5$, $b=0.2$) it certifies, to machine precision:

| claim | quantity | measured |
|---|---|---|
| blind-spot identity (2.2) | $\max_h\lvert\langle h,Th\rangle-\langle h,Sh\rangle\rvert$ over 202 encoders | $3.3\times10^{-16}$ |
| skew form vanishes (Lemma 1) | $\max_h\lvert\langle h,Ah\rangle\rvert$ | $2.1\times10^{-17}$ |
| time-reversal degeneracy (Thm 1.1) | $\lVert S_{\text{fwd}}-S_{\text{rev}}\rVert_F$ / loss gap | $1.9\times10^{-16}$ / $8.9\times10^{-16}$ |
| the worlds genuinely differ | $\lVert T_{\text{fwd}}-T_{\text{rev}}\rVert_F = 2\lVert A\rVert_F$ | $1.47$ |
| arrow in the spectrum | closed-form $\lambda_1$ vs numeric | $4.5\times10^{-16}$ |
| predictor block (Thm 2) | $\lVert B-[[\alpha,\beta],[-\beta,\alpha]]\rVert_F$ | $1.1\times10^{-16}$ |
| predictor separates the arrow | $\lVert B_{\text{fwd}}-B_{\text{rev}}\rVert_F$ | $0.42 \;(>0)$ |
| **control: blind spot OFF at $q=b$** | $\lVert A\rVert_F$, $\beta$, antisym$(B)$ | $0,\,0,\,0$ (exact) |

A sweep at fixed total rate $q+b$ confirms the picture in one image (the certificate's figure): the single-encoder objective is **flat** across all drift asymmetries while the arrow $\lvert\beta\rvert$ rises linearly, and the predictor's flow on the ring embedding spins forward or backward with the current. The same identity is verified on the conductor-blind-spot moduli $n\in\{6,8,12,18,30\}$ — it is literally the CBS ring with a current added.

**The general non-normal form (E10)** — [`e10_nonnormal_svd.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e10_nonnormal_svd.py), also in the gate — certifies §3.2 on a **conveyor**: a $\mathbb{Z}/n$ walk with position-dependent forward rates $a_k$ (stationary law $\pi_k\propto 1/a_k$ in closed form), the maximally irreversible chain. It is genuinely non-normal ($\lVert TT^{*}-T^{*}T\rVert=0.97$), Theorem 1's identity still holds exactly ($4\times10^{-16}$), and the recovered input and output charts **differ by up to $60^{\circ}$** (principal angles of the top singular functions), with the two-encoder optimum matching $\sum\sigma_i$ to $10^{-15}$ and beating every random whitened pair. The irreversibility gap is $\Delta=0.40$ on the conveyor and $\Delta=0.025$ on the normal drift ring (the phase deficit) — and **exactly $0$** at detailed balance, where the chart split and the arrow vanish together. Time reversal swaps the left and right charts to $10^{-6}$ degrees. The contrast is the paper in one certificate: a current makes $\Delta>0$; non-normality additionally splits the charts.

**The topology (E12)** — [`e12_topological_blindspot.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e12_topological_blindspot.py), also in the gate — certifies Theorem 4 on four graphs of increasing cycle rank: a ring ($\beta_1=1$), a theta graph ($\beta_1=2$), the complete graph $K_4$ ($\beta_1=3$), and a $3\times3$ torus ($\beta_1=10$). For each it confirms $\beta_1$ three independent ways ($E-V+1$, the nullity of the boundary operator, and the number of fundamental cycles, all agreeing), builds the $\beta_1$ circulations, and verifies that across the resulting $\beta_1$-parameter family the single-encoder objective is **exactly invariant** ($0$ to machine precision) while the predictor recovers the current to $10^{-17}$, that the antisymmetric part lives **entirely** in the cycle space (residual $\sim10^{-17}$), and that the recovered current dimension **equals $\beta_1$** in every case. The current is divergence-free (Kirchhoff residual $\sim10^{-17}$), the gap $\Delta$ grows with it, and the detailed-balance control gives $\Delta=0$ exactly. The dimension of the arrow is the cycle rank, measured. Finally E12 computes the **continuum refinement** of Theorem 5 by the Hodge $1$-Laplacian $L_1=d_1^\top d_1+d_2 d_2^\top$: filling the torus's square faces collapses $\beta_1=10$ to $b_1(T^2)=2$, and filling the tetrahedral $K_4$ collapses $\beta_1=3$ to $b_1(S^2)=0$ (each a valid complex, $d_1 d_2=0$ exactly) — the mesh-dependent eddies fall away and only the topological modes remain.

**Estimation in practice (E13)** — [`e13_estimating_the_arrow.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e13_estimating_the_arrow.py) — backs the remark above with numbers, and draws the practical line cleanly. The arrow's *strength* $\Delta$ is estimable from finite sampled pairs and converges to the exact operator value (relative error $2.6\%\!\to\!0.2\%$ as the sample grows from $10^3$ to $10^5$ pairs, variance shrinking as $1/\sqrt m$), at $O(d^3)$ eval-time cost — the in-loop-friendly signal, free from the predictor. Its *dimension* $b_1$ is **not** loop-cheap: a single-scale geometric (kNN flag-complex) estimate is wrong and scale-dependent (the torus's $b_1$ is read as $47,19,13$ for $k=6,9,12$ against the true $2$; the sphere's as $31,19,9$ against $0$), and exact Vietoris–Rips persistence times out on a few hundred points offline. Topology is therefore an **offline, sub-sampled audit**, never an in-training computation — exactly as the overhead worry anticipates, and exactly why $\Delta$ (strength), not $b_1$ (count), is the quantity to track during learning.

**The two axes are orthogonal (E14)** — [`e14_two_axes.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e14_two_axes.py) — composes this paper's reversibility axis with the parent's cumulant axis on one product world (a warped Gaussian factor $U$ and a drift ring $V$). Sweeping non-Gaussianity $\gamma$ and irreversibility $\rho$, the linear-probe curvature $\nu_D$ depends on $\gamma$ alone ($0\!\to\!0.27$) and the irreversibility gap $\Delta$ on $\rho$ alone ($0\!\to\!0.02$), with **zero cross-talk to machine precision** — so all four corners of (Gaussian/non-Gaussian)$\times$(reversible/irreversible) are realised independently. The arrow lives in the single product operator (its $L^2(\pi)$-antisymmetric part grows with $\rho$ and vanishes exactly at $\rho=0$). A representation can be defeated on either axis or both; an audit that checks only linearity misses the arrow, and one that checks only reversibility misses the curved chart.

**The offline audit, made efficient (E15)** — [`e15_efficient_betti.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e15_efficient_betti.py) — turns Theorem 5's $b_1$ from a prohibitive computation into a cheap one. The key is to read $b_1=\dim\ker L_1$ as the count of **near-zero eigenvalues of the sparse Hodge $1$-Laplacian**, obtained by shift-invert Lanczos that touches only the bottom of the spectrum — $O(\text{simplices}\times b_1)$ work, never the $O(E^3)$ dense boundary-rank E12 used. On complexes with known topology the nullity is *exact*: triangulated tori ($b_1=2$) scale to thousands of simplices in well under a second (where dense rank is hopeless), a triangulated sphere gives $b_1=0$ with no false positives, two disjoint tori give $b_1=4$. The honest residual is the *faithful sparse complex* from a raw point cloud: a single-scale flag complex has no good scale — a sampling gap breaks a loop (so a sampled circle reads $b_1=0$ at most $k$), larger $k$ fills it, and a surface grows phantom loops (a Clifford-torus sample over-counts) — so the complex wants a persistence/alpha library on a sub-sample of the encoder embedding, after which this engine reads $b_1$ cheaply. The engine is solved here; E16 supplies the complex.

**Closing the loop: the faithful complex (E16)** — [`e16_pointcloud_betti.py`](https://github.com/leomurillo/AI-ConductorBlindSpot/blob/main/empirical/apex_recovery/e16_pointcloud_betti.py) — turns a raw point cloud into the right $b_1$ via **persistent homology** (GUDHI), the proper multi-scale answer to the scale problem E15 exposes. From samples of a circle, a flat Clifford torus, a sphere, and three circles it recovers $b_1=1,2,0,3$ **exactly**, robustly (stable across sample size and additive noise $\sigma\approx0.05$), at C++ speed (sub-second to $\sim$2.6 s). The count uses a data-driven rule with no hand-tuned scale: $b_1$ is the number of $H_1$ bars clearing a dominance floor (a fixed fraction of the diameter), *confirmed* by a persistence-stability gap across that floor — one rule, four honest outcomes (all-noise $\to 0$, as the sphere correctly reads; all-signal $\to$ count; gap-confirmed $\to$ count; ambiguous $\to 0$). It composes exactly with E15: the faithful $2$-skeleton at the persistence-chosen scale, handed to E15's sparse-Hodge `hodge_b1`, returns the same $b_1=2$ on the Clifford torus. And it reconfirms the embedding lesson — the curved $3$-D donut at matched $N$ reads $0$ because its uneven density raises the noise floor and collapses the gap, whereas the flat (encoder/diffusion) embedding is uniformly sampled and reads $2$. (E16 needs a TDA library and so, like the trained certificates, is runnable but outside the dependency-light deterministic gate.)

---

## 5. Scope, and what comes next {#sec:scope}

The theorem is exact and assumption-light: it needs only a stationary pair and the $L^2(p)$ adjoint, and it holds for *every* encoder, not merely the optimum. Three honest boundaries. (i) **Normal, non-normal, topological, and continuum — settled (Theorems 3–5, E10/E12).** On the symmetric ring the operator is normal and the arrow is the *imaginary spectrum*; a non-normal transition separates the **left and right singular charts** (§3.2, E10); on a graph the arrow is a $\beta_1$-dimensional current space (§3.3, E12); and in the continuum it is the harmonic currents $H^1(M)$, of dimension $b_1(M)$ (§3.4, Theorem 5), to which the graph count refines as the mesh fills (torus $10\to2$). What remains genuinely open is **robust estimation from finite data**, and E13 maps it honestly: the arrow's *strength* $\Delta$ is cheap and robust to estimate (it converges from sampled pairs at $O(d^3)$ eval cost, free from the predictor), but its *dimension* $b_1$ resists in-loop computation — single-scale geometric estimates are wrong and scale-dependent, and exact persistent homology times out on a few hundred points — so $b_1$ belongs to an offline, sub-sampled audit — and that audit is now end-to-end: **E16** supplies the *faithful complex* from raw samples by persistent homology (GUDHI), recovering $b_1$ exactly and robustly at C++ speed, and **E15** reads the *count* as the sparse-Hodge nullity, exact and scaling sub-second. The two compose (E16's persistence-scale $2$-skeleton $\to$ E15's engine $\to$ the same $b_1$). What was the open practical frontier is closed for the cases studied; hardening it on large real encoders (sub-sampling strategy, the embedding's effective dimension, confidence from persistence stability) is the natural engineering follow-up. (ii) **Population and trained — confirmed (E11).** The theorems are about the objective and its optimum; E11 is the trained counterpart (the analogue of the parent's E8), and it confirms them. Trained off-host by SGD, the single-encoder objective is **reversal-blind** — the loss gap and the Procrustes distance between a world and its time-reverse are both $\approx 0.01$ (it converges to the same representation either way) — while the trained **predictor recovers the signed arrow**: $\lvert\hat\beta\rvert\approx 0.17$ against the ground-truth $0.15$, flipping sign ($+0.17\to-0.17$) under time reversal in a fixed gauge. SGD reaches the predictor optimum closely enough to read the current the single encoder cannot. (iii) **The two axes compose but are not the same.** Corollary 1 shows the reversibility blind spot sits *on top of* the cumulant blind spot: the recovered chart can be both arrow-erased (this paper) and curved-and-linearly-unreadable (parent). A representation can be defeated on either axis independently — measured directly in E14, where the curvature $\nu_D$ and the gap $\Delta$ vary with *zero cross-talk* across all four corners of (Gaussian/non-Gaussian)$\times$(reversible/irreversible) — so an audit must check both.

The lesson for practice is sharp and architectural. If the world has an arrow — and video, language, and physical causality all do — a symmetric self-supervised objective will recover a time-reversal-blind representation no matter how much data or capacity it is given, because the arrow is not in the objective's range. The fix is not scale; it is a predictor. The conductor blind spot taught that a second-order instrument cannot read an order-three fact; this is its sibling on the axis of time.

---

## References

[1] Murillo Montero, L. (2026). *Beyond the Conductor Blind Spot: Eigenfunction Identifiability and Planning in Non-Gaussian Worlds.* Companion preprint (this program).

[2] Murillo Montero, L. (2026). *The Conductor Blind Spot of the Quadratic Curvature Class on Ring-Structured Categorical Heads.* Zenodo preprint, DOI [10.5281/zenodo.20330864](https://doi.org/10.5281/zenodo.20330864).

[3] Fan, K. (1949). *On a theorem of Weyl concerning eigenvalues of linear transformations.* PNAS 35(11), 652–655.

[4] Rényi, A. (1959). *On measures of dependence.* Acta Mathematica Academiae Scientiarum Hungaricae 10, 441–451. (Maximal correlation; the SVD of the conditional-expectation operator.)

[5] Grill, J.-B., et al. (2020). *Bootstrap your own latent (BYOL).* NeurIPS 2020. (The predictor head.)

[6] Chen, X., & He, K. (2021). *Exploring simple Siamese representation learning (SimSiam).* CVPR 2021. (Predictor + stop-gradient asymmetry.)

[7] Assran, M., et al. (2023). *Self-supervised learning from images with a joint-embedding predictive architecture (I-JEPA).* CVPR 2023.

[8] Levin, D. A., & Peres, Y. (2017). *Markov Chains and Mixing Times* (2nd ed.). AMS. (Detailed balance, reversibility, the current.)

[9] Schnakenberg, J. (1976). *Network theory of microscopic and macroscopic behavior of master equation systems.* Reviews of Modern Physics, 48(4), 571–585. (Stationary cycle currents and nonequilibrium steady states on graphs.)

[10] Jiang, X., Lim, L.-H., Yao, Y., & Ye, Y. (2011). *Statistical ranking and combinatorial Hodge theory.* Mathematical Programming, 127(1), 203–244. (Hodge/Helmholtz decomposition of edge flows; the cycle space.)

[11] Schwarz, G. (1995). *Hodge Decomposition — A Method for Solving Boundary Value Problems.* Lecture Notes in Mathematics, Vol. 1607. Springer. (Hodge decomposition of $1$-forms; harmonic forms and $H^1_{\mathrm{dR}}$.)

[12] Qian, H. (2001). *Mathematical formalism for isothermal linear irreversibility.* Proceedings of the Royal Society A, 457(2011), 1645–1655. (Stationary probability current as the circulatory, detailed-balance-breaking part of a diffusion.)
