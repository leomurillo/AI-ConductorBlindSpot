# Poole-OTG Bridge Theorem

Draft 2 — 2026-06-02

*Draft 1 (ChatGPT 5.5) stated the five bridge theorems and sketched their proofs.
Draft 2 closes them: each proof is tightened to referee standard, and each is
settled by a deterministic, exact-arithmetic certificate (P1-P5) that also runs
the whole pipeline end to end on a concrete representative Poole rule. Nothing
below depends on a metaphor; the open work is named precisely and is Rooke's
team's to discharge.*

## Abstract

This note closes the first rigorous bridge from a finite Poole-style cellular
automaton to the program's quantum, leakage, curvature, and audit language. Every
claim is now both proved and certified in exact arithmetic.

The closed results are modest but real, and complete:

1. **(Theorem A)** Every fixed Poole rule has an exact open-quantum lift whose
   decoherent restriction is precisely the classical cellular automaton.
2. **(Theorem B / Corollary B1)** Every fixed Poole rule has an exact
   reversible/unitary dilation after adding an environment register; a same-space
   unitary on configurations alone exists if and only if the rule is injective.
3. **(Theorem C)** Every proposed reduced observable sector has an exact finite
   closure test. Failure of closure is not a metaphor: it is witnessed by two
   configurations that are observationally identical before evolution and
   observationally distinct after evolution.
4. **(Theorem D)** For empirical or noisy Poole transition data, the
   symmetric/current decomposition gives a finite certificate for blind-spot
   behavior, arrow, and cycle/topology content — now at arbitrary stationary law,
   not only at the uniform point.
5. **(Theorem E)** The finite bridge emits only relabelling-invariant
   (dimensionless) structural invariants; an SI-dimensionful output requires an
   external Archimedean completion. This is a checked property, not a disclaimer.

The note does **not** prove that Poole dynamics is quantum gravity. It proves the
correct structural place where such a claim would have to live: in a *nontrivial*
coherent lift, a *stable* boundary-leakage certificate, and *dimensionless*
refinement invariants. The first is the only one of the five that the bridge does
not hand you for free — and the note pins down exactly the floor it must rise
above.

All five theorems are certified by the suite `empirical/poole_bridge/` in the
companion repository (`github.com/leomurillo/AI-ConductorBlindSpot`); `run_all.py`
reproduces every number below in a few CPU-seconds with no GPU, no network, and no
machine learning. Section 7 is the certificate ledger.

---

## 1. Scope and conventions

Let a **Poole-style rule** mean a deterministic finite-radius Boolean update rule
on a finite torus. The exact threshold convention, prime-resonance convention,
boundary convention, and update schedule must be fixed before applying the
theorem. For a box `Lambda = (Z/Lx) x (Z/Ly) x (Z/Lz)`, write

```text
X = {0,1}^Lambda                     (configurations)
F : X -> X                           (the global synchronous update).
```

Once the local rule is fixed, `F` is a fixed finite map and everything below is
finite and exact. Theorems A, C, D, E hold for **any** finite `F`; Theorem B uses
only `X = {0,1}^N`. The three-dimensionality of the torus plays no role in the
mathematics — it is a parameter — so we state the theorems for a general finite
`F` and exhibit them on a concrete rule.

**The rule (Obligation 1 discharged).** As of this draft the *actual* OTG rule is
wired in and validated, not stood in for. Section 7A transcribes Rooke's canonical
**B5-7/S5-9 prime-resonance engine** (the `PooleEngine.step` of the Delta RPM
Protocol) into a torch-free `otg_rule.py`, reproduces his own canonical unit tests
bit-for-bit, and re-runs the closure and current certificates (P6) on it. The rule
is a 3-D outer-totalistic Boolean update on a **Moore-26** neighbourhood
(`3x3x3` shell, centre excluded), circular boundary, synchronous, with potential
`total_phi = m + sum_p alpha e^{-(m-p)^2/sigma^2}` and bands `5 <= total_phi <= 7`
(birth, dead cell) and `5 <= total_phi <= 9` (survival, live cell); the sharp
resonance (`sigma^2 = 0.01`) makes the *effective* integer rule exactly `B = {5,6}`,
`S = {5,6,7,8,9}` (the prime kick `7 -> 7.35` ejects `m=7` from birth).

A second, simpler **representative** rule (`PooleRule` in `poole_world.py`,
outer-totalistic with a hard prime input-shift) is retained for the rule-agnostic
demonstrations P1-P5 — it makes the theorems' independence from the rule visible,
and its specific witnesses in Section 4 belong to it. Both rules are objects one
line apart; every certificate re-runs unchanged on either.

**Epistemic tiers.** We mark every load-bearing statement: `[P]` proved and
certified in exact arithmetic; `[A]` argued/conditional; `[C]` conjectural or
contingent on the actual rule. The bridge theorems are `[P]`; the OTG-physics
claims they would support are `[C]` and named as such (Section 11).

---

## 2. Theorem A: Exact decoherent quantum-channel lift `[P]`

Let `X` be any finite configuration space, `F : X -> X` any deterministic update,
`H_X = C^X` with computational basis `{|x> : x in X}`. Define Kraus operators and
channel

```text
A_x = |F(x)><x|,          E_F(rho) = sum_x A_x rho A_x^†.
```

**Theorem A.** *(1) `E_F` is completely positive and trace preserving (CPTP).
(2) On diagonal states it is the classical Poole update: `E_F(D(p)) = D(F_# p)`,
and on a point mass `E_F(|x><x|) = |F(x)><F(x)|`. (3) Every coherence is
annihilated: `E_F(|u><v|) = 0` for `u != v`.*

**Proof.** *Complete positivity* is automatic for any operator-sum (Kraus) form,
by Choi's theorem; equivalently the Choi matrix
`J(E_F) = sum_{i,j} E_F(|i><j|) ⊗ |i><j| = sum_i |F(i)><F(i)| ⊗ |i><i|` is a sum
of mutually orthogonal rank-one projectors, hence positive semidefinite with
spectrum in `{0,1}`.

*Trace preservation:*

```text
sum_x A_x^† A_x = sum_x |x><F(x)|F(x)><x| = sum_x |x><x| = I.
```

*Diagonal action:* with `D(p) = sum_x p_x |x><x|`,

```text
E_F(D(p)) = sum_x p_x |F(x)><F(x)|,
```

so the coefficient of `|y><y|` is `sum_{x : F(x)=y} p_x`, exactly the pushforward
`(F_# p)(y)`. Taking `p` a point mass gives `E_F(|x><x|) = |F(x)><F(x)|`.

*Coherence annihilation:* for any `u, v`,

```text
E_F(|u><v|) = sum_x <x|u><v|x> |F(x)><F(x)| = delta_{uv} |F(u)><F(u)|,
```

which vanishes for `u != v`. QED.

### Corollary A1: the T7 classical-to-quantum bridge, and its warning `[P]`

The classical probability simplex on `X` is the decoherent (diagonal) face of the
density matrices on `H_X`; the Born/dephasing projection returns classical
probabilities, and on that face `E_F` is the Poole rule. So a Poole rule already
sits inside quantum formalism:

```text
density matrices  --dephase-->  probability simplex  --F_#-->  Poole update.
```

This closes the basic T7 gap: the classical rule is not *outside* quantum
language; it is an exact decoherent face of a quantum channel.

The corollary is equally a **warning, made quantitative by P1**. The channel above
is the *minimal* lift, and it destroys coherence completely: an equal
superposition over two configurations decoheres to a 50/50 classical mixture
(P1 measures the off-diagonal mass falling from `0.5` to `0`). It therefore
contains no new quantum physics by itself. **Any nontrivial OTG quantum claim must
specify a coherent channel, unitary, or Lindblad dynamics whose dephased
restriction recovers the Poole rule while the off-diagonal sector carries testable
additional structure.** Theorem A fixes the floor; the OTG content, if any, is the
height above it.

---

## 3. Theorem B: Exact reversible dilation `[P]`

Let `X = {0,1}^N` and `F : X -> X` any deterministic Boolean update. Add an
environment register with basis `X` and define

```text
U_F |x>|y> = |x>|y XOR F(x)>.
```

**Theorem B.** *`U_F` is unitary, `U_F|x>|0> = |x>|F(x)>`, and if the input is a
diagonal classical mixture and the first register is traced out, the second
register carries exactly the pushforward `F_# p` — the same classical face as
Theorem A.*

**Proof.** The map `(x,y) -> (x, y XOR F(x))` is an involution of `X x X` (apply it
twice: `y XOR F(x) XOR F(x) = y`), hence a bijection of the computational basis,
hence `U_F` extends to a unitary. Starting from `|x>|0>` gives `|x>|F(x)>`. For
`D(p) ⊗ |0><0|`, `U_F` produces `sum_x p_x |x><x| ⊗ |F(x)><F(x)|`; tracing out
register 1 leaves `sum_x p_x |F(x)><F(x)| = D(F_# p)`. QED.

### Corollary B1: irreversibility is an environment statement `[P]`

**Corollary B1.** *A same-space unitary `V` on `H_X` with `V|x> = |F(x)>` for all
`x` exists if and only if `F` is injective. When `F` is not injective the
obstruction is exact and witnessed: a collision `F(x) = F(x')`, `x != x'`, would
force `V` to send two orthonormal vectors to one, contradicting unitarity; the
target Gram matrix `G[i,j] = <F(i)|F(j)> = [F(i)=F(j)]` then has rank `|image F| <
|X|` and cannot equal the identity.*

Thus non-injective Poole dynamics — the generic case, since most cellular rules
contract — can be quantum-lifted exactly only as an open system (Theorem A) or as
a reversible system with an environment/history register (Theorem B), never as a
same-space unitary on configurations alone. For a finite-radius Boolean rule the
dilation is implemented by standard reversible Boolean circuitry with work
registers; exact reversibilisation is automatic, and the only design questions
(circuit depth, locality-preserving schedule) are Poole-specific engineering.

*Certified (P2):* the representative rule on the `2x2x1` torus is strongly
non-injective — its image has rank `4` of `16` — and P2 exhibits a collision
witness `F(0) = F(1)`, while a genuine permutation rule admits the same-space
unitary explicitly. This is the precise bridge from classical irreversible
dynamics to the open/decoherence story.

---

## 4. Theorem C: Exact closure and leakage test `[P]`

Let `Q : X -> Y` be any proposed retained observable sector (local neighbour
count, centre-plus-count, density, patch type, Betti data, current class, any
finite feature map).

**Theorem C.** *`Q` is closed under `F` — there exists `G : Y -> Y` with
`Q o F = G o Q` — if and only if*

```text
Q(x) = Q(x')   implies   Q(F(x)) = Q(F(x'))   for all x, x'.
```

*When the condition fails, there is an exact boundary-leakage witness: a pair
with `Q(x) = Q(x')` but `Q(F(x)) != Q(F(x'))`.*

**Proof.** If `G` exists and `Q(x)=Q(x')`, then `Q(F(x)) = G(Q(x)) = G(Q(x')) =
Q(F(x'))`. Conversely, suppose the implication holds. Define `G` on the image of
`Q` by choosing any `x` with `Q(x)=y` and setting `G(y) = Q(F(x))`; the
implication is exactly the statement that this is independent of the
representative, so `G` is well defined, and `Q o F = G o Q` by construction
(extend `G` arbitrarily off the image of `Q`). The contrapositive of the
implication is the witness. QED.

The content is that `Q` is closed iff the partition of `X` into `Q`-fibres is a
**congruence** for `F` — `F` descends to the quotient `X/Q`. The induced `G` is
the renormalised dynamics; the witness pair is the obstruction.

### Corollary C1: the T11/T13/T14 boundary dictionary, realised `[P]`

```text
closed Q-sector          = projection closure              (T11)
failure of closure       = boundary leakage                (T14)
minimal witness pair     = finite Noether-defect certificate
renormalised observable  = quotient dynamics G when closed (T13)
curvature-like source    = stable projection noncommutation under refinement
```

A Poole sector is **not** declared closed by interpretation; it is closed only
when the witness test has no counterexample. P3 runs the exact enumeration on the
representative rule (`3x3x1`, 9 cells, 512 states — and a genuinely 3-D `2x2x2`
box) and sorts the candidate sectors into the dictionary:

| sector `Q` | `|Y|` | verdict | structure |
|---|---|---|---|
| identity | 512 | **closed** | `G = F` (sanity) |
| attractor/basin id | 34 | **closed** | `G = id` (an `F`-invariant) |
| translation-orbit rep | 64 | **closed** | `G` = induced symmetry quotient |
| density (live count) | 10 | **leaks** (6 classes) | witness below |
| single fixed cell | 2 | **leaks** (2 classes) | local observable leaks |
| local window (cell+nbrs) | 32 | **leaks** (all 32) | overlapping neighbourhoods leak |

The closures are not accidents: the rule is **translation-equivariant** (P3
verifies `F(T_v s) = T_v F(s)` for every shift, exactly), so `F` descends to the
64 translation orbits — a genuine symmetry-renormalised dynamics — and every
basin is `F`-invariant by definition. The leaks are the substance. A fully decoded
density witness (both configurations have two live cells):

```text
x  = 100 / 100 / 000   --F-->   000 / 000 / 100    (1 live)
x' = 010 / 100 / 000   --F-->   100 / 010 / 000    (2 live)
```

Identical density before, different density after: the coarse sector leaks, and
the pair is the certificate. This is exactly Corollary C1's reading that a
totalistic *single-cell update* is closed on its retained local inputs, while
*coarse density*, *single-cell value*, and *patch/window evolution* generally leak
through spatial correlations and overlapping neighbourhoods. The closure theorem
turns each into an exact enumeration with a concrete witness.

*Honesty on scale.* Full enumeration is exact only up to a few million states; for
a large `3-D` torus the same test runs on sampled or symmetry-reduced
configurations, where **finding** a witness still proves non-closure but **not
finding** one does not prove closure. P3 reports the enumerable regime exactly and
flags the boundary.

---

## 5. Theorem D: Current/blind-spot audit for transition data `[P]`

Let `K` be a finite Markov transition matrix on a state set `Y`, and `pi` a
stationary law with positive support. Define the `L^2(pi)` adjoint, the
symmetric/antisymmetric split, and the stationary edge current

```text
K*(x,y) = pi_y K(y,x) / pi_x,
S = (K + K*)/2,   A = (K - K*)/2,   J(x,y) = pi_x K(x,y) - pi_y K(y,x).
```

**Theorem D.** *(1) `S` is `pi`-self-adjoint (the time-symmetric part) and `A` is
`pi`-skew (the antisymmetric current/arrow part). (2) Detailed balance holds iff
`A = 0` iff `J = 0`. (3) `J` is antisymmetric and has zero stationary divergence,
`sum_y J(x,y) = 0`.*

**Proof.** A direct computation gives `<f, K g>_pi = <K* f, g>_pi`, so `S* = S` and
`A* = -A` in `L^2(pi)`. Detailed balance `pi_x K(x,y) = pi_y K(y,x)` is exactly
`K = K*`, i.e. `A = 0`, i.e. `J = 0`. Antisymmetry `J(y,x) = -J(x,y)` is immediate,
and

```text
sum_y J(x,y) = pi_x sum_y K(x,y) - sum_y pi_y K(y,x) = pi_x - (pi^T K)_x = pi_x - pi_x = 0
```

by row-stochasticity and stationarity (Kirchhoff's current law). QED.

### Corollary D1: what a predictor can and cannot see `[P]`

A representation or predictor trained only to recover next-state probabilities is
a functional of `S`; the antisymmetric current `A` lies in its kernel. Mutual
information between retained features and the next state measures *predictive
closure*, and does not by itself certify detailed balance, arrow, or cycle
current. A Poole audit should therefore report **both** axes:

```text
predictive : entropy, mutual information, conditional entropy, spectral gap
arrow      : current norm ||A||, antisymmetric spectrum, cycle circulation
topology   : cycle rank, persistent homology, basin-graph structure
```

This is the Conductor / Irreversible Blind Spot connection: a symmetric objective
sees `S` and is blind to `A`, while a predictor (two-encoder) objective recovers
`A`. The companion certificates `apex_recovery/E9` (uniform-`pi` drift ring) and
`E10` (non-normal conveyor) prove this on the ring; **P4 is the general-`pi`
companion**, and it adds the Poole-data instance:

- *Drift ring* `Z/12` (`q=0.5, b=0.2`): all four identities hold to `1.4e-16`,
  `||A|| = 0.735`, the arrow matches E9's closed form `beta = 0.150`, and the
  detailed-balance control `q=b` gives `A = J = 0` exactly.
- *General `pi`* (a non-uniform irreducible chain, `pi = [0.328, 0.379, 0.293]`):
  the adjoint identity `<Kf,g>_pi = <f,K*g>_pi` holds to `4.4e-16`, validating the
  `L^2(pi)` machinery off the uniform point; a conductance (reversible) chain is
  the `A = 0` control.
- *Poole data*: the density-coarsened representative rule with a 2% noise floor
  (the note's "empirical or noisy Poole transition data") is irreducible with
  `pi > 0`, carries a genuine current `||A|| = 1.05` — an arrow in the coarse
  Poole dynamics — whose symmetric part `S` is reversal-invariant while `J` flips
  sign, and whose current has cycle-rank dimension `b_1 = E - V + C = 35` (the
  dimension of the arrow, ties to `apex_recovery/E12`).

So prediction and arrow are different axes, certified on the actual coarse-grained
Poole dynamics, not only on a toy ring.

---

## 6. Theorem E: Dimensionless physics firewall `[P]`

**Theorem E.** *Every quantity the finite bridge constructs from `(X, F, Q, K,
pi)` that is invariant under relabelling the state set is a function of the
isomorphism class alone — a pure number. A dimensionful SI quantity (mass, length,
time, `G`, cosmological distance) is therefore not a function of the finite data;
it requires an additional map from the combinatorial object to an Archimedean
measured scale.*

**Proof.** The inputs are finite sets, Boolean states, maps, distributions, and
transition operators. Relabelling the states by a permutation `sigma` acts by
`F -> sigma F sigma^{-1}`, `Q -> Q sigma^{-1}`, `K -> P K P^{-1}`, `pi -> P pi`. Any
quantity unchanged under all such `sigma` depends only on the isomorphism class of
the data and so carries no distinguished unit — it is dimensionless. A dimensionful
output would distinguish a scale, hence would not be a function of the
relabelling-invariant data alone. QED.

This is a *checked* property, not a disclaimer. P5 verifies the positive half
operationally: the channel's Choi spectrum, the pushforward entropy, the
closure verdict and leaking-class count, and the spectral gap / `||A||` / `||J||`
/ cycle rank are all invariant to `0` over 200 random relabellings; and a
deliberately labelling-dependent control (the integer index `F(0)`, an
index-weighted pushforward sum) **moves** — taking 16 and 109 distinct values —
so the test has teeth. The closed finite outputs are exactly:

```text
closure/leakage witness counts · entropy and mutual information · normalized
density and flux · spectral gaps · current norms · cycle ranks and homology
statistics · dimensionless scaling ratios under refinement.
```

SI-scale masses, lengths, times, and constants require the external unit anchors
and refinement limits of the T15/P0/T30/T32 stack. The firewall holds at the
gate.

---

## 7. Certificates `[P]`

Every theorem is settled by a self-contained, deterministic, CPU-second program in
`empirical/poole_bridge/` (shared toolkit `poole_world.py`; one-command
`run_all.py` with a pass/fail gate). There is no machine learning: the bridge
collapses each claim to exact linear algebra on small finite objects.

| # | theorem | claim certified | headline (exact) |
|---|---|---|---|
| P1 | A | CPTP + Choi PSD; classical face = rule; coherence killed | all residuals `0`; Choi spectrum `{0,1}`; superposition off-diag `0.5 -> 0` |
| P2 | B, B1 | dilation unitary; marginal = pushforward; same-space unitary `<=>` injective | dilation exact; `2x2x1` rule non-injective (image rank `4/16`), collision witnessed |
| P3 | C | closure decided per sector; leakage witnessed | id / basin / translation-orbit **close**; density / cell / window **leak**, witnesses decoded |
| P4 | D, D1 | `S/A/J` at general `pi`; DB `<=>` `A=0=J`; `J` divergence-free; arrow in Poole data | identities `~1e-16`; ring arrow `= 0.150` (matches E9); coarse-Poole `||A||=1.05`, `b_1=35` |
| P5 | E | every invariant relabelling-invariant; control moves | invariants drift `0` over 200 relabellings; control takes 16 / 109 values |
| P6 | A-E on the **real rule** | his canonical unit tests reproduced; closure decided; succession flux is arrow-blind | see Section 7A |

```text
python empirical/poole_bridge/run_all.py     # reproduces every number above
```

P1-P5 run on the representative rule (rule-agnostic demonstrations); **P6 runs the
same machinery on Rooke's actual B5-7/S5-9 rule** (Section 7A).

---

## 7A. The real rule: Obligation 1 discharged `[P]`

Rooke's canonical engine (`PooleEngine.step`, Delta RPM Protocol) is transcribed
torch-free into `otg_rule.py` and certified in P6. We never run his torch code; we
read it and reimplement it in numpy, then prove the reimplementation faithful.

**Faithfulness.** P6 reproduces his own canonical unit tests bit-for-bit — the
vacuum stays empty, and the core of a solid `3x3x3` block evaporates (its 26 live
neighbours exceed `S_HIGH = 9`) — and confirms the exact effective integer rule
`B = {5,6}`, `S = {5,6,7,8,9}`. As an independent cross-check, the vectorised batch
update agrees with the per-cell convolution on every sampled state, and the
equilibrium density from a matched initialisation is `0.3997` — numerically his
quoted `Phi ~ 0.4002`.

**Closure on his rule (Theorem C).** On the two smallest enumerable tori carrying
his exact `3x3x3` kernel — `2x2x2` (256 states) and `4x2x2` (65 536 states), both
translation-equivariant to the last state — the dictionary holds: `id`, the
`attractor/basin` sector (`G = id`; 7 and 495 basins respectively), and the
`translation-orbit` sector (`G` = quotient; 46 and 4216 orbits) **close**, while
`density` and `single-cell` **leak**, each with a decoded before-identical /
after-distinct witness. So his substrate *does* admit exact renormalised sectors
(the symmetry quotient and the basins) and *does* leak coarse density — both
decided, not asserted.

**The succession flux is arrow-blind (Theorem D, and a correction).** Rooke's
*succession flux* `Phi` is defined as a scalar — the fraction of cells changing
state, equivalently the equilibrium occupancy `~0.40`. It is therefore a
*time-symmetric* quantity, and it is **not** the probability current. P6 builds the
exact density-coarsened transition operator of his rule and splits it `S + A` in
`L^2(pi)` (identities to `1.3e-15`): there *is* a genuine current `||A|| = 0.762`
(detailed balance fails), of cycle-rank dimension `b_1 = 114`; and under time
reversal the symmetric part is invariant (`||S - S_rev|| = 4e-17`) while the
current flips (`||J - J_rev|| = 0.02`). The consequence is exact and useful: **any
scalar activity like `Phi` reads only `S` and is blind to the arrow `A`** it is
asked to carry. If the cosmological "succession" is to have a direction, that
direction is the antisymmetric current — a distinct object the flux scalar does
not measure, and which a predictor/current audit recovers (Corollary D1).

This is the sharp, corrected form of the bridge to his program: not "`Phi` is the
current" (it is not), but "`Phi` is exactly the kind of symmetric scalar that the
current audit proves cannot see the arrow — so name the current."

---

## 8. Status ledger

| statement | tier | basis |
|---|---|---|
| Thm A: decoherent CPTP lift, classical face = rule | `[P]` | proof §2 + P1 |
| Thm B / Cor B1: reversible dilation; injective `<=>` same-space unitary | `[P]` | proof §3 + P2 |
| Thm C: closure congruence + leakage witness | `[P]` | proof §4 + P3 |
| Thm D / Cor D1: `S/A/J` audit at general `pi`; prediction `!=` arrow | `[P]` | proof §5 + P4 (+ E9/E10/E12) |
| Thm E: dimensionless firewall | `[P]` | proof §6 + P5 |
| **Real rule** (B5-7/S5-9) faithful + closure decided + current audit | `[P]` | §7A + P6 (his unit tests reproduced bit-for-bit) |
| Succession flux `Phi` is a symmetric scalar, blind to the arrow `A` | `[P]` | §7A + P6 |
| The representative rule's specific closures/witnesses | `[representative]` | P3-P5 on the stand-in rule |
| A *nontrivial* coherent lift whose dephasing is Poole | `[C]` | open; the real OTG content (§11) |
| Poole dynamics is/encodes quantum gravity | `[C]` | not addressed; firewall + open work bound it |

### 8A. How the theorems meet Rooke's own `KNOWN_GAPS.md`

The bridge is not external structure imposed on the program; it closes, or sharply
poses, gaps the OTG team has itself listed.

| Rooke's listed gap | what the bridge contributes |
|---|---|
| "Planck-scale unit mapping — lattice spacing to SI not rigorously derived" | **Theorem E** is exactly this statement, proved: the finite substrate emits only dimensionless invariants; SI scale needs an external Archimedean map. The firewall says *where* the unit assumption must live. |
| "Continuum limit — rigorous renormalisation needed" | **Theorem C** is the finite core of renormalisation: the closure test decides which observables survive coarse-graining (an induced `G`) versus leak. His symmetry-quotient and basin sectors are certified closed (§7A). |
| "Quantum coherence limits / coherence failure" | **Theorem A / Corollary A1** give the exact decoherent floor; his coherence-failure prediction lives in the off-diagonal sector above it — the one genuinely open lift (§11). |
| Succession flux = dark energy (BENCHMARKED) | **§7A**: `Phi` is a symmetric scalar, arrow-blind; the directional content is the current `A`, which a flux scalar cannot see. Sharpens the claim's object. |

---

## 9. Poole-specific closure program (the proof obligations)

The bridge is rule-agnostic; applying it to Rooke's OTG goals is a finite, exact
program with five obligations. Obligations 2-3 and 5's *structure* are already
executable today (the tests exist and run); Obligations 1 and 4 are the team's.

1. **Freeze the actual rule** `[done]`. Now discharged: the canonical B5-7/S5-9
   prime-resonance engine (Moore-26, circular, synchronous) is transcribed in
   `otg_rule.py`, validated against Rooke's own unit tests, and certified in P6
   (§7A). The one convention the verbal spec left ambiguous — input-shift vs
   band-shift resonance — is settled by his code: it is an *additive Gaussian
   potential* on the neighbour count, which for integer counts adds `0.35` at
   primes and so ejects `m=7` from birth, giving effective `B={5,6}`, `S={5,6,7,8,9}`.
2. **Choose the retained sector** and run Theorem C `[ready]`. For each candidate
   `Q` (single-cell local inputs, patch class, density+flux, neighbour-count or
   prime-resonance histogram, basin/attractor class, homology/cycle signature,
   coarse-graph node) either produce the induced `G` or produce leakage witnesses.
   P3 is the harness.
3. **Separate prediction from arrow** via Theorem D `[ready]`. Estimate a coarse or
   noisy `K` on the retained sector and compute `S`, `A`, `J`. This prevents a
   high mutual-information result from being mistaken for an arrow or curvature
   result. P4 is the harness.
4. **Supply a nontrivial quantum lift** `[open, the real work]`. Theorems A-B give
   the guaranteed lift, which decoheres. The next claim must define a coherent
   phase dynamics with `dephase(QuantumStep(rho)) = K_F(dephase(rho))` on the
   classical face while the off-diagonal sector carries testable invariants before
   decoherence. This is the only obligation the bridge does not discharge for free.
5. **Test refinement** `[ready, needs the rule]`. Re-run the invariants over
   increasing `L`, neighbourhood choices, and coarse-grainings. A gravity-like or
   continuum claim attaches only to *stable* dimensionless ratios or *stable*
   leakage/current structures under refinement (Theorem E bounds what can be
   stable).

---

## 10. What we have proved, and what remains open

**Proved and certified.**

```text
fixed finite Poole rule
  -> exact CPTP channel lift                         (Thm A, P1)
  -> exact unitary dilation with environment         (Thm B, P2)
  -> exact decoherent recovery of the automaton      (Cor A1)
  -> injective <=> same-space unitary, witnessed     (Cor B1, P2)
  -> exact observable-sector closure/leakage test    (Thm C, P3)
  -> general-pi current/blind-spot audit             (Thm D, P4)
  -> strict relabelling-invariant (dimensionless) output (Thm E, P5)
```

This is enough to tell Rooke something precise: the bridge from classical OTG
dynamics to *quantum language* is available, but it does not come from analogy —
it comes from channel/dilation theory plus the T7 decoherent-face reading. The
bridge from Poole dynamics to *curvature or gravity-like* language is **not**
automatic; it must pass the T11/T13/T14 closure-leakage test (now executable) and
then survive T15/P0 refinement.

**Open — computational and theorem-level, not philosophical.**

1. Enumerate (or sample, with the honest one-sided guarantee) closure/leakage
   witnesses for the *actual* Poole rule, once frozen. `[C]`
2. Identify which leakage witnesses are *stable* under scale/refinement — the
   curvature-like content of Corollary C1. `[C]`
3. Build a *coherent* phase dynamics whose decoherent face is Poole (Obligation 4).
   This is the substantive physics step the minimal lift deliberately omits. `[C]`
4. Show the off-diagonal sector carries *nontrivial, testable* invariants. `[C]`
5. Prove or falsify whether those invariants match OTG's claimed decoherence,
   phase transition, or gravity-adjacent structure. `[C]`

The note's value is that items 1-5 are now sharply posed against working
harnesses, and the four structural theorems beneath them are closed.

---

## 11. Local program anchors

```text
T7  - complex amplitudes, projective phase, decoherent classical face
T11 - finite apex vertex calculus and projection closure
T13 - exact simplicial renormalization as boundary adjunction
T14 - curvature as projection noncommutation and boundary leakage
T15 - finite-to-Archimedean completion
P0  - place-completion discipline (the adelic keystone)
T30 - finite grid + leakage channel + completion as physics bridge
T32 - physics-marriage firewall: dimensionless structural ratios first
ConductorBlindSpot / Irreversible Blind Spot - transition operator split into
      symmetric and current axes; E9/E10/E12 are the uniform-pi / non-normal /
      topological certificates that P4 generalises to arbitrary pi and Poole data.
```

Certificate suite: `empirical/poole_bridge/` in `github.com/leomurillo/AI-ConductorBlindSpot`
— `poole_world.py` (toolkit) + `otg_rule.py` (Rooke's real rule, torch-free) +
`p1`-`p6` + `run_all.py` (gate, green in ~20 s).

**External anchors (Rooke's program), `github.com/rookepoole/SVP-OTG-Poole-Manifold-tests`:**

```text
OTG    - Observative Tetrahedral Gravity: the cosmological framework derived from
         the substrate (BAO fit benchmarked vs LambdaCDM on DESI DR2).
Poole Manifold - the 3-D B5-7/S5-9 prime-resonance cellular automaton (the substrate).
Succession Flux (Phi ~ 0.4002) - his scalar activity/occupancy; §7A shows it is
         arrow-blind and names the current it cannot carry.
Delta RPM Protocol - his canonical engine + WORM-audited test harness (the rule we
         transcribed in otg_rule.py and validated against in P6).
```
