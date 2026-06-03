# Continuous-theory certificates (E1–E16) — two axes of one operator

Companion code for the continuous-theory papers in `../`: *Beyond the Conductor
Blind Spot* (the **cumulant axis**), *The Irreversible Blind Spot* (the
**reversibility axis**), and the unified *Curvature and Current*. Everything the
aligned-whitened learner recovers is governed by one transition operator
`T = S + A`; the certificates split into the curvature of `S`'s slow
eigenfunctions (E1–E8) and the antisymmetric current `A`, the arrow of time
(E9–E16). These scripts are separate from the `paper34_*` Conductor-Blind-Spot
artifacts in `../`; they certify the *new* results and are written to be read,
not just run.

> **The one sentence.** Minimising alignment subject to whitening recovers the
> latent variable *through the slowest eigenfunction* `phi_1` of the transition
> operator, up to rotation. `phi_1` is a straight line **iff** the world is
> Gaussian (the Gaussian corner); otherwise it is a monotone curve, and a
> *linear* probe necessarily fails where the *eigenfunction* probe succeeds.

## What each script certifies

| Script | Result it certifies | Headline numbers |
|---|---|---|
| `e1_eigenfunction_recovery.py` | **Exact theorem.** `phi_1` affine ⟺ Gaussian; otherwise monotone-nonlinear. Recovery is intrinsic to the operator (mixing-invariant). | nonlinearity `nu`: gaussian `0.000`, uniform `0.014`, bimodal `0.068`, laplace `0.131`; cube-warp recovers closed-form `cbrt(z)` while a linear probe is stuck at `R²=0.61` |
| `e2_approximate_bound.py` | **Approximate recovery theorem** (Thm 3): `min_U‖h−UΦ₁‖² ≤ (ε+√(2δ/γ))²`, plus the product-spectrum / Assumption-(G) check. | bound holds **15/15** trials; `(G)` verified (`γ=0.0224`, top-2 eigenspace = single-coordinate modes exactly); one heterogeneous pair shown to *violate* `(G)` |
| `e3_hankel_reconstruction.py` | **The distributional tower** (§3.4): finite determinacy of the stationary law in exact rational arithmetic, and the tie to CBS. *Not* the recovery map — the two towers coincide only at the Gaussian/rank-2 corner. | rank-3 world: Hankel dets `1,5,81,0` → closes at 3; Prony recovers levels `{−2,1,4}` exactly; rank-2 = the affine (Gaussian) corner; CBS cross-packet cubic counts `18,42,108,252,774` **reproduced** |
| `e4_cross_register_bridge.py` | **The bridge, measured** (§3.4 → measured): the dynamical curvature `ν_D` and the distributional cumulants vanish together at the Gaussian, locked by a square law `ν_D ∝ (leading cumulant)²`. | skew family: `ν_D/k₃² → 0.0555` (stable 1.3%, **order 3** — CBS-cubic); symmetric family: `k₃=0` to 1e-17, `ν_D/k₄² → 0.0104` (**order 4**); ring: scalar skewness `0` vs cross-packet cubic `18…774` (ring keeps order-3 alive) |
| `e5_real_model_bridge.py` *(optional; needs `torch`+`transformers`)* | **The bridge on a real model** (§6.5): each residual-stream direction of Pythia is a world; the depth profile of `corr(ν_D, leading-cumulant²)`. *Observational, not gated.* | **middle-layer phenomenon**: Spearman rises to **+0.41** at layer 4 (band 3–5), `≈0` at embedding/early/final; peak layer most-Gaussian dir `ν_D=0.03` (affine) vs rogue dim (skew 11.8, kurt 142) `ν_D=0.51` (curved) |
| `e6_grokking_bridge.py` *(needs `torch`)* | **Grokking as the bridge in time** (§7): trains modular addition and tracks the distributional (logit additivity) and dynamical (embedding Fourier) registers vs validation accuracy + a control. | two registers rise **together** to val-acc 1, flat on a structure-free control (`p=113`, `n=110`, `n=121`); persistent cross-packet mass accumulates on the ring task |
| `e7_planning_certificate.py` | **Planning faithfulness** (Prop 1 + Thm 4): the recovered chart is a faithful coordinate for control. | chart agent exact (`<1e-9`); a linear planner that ignores the warp is sub-optimal off-Gaussian; the Thm 4 regret bound holds |
| `e8_trained_encoder.py` *(needs `torch`)* | **SGD reaches the chart** (§6.7): a trained encoder reaches the slow-feature `phi_1`. | recovery corr `0.96–1.00`; closes the population-optimum-vs-trained gap (2-D partial) |

### The reversibility axis (E9–E16): the arrow of time

| Script | Result it certifies | Headline numbers |
|---|---|---|
| `e9_irreversible_ring.py` | **The Irreversible Blind Spot** on the drift ring `Z/n`: the single-encoder objective is a functional of the symmetric part `S` alone; the predictor recovers the antisymmetric arrow. | identity `3e-16`; loss reversal-invariant `9e-16`; predictor block exact; blind spot **OFF** at detailed balance `q=b` |
| `e10_nonnormal_svd.py` | **Non-normal SVD** (§3.2): left/right charts split; the irreversibility gap `Δ = Σσ − Σλ(S) ≥ 0`. | charts differ up to `60°`; `Δ = 0.40 / 0.025 / 0` (conveyor / ring / reversible) |
| `e11_trained_two_encoder.py` *(needs `torch`)* | **SGD reaches the predictor**: single encoder reversal-blind; the trained predictor reads the signed arrow. | `β̂` flips `+0.17 → −0.17` under time reversal (ground truth `0.15`) |
| `e12_topological_blindspot.py` | **Topological dimension**: recovered current dim `= β₁ = E−V+1`, refining to the Hodge Betti number `b₁`. | ring 1, theta 2, `K₄` 3, torus 10; refinement torus `10→2`, `K₄ 3→0` |
| `e13_estimating_the_arrow.py` | **What is cheaply estimable**: the arrow's *strength* `Δ` converges from samples; its *dimension* `b₁` is an offline audit. | `Δ` rel-err `2.6% → 0.2%` (10³→10⁵ pairs); single-scale `b₁` is wrong/scale-dependent |
| `e14_two_axes.py` | **The two axes are orthogonal**: `ν_D` tracks non-Gaussianity, `Δ` tracks irreversibility, with zero cross-talk. | all four corners realised; cross-talk `< 1e-9` both ways |
| `e15_efficient_betti.py` | **Efficient `b₁` engine**: `b₁ =` nullity of the sparse Hodge 1-Laplacian, read from the bottom of the spectrum. | exact vs ground truth; torus `b₁=2` to `L=40` in `0.4 s`; sphere 0; two tori 4 |
| `e16_pointcloud_betti.py` *(needs GUDHI)* | **Faithful complex from samples** by persistent homology; composes with E15. | `b₁ = 1,2,0,3` exact & robust (circle / torus / sphere / 3 circles); sub-second to ~2.6 s |

The figures (`reports/*.png`) are produced when `matplotlib` is available and
skipped otherwise; the JSON certificates in `reports/` are the source of truth.
The deterministic pass/fail gate in `run_all.py` covers **E1–E4, E7, E9, E10,
E12–E15** (numpy/scipy/sympy only). The trained certificates **E8, E11** and the
grokking testbed **E6** need `torch`; the real-model bridge **E5** needs
`torch`+`transformers`+a cached Pythia checkpoint; the point-cloud audit **E16**
needs a TDA library (GUDHI). All are runnable; all sit outside the
dependency-light gate.

## How the worlds are built

A "world" is a latent `z` with a stationary law `p(z)`, observed through
reversible additive-noise pairs `(z, z')`. We realise it as a **reversible
nearest-neighbour Metropolis chain** on a grid — the exact, fully
diagonalisable discrete cousin of the Ornstein–Uhlenbeck transition. The
shared toolkit is [`apex_world.py`](apex_world.py); it has no machine learning
in it, because the theory says the learning problem *collapses to the
eigenproblem of one self-adjoint operator*. All operators are deterministic;
the only randomness is the seeded contamination in E2.

## Reproducing

```bash
# full suite + self-check gate (a few seconds, CPU, no network)
python empirical/apex_recovery/run_all.py     # use `py` instead of `python`
                                              # to also render the PNG figures
# or individually
python empirical/apex_recovery/e1_eigenfunction_recovery.py
python empirical/apex_recovery/e2_approximate_bound.py
python empirical/apex_recovery/e3_hankel_reconstruction.py
python empirical/apex_recovery/e4_cross_register_bridge.py

# optional real-model certificate (needs torch+transformers + cached Pythia):
py empirical/apex_recovery/e5_real_model_bridge.py
```

`run_all.py` exits nonzero if any built-in check fails — E2's bound holds in
every trial; E3 is exact and matches the CBS counts; E7's chart agent is exact;
E9's blind-spot identity is exact and switches off at detailed balance; E10's
gap is `0` iff reversible; E12's recovered current dimension equals the cycle
rank `β₁` (refining to `b₁`); E14's two axes have zero cross-talk; E15's
sparse-Hodge `b₁` matches ground truth.

## Dependencies

`numpy`, `scipy`, `sympy` (E3 exact arithmetic); `matplotlib` optional (figures).
The gated certificates (E1–E4, E7, E9, E10, E12–E15) need no GPU and no network.
**Optional heavier stacks:** `torch` for the trained certificates (E8, E11) and
the grokking testbed (E6); `torch`+`transformers`+a cached Pythia checkpoint for
the real-model bridge (E5); a TDA library (`gudhi`) for the point-cloud audit
(E16). Tested on Python 3.12/3.13, numpy 2.4, scipy 1.17, sympy 1.14, torch 2.12
(CPU), transformers 5.9.

## Relationship to the manuscripts

* **Conductor Blind Spot** (`../ConductorBlindSpot.md`) is the finite,
  distributional register this paper builds on: the discrete witness that the
  "third rung" (order-3 Amari–Chentsov cubic) is occupied on cyclic heads. E3
  regenerates its Section 4 counts exactly.
* The interior **T-series** T0–T7 (Zenodo) supply the gauge corridor and the
  cumulant tower. The finite-determinacy (Hankel/Prony) machinery E3 uses is
  classical and is restated in-house in the paper's appendix.
* The **Gaussian/linear case** is established concurrently in the
  self-supervised-learning literature (Klindt, LeCun & Balestriero 2026,
  arXiv:2605.26379); it is the Gaussian corner of every certificate here, cited
  as the boundary case the general theory contains.
