# Eigenfunction recovery — computational certificates

Companion code for the (in-preparation) paper *Beyond the Conductor Blind Spot:
Eigenfunction Identifiability and Planning in Non-Gaussian Worlds*
(`../BeyondTheConductorBlindSpot.md`) — the continuous theory of what a
self-supervised representation recovers when the world is not Gaussian. These
scripts are separate from the `paper34_*` Conductor-Blind-Spot artifacts in
`../`; they certify the *new* results and are written to be read, not just run.

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
| `e6_grokking_bridge.py` *(forward experiment; needs `torch`)* | **Grokking as the bridge in time** (§7 prediction): trains modular addition and tracks the distributional (logit additivity) and dynamical (embedding Fourier) registers vs validation accuracy + a control. *Not a completed result.* | pre-grok baseline reproduced (memorise→1.0, val≈chance, both registers at order-2 baseline); the in-training co-emergence is the follow-up to run |

The figures (`reports/e1_eigenfunctions.png`, `reports/e2_bound_scatter.png`,
`reports/e4_bridge.png`, `reports/e5_real_model_bridge.png`) are produced when
`matplotlib` is available and skipped otherwise; the JSON certificates in
`reports/` are the source of truth. E5 is the only experiment needing a GPU-free
but heavier stack (`torch`+`transformers`+a cached Pythia checkpoint); it is
**not** part of the `run_all.py` pass/fail gate.

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

`run_all.py` exits nonzero if any built-in check fails (E2's bound must hold in
every trial; E3's exact reconstructions and the CBS count match are asserted).

## Dependencies

`numpy`, `scipy`, `sympy` (E3 exact arithmetic); `matplotlib` optional (figures).
E1–E4 need no GPU and no network. **E5 only**: `torch` (CPU is fine) +
`transformers` + a cached Pythia checkpoint (runs fully offline from the HF
cache). Tested on Python 3.12 and 3.13, numpy 2.4, scipy 1.17, sympy 1.14,
torch 2.12 (CPU), transformers 5.9.

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
