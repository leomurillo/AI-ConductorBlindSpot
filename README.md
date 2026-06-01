# Conductor Blind Spot — computational artifacts

Companion code and run outputs for the manuscript

> **The Conductor Blind Spot of the Quadratic Curvature Class on Ring-Structured Categorical Heads**
> *From Curvature Preconditioning to Gradient-Based Mechanism Attribution*

The manuscript proves that the curvature model used by every purely second-order preconditioner on a categorical head with cyclically indexed outcomes is constitutionally unable to represent a coupling that the local information geometry provably carries (a finite character-orthogonality identity on $\mathbb{Z}/n\mathbb{Z}$). It then (i) verifies the identity by exact integer arithmetic across a five-ring ladder, (ii) extends to non-cyclic finite abelian groups, (iii) derives a retraining-free diagnostic $\rho_\times$ and demonstrates it on a synthetic modular-addition MLP and on Pythia checkpoints, and (iv) pre-specifies and runs three head-only cubic-aware interventions.

This repository contains the scripts that produce every empirical number, table, and figure in the paper, together with the run outputs they produced.

It also hosts the **sibling preprint** — *Beyond the Conductor Blind Spot: Eigenfunction Identifiability and Planning in Non-Gaussian Worlds* (`BeyondTheConductorBlindSpot.md` / `.pdf`) — the continuous theory of what a self-supervised representation recovers when the world is not Gaussian, with its own self-contained certificate suite in [`empirical/apex_recovery/`](empirical/apex_recovery/).

## Layout

```
ConductorBlindSpot.md             Manuscript source (Markdown; Pandoc → LaTeX → PDF)
ConductorBlindSpot.pdf            Compiled manuscript
BeyondTheConductorBlindSpot.md    Sibling manuscript (non-Gaussian recovery & planning)
BeyondTheConductorBlindSpot.pdf   Compiled sibling manuscript
build.ps1 / build.sh / BUILD.md   Build scripts (take the source .md as an argument)
docs/                            Reproducibility, validation, source, and claims docs
latex/preamble.tex                Shared LaTeX preamble used by both builds
Paper abstracts/                  External-paper abstract notes used while writing
empirical/
  paper34_cyclic_exact_certificate.py     §4 / Appendix C: exact cyclic ladder certificate
  paper34_C4_noncyclic_certificate.py     Appendix B: non-cyclic finite abelian extension
  paper34_conductor_blindspot_demo.py     §6.5: synthetic modular-addition demo
  paper34_layer3_cubic_experiment.py      §8.6: three head-only cubic-aware interventions
  reports/                                Run outputs (JSON / Markdown / PNG)
  pythia_rho_x_sweep/                     §6.6: Pythia checkpoint sweep
  apex_recovery/                          Sibling-paper certificate suite (own README)
    apex_world.py                         shared toolkit (reversible-chain eigenproblem)
    e1_eigenfunction_recovery.py          §6.1 exact recovery + Gaussian boundary
    e2_approximate_bound.py               §6.2 approximate-recovery bound
    e3_hankel_reconstruction.py           §6.3 exact distributional tower + blind-spot counts
    e4_cross_register_bridge.py           §6.4 measured square law (log–log slope 2)
    e5_real_model_bridge.py               §6.5 real-model depth profile (opt.: torch+transformers)
    e6_grokking_bridge.py                 §7 grokking testbed (forward experiment)
    run_all.py                            one-command runner + self-check gate
    reports/                              JSON certificates + figures
```

### Sibling suite: Beyond the Conductor Blind Spot

`empirical/apex_recovery/` holds the certificates for the sibling preprint —
the continuous theory of what a self-supervised representation recovers when
the world is not Gaussian (exact recovery, the approximate-recovery bound, the
exact distributional tower, the measured cross-register square law, and a
real-model depth profile). It is self-contained with its own
[`README.md`](empirical/apex_recovery/README.md) and one-command runner
`python empirical/apex_recovery/run_all.py` (E1–E4 need only `numpy`/`scipy`/`sympy`;
E5 optionally `torch`+`transformers`). Its `e3` certificate regenerates this
manuscript's Section 4 cross-packet cubic counts (`18, 42, 108, 252, 774`)
exactly, tying the two papers together.

## Manuscript ↔ artifact map

| Manuscript section | Script | Outputs |
|---|---|---|
| §4 / Appendix C — cyclic five-ring ladder (full tabulation) | `empirical/paper34_cyclic_exact_certificate.py` | `empirical/reports/paper34_cyclic_exact_certificate.json` |
| Appendix B — non-cyclic finite abelian extension | `empirical/paper34_C4_noncyclic_certificate.py` | `empirical/reports/paper34_C4_*.json`, `paper34_C4_noncyclic_summary.md` |
| §6.5 — synthetic-MLP demo on $n \in \{6, 8, 12, 18, 30\}$ | `empirical/paper34_conductor_blindspot_demo.py` | `empirical/reports/paper34_demo_n{n}_certificate.{json,png}`, `paper34_demo_n{n}_curve.{json,png}`, `paper34_demo_summary.md` |
| §6.6 — Pythia calendar-months sweep ($N$- and $D$-axes) | `empirical/pythia_rho_x_sweep/sweep_n_axis.py`, `sweep_d_axis.py` (with `rho_x.py`, `run_diagnostic.py`, `tokenizer_probe.py`, `smoke_test.py`) | `empirical/pythia_rho_x_sweep/results_{n,d}_axis.json`, `RESULTS.md` |
| §8.6 — three head-only cubic-aware interventions (forms A/B/C, 3 seeds, $\alpha = 0.1$, $n = 30$) | `empirical/paper34_layer3_cubic_experiment.py` | `empirical/reports/paper34_layer3_n30_seed{0,1,2}_adam{,_neumann,_within_packet}_alpha0p1.json`, `paper34_layer3_summary_adam{,_neumann,_within_packet}_alpha0p1.md` |

## Reproducing the runs

For the complete documentation set, start with [`docs/README.md`](docs/README.md).
The detailed artifact matrix and validation checklist live in
[`docs/reproducibility.md`](docs/reproducibility.md) and
[`docs/validation.md`](docs/validation.md). The final evidence-tiered claims
assessment is [`docs/final_claims_assessment.md`](docs/final_claims_assessment.md).

Each script is self-contained. From the repository root:

```bash
# Appendix B — non-cyclic groups (exact integer arithmetic, seconds)
python empirical/paper34_C4_noncyclic_certificate.py

# §4 / Appendix C — cyclic ladder (exact integer arithmetic, seconds)
python empirical/paper34_cyclic_exact_certificate.py

# §6.5 — synthetic-MLP demo (CPU-friendly; ~minutes per ring)
python empirical/paper34_conductor_blindspot_demo.py --rings 6,8,12,18,30 --include_strong

# §8.6 — three head-only cubic-aware interventions (one GPU recommended)
python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form newton --tag adam_alpha0p1
python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form neumann --tag adam_neumann_alpha0p1
python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form within_packet --tag adam_within_packet_alpha0p1

# Regenerate only the checked Layer-3 summaries from their JSON artifacts
python empirical/paper34_layer3_cubic_experiment.py --rings 30 --seeds 0,1,2 --tag adam_alpha0p1 --summarize_only
python empirical/paper34_layer3_cubic_experiment.py --rings 30 --seeds 0,1,2 --tag adam_neumann_alpha0p1 --summarize_only
python empirical/paper34_layer3_cubic_experiment.py --rings 30 --seeds 0,1,2 --tag adam_within_packet_alpha0p1 --summarize_only

# §6.6 — Pythia sweep (single consumer GPU; fp16; the run reported in the
# paper used an RTX 3050 Laptop / 4 GB)
python empirical/pythia_rho_x_sweep/sweep_n_axis.py --out empirical/pythia_rho_x_sweep/results_n_axis.json
python empirical/pythia_rho_x_sweep/sweep_d_axis.py --out empirical/pythia_rho_x_sweep/results_d_axis.json
```

The `empirical/reports/` and `empirical/pythia_rho_x_sweep/results_*.json` files in this repository are the exact outputs cited in the manuscript and are checked in for verifiability; re-running the scripts overwrites them.

The checked Pythia JSON files were produced before HuggingFace commit-hash
capture was added; new runs record both the requested revision and the resolved
model commit when the `transformers` objects expose it.

## Dependencies

- Python 3.10+
- `numpy`, `scipy` (Appendix B / §6.5)
- `torch` (§6.5, §8.6)
- `transformers`, `torch` (§6.6 — Pythia via the GPT-NeoX tokenizer)
- `matplotlib` (figures in `empirical/reports/`)

## Building the manuscript

See `BUILD.md`. In short:

```bash
bash build.sh BeyondTheConductorBlindSpot.md          # the sibling paper (POSIX)
./build.ps1 -Source BeyondTheConductorBlindSpot.md    # Windows PowerShell
bash build.sh                                         # ConductorBlindSpot.md (default)
```

Each build produces the compiled `<source>.pdf` via Pandoc → LaTeX (XeLaTeX), using the shared `latex/preamble.tex`.

## Citation

A BibTeX entry will be added once an arXiv identifier is assigned.

## License

Code and manuscript are released for academic use; a formal license file will be added with the arXiv submission.
