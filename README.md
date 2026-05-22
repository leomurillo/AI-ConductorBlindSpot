# Conductor Blind Spot — computational artifacts

Companion code and run outputs for the manuscript

> **The Conductor Blind Spot of the Quadratic Curvature Class on Ring-Structured Categorical Heads**
> *From Curvature Preconditioning to Gradient-Based Mechanism Attribution*

The manuscript proves that the curvature model used by every purely second-order preconditioner on a categorical head with cyclically indexed outcomes is constitutionally unable to represent a coupling that the local information geometry provably carries (a finite character-orthogonality identity on $\mathbb{Z}/n\mathbb{Z}$). It then (i) verifies the identity by exact integer arithmetic across a five-ring ladder, (ii) extends to non-cyclic finite abelian groups, (iii) derives a retraining-free diagnostic $\rho_\times$ and demonstrates it on a synthetic modular-addition MLP and on Pythia checkpoints, and (iv) pre-specifies and runs three head-only cubic-aware interventions.

This repository contains the scripts that produce every empirical number, table, and figure in the paper, together with the run outputs they produced.

## Layout

```
ConductorBlindSpot.md             Manuscript source (Markdown; Pandoc → LaTeX → PDF)
ConductorBlindSpot.pdf            Compiled manuscript
build.ps1 / build.sh / BUILD.md   Build scripts and instructions
latex/preamble.tex                LaTeX preamble used by the build
Paper abstracts/                  External-paper abstract notes used while writing
empirical/
  paper34_C4_noncyclic_certificate.py     Appendix B: non-cyclic finite abelian extension
  paper34_conductor_blindspot_demo.py     §6.5: synthetic modular-addition demo
  paper34_layer3_cubic_experiment.py      §8.6: three head-only cubic-aware interventions
  reports/                                Run outputs (JSON / Markdown / PNG)
  pythia_rho_x_sweep/                     §6.6: Pythia checkpoint sweep
```

## Manuscript ↔ artifact map

| Manuscript section | Script | Outputs |
|---|---|---|
| §4 / Appendix C — cyclic five-ring ladder (full tabulation) | (embedded in manuscript) | — |
| Appendix B — non-cyclic finite abelian extension | `empirical/paper34_C4_noncyclic_certificate.py` | `empirical/reports/paper34_C4_*.json`, `paper34_C4_noncyclic_summary.md` |
| §6.5 — synthetic-MLP demo on $n \in \{6, 8, 12, 18, 30\}$ | `empirical/paper34_conductor_blindspot_demo.py` | `empirical/reports/paper34_demo_n{n}_certificate.{json,png}`, `paper34_demo_n{n}_curve.{json,png}`, `paper34_demo_summary.md` |
| §6.6 — Pythia calendar-months sweep ($N$- and $D$-axes) | `empirical/pythia_rho_x_sweep/sweep_n_axis.py`, `sweep_d_axis.py` (with `rho_x.py`, `run_diagnostic.py`, `tokenizer_probe.py`, `smoke_test.py`) | `empirical/pythia_rho_x_sweep/results_{n,d}_axis.json`, `RESULTS.md` |
| §8.6 — three head-only cubic-aware interventions (forms A/B/C, 3 seeds, $\alpha = 0.1$, $n = 30$) | `empirical/paper34_layer3_cubic_experiment.py` | `empirical/reports/paper34_layer3_n30_seed{0,1,2}_adam{,_neumann,_within_packet}_alpha0p1.json`, `paper34_layer3_summary_adam{,_neumann,_within_packet}_alpha0p1.md` |

## Reproducing the runs

Each script is self-contained. From the repository root:

```bash
# Appendix B — non-cyclic groups (exact integer arithmetic, seconds)
python empirical/paper34_C4_noncyclic_certificate.py

# §6.5 — synthetic-MLP demo (CPU-friendly; ~minutes per ring)
python empirical/paper34_conductor_blindspot_demo.py

# §8.6 — three head-only cubic-aware interventions (one GPU recommended)
python empirical/paper34_layer3_cubic_experiment.py

# §6.6 — Pythia sweep (single consumer GPU; fp16; the run reported in the
# paper used an RTX 3050 Laptop / 4 GB)
python empirical/pythia_rho_x_sweep/sweep_n_axis.py
python empirical/pythia_rho_x_sweep/sweep_d_axis.py
```

The `empirical/reports/` and `empirical/pythia_rho_x_sweep/results_*.json` files in this repository are the exact outputs cited in the manuscript and are checked in for verifiability; re-running the scripts overwrites them.

## Dependencies

- Python 3.10+
- `numpy`, `scipy` (Appendix B / §6.5)
- `torch` (§6.5, §8.6)
- `transformers`, `torch` (§6.6 — Pythia via the GPT-NeoX tokenizer)
- `matplotlib` (figures in `empirical/reports/`)

## Building the manuscript

See `BUILD.md`. In short:

```bash
./build.sh       # POSIX
./build.ps1      # Windows PowerShell
```

The build produces `build/paper.pdf` via Pandoc → LaTeX (XeLaTeX), using `latex/preamble.tex`.

## Citation

A BibTeX entry will be added once an arXiv identifier is assigned.

## License

Code and manuscript are released for academic use; a formal license file will be added with the arXiv submission.
