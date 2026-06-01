# Reproducibility guide

Run commands from the repository root. The checked outputs are tracked under
`empirical/reports/` and `empirical/pythia_rho_x_sweep/`; re-running the
producer scripts overwrites those files unless an explicit `--out` path is
provided.

## Checked artifact commands

```bash
# Appendix B: non-cyclic finite abelian groups.
python empirical/paper34_C4_noncyclic_certificate.py

# Section 4 / Appendix C: exact cyclic ladder.
python empirical/paper34_cyclic_exact_certificate.py

# Section 6.5: synthetic MLP diagnostic, including strong controls.
python empirical/paper34_conductor_blindspot_demo.py --rings 6,8,12,18,30 --include_strong

# Section 8.6: three head-only cubic-aware interventions.
python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form newton --tag adam_alpha0p1
python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form neumann --tag adam_neumann_alpha0p1
python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form within_packet --tag adam_within_packet_alpha0p1

# Regenerate Layer-3 summaries only from checked JSON files.
python empirical/paper34_layer3_cubic_experiment.py --rings 30 --seeds 0,1,2 --tag adam_alpha0p1 --summarize_only
python empirical/paper34_layer3_cubic_experiment.py --rings 30 --seeds 0,1,2 --tag adam_neumann_alpha0p1 --summarize_only
python empirical/paper34_layer3_cubic_experiment.py --rings 30 --seeds 0,1,2 --tag adam_within_packet_alpha0p1 --summarize_only

# Section 6.6: Pythia sweep. These can download models and use substantial GPU/disk.
python empirical/pythia_rho_x_sweep/sweep_n_axis.py --out empirical/pythia_rho_x_sweep/results_n_axis.json
python empirical/pythia_rho_x_sweep/sweep_d_axis.py --out empirical/pythia_rho_x_sweep/results_d_axis.json
```

## Cheap versus expensive runs

| Artifact family | Default validation | Full reproduction cost |
|---|---|---|
| Exact cyclic certificate | Fast; can be regenerated to a temporary `--out` path. | Seconds, offline. |
| Non-cyclic certificate | Fast deterministic function checks in tests. | Seconds, overwrites checked JSON and summary. |
| Synthetic MLP diagnostic | JSON and summary consistency checks. | Minutes per ring; overwrites reports. |
| Layer-3 interventions | Summary and seed-file consistency checks. | GPU recommended; long training run. |
| Pythia sweep | Offline JSON provenance and range checks. | Model downloads, GPU/VRAM, HuggingFace cache. |
| Manuscript PDF | `bash build.sh --clean` smoke build. | Requires Pandoc, XeLaTeX, and `latexmk`. |

## Provenance notes

- The exact cyclic certificate is the authority for the Section 4 ladder counts.
- The synthetic demo's Layer-1 table should match the exact-certificate ladder,
  not an internal authoring artifact name.
- The checked Pythia JSON files were produced before resolved HuggingFace commit
  capture was added. These legacy checked rows remain exact local artifacts but
  do not include the original resolved model commit. New Pythia runs record
  `model_revision_requested` and
  `model_revision_resolved` when the loaded `transformers` object exposes a
  resolved commit hash.
- The checked Layer-3 summaries must include seeds `0`, `1`, and `2`; summary
  means and sample standard deviations are derived from the JSON seed files.

## Build outputs

The PDF build writes:

- `build/paper.md`
- `build/paper.tex`
- `build/paper.pdf`
- `ConductorBlindSpot.pdf`

Use `bash build.sh --clean` for the publication smoke build. Use
`bash build.sh --skip-pdf` when only the Pandoc-to-TeX path needs inspection.
