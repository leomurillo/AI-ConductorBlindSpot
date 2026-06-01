# Validation guide

This repository intentionally keeps validation deterministic and offline by
default. Expensive model reruns are documented separately from the tests so a
fresh clone can still check artifact consistency without downloading Pythia.

## Required local checks

```bash
git diff --check
bash -n build.sh
python3 - <<'PY'
import ast
from pathlib import Path
for path in sorted(Path(".").rglob("*.py")):
    if any(part.startswith(".") for part in path.parts):
        continue
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("parsed python files")
PY
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q
python3 empirical/paper34_cyclic_exact_certificate.py --out /tmp/paper34_cyclic_exact_certificate.json
bash build.sh --clean
```

If PowerShell is installed, also parse the Windows build script:

```bash
pwsh -NoProfile -Command '[scriptblock]::Create((Get-Content -Raw build.ps1)) | Out-Null'
```

After a successful PDF build, verify:

```bash
test -s ConductorBlindSpot.pdf
test -s build/paper.pdf
cmp -s build/paper.pdf ConductorBlindSpot.pdf
pdfinfo ConductorBlindSpot.pdf | rg '^Pages:'
```

## What the tests cover

- README and docs command contracts point at real checked outputs.
- The exact cyclic ladder is regenerated to a temporary path using integer
  arithmetic only.
- Non-cyclic candidate JSON files match the deterministic verifier functions.
- Synthetic demo summaries match per-ring JSON artifacts.
- Layer-3 summaries include all seed JSONs and compute the manuscript aggregate
  rows from those source files.
- Pythia JSON rows keep repo-root-safe output defaults, requested revisions,
  resolved-revision keys, and bounded diagnostic values.
- The source ledger covers all manuscript arXiv references used for empirical
  support and records DCD as `under_review`.
- Build scripts avoid environment-specific hard-coded font overrides.
- The checked `build/paper.pdf` and root `ConductorBlindSpot.pdf` are identical.

## Extended validations

The following commands are not part of the default offline test suite because
they are compute-heavy or may download models:

```bash
python empirical/paper34_conductor_blindspot_demo.py --rings 6,8,12,18,30 --include_strong
python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form newton --tag adam_alpha0p1
python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form neumann --tag adam_neumann_alpha0p1
python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form within_packet --tag adam_within_packet_alpha0p1
python empirical/pythia_rho_x_sweep/sweep_n_axis.py --out empirical/pythia_rho_x_sweep/results_n_axis.json
python empirical/pythia_rho_x_sweep/sweep_d_axis.py --out empirical/pythia_rho_x_sweep/results_d_axis.json
```

Run these only when the goal is to regenerate the checked empirical artifacts,
not merely to verify repository consistency.
