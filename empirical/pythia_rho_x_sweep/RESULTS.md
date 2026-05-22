# Pythia rho_x sweep — first-run results

First-run application of the §6 diagnostic to Pythia checkpoints on a
$\mathbb{Z}/12\mathbb{Z}$-indexed conditional head (twelve single-token
English month abbreviations under templated cyclic-shift prompts).

Substrate: GPT-NeoX BPE tokenizer; every short-form month (Jan, ..., Dec)
is a single token in both bare and leading-space contexts. Prompts:
five context lengths $k \in \{3, 5, 7, 9, 11\}$ x twelve cyclic shifts =
60 prompts per checkpoint. Conditional softmax over 12 month tokens in
whichever style (bare vs leading-space) carries more average mass.
50 random permutations per measurement. fp16, RTX 3050 Laptop 4 GB.

Structural ceiling for $n = 12$: $\rho_\times = 108/110 \approx 0.982$.
The §6.5 synthetic-MLP demo at $n = 12$ pre-grokking gave
$\overline{\rho_\times^{D1}} \approx 0.847$ (ring) vs
$\overline{\rho_\times^{D2\text{-weak}}} \approx 0.995$ (permuted control).

## N-axis (final checkpoint across model sizes)

| model        | top1  | ring_mass | bm_ring | bm_perm | bm_sep   | pe_ring | pe_perm | pe_sep   |
|--------------|------:|----------:|--------:|--------:|---------:|--------:|--------:|---------:|
| pythia-70m   | 0.083 |    0.014  |  0.891  |  0.970  |  +0.079  |  0.979  |  0.979  |  +0.000  |
| pythia-160m  | 0.017 |    0.253  |  0.924  |  0.954  |  +0.030  |  0.978  |  0.980  |  +0.001  |
| pythia-410m  | 0.900 |    0.848  |  0.953  |  0.969  |  +0.015  |  0.960  |  0.975  |  +0.015  |
| pythia-1b    | 0.533 |    0.625  |  0.952  |  0.970  |  +0.018  |  0.973  |  0.977  |  +0.005  |
| pythia-1.4b  | 0.700 |    0.613  |  1.000  |  0.954  |  -0.046  |  0.973  |  0.977  |  +0.004  |

## D-axis (Pythia-1B intermediate checkpoints)

| step    | tokens   | top1  | ring_mass | bm_ring | bm_sep   | pe_ring | pe_sep   |
|--------:|---------:|------:|----------:|--------:|---------:|--------:|---------:|
|     128 |  2.7e+08 | 0.083 |    0.000  |  0.971  |  +0.001  |  0.980  |  +0.000  |
|    1000 |  2.1e+09 | 0.050 |    0.171  |  0.983  |  -0.008  |  0.980  |  -0.001  |
|    3000 |  6.3e+09 | 0.033 |    0.007  |  0.871  |  +0.099  |  0.981  |  -0.001  |
|   10000 |  2.1e+10 | 0.033 |    0.052  |  0.929  |  +0.035  |  0.983  |  -0.004  |
|   30000 |  6.3e+10 | 0.300 |    0.456  |  0.984  |  -0.017  |  0.980  |  -0.001  |
|   70000 |  1.5e+11 | 0.333 |    0.239  |  0.839  |  +0.134  |  0.977  |  +0.002  |
|  143000 |  3.0e+11 | 0.533 |    0.625  |  0.952  |  +0.018  |  0.973  |  +0.005  |

## Reading

Per-example $\rho_\times$ on the ring task sits in $[0.96, 0.98]$ across
the entire N x D sweep — at or just below the structural ceiling 0.982,
~0.13 above the §6.5 synthetic n=12 pre-grokking value of 0.847.
Per-example separation against the permuted control is at the
per-permutation noise floor across all 12 measurements.

The §6.1 batch-mean form fluctuates between -0.046 and +0.134 across
the sweep without monotone $N$ or $D$ trend; we attribute this to phase
cancellation of the per-prompt centered scores when prompts are
cyclically balanced and the model is either confident-and-correct
(u^(i) -> 0) or confident-and-wrong-shift-equivariantly (per-prompt
phase washes out in the batch mean). Per-example is the robust statistic
on this substrate.

By the §8.6 directionality insight (ring-coherent learning concentrates
the gradient's cubic mass *within* a conductor packet, driving rho_x
below the structural default), values sitting AT the default mean
no ring consolidation has occurred. Pythia's calendar-months head is
*off the ring-structure trajectory* at every tested scale point.

This is a third diagnostic-boundary observation, complementing
- structural-degeneracy boundary (n=8, rho_x identically 1)
- insufficient-budget boundary (n=18, pre-representation-learning)
- and now: off-trajectory boundary (calendar tokens in a generic LM head)

Captured in the paper as §6.6 and §9 (Diagnostic deployment scope).

## Reproduction

```
.venv\Scripts\python.exe pythia_rho_x_sweep\sweep_n_axis.py
.venv\Scripts\python.exe pythia_rho_x_sweep\sweep_d_axis.py
```

Total compute: ~25 min on RTX 3050 Laptop 4 GB.
Cached models: ~12 GB on disk under ~/.cache/huggingface/hub.
Raw JSON outputs:
- pythia_rho_x_sweep/results_n_axis.json
- pythia_rho_x_sweep/results_d_axis.json
