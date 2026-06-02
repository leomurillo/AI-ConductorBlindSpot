# E6 Grokking-Bridge GCP Runbook

Runs `empirical/apex_recovery/e6_grokking_bridge.py` on a cloud GPU: the
in-training crossing of the order-2 -> order-3 boundary on modular addition, with
a matched structure-free control. This is the dynamical, in-training twin of E4
(synthetic limit) and E5 (static real model).

This is a **much lighter** job than the `quantum_lift` audit. That one loads
7-8B pretrained LLMs (24 GB VRAM, HF token, gated access, ~15 GB download). E6
trains a **tiny 1-layer transformer from scratch**: no pretrained model, no HF
token, no download, <1 GB VRAM. The same L4 VM works and is overkill.

## What The Experiment Measures

Task: exhaustive modular addition `(a + b) mod n` for all `n^2` pairs, 1-layer
transformer, AdamW with strong weight decay (the canonical grokking recipe;
Power et al. 2022, Nanda et al. 2023).

Three quantities are logged each step (all in `e6_diagnostics.py`, validated
torch-free against CBS's ladder counts):

- **D (dynamical)** -- embedding Fourier concentration: the token embedding
  aligning to the ring's slow eigenfunctions (its character chart).
- **S (distributional, proxy)** -- logit additivity: share of the logit tensor
  explained by the sum-class `s=(a+b) mod n` (the head depending only on a+b).
- **rho_x (distributional, CBS Definition 6.1)** -- cross-packet cubic mass of
  the batch-averaged centered score. The **actual** Conductor-Blind-Spot
  diagnostic; its trajectory is CBS Conjecture 5.8 (closed analytically in
  Remark 5.9, left open as a trajectory question -- this run is that test).

Pre-registered prediction: at the grok, D and S (and rho_x, composite n) **rise
together** and track val accuracy; **all flat on the control**. Different rise
times, or a moving control, falsifies the single-boundary reading.

### PRIME vs COMPOSITE -- run both

`rho_x` is **identically 0 for a prime modulus** (every nonzero character has
conductor `n`, so there are no cross-packet triples). So:

- **prime `--p 113`** -- canonical clean grokking; gives the **D/S** co-emergence.
- **composite `--p 30`** (and/or `12`) -- the **rho_x / Conjecture-5.8** test.

The script prints which regime it is in at startup and forces `rho_x = 0` in the
prime case rather than reporting noise.

## GCP Hardware Choice

Reuse the proven config from the quantum-lift runbook -- one `g2-standard-8`
(1x NVIDIA L4, 24 GB). E6 needs almost none of it, so cheaper options all work:
`g2-standard-4`, an `n1-standard-4` + T4, or even CPU-only (`e2-standard-8`,
~2x slower but zero GPU cost). The L4 is the safe, fast default.

```bash
gcloud compute instances create cbs-e6-vm \
  --project=causalab-llama \
  --zone=us-east4-c \
  --machine-type=g2-standard-8 \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True
```

Notes vs the quantum-lift command: boot disk **100 GB** (no 15 GB model cache);
**no `huggingface-cli login`** (nothing is downloaded); conservative image
fallback is still `pytorch-2-7-cu128-ubuntu-2204-nvidia-570`. If GCP rejects the
accelerator flag on `g2-standard-8`, drop it (G2 has a fixed L4 attached).

## Runtime Estimate

The model is tiny; the only real variable is how many steps grokking takes, which
the auto-stop bounds (`--stop_after_grok`, halts ~4000 steps past grok).

| Run | Steps | Hardware | Wall time |
|---|---:|---|---:|
| Smoke (pre-grok only) | 600 | 1x L4 | ~1 min |
| One task (prime or composite), to grok + buffer | <= 40000 | 1x L4 | 10 to 20 min |
| Control (never groks, full cap) | 40000 | 1x L4 | 10 to 20 min |
| Task + control, run in parallel on the idle L4 | -- | 1x L4 | 15 to 25 min total |

Add 5 to 10 min for VM create + driver + clone (no model download). **A full
session -- both regimes if you want -- is well under an hour and roughly $1.**

## VM Setup (one-time, after SSH)

The Deep Learning image already has CUDA + PyTorch. E6 only needs numpy.

```bash
nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
git clone https://github.com/leomurillo/AI-ConductorBlindSpot.git
cd AI-ConductorBlindSpot
python -m pip install -U numpy            # scipy/matplotlib NOT needed on the VM
```

No Hugging Face login, no model license -- E6 downloads nothing.

## Smoke Test First (always)

Confirms the apparatus, the diagnostics, and the control path all run before you
spend the long run. 600 steps will NOT grok (it captures the pre-grok baseline,
matching the earlier partial run: train_acc -> 1, val_acc ~ chance, both
registers low). You are checking it does not crash and the JSONL has rho_x/u.

```bash
python empirical/apex_recovery/e6_grokking_bridge.py --p 30 --steps 600 --log_every 200
```

Expected: a `reports/e6_modadd_p30_metrics.jsonl` with rows carrying
`rho_x`, `u`, `emb_shares`; startup line says `COMPOSITE (774 cross-packet triples)`.

## The Experiment Commands

Run both regimes; task and control in parallel on the idle L4 (`&` then `wait`):

```bash
# prime p=113 -- canonical grokking, the D/S co-emergence (rho_x is 0 here)
python empirical/apex_recovery/e6_grokking_bridge.py --p 113 &
python empirical/apex_recovery/e6_grokking_bridge.py --p 113 --control &
wait

# composite n=30 -- the rho_x / CBS Conjecture-5.8 trajectory + control
python empirical/apex_recovery/e6_grokking_bridge.py --p 30 &
python empirical/apex_recovery/e6_grokking_bridge.py --p 30 --control &
wait
```

Defaults match the grokking recipe: `--frac_train 0.3 --lr 1e-3 --wd 1.0
--steps 40000 --stop_after_grok 4000 --log_every 200`. If a regime does not grok
within 40000 steps, lower the modulus or nudge `--wd` / `--frac_train` and rerun
(cheap). Composite-n grokking is less documented than prime -- treat `--p 30` as
the experimental arm and `--p 12` as a fallback if 30 will not grok.

## Artifact Retrieval (before you delete anything)

Everything needed lives in `reports/` (JSONL with raw `u` + spectra, summary
JSON). Pull it to the local machine:

```bash
gcloud compute scp --recurse \
  cbs-e6-vm:~/AI-ConductorBlindSpot/empirical/apex_recovery/reports \
  ./e6_gcp_reports \
  --zone=us-east4-c --project=causalab-llama
```

## Teardown

```bash
gcloud compute instances delete cbs-e6-vm --zone=us-east4-c --project=causalab-llama --quiet
```

Because the JSONL carries the raw centered score `u` and embedding spectra per
step, **every diagnostic is recomputable offline** -- safe to delete the drive;
nothing of value is left on it.

## Offline Analysis (local, CPU, no torch)

```bash
# figure (needs matplotlib -> use `py` = 3.13):
py empirical/apex_recovery/e6_plot.py --p 113
py empirical/apex_recovery/e6_plot.py --p 30

# re-derive / correct any diagnostic from the logged raw u, no GPU rerun needed:
python empirical/apex_recovery/e6_diagnostics.py      # self-test / ladder check
```

## Decision Rule

Support for the two-register-boundary reading:

- on the ring task, **val_acc, D, and S rise in the same step window** (a sharp
  joint transition, not staggered);
- on **composite n**, **rho_x rises with them** (the cross-packet cubic emerges)
  -- this is the Conjecture-5.8 positive;
- the **control stays flat** in every register (no group structure to grok).

Falsification / null:

- the registers rise at clearly different times -> the single-boundary reading
  is wrong (the headline falsifier);
- rho_x moves on the control too -> the signal is not ring structure;
- no grok within budget on any modulus -> task/recipe unsuitable, retune.

## Honest Scope Notes

- The individual register metrics (embedding Fourier concentration, logit
  additivity) **overlap Nanda et al.'s mechanistic progress measures** -- we do
  not claim to have discovered them. The contribution is (1) identifying them as
  the order-2/3 boundary of CBS + the eigenfunction theory, (2) the **rho_x
  trajectory test of CBS's open Conjecture 5.8** with a matched control, and (3)
  the falsifiable co-emergence (ideally a quantitative lock, not mere
  co-occurrence).
- rho_x is **0 by construction on prime p** -- do not read the prime run's
  rho_x panel as a null result; run composite n for that question.
- The earlier laptop run reached only step ~1800 (pre-grok) before a hardware
  failure; this runbook exists to complete the trajectory on disposable cloud
  hardware. CUDA is confined to the VM.
