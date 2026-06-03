# Poole–OTG Bridge — exact certificates (P1–P6)

> **The note this certifies:** [`Poole-OTG-Bridge-Theorem.md`](Poole-OTG-Bridge-Theorem.md)
> — the five bridge theorems, their proofs, the status ledger, and how they meet
> Rooke Poole's own `KNOWN_GAPS.md`. **P6** runs the whole machinery on his actual
> [B5-7/S5-9 engine](https://github.com/rookepoole/SVP-OTG-Poole-Manifold-tests).

Deterministic, exact-arithmetic certificates for the five theorems of the
**Poole–OTG Bridge Theorem** note. They settle, in machine-precision linear
algebra (no machine learning, no GPU, no network), the structural place where a
claim that "Poole dynamics is quantum gravity" would have to live: a nontrivial
coherent lift above a decoherent floor, a stable closure/leakage certificate, and
relabelling-invariant (dimensionless) refinement invariants.

```
python empirical/poole_bridge/run_all.py     # one-command pass/fail gate, ~seconds
```

| # | theorem | what is certified | headline result |
|---|---------|-------------------|-----------------|
| P1 | **A** decoherent channel lift | `A_x=|F(x)><x|` gives a CPTP channel; Choi ⪰ 0; the diagonal face is the rule; every coherence is annihilated | all residuals `0`; Choi spectrum `{0,1}`; superposition off-diagonal `0.5 → 0` |
| P2 | **B** reversible dilation + **B1** | `U_F|x,y>=|x,y⊕F(x)>` unitary; traced marginal = pushforward; same-space unitary ⇔ injective | dilation exact; the 2×2×1 rule is non-injective (image rank `4/16`), witnessed by a collision |
| P3 | **C** closure / leakage | on the representative rule: which sectors close (induced `G`) vs leak (witness pair) | id / basin / **translation-orbit** close; **density / cell / window leak**, each with a decoded before-identical / after-distinct witness |
| P4 | **D** current audit | `S/A/J` split at **general** `π`; detailed balance ⇔ `A=0=J`; `J` divergence-free; arrow in noisy Poole data | identities `~1e-16` at non-uniform `π`; ring arrow matches E9 closed form; coarse-Poole current dim `b₁` |
| P5 | **E** dimensionless firewall | every reported invariant is relabelling-invariant; a labelling-dependent control moves | invariants drift `0` over 200 relabellings; control takes 16 / 109 distinct values |
| P6 | **A–E on the REAL rule** (license-gated) | Rooke's B5-7/S5-9 engine; his unit tests reproduced; closure + succession-flux audited. *Engine not shipped here — see "On Rooke's engine".* | vacuum/overpopulation pass; effective `B{5,6}/S{5,6,7,8,9}`; `id`/basin/orbit close, density/cell leak; current `‖A‖=0.76`, **Φ shown arrow-blind**, `b₁=114` |

## Files

- `poole_world.py` — the shared toolkit. **Layer A** is the rule-agnostic
  finite-dynamics machinery the *theorems* use (Kraus channel, Choi, XOR
  dilation, closure test, `L²(π)` current split). **Layer B** is a concrete,
  representative Poole rule (for the rule-agnostic demos P1–P5).
- `otg_rule.py` — **not included here (license-gated).** A torch-free transcription
  of Rooke's B5-7/S5-9 engine; his repository is All-Rights-Reserved, so the port is
  withheld pending his written permission (see "On Rooke's engine" below).
- `p1_…`–`p5_…` — one self-contained certificate per theorem (representative rule).
- `p6_otg_real_rule.py` — the same machinery on **the real rule** (Obligation 1);
  runs only where `otg_rule.py` is supplied, and is skipped by the public gate.
- `run_all.py` — runs the suite and gates on the JSON outputs; P6 auto-skips if the
  license-gated engine is absent, so P1–P5 are the public gate.
- `reports/*.json` — machine-readable results (the source of truth).

## On Rooke's engine (license)

Rooke Poole's repository — `github.com/rookepoole/SVP-OTG-Poole-Manifold-tests` —
is **All Rights Reserved** (his `LICENSE`): reading, running for research, and
citing with attribution are permitted, but redistributing his code "as part of
another project" requires his written permission. Accordingly, **we do not ship
`otg_rule.py`** (our transcription of his engine) in this repo. P6's *results* —
our own academic output from running the rule — are reported in the note and in
`reports/p6_otg_real_rule.json`, attributed to **Rooke Alan Poole, "The Poole
Manifold" (2026)**. To reproduce P6, obtain his engine and supply a compatible
`otg_rule.py` locally. (His `CITATION.cff` lists MIT, which contradicts the
`LICENSE`; until he resolves that, we treat the stricter `LICENSE` as governing.)

## The rule: Obligation 1 discharged (read this)

The note's *Obligation 1* ("freeze the actual rule") is now **discharged**.
`otg_rule.py` is a torch-free transcription of Rooke's canonical `PooleEngine.step`
(Delta RPM Protocol): Moore-26 neighbourhood, circular boundary, synchronous,
`total_phi = m + Σ_p α·exp(−(m−p)²/σ²)`, birth `5 ≤ total_phi ≤ 7`, survive
`5 ≤ total_phi ≤ 9`. P6 reproduces his own unit tests bit-for-bit and confirms the
effective integer rule `B={5,6}`, `S={5,6,7,8,9}` (the sharp resonance ejects `m=7`
from birth: `7 + 0.35 = 7.35 > 7`).

Epistemic tiers:

- **Theorems A–E and the closure/current/firewall *tests*** are rule-agnostic,
  proved and certified — they hold for *any* deterministic finite `F`. `[P]`
- **P6's verdicts on the real rule** (closure decided; succession flux arrow-blind;
  current `‖A‖>0`) are claims about Rooke's *actual* substrate, on the enumerable
  tori carrying his exact kernel. `[P]`
- **P1–P5's witness pairs** belong to the simpler representative rule, kept because
  it makes the theorems' independence from the rule visible. `[representative]`

One honest note carried into P6: Rooke's *succession flux* `Φ ≈ 0.4002` is his
scalar activity/occupancy, **not** a probability current. P6 measures the
equilibrium density at `0.3997` (matching `Φ`) and shows that, being a symmetric
scalar, `Φ` is blind to the genuine antisymmetric current `A` (the arrow) — which
is the object the current audit recovers.

## Relationship to the rest of the suite

P4 is the **general-`π` companion** to `apex_recovery/`'s **E9** (drift ring, uniform
`π`) and **E10** (non-normal conveyor): the same symmetric/antisymmetric
decomposition that powers the *Irreversible Blind Spot*, here generalised to
arbitrary positive stationary law and applied to Poole-coarsened transition data.
The cycle-rank readout in P4 ties to **E12** (dimension of the arrow = `b₁`).
