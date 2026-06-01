# Final claims assessment

Assessment date: 2026-05-22.

This report assesses the original manuscript's claims against the current
repository state: manuscript source, checked JSON/Markdown/PNG artifacts,
source-verification ledger, build outputs, deterministic tests, and selected
primary-source checks. It does not treat Perplexity output as evidence.

## Evidence tiers

| Support | Tier | Meaning |
|---|---|---|
| 🟢 | Strong theorem/exact support | Supported by manuscript derivation, deterministic integer enumeration, or both. |
| 🟡 | Bounded empirical support | Supported by checked artifacts for the stated configuration, seeds, checkpoints, and controls. |
| 🟠 | Context or conjectural support | Useful as positioning, concordance, or a pre-specified follow-up, but not direct evidence for the claim. |
| 🔴 | Not supported as originally phrased | The original wording was too strong, contradicted by checked artifacts, or must remain softened. |

## Claim inventory method

The audit scanned the manuscript abstract, introduction, contribution list,
limitations, conclusion, artifact documentation, checked summaries, and the
current source ledger. It explicitly looked for claim-bearing phrases and
sections: `We prove`, `We verify`, `We derive`, `We report`, `We propose`,
`Conjecture`, `Branch B`, and explicit limitation/non-claim language.

Each row is anchored to a short quote from `ConductorBlindSpot.md` wherever the
claim appears in the paper. The quote is not a substitute for evidence; it is a
traceability guard so the assessment addresses the authors' own words before it
assigns support level, scope boundary, required wording, and failure mode.
Perplexity was used only as a second-opinion check on claims-audit structure,
not as evidence for any support rating.

Mathematical claims are marked 🟢 only when the report can point to a theorem,
proposition, corollary, or exact integer artifact. Empirical claims are marked
🟡 only when checked JSON/Markdown artifacts define the configuration tightly
enough to reproduce the reported reading. External papers never upgrade a
repository claim above 🟠 unless the claim is only bibliographic or source-status
context. Source-status-sensitive claims must match
`sources/verification_ledger.json` and the direct primary source.

Primary sources rechecked for the highest-risk rows:

- arXiv:2605.09129 is a 2026 arXiv preprint whose comments list it as under
  review; it can support DCD as motivating concordant evidence, not as a
  published or diagnostic-confirming result.
- arXiv:2304.01373 supports Pythia as a public checkpoint suite for studying
  language-model training dynamics; it does not validate this repository's
  calendar-months diagnostic result.
- arXiv:2312.06581 and arXiv:1412.1193 were checked as representative
  ledger-backed context sources for grokking/cosets and natural-gradient
  background, not as direct evidence for this repository's empirical claims.

## Claim assessment

| Claim | Paper quote | Support | Evidence checked | Scope boundary | Required wording | Failure mode if overstated |
|---|---|---:|---|---|---|---|
| Quadratic curvature at the uniform cyclic categorical head is conductor-packet block-diagonal in the additive-character basis. | "the Fisher information form $g_{p_*}$ is diagonal" | 🟢 | Theorem 3.5, Lemmas 3.1 and 3.4, and Section 2.3 scope language in `ConductorBlindSpot.md`. | Composite cyclic label set at the maximum-entropy point on the output tangent space. | State as an exact local categorical-head result at p star. | Becomes a false universal optimizer claim if applied to arbitrary trained networks or off-centroid trajectories. |
| The Amari-Chentsov cubic carries cross-packet selection-rule triples invisible to the quadratic class. | "outside the representational capacity of $g_{p_*}$" | 🟢 | Theorem 3.5 plus `empirical/paper34_cyclic_exact_certificate.py` and `empirical/reports/paper34_cyclic_exact_certificate.json`. | Cross-packet means conductors not all equal; pairwise-distinct conductors are a stronger Appendix A condition. | State as cubic content outside the representational capacity of the quadratic curvature form at p star. | Overstates if rewritten as task relevance or performance cost. |
| The five-ring cyclic ladder has cross-packet counts 18, 42, 108, 252, and 774. | "We verify the result computationally on the ladder" | 🟢 | Exact cyclic JSON rows for n equals 6, 8, 12, 18, 30 with Python integer arithmetic only. | The checked ladder is finite and small. | Use these exact counts only for the five checked rings. | Overstates if used as evidence for all moduli or large-scale rings. |
| The non-cyclic finite abelian extension holds on five named candidate groups. | "verifies it on five non-cyclic candidates" | 🟢 | `empirical/paper34_C4_noncyclic_certificate.py`, five `paper34_C4_*.json` files, and `paper34_C4_noncyclic_summary.md`. | Verified candidate set, not an exhaustive generated census of all finite abelian groups. | Say five non-cyclic candidates are verified. | Overstates if phrased as a broad empirical survey or all-group computation. |
| Natural-gradient and operator-side lifts preserve the packet-null structure at p star where the stated hypotheses apply. | "to the extent named in the four-tier scope" | 🟢 | Corollaries 3.8, 3.9, and 3.10 plus the four-tier method-scope paragraph. | Output-side exact natural gradient and specified lifts; deeper-architecture Jacobian preservation remains conditional. | Tie each named method to its tier and condition. | Overstates if every K-FAC, Adam, NTK, or deep-network instance is treated as identical to g at p star. |
| The off-centroid Fisher expansion has a leading Fourier selection rule and an all-orders convolution form. | "The expansion extends to all orders in $h$" | 🟢 | Lemma 5.1, Theorem 5.2, Theorem 5.6, and Appendix D. | Analytic expansion along a specified displacement from p star. | State as deterministic analytic geometry for fixed displacements. | Overstates if made into a claim about what SGD trajectories actually do. |
| Conjecture 5.8 predicts measurable off-centroid persistence on ring-structured trajectories. | "is stated as a conjecture (Conjecture 5.8) rather than a theorem" | 🟠 | Conjecture 5.8 and Section 8 pre-specification. | Explicit conjecture, not a theorem. Tested head-only interventions did not support the operational steering corollary. | Call it conjectural or an open follow-up. | Becomes false if described as proved or established by the diagnostic demo. |
| Definition 6.1 defines a retraining-free diagnostic rho_x on p-star-anchored selection-rule triples. | "The diagnostic makes no performance claim" | 🟢 | Definition 6.1 and synthetic/Pythia artifact code paths. | Measurement definition, not a performance theorem. | Say the diagnostic converts the certificate into a measurable quantity. | Overstates if claimed to prove task relevance or optimizer failure. |
| The Section 8 experiment is internally pre-specified with a negative-control branch. | "Both outcome branches are pre-claimed" | 🟢 | Section 8 protocol language and the checked Layer-3 Branch B summaries. | Internal manuscript/repository pre-specification, not an external registry timestamp. | Say internally pre-specified protocol with both branches reportable. | Overstates if called independent preregistration or external replication. |
| The synthetic modular-addition MLP demo separates ring-structured and structureless controls in the checked setting. | "identifying three regimes — *structural degeneracy*" | 🟡 | `paper34_demo_summary.md`, per-ring curve/certificate JSONs, and strong controls where present. | Checked architecture, seed, budget, rings 6, 8, 12, 18, 30; strongest clean positive separation is n equals 12 pre-grokking. | Say bounded empirical support for the checked synthetic setup. | Overstates if presented as universal behavior of modular arithmetic learning. |
| The Pythia calendar-months sweep identifies an off-trajectory diagnostic boundary. | "We report a first-run application to the Pythia suite" | 🟡 | `results_n_axis.json`, `results_d_axis.json`, `RESULTS.md`, and arXiv:2304.01373 for Pythia checkpoint-suite context. Checked artifacts contain 5 N-axis rows and 7 D-axis rows; per-example ring rho_x spans about 0.9601 to 0.9831. | Calendar-month conditional head on selected Pythia checkpoints; generic LM head is not a modular-arithmetic training trajectory. | Say first-run diagnostic-boundary observation. | Overstates if framed as a general claim about language-model calendar reasoning or Conjecture 5.8. |
| Historical checked Pythia rows have exact HuggingFace commit provenance. | "All checked-in run artifacts in `empirical/reports/` are the exact outputs cited" | 🔴 | Pythia scripts now capture requested/resolved revisions when available, but checked JSON predates resolved commit capture. | Legacy checked rows are local artifacts with requested revision fields but no original resolved commit hashes. | Say future runs improve provenance; old rows lack original resolved model commits. | False reproducibility claim if called exact remote-model provenance. |
| Three head-only cubic-aware interventions steer ring-task acquisition. | "Three head-only cubic-aware interventions on $n = 30$ all return Branch B" | 🔴 | Layer-3 JSONs and summaries for newton, neumann, and within-packet forms, seeds 0, 1, 2. Mean interactions are approximately +0.003, -0.011, and -0.004. | Tested n equals 30, alpha equals 0.1, Adam baseline, head-side correction forms only. | Say all three tested forms returned Branch B and the certificate is measurement evidence, not steering evidence. | Directly contradicted if described as successful steering. |
| Branch B refutes only the operational steering cell, not the theorem-level certificate. | "the specific *operational* claim" | 🟢 | Section 8.6, Section 9 limitations, and Layer-3 summaries. | Refutation is limited to the tested interventions and operating point. | Say Branch B leaves Theorem 3.5 and the diagnostic definition intact. | Overstates if treated as refuting the finite certificate or all cubic-aware follow-ups. |
| The head-side lift applies the blind-spot statement to linear gradient-based edge-attribution inputs. | "lifts the static block-diagonality from the curvature object" | 🟢 | Proposition 3.11, Corollary 3.12, and Section 2.3 tier T4. | At p star and at the head-side input to linear or first-order attribution scores. | Say head-side first-order attribution input has the stated blind spot. | Overstates if applied to full nonlinear circuit-discovery pipelines without the Section 6.7 artifact test. |
| The Section 6.7 attribution-side diagnostic is implemented on released DCD artifacts. | "the present paper does not run the diagnostic on those artifacts" | 🔴 | Manuscript states this as pre-specified future work; no checked DCD diagnostic output exists in this repository. | Not run in the checked artifact set. | Say pre-specified diagnostic test pending released artifacts. | False if described as completed external validation. |
| DCD-style external results provide concordant context for the attribution-side concern. | "motivating concordant evidence rather than as a confirmed instance" | 🟠 | `sources/verification_ledger.json` and arXiv:2605.09129, whose comments list under review. | External under-review preprint; not this repository's diagnostic. | Use motivating concordant evidence only. | Overstates if called direct empirical support for rho_x or the certificate. |
| The DCD mechanism-mixing discussion proposes a candidate structural reading, not a confirmed instance. | "We propose this as a candidate structural reading rather than a confirmed instance" | 🟠 | Section 7.7 discussion, source ledger entry for arXiv:2605.09129, and the absence of checked Section 6.7 DCD diagnostic artifacts. | Interpretive bridge only; alternative non-conductor explanations remain possible. | Say candidate structural reading pending the pre-specified diagnostic test. | Overstates if described as confirmed DCD validation of the certificate. |
| The DCD paper can be used as definitive published circuit-discovery evidence. | "under-review large-scale circuit-discovery preprint" | 🔴 | Ledger entry for arXiv:2605.09129 and direct arXiv page both mark source status as under review. | No publication-status support as of 2026-05-22. | Say under-review preprint. | False if described as published evidence. |
| Recent optimization and spectral-feature papers confirm the conductor blind spot. | "concordant with the model-class breadth of the claim, pending direct tests" | 🟠 | Ledger-backed context citations and Section 7 discussion. | Positioning and concordance only; they are not tests of this certificate. | Say compatible or complementary external context. | Overstates if used as independent proof of the manuscript's certificate. |
| The paper proves the missed coupling is task-relevant or explains grokking. | "It does not prove the missed coupling is *task-relevant*" | 🔴 | Section 9 explicitly says the certificate does not prove task relevance and makes no claim that it explains grokking. | Task relevance remains empirical and partly refuted for tested steering cells. | Say the paper emits a theorem, a measurement procedure, and a pre-specified protocol. | False if written as a full grokking explanation or performance theorem. |
| The manuscript endorses a benchmark number, learning rate, or tuning recipe. | "it emits no benchmark number" | 🔴 | Section 9 says the paper emits no benchmark number and does not endorse specific learning-rate or baseline tuning. | No benchmark claim is supported. | Keep benchmarks scoped to checked artifacts only. | False if turned into a benchmark recommendation. |
