# AI-ConductorBlindSpot documentation

This folder collects the repository-level operating documentation for the
Conductor Blind Spot manuscript and its checked artifacts.

## Documents

- [Reproducibility](reproducibility.md): exact commands for every checked
  artifact, expected output paths, and which commands overwrite tracked files.
- [Validation](validation.md): local checks used to verify syntax, tests,
  exact certificates, generated artifacts, source-ledger coverage, and PDF
  consistency.
- [Source verification](source-verification.md): policy for the
  `sources/verification_ledger.json` ledger, including the rule that primary
  sources are authoritative and Perplexity is only a discovery aid.
- [Final claims assessment](final_claims_assessment.md): evidence-tiered
  assessment of the original manuscript claims against the current checked
  repository state.

## Evidence hierarchy

The repository uses this evidence order when documentation, manuscript prose,
and generated artifacts disagree:

1. Deterministic source code and checked JSON artifacts.
2. Regenerated summaries produced from those JSON artifacts.
3. Direct primary sources, especially arXiv pages recorded in the source
   ledger.
4. Secondary discovery tools, including Perplexity, only as prompts for
   primary-source verification.

Do not cite Perplexity output as independent support for manuscript claims.
