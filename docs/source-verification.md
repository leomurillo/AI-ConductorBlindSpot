# Source verification policy

The source ledger is `sources/verification_ledger.json`. It records arXiv IDs,
canonical URLs, source status, checked date, claim locations, and the exact
claim each source is used to support.

## Authority rule

Primary sources are authoritative. For manuscript-supporting citations, use:

1. The direct arXiv page or publisher page.
2. The paper text when the claim depends on a result rather than metadata.
3. Local checked artifacts when the claim is about this repository's empirical
   outputs.

Perplexity may be used for discovery, contradiction hunting, or second-opinion
triage. It is not evidence by itself, and a Perplexity answer must not override
the direct primary source or local artifacts.

## Required ledger fields

Each entry must contain:

- `id`: arXiv identifier without the `arXiv:` prefix.
- `primary_url`: canonical `https://arxiv.org/abs/<id>` URL for arXiv sources.
- `source_status`: one of `peer_reviewed`, `preprint`, or `under_review`.
- `claim_locations`: manuscript or documentation locations that rely on the
  source.
- `claims_verified`: concise statements of the verified claim.
- `reference_text`: the reference line or other bibliographic support.

The top-level `checked_as_of` date is currently `2026-05-22`.

## Status-sensitive claims

The DCD paper, arXiv:2605.09129, is recorded as `under_review`. Repository
language should therefore describe it as an under-review preprint or motivating
concordant evidence, not as published circuit-discovery evidence.

Pythia, arXiv:2304.01373, is recorded as peer-reviewed and supports the fact
that the Pythia checkpoints are a public checkpoint suite for analyzing
language models across training and scaling.

## Maintenance checklist

When adding or changing a manuscript citation:

1. Add or update the ledger entry before strengthening any manuscript claim.
2. Prefer softer manuscript wording when the source is a preprint or
   under-review work.
3. Record the exact manuscript or documentation location that relies on the
   source.
4. Re-run `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q`.
