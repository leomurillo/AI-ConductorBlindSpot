# Zenodo — submitting version 2 of the Conductor Blind Spot preprint

This is an operator's note for posting a **new version** of the existing Zenodo
record. It does not change the manuscript.

- **Concept DOI (all versions):** [10.5281/zenodo.20330864](https://doi.org/10.5281/zenodo.20330864)
- **Record:** <https://zenodo.org/records/20330864>
- **Artifact to upload:** `ConductorBlindSpot.pdf` (rebuilt from the current source;
  66 pp). The companion code is the GitHub repository, linked from the
  manuscript's *Code and data availability* section, so a code zip is optional —
  upload one only if you want a frozen snapshot in the record.

## Version notes (paste into Zenodo's "What's new" field)

> **Version 2.** Substantial revision. Expanded the information-geometric
> substrate and related work: the manuscript now situates the order-2/order-3
> boundary within the simplicial gauge program (the ל0–ל7 series) and against the
> linear-identifiability boundary of self-supervised representation learning
> (§7.11–§7.12). Added the linear-time σ_PI packet diagnostic and the σ_H
> harmonic-skew scalar witness (§6.1), and the replicator (β-flow) structural
> analysis showing the head-only cross-packet content is flow-invariant
> (Remark 5.9 / Conjecture 5.8′), which reframes the §8.6 head-only null as the
> predicted signature rather than an anomaly. Wired every empirical artifact to
> the public GitHub repository (new *Code and data availability* section with
> per-script links) and filled in the Zenodo archival DOI. References expanded
> to 52; numerous proof expansions and editorial refinements throughout. No
> change to the main theorem (the finite character-orthogonality certificate) or
> its computational verification.

## Manual procedure (always works)

1. Sign in to Zenodo and open the record <https://zenodo.org/records/20330864>.
2. Click **New version** (top right). Zenodo clones the metadata and reserves a
   new version DOI under the same concept DOI.
3. **Remove** the old PDF and **upload** the freshly built `ConductorBlindSpot.pdf`.
   (Optional: also attach a code snapshot — GitHub → *Code* → *Download ZIP*, or a
   release tarball.)
4. Set **Version** to `v2` (or `2.0`) and paste the version notes above into the
   *What's new in this version?* field.
5. Confirm metadata: title, author (Leonardo Murillo Montero), keywords,
   license; under *Related identifiers* keep/point to the GitHub URL
   <https://github.com/leomurillo/AI-ConductorBlindSpot>.
6. **Publish.** The new version DOI is minted; the concept DOI 10.5281/zenodo.20330864
   continues to resolve to the latest version.

## Optional automated path (only if this repo is connected to Zenodo)

If the GitHub repository is linked to Zenodo via the GitHub–Zenodo webhook (check
Zenodo → *GitHub* → the repo toggle), then publishing a **GitHub release** mints
the new version automatically. In that case add a root `.zenodo.json` describing
the record and cut a tagged release (e.g. `v2`). Do **not** use this path if the
existing record was deposited manually — a release would create a *separate*
record with its own concept DOI rather than a new version of 20330864.
