from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SUPPORT_EMOJIS = {"🟢", "🟡", "🟠", "🔴"}
CLAIM_COLUMNS = (
    "claim",
    "quote",
    "support",
    "evidence",
    "scope",
    "wording",
    "failure_mode",
)


def read_text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _delink(text: str) -> str:
    """Strip clickable internal-link markup so anchor quotes that became
    \\hyperref / \\hyperlink targets in the T-series catch-up pass still match
    (e.g. a quote containing "(Conjecture 5.8)" now rendered as
    "(\\hyperlink{stmt:5-8}{Conjecture~5.8})")."""
    text = re.sub(r"\\hyperref\[[^\]]*\]\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\hyperlink\{[^}]*\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\hypertarget\{[^}]*\}\{\}", "", text)
    return text.replace("~", " ")


def final_claim_rows() -> list[dict[str, str]]:
    report = read_text("docs/final_claims_assessment.md")
    rows: list[dict[str, str]] = []
    in_table = False
    for line in report.splitlines():
        if line.startswith("| Claim | Paper quote | Support | Evidence checked | Scope boundary |"):
            in_table = True
            continue
        if not in_table or line.startswith("|---"):
            continue
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(CLAIM_COLUMNS):
            continue
        rows.append(dict(zip(CLAIM_COLUMNS, cells)))
    return rows


def test_docs_folder_has_required_public_documents() -> None:
    required = {
        "README.md",
        "reproducibility.md",
        "validation.md",
        "source-verification.md",
        "final_claims_assessment.md",
    }
    assert required <= {path.name for path in DOCS.glob("*.md")}


def test_readme_and_build_docs_link_to_docs_suite() -> None:
    readme = read_text("README.md")
    build = read_text("BUILD.md")

    for rel in (
        "docs/README.md",
        "docs/reproducibility.md",
        "docs/validation.md",
        "docs/final_claims_assessment.md",
    ):
        assert rel in readme
    assert "docs/validation.md" in build


def test_docs_index_links_all_required_documents() -> None:
    index = read_text("docs/README.md")
    for name in (
        "reproducibility.md",
        "validation.md",
        "source-verification.md",
        "final_claims_assessment.md",
    ):
        assert f"]({name})" in index


def test_reproducibility_doc_lists_exact_checked_commands() -> None:
    doc = read_text("docs/reproducibility.md")

    required_commands = [
        "python empirical/paper34_C4_noncyclic_certificate.py",
        "python empirical/paper34_cyclic_exact_certificate.py",
        "python empirical/paper34_conductor_blindspot_demo.py --rings 6,8,12,18,30 --include_strong",
        "python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form newton --tag adam_alpha0p1",
        "python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form neumann --tag adam_neumann_alpha0p1",
        "python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form within_packet --tag adam_within_packet_alpha0p1",
        "python empirical/pythia_rho_x_sweep/sweep_n_axis.py --out empirical/pythia_rho_x_sweep/results_n_axis.json",
        "python empirical/pythia_rho_x_sweep/sweep_d_axis.py --out empirical/pythia_rho_x_sweep/results_d_axis.json",
    ]
    for command in required_commands:
        assert command in doc

    assert "overwrites" in doc
    assert "legacy checked rows" in doc


def test_validation_doc_records_required_checks_and_expensive_boundary() -> None:
    doc = read_text("docs/validation.md")

    for command in (
        "git diff --check",
        "bash -n build.sh",
        "PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q",
        "python3 empirical/paper34_cyclic_exact_certificate.py --out /tmp/paper34_cyclic_exact_certificate.json",
        "bash build.sh --clean",
        "cmp -s build/paper.pdf ConductorBlindSpot.pdf",
    ):
        assert command in doc

    assert "not part of the default offline test suite" in doc


def test_source_verification_doc_sets_primary_source_policy() -> None:
    doc = read_text("docs/source-verification.md")

    assert "sources/verification_ledger.json" in doc
    assert "Perplexity may be used for discovery" in doc
    assert "not evidence by itself" in doc
    assert "2605.09129" in doc
    assert "`under_review`" in doc
    assert "not as published circuit-discovery evidence" in doc


def test_final_claims_assessment_uses_required_evidence_tiers() -> None:
    report = read_text("docs/final_claims_assessment.md")

    for emoji, tier in (
        ("🟢", "Strong theorem/exact support"),
        ("🟡", "Bounded empirical support"),
        ("🟠", "Context or conjectural support"),
        ("🔴", "Not supported as originally phrased"),
    ):
        assert f"| {emoji} | {tier} |" in report

    for rule in (
        "Mathematical claims are marked 🟢 only",
        "Empirical claims are marked\n🟡 only",
        "External papers never upgrade a\nrepository claim above 🟠",
        "Source-status-sensitive claims must match\n`sources/verification_ledger.json`",
    ):
        assert rule in report

    assert "It does not treat Perplexity output as evidence." in report
    assert "Perplexity was used only as a second-opinion check" in report
    assert "all three tested forms returned Branch B" in report
    assert "old rows lack original resolved model commits" in report


def test_final_claim_table_has_required_schema_and_support_emoji() -> None:
    rows = final_claim_rows()
    assert len(rows) >= 19

    for row in rows:
        assert row["support"] in SUPPORT_EMOJIS
        for column in CLAIM_COLUMNS:
            assert row[column], row


def test_final_claim_rows_are_anchored_to_paper_quotes() -> None:
    manuscript = _delink(read_text("ConductorBlindSpot.md"))
    for row in final_claim_rows():
        quote = row["quote"]
        assert quote.startswith('"') and quote.endswith('"'), row
        assert "No direct" not in quote
        assert len(quote.strip('"').split()) >= 4, row
        assert quote.strip('"') in manuscript, row


def test_high_risk_claims_have_expected_support_levels() -> None:
    rows = final_claim_rows()

    def support_for(fragment: str) -> str:
        matches = [row["support"] for row in rows if fragment in row["claim"]]
        assert len(matches) == 1, fragment
        return matches[0]

    assert support_for("Quadratic curvature") == "🟢"
    assert support_for("off-centroid Fisher expansion") == "🟢"
    assert support_for("five-ring cyclic ladder") == "🟢"
    assert support_for("synthetic modular-addition MLP") == "🟡"
    assert support_for("Pythia calendar-months sweep") == "🟡"
    assert support_for("Historical checked Pythia rows") == "🔴"
    assert support_for("head-only cubic-aware interventions steer") == "🔴"
    assert support_for("Section 6.7 attribution-side diagnostic") == "🔴"
    assert support_for("DCD-style external results") == "🟠"
    assert support_for("definitive published circuit-discovery evidence") == "🔴"
    assert support_for("explains grokking") == "🔴"


def test_final_claim_report_represents_major_manuscript_claim_classes() -> None:
    report = read_text("docs/final_claims_assessment.md")

    for phrase in (
        "`We prove`",
        "`We verify`",
        "`We derive`",
        "`We report`",
        "`We propose`",
        "`Conjecture`",
        "`Branch B`",
        "explicit limitation/non-claim language",
    ):
        assert phrase in report

    for claim_fragment in (
        "Natural-gradient and operator-side lifts",
        "Definition 6.1 defines a retraining-free diagnostic",
        "Branch B refutes only the operational steering cell",
        "The paper proves the missed coupling is task-relevant",
        "The manuscript endorses a benchmark number",
    ):
        assert any(claim_fragment in row["claim"] for row in final_claim_rows())


def test_final_claim_rows_have_scope_wording_and_failure_modes() -> None:
    for row in final_claim_rows():
        assert row["scope"] not in {"-", "N/A"}
        assert row["wording"].startswith(("State", "Say", "Use", "Call", "Tie", "Keep"))
        assert row["failure_mode"].startswith(("Becomes", "Overstates", "False", "Directly"))
