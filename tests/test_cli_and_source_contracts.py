from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def argparse_default(script_rel: str, option: str):
    tree = ast.parse(read_text(script_rel))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        option_names = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        if option not in option_names:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return ast.literal_eval(keyword.value)
    raise AssertionError(f"{script_rel} does not define {option}")


def test_demo_defaults_reproduce_checked_synthetic_report_shape() -> None:
    assert argparse_default("empirical/paper34_conductor_blindspot_demo.py", "--rings") == "6,8,12,18,30"
    assert argparse_default("empirical/paper34_conductor_blindspot_demo.py", "--include_strong") is True


def test_pythia_default_outputs_are_repo_root_safe() -> None:
    assert argparse_default("empirical/pythia_rho_x_sweep/sweep_n_axis.py", "--out") == (
        "empirical/pythia_rho_x_sweep/results_n_axis.json"
    )
    assert argparse_default("empirical/pythia_rho_x_sweep/sweep_d_axis.py", "--out") == (
        "empirical/pythia_rho_x_sweep/results_d_axis.json"
    )


def test_source_ledger_schema_unique_ids_and_status_values() -> None:
    ledger = json.loads(read_text("sources/verification_ledger.json"))
    entries = ledger["entries"]
    ids = [entry["id"] for entry in entries]

    assert ledger["schema_version"] == 1
    assert ledger["checked_as_of"] == "2026-05-22"
    assert "not treated as authoritative" in ledger["verification_method"]
    assert len(ids) == len(set(ids))

    for entry in entries:
        assert entry["primary_url"] == f"https://arxiv.org/abs/{entry['id']}"
        assert entry["source_status"] in {"peer_reviewed", "preprint", "under_review"}
        assert entry["claim_locations"]
        assert entry["claims_verified"]
        assert entry["reference_text"]


def test_manuscript_arxiv_ids_have_ledger_entries() -> None:
    manuscript = read_text("ConductorBlindSpot.md")
    cited_ids = set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", manuscript))
    ledger_ids = {entry["id"] for entry in json.loads(read_text("sources/verification_ledger.json"))["entries"]}

    assert cited_ids
    assert cited_ids <= ledger_ids


def test_dcd_status_language_is_softened() -> None:
    ledger = json.loads(read_text("sources/verification_ledger.json"))
    dcd = next(entry for entry in ledger["entries"] if entry["id"] == "2605.09129")
    assert dcd["source_status"] == "under_review"

    public_text = "\n".join(
        read_text(rel)
        for rel in (
            "README.md",
            "ConductorBlindSpot.md",
            "docs/source-verification.md",
            "docs/final_claims_assessment.md",
        )
    )
    assert "published frontier-LLM circuit-discovery study" not in public_text
    assert "published frontier-LLM circuit-discovery result" not in public_text
    assert "under review" in public_text.lower()


def test_claim_assessment_source_status_sensitive_rows_match_ledger() -> None:
    ledger = json.loads(read_text("sources/verification_ledger.json"))
    entries = {entry["id"]: entry for entry in ledger["entries"]}
    report = read_text("docs/final_claims_assessment.md")

    assert entries["2605.09129"]["source_status"] == "under_review"
    assert entries["2304.01373"]["source_status"] == "peer_reviewed"

    assert "arXiv:2605.09129" in report
    assert "under review" in report.lower()
    assert "arXiv:2304.01373" in report
    assert "Pythia checkpoint-suite" in report


def test_claim_assessment_arxiv_ids_are_in_source_ledger() -> None:
    ledger = json.loads(read_text("sources/verification_ledger.json"))
    ledger_ids = {entry["id"] for entry in ledger["entries"]}
    report_ids = set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", read_text("docs/final_claims_assessment.md")))

    assert report_ids
    assert report_ids <= ledger_ids


def test_pythia_claim_row_does_not_use_primary_source_as_diagnostic_validation() -> None:
    report = read_text("docs/final_claims_assessment.md")
    pythia_rows = [
        line
        for line in report.splitlines()
        if line.startswith("| The Pythia calendar-months sweep")
    ]
    assert len(pythia_rows) == 1
    row = pythia_rows[0]

    assert "arXiv:2304.01373" in row
    assert "checkpoint-suite context" in row
    assert "does not validate this repository's\n  calendar-months diagnostic result" in report
    assert "diagnostic-confirming" not in row
