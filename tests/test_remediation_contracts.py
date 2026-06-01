from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "empirical" / "reports"
LAYER3_TAGS = {
    "adam_alpha0p1": "newton",
    "adam_neumann_alpha0p1": "neumann",
    "adam_within_packet_alpha0p1": "within_packet",
}


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def argparse_default(script: Path, option: str) -> str:
    tree = ast.parse(script.read_text(encoding="utf-8"))
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
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value
    raise AssertionError(f"{script} does not define {option}")


def markdown_seed_rows(summary: str) -> list[int]:
    seeds: list[int] = []
    for line in summary.splitlines():
        match = re.match(r"^\|\s*30\s*\|\s*(\d+)\s*\|", line)
        if match:
            seeds.append(int(match.group(1)))
    return seeds


def layer3_results_for_tag(tag: str) -> list[dict]:
    paths = sorted(REPORTS.glob(f"paper34_layer3_n30_seed*_{tag}.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def test_readme_lists_exact_checked_reproduction_commands() -> None:
    readme = read_text("README.md")

    assert "python empirical/paper34_conductor_blindspot_demo.py --rings 6,8,12,18,30 --include_strong" in readme
    assert "python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form newton --tag adam_alpha0p1" in readme
    assert "python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form neumann --tag adam_neumann_alpha0p1" in readme
    assert "python empirical/paper34_layer3_cubic_experiment.py --steps 15000 --seeds 0,1,2 --baseline_mode adam --cubic_form within_packet --tag adam_within_packet_alpha0p1" in readme
    assert "--out empirical/pythia_rho_x_sweep/results_n_axis.json" in readme
    assert "--out empirical/pythia_rho_x_sweep/results_d_axis.json" in readme
    assert "python empirical/paper34_cyclic_exact_certificate.py" in readme


def test_build_docs_use_bash_and_scripts_avoid_environment_specific_fonts() -> None:
    readme = read_text("README.md")
    build_doc = read_text("BUILD.md")
    build_sh = read_text("build.sh")
    build_ps1 = read_text("build.ps1")

    assert "bash build.sh" in readme
    assert "bash build.sh --clean" in build_doc
    assert "./build.sh" not in readme
    for forbidden in ("mainfont=Latin Modern", "mathfont=Latin Modern", "monofont=Latin Modern"):
        assert forbidden not in build_sh
        assert forbidden not in build_ps1


def test_pythia_sweep_defaults_are_repo_root_paths() -> None:
    assert argparse_default(ROOT / "empirical/pythia_rho_x_sweep/sweep_n_axis.py", "--out") == (
        "empirical/pythia_rho_x_sweep/results_n_axis.json"
    )
    assert argparse_default(ROOT / "empirical/pythia_rho_x_sweep/sweep_d_axis.py", "--out") == (
        "empirical/pythia_rho_x_sweep/results_d_axis.json"
    )


def test_pythia_scripts_record_revision_provenance() -> None:
    for rel in (
        "empirical/pythia_rho_x_sweep/sweep_n_axis.py",
        "empirical/pythia_rho_x_sweep/sweep_d_axis.py",
        "empirical/pythia_rho_x_sweep/run_diagnostic.py",
        "empirical/pythia_rho_x_sweep/tokenizer_probe.py",
        "empirical/pythia_rho_x_sweep/smoke_test.py",
    ):
        text = read_text(rel)
        assert "--model-revision" in text
        assert "revision=" in text

    for rel in (
        "empirical/pythia_rho_x_sweep/results_n_axis.json",
        "empirical/pythia_rho_x_sweep/results_d_axis.json",
    ):
        rows = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        assert rows
        for row in rows:
            assert row["model_revision_requested"]
            assert "model_revision_resolved" in row


def test_layer3_summaries_include_all_seed_jsons_and_aggregate_rows() -> None:
    for tag in LAYER3_TAGS:
        results = layer3_results_for_tag(tag)
        assert [r["seed"] for r in results] == [0, 1, 2]
        interactions = [r["summary"]["interaction"] for r in results]
        mean_interaction = sum(interactions) / len(interactions)
        std_interaction = math.sqrt(
            sum((x - mean_interaction) ** 2 for x in interactions) / (len(interactions) - 1)
        )

        summary_path = REPORTS / f"paper34_layer3_summary_{tag}.md"
        summary = summary_path.read_text(encoding="utf-8")
        assert markdown_seed_rows(summary) == [0, 1, 2]
        assert "## Aggregate summary" in summary
        assert f"**{mean_interaction:+.3f}**" in summary
        assert f"{std_interaction:.3f}" in summary
        assert "Head-only exact-Fisher variant" not in summary
        assert "Pre-specified branches" in summary


def test_exact_cyclic_certificate_script_and_checked_output_match_ladder(tmp_path: Path) -> None:
    script = ROOT / "empirical" / "paper34_cyclic_exact_certificate.py"
    source = script.read_text(encoding="utf-8")
    assert "numpy" not in source
    assert "float(" not in source

    out_path = tmp_path / "cyclic_certificate.json"
    subprocess.run(
        [sys.executable, str(script), "--out", str(out_path)],
        cwd=ROOT,
        check=True,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    generated = json.loads(out_path.read_text(encoding="utf-8"))
    checked = json.loads((REPORTS / "paper34_cyclic_exact_certificate.json").read_text(encoding="utf-8"))

    expected = {
        6: (20, 2, 18),
        8: (42, 0, 42),
        12: (110, 2, 108),
        18: (272, 20, 252),
        30: (812, 38, 774),
    }
    for payload in (generated, checked):
        assert payload["arithmetic"] == "Python integer arithmetic only"
        rows = {row["n"]: row for row in payload["rings"]}
        assert sorted(rows) == sorted(expected)
        for n, (total, same, cross) in expected.items():
            assert rows[n]["surviving_triples_total"] == total
            assert rows[n]["same_packet_triples"] == same
            assert rows[n]["cross_packet_triples"] == cross
            assert rows[n]["bad_cross_packet_fisher_pairs"] == []


def test_source_verification_ledger_covers_manuscript_arxiv_ids() -> None:
    manuscript = read_text("ConductorBlindSpot.md")
    ids = sorted(set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", manuscript)))
    ledger = json.loads((ROOT / "sources" / "verification_ledger.json").read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in ledger["entries"]}

    assert ids
    assert set(ids) <= set(entries)
    assert ledger["checked_as_of"] == "2026-05-22"

    for arxiv_id in ids:
        entry = entries[arxiv_id]
        assert entry["primary_url"] == f"https://arxiv.org/abs/{arxiv_id}"
        assert entry["source_status"] in {"peer_reviewed", "preprint", "under_review"}
        assert entry["claim_locations"]

    dcd = entries["2605.09129"]
    assert dcd["source_status"] == "under_review"
    assert "87" in " ".join(dcd["claims_verified"])
