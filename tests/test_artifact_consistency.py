from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "empirical" / "reports"


def load_json(rel: str) -> object:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def load_module(rel: str, name: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def demo_summary_rows() -> dict[int, dict[str, int]]:
    summary = (REPORTS / "paper34_demo_summary.md").read_text(encoding="utf-8")
    rows: dict[int, dict[str, int]] = {}
    in_layer1 = False
    for line in summary.splitlines():
        if line.startswith("## Layer 1"):
            in_layer1 = True
            continue
        if line.startswith("## Layer 2"):
            break
        if not in_layer1:
            continue
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 5:
            continue
        n = int(parts[0])
        rows[n] = {
            "cross": int(parts[2]),
            "exact_ladder": int(parts[3]),
            "total": int(parts[4]),
        }
    return rows


def test_demo_layer1_certificates_match_exact_ladder_and_summary() -> None:
    cyclic = load_json("empirical/reports/paper34_cyclic_exact_certificate.json")
    exact_by_n = {row["n"]: row for row in cyclic["rings"]}
    summary_by_n = demo_summary_rows()

    assert set(summary_by_n) == {6, 8, 12, 18, 30}
    assert "INSERT_16" not in (REPORTS / "paper34_demo_summary.md").read_text(encoding="utf-8")

    for n, exact in exact_by_n.items():
        cert = load_json(f"empirical/reports/paper34_demo_n{n}_certificate.json")
        assert cert["n"] == n
        assert cert["tangent_dim"] == n - 1
        assert cert["fisher_block_diagonal"] is True
        assert cert["fisher_offdiag_max_abs"] <= 1e-9
        assert cert["selection_rule_total_triples"] == exact["surviving_triples_total"]
        assert cert["expected_cross_packet_triples"] == exact["cross_packet_triples"]
        assert cert["cross_packet_triples_count"] == exact["cross_packet_triples"]

        summary = summary_by_n[n]
        assert summary["cross"] == exact["cross_packet_triples"]
        assert summary["exact_ladder"] == exact["cross_packet_triples"]
        assert summary["total"] == exact["surviving_triples_total"]


def test_demo_curve_jsons_have_required_arms_and_bounded_metrics() -> None:
    for n in (6, 8, 12, 18, 30):
        curve = load_json(f"empirical/reports/paper34_demo_n{n}_curve.json")
        assert curve["n"] == n
        assert curve["steps"] > 0
        assert curve["layer1"]["cross_packet_triples_count"] == curve["layer1"]["expected_cross_packet_triples"]

        for arm in ("d1", "d2"):
            final = curve[f"{arm}_final"]
            records = curve[f"{arm}_records"]
            assert records
            assert records[-1]["step"] == final["step"]
            for key in ("train_acc", "test_acc", "rho_cross_raw", "rho_cross_prec"):
                assert 0.0 <= final[key] <= 1.0
            assert -1.0 <= final["discard_ratio_clamped"] <= 1.0

        if n != 6:
            assert curve["d2_strong_records"]
            strong = curve["d2_strong_final"]
            assert 0.0 <= strong["rho_cross_raw"] <= 1.0


def test_noncyclic_checked_jsons_match_deterministic_verifier() -> None:
    module = load_module("empirical/paper34_C4_noncyclic_certificate.py", "noncyclic_certificate")

    for name, latex, moduli in module.CANDIDATES:
        expected = module.verify_group(name, latex, moduli)
        filename = f"paper34_C4_{name.replace(' ', '_').replace('/', '').replace('(', '').replace(')', '').replace('^', '')}.json"
        checked = load_json(f"empirical/reports/{filename}")

        for key in (
            "name",
            "moduli",
            "order",
            "tangent_dim",
            "packet_sizes",
            "fisher_block_diagonal",
            "selection_rule_total_triples",
            "expected_total_triples_closed_form",
            "cross_packet_triples_count",
            "same_packet_triples_count",
            "diagnostic_applicable",
        ):
            if key == "packet_sizes":
                assert checked[key] == {str(k): v for k, v in expected[key].items()}
            else:
                assert checked[key] == expected[key]
        assert checked["cross_packet_triples_count"] > 0


def test_pythia_results_have_revision_keys_and_bounded_values() -> None:
    for rel in (
        "empirical/pythia_rho_x_sweep/results_n_axis.json",
        "empirical/pythia_rho_x_sweep/results_d_axis.json",
    ):
        rows = load_json(rel)
        assert rows
        for row in rows:
            assert row["model_revision_requested"]
            assert "model_revision_resolved" in row
            assert 0.0 <= row["top1_acc"] <= 1.0
            assert 0.0 <= row["ring_mass_bare"] <= 1.0
            assert 0.0 <= row["ring_mass_space"] <= 1.0
            assert row["style"] in {"bare", "space"}

            for block_name in ("batch_mean", "per_example"):
                block = row[block_name]
                assert 0.0 <= block["ring_rho_x"] <= 1.0
                assert 0.0 <= block["perm_rho_x_mean"] <= 1.0
                assert block["perm_rho_x_std"] >= 0.0
                assert -1.0 <= block["separation"] <= 1.0


def test_layer3_json_summaries_record_branch_b_interpretation() -> None:
    for tag in (
        "adam_alpha0p1",
        "adam_neumann_alpha0p1",
        "adam_within_packet_alpha0p1",
    ):
        paths = sorted(REPORTS.glob(f"paper34_layer3_n30_seed*_{tag}.json"))
        assert len(paths) == 3
        interactions = []
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["n"] == 30
            assert payload["baseline_mode"] == "adam"
            assert payload["cubic_alpha"] == 0.1
            assert payload["steps"] == 15000
            interactions.append(payload["summary"]["interaction"])

        mean_interaction = sum(interactions) / len(interactions)

        summary = (REPORTS / f"paper34_layer3_summary_{tag}.md").read_text(encoding="utf-8")
        assert "Branch B" in summary
        assert f"**{mean_interaction:+.3f}**" in summary


def test_checked_pdf_outputs_match() -> None:
    root_pdf = ROOT / "ConductorBlindSpot.pdf"
    build_pdf = ROOT / "build" / "paper.pdf"

    assert root_pdf.read_bytes().startswith(b"%PDF")
    assert build_pdf.read_bytes().startswith(b"%PDF")
    assert root_pdf.stat().st_size > 100_000
    assert root_pdf.read_bytes() == build_pdf.read_bytes()


def test_claim_report_numeric_artifact_claims_match_checked_outputs() -> None:
    report = (ROOT / "docs" / "final_claims_assessment.md").read_text(encoding="utf-8")

    cyclic = load_json("empirical/reports/paper34_cyclic_exact_certificate.json")
    cross_counts = [row["cross_packet_triples"] for row in cyclic["rings"]]
    assert ", ".join(str(count) for count in cross_counts[:-1]) + f", and {cross_counts[-1]}" in report

    noncyclic_paths = sorted(REPORTS.glob("paper34_C4_*.json"))
    assert len(noncyclic_paths) == 5
    assert "five `paper34_C4_*.json` files" in report

    n_axis = load_json("empirical/pythia_rho_x_sweep/results_n_axis.json")
    d_axis = load_json("empirical/pythia_rho_x_sweep/results_d_axis.json")
    all_pythia_rows = n_axis + d_axis
    ring_rhos = [row["per_example"]["ring_rho_x"] for row in all_pythia_rows]
    assert f"{len(n_axis)} N-axis rows and {len(d_axis)} D-axis rows" in report
    assert f"{min(ring_rhos):.4f} to {max(ring_rhos):.4f}" in report

    means = []
    for tag in (
        "adam_alpha0p1",
        "adam_neumann_alpha0p1",
        "adam_within_packet_alpha0p1",
    ):
        interactions = []
        seeds = []
        for path in sorted(REPORTS.glob(f"paper34_layer3_n30_seed*_{tag}.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            seeds.append(payload["seed"])
            interactions.append(payload["summary"]["interaction"])
        assert seeds == [0, 1, 2]
        means.append(sum(interactions) / len(interactions))
    expected_means = ", ".join(f"{mean:+.3f}" for mean in means[:-1]) + f", and {means[-1]:+.3f}"
    assert expected_means in report
