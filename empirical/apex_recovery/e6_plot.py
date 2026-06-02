"""
E6 figure -- made OFFLINE from the JSONL the GPU run captured (no torch, no GPU).

Reads reports/e6_modadd_p{P}_metrics.jsonl (+ the control if present) and draws
the co-emergence: val accuracy and the registers D (embedding Fourier), S (logit
additivity), and rho_x (cross-packet cubic, composite n only) against step --
task vs control. A vertical line marks the grok step.

Usage (run with `py` = 3.13 which has matplotlib):
  py empirical/apex_recovery/e6_plot.py --p 113     # prime: D/S co-emergence
  py empirical/apex_recovery/e6_plot.py --p 30      # composite: adds rho_x panel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPORTS = Path(__file__).resolve().parent / "reports"


def load(tag, p):
    fp = REPORTS / f"e6_{tag}_p{p}_metrics.jsonl"
    if not fp.exists():
        return None
    rows = [json.loads(line) for line in fp.read_text().splitlines() if line.strip()]
    return rows


def col(rows, key):
    return [r["step"] for r in rows], [r.get(key) for r in rows]


def grok_step(rows):
    for r in rows:
        if r["train_acc"] > 0.9 and r["val_acc"] > 0.9:
            return r["step"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=113)
    args = ap.parse_args()

    task = load("modadd", args.p)
    ctrl = load("control", args.p)
    if task is None:
        raise SystemExit(f"no task metrics for p={args.p} in {REPORTS}")

    composite = any((r.get("n_triples_cross") or 0) > 0 for r in task)
    has_logit = composite and ("rho_x_logit" in task[0])
    panels = ["val_acc", "D_fourier_max", "S_logit_additivity"]
    titles = ["validation accuracy", "D  -  embedding Fourier concentration",
              "S  -  logit additivity (R^2 by sum-class)"]
    if has_logit:
        panels += ["rho_x_logit_cross_mass"]
        titles += ["cross-packet cubic mass, offset profile (CBS Def 6.1 / Conj 5.8)  [log scale]"]
    elif composite:
        panels.append("rho_x")
        titles.append("rho_x  -  cross-packet cubic mass (CBS Def 6.1)")

    gs = grok_step(task)
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.3 * len(panels)),
                             sharex=True)
    for ax, key, title in zip(axes, panels, titles):
        xt, yt = col(task, key)
        ax.plot(xt, yt, color="C3", lw=2, label="mod-add (ring task)")
        if ctrl is not None:
            xc, yc = col(ctrl, key)
            ax.plot(xc, yc, color="0.55", lw=1.5, ls="--", label="control (random table)")
        if key in ("rho_x_logit_total_mass", "rho_x_logit_cross_mass"):
            ax.set_yscale("log")
        if gs is not None:
            ax.axvline(gs, color="C0", lw=1, alpha=0.6)
            ax.text(gs, ax.get_ylim()[1], "  grok", color="C0", va="top", fontsize=8)
        ax.set_ylabel(key)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="center right", fontsize=8)
    axes[-1].set_xlabel("training step")
    kind = "composite" if composite else "prime"
    fig.suptitle(f"E6  -  grokking as the two-register boundary  (p={args.p}, {kind})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = REPORTS / f"e6_coemergence_p{args.p}.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    if not composite:
        print("note: prime modulus -> rho_x panel omitted (rho_x==0 by construction); "
              "run a composite n (e.g. --p 30) for the CBS Conjecture-5.8 panel.")


if __name__ == "__main__":
    main()
