"""Offline read of the E6 GCP runs (no torch). Prints grok step, the D/S
co-emergence, and whether rho_x discriminates task vs control."""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "e6_gcp_reports"


def load(tag, p):
    f = D / f"e6_{tag}_p{p}_metrics.jsonl"
    if not f.exists():
        return None
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def grok(rows):
    for r in rows:
        if r["train_acc"] > 0.9 and r["val_acc"] > 0.9:
            return r["step"]
    return None


def at(rows, step):
    return min(rows, key=lambda r: abs(r["step"] - step))


def show(label, r):
    print(f"    {label:11s} step{r['step']:6d}: val={r['val_acc']:.3f} "
          f"D={r['D_fourier_max']:.3f} S={r['S_logit_additivity']:.3f} "
          f"rho_x={r['rho_x']:.4f} totmass={r['rho_x_total_mass']:.3g}")


for p in (113, 110):
    task = load("modadd", p)
    ctrl = load("control", p)
    if not task:
        continue
    gs = grok(task)
    kind = "prime" if p == 113 else "composite"
    print(f"\n===== p={p} ({kind}) =====")
    print(f"  task: grok_step={gs}  final step={task[-1]['step']} val={task[-1]['val_acc']:.3f}")
    if ctrl:
        print(f"  ctrl: grok_step={grok(ctrl)}  final step={ctrl[-1]['step']} val={ctrl[-1]['val_acc']:.3f}")
    if gs:
        show("task pre", at(task, max(0, gs - 6000)))
        show("task GROK", at(task, gs))
        show("task post", at(task, gs + 3800))
    else:
        show("task end", task[-1])
    if ctrl:
        show("ctrl @grok", at(ctrl, gs if gs else ctrl[-1]["step"]))

    # D/S co-emergence magnitude
    if gs:
        pre = at(task, max(0, gs - 6000))
        post = at(task, gs + 3800)
        print(f"    D rise {pre['D_fourier_max']:.3f}->{post['D_fourier_max']:.3f}   "
              f"S rise {pre['S_logit_additivity']:.3f}->{post['S_logit_additivity']:.3f}")
        if ctrl:
            ce = at(ctrl, post["step"])
            print(f"    control at same step: D={ce['D_fourier_max']:.3f} S={ce['S_logit_additivity']:.3f}")

    # rho_x discrimination (composite only)
    if p != 113 and ctrl:
        cser = {r["step"]: r["rho_x"] for r in ctrl}
        diffs = [abs(r["rho_x"] - cser[r["step"]]) for r in task if r["step"] in cser]
        tr = [r["rho_x"] for r in task]
        cr = [r["rho_x"] for r in ctrl]
        ncross = task[0].get("n_triples_cross", 0)
        ntot = next((r for r in task), {})
        print(f"    rho_x task range [{min(tr):.3f},{max(tr):.3f}]  "
              f"ctrl range [{min(cr):.3f},{max(cr):.3f}]")
        print(f"    rho_x mean|task-ctrl| (matched steps) = {statistics.mean(diffs):.4f}")
        # total mass behaviour near grok (u->0 check)
        if gs:
            print(f"    score-rho_x totmass: pre={at(task,max(0,gs-6000))['rho_x_total_mass']:.3g} "
                  f"grok={at(task,gs)['rho_x_total_mass']:.3g} "
                  f"post={at(task,gs+3800)['rho_x_total_mass']:.3g}  (collapse => score-rho_x noisy)")

        # persistent logit-cubic (offset-profile rho_x), if logged
        if "rho_x_logit" in task[0]:
            print("    --- persistent logit-cubic (offset-profile) ---")
            tm = [r["rho_x_logit_total_mass"] for r in task]
            cm = [r["rho_x_logit_total_mass"] for r in ctrl]
            if gs:
                pre, gk, po = at(task, max(0, gs - 6000)), at(task, gs), at(task, gs + 3800)
                cgk = at(ctrl, gk["step"])
                print(f"    MASS  task pre={pre['rho_x_logit_total_mass']:.3g} "
                      f"grok={gk['rho_x_logit_total_mass']:.3g} post={po['rho_x_logit_total_mass']:.3g}"
                      f"   ctrl@grok={cgk['rho_x_logit_total_mass']:.3g}")
                ratio = po["rho_x_logit_total_mass"] / max(cgk["rho_x_logit_total_mass"], 1e-12)
                print(f"    MASS  task/ctrl at post-grok = {ratio:.1f}x  (grow+separate => persistent signal)")
                print(f"    RATIO (cross-packet share) task pre={pre['rho_x_logit']:.3f} "
                      f"grok={gk['rho_x_logit']:.3f} post={po['rho_x_logit']:.3f}  ctrl={cgk['rho_x_logit']:.3f}")
            print(f"    logit-cubic mass range: task [{min(tm):.3g},{max(tm):.3g}]  "
                  f"ctrl [{min(cm):.3g},{max(cm):.3g}]")
