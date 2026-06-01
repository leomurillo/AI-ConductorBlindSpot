#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3: CBS x T5/T6 mixed-gauge 4th arm for the §8.6 experimental matrix
====================================================================

Adds **Form D (harmonic-dual gauge correction)** to the §8.6 framework.
Closes P3 of T0_T6_HARVEST.md.

Motivation
----------

The §8.6 forms A/B/C all act in the RMS gauge (the natural gauge of
AdamW): they inject an additive cross-packet correction into the
batch-mean logit gradient. T6 Theorem 1.1 says the RMS gauge is a
perfectly closed L^2-normalization geometry; T5 Theorem 4.2 says the
ODD cumulant tower is detected by log(A*H/G^2), the harmonic-skew
SCALAR witness paired against the RMS gauge.

Form D operationalizes this for the §8.6 setup: instead of injecting
cubic content (which the RMS observer cannot represent), we inject
the gap between the harmonic-mean and arithmetic-mean per-class
gradient. This is a T5-native CHANGE OF OBSERVER intervention.

Definition (Form D)
-------------------

Given per-example head logit gradients u_per of shape (B, n) (with
B = batch size, n = output classes), define:

  g_R[y] = mean_i u_per[i, y]                     (arithmetic / RMS)
  H[y]   = B / sum_i (1 / (|u_per[i, y]| + eps))  (harmonic of |u|)
  g_H[y] = sign(g_R[y]) * H[y]                    (signed harmonic)
  w[y]   = g_H[y] - g_R[y]                        (harmonic-arithmetic gap)
  w     -= mean(w)                                (center for T_{p_*})

Then dlogits_D = dlogits_A1 + (alpha * ||g_R||/||w||) * w / B.

By T5 Theorem 4.2, w is a SCALAR-PER-CLASS witness of the odd-cumulant
content of the per-example gradient distribution that the RMS-gauge
preconditioner (Adam's v_t) cannot absorb. The pre-claim is that on
the ring task, the odd-cumulant content has a SPECIFIC packet-aligned
shape (a ring-coherent harmonic skew); on the structureless control,
it is symmetric and washes out.

Pre-registered branches
-----------------------

Branch A (T5 + T6 mixed-gauge confirmed): interaction > 0, with
Delta_T1 > 0 and Delta_T2s within noise of zero. Form D preferentially
accelerates the ring task. Reads as the harmonic observer recovers
the odd-cumulant signal RMS-gauge optimizers miss.

Branch B (refutation, joining forms A/B/C in null): interaction <= 0
within noise. Reads as the §8.6 null is universal across observer
choices on the head-only intervention substrate; the depth contribution
(P1 result, 12.5%) is the only meaningful gap and any head-side
intervention is structurally incapable of compressing it.

Outputs
-------
  empirical/reports/cbs_p3_mixed_gauge_<seed>.json
  empirical/reports/cbs_p3_mixed_gauge_summary.md

Usage
-----
  python empirical/cbs_p3_mixed_gauge_experiment.py [--n N] [--steps S]
                                                    [--seeds 0,1,...]
                                                    [--alpha 0.1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Reuse the existing layer3 infrastructure.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper34_conductor_blindspot_demo as demo            # noqa: E402
import paper34_layer3_cubic_experiment as layer3           # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "empirical" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Form D: harmonic-dual correction (T5 Theorem 4.2).
# ---------------------------------------------------------------------------


def compute_harmonic_dual_correction(u_per: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Form D: per-class gap between harmonic and arithmetic mean of |u_per|.

    T5 Theorem 4.2: log(A*H/G^2) measures the odd cumulant content of
    a positive ledger. We apply this column-wise to |u_per|, then
    reattach the sign and center.

    Arguments
    ---------
    u_per : (B, n) array of per-example head logit gradients
    eps   : stabilizer for 1/(|u| + eps)

    Returns
    -------
    w : (n,) real, mean-zero array; the harmonic-arithmetic gap per
        class, sign-preserving.
    """
    B, n = u_per.shape
    abs_u = np.abs(u_per)
    # arithmetic mean per class
    g_R = u_per.mean(axis=0)                                # (n,)
    # harmonic mean of |u_per| per class (positive)
    H = B / np.sum(1.0 / (abs_u + eps), axis=0)              # (n,)
    # signed harmonic
    g_H = np.sign(g_R) * H
    w = g_H - g_R
    w = w - w.mean()
    return w


def backward_with_form_d(
    params: demo.MLPParams,
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    cubic_alpha: float = 0.0,
    eps_head: float = 1e-3,
    use_head_ng: bool = False,
) -> Tuple[demo.MLPParams, np.ndarray]:
    """
    Forward + Form D logit-gradient + standard backward through MLP.

    Standard plain-AdamW dlogits (use_head_ng=False) augmented with
    the harmonic-dual correction w scaled to alpha * ||u_bar|| / ||w||.
    """
    logits, h, z1, emb, a_idx, b_idx = demo.forward(params, X)
    p = demo.softmax(logits)
    B = X.shape[0]
    d_emb = params.E_a.shape[1]

    yh = np.zeros_like(p)
    yh[np.arange(B), y] = 1.0

    if use_head_ng:
        u_per = (p - yh) / (p + eps_head)
    else:
        u_per = p - yh
    dlogits = u_per / B

    if cubic_alpha > 0:
        # u_bar = batch-summed centered version of u_per (i.e. mean * B).
        u_bar = dlogits.mean(axis=0) * B
        u_bar = u_bar - u_bar.mean()
        norm_u = float(np.linalg.norm(u_bar))
        # Form D correction (per-class, NOT cubic; T5-derived).
        w = compute_harmonic_dual_correction(u_per)
        norm_w = float(np.linalg.norm(w)) + 1e-12
        w_scaled = w * (norm_u / norm_w) * cubic_alpha
        dlogits = dlogits + (w_scaled / B)[None, :]

    # Standard backward through the MLP -- copy-pasted from layer3.
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ params.W2.T
    dz1 = dh * demo.gelu_grad(z1)
    dW1 = emb.T @ dz1
    db1 = dz1.sum(axis=0)
    de_emb = dz1 @ params.W1.T
    de_a = de_emb[:, :d_emb]
    de_b = de_emb[:, d_emb:]
    dE_a = np.zeros_like(params.E_a)
    dE_b = np.zeros_like(params.E_b)
    np.add.at(dE_a, a_idx, de_a)
    np.add.at(dE_b, b_idx, de_b)
    grads = demo.MLPParams(dE_a, dE_b, dW1, db1, dW2, db2)
    return grads, p


def run_arm_form_d(
    n: int,
    seed: int,
    steps: int,
    ablate_ring: bool,
    ablate_strong: bool,
    cubic_alpha: float,
    use_head_ng: bool = False,
    log_every: int = 100,
    hidden: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    eps_head: float = 0.1,
) -> List[dict]:
    """Train one (arm, task) configuration with Form D correction enabled."""
    rng = np.random.default_rng(seed)
    X_tr, y_tr, X_te, y_te = demo.make_dataset(
        n, ablate_ring=ablate_ring, seed=seed, ablate_strong=ablate_strong
    )
    params = demo.init_mlp(n, hidden, rng)
    state = demo.init_adam(params, lr=lr, weight_decay=weight_decay)

    records: List[dict] = []
    t0 = time.time()
    for step in range(1, steps + 1):
        grads, _ = backward_with_form_d(
            params, X_tr, y_tr, n,
            cubic_alpha=cubic_alpha,
            eps_head=eps_head,
            use_head_ng=use_head_ng,
        )
        demo.adam_step(params, grads, state)

        if step % log_every == 0 or step == 1:
            logits_tr, _, _, _, _, _ = demo.forward(params, X_tr)
            logits_te, _, _, _, _, _ = demo.forward(params, X_te)
            loss_tr = demo.cross_entropy(logits_tr, y_tr)
            loss_te = demo.cross_entropy(logits_te, y_te)
            acc_tr = demo.accuracy(logits_tr, y_tr)
            acc_te = demo.accuracy(logits_te, y_te)
            records.append(dict(
                step=int(step),
                wall_time_s=float(time.time() - t0),
                train_loss=float(loss_tr),
                test_loss=float(loss_te),
                train_acc=float(acc_tr),
                test_acc=float(acc_te),
            ))
    return records


def late_window_mean(records: List[dict], key: str, window_frac: float = 0.10) -> float:
    if not records:
        return float("nan")
    n_keep = max(1, int(round(len(records) * window_frac)))
    tail = records[-n_keep:]
    return float(np.mean([r[key] for r in tail]))


def run_p3_experiment(
    n: int = 30,
    seed: int = 0,
    steps: int = 7500,
    cubic_alpha: float = 0.1,
    hidden: int = 128,
) -> dict:
    """The full 2-arm x 2-task experiment for one (n, seed)."""
    print(f"\n[P3 n={n} seed={seed} alpha={cubic_alpha} steps={steps}]")

    tasks = [
        ("T1_ring", dict(ablate_ring=False, ablate_strong=False)),
        ("T2_strong", dict(ablate_ring=False, ablate_strong=True)),
    ]
    arms = [
        ("A1_baseline_adam", dict(cubic_alpha=0.0)),
        ("A2_form_d_harmonic", dict(cubic_alpha=cubic_alpha)),
    ]

    records = {}
    for arm_name, arm_kwargs in arms:
        records[arm_name] = {}
        for task_name, task_kwargs in tasks:
            print(f"  {arm_name} x {task_name} ...", end=" ", flush=True)
            t0 = time.time()
            rec = run_arm_form_d(
                n=n, seed=seed, steps=steps, hidden=hidden,
                **arm_kwargs, **task_kwargs
            )
            elapsed = time.time() - t0
            final = rec[-1]
            print(f"final test_acc={final['test_acc']:.3f} ({elapsed:.0f}s)")
            records[arm_name][task_name] = rec

    # Endpoint computation.
    summary = {}
    for arm_name, _ in arms:
        for task_name, _ in tasks:
            recs = records[arm_name][task_name]
            key = f"{arm_name}_{task_name}"
            summary[f"{key}_test_acc_final"] = recs[-1]["test_acc"]
            summary[f"{key}_test_acc_latemean"] = late_window_mean(recs, "test_acc")

    a1, a2 = arms[0][0], arms[1][0]
    delta_t1 = (summary[f"{a2}_T1_ring_test_acc_latemean"]
                - summary[f"{a1}_T1_ring_test_acc_latemean"])
    delta_t2 = (summary[f"{a2}_T2_strong_test_acc_latemean"]
                - summary[f"{a1}_T2_strong_test_acc_latemean"])
    summary["delta_T1_ring"] = delta_t1
    summary["delta_T2_strong"] = delta_t2
    summary["interaction"] = delta_t1 - delta_t2
    summary["a1_name"] = a1
    summary["a2_name"] = a2

    return dict(
        n=int(n), seed=int(seed), steps=int(steps),
        cubic_alpha=float(cubic_alpha), hidden=int(hidden),
        form="harmonic_dual_t5_derived",
        records=records,
        summary=summary,
    )


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--steps", type=int, default=7500,
                   help="reduced from §8.6 default 15000 for P3 compute budget")
    p.add_argument("--seeds", type=str, default="0,1",
                   help="comma-separated seeds (default: 2-seed P3 sweep)")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--smoke", action="store_true",
                   help="quick smoke test: n=12, single seed, 1500 steps")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    if args.smoke:
        rs = [run_p3_experiment(n=12, seed=0, steps=1500, cubic_alpha=args.alpha)]
    else:
        seeds = [int(x) for x in args.seeds.split(",")]
        rs = [run_p3_experiment(
            n=args.n, seed=s, steps=args.steps, cubic_alpha=args.alpha
        ) for s in seeds]

    # Save per-run JSONs.
    for r in rs:
        path = REPORTS_DIR / f"cbs_p3_form_d_n{r['n']}_seed{r['seed']}.json"
        path.write_text(json.dumps(r, indent=2))

    # Aggregate.
    interactions = [r["summary"]["interaction"] for r in rs]
    delta_t1s = [r["summary"]["delta_T1_ring"] for r in rs]
    delta_t2s = [r["summary"]["delta_T2_strong"] for r in rs]

    md = [
        "# P3: CBS x T5/T6 mixed-gauge (Form D harmonic-dual) experiment",
        "",
        "Form D: w[y] = sign(g_R[y]) * H[y] - g_R[y], where g_R[y] is "
        "the per-class arithmetic mean of u_per[:, y] and H[y] is the "
        "per-class harmonic mean of |u_per[:, y]|. By T5 Theorem 4.2 "
        "this is a per-class scalar witness of the ODD cumulant content "
        "the RMS-gauge optimizer (Adam) cannot absorb.",
        "",
        f"**Setup.** n = {rs[0]['n']}, alpha = {rs[0]['cubic_alpha']}, "
        f"steps = {rs[0]['steps']}, hidden = {rs[0]['hidden']}.",
        "",
        "## Per-seed result",
        "",
        "| seed | A1xT1 | A2xT1 | DeltaT1 | A1xT2s | A2xT2s | DeltaT2s | **interaction** |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rs:
        s = r["summary"]
        a1, a2 = s["a1_name"], s["a2_name"]
        md.append(
            f"| {r['seed']} | "
            f"{s[f'{a1}_T1_ring_test_acc_latemean']:.3f} | "
            f"{s[f'{a2}_T1_ring_test_acc_latemean']:.3f} | "
            f"{s['delta_T1_ring']:+.3f} | "
            f"{s[f'{a1}_T2_strong_test_acc_latemean']:.3f} | "
            f"{s[f'{a2}_T2_strong_test_acc_latemean']:.3f} | "
            f"{s['delta_T2_strong']:+.3f} | "
            f"**{s['interaction']:+.3f}** |"
        )
    md += [
        f"| **mean** | | | {np.mean(delta_t1s):+.3f} | | | "
        f"{np.mean(delta_t2s):+.3f} | **{np.mean(interactions):+.3f}** |",
        f"| **std**  | | | {np.std(delta_t1s):.3f} | | | "
        f"{np.std(delta_t2s):.3f} | {np.std(interactions):.3f} |",
        "",
        "## Comparison to forms A/B/C (existing §8.6 results)",
        "",
        "From CBS §8.6 (Adam baseline, alpha=0.1, n=30, 15000 steps, "
        "3 seeds):",
        "",
        "| Form | mean interaction | reading |",
        "|---|---:|---|",
        "| A (Newton cubic) | +0.003 (std 0.007) | Branch B (null) |",
        "| B (Neumann cubic) | -0.011 (std 0.020) | Branch B (mild negative) |",
        "| C (within-packet amp) | -0.004 (std 0.007) | Branch B (null) |",
        f"| **D (T5 harmonic-dual)** | **{np.mean(interactions):+.3f} "
        f"(std {np.std(interactions):.3f})** | "
        f"{'**Branch A**' if np.mean(interactions) > 0.02 else '**Branch B**'} |",
        "",
        "## Pre-registered reading",
        "",
        f"Mean interaction across {len(rs)} seeds at n = {rs[0]['n']}, "
        f"alpha = {rs[0]['cubic_alpha']}: "
        f"**{np.mean(interactions):+.3f}** (std {np.std(interactions):.3f}).",
        "",
    ]
    inter_mean = float(np.mean(interactions))
    if inter_mean > 0.02:
        md += [
            "**Branch A (T5 + T6 + T0 mixed-gauge confirmed).** "
            "Form D preferentially accelerates the ring task; the "
            "harmonic-dual observer recovers odd-cumulant signal that "
            "the four §8.6 RMS-gauge forms (A/B/C) cannot detect. The "
            "T5 simplification claim is empirically supported on this "
            "substrate.",
        ]
    elif inter_mean < -0.02:
        md += [
            "**Branch B (universal head-side null, with negative bias).** "
            "Form D's negative interaction places it alongside §8.6's "
            "form (B) Neumann: any head-side intervention -- "
            "additive cubic OR observer-change harmonic -- opposes "
            "the ring-task descent, consistent with P1's analytical "
            "result that the head-only beta-flow has rho_x(t) = const "
            "and any change of head-side dynamics moves the trajectory "
            "off the ring-coherent direction. The 10%+ depth deviation "
            "(P1) remains the only meaningful gap and requires an "
            "upstream-feature intervention, not a head-side one.",
        ]
    else:
        md += [
            "**Branch B (universal head-side null).** Form D joins forms "
            "A/B/C in producing a null interaction on this substrate. "
            "The §8.6 result is therefore observer-independent: NO "
            "head-side correction -- additive cubic (A/B), within-"
            "packet amplification (C), OR harmonic-dual observer "
            "change (D) -- preferentially helps the ring task on "
            "AdamW + MLP. Consistent with P1: the head-only beta-flow "
            "has rho_x(t) = const, and any head-side modification "
            "is structurally orthogonal to the depth-feature dynamics "
            "that drive the empirical Conjecture 5.8 signal.",
        ]
    md += [
        "",
        "## Implication for CBS §8.6 sharpening",
        "",
        "Form D is the first §8.6 arm motivated by an *observer-change* "
        "rather than an *additive correction*. Its result (whichever "
        "branch) constrains the §8.6 reading:",
        "",
        "- If Branch A: change-of-observer interventions DO carry the "
        "ring-task asymmetry the additive corrections miss; the right "
        "operational program is to climb the T0 gauge corridor and "
        "find the observer matched to the ring's odd-cumulant "
        "structure. (Theoretical predictions: T5 paper §6.3 + T6 §5.)",
        "",
        "- If Branch B: the head-side intervention space is **closed "
        "under both additive and observer-change moves** on this "
        "substrate. Any non-trivial intervention must touch upstream "
        "features. The natural next experiments are the EGD/PGD arms "
        "of §7.8 (uniform within-mode rescaling, an upstream effect) "
        "or a parameter-side K-FAC variant.",
        "",
        "",
        "## Files",
        "",
        f"- Per-seed JSON: cbs_p3_form_d_n{rs[0]['n']}_seed*.json",
        "- This report.",
        "",
        "Generated by `empirical/cbs_p3_mixed_gauge_experiment.py`.",
    ]
    out_md = REPORTS_DIR / "cbs_p3_mixed_gauge_summary.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSummary: {out_md}")
    print(f"Interactions: {interactions}")
    print(f"Mean interaction: {inter_mean:+.4f}")
    print(f"Reading: {'Branch A' if inter_mean > 0.02 else 'Branch B'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
