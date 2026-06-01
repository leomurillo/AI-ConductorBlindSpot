#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper 34 -- Layer 3 (Section 8) pre-registered cubic-aware experiment.

HEAD-ONLY EXACT-FISHER VARIANT
==============================

The §8 hypothesis (Conjecture 5.8) predicts an INTERACTION between a
cubic-aware preconditioner correction and the ring-respecting task: A2
should help T1 more than it helps T2-strong.

Per the recommendation in the plan, we test the conjecture in the cleanest
form: on the OUTPUT side of a single-layer softmax head, where the
certificate is exact (Corollary 3.3 / 3.8) and the cubic correction is a
finite-dim character-basis computation. The rest of the network is trained
with the same AdamW as in §6.5's demo; only the LOGIT GRADIENT entering
backprop is modified by the two preconditioning choices below.

Arms
----
  A1 (matched second-order baseline) -- head exact-Fisher natural gradient
      = elementwise (p - y_onehot) / (p + eps).
  A2 (cubic-aware) -- A1, plus a cross-packet cubic injection
      computed from the batch-mean head-NG logit gradient, scaled to a
      fraction `cubic_alpha` of that gradient's norm.

Tasks
-----
  T1 (ring)        -- target = (a + b) mod n. The canonical grokking testbed.
  T2-strong (control) -- target = f(a, b) for a fixed uniformly-random
      f. Structureless random function; blocks internal ring composition
      (Section 6.3 D2-strong / Section 8.3 T2-strong).

Primary endpoint
----------------
  Interaction := [delta_T1_acc] - [delta_T2-strong_acc]
  where delta_X_acc = mean test_acc under A2 on X - mean test_acc under A1
  on X, evaluated over a fixed late-training window (default: the last 10%
  of training steps, averaging over the trajectory).

Branches (pre-claimed)
----------------------
  Branch A (Conjecture 5.8 confirmed): positive interaction; A2 - A1 helps
  T1 significantly more than T2-strong, with the T2-strong effect within
  noise of zero. Reported as confirmation; experiment in this form does
  not address persistence beyond the tested ring.

  Branch B (refutation on tested instance): interaction non-positive (A2
  helps both or neither task significantly, or helps T2-strong as much as
  T1). Reported as refutation on the tested instance. Theorem 3.5,
  Corollary 3.3, the diagnostic of Section 6, and the off-centroid
  expansion of Section 5 stand regardless.

Author: Leo. Layer 3 of paper 34 / C5 of §11.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Re-use utilities from the demo (cubic_triples, MLP architecture, etc.).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper34_conductor_blindspot_demo as demo

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "empirical" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_TAG_TO_FORM = {
    "adam_alpha0p1": "newton",
    "adam_neumann_alpha0p1": "neumann",
    "adam_within_packet_alpha0p1": "within_packet",
}
FORM_DESCRIPTIONS = {
    "newton": "Form A: plain AdamW baseline plus cross-packet Newton-style T(u,u,.) cubic injection.",
    "neumann": "Form B: plain AdamW baseline plus cross-packet Neumann-style T(h,u,.) cubic injection.",
    "within_packet": "Form C: plain AdamW baseline plus within-packet projection/amplification control.",
}


# ---------------------------------------------------------------------------
# 1. The cubic correction (cross-packet, character-basis).
# ---------------------------------------------------------------------------


_cross_packet_index_arrays_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _cross_packet_index_arrays(
    n: int, triples: List[Tuple[Tuple[int, int, int], bool]]
):
    """Cache (ks, ls, ms) NumPy arrays for cross-packet selection-rule triples.

    The triples list is built once per ring; this helper extracts the cross-
    packet subset into three integer arrays for vectorized scatter-add.
    """
    cache_key = (n, len(triples))
    cached = _cross_packet_index_arrays_cache.get(cache_key)
    if cached is not None:
        return cached
    ks: List[int] = []
    ls: List[int] = []
    ms: List[int] = []
    for (k, l, m), cross_packet in triples:
        if cross_packet:
            ks.append(k)
            ls.append(l)
            ms.append(m)
    out = (np.array(ks, dtype=np.intp), np.array(ls, dtype=np.intp), np.array(ms, dtype=np.intp))
    _cross_packet_index_arrays_cache[cache_key] = out
    return out


def compute_cubic_correction(
    u_bar: np.ndarray, n: int, triples: List[Tuple[Tuple[int, int, int], bool]]
) -> np.ndarray:
    """Form (A): cubic-Newton-style correction w = T(u, u, ·)|_{cross-packet}.

    Given batch-mean centered update direction u_bar in R^n, this returns
    a real, centered correction vector w computed as the cross-packet
    projection of the cubic contraction T(u_bar, u_bar, ·). Quadratic in u.

      1. u_hat[k] = (1/n) sum_y u_bar[y] exp(-2 pi i k y / n)  (DFT)
      2. w_hat[m] = sum_{(k, l) : k+l+m == 0 mod n, (k,l,m) cross-packet}
                      u_hat[k] * u_hat[l]
      3. w[y]    = real( sum_m w_hat[m] exp(2 pi i m y / n) )    (IDFT)
      4. Center: w -= w.mean()
    """
    y_idx = np.arange(n)
    Wm = np.exp(-2j * np.pi * np.arange(n)[:, None] * y_idx / n)  # (k, y)
    u_hat = (Wm @ u_bar) / n  # (n,) complex

    ks, ls, ms = _cross_packet_index_arrays(n, triples)
    w_hat = np.zeros(n, dtype=complex)
    np.add.at(w_hat, ms, u_hat[ks] * u_hat[ls])

    Wp = np.exp(2j * np.pi * np.arange(n)[:, None] * y_idx / n).T  # (y, m)
    w = np.real(Wp @ w_hat)
    w -= w.mean()
    return w


_conductor_packet_indices_cache: Dict[int, Dict[int, np.ndarray]] = {}


def _conductor_packet_indices(n: int) -> Dict[int, np.ndarray]:
    """Cache the per-packet character-index arrays for n.

    Returns a dict {d -> array of k in {1, ..., n-1} with cond(k) == d}.
    """
    cached = _conductor_packet_indices_cache.get(n)
    if cached is not None:
        return cached
    from math import gcd
    out: Dict[int, List[int]] = {}
    for k in range(1, n):
        d = n // gcd(k, n)
        out.setdefault(d, []).append(k)
    arr = {d: np.array(sorted(ks), dtype=np.intp) for d, ks in out.items()}
    _conductor_packet_indices_cache[n] = arr
    return arr


def compute_within_packet_correction(
    u_bar: np.ndarray, n: int
) -> np.ndarray:
    """Form (C): within-packet amplification — the sign-reversed cubic correction.

    Identifies the dominant conductor packet of the batch-mean update direction
    u_bar (the packet on which u_bar's Fourier mass is largest) and returns
    the projection of u_bar onto that packet's characters, IDFT'd back to
    real space and centered. Adding this correction to u_bar amplifies the
    within-packet direction the model is already descending in, which the
    §6.5 diagnostic and the §8.6 form-(B) interaction together identify as
    the consolidation direction on ring-structured modular addition.

    The certificate's content predicts that the quadratic curvature model
    class can represent within-packet structure (Corollary 3.3); the
    within-packet-amplifying correction is therefore IN the model class's
    span, not outside it. Whether amplifying it preferentially helps ring
    tasks (Branch A) or is no-op / hurts (Branch B) is the cleanest
    remaining test of the certificate's operational relevance: a positive
    interaction here means the optimization's bottleneck is the *rate* of
    within-packet consolidation, not the missing cross-packet content.

      1. u_hat[k] = (1/n) sum_y u_bar[y] exp(-2 pi i k y / n)
      2. for each packet d: energy[d] = sum_{k in P_d} |u_hat[k]|^2
      3. d_star = argmax_d energy[d]   (dominant packet at this step)
      4. project u_hat -> only characters with cond(k) = d_star, zero else
      5. IDFT, center.

    Returns w such that ||w|| = ||u_bar restricted to dominant packet||;
    the caller scales by alpha * ||u_bar|| / ||w|| as usual.
    """
    y_idx = np.arange(n)
    Wm = np.exp(-2j * np.pi * np.arange(n)[:, None] * y_idx / n)
    u_hat = (Wm @ u_bar) / n  # (n,) complex

    packets = _conductor_packet_indices(n)
    # Energy per packet from nontrivial characters.
    best_d = None
    best_energy = -1.0
    for d, ks in packets.items():
        energy = float(np.sum(np.abs(u_hat[ks]) ** 2))
        if energy > best_energy:
            best_energy = energy
            best_d = d
    # Project u_hat onto dominant packet.
    u_hat_proj = np.zeros_like(u_hat)
    u_hat_proj[packets[best_d]] = u_hat[packets[best_d]]

    # IDFT back to real space.
    Wp = np.exp(2j * np.pi * np.arange(n)[:, None] * y_idx / n).T
    w = np.real(Wp @ u_hat_proj)
    w -= w.mean()
    return w


def compute_cubic_correction_neumann(
    u_bar: np.ndarray,
    h_bar: np.ndarray,
    n: int,
    triples: List[Tuple[Tuple[int, int, int], bool]],
) -> np.ndarray:
    """Form (B): Neumann-series cubic correction w ∝ T_{p_*}(h, u, ·)|_{cross-packet}.

    This is the certificate's natural operationalization: the inverse
    Fisher at p_*+h, expanded to first order in h via the Neumann series

         g_{p_*+h}^{-1} ≈ g_{p_*}^{-1} + g_{p_*}^{-1} T_{p_*}(h, ·, ·) g_{p_*}^{-1}

    contributes a correction term T(h, u, ·) to the preconditioned update.
    Restricting to cross-packet triples isolates the content the certificate
    proves the quadratic class cannot represent (Theorem 5.2 / 5.6).

    Construction:
      1. u_hat[l] = (1/n) sum_y u_bar[y] exp(-2 pi i l y / n)
      2. h_hat[k] = (1/n) sum_y h_bar[y] exp(-2 pi i k y / n)
      3. w_hat[m] = sum_{(k, l, m) cross-packet, k+l+m ≡ 0 mod n}
                      h_hat[k] * u_hat[l]
      4. w[y]    = real( sum_m w_hat[m] chi_m(y) )
      5. Center.

    Linear in h, linear in u. Magnitude scales as ||h|| * ||u||, matching
    the certificate's prediction (correction grows as model leaves p_*).
    """
    y_idx = np.arange(n)
    Wm = np.exp(-2j * np.pi * np.arange(n)[:, None] * y_idx / n)
    u_hat = (Wm @ u_bar) / n
    h_hat = (Wm @ h_bar) / n

    ks, ls, ms = _cross_packet_index_arrays(n, triples)
    w_hat = np.zeros(n, dtype=complex)
    # Per ordered triple (k = h-index, l = u-index, m = output): h_hat[k] * u_hat[l].
    np.add.at(w_hat, ms, h_hat[ks] * u_hat[ls])

    Wp = np.exp(2j * np.pi * np.arange(n)[:, None] * y_idx / n).T
    w = np.real(Wp @ w_hat)
    w -= w.mean()
    return w


# ---------------------------------------------------------------------------
# 2. Modified backward: head NG + optional cubic correction.
# ---------------------------------------------------------------------------


def head_ng_logit_gradient(p: np.ndarray, yh: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Elementwise head exact-Fisher natural gradient.

    For categorical with prediction p in Delta_R and target one-hot yh:
       u   = p - yh           (raw cross-entropy logit gradient, in T_0)
       NG  = u / (p + eps)    (elementwise; equivalent to F^{-1} u where F
                               is the parameter Fisher of the softmax head)

    Adding eps avoids the blow-up on memorized directions where p_y -> 0.
    """
    return (p - yh) / (p + eps)


def backward_with_modifications(
    params: demo.MLPParams,
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    cubic_triples_cache: List[Tuple[Tuple[int, int, int], bool]],
    head_ng: bool = True,
    cubic_alpha: float = 0.0,
    eps: float = 1e-3,
    cubic_form: str = "newton",  # "newton" = T(u, u, ·); "neumann" = T(h, u, ·)
):
    """Forward + (modified) logit-gradient + standard backward through the MLP.

    Replaces the standard cross-entropy logit gradient with
       dlogits[i] = (p[i] - y_onehot[i]) / B            (head_ng=False)
       dlogits[i] = (p[i] - y_onehot[i]) / ((p[i]+eps) * B)  (head_ng=True)

    And optionally adds a per-batch cubic-correction direction w to every
    example's dlogits, scaled to be `cubic_alpha` times the norm of the
    batch-mean modified gradient. Two cubic-correction forms:

      cubic_form='newton'  : w = cross-packet projection of T(u, u, ·)
                             (quadratic in u; the form (A) of §8.6 v1).
      cubic_form='neumann' : w = cross-packet projection of T(h, u, ·)
                             (linear in h = p_bar - 1/n, linear in u; the
                             certificate's natural operationalization via
                             the Neumann series for g_{p_*+h}^{-1}).
    """
    logits, h, z1, emb, a_idx, b_idx = demo.forward(params, X)
    p = demo.softmax(logits)
    B = X.shape[0]
    d = params.E_a.shape[1]

    yh = np.zeros_like(p)
    yh[np.arange(B), y] = 1.0

    if head_ng:
        u_per = (p - yh) / (p + eps)
    else:
        u_per = p - yh
    dlogits = u_per / B

    if cubic_alpha > 0:
        u_bar = dlogits.mean(axis=0) * B  # batch-summed (mean over examples of u_per)
        u_bar = u_bar - u_bar.mean()
        if cubic_form == "newton":
            w = compute_cubic_correction(u_bar, n, cubic_triples_cache)
        elif cubic_form == "neumann":
            # h_bar = batch-mean prediction minus the uniform point p_* = 1/n.
            p_bar = p.mean(axis=0)
            p_bar = p_bar / p_bar.sum()
            h_bar = p_bar - 1.0 / n
            h_bar = h_bar - h_bar.mean()
            w = compute_cubic_correction_neumann(u_bar, h_bar, n, cubic_triples_cache)
        elif cubic_form == "within_packet":
            # Sign-reversed: project u_bar onto its dominant conductor packet,
            # amplify that direction. No cubic contraction; pure character-
            # basis projection. The §8.6 directionality insight predicts a
            # POSITIVE interaction here if ring-task progress is bottlenecked
            # by within-packet consolidation.
            w = compute_within_packet_correction(u_bar, n)
        else:
            raise ValueError(f"Unknown cubic_form: {cubic_form}")
        norm_u = float(np.linalg.norm(u_bar))
        norm_w = float(np.linalg.norm(w)) + 1e-12
        # Scale w so |alpha * w_scaled| / |u_bar| = alpha (relative-norm strength).
        w_scaled = w * (norm_u / norm_w) * cubic_alpha
        dlogits = dlogits + (w_scaled / B)[None, :]

    # Standard backward pass from dlogits through W2/b2/W1/b1/E_a/E_b.
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ params.W2.T
    dz1 = dh * demo.gelu_grad(z1)
    dW1 = emb.T @ dz1
    db1 = dz1.sum(axis=0)
    de_emb = dz1 @ params.W1.T
    de_a = de_emb[:, :d]
    de_b = de_emb[:, d:]
    dE_a = np.zeros_like(params.E_a)
    dE_b = np.zeros_like(params.E_b)
    np.add.at(dE_a, a_idx, de_a)
    np.add.at(dE_b, b_idx, de_b)
    grads = demo.MLPParams(dE_a, dE_b, dW1, db1, dW2, db2)
    return grads, p


# ---------------------------------------------------------------------------
# 3. Arm runner.
# ---------------------------------------------------------------------------


def run_arm(
    n: int,
    seed: int,
    steps: int,
    ablate_ring: bool,
    ablate_strong: bool,
    head_ng: bool,
    cubic_alpha: float,
    log_every: int = 50,
    hidden: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    eps: float = 0.1,
    cubic_form: str = "newton",
) -> List[dict]:
    """Train one (arm, task) configuration and return per-step records."""
    rng = np.random.default_rng(seed)
    X_tr, y_tr, X_te, y_te = demo.make_dataset(
        n, ablate_ring=ablate_ring, seed=seed, ablate_strong=ablate_strong
    )
    params = demo.init_mlp(n, hidden, rng)
    state = demo.init_adam(params, lr=lr, weight_decay=weight_decay)
    triples = demo.cubic_triples(n)

    records: List[dict] = []
    t0 = time.time()
    for step in range(1, steps + 1):
        grads, _ = backward_with_modifications(
            params,
            X_tr,
            y_tr,
            n,
            triples,
            head_ng=head_ng,
            cubic_alpha=cubic_alpha,
            eps=eps,
            cubic_form=cubic_form,
        )
        demo.adam_step(params, grads, state)

        if step % log_every == 0 or step == 1:
            logits_tr, _, _, _, _, _ = demo.forward(params, X_tr)
            logits_te, _, _, _, _, _ = demo.forward(params, X_te)
            loss_tr = demo.cross_entropy(logits_tr, y_tr)
            loss_te = demo.cross_entropy(logits_te, y_te)
            acc_tr = demo.accuracy(logits_tr, y_tr)
            acc_te = demo.accuracy(logits_te, y_te)
            records.append(
                {
                    "step": int(step),
                    "wall_time_s": float(time.time() - t0),
                    "train_loss": float(loss_tr),
                    "test_loss": float(loss_te),
                    "train_acc": float(acc_tr),
                    "test_acc": float(acc_te),
                }
            )
    return records


# ---------------------------------------------------------------------------
# 4. Experiment driver (4-arm matched-pair).
# ---------------------------------------------------------------------------


def make_arms(
    baseline_mode: str, cubic_alpha: float, head_ng_eps: float, cubic_form: str
):
    """Return the (A1, A2) arm specs for the chosen baseline mode.

    baseline_mode='head_ng' : A1 = head exact-Fisher NG (Adam on rest of net).
                              A2 = A1 + cubic correction.
    baseline_mode='adam'    : A1 = plain AdamW (no head NG).
                              A2 = A1 + cubic correction.

    cubic_form ∈ {'newton', 'neumann'} chooses the form of the cubic
    correction added in A2 (see compute_cubic_correction* docstrings).
    """
    common = dict(eps=head_ng_eps, cubic_form=cubic_form)
    if baseline_mode == "head_ng":
        return [
            ("A1_baseline_head_ng", dict(head_ng=True, cubic_alpha=0.0, **common)),
            ("A2_cubic_aware", dict(head_ng=True, cubic_alpha=cubic_alpha, **common)),
        ]
    elif baseline_mode == "adam":
        return [
            ("A1_baseline_adam", dict(head_ng=False, cubic_alpha=0.0, **common)),
            ("A2_cubic_aware", dict(head_ng=False, cubic_alpha=cubic_alpha, **common)),
        ]
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

TASKS = [
    ("T1_ring", dict(ablate_ring=False, ablate_strong=False)),
    ("T2_strong", dict(ablate_ring=False, ablate_strong=True)),
]


def _late_window_mean(records: List[dict], key: str, window_frac: float = 0.10) -> float:
    """Mean of `key` over the last `window_frac` of the trajectory."""
    if not records:
        return float("nan")
    n_keep = max(1, int(round(len(records) * window_frac)))
    tail = records[-n_keep:]
    return float(np.mean([r[key] for r in tail]))


def _time_to_threshold(records: List[dict], key: str, threshold: float) -> int:
    """First step at which `records[t][key] >= threshold`, or -1 if never."""
    for r in records:
        if r[key] >= threshold:
            return int(r["step"])
    return -1


def run_experiment(
    n: int,
    seed: int,
    steps: int,
    cubic_alpha: float,
    baseline_mode: str = "adam",
    head_ng_eps: float = 0.1,
    cubic_form: str = "newton",
    hidden: int = 128,
) -> dict:
    """Run the 4-arm matched-pair experiment for one ring at one seed."""
    print(f"\n{'=' * 78}\n  paper 34 Layer 3 -- n = {n}, seed = {seed}, "
          f"alpha = {cubic_alpha}, baseline = {baseline_mode}, "
          f"form = {cubic_form}\n{'=' * 78}")

    arms = make_arms(baseline_mode, cubic_alpha, head_ng_eps, cubic_form)
    a1_name = arms[0][0]
    a2_name = arms[1][0]

    records: Dict[str, Dict[str, List[dict]]] = {}
    for arm_name, arm_kwargs in arms:
        records[arm_name] = {}
        for task_name, task_kwargs in TASKS:
            print(f"  [{arm_name} x {task_name}] ...")
            t0 = time.time()
            rec = run_arm(
                n=n,
                seed=seed,
                steps=steps,
                hidden=hidden,
                **arm_kwargs,
                **task_kwargs,
            )
            print(
                f"    final: train_acc={rec[-1]['train_acc']:.3f}, "
                f"test_acc={rec[-1]['test_acc']:.3f}  "
                f"({time.time()-t0:.1f}s)"
            )
            records[arm_name][task_name] = rec

    # Endpoint computation.
    summary: Dict[str, float] = {}
    for arm_name, _ in arms:
        for task_name, _ in TASKS:
            key = f"{arm_name}_{task_name}"
            recs = records[arm_name][task_name]
            summary[f"{key}_test_acc_final"] = recs[-1]["test_acc"]
            summary[f"{key}_test_acc_latemean"] = _late_window_mean(recs, "test_acc")
            summary[f"{key}_time_to_test_acc_50"] = _time_to_threshold(recs, "test_acc", 0.5)
            summary[f"{key}_time_to_test_acc_95"] = _time_to_threshold(recs, "test_acc", 0.95)

    # The primary endpoint: interaction in late-mean test_acc.
    delta_t1 = (
        summary[f"{a2_name}_T1_ring_test_acc_latemean"]
        - summary[f"{a1_name}_T1_ring_test_acc_latemean"]
    )
    delta_t2 = (
        summary[f"{a2_name}_T2_strong_test_acc_latemean"]
        - summary[f"{a1_name}_T2_strong_test_acc_latemean"]
    )
    summary["delta_T1_ring"] = delta_t1
    summary["delta_T2_strong"] = delta_t2
    summary["interaction"] = delta_t1 - delta_t2  # primary endpoint
    summary["a1_name"] = a1_name
    summary["a2_name"] = a2_name

    out = {
        "n": int(n),
        "seed": int(seed),
        "steps": int(steps),
        "cubic_alpha": float(cubic_alpha),
        "baseline_mode": baseline_mode,
        "head_ng_eps": float(head_ng_eps),
        "cubic_form": cubic_form,
        "hidden": int(hidden),
        "records": records,
        "summary": summary,
    }
    return out


def plot_layer3(n: int, result: dict):
    if not HAVE_MPL:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    s = result["summary"]
    a1_name = s["a1_name"]
    a2_name = s["a2_name"]
    palette = {a1_name: "tab:blue", a2_name: "tab:green"}
    linestyle = {"T1_ring": "-", "T2_strong": "--"}
    for ax_idx, (key, ylabel) in enumerate(
        [
            ("test_acc", "test accuracy"),
            ("train_loss", "train loss"),
        ]
    ):
        ax = axes[ax_idx]
        for arm_name in [a1_name, a2_name]:
            for task_name in ["T1_ring", "T2_strong"]:
                recs = result["records"][arm_name][task_name]
                steps = [r["step"] for r in recs]
                ys = [r[key] for r in recs]
                ax.plot(
                    steps,
                    ys,
                    linestyle[task_name],
                    color=palette[arm_name],
                    label=f"{arm_name} x {task_name}",
                    alpha=0.8,
                )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        if key == "train_loss":
            ax.set_yscale("log")
    axes[-1].set_xlabel("training step")
    fig.suptitle(
        f"Paper 34 Layer 3 (Section 8) -- n = {n}, seed = {result['seed']}, "
        f"cubic_alpha = {result['cubic_alpha']}, baseline = {result.get('baseline_mode', 'adam')}\n"
        f"interaction = (A2-A1)_T1 - (A2-A1)_T2_strong = "
        f"{s['delta_T1_ring']:+.3f} - {s['delta_T2_strong']:+.3f} = "
        f"{s['interaction']:+.3f}  "
        f"(>0 confirms Conjecture 5.8 on this instance)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"paper34_layer3_n{n}_seed{result['seed']}.png", dpi=130)
    plt.close(fig)


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _sample_std(values: List[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mu = _mean(values)
    return float(math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1)))


def _format_std(values: List[float]) -> str:
    return "n/a" if len(values) < 2 else f"{_sample_std(values):.3f}"


def _summary_form(results: List[dict], tag: str = "") -> str:
    for r in results:
        if r.get("cubic_form"):
            return str(r["cubic_form"])
    return SUMMARY_TAG_TO_FORM.get(tag, "newton")


def emit_layer3_summary(results: List[dict], out_path: Path = None, tag: str = "") -> None:
    if not results:
        raise ValueError("cannot emit an empty Layer 3 summary")
    results = sorted(results, key=lambda r: (r["n"], r["seed"]))
    form = _summary_form(results, tag)
    deltas_t1 = [float(r["summary"]["delta_T1_ring"]) for r in results]
    deltas_t2 = [float(r["summary"]["delta_T2_strong"]) for r in results]
    interactions = [float(r["summary"]["interaction"]) for r in results]
    md = [
        "# Paper 34 -- Layer 3 (Section 8) Cubic-Aware Experiment Summary",
        "",
        FORM_DESCRIPTIONS.get(form, f"Cubic form: {form}."),
        "T1 = (a+b) mod n; T2-strong = random function f(a,b).",
        "",
        "## Per-run summary",
        "",
        "| $n$ | seed | $\\alpha_{\\mathrm{cubic}}$ | A1xT1 | A2xT1 | $\\Delta_{T1}$ | A1xT2s | A2xT2s | $\\Delta_{T2s}$ | **interaction** |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        s = r["summary"]
        a1 = s["a1_name"]
        a2 = s["a2_name"]
        md.append(
            f"| {r['n']} | {r['seed']} | {r['cubic_alpha']} "
            f"| {s[f'{a1}_T1_ring_test_acc_latemean']:.3f} "
            f"| {s[f'{a2}_T1_ring_test_acc_latemean']:.3f} "
            f"| {s['delta_T1_ring']:+.3f} "
            f"| {s[f'{a1}_T2_strong_test_acc_latemean']:.3f} "
            f"| {s[f'{a2}_T2_strong_test_acc_latemean']:.3f} "
            f"| {s['delta_T2_strong']:+.3f} "
            f"| **{s['interaction']:+.3f}** |"
        )
    md += [
        "",
        "## Aggregate summary",
        "",
        "| statistic | runs | $\\Delta_{T1}$ | $\\Delta_{T2s}$ | **interaction** |",
        "|---|---:|---:|---:|---:|",
        f"| mean | {len(results)} | {_mean(deltas_t1):+.3f} | {_mean(deltas_t2):+.3f} | **{_mean(interactions):+.3f}** |",
        f"| std | {len(results)} | {_format_std(deltas_t1)} | {_format_std(deltas_t2)} | {_format_std(interactions)} |",
    ]
    md += [
        "",
        "## Pre-specified branches",
        "",
        "**Branch A (Conjecture 5.8 confirmed on tested instance):** ",
        "interaction > 0, with $\\Delta_{T1} > 0$ and $\\Delta_{T2s} \\approx 0$ within noise.",
        "",
        "**Branch B (refutation):** interaction $\\leq 0$ within noise, or $\\Delta_{T2s}$ ",
        "is positive with magnitude comparable to $\\Delta_{T1}$. Theorem 3.5, ",
        "Corollary 3.3, the diagnostic of Section 6, and the off-centroid expansion of ",
        "Section 5 stand regardless. Conjecture 5.8 is reported as not supported by the ",
        "experiment on the tested instance.",
        "",
    ]
    target = out_path if out_path is not None else REPORTS_DIR / "paper34_layer3_summary.md"
    target.write_text("\n".join(md), encoding="utf-8")


def load_layer3_results(ns: List[int], seeds: List[int], tag_suffix: str) -> List[dict]:
    results: List[dict] = []
    missing: List[Path] = []
    for n in ns:
        for seed in seeds:
            path = REPORTS_DIR / f"paper34_layer3_n{n}_seed{seed}{tag_suffix}.json"
            if not path.exists():
                missing.append(path)
                continue
            results.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        missing_list = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing Layer 3 source JSON files:\n{missing_list}")
    return results


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rings",
        type=str,
        default="30",
        help="comma-separated n values (default: 30, where the architecture groks on D1)",
    )
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="comma-separated seeds (default: 0,1,2 for 3-seed validation)",
    )
    p.add_argument(
        "--cubic_alpha",
        type=float,
        default=0.1,
        help="strength of cubic correction relative to ||u||",
    )
    p.add_argument(
        "--baseline_mode",
        type=str,
        default="adam",
        choices=["adam", "head_ng"],
        help="A1 baseline: 'adam' (plain AdamW) or 'head_ng' (head exact-Fisher NG + Adam on rest).",
    )
    p.add_argument(
        "--head_ng_eps",
        type=float,
        default=0.1,
        help="regularizer in head NG (1/(p+eps)); larger eps avoids blow-up on memorized directions",
    )
    p.add_argument(
        "--cubic_form",
        type=str,
        default="newton",
        choices=["newton", "neumann", "within_packet"],
        help=(
            "cubic correction form: 'newton' = T(u, u, .)|_cross-packet; "
            "'neumann' = T(h, u, .)|_cross-packet; "
            "'within_packet' = project u onto dominant packet, amplify (sign-reversed)"
        ),
    )
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="optional tag appended to output filenames to keep runs separate",
    )
    p.add_argument(
        "--summarize_only",
        action="store_true",
        help="regenerate the summary Markdown from existing JSON artifacts without training",
    )
    return p.parse_args(argv)


def main(argv: List[str] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    ns = [int(x) for x in args.rings.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    tag_suffix = f"_{args.tag}" if args.tag else ""
    if args.summarize_only:
        all_results = load_layer3_results(ns, seeds, tag_suffix)
        summary_path = REPORTS_DIR / f"paper34_layer3_summary{tag_suffix}.md"
        emit_layer3_summary(all_results, out_path=summary_path, tag=args.tag)
        print(f"\n[done] summary regenerated at {summary_path}")
        return 0

    all_results: List[dict] = []
    for n in ns:
        for seed in seeds:
            r = run_experiment(
                n=n,
                seed=seed,
                steps=args.steps,
                cubic_alpha=args.cubic_alpha,
                baseline_mode=args.baseline_mode,
                head_ng_eps=args.head_ng_eps,
                cubic_form=args.cubic_form,
            )
            all_results.append(r)
            plot_layer3(n, r)
            (REPORTS_DIR / f"paper34_layer3_n{n}_seed{seed}{tag_suffix}.json").write_text(
                json.dumps(r, indent=2)
            )
    summary_path = REPORTS_DIR / f"paper34_layer3_summary{tag_suffix}.md"
    emit_layer3_summary(all_results, out_path=summary_path, tag=args.tag)
    print(f"\n[done] artifacts in {REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
