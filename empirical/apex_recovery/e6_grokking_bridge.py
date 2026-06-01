"""
E6 — Grokking as the two-register boundary, crossed in time  (GPU)
================================================================================

THE CLAIM
---------
Grokking on modular addition is the order-2 -> order-3 boundary of this paper,
traversed DURING TRAINING. The memorising solution is order-2-sufficient (a
lookup code with no ring structure); the generalising "Fourier circuit" is the
order-3 cubic (the selection rule k+l+m=0 mod p, = the Amari-Chentsov cubic of
the Conductor Blind Spot). The pre-registered prediction:

    at the grokking step, BOTH registers rise together and track val accuracy,
    and BOTH stay flat on a structure-free control:

      * DYNAMICAL register  D  = the token embedding aligning to the ring's slow
        eigenfunctions (its Fourier/character chart): "embedding Fourier
        concentration" C_F = max-over-frequency share of embedding variance.
        Pre-grok ~ uniform over frequencies (small); post-grok a few frequencies
        dominate (large) -- the curved character chart is acquired.

      * DISTRIBUTIONAL register  S  = the head acquiring the ring-additive cubic:
        "logit additivity" R2_add = share of the logit tensor explained by the
        sum class s=(a+b) mod p alone. Pre-grok the head depends on (a,b)
        jointly (a lookup); post-grok it depends only on a+b (the cubic
        constraint a+b-c=0). This is the order-3 content CBS's rho_x measures;
        R2_add is its cheap, exact, retraining-free proxy on this task.

CONTROL: replace c=(a+b) mod p by a fixed random table c=R[a,b] -- memorisable,
but no group structure to grok into. Prediction: never groks, both registers
flat. (Run with --control.)

This is the canonical grokking testbed (Power et al. 2022; the Fourier-circuit
mechanism is Nanda et al. 2023). Our contribution is not the mechanism but the
reading: grokking is the cross-register bridge of E4/E5 crossed in time, on the
field's hello-world. E4 = synthetic/continuous-limit twin; E5 = static real
model; E6 = the dynamical, in-training version.

Run (GPU):
  py empirical/apex_recovery/e6_grokking_bridge.py
  py empirical/apex_recovery/e6_grokking_bridge.py --control
Writes a metrics JSONL (one row per logged step) + a final JSON/figure to reports/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# A minimal 1-layer transformer (the standard grokking architecture).
# Sequence is [a, b, '='] -> predict c at the last position.
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, d, h, d_mlp):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d_mlp), nn.GELU(), nn.Linear(d_mlp, d))

    def forward(self, x):
        # causal-free full attention over the 3 positions
        a, _ = self.attn(x, x, x, need_weights=False)
        x = x + a
        x = x + self.mlp(x)
        return x


class GrokFormer(nn.Module):
    def __init__(self, p, d=128, h=4, d_mlp=512, seq=3):
        super().__init__()
        self.p = p
        self.tok = nn.Embedding(p + 1, d)   # tokens 0..p-1 plus '=' = index p
        self.pos = nn.Embedding(seq, d)
        self.block = Block(d, h, d_mlp)
        self.unembed = nn.Linear(d, p, bias=False)
        self.seq = seq

    def forward(self, idx):
        # idx: [B, 3]
        pos = torch.arange(self.seq, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        x = self.block(x)
        return self.unembed(x[:, -1])       # logits at '=' position: [B, p]


# ---------------------------------------------------------------------------
# Data: all (a,b) pairs; target (a+b)%p, or a fixed random table for control.
# ---------------------------------------------------------------------------


def make_data(p, control, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)              # '=' token
    X = torch.stack([a, b, eq], dim=1)
    if control:
        table = torch.randint(0, p, (p, p), generator=g)  # no group structure
        Y = table[a, b]
    else:
        Y = (a + b) % p
    return X.to(device), Y.to(device)


# ---------------------------------------------------------------------------
# The two registers.
# ---------------------------------------------------------------------------


def embedding_fourier_concentration(model):
    """D-register: max share of the number-token embedding variance held by a
    single ring frequency (cos_k, sin_k). Uniform-ish pre-grok, peaked post-grok."""
    p = model.p
    W = model.tok.weight.detach()[:p].float().cpu().numpy()   # [p, d]
    W = W - W.mean(0, keepdims=True)
    total = float((W**2).sum()) + 1e-12
    n = np.arange(p)
    shares = []
    for k in range(1, p // 2 + 1):
        c = np.cos(2 * np.pi * k * n / p)
        s = np.sin(2 * np.pi * k * n / p)
        c /= np.linalg.norm(c) + 1e-12
        s /= np.linalg.norm(s) + 1e-12
        ek = float((W.T @ c) @ (W.T @ c) + (W.T @ s) @ (W.T @ s))
        shares.append(ek / total)
    shares = np.sort(shares)[::-1]
    return float(shares[0]), float(shares[:5].sum())


def logit_additivity(model, X, Y, p):
    """S-register: share of the logit tensor explained by the sum class s=(a+b)%p
    alone (R^2 of grouping logits by s). Low pre-grok (lookup over (a,b)),
    -> 1 post-grok (head depends only on a+b: the cubic constraint a+b-c=0)."""
    with torch.no_grad():
        L = model(X).float()                       # [p^2, p]
    a = X[:, 0]
    b = X[:, 1]
    s = (a + b) % p                                # [p^2]
    Lc = L - L.mean(0, keepdim=True)
    total = float((Lc**2).sum()) + 1e-12
    # within-sum-class residual: subtract each s-class mean
    resid = torch.zeros_like(L)
    for sv in range(p):
        m = s == sv
        if m.any():
            resid[m] = L[m] - L[m].mean(0, keepdim=True)
    within = float((resid**2).sum())
    return 1.0 - within / total                    # R^2 explained by s-class


@torch.no_grad()
def accuracy(model, X, Y):
    pred = model(X).argmax(-1)
    return float((pred == Y).float().mean())


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--frac_train", type=float, default=0.3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d_mlp", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tag = "control" if args.control else "modadd"
    print(f"E6 grokking bridge | task={tag} p={args.p} device={dev}")

    X, Y = make_data(args.p, args.control, dev, seed=args.seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_tr = int(args.frac_train * n)
    tr, va = perm[:n_tr].to(dev), perm[n_tr:].to(dev)
    Xtr, Ytr, Xva, Yva = X[tr], Y[tr], X[va], Y[va]
    print(f"  examples={n}  train={n_tr}  val={n - n_tr}")

    model = GrokFormer(args.p, args.d, args.heads, args.d_mlp).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.98))

    log_path = REPORTS / f"e6_{tag}_p{args.p}_metrics.jsonl"
    log = log_path.open("w")
    t0 = time.time()
    grok_step = None
    for step in range(args.steps + 1):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr), Ytr)
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            model.eval()
            tr_acc = accuracy(model, Xtr, Ytr)
            va_acc = accuracy(model, Xva, Yva)
            cF_max, cF_top5 = embedding_fourier_concentration(model)
            r2_add = logit_additivity(model, X, Y, args.p)
            row = dict(step=step, train_loss=loss.item(), train_acc=tr_acc,
                       val_acc=va_acc, D_fourier_max=cF_max, D_fourier_top5=cF_top5,
                       S_logit_additivity=r2_add, sec=round(time.time() - t0, 1))
            log.write(json.dumps(row) + "\n")
            log.flush()
            if grok_step is None and va_acc > 0.9 and tr_acc > 0.9:
                grok_step = step
            print(f"  step {step:6d} | tr {tr_acc:.3f} va {va_acc:.3f} | "
                  f"D(C_F max) {cF_max:.3f} | S(add R2) {r2_add:.3f} | "
                  f"{row['sec']:.0f}s" + ("  <== GROK" if grok_step == step else ""))
    log.close()

    out = dict(experiment="E6_grokking_bridge", task=tag, p=args.p, device=dev,
               grok_step=grok_step, steps=args.steps, metrics_jsonl=str(log_path),
               prediction=("D (embedding Fourier concentration) and S (logit "
                           "additivity) rise together at the grok and track val "
                           "acc; both flat on the control"))
    (REPORTS / f"e6_{tag}_p{args.p}_summary.json").write_text(json.dumps(out, indent=2))
    print(f"done. grok_step={grok_step}. metrics -> {log_path}")


if __name__ == "__main__":
    main()
