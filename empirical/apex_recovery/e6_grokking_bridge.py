"""
E6 -- Grokking as the two-register boundary, crossed in time   (GPU / cloud)
================================================================================

THE CLAIM
---------
Grokking on modular addition is the order-2 -> order-3 boundary of this paper,
traversed DURING TRAINING. The memorising solution is order-2-sufficient (a
lookup code with no ring structure); the generalising "Fourier circuit" is the
order-3 cubic (the selection rule k+l+m=0 mod n, = the Amari-Chentsov cubic of
the Conductor Blind Spot). Pre-registered prediction:

    at the grokking step BOTH registers rise together and track val accuracy,
    and BOTH stay flat on a structure-free control.

THREE THINGS ARE LOGGED EACH STEP (all defined in e6_diagnostics.py):

  D  (dynamical)        embedding Fourier concentration C_F
        the token embedding aligning to the ring's slow eigenfunctions.

  S  (distributional, cheap proxy)   logit additivity R2_add
        share of the logit tensor explained by the sum-class s=(a+b) mod n.

  rho_x (distributional, CBS Definition 6.1)
        cross-packet cubic mass of the batch-averaged centered score on Z/nZ --
        the ACTUAL Conductor-Blind-Spot diagnostic. This is the object of CBS
        Conjecture 5.8 (does the cross-packet cubic emerge along a training
        trajectory and stay null on a control?). The published paper closed only
        the analytic half (Remark 5.9); this run is the trajectory half.

PRIME vs COMPOSITE -- read this before choosing --p:
  * rho_x is identically 0 for a PRIME modulus (every nonzero character has the
    same conductor p, so there are no cross-packet triples). Prime p (113) is the
    canonical grokking testbed and gives the clean D/S co-emergence.
  * the rho_x / Conjecture-5.8 test needs a COMPOSITE n (12 or 30, the CBS §6.5
    ladder). So the full experiment runs BOTH: prime for D/S, composite for rho_x.

RAW LOGGING -- the GPU run is a DATA-CAPTURE, not the final analysis. Every row
also stores the raw centered score u (length n) and the embedding spectrum, so
any diagnostic can be recomputed / corrected OFFLINE (CPU, no GPU) from the
JSONL without paying for another cloud run.

AUTO-STOP -- once the task groks (train & val acc > 0.9) the run continues for
--stop_after_grok more steps (to capture the co-emergence + a little plateau)
and then halts, so we never pay for dead post-grok steps. The control never
groks and runs to --steps.

CONTROL: replace c=(a+b) mod n by a fixed random table c=R[a,b] (memorisable,
no group structure). Prediction: never groks; D, S, rho_x all flat. (--control.)

The figure is made OFFLINE by e6_plot.py from the JSONL (no matplotlib needed on
the VM). Our contribution is not the Fourier-circuit mechanism (Power et al.
2022; Nanda et al. 2023) but the reading: grokking is the cross-register bridge
of E4/E5 crossed in time, and the rho_x trajectory is CBS Conjecture 5.8 tested.

Run (GPU):
  python empirical/apex_recovery/e6_grokking_bridge.py --p 113                 # prime: D/S
  python empirical/apex_recovery/e6_grokking_bridge.py --p 113 --control
  python empirical/apex_recovery/e6_grokking_bridge.py --p 30                  # composite: rho_x
  python empirical/apex_recovery/e6_grokking_bridge.py --p 30  --control
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e6_diagnostics import (
    RhoX,
    embedding_fourier_concentration,
    logit_additivity,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# A minimal 1-layer transformer (the standard grokking architecture).
# Sequence [a, b, '='] -> predict c at the last position.
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, d, h, d_mlp):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d_mlp), nn.GELU(), nn.Linear(d_mlp, d))

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)   # full attention, 3 positions
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
        pos = torch.arange(self.seq, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        x = self.block(x)
        return self.unembed(x[:, -1])       # logits at '=' position: [B, p]


# ---------------------------------------------------------------------------
# Data: all (a,b) pairs; target (a+b)%n, or a fixed random table for control.
# ---------------------------------------------------------------------------


def make_data(p, control, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    X = torch.stack([a, b, eq], dim=1)
    if control:
        table = torch.randint(0, p, (p, p), generator=g)   # no group structure
        Y = table[a, b]
    else:
        Y = (a + b) % p
    return X.to(device), Y.to(device)


@torch.no_grad()
def accuracy(model, X, Y):
    return float((model(X).argmax(-1) == Y).float().mean())


@torch.no_grad()
def head_tensors(model, X, Y, p):
    """Forward once over the full ring; return (L_numpy, u_numpy):
        L  = logits  [N, p]
        u  = batch-averaged centered score on Z/n: mean_i (onehot(Y_i) - p_i),
             a tangent vector (sum_y u_y = 0) -- the input to CBS's rho_x."""
    L = model(X).float()
    P = torch.softmax(L, dim=-1)
    oh = F.one_hot(Y, p).float()
    u = (oh - P).mean(0)
    return L.cpu().numpy(), u.cpu().numpy()


def _r(x, k=6):
    return [round(float(v), k) for v in x]


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=113,
                    help="modulus; PRIME -> D/S co-emergence (rho_x==0); "
                         "COMPOSITE (12,30) -> rho_x / Conjecture-5.8 test")
    ap.add_argument("--frac_train", type=float, default=0.3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d_mlp", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=40000, help="hard step cap")
    ap.add_argument("--stop_after_grok", type=int, default=4000,
                    help="halt this many steps after grok is detected (0 = run "
                         "to --steps; the control never groks so always runs full)")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tag = "control" if args.control else "modadd"
    rho = RhoX(args.p)
    kind = "PRIME (rho_x degenerate -> 0)" if rho.n_cross == 0 else \
        f"COMPOSITE ({rho.n_cross} cross-packet triples)"
    print(f"E6 grokking bridge | task={tag} p={args.p} [{kind}] device={dev}")

    X, Y = make_data(args.p, args.control, dev, seed=args.seed)
    a_np = X[:, 0].cpu().numpy()
    b_np = X[:, 1].cpu().numpy()
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
            W = model.tok.weight.detach()[:args.p].float().cpu().numpy()
            cF_max, cF_top5, shares = embedding_fourier_concentration(W, args.p)
            L, u = head_tensors(model, X, Y, args.p)
            r2_add = logit_additivity(L, a_np, b_np, args.p)
            rx = rho(u)
            row = dict(step=step, train_loss=round(loss.item(), 6),
                       train_acc=tr_acc, val_acc=va_acc,
                       D_fourier_max=cF_max, D_fourier_top5=cF_top5,
                       S_logit_additivity=r2_add,
                       rho_x=rx["rho_x"], rho_x_cross_mass=rx["cross_mass"],
                       rho_x_total_mass=rx["total_mass"],
                       n_triples_cross=rx["n_triples_cross"],
                       sec=round(time.time() - t0, 1),
                       # raw vectors for offline recompute / correction:
                       u=_r(u), emb_shares=_r(shares))
            log.write(json.dumps(row) + "\n")
            log.flush()
            newly = grok_step is None and va_acc > 0.9 and tr_acc > 0.9
            if newly:
                grok_step = step
            rxs = "  rho_x --" if rho.n_cross == 0 else f"  rho_x {rx['rho_x']:.3f}"
            print(f"  step {step:6d} | tr {tr_acc:.3f} va {va_acc:.3f} | "
                  f"D {cF_max:.3f} | S {r2_add:.3f} |{rxs} | {row['sec']:.0f}s"
                  + ("  <== GROK" if newly else ""))

            if (grok_step is not None and args.stop_after_grok > 0
                    and step - grok_step >= args.stop_after_grok):
                print(f"  auto-stop: {args.stop_after_grok} steps past grok "
                      f"(grok_step={grok_step}); co-emergence captured.")
                break
    log.close()

    out = dict(experiment="E6_grokking_bridge", task=tag, p=args.p, device=dev,
               modulus_kind=("prime" if rho.n_cross == 0 else "composite"),
               n_triples_cross=rho.n_cross, grok_step=grok_step,
               steps_run=step, steps_cap=args.steps,
               wd=args.wd, lr=args.lr, frac_train=args.frac_train,
               metrics_jsonl=str(log_path),
               prediction=("D (embedding Fourier), S (logit additivity), and "
                           "rho_x (cross-packet cubic, composite n only) rise "
                           "together at the grok and track val acc; all flat on "
                           "the control. rho_x trajectory = CBS Conjecture 5.8."))
    (REPORTS / f"e6_{tag}_p{args.p}_summary.json").write_text(json.dumps(out, indent=2))
    print(f"done. grok_step={grok_step}  steps_run={step}  metrics -> {log_path}")


if __name__ == "__main__":
    main()
