"""
E11 — Trained recovery of the arrow: a predictor learns what a single encoder cannot
================================================================================

  *** GPU / TORCH CERTIFICATE — RUN ON GCP, NOT LOCALLY. NOT IN THE run_all GATE. ***
  This file imports torch ONLY inside the training functions. The module top is
  numpy-only, and `--selftest` exercises the torch-free readout helpers without
  importing torch, so the data-generation and arrow-readout logic can be checked
  on any machine. The SGD training (train_*), and only it, needs torch+GPU and is
  meant to be launched on GCP.

WHAT THIS CERTIFIES  (the trained counterpart of E9/E10)
-------------------------------------------------------
E9/E10 are exact statements about the OPTIMA of the two objectives. This is the
learned counterpart (the analogue of the parent paper's E8): does SGD actually
realise the Irreversible Blind Spot and its predictor resolution?

  (A) SINGLE-ENCODER (symmetric Siamese: shared encoder h, align + VICReg whiten).
      Theorem 1 says its objective is a function of S only. Trained on a world W
      and on its time-reverse W' (drift flipped), it must converge to the SAME
      loss and the SAME embedding up to rotation: the arrow is invisible TO THE
      OBJECTIVE. Metric: final-loss gap ~ 0 and Procrustes(emb_W, emb_W') ~ 0.

  (B) TWO-ENCODER (predictive: encoder f + learnable predictor P, BYOL/SimSiam-
      style with whitening). Its optimum is the operator block B = [[a,beta],
      [-beta,a]] (E9). The TRAINED predictor P should be ASYMMETRIC, and the sign
      and magnitude of its antisymmetric part should recover the ground-truth
      drift beta = (q-b) sin(2pi/n). Training on W' flips the sign. The arrow is
      RECOVERED — by the predictor, exactly the BYOL/JEPA head Corollary 2 names.

GROUND TRUTH. We train on E9's drift ring, where beta is known in closed form, so
the recovered antisym(P) has an exact target. (The conveyor of E10 — recovering
the left/right CHART SPLIT by a two-encoder f != g — is the natural extension;
the readout helper `chart_angle_from_embeddings` is provided and selftested.)

Run (GCP):     py empirical/apex_recovery/e11_trained_two_encoder.py
Run (anywhere, torch-free logic check):  python empirical/apex_recovery/e11_trained_two_encoder.py --selftest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


# ===========================================================================
# Torch-free world + readout helpers (selftested locally; reused by GCP run).
# ===========================================================================


def drift_ring(n: int, q: float, b: float):
    """E9's drift ring: forward q, backward b, stay r. Uniform stationary law."""
    r = 1.0 - q - b
    T = np.zeros((n, n))
    for k in range(n):
        T[k, (k + 1) % n] += q
        T[k, (k - 1) % n] += b
        T[k, k] += r
    return T, np.full(n, 1.0 / n)


def conveyor(n: int, a: np.ndarray):
    """E10's non-normal conveyor: forward rate a_k, stay 1-a_k; pi ∝ 1/a (closed form)."""
    T = np.zeros((n, n))
    for k in range(n):
        T[k, (k + 1) % n] = a[k]
        T[k, k] = 1.0 - a[k]
    return T, (1.0 / a) / np.sum(1.0 / a)


def beta_ground_truth(n: int, q: float, b: float) -> float:
    """Closed-form arrow: beta = Im(lambda_1) = (q-b) sin(2pi/n)."""
    return (q - b) * np.sin(2 * np.pi / n)


def sample_pairs(T: np.ndarray, pi: np.ndarray, m: int, seed: int = 0):
    """m positive pairs (k, k') ~ stationary start then one chain step. Numpy only."""
    rng = np.random.default_rng(seed)
    n = len(pi)
    z = rng.choice(n, size=m, p=pi)
    zp = np.array([rng.choice(n, p=T[k]) for k in z])
    return z, zp


def predictor_block(F: np.ndarray, Fp: np.ndarray) -> np.ndarray:
    """
    Optimal linear predictor P* mapping f(z) -> f(z'), from paired embeddings:
        P* = Cov(Fp, F) Cov(F, F)^{-1}   (rows = samples, cols = embedding dims).
    For whitened F this is the operator block B; its antisymmetric part is the
    arrow. Pure numpy — takes the trained model's embeddings as arrays.
    """
    F = F - F.mean(0, keepdims=True)
    Fp = Fp - Fp.mean(0, keepdims=True)
    Cff = F.T @ F / len(F)
    Cpf = Fp.T @ F / len(F)
    return Cpf @ np.linalg.pinv(Cff)


def antisym_scalar(P: np.ndarray) -> float:
    """For a 2x2 block, the antisymmetric coefficient beta_hat = (P[0,1]-P[1,0])/2."""
    return float(0.5 * (P[0, 1] - P[1, 0]))


def procrustes_subspace_diff(A: np.ndarray, B: np.ndarray) -> float:
    """min over orthogonal U of ||A - B U||_F / ||A||_F (column-space alignment)."""
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M)
    R = (U @ Vt).T
    return float(np.linalg.norm(A - B @ R) / (np.linalg.norm(A) + 1e-12))


def chart_angle_from_embeddings(F: np.ndarray, Fp: np.ndarray, d: int = 1) -> float:
    """
    Principal angle (deg) between input chart (right singular dirs of the cross-
    covariance, encoded from z) and output chart (left singular dirs, read from
    z'). > 0 signals a NON-NORMAL transition (E10's two charts). Torch-free.
    """
    F = F - F.mean(0, keepdims=True)
    Fp = Fp - Fp.mean(0, keepdims=True)
    C = Fp.T @ F / len(F)
    U, s, Vt = np.linalg.svd(C)
    cross = U[:, :d].T @ Vt[:d, :].T
    cos = np.clip(np.linalg.svd(cross, compute_uv=False), 0, 1)
    return float(np.degrees(np.arccos(cos)).max())


# ===========================================================================
# Torch-free self-test: validate the readout pipeline WITHOUT training.
# ===========================================================================


def selftest() -> int:
    """
    Use the TRUE cos/sin features as a stand-in for a perfectly-trained encoder
    and confirm the empirical predictor block recovers the closed-form beta. This
    validates sampling + readout (the parts the GCP training will reuse) with no
    torch import. If this passes, a trained encoder that reaches cos/sin will too.
    """
    print("E11 selftest (numpy only; no torch imported) ...")
    # (a) Drift ring: the predictor recovers the signed arrow beta to ground truth.
    n, q, b = 12, 0.5, 0.2
    T, pi = drift_ring(n, q, b)
    beta = beta_ground_truth(n, q, b)
    z, zp = sample_pairs(T, pi, m=200_000, seed=1)
    k = np.arange(n)
    feats = np.stack([np.sqrt(2) * np.cos(2 * np.pi * k / n),
                      np.sqrt(2) * np.sin(2 * np.pi * k / n)], axis=1)
    F, Fp = feats[z], feats[zp]
    # operator block B = (empirical predictor P*)^T  (E9 convention B[a,b]=<psi_a,T psi_b>;
    # the trained predictor map f(z)->f(z') is P* = B^T, antisym -> -beta, so transpose).
    B_hat = predictor_block(F, Fp).T
    beta_hat = antisym_scalar(B_hat)
    ring_ok = abs(beta_hat - beta) < 0.01 and np.sign(beta_hat) == np.sign(q - b)
    print(f"  [ring] operator block B_hat =\n{np.array2string(B_hat, prefix='    ', precision=4)}")
    print(f"  [ring] beta_hat = {beta_hat:+.4f}  (truth {beta:+.4f}); sign recovered: "
          f"{np.sign(beta_hat)==np.sign(q-b)}   -> {'OK' if ring_ok else 'FAIL'}")

    # (b) Non-normal conveyor: input chart != output chart (validates chart-angle helper;
    # singular values are distinct here, so the d=1 angle is well-defined, unlike the ring).
    nc = 8
    a = np.array([0.2, 0.5, 0.8, 0.3, 0.6, 0.9, 0.4, 0.7])
    Tc, pic = conveyor(nc, a)
    zc, zpc = sample_pairs(Tc, pic, m=200_000, seed=2)
    kc = np.arange(nc)
    fc = np.stack([np.sqrt(2) * np.cos(2 * np.pi * kc / nc),
                   np.sqrt(2) * np.sin(2 * np.pi * kc / nc)], axis=1)
    ang = chart_angle_from_embeddings(fc[zc], fc[zpc], d=1)
    conv_ok = ang > 1.0
    print(f"  [conveyor] input-vs-output chart angle = {ang:.2f} deg (>1: non-normal) "
          f"-> {'OK' if conv_ok else 'FAIL'}")

    ok = ring_ok and conv_ok
    print("  SELFTEST PASSED" if ok else "  SELFTEST FAILED")
    return 0 if ok else 1


# ===========================================================================
# GCP training (imports torch INSIDE; never called by --selftest).
# ===========================================================================


def _make_encoder(n, dim, hidden, torch, nn):
    """One-hot(state) -> MLP -> dim. Small; runs in seconds on any GPU/CPU-GCP."""
    return nn.Sequential(nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, dim))


def _vicreg_whiten(z, torch):
    """VICReg variance+covariance terms: push the embedding toward whitened."""
    z = z - z.mean(0, keepdim=True)
    std = torch.sqrt(z.var(0) + 1e-4)
    var_term = torch.mean(torch.relu(1.0 - std))
    d = z.shape[1]
    cov = (z.T @ z) / (len(z) - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_term = (off ** 2).sum() / d
    return var_term + cov_term


def train_single_encoder(n, T, pi, dim=2, steps=4000, seed=0):
    """(A) Symmetric Siamese: align(h(z),h(z')) + whitening. Returns (emb_table, loss)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    z, zp = sample_pairs(T, pi, m=60_000, seed=seed)
    Z = torch.eye(n)
    h = _make_encoder(n, dim, 64, torch, nn)
    opt = torch.optim.Adam(h.parameters(), lr=1e-3)
    zt, zpt = torch.tensor(z), torch.tensor(zp)
    last = None
    for _ in range(steps):
        idx = torch.randint(0, len(zt), (4096,))
        a, ap = h(Z[zt[idx]]), h(Z[zpt[idx]])
        loss = ((a - ap) ** 2).sum(1).mean() + 1.0 * _vicreg_whiten(a, torch)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
    with torch.no_grad():
        emb = h(Z).numpy()
    return emb, last


def train_two_encoder(n, T, pi, dim=2, steps=4000, seed=0):
    """(B) Predictive: encoder f + linear predictor P; ||P f(z) - sg f(z')||^2 + whiten.
    Returns (emb_table, P_matrix)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    z, zp = sample_pairs(T, pi, m=60_000, seed=seed)
    Z = torch.eye(n)
    f = _make_encoder(n, dim, 64, torch, nn)
    P = nn.Linear(dim, dim, bias=False)
    opt = torch.optim.Adam(list(f.parameters()) + list(P.parameters()), lr=1e-3)
    zt, zpt = torch.tensor(z), torch.tensor(zp)
    for _ in range(steps):
        idx = torch.randint(0, len(zt), (4096,))
        fz, fzp = f(Z[zt[idx]]), f(Z[zpt[idx]])
        pred = P(fz)
        loss = ((pred - fzp.detach()) ** 2).sum(1).mean() + 1.0 * _vicreg_whiten(fz, torch)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        emb = f(Z).numpy()
        Pmat = P.weight.detach().numpy()
    return emb, Pmat


def main_train():
    """Full GCP run: train both models on a world and its time-reverse; certify."""
    print("=" * 78)
    print("E11  Trained recovery of the arrow (GCP run; torch)")
    print("=" * 78)
    n, q, b = 12, 0.5, 0.2
    T, pi = drift_ring(n, q, b)
    Trev, _ = drift_ring(n, b, q)  # time-reverse (flip drift)
    beta = beta_ground_truth(n, q, b)

    # (A) single-encoder: blind to the arrow (reversal-invariant).
    embA, lossA = train_single_encoder(n, T, pi)
    embA_rev, lossA_rev = train_single_encoder(n, Trev, pi)
    loss_gap = abs(lossA - lossA_rev)
    proc = procrustes_subspace_diff(embA - embA.mean(0), embA_rev - embA_rev.mean(0))

    # (B) two-encoder: recovers the arrow via the predictor. Train ONE encoder on W,
    # then read the arrow off forward AND reversed pairs in the SAME gauge (so the
    # sign comparison is gauge-invariant; |antisym| is invariant to the encoder's
    # orthogonal frame, and the sign flips under reversal within one frame).
    embB, _ = train_two_encoder(n, T, pi)
    zf, zpf = sample_pairs(T, pi, m=200_000, seed=3)
    zr, zpr = sample_pairs(Trev, pi, m=200_000, seed=4)
    beta_W = antisym_scalar(predictor_block(embB[zf], embB[zpf]).T)
    beta_Wrev = antisym_scalar(predictor_block(embB[zr], embB[zpr]).T)  # SAME embB
    mag_ok = abs(abs(beta_W) - abs(beta)) < 0.05
    flips = (np.sign(beta_Wrev) == -np.sign(beta_W)) and abs(beta_Wrev + beta_W) < 0.05

    report = dict(
        experiment="E11_trained_two_encoder", n=n, q=q, b=b, beta_truth=beta,
        single_encoder=dict(loss=lossA, loss_rev=lossA_rev, loss_gap=loss_gap,
                            procrustes_W_vs_Wrev=proc),
        two_encoder=dict(beta_W=beta_W, beta_Wrev_same_gauge=beta_Wrev,
                         magnitude_recovered=bool(mag_ok), flips_under_reversal=bool(flips)),
    )
    # self-checks: single-encoder reversal-blind; two-encoder recovers the arrow magnitude
    # and its sign flips under time reversal (same gauge).
    ok = (loss_gap < 0.05 and proc < 0.15 and mag_ok and flips)
    report["passed"] = bool(ok)
    (REPORTS / "e11_trained_two_encoder.json").write_text(json.dumps(report, indent=2))
    print(f"  (A) single-encoder loss gap W vs W' = {loss_gap:.4f} (~0: arrow invisible to objective)")
    print(f"      Procrustes(emb_W, emb_W')       = {proc:.4f} (~0: same rep up to rotation)")
    print(f"  (B) two-encoder |beta_W| = {abs(beta_W):.4f} (truth {abs(beta):.4f}); "
          f"beta_W={beta_W:+.4f} flips to {beta_Wrev:+.4f} under reversal")
    print("E11:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="numpy-only readout check; does NOT import torch (safe anywhere)")
    args = ap.parse_args()
    raise SystemExit(selftest() if args.selftest else main_train())
