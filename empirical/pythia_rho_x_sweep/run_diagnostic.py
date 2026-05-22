"""Run the rho_x diagnostic on one Pythia checkpoint, ring + permuted control.

Usage:
    python run_diagnostic.py --model EleutherAI/pythia-410m

Ring task (T1):
    Conditional categorical over the 12 month tokens, labels = correct
    next-month in calendar order, batch-averaged centered score.
Permuted control (T2-weak, post-hoc):
    Same model outputs and same prompts, but the 12-element ring index
    assignment is permuted by a random sigma in S_12. Repeated n_perms
    times and averaged. Destroys the additive-group structure on the
    diagnostic side.

Output: rho_x_ring vs rho_x_permuted (mean +/- std).
Prediction from Theorem 3.5 + Conjecture 5.8:
    rho_x_ring < rho_x_permuted on models that have learned the cycle.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rho_x import rho_x, packet_summary

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
N = 12


def build_prompts(ring_labels: list[str], context_lengths=(3, 5, 7, 9, 11)) -> list[tuple[str, int]]:
    """Cyclic-shift prompts. For each (k, i): prompt = ring[i:i+k] as
    space-joined string; label = (i + k) mod n. Returns (prompt, label_idx).
    """
    n = len(ring_labels)
    out = []
    for k in context_lengths:
        for i in range(n):
            seq = [ring_labels[(i + j) % n] for j in range(k)]
            prompt = " ".join(seq)
            label = (i + k) % n
            out.append((prompt, label))
    return out


def forward_batch(model, tok, prompts: list[str], device: str) -> torch.Tensor:
    """Returns (P, V) tensor of last-position logits, one row per prompt."""
    out = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :].float().cpu()
        out.append(logits)
    return torch.stack(out)


def diagnostic_for_assignment(
    logits: torch.Tensor,
    ids_ring: list[int],
    labels: list[int],
    n: int,
    permute: list[int] | None = None,
) -> dict:
    """logits: (P, V). ids_ring: original ring token IDs in calendar order.
    labels: per-prompt correct index in [0, n). permute: optional sigma
    applied to the ring index assignment (re-orders both ids_ring and labels).
    """
    if permute is not None:
        ids_used = [ids_ring[permute[i]] for i in range(n)]
        labels_used = [permute.index(y) for y in labels]
    else:
        ids_used = ids_ring
        labels_used = labels

    sub = logits[:, ids_used]  # (P, n)
    p_cond = torch.softmax(sub, dim=-1).numpy()
    P = p_cond.shape[0]
    u_batch = np.zeros((P, n))
    for i, y in enumerate(labels_used):
        u_batch[i, y] = 1.0
    u_batch -= p_cond
    u_mean = u_batch.mean(axis=0).tolist()
    return rho_x(u_mean, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--n-perms", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=("fp32", "fp16"), default="fp16")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print(f"=== {args.model}  (device={device}, dtype={args.dtype}) ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    print(f"loaded in {time.time() - t0:.1f}s  vram={torch.cuda.memory_allocated() / 1e9:.2f}GB")

    ids_bare = [tok.encode(m, add_special_tokens=False)[0] for m in MONTHS]
    ids_space = [tok.encode(" " + m, add_special_tokens=False)[0] for m in MONTHS]

    pairs = build_prompts(MONTHS)
    prompts = [p for p, _ in pairs]
    labels = [y for _, y in pairs]
    print(f"prompts: {len(prompts)}")

    t0 = time.time()
    logits = forward_batch(model, tok, prompts, device)
    print(f"forward time: {time.time() - t0:.1f}s for {len(prompts)} prompts")

    # Style selection: which V_ring captures more mass on the avg row?
    probs_avg = torch.softmax(logits, dim=-1).mean(dim=0)
    mb = probs_avg[ids_bare].sum().item()
    ms = probs_avg[ids_space].sum().item()
    style = "space" if ms > mb else "bare"
    ids = ids_space if style == "space" else ids_bare
    print(f"\nmass on V_ring (averaged): bare={mb:.4f}, space={ms:.4f} -> style={style}")

    # Top-1 accuracy diagnostic
    p_cond_avg = torch.softmax(logits[:, ids], dim=-1).numpy()
    top1 = p_cond_avg.argmax(axis=1)
    correct = sum(int(t == y) for t, y in zip(top1, labels))
    print(f"top-1 on V_ring conditional: {correct}/{len(prompts)} = {correct/len(prompts):.3f}")

    # T1 (ring) diagnostic
    res_ring = diagnostic_for_assignment(logits, ids, labels, N)
    print(f"\n[T1 ring]      rho_x = {res_ring['rho_x']:.4f}")

    # T2-weak (label permutation) diagnostic
    rng = np.random.default_rng(args.seed)
    rho_perms = []
    for _ in range(args.n_perms):
        perm = list(map(int, rng.permutation(N)))
        res = diagnostic_for_assignment(logits, ids, labels, N, permute=perm)
        rho_perms.append(res["rho_x"])
    rho_perms = np.array(rho_perms)
    print(f"[T2-weak perm] rho_x = {rho_perms.mean():.4f} +/- {rho_perms.std():.4f}  "
          f"(n={args.n_perms} perms)")

    sep = rho_perms.mean() - res_ring['rho_x']
    print(f"\nseparation (perm - ring) = {sep:+.4f}")
    print(f"prediction direction: positive (ring < perm) iff model has ring structure")
    print(f"uniform-baseline rho_x = {108/110:.4f} (from {res_ring['n_triples_cross']}/"
          f"{res_ring['n_triples_total']} cross-packet triples on n=12)")


if __name__ == "__main__":
    main()
