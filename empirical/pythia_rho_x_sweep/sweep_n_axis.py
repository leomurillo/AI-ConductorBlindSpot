"""Run the diagnostic across Pythia model sizes (N-axis). Saves JSON.

Adds a per-example variant (mean over examples of single-prompt rho_x)
alongside the §6.1 batch-mean variant, because on pretrained LLMs the
batch-mean cancels when the model is either uniformly accurate or
uniformly random and the per-example version preserves the signal.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rho_x import rho_x

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
N = 12

PYTHIA_SIZES = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]


def build_prompts(ring_labels):
    n = len(ring_labels)
    out = []
    for k in (3, 5, 7, 9, 11):
        for i in range(n):
            seq = [ring_labels[(i + j) % n] for j in range(k)]
            prompt = " ".join(seq)
            label = (i + k) % n
            out.append((prompt, label))
    return out


def forward_all(model, tok, prompts, device):
    out = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :].float().cpu()
        out.append(logits)
    return torch.stack(out)


def diagnostic(logits, ids_ring, labels, n, permute=None, mode="batch_mean"):
    """mode='batch_mean' implements Def 6.1 (Section 6 of the paper).
    mode='per_example' computes rho_x per row and averages -- not in the paper
    but more robust on pretrained-LLM heads where per-prompt phases cancel.
    """
    if permute is not None:
        ids_used = [ids_ring[permute[i]] for i in range(n)]
        labels_used = [permute.index(y) for y in labels]
    else:
        ids_used, labels_used = ids_ring, labels

    sub = logits[:, ids_used]
    p_cond = torch.softmax(sub, dim=-1).numpy()
    P = p_cond.shape[0]
    u_batch = np.zeros((P, n))
    for i, y in enumerate(labels_used):
        u_batch[i, y] = 1.0
    u_batch -= p_cond

    if mode == "batch_mean":
        u_mean = u_batch.mean(axis=0).tolist()
        return rho_x(u_mean, n)
    elif mode == "per_example":
        rho_vals = []
        for i in range(P):
            r = rho_x(u_batch[i].tolist(), n)
            if r["total_mass"] > 0:
                rho_vals.append(r["rho_x"])
        return {
            "rho_x": float(np.mean(rho_vals)) if rho_vals else float("nan"),
            "rho_x_std": float(np.std(rho_vals)) if rho_vals else float("nan"),
            "n_used": len(rho_vals),
        }
    else:
        raise ValueError(mode)


def run_one(model_short, n_perms, seed, device):
    model_name = f"EleutherAI/{model_short}"
    print(f"\n=== {model_name} ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device).eval()
    load_time = time.time() - t0
    vram = torch.cuda.memory_allocated() / 1e9

    ids_bare = [tok.encode(m, add_special_tokens=False)[0] for m in MONTHS]
    ids_space = [tok.encode(" " + m, add_special_tokens=False)[0] for m in MONTHS]

    pairs = build_prompts(MONTHS)
    prompts = [p for p, _ in pairs]
    labels = [y for _, y in pairs]

    t0 = time.time()
    logits = forward_all(model, tok, prompts, device)
    fwd_time = time.time() - t0

    probs_avg = torch.softmax(logits, dim=-1).mean(dim=0)
    mb = probs_avg[ids_bare].sum().item()
    ms = probs_avg[ids_space].sum().item()
    style = "space" if ms > mb else "bare"
    ids = ids_space if style == "space" else ids_bare

    p_cond_avg = torch.softmax(logits[:, ids], dim=-1).numpy()
    top1 = p_cond_avg.argmax(axis=1)
    top1_acc = float(np.mean(top1 == np.array(labels)))

    # Batch-mean Def 6.1
    res_bm_ring = diagnostic(logits, ids, labels, N, mode="batch_mean")
    rng = np.random.default_rng(seed)
    rho_bm_perms = []
    for _ in range(n_perms):
        perm = list(map(int, rng.permutation(N)))
        rho_bm_perms.append(diagnostic(logits, ids, labels, N, permute=perm, mode="batch_mean")["rho_x"])
    rho_bm_perms = np.array(rho_bm_perms)

    # Per-example
    res_pe_ring = diagnostic(logits, ids, labels, N, mode="per_example")
    rng = np.random.default_rng(seed + 1)
    rho_pe_perms = []
    for _ in range(n_perms):
        perm = list(map(int, rng.permutation(N)))
        rho_pe_perms.append(diagnostic(logits, ids, labels, N, permute=perm, mode="per_example")["rho_x"])
    rho_pe_perms = np.array(rho_pe_perms)

    result = {
        "model": model_name,
        "load_time_s": round(load_time, 2),
        "fwd_time_s": round(fwd_time, 2),
        "vram_gb": round(vram, 3),
        "ring_mass_space": round(ms, 4),
        "ring_mass_bare": round(mb, 4),
        "style": style,
        "top1_acc": round(top1_acc, 4),
        "batch_mean": {
            "ring_rho_x": round(res_bm_ring["rho_x"], 4),
            "perm_rho_x_mean": round(float(rho_bm_perms.mean()), 4),
            "perm_rho_x_std": round(float(rho_bm_perms.std()), 4),
            "separation": round(float(rho_bm_perms.mean() - res_bm_ring["rho_x"]), 4),
        },
        "per_example": {
            "ring_rho_x": round(res_pe_ring["rho_x"], 4),
            "ring_rho_x_std": round(res_pe_ring["rho_x_std"], 4),
            "perm_rho_x_mean": round(float(rho_pe_perms.mean()), 4),
            "perm_rho_x_std": round(float(rho_pe_perms.std()), 4),
            "separation": round(float(rho_pe_perms.mean() - res_pe_ring["rho_x"]), 4),
        },
    }

    print(f"  vram={vram:.2f}GB  load={load_time:.1f}s  fwd={fwd_time:.1f}s  "
          f"top1={top1_acc:.3f}  ring_mass={ms:.3f}")
    print(f"  batch_mean : ring={result['batch_mean']['ring_rho_x']:.4f}  "
          f"perm={result['batch_mean']['perm_rho_x_mean']:.4f}+/-{result['batch_mean']['perm_rho_x_std']:.4f}  "
          f"sep={result['batch_mean']['separation']:+.4f}")
    print(f"  per_example: ring={result['per_example']['ring_rho_x']:.4f}  "
          f"perm={result['per_example']['perm_rho_x_mean']:.4f}+/-{result['per_example']['perm_rho_x_std']:.4f}  "
          f"sep={result['per_example']['separation']:+.4f}")

    del model
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=PYTHIA_SIZES)
    ap.add_argument("--n-perms", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pythia_rho_x_sweep/results_n_axis.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sweep on {device}; sizes={args.sizes}")
    results = []
    for s in args.sizes:
        r = run_one(s, args.n_perms, args.seed, device)
        results.append(r)
        Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"\nResults saved to {args.out}")
    print("\nSummary:")
    print(f"{'model':<24} {'top1':>6} {'bm_ring':>8} {'bm_perm':>8} {'bm_sep':>8} "
          f"{'pe_ring':>8} {'pe_perm':>8} {'pe_sep':>8}")
    for r in results:
        print(f"{r['model']:<24} {r['top1_acc']:>6.3f} "
              f"{r['batch_mean']['ring_rho_x']:>8.4f} {r['batch_mean']['perm_rho_x_mean']:>8.4f} "
              f"{r['batch_mean']['separation']:>+8.4f} "
              f"{r['per_example']['ring_rho_x']:>8.4f} {r['per_example']['perm_rho_x_mean']:>8.4f} "
              f"{r['per_example']['separation']:>+8.4f}")


if __name__ == "__main__":
    main()
