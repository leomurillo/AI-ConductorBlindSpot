"""Smoke test: end-to-end rho_x on a single Pythia-70M forward pass.

Loads the smallest Pythia, runs one forward on a templated month-sequence
prompt, identifies which token-style (bare vs leading-space) the model
puts continuation mass on, and computes rho_x for that one example.
A single example will be noisy by construction — purpose is to verify
the pipeline end-to-end, not to test the certificate.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rho_x import rho_x, packet_summary


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEFAULT_MODEL = "EleutherAI/pythia-70m"
DEFAULT_MODEL_REVISION = "main"


def resolved_model_revision(tok, model) -> str | None:
    """Best-effort HuggingFace commit hash captured after loading."""
    for obj in (model, tok):
        config = getattr(obj, "config", None)
        commit = getattr(config, "_commit_hash", None)
        if commit:
            return str(commit)
        init_kwargs = getattr(obj, "init_kwargs", None)
        if isinstance(init_kwargs, dict) and init_kwargs.get("_commit_hash"):
            return str(init_kwargs["_commit_hash"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    args = parser.parse_args()

    model_name = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name} @ {args.model_revision} on {device}...")
    tok = AutoTokenizer.from_pretrained(model_name, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=args.model_revision, torch_dtype=torch.float32
    ).to(device).eval()
    print(f"Vocab size: {tok.vocab_size}, model dtype: {next(model.parameters()).dtype}")
    print(f"resolved HF revision: {resolved_model_revision(tok, model) or 'unknown'}")

    ids_bare = [tok.encode(m, add_special_tokens=False)[0] for m in MONTHS]
    ids_space = [tok.encode(" " + m, add_special_tokens=False)[0] for m in MONTHS]
    print(f"bare IDs  = {ids_bare}")
    print(f"space IDs = {ids_space}")

    # Smoke prompt: cyclic month sequence, predict next.
    prompt = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov"
    label_idx = 11  # Dec

    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    print(f"\nprompt: {prompt!r}")
    print(f"tokens (decoded): {[tok.decode([int(i)]) for i in input_ids[0]]}")

    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :].float().cpu()
    probs = torch.softmax(logits, dim=-1)

    mass_bare = probs[ids_bare].sum().item()
    mass_space = probs[ids_space].sum().item()
    print(f"\nmass on bare month tokens:  {mass_bare:.4f}")
    print(f"mass on space month tokens: {mass_space:.4f}")

    style = "space" if mass_space > mass_bare else "bare"
    ids = ids_space if style == "space" else ids_bare
    print(f"-> using style={style} (top-{1} token = {tok.decode([int(probs.argmax())])!r})")

    sub_logits = logits[ids]
    p_cond = torch.softmax(sub_logits, dim=-1).numpy()
    argmax_idx = int(p_cond.argmax())
    print(f"\np_cond over months: {[f'{MONTHS[i]}:{p_cond[i]:.3f}' for i in range(12)]}")
    print(f"argmax  = {MONTHS[argmax_idx]}")
    print(f"label   = {MONTHS[label_idx]}")
    print(f"top-1 correct? {argmax_idx == label_idx}")

    u = [(1.0 if i == label_idx else 0.0) - float(p_cond[i]) for i in range(12)]
    print(f"\ncentered score u = {[round(x, 4) for x in u]}")
    print(f"sum(u) = {sum(u):.6e}  (must be ~0)")

    result = rho_x(u, 12)
    print(f"\nn=12 packets: {packet_summary(12)}")
    print(f"selection-rule triples: total={result['n_triples_total']}, "
          f"cross-packet={result['n_triples_cross']}  "
          f"(uniform baseline rho_x = {result['n_triples_cross']/result['n_triples_total']:.4f})")
    print(f"\nrho_x       = {result['rho_x']:.4f}")
    print(f"total_mass  = {result['total_mass']:.4e}")
    print(f"cross_mass  = {result['cross_mass']:.4e}")


if __name__ == "__main__":
    main()
