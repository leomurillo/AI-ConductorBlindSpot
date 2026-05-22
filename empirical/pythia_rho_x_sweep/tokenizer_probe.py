"""Probe Pythia's tokenizer to see which rings have clean single-token labels.

For each candidate ring (weekdays, months, hours, digits), test each label
in three context positions:
    "X"        bare
    " X"       leading space (common BPE prefix for mid-sentence tokens)
    "\nX"      newline-prefixed

A ring is "clean" in a position iff every label is a single token in that
position. The diagnostic needs all n labels to share one clean position.
"""

from __future__ import annotations
from transformers import AutoTokenizer


WEEKDAYS_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAYS_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
MONTHS_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_LONG = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
DIGITS = [str(i) for i in range(10)]
HOURS_24 = [f"{i:02d}" for i in range(24)]


def probe_ring(tok, labels: list[str], name: str) -> dict:
    """For each context style, list (label, token-ids) and count single-token labels."""
    styles = {"bare": "", "space": " ", "newline": "\n"}
    results = {}
    for style_name, prefix in styles.items():
        rows = []
        n_single = 0
        for lab in labels:
            ids = tok.encode(prefix + lab, add_special_tokens=False)
            if len(ids) == 1:
                n_single += 1
            rows.append((lab, ids))
        results[style_name] = {"rows": rows, "n_single": n_single, "n_total": len(labels)}
    return {"name": name, "labels": labels, "by_style": results}


def main() -> None:
    print("Loading Pythia tokenizer...")
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    print(f"Vocab size: {tok.vocab_size}")
    print()

    rings = [
        ("weekdays_short (n=7)", WEEKDAYS_SHORT),
        ("weekdays_long (n=7)", WEEKDAYS_LONG),
        ("months_short (n=12)", MONTHS_SHORT),
        ("months_long (n=12)", MONTHS_LONG),
        ("digits (n=10)", DIGITS),
        ("hours_24 (n=24)", HOURS_24),
    ]

    for name, labels in rings:
        r = probe_ring(tok, labels, name)
        print(f"=== {name} ===")
        for style_name, style_r in r["by_style"].items():
            n_s = style_r["n_single"]
            n_t = style_r["n_total"]
            verdict = "CLEAN" if n_s == n_t else "split"
            print(f"  style={style_name:8s}  {n_s}/{n_t} single-token  [{verdict}]")
            if verdict == "split":
                for lab, ids in style_r["rows"]:
                    if len(ids) != 1:
                        toks = [tok.decode([i]) for i in ids]
                        print(f"    {lab!r:14s} -> {ids}  decoded={toks}")
        print()


if __name__ == "__main__":
    main()
