"""
receipts.py -- SHA-256 provenance + reproducibility receipts for the whole project.

Three things, one command:
  * SOURCE  : raw-byte SHA-256 of every certificate script and manuscript
              (tamper-evident provenance -- "these exact bytes existed").
  * RESULT  : SHA-256 of each reports/*.json AFTER stripping volatile keys
              (timing, environment, figure paths) and canonicalizing (sorted keys,
              compact). A deterministic certificate's RESULT hash is therefore
              REPRODUCIBLE across runs and machines, even though wall-clock timing
              is not -- so re-running run_all.py and re-hashing must match.
  * ROLLUP  : one SHA-256 over all (tag, hash, name) lines -- a single fingerprint
              of the entire verified state. Print it, paste it in a paper, diff it.

PDFs are intentionally excluded from the reproducible rollup: xelatex embeds build
timestamps, so PDF bytes are not reproducible (they appear in SHA256SUMS.txt as
plain provenance only).

Usage:
  python receipts.py          # write RECEIPTS.sha256 and print the ROLLUP
  python receipts.py --check  # recompute; exit 0 iff the ROLLUP matches the stored one
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APEX = ROOT / "empirical" / "apex_recovery"

# keys whose values are run/machine-specific, not results (stripped before hashing):
VOLATILE = {"t", "time", "times", "secs", "seconds", "duration", "wall_time",
            "elapsed", "runtime", "_environment", "environment", "figure",
            "timestamp", "date"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def strip_volatile(o):
    if isinstance(o, dict):
        return {k: strip_volatile(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [strip_volatile(x) for x in o]
    return o


def canon_result_hash(path: Path) -> str:
    """Hash the canonical, volatile-stripped JSON -> reproducible result fingerprint."""
    obj = json.loads(path.read_text())
    canon = json.dumps(strip_volatile(obj), sort_keys=True, separators=(",", ":"))
    return sha(canon.encode())


def build_lines():
    sources = sorted(APEX.glob("*.py")) + sorted(ROOT.glob("*.md"))
    results = sorted((APEX / "reports").glob("*.json"))
    lines = []
    for p in sources:
        lines.append(f"SOURCE {sha(p.read_bytes())}  {p.relative_to(ROOT).as_posix()}")
    for p in results:
        lines.append(f"RESULT {canon_result_hash(p)}  {p.relative_to(ROOT).as_posix()}")
    rollup = sha("\n".join(sorted(lines)).encode())
    return lines, rollup, len(sources), len(results)


def main():
    out = ROOT / "RECEIPTS.sha256"
    lines, rollup, n_src, n_res = build_lines()

    if "--check" in sys.argv:
        saved = None
        if out.exists():
            for ln in out.read_text().splitlines():
                if ln.startswith("ROLLUP"):
                    saved = ln.split("=")[-1].strip()
        ok = saved == rollup
        print(f"ROLLUP now   = {rollup}")
        print(f"ROLLUP saved = {saved}")
        print("MATCH -- outputs verified reproducible" if ok else "MISMATCH -- something changed")
        sys.exit(0 if ok else 1)

    header = ("# SHA-256 receipts for Curvature-and-Current certificate suite.\n"
              "# SOURCE = raw bytes; RESULT = canonical (volatile-stripped) JSON; "
              "verify results with: python receipts.py --check\n\n")
    out.write_text(header + "\n".join(lines) + f"\n\nROLLUP sha256 = {rollup}\n")
    print(f"wrote {out.name}  ({n_src} sources + {n_res} results)")
    print(f"ROLLUP sha256 = {rollup}")


if __name__ == "__main__":
    main()
