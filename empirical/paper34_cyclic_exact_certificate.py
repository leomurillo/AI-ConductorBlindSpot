"""Exact cyclic conductor-packet certificate for the five-ring ladder.

This is the machine-readable counterpart to Section 4 / Appendix C of the
manuscript. It deliberately uses only Python integer arithmetic: no NumPy, no
floating point, and no tolerance-based vanishing decisions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LADDER = (6, 8, 12, 18, 30)
DEFAULT_OUT = Path("empirical/reports/paper34_cyclic_exact_certificate.json")


def conductor(n: int, k: int) -> int:
    return n // math.gcd(n, k)


def conductor_packets(n: int) -> dict[int, list[int]]:
    packets: dict[int, list[int]] = {}
    for k in range(1, n):
        packets.setdefault(conductor(n, k), []).append(k)
    return dict(sorted(packets.items()))


def verify_ring(n: int) -> dict:
    packets = conductor_packets(n)
    cond = {k: conductor(n, k) for k in range(1, n)}

    bad_fisher_pairs = []
    for k in range(1, n):
        for ell in range(1, n):
            if cond[k] != cond[ell] and (k - ell) % n == 0:
                bad_fisher_pairs.append([k, ell])

    total = 0
    same_packet = 0
    cross_packet = 0
    sample_cross_packet_triple = None
    for k in range(1, n):
        for ell in range(1, n):
            for m in range(1, n):
                if (k + ell + m) % n != 0:
                    continue
                total += 1
                conductors = [cond[k], cond[ell], cond[m]]
                if conductors[0] == conductors[1] == conductors[2]:
                    same_packet += 1
                else:
                    cross_packet += 1
                    if sample_cross_packet_triple is None:
                        sample_cross_packet_triple = {
                            "triple": [k, ell, m],
                            "conductors": conductors,
                            "distinct_conductors": len(set(conductors)),
                        }

    return {
        "n": n,
        "packets": {str(key): value for key, value in packets.items()},
        "bad_cross_packet_fisher_pairs": bad_fisher_pairs,
        "surviving_triples_total": total,
        "closed_form_total": (n - 1) * (n - 2),
        "same_packet_triples": same_packet,
        "cross_packet_triples": cross_packet,
        "sample_cross_packet_triple": sample_cross_packet_triple,
    }


def build_certificate(ladder: tuple[int, ...] = LADDER) -> dict:
    rings = [verify_ring(n) for n in ladder]
    return {
        "schema_version": 1,
        "description": "Exact cyclic Fisher/cubic conductor-packet certificate for Section 4 / Appendix C.",
        "arithmetic": "Python integer arithmetic only",
        "floating_point_used": False,
        "rings": rings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output JSON path (default: {DEFAULT_OUT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_certificate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in payload["rings"]:
        print(
            f"n={row['n']} total={row['surviving_triples_total']} "
            f"same={row['same_packet_triples']} cross={row['cross_packet_triples']} "
            f"bad_fisher_pairs={len(row['bad_cross_packet_fisher_pairs'])}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
