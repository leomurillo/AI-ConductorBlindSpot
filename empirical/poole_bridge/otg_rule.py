"""
otg_rule.py — Rooke Poole's ACTUAL canonical rule, reimplemented in numpy.
================================================================================

This is a faithful, torch-free transcription of the `PooleEngine.step` of the
Delta RPM Protocol master engine in
    github.com/rookepoole/SVP-OTG-Poole-Manifold-tests
        Docs/DELTA RPM PROTOCOL - MASTER INTEGRATION TEST SUITE
discharging *Obligation 1* of the bridge note ("freeze the actual rule"). It is
numpy only — we never import or run his torch engine (only read it) — and it
reproduces his own canonical unit tests bit-for-bit (see `selfcheck()`):

  rule        : 3-D outer-totalistic on the torus, Moore-26 neighbourhood
                (3x3x3 ones-kernel, centre zeroed), circular boundary, synchronous.
  potential   : integer live-neighbour count m in {0..26}.
  resonance   : R(m) = sum_p alpha * exp(-(m - p)^2 / sigma_sq),
                primes = {2,3,5,7,11,13,17,19,23}, alpha=0.35, sigma_sq=0.01.
  total_phi   : m + R(m).
  birth  (m, dead cell)  : B_LOW=5 <= total_phi <= B_HIGH=7.
  survive(m, live cell)  : S_LOW=5 <= total_phi <= S_HIGH=9.

Because sigma_sq = 0.01 is sharp and m is an integer, R(m) = 0.35 exactly when m is
prime and ~0 otherwise. The only band-edge it moves is m=7 (prime): 7 + 0.35 =
7.35 > B_HIGH, so 7 is ejected from BIRTH. The exact effective integer rule is
therefore  B = {5,6},  S = {5,6,7,8,9}  (verified in selfcheck). We keep his exact
float computation as the definition and use the integer reduction only to label.

The optional inhibitor/threshold/amplifier control fields of his step() are used
in his logic-gate experiments, not in the bare substrate; the substrate
certificates run the base rule (all None), exactly as his variance audit does.
"""

from __future__ import annotations

from itertools import product

import numpy as np

PRIMES = (2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0)
ALPHA = 0.35
SIGMA_SQ = 0.01
B_LOW, B_HIGH = 5.0, 7.0
S_LOW, S_HIGH = 5.0, 9.0

# the 26 Moore offsets (centre excluded), matching his 3x3x3 ones-kernel
_OFFSETS = [o for o in product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]


def resonance(potential: np.ndarray) -> np.ndarray:
    """R = sum_p alpha exp(-(potential - p)^2 / sigma_sq). Exactly his loop."""
    out = np.zeros_like(potential, dtype=float)
    for p in PRIMES:
        out += ALPHA * np.exp(-((potential - p) ** 2) / SIGMA_SQ)
    return out


_KERNEL = np.ones((3, 3, 3), dtype=float)
_KERNEL[1, 1, 1] = 0.0  # centre excluded, exactly his self.kernel

try:
    from scipy.ndimage import convolve as _convolve  # fast C path, mode='wrap'

    def neighbour_count(field: np.ndarray) -> np.ndarray:
        """Moore-26 circular neighbour sum (his conv3d, circular pad, centre 0)."""
        return _convolve(field.astype(float), _KERNEL, mode="wrap")
except ImportError:  # pragma: no cover  — pure-numpy fallback (slower, identical)
    def neighbour_count(field: np.ndarray) -> np.ndarray:
        P = np.zeros(field.shape, dtype=float)
        for (dx, dy, dz) in _OFFSETS:
            P += np.roll(field, shift=(dx, dy, dz), axis=(0, 1, 2))
        return P


def step_field(field: np.ndarray) -> np.ndarray:
    """One synchronous OTG update of a 3-D 0/1 field. Faithful to PooleEngine.step."""
    f = field.astype(np.int8)
    m = neighbour_count(f)
    total_phi = m + resonance(m)
    birth = (f == 0) & (total_phi >= B_LOW) & (total_phi <= B_HIGH)
    survive = (f == 1) & (total_phi >= S_LOW) & (total_phi <= S_HIGH)
    return (birth | survive).astype(np.int8)


def effective_rule() -> dict:
    """The exact effective integer rule, by evaluating his real-valued bands at m=0..26."""
    m = np.arange(0, 27, dtype=float)
    phi = m + resonance(m)
    birth = sorted(int(k) for k in m[(phi >= B_LOW) & (phi <= B_HIGH)])
    survive = sorted(int(k) for k in m[(phi >= S_LOW) & (phi <= S_HIGH)])
    return {"birth": birth, "survive": survive,
            "phi_at_7": float(7 + resonance(np.array([7.0]))[0])}


# ---- small-torus enumeration (bit-encoded), for the EXACT certificates ------


class OTGTorus:
    """His exact rule on a small enumerable torus, states <-> integers (bit c = cell c)."""

    def __init__(self, shape: tuple):
        self.shape = shape
        self.cells = list(product(range(shape[0]), range(shape[1]), range(shape[2])))
        self.index = {c: i for i, c in enumerate(self.cells)}

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_states(self) -> int:
        return 1 << self.n_cells

    def decode(self, s: int) -> np.ndarray:
        f = np.zeros(self.shape, dtype=np.int8)
        for c, (x, y, z) in enumerate(self.cells):
            if (s >> c) & 1:
                f[x, y, z] = 1
        return f

    def encode(self, f: np.ndarray) -> int:
        s = 0
        for c, (x, y, z) in enumerate(self.cells):
            if f[x, y, z]:
                s |= (1 << c)
        return s

    def step_int(self, s: int) -> int:
        return self.encode(step_field(self.decode(s)))

    # --- vectorised batch step over ALL states (fast). step_int above is kept
    #     for reference; the certificate checks the two agree on a sample. -------
    def _decode_all(self) -> np.ndarray:
        d = self.n_states
        states = np.arange(d, dtype=np.int64)
        field = np.zeros((d,) + self.shape, dtype=np.int8)
        for c, (x, y, z) in enumerate(self.cells):
            field[:, x, y, z] = ((states >> c) & 1).astype(np.int8)
        return field

    def _encode_all(self, field: np.ndarray) -> np.ndarray:
        out = np.zeros(field.shape[0], dtype=np.int64)
        for c, (x, y, z) in enumerate(self.cells):
            out |= field[:, x, y, z].astype(np.int64) << c
        return out

    def F_array(self, cap: int = 1 << 18) -> np.ndarray:
        d = self.n_states
        if d > cap:
            raise ValueError(f"{d} states exceeds cap {cap}")
        f0 = self._decode_all()
        P = np.zeros(f0.shape, dtype=float)
        for (dx, dy, dz) in _OFFSETS:  # batched Moore-26 wrap sum over spatial axes
            P += np.roll(f0, shift=(dx, dy, dz), axis=(1, 2, 3))
        phi = P + resonance(P)
        birth = (f0 == 0) & (phi >= B_LOW) & (phi <= B_HIGH)
        survive = (f0 == 1) & (phi >= S_LOW) & (phi <= S_HIGH)
        return self._encode_all((birth | survive).astype(np.int8))

    def translate_int(self, s: int, shift: tuple) -> int:
        Lx, Ly, Lz = self.shape
        dx, dy, dz = shift
        out = 0
        for c, (x, y, z) in enumerate(self.cells):
            if (s >> c) & 1:
                out |= (1 << self.index[((x + dx) % Lx, (y + dy) % Ly, (z + dz) % Lz)])
        return out

    def translate_perm(self, shift: tuple) -> np.ndarray:
        """Translation by `shift` as a permutation of all states (vectorised)."""
        Lx, Ly, Lz = self.shape
        dx, dy, dz = shift
        d = self.n_states
        states = np.arange(d, dtype=np.int64)
        perm = np.zeros(d, dtype=np.int64)
        for c, (x, y, z) in enumerate(self.cells):
            tc = self.index[((x + dx) % Lx, (y + dy) % Ly, (z + dz) % Lz)]
            perm |= ((states >> c) & 1) << tc
        return perm

    def all_shifts(self):
        return list(product(range(self.shape[0]), range(self.shape[1]), range(self.shape[2])))


# ---- faithfulness self-check against HIS canonical unit tests ---------------


def selfcheck() -> dict:
    """Reproduce his Delta RPM physics unit tests, bit-for-bit, in numpy."""
    out = {}

    # test_vacuum: 10^3 of zeros stays empty.
    vac = np.zeros((10, 10, 10), dtype=np.int8)
    out["vacuum_stays_empty"] = bool(step_field(vac).sum() == 0)

    # test_overpopulation: a solid 3x3x3 block's core evaporates (26 nbrs > S_HIGH).
    blk = np.zeros((10, 10, 10), dtype=np.int8)
    blk[4:7, 4:7, 4:7] = 1
    out["overpopulation_core_evaporates"] = bool(step_field(blk)[5, 5, 5] == 0)

    # effective integer rule
    eff = effective_rule()
    out["effective_birth"] = eff["birth"]
    out["effective_survive"] = eff["survive"]
    out["seven_ejected_from_birth"] = bool(eff["phi_at_7"] > B_HIGH and 7 not in eff["birth"])
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(selfcheck(), indent=2))
