# Paper 34 -- Layer 3 (Section 8) Cubic-Aware Experiment Summary

Form A: plain AdamW baseline plus cross-packet Newton-style T(u,u,.) cubic injection.
T1 = (a+b) mod n; T2-strong = random function f(a,b).

## Per-run summary

| $n$ | seed | $\alpha_{\mathrm{cubic}}$ | A1xT1 | A2xT1 | $\Delta_{T1}$ | A1xT2s | A2xT2s | $\Delta_{T2s}$ | **interaction** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0 | 0.1 | 0.002 | 0.002 | +0.000 | 0.038 | 0.038 | +0.001 | **-0.001** |

## Aggregate summary

| statistic | runs | $\Delta_{T1}$ | $\Delta_{T2s}$ | **interaction** |
|---|---:|---:|---:|---:|
| mean | 1 | +0.000 | +0.001 | **-0.001** |
| std | 1 | n/a | n/a | n/a |

## Pre-specified branches

**Branch A (Conjecture 5.8 confirmed on tested instance):** 
interaction > 0, with $\Delta_{T1} > 0$ and $\Delta_{T2s} \approx 0$ within noise.

**Branch B (refutation):** interaction $\leq 0$ within noise, or $\Delta_{T2s}$ 
is positive with magnitude comparable to $\Delta_{T1}$. Theorem 3.5, 
Corollary 3.3, the diagnostic of Section 6, and the off-centroid expansion of 
Section 5 stand regardless. Conjecture 5.8 is reported as not supported by the 
experiment on the tested instance.
