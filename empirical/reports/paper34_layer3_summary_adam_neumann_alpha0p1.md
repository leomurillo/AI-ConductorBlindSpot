# Paper 34 -- Layer 3 (Section 8) Cubic-Aware Experiment Summary

Form B: plain AdamW baseline plus cross-packet Neumann-style T(h,u,.) cubic injection.
T1 = (a+b) mod n; T2-strong = random function f(a,b).

## Per-run summary

| $n$ | seed | $\alpha_{\mathrm{cubic}}$ | A1xT1 | A2xT1 | $\Delta_{T1}$ | A1xT2s | A2xT2s | $\Delta_{T2s}$ | **interaction** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0 | 0.1 | 0.955 | 0.911 | -0.044 | 0.047 | 0.039 | -0.008 | **-0.036** |
| 30 | 1 | 0.1 | 1.000 | 0.990 | -0.010 | 0.043 | 0.041 | -0.002 | **-0.008** |
| 30 | 2 | 0.1 | 0.981 | 0.990 | +0.009 | 0.049 | 0.047 | -0.003 | **+0.012** |

## Aggregate summary

| statistic | runs | $\Delta_{T1}$ | $\Delta_{T2s}$ | **interaction** |
|---|---:|---:|---:|---:|
| mean | 3 | -0.015 | -0.004 | **-0.011** |
| std | 3 | 0.027 | 0.004 | 0.024 |

## Pre-specified branches

**Branch A (Conjecture 5.8 confirmed on tested instance):** 
interaction > 0, with $\Delta_{T1} > 0$ and $\Delta_{T2s} \approx 0$ within noise.

**Branch B (refutation):** interaction $\leq 0$ within noise, or $\Delta_{T2s}$ 
is positive with magnitude comparable to $\Delta_{T1}$. Theorem 3.5, 
Corollary 3.3, the diagnostic of Section 6, and the off-centroid expansion of 
Section 5 stand regardless. Conjecture 5.8 is reported as not supported by the 
experiment on the tested instance.
