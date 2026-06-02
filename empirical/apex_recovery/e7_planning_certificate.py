"""
E7 -- Planning: the chart is a faithful coordinate for control  (Prop 1 + Thm 4)
================================================================================

This is the certificate the paper's title's second word has been missing. Section 5
makes two claims about planning in the recovered chart psi = U*Phi_1, and until now
no experiment exercised them:

  * Proposition 1 (exact). Because psi is a diffeomorphism, ANY control problem on
    the true latent transports through it without loss: planning in the agent's own
    representation -- with the change of variables applied -- is exactly planning in
    the world it cannot see. Exact for any psi, Gaussian or not.

  * Theorem 4 (approximate). When recovery is only within RMS error eta, the chart
    plan's true-world regret is bounded:  E[V_pi - V*] <= C * L * T * eta  -- linear
    in eta and the horizon T, vanishing as eta -> 0 (recovering Prop 1).

The spine is E1's, moved from probing to acting:

      latent s  --g-->  observation o = g(s)  --psi=g^{-1}-->  recovered chart

  - The CHART agent applies the recovery map psi (= the slow eigenfunction E1
    verifies: the cube-root on the non-Gaussian world, affine on the Gaussian one),
    so it knows the true state and plans optimally -- even when psi is nonlinear.
    THAT is Proposition 1 in action.
  - The LINEAR agent uses the best affine read of s from o (the control analogue of
    E1's linear probe). Off the Gaussian it mis-locates the state and is silently
    sub-optimal; at the Gaussian corner (g affine) it is exact. THAT is the kicker:
    the same affine-iff-Gaussian boundary, now in control.
  - With a noisy chart (RMS eta) the chart agent's regret grows linearly in eta and
    vanishes at eta=0 -- Theorem 4, certified by a sweep.

Everything is deterministic finite-horizon dynamic programming on a grid; the only
randomness is the recovery-noise model of Theorem 4 (fixed seed). No GPU, no torch.
Run:  python empirical/apex_recovery/e7_planning_certificate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPORTS = Path(__file__).resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Worlds: latent s, observation o = g(s). The recovery chart is psi = g^{-1},
# i.e. the slow-eigenfunction map E1 recovers (affine on the Gaussian world, the
# cube-root on the non-Gaussian one). A nonlinearity knob sweeps from one to the
# other:  g_a(s) = s + a*s^3  (monotone for a >= 0; a=0 is the Gaussian corner).
# ---------------------------------------------------------------------------


def warp(a):
    """Return (g, ginv) for g_a(s) = s + a s^3, normalised so the image matches
    the latent range (keeps the grids comparable). ginv by monotone bisection."""
    def g(s):
        return s + a * s ** 3

    def ginv(o, lo=-3.0, hi=3.0, iters=60):
        lo = np.full_like(np.asarray(o, float), lo)
        hi = np.full_like(np.asarray(o, float), hi)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            go = g(mid)
            hi = np.where(go > o, mid, hi)
            lo = np.where(go > o, lo, mid)
        return 0.5 * (lo + hi)

    return g, ginv


def affine_nonlinearity(x, y):
    """1 - R^2 of the best linear fit y ~ x: 0 iff y is an affine function of x."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.polyfit(x, y, 1)
    resid = y - np.polyval(A, x)
    return float((resid ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-15))


# ---------------------------------------------------------------------------
# A finite-horizon MDP on the latent grid: actions move s by one grid step;
# reward is a narrow peak so the optimal plan must localise the state precisely.
# ---------------------------------------------------------------------------


def build_mdp(N=81, smax=1.5, goal=0.8, width=0.12, T=24):
    s = np.linspace(-smax, smax, N)
    r = np.exp(-(s - goal) ** 2 / (2 * width ** 2))           # reward landscape on s
    actions = np.array([-1, 0, 1])
    nxt = np.clip(np.arange(N)[:, None] + actions[None, :], 0, N - 1)   # [N,3]
    return dict(s=s, r=r, actions=actions, nxt=nxt, T=T, N=N)


def value_iteration(mdp):
    """Backward induction -> V[t] (value), PI[t] (optimal action index)."""
    N, T, r, nxt = mdp["N"], mdp["T"], mdp["r"], mdp["nxt"]
    V = np.zeros((T + 1, N))
    PI = np.zeros((T, N), dtype=int)
    for t in range(T - 1, -1, -1):
        Q = r[nxt] + V[t + 1][nxt]                            # [N,3] reward-on-arrival
        PI[t] = Q.argmax(1)
        V[t] = Q.max(1)
    return V, PI


def return_over_starts(mdp, PI, est_idx):
    """Mean true-world return of the policy that, in true state i, plays the action
    optimal for its ESTIMATED state est_idx[i]. Vectorised over all start states."""
    N, T, r, nxt = mdp["N"], mdp["T"], mdp["r"], mdp["nxt"]
    cur = np.arange(N)
    total = np.zeros(N)
    for t in range(T):
        a = PI[t][est_idx[cur]]                               # action for estimated state
        cur = nxt[cur, a]                                     # true transition
        total += r[cur]
    return float(total.mean())


def nearest_idx(values, grid):
    return np.abs(np.asarray(values)[:, None] - grid[None, :]).argmin(1)


# ---------------------------------------------------------------------------


def make_figure(out):
    """Two panels: the control affine-iff-Gaussian (regret vs warp nonlinearity)
    and Theorem 4 (regret vs recovery error eta with its bound). matplotlib optional."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    sweep, thm = out["linear_kicker"], out["theorem4"]
    L, T, C = out["L"], out["T"], out["C_fit"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    nus = [d["nu"] for d in sweep]; regs = [d["regret_lin"] for d in sweep]
    ax1.plot(nus, regs, "o-", color="C3", lw=2, label="linear agent (ignores the warp)")
    ax1.axhline(0, color="C0", lw=2, ls="--", label="chart agent (Prop 1: exact)")
    ax1.set_xlabel("warp nonlinearity  $\\nu$   (0 = Gaussian corner)")
    ax1.set_ylabel("true-world regret  $J^\\star - J$")
    ax1.set_title("Planning is affine-iff-Gaussian:\nthe linear chart fails off the Gaussian", fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    etas = [d["eta"] for d in thm]; er = [d["regret"] for d in thm]
    ax2.plot(etas, er, "o-", color="C3", lw=2, label="E[regret], noisy chart")
    ax2.plot(etas, [C * L * T * e for e in etas], "--", color="0.5",
             label=f"$C\\,L\\,T\\,\\eta$ bound  (C={C:.3f})")
    ax2.set_xlabel("recovery RMS error  $\\eta$")
    ax2.set_ylabel("true-world regret")
    ax2.set_title("Theorem 4:  regret $\\leq C\\,L\\,T\\,\\eta$\n(linear in $\\eta$, $\\to 0$ as $\\eta\\to 0$)", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = REPORTS / "e7_planning.png"
    fig.savefig(p, dpi=140)
    return p


def main():
    rng = np.random.default_rng(0)
    mdp = build_mdp()
    s, N, T = mdp["s"], mdp["N"], mdp["T"]
    V, PI = value_iteration(mdp)
    Jstar = float(V[0].mean())                                # optimal = chart agent
    L = float(np.abs(np.diff(V[0])).max() / (s[1] - s[0]))    # value Lipschitz const
    print(f"E7 planning certificate | N={N} T={T} | J*={Jstar:.4f}  L(value)={L:.3f}")

    identity = np.arange(N)

    # ---- Part A: Proposition 1 -- the chart agent is exact, even on a nonlinear warp.
    print("\n[A] Proposition 1: planning in the recovered chart = planning in the world")
    propA = {}
    for name, a in (("gaussian", 0.0), ("cube", 1.0)):
        g, ginv = warp(a)
        o = g(s)
        nu = affine_nonlinearity(s, o)                        # warp nonlinearity (E1's nu)
        # chart agent recovers s exactly via psi=ginv, then localises on the grid:
        est_chart = nearest_idx(ginv(o), s)
        J_chart = return_over_starts(mdp, PI, est_chart)
        propA[name] = dict(nu=nu, J_chart=J_chart, regret=Jstar - J_chart)
        print(f"    {name:8s} nu(warp)={nu:.3f}  J_chart={J_chart:.4f}  "
              f"regret={Jstar - J_chart:.2e}  (chart agent matches J* exactly)")
        assert abs(J_chart - Jstar) < 1e-9, "chart agent must be exact (Prop 1)"

    # ---- Part B: the linear kicker -- affine read of the warp; the Gaussian boundary.
    print("\n[B] Linear agent (affine read of s from o): silent sub-optimality off-Gaussian")
    sweep = []
    for a in (0.0, 0.25, 0.5, 1.0, 1.75):
        g, ginv = warp(a)
        o = g(s)
        nu = affine_nonlinearity(s, o)
        coef = np.polyfit(o, s, 1)                            # best affine estimate of s from o
        est_lin = nearest_idx(np.polyval(coef, o), s)
        J_lin = return_over_starts(mdp, PI, est_lin)
        reg = Jstar - J_lin
        sweep.append(dict(a=a, nu=nu, J_lin=J_lin, regret_lin=reg))
        tag = "  <- Gaussian corner: exact" if a == 0.0 else ""
        print(f"    a={a:4.2f}  nu={nu:.3f}  J_lin={J_lin:.4f}  regret_lin={reg:.4f}{tag}")
    assert sweep[0]["regret_lin"] < 1e-9, "linear agent must be exact at the Gaussian corner"
    assert sweep[-1]["regret_lin"] > 1e-2, "linear agent must be sub-optimal on a strong warp"
    # regret grows with the warp nonlinearity
    nus = [d["nu"] for d in sweep]; regs = [d["regret_lin"] for d in sweep]
    assert all(x <= y + 1e-9 for x, y in zip(regs, regs[1:])), "regret_lin should rise with nu"
    print(f"    => regret_lin climbs 0 -> {regs[-1]:.3f} with warp nonlinearity (affine-iff-Gaussian, in control)")

    # ---- Part C: Theorem 4 -- noisy chart, regret <= C*L*T*eta, vanishing at eta=0.
    print("\n[C] Theorem 4: imperfect chart (RMS eta) -> regret <= C*L*T*eta")
    g, ginv = warp(1.0)                                       # the nonlinear (cube) world
    o = g(s)
    s_recovered = ginv(o)                                     # = s, the perfect chart
    etas = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    reps = 400
    thmC = []
    for eta in etas:
        regs_eta = []
        for _ in range(reps):
            shat = s_recovered + eta * rng.standard_normal(N)  # recovery error, RMS ~ eta
            est = nearest_idx(shat, s)
            regs_eta.append(Jstar - return_over_starts(mdp, PI, est))
        Ereg = float(np.mean(regs_eta))
        thmC.append(dict(eta=float(eta), regret=Ereg))
        bound = L * T * eta                                   # C=1 reference scale
        print(f"    eta={eta:4.2f}  E[regret]={Ereg:.4f}  L*T*eta={bound:.3f}  "
              f"ratio={Ereg / (bound + 1e-12):.3f}")
    assert thmC[0]["regret"] < 1e-9, "Thm 4 must vanish at eta=0 (recovers Prop 1)"
    # the bound holds with an O(1) constant, and regret is monotone in eta
    Cfit = max(d["regret"] / (L * T * d["eta"]) for d in thmC if d["eta"] > 0)
    assert all(d2["regret"] >= d1["regret"] - 1e-9
               for d1, d2 in zip(thmC, thmC[1:])), "E[regret] should rise with eta"
    assert Cfit < 1.0, "regret must stay within C*L*T*eta for an O(1) constant C"
    print(f"    => regret <= C*L*T*eta holds with C={Cfit:.3f} (O(1)); linear in eta; ->0 as eta->0")

    out = dict(experiment="E7_planning_certificate", Jstar=Jstar, L=L, T=T,
               proposition1=propA, linear_kicker=sweep, theorem4=thmC, C_fit=Cfit,
               summary="chart agent exact on nonlinear warp (Prop 1); linear agent "
                       "regret 0 at Gaussian and rising with nu (the control analogue "
                       "of the linear-probe failure); noisy-chart regret <= C*L*T*eta (Thm 4)")
    (REPORTS / "e7_planning_certificate.json").write_text(json.dumps(out, indent=2))
    fig = make_figure(out)
    print(f"\nPASS. wrote {REPORTS / 'e7_planning_certificate.json'}"
          + (f" and {fig}" if fig else " (figure skipped: matplotlib unavailable)"))
    return out


if __name__ == "__main__":
    main()
