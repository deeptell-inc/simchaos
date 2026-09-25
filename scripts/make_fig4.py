"""Fig 4: (a) crossover s*/CSR vs lambda; (b) elliptic family theory curves;
(c) discrete-map slope separation."""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from functools import lru_cache
from math import factorial, comb

R = RESULTS
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5))

# (a) crossover
t6 = json.loads(_in("task6_crossover.json").read_text())
lam = [r["lam"] for r in t6]
ax2 = ax[0].twinx()
ax[0].errorbar(lam, [r["s"] for r in t6], yerr=[r["s_std"] for r in t6],
               fmt="o-", color="firebrick", ms=4, label=r"$s^*$")
ax2.errorbar(lam, [r["r"] for r in t6], yerr=[r["r_std"] for r in t6],
             fmt="s--", color="gray", ms=3.5)
ax[0].axhspan(1.30, 1.60, color="orange", alpha=0.15)
ax[0].set_xlabel(r"$\lambda$ (integrability breaking)")
ax[0].set_ylabel(r"$s^*$", color="firebrick")
ax2.set_ylabel(r"$\langle r\rangle$ (CSR)", color="gray")
ax2.axhline(0.738, color="gray", ls=":", lw=0.7)
ax[0].set_title("(a) crossover tracking", fontsize=9)

# (b) elliptic family theory
def M_mn(m, n, tau):
    word = [0]*m + [1]*n
    @lru_cache(maxsize=None)
    def f(i, j):
        if i >= j: return 1.0
        if (j-i) % 2: return 0.0
        return sum((tau if word[i]==word[k] else 1.0)*f(i+1,k)*f(k+1,j)
                   for k in range(i+1, j, 2))
    return f(0, m+n)
def lam_mom(m, n, tau):
    return sum(comb(m,p)*comb(n,q)*tau**((m-p)+(n-q))/(p+n-q+1)
               for p in range(m+1) for q in range(n+1) if p+n-q == m-p+q)
def ratio(t, th, tau, K=22):
    a = t*np.exp(1j*th); P = P0 = 0j
    for m in range(K+1):
        for n in range(K+1-m):
            if (m+n)%2: continue
            c = a**m*np.conj(a)**n/(factorial(m)*factorial(n))
            P += c*M_mn(m,n,tau); P0 += c*lam_mom(m,n,tau)
    return (P/P0).real - 1
ts = np.linspace(0.25, 3.0, 12)
for tau, col in ((0.0,"crimson"),(0.3,"darkorange"),(0.6,"seagreen"),(0.9,"steelblue")):
    ax[1].plot(ts, [ratio(t, 0, tau) for t in ts], "-", color=col, lw=1.4,
               label=fr"$\tau={tau}$")
ax[1].set_xlabel(r"$t$"); ax[1].set_ylabel(r"$X/\Phi_0$ ($\theta{=}0$, planar)")
ax[1].legend(fontsize=6); ax[1].set_title("(b) elliptic family (theory)", fontsize=9)

# (c) discrete maps
t8 = json.loads(_in("task8_maps.json").read_text())
ks = np.arange(1, 13)
for r in t8:
    col = "firebrick" if r["kind"]=="haar" else "steelblue"
    ax[2].semilogy(ks, np.abs(r["x_profile"]), "-o" if r["kind"]=="haar" else "-s",
                   color=col, ms=2.5, lw=0.9, alpha=0.7)
ax[2].set_xlabel(r"$k$ (map iterations)"); ax[2].set_ylabel(r"$|X_k/\Phi_{0,k}|$")
from matplotlib.lines import Line2D
ax[2].legend(handles=[Line2D([],[],color="firebrick",marker="o",ls="-",label="Haar dilation (chaotic)"),
                      Line2D([],[],color="steelblue",marker="s",ls="-",label="product channel (integrable)")],
             fontsize=6)
ax[2].set_title("(c) discrete-time maps", fontsize=9)
fig.tight_layout()
for ext in ("png","svg"):
    fig.savefig(R/f"fig4.{ext}", dpi=200)
print("fig4 saved")
