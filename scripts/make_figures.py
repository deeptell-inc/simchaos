"""S9: build Fig 2 (classification) and Fig 3 (robustness) for the manuscript.
Outputs: ../results/fig2.png/svg, fig3.png/svg
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
from models import MODELS
from diagnostics import excess_profile

R = RESULTS
TS = 0.25 * np.arange(1, 13)
TH0 = np.array([0.0])
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})

CORE = [  # name, size, label, class-color, ls
    ("ginue", 1024, "GinUE", "crimson", "-"),
    ("random_liouvillian", 32, "random Liouvillian", "darkorange", "-"),
    ("nh_syk_q4", 18, "nH-SYK $q{=}4$", "firebrick", "--"),
    ("nh_syk_q2", 18, "nH-SYK $q{=}2$", "steelblue", "--"),
    ("hatano_nelson", 1024, "Hatano–Nelson", "seagreen", ":"),
    ("poisson_normal", 1024, "normal 2D-Poisson", "navy", "-."),
]

# ---------------- Fig 2
p3 = json.loads(_in("p3_results.json").read_text())
s6 = json.loads(_in("s6_scan.json").read_text())

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
rng_seed = 20260717
for name, size, lab, col, ls in CORE:
    rng = np.random.default_rng(rng_seed)
    L = MODELS[name](size, rng)
    x = excess_profile(L, TS, TH0)["profile"][0]
    comp = np.abs(x) / TS**2
    ax[0].semilogy(TS, comp / comp[1], ls, color=col, label=lab, lw=1.3)
ax[0].axhline(1.0, color="k", ls=":", lw=0.8)
ax[0].text(2.55, 1.12, r"$\propto t^2$", fontsize=8)
ax[0].set_xlabel(r"$t$ (units of $1/\alpha$)")
ax[0].set_ylabel(r"$(X/\Phi_0)/t^2$ (norm.)")
ax[0].set_ylim(5e-2, 3e0)
ax[0].legend(fontsize=6.2, loc="lower left", framealpha=0.9)

# (chi, s*) plane from run_p3 rows (chaotic/integrable) + HN
for name, res in p3.items():
    if name.startswith("xxz"):
        continue  # excluded family (F8): ground truth unresolved
    cls = res["class"]
    col = "crimson" if cls.startswith("chaotic") else ("seagreen" if "stress" in cls else "steelblue")
    mk = "^" if "stress" in cls else ("o" if cls.startswith("chaotic") else "s")
    for row in res["rows"]:
        if row["chi"] < 1e-10:
            continue  # normal-integrable: classified at stage 1, off this panel
        ax[1].plot(row["s_late"], row["chi"], mk, color=col, ms=4.5, alpha=0.75)
ax[1].axvline(1.41, color="gray", ls="--", lw=1)
ax[1].text(1.415, 4.5, r"$s_c$", fontsize=8, color="gray")
ax[1].set_xlabel(r"late-time power $s^*$")
ax[1].set_ylabel(r"$\chi=\max X/\Phi_0$")
ax[1].set_yscale("log")
from matplotlib.lines import Line2D
ax[1].legend(handles=[
    Line2D([], [], marker="o", color="crimson", ls="", label="chaotic (CSR)"),
    Line2D([], [], marker="s", color="steelblue", ls="", label="integrable (CSR)"),
    Line2D([], [], marker="^", color="seagreen", ls="", label="HN (quasi-1D)")],
    fontsize=6.2, loc="upper right")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(R / f"fig2.{ext}", dpi=200)

# ---------------- Fig 3
sn = json.loads(_in("shotnoise.json").read_text())
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5))
# (a) chi vs D
for model, col, mk in (("ginue", "crimson", "o"), ("hatano_nelson", "seagreen", "^")):
    Ds, chis = {}, {}
    for r in s6["scan"]:
        if r["model"] == model:
            Ds.setdefault(r["D"], []).append(r["chi"])
    D_sorted = sorted(Ds)
    m = [np.mean(Ds[D]) for D in D_sorted]
    sd = [np.std(Ds[D]) for D in D_sorted]
    ax[0].errorbar(D_sorted, m, yerr=sd, fmt=mk + "-", color=col, ms=4,
                   label="GinUE" if model == "ginue" else "Hatano–Nelson")
ax[0].plot([256, 2048], [0.9 * (256 / 256) ** -0.5, 0.9 * (2048 / 256) ** -0.5], "k:", lw=0.8)
ax[0].text(600, 0.32, r"$\propto D^{-1/2}$", fontsize=7)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel(r"$D$"); ax[0].set_ylabel(r"$\chi$")
ax[0].legend(fontsize=6.2)
# (b) s* vs N for SYK
for q, col in ((4, "firebrick"), (2, "steelblue")):
    Ns, ss = {}, {}
    for r in s6["scan"]:
        if r["model"] == f"nh_syk_q{q}":
            # recover N from D: D = 2^(N/2 -1)
            N = int(2 * (np.log2(r["D"]) + 1))
            Ns.setdefault(N, []).append(r["s_late"])
    N_sorted = sorted(Ns)
    ax[1].errorbar(N_sorted, [np.mean(Ns[N]) for N in N_sorted],
                   yerr=[np.std(Ns[N]) for N in N_sorted], fmt="o-", color=col,
                   ms=4, label=f"$q={q}$")
ax[1].axhline(1.41, color="gray", ls="--", lw=1)
ax[1].set_xlabel(r"$N$ (Majoranas)"); ax[1].set_ylabel(r"$s^*$")
ax[1].set_xticks([14, 16, 18]); ax[1].legend(fontsize=6.2)
# (c) flip rate vs eps
for name, col, mk, lab in (("nh_syk_q4_N18", "firebrick", "o", "SYK $q{=}4$"),
                           ("nh_syk_q2_N18", "steelblue", "s", "SYK $q{=}2$"),
                           ("ginue_D1024", "crimson", "^", "GinUE ($s^*$ only)")):
    eps = [r["eps_rel"] for r in sn[name]["rows"]]
    fl = [max(r["flip_rate"], 5e-4) for r in sn[name]["rows"]]
    ax[2].loglog(eps, fl, mk + "-", color=col, ms=4, label=lab)
ax[2].set_xlabel(r"relative noise $\varepsilon_{\rm rel}$")
ax[2].set_ylabel("misclassification rate")
ax[2].legend(fontsize=6.2)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(R / f"fig3.{ext}", dpi=200)
print("figures saved")
