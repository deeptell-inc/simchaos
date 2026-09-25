"""P3 decision run: does chi = max_(t,theta) X/Phi0 separate chaotic from
integrable non-Hermitian generators, in agreement with CSR ground truth?

Usage: python run_p3.py [--quick]
Outputs: ../results/p3_results.json, ../results/p3_summary.png
Seeds: 20260717 + i, i = 0..N_SEEDS-1 (mean +/- std reported, convention 3-10 seeds)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
from models import MODELS
from diagnostics import csr_stats, eig_overlap_data, excess_profile


# model -> (size argument, CSR options)
CONFIG = {
    "ginue":                 dict(size=1024, csr=dict(bulk_fraction=0.9), cls="chaotic"),
    "poisson_normal":        dict(size=1024, csr=dict(bulk_fraction=0.9), cls="integrable"),
    "random_liouvillian":    dict(size=32,   csr=dict(drop_real_axis=1e-9, bulk_fraction=0.9), cls="chaotic"),
    "dephasing_liouvillian": dict(size=32,   csr=dict(drop_real_axis=1e-9, bulk_fraction=0.9), cls="integrable"),
    "nh_syk_q4":             dict(size=18,   csr=dict(bulk_fraction=0.9), cls="chaotic"),
    "nh_syk_q2":             dict(size=18,   csr=dict(bulk_fraction=0.9), cls="integrable"),
    "xxz_liou_chaotic":      dict(size=6,    csr=dict(drop_real_axis=1e-9, bulk_fraction=0.9), cls="chaotic"),
    "xxz_liou_integrable":   dict(size=6,    csr=dict(drop_real_axis=1e-9, bulk_fraction=0.9), cls="integrable"),
    "hatano_nelson":         dict(size=1024, csr=dict(bulk_fraction=0.9), cls="integrable(non-normal stress test)"),
}

TS = 0.25 * np.arange(1, 13)                     # t = 0.25 .. 3.0
THETAS = np.array([0.0, np.pi / 6, np.pi / 3, np.pi / 2,
                   2 * np.pi / 3, 5 * np.pi / 6, np.pi])


def run(n_seeds: int = 5) -> dict:
    out = {}
    for name, cfg in CONFIG.items():
        rows = []
        for s in range(n_seeds):
            rng = np.random.default_rng(20260717 + s)
            t0 = time.time()
            L = MODELS[name](cfg["size"], rng)
            lam = np.linalg.eigvals(L)
            csr = csr_stats(lam, **cfg["csr"])
            prof = excess_profile(L, TS, THETAS, lam=lam)
            x0 = prof["profile"][0]
            tail = slice(-5, None)
            s_late = float(np.polyfit(np.log(TS[tail]), np.log(np.abs(x0[tail]) + 1e-300), 1)[0])
            okk = eig_overlap_data(L)["O_kk"] if L.shape[0] <= 1100 else None
            rows.append({
                "seed": 20260717 + s,
                "D": int(L.shape[0]),
                "r_mean": csr["r_mean"],
                "mcos_mean": csr["mcos_mean"],
                "n_csr": csr["n_used"],
                "chi": prof["chi"],
                "chi_theta0": prof["chi_theta0"],
                "s_late": s_late,
                "log10_mean_Okk": float(np.log10(np.mean(okk))) if okk is not None else None,
                "runtime_s": round(time.time() - t0, 1),
            })
            print(f"{name} seed{s}: D={rows[-1]['D']} r={csr['r_mean']:.3f} "
                  f"-cos={csr['mcos_mean']:.3f} chi={prof['chi']:.3g} s={s_late:.2f} "
                  f"({rows[-1]['runtime_s']}s)", flush=True)
        agg = {k: (float(np.mean([r[k] for r in rows if r[k] is not None])),
                   float(np.std([r[k] for r in rows if r[k] is not None])))
               for k in ("r_mean", "mcos_mean", "chi", "chi_theta0", "s_late")}
        out[name] = {"class": cfg["cls"], "rows": rows, "agg": agg}
    return out


def plot(results: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, res in results.items():
        chaotic = res["class"].startswith("chaotic")
        color = "crimson" if chaotic else "steelblue"
        marker = "^" if "stress" in res["class"] else ("o" if chaotic else "s")
        r, dr = res["agg"]["r_mean"]
        c, dc = res["agg"]["mcos_mean"]
        chi, dchi = res["agg"]["chi"]
        ax[0].errorbar(r, c, xerr=dr, yerr=dc, fmt=marker, color=color, label=name)
        ax[1].errorbar([r], [chi], xerr=[dr], yerr=[dchi], fmt=marker, color=color)
    ax[0].axvline(2 / 3, ls=":", c="gray"); ax[0].axvline(0.738, ls=":", c="gray")
    ax[0].set_xlabel(r"$\langle r\rangle$"); ax[0].set_ylabel(r"$-\langle\cos\theta\rangle$")
    ax[0].set_title("CSR ground truth"); ax[0].legend(fontsize=7, loc="best")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\langle r\rangle$ (CSR)"); ax[1].set_ylabel(r"$\chi=\max X/\Phi_0$")
    ax[1].set_title("Overlap excess discriminant vs CSR")
    fig.tight_layout(); fig.savefig(path, dpi=160)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    res = run(n_seeds=2 if args.quick else 5)
    (RESULTS / "p3_results.json").write_text(json.dumps(res, indent=1))
    plot(res, RESULTS / "p3_summary.png")
    print("saved:", RESULTS / "p3_results.json")
