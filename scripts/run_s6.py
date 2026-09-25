"""S6 scans: (A) D-scaling of s*, (B) theta-anisotropy for HN separation,
(C) XXZ Liouvillian retune with bulk damping.
Output: ../results/s6_scan.json
"""
import json
import numpy as np
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
from models import (ginue, poisson_normal, nh_syk, hatano_nelson,
                    xxz_liouvillian, _vectorized_lindblad, _spin_ops, _rescale)
from diagnostics import csr_stats, excess_profile

TS = 0.25 * np.arange(1, 13)
TH = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])


def s_late_and_aniso(L, lam=None):
    res = excess_profile(L, TS, TH, lam=lam)
    x0 = res["profile"][0]
    tail = slice(-5, None)
    s = float(np.polyfit(np.log(TS[tail]), np.log(np.abs(x0[tail]) + 1e-300), 1)[0])
    # anisotropy of late-time excess across theta
    late = res["profile"][:, -1]
    aniso = float(late.max() / max(late.min(), 1e-300))
    return s, aniso, float(res["chi"])


def xxz_bulk(n, chaotic, rng, g_bulk=0.35):
    """XXZ Liouvillian + weak bulk amplitude damping on every site (breaks
    the weak-U(1)? no: sigma- carries charge -> delta_m still conserved;
    keep the delta_m=0 projection by reusing xxz_liouvillian's H but adding
    bulk jumps here)."""
    ops = _spin_ops(n)
    d = 2 ** n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n - 1):
        H += ops["x"][i] @ ops["x"][i + 1] + ops["y"][i] @ ops["y"][i + 1]
        H += 0.5 * (ops["z"][i] @ ops["z"][i + 1])
    if chaotic:
        for i in range(n):
            H += 0.5 * ((-1) ** i) * ops["z"][i]
        for i in range(n - 2):
            H += 0.8 * (ops["z"][i] @ ops["z"][i + 2])
    sp = lambda i: ops["x"][i] + 1j * ops["y"][i]
    sm = lambda i: ops["x"][i] - 1j * ops["y"][i]
    jumps = [np.sqrt(0.6) * sp(0), np.sqrt(0.3) * sm(0),
             np.sqrt(0.6) * sm(n - 1), np.sqrt(0.3) * sp(n - 1)]
    jumps += [np.sqrt(g_bulk * (1 + 0.3 * ((-1) ** i))) * sm(i) for i in range(n)]
    L = _vectorized_lindblad(H, jumps)
    pop = np.array([bin(i).count("1") for i in range(d)])
    bi, bj = np.divmod(np.arange(d * d), d)
    P = np.eye(d * d)[:, pop[bi] == pop[bj]]
    return _rescale(P.T @ L @ P)


out = {}

# (A) D-scaling of s* + (B) anisotropy, 3 seeds each
scan = []
for D in (256, 512, 1024, 2048):
    for s_ix in range(3):
        rng = np.random.default_rng(20260717 + s_ix)
        for name, gen in (("ginue", lambda: ginue(D, rng)),
                          ("hatano_nelson", lambda: hatano_nelson(D, rng))):
            L = gen()
            s, an, chi = s_late_and_aniso(L)
            scan.append(dict(model=name, D=D, seed=s_ix, s_late=s, aniso=an, chi=chi))
            print(f"A: {name} D={D} seed{s_ix}: s*={s:.3f} aniso={an:.2f} chi={chi:.3g}", flush=True)
for N in (14, 16, 18):
    for s_ix in range(3):
        rng = np.random.default_rng(20260717 + s_ix)
        for q in (4, 2):
            L = nh_syk(N, q, rng)
            s, an, chi = s_late_and_aniso(L)
            scan.append(dict(model=f"nh_syk_q{q}", D=L.shape[0], seed=s_ix, s_late=s, aniso=an, chi=chi))
            print(f"A: syk q{q} N={N} seed{s_ix}: s*={s:.3f} aniso={an:.2f} chi={chi:.3g}", flush=True)
out["scan"] = scan

# (C) XXZ retune with bulk damping
xxz = []
for chaotic in (True, False):
    rng = np.random.default_rng(20260717)
    L = xxz_bulk(6, chaotic, rng)
    lam = np.linalg.eigvals(L)
    csr = csr_stats(lam, drop_real_axis=1e-9, bulk_fraction=0.9)
    s, an, chi = s_late_and_aniso(L, lam=lam)
    xxz.append(dict(chaotic=chaotic, D=L.shape[0], r=csr["r_mean"], mcos=csr["mcos_mean"],
                    n=csr["n_used"], s_late=s, aniso=an, chi=chi))
    print(f"C: xxz_bulk chaotic={chaotic}: r={csr['r_mean']:.3f} -cos={csr['mcos_mean']:.3f} "
          f"n={csr['n_used']} s*={s:.3f} chi={chi:.3g}", flush=True)
out["xxz_bulk"] = xxz

RESULTS.joinpath("s6_scan.json").write_text(json.dumps(out, indent=1))
print("saved")
