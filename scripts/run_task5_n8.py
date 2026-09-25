"""Task 5 (v2, memory-safe): boundary-driven XXZ pair at n=8, delta-m=0 sector
(D=12870), assembled block-by-block over magnetization m (largest block 70^2)
instead of slicing the 65536^2 full Liouvillian.

L acts on pairs (i,j) with pop(i)=pop(j)=m. Blocks:
  diagonal (m):  -i(kron(H_m,I) - kron(I,H_m^T)) - 1/2 sum_K [kron((K+K)_m,I)+kron(I,conj((K+K)_m))]
  off-diag  (m -> m+c) for jump K of charge c: kron(K_{m+c,m}, conj(K_{m+c,m}))
Output: ../results/task5_n8.json
"""
import json, time
import numpy as np
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
from models import _spin_ops, _rescale
from diagnostics import csr_stats, excess_profile

R = RESULTS
TS = 0.25 * np.arange(1, 13)
TAIL = slice(-5, None)
N = 8


def build_sector_liouvillian(n, chaotic, rng, gp=1.2, gm=0.6, delta=0.5):
    ops = _spin_ops(n)
    d = 2 ** n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n - 1):
        H += ops["x"][i] @ ops["x"][i + 1] + ops["y"][i] @ ops["y"][i + 1] \
             + delta * (ops["z"][i] @ ops["z"][i + 1])
    if chaotic:
        for i in range(n):
            H += (0.5 * ((-1) ** i) + 0.15 * rng.uniform(-1, 1)) * ops["z"][i]
        for i in range(n - 2):
            H += 0.8 * (ops["z"][i] @ ops["z"][i + 2])
    sp = lambda i: ops["x"][i] + 1j * ops["y"][i]
    sm = lambda i: ops["x"][i] - 1j * ops["y"][i]
    jumps = []  # (K, charge)
    for (site, fac) in ((0, 1.0), (1, 0.5), (n - 2, 0.5), (n - 1, 1.0)):
        jumps.append((np.sqrt(gp * fac) * sp(site), +1))
        jumps.append((np.sqrt(gm * fac) * sm(site), -1))

    pop = np.array([bin(i).count("1") for i in range(d)])
    idx = {m: np.where(pop == m)[0] for m in range(n + 1)}
    sizes = {m: len(idx[m]) ** 2 for m in range(n + 1)}
    offs, off = {}, 0
    for m in range(n + 1):
        offs[m] = off; off += sizes[m]
    D = off
    L = np.zeros((D, D), dtype=complex)
    for m in range(n + 1):
        im = idx[m]; dm = len(im); Im = np.eye(dm)
        Hm = H[np.ix_(im, im)]
        A = -1j * (np.kron(Hm, Im) - np.kron(Im, Hm.T))
        for K, c in jumps:
            KdK = (K.conj().T @ K)[np.ix_(im, im)]
            A -= 0.5 * (np.kron(KdK, Im) + np.kron(Im, KdK.conj()))
        L[offs[m]:offs[m] + sizes[m], offs[m]:offs[m] + sizes[m]] = A
        for K, c in jumps:
            m2 = m + c
            if 0 <= m2 <= n:
                Kb = K[np.ix_(idx[m2], im)]
                B = np.kron(Kb, Kb.conj())
                L[offs[m2]:offs[m2] + sizes[m2], offs[m]:offs[m] + sizes[m]] += B
    return _rescale(L)


out = []
for chaotic in (True, False):
    t0 = time.time()
    rng = np.random.default_rng(20260717)
    L = build_sector_liouvillian(N, chaotic, rng)
    print(f"built chaotic={chaotic} D={L.shape[0]} ({time.time()-t0:.0f}s)", flush=True)
    lam = np.linalg.eigvals(L)
    print(f"eig done ({time.time()-t0:.0f}s)", flush=True)
    csr = csr_stats(lam, drop_real_axis=1e-9, bulk_fraction=0.9)
    prof = excess_profile(L, TS, np.array([0.0]), lam=lam)
    x0 = prof["profile"][0]
    s = float(np.polyfit(np.log(TS[TAIL]), np.log(np.abs(x0[TAIL]) + 1e-300), 1)[0])
    row = dict(chaotic=chaotic, D=int(L.shape[0]), r=csr["r_mean"], mcos=csr["mcos_mean"],
               n_csr=csr["n_used"], chi=prof["chi"], s_late=s,
               profile=x0.tolist(), runtime_s=round(time.time() - t0))
    out.append(row)
    print(f"RESULT chaotic={chaotic}: r={row['r']:.3f} -cos={row['mcos']:.3f} "
          f"chi={row['chi']:.3g} s*={s:.3f} ({row['runtime_s']}s)", flush=True)
    R.joinpath("task5_n8.json").write_text(json.dumps(out, indent=1))
print("saved")
