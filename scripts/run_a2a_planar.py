"""A2a: nH-SYK planar reduction to elliptic family (Theorem 3).

1. eta(N,q) exact Q-Hermite crossing weight  -> q=4 is planar, q=2 is not
2. tau_eff = (1-kappa^2)/(1+kappa^2) moment check against Tr[H^2]/Tr[HH+]
3. SYK q4 measured excess profile vs parameter-free Theorem-3 elliptic curve

Output: ../results/a2a_planar.json
Recovered from session 5c1ba5e3 (2026-07-24 run); numbers match claims-ledger A2a row.
"""
import json
import numpy as np
from math import comb, factorial
from functools import lru_cache
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
from models import nh_syk
from diagnostics import excess_profile

out = {}

# 1. eta(N,q): Q-Hermite crossing weight
def eta(N, q):
    return sum((-1)**k * comb(q, k) * comb(N - q, q - k) for k in range(q + 1)) / comb(N, q)

out["eta"] = {f"N{N}": {"q4": eta(N, 4), "q2": eta(N, 2)} for N in (14, 16, 18, 20)}
for N in (14, 16, 18, 20):
    print(f"eta(N={N}, q=4) = {eta(N,4):+.4f}   eta(N={N}, q=2) = {eta(N,2):+.4f}")

# 2. tau_eff from coupling covariance vs measured moment ratio
kappa = 0.4
tau_pred = (1 - kappa**2) / (1 + kappa**2)
out["tau_pred"] = tau_pred
out["moment_ratio"] = {}
for N in (16, 18):
    rng = np.random.default_rng(20260717)
    L = nh_syk(N, 4, rng)  # ratio Tr[H^2]/Tr[HH+] is scale-invariant
    r2 = np.trace(L @ L) / np.trace(L @ L.conj().T)
    out["moment_ratio"][f"N{N}"] = [float(r2.real), float(r2.imag)]
    print(f"N={N}: Tr[H^2]/Tr[HH+] = {r2.real:+.4f}{r2.imag:+.4f}i   predicted tau_eff = {tau_pred:.4f}")

# 3. measured SYK q4 profile vs elliptic theory at tau_eff (no free parameters)
def M_mn(m, n, tau):
    word = [0] * m + [1] * n
    @lru_cache(maxsize=None)
    def f(i, j):
        if i >= j: return 1.0
        if (j - i) % 2: return 0.0
        return sum((tau if word[i] == word[k] else 1.0) * f(i + 1, k) * f(k + 1, j)
                   for k in range(i + 1, j, 2))
    return f(0, m + n)

def lam_mom(m, n, tau):
    return sum(comb(m, p) * comb(n, q2) * tau**((m - p) + (n - q2)) / (p + n - q2 + 1)
               for p in range(m + 1) for q2 in range(n + 1) if p + n - q2 == m - p + q2)

def theory_ratio(t, tau, K=24):
    a = t; P = P0 = 0.0
    for m in range(K + 1):
        for n in range(K + 1 - m):
            if (m + n) % 2: continue
            c = a**m * a**n / (factorial(m) * factorial(n))
            P += c * M_mn(m, n, tau); P0 += c * lam_mom(m, n, tau)
    return P / P0 - 1

TS = 0.25 * np.arange(1, 13)
r_scale = 1 + tau_pred  # spectral radius (semi-major axis 1+tau) rescaled to 1
prof = []
for s in range(3):
    L = nh_syk(18, 4, np.random.default_rng(20260717 + s))
    prof.append(excess_profile(L, TS, np.array([0.0]))["profile"][0])
meas = np.mean(prof, axis=0)
rows = []
print("\nt_model  measured(q4,N=18)  elliptic-theory(tau=0.724, t/r)   ratio")
for i, t in enumerate(TS):
    th = theory_ratio(t / r_scale, tau_pred)
    rows.append({"t": float(t), "measured": float(meas[i]), "theory": float(th),
                 "ratio": float(meas[i] / th)})
    print(f"{t:5.2f}   {meas[i]:8.4f}          {th:8.4f}                  {meas[i]/th:6.3f}")
out["curve"] = rows

with open((RESULTS / "a2a_planar.json"), "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote ../results/a2a_planar.json")
