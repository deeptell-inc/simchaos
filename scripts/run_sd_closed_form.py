"""Schwinger-Dyson complete solution of the elliptic family (Theorem 3 upgrade).

Closed forms verified here:
  Ghat(p,q)   = phihat(p)phihat(q)/(1 - phihat(p)phihat(q)),
                phihat(p) = (p - sqrt(p^2-4tau))/(2tau)   [key identity
                p - tau*phihat(p) = 1/phihat(p) from the Catalan equation]
  Phi_tau(t;theta) = (1/t^2) sum_{k>=1} k^2 tau^{-k} |I_k(2 sqrt(tau) t e^{i theta})|^2
  Phi0(t;theta)    = I_1(2b)/b,  b = t sqrt(1 + 2 tau cos 2theta + tau^2)
  Family ramp:  X/Phi0 -> (1-tau)^2 (1+tau) t   (t -> inf, every 0 <= tau < 1)

Checks: (a) coefficient-level match with DP mixed moments (m,n<=8, two tau)
        (b) Phi closed vs direct series   (c) tau->1 Hermitian limit I1(4t)/2t
        (d) X anisotropy reproduces (2/3)tau(1-tau^2)cos2theta t^4
        (e) ramp ratio -> 1 in log domain up to t=150
Output: ../results/sd_closed_form.json
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
import json
import numpy as np
from scipy.special import iv, ive
from math import factorial, comb
from functools import lru_cache
from pathlib import Path

out = {}

def M_mn(m, n, tau):
    word = [0]*m + [1]*n
    @lru_cache(maxsize=None)
    def f(i, j):
        if i >= j: return 1.0
        if (j - i) % 2: return 0.0
        return sum((tau if word[i]==word[k] else 1.0)*f(i+1,k)*f(k+1,j)
                   for k in range(i+1, j, 2))
    return f(0, m+n)

def lam_mom(m, n, tau):
    return sum(comb(m,p)*comb(n,q)*tau**((m-p)+(n-q))/(p+n-q+1)
               for p in range(m+1) for q in range(n+1) if p+n-q == m-p+q)

def phi_closed(t, tau, theta=0.0, kmax=200):
    if tau == 0.0: return float(iv(0, 2*t))
    a = 2*np.sqrt(tau)*t*np.exp(1j*theta)
    return float(sum(k**2*tau**(-k)*abs(iv(k, a))**2 for k in range(1, kmax))/t**2)

def phi0_closed(t, tau, theta=0.0):
    b = t*np.sqrt(1 + 2*tau*np.cos(2*theta) + tau**2)
    return float(iv(1, 2*b)/b)

# (a) coefficient check: G = sum_k f_k(u) f_k(v), f_k = k I_k(2 rt t)/(tau^{k/2} t)
def fk_coeff(k, m, tau):
    if (m - k + 1) % 2 or m < k - 1: return 0.0
    j = (m - k + 1)//2
    return k*tau**j/(factorial(j)*factorial(j+k))
mism = 0
for tau in (0.3, 0.724):
    for m in range(1, 9):
        for n in range(1, 9):
            if (m+n) % 2: continue
            pred = sum(fk_coeff(k,m,tau)*fk_coeff(k,n,tau) for k in range(1,m+n+2)) \
                   * factorial(m)*factorial(n)
            if abs(pred - M_mn(m,n,tau)) > 1e-9*max(1, abs(M_mn(m,n,tau))): mism += 1
out["moment_check_mismatches"] = mism
print("(a) moment check mismatches:", mism)

# (b) closed vs direct series
rows = []
for tau in (0.2, 0.5, 0.724):
    for t in (0.5, 1.5, 3.0):
        a = t
        s = sum((a**m*a**n*M_mn(m,n,tau)/(factorial(m)*factorial(n)))
                for m in range(26) for n in range(26-m) if (m+n) % 2 == 0 and M_mn(m,n,tau))
        rows.append(dict(tau=tau, t=t, closed=phi_closed(t,tau), series=float(s)))
out["closed_vs_series"] = rows
print("(b) worst ratio:", max(abs(r["closed"]/r["series"]-1) for r in rows))

# (c) Hermitian limit
out["hermitian_limit"] = [dict(t=t, closed=phi_closed(t, 0.9999),
                               ref=float(iv(1,4*t)/(2*t))) for t in (0.5,1.5,3.0)]
print("(c) tau->1 worst dev:",
      max(abs(r["closed"]/r["ref"]-1) for r in out["hermitian_limit"]))

# (d) X anisotropy
rows = []
for tau in (0.3, 0.5, 0.724):
    t = 0.05
    X0  = phi_closed(t,tau,0.0)     - phi0_closed(t,tau,0.0)
    X90 = phi_closed(t,tau,np.pi/2) - phi0_closed(t,tau,np.pi/2)
    rows.append(dict(tau=tau, measured=X0-X90, predicted=(2/3)*tau*(1-tau**2)*t**4))
out["anisotropy_t4"] = rows
print("(d) aniso worst ratio dev:",
      max(abs(r["measured"]/r["predicted"]-1) for r in rows))

# (e) family ramp in log domain
def ramp_ratio(t, tau, kmax=2000):
    if tau == 0.0:
        return float(t*ive(0,2*t)/ive(1,2*t) - 1)
    z = 2*np.sqrt(tau)*t; b = t*(1+tau)
    ks = np.arange(1, kmax)
    lt = 2*np.log(ks) - ks*np.log(tau) + 2*np.log(ive(ks, z)) + 2*z
    m = lt.max()
    logPhi_t2 = m + np.log(np.exp(lt-m).sum())
    return float(np.exp(logPhi_t2 - 2*np.log(t) - np.log(ive(1,2*b)/b) - 2*b) - 1)
rows = []
for tau in (0.0, 0.2, 0.5, 0.724):
    for t in (20.0, 60.0, 150.0):
        rows.append(dict(tau=tau, t=t, ratio=ramp_ratio(t,tau),
                         pred=(1-tau)**2*(1+tau)*t))
out["family_ramp"] = rows
print("(e) ramp ratio/pred at t=150:",
      [round(r["ratio"]/r["pred"],5) for r in rows if r["t"]==150.0])

(RESULTS / "sd_closed_form.json").write_text(json.dumps(out, indent=1))
print("SAVED results/sd_closed_form.json")
