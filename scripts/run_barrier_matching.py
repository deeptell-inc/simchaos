"""All-protocol query barrier: numerical ingredients (Appendix D upgrade).

1. Bottleneck (minimax) matching GinUE spectrum <-> iid-uniform (Poisson)
   configuration of the same intensity: max displacement delta scales as
   D^{-1/2} (log D)^{3/4} (Leighton-Shor type), while the two configurations
   keep O(1)-different CSR statistics (<r> ~ 0.73 vs 2/3).
2. Lipschitz constant of the canonical 2x2 dilation block encoding
   V(lam) = [[lam, -s],[s, conj(lam)]], s = sqrt(1-|lam|^2), on |lam| <= 1/2:
   empirical max ~1.15, analytic bound 1 + 1/sqrt(3/4) ~ 2.155.

With the BBBV hybrid bound |p_V - p_V'| <= 2T ||V - V'||, these give
T = Omega(sqrt(D)/(log D)^{3/4}) for any protocol distinguishing the pair.
Output: ../results/barrier_matching.json
"""
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
from models import ginue
from diagnostics import csr_stats


def bottleneck_match(P, Q):
    dif = np.abs(P[:, None] - Q[None, :])
    lo, hi = 0.0, dif.max()
    def feasible(r):
        m = maximum_bipartite_matching(csr_matrix(dif <= r), perm_type='column')
        return (m >= 0).all()
    for _ in range(40):
        mid = (lo + hi)/2
        if feasible(mid): hi = mid
        else: lo = mid
    return hi

rng = np.random.default_rng(7)
rows = []
for D in (256, 512, 1024, 2048):
    lamG = np.linalg.eigvals(ginue(D, rng))
    R = np.abs(lamG).max()
    lamP = np.sqrt(rng.uniform(0, R**2, D))*np.exp(1j*rng.uniform(0, 2*np.pi, D))
    d = bottleneck_match(lamG, lamP)
    rows.append(dict(D=D, delta=float(d),
                     delta_sqrtD=float(d*np.sqrt(D)),
                     delta_scaled=float(d*np.sqrt(D)/np.log(D)**0.75),
                     r_ginue=csr_stats(lamG, bulk_fraction=0.9)["r_mean"],
                     r_poisson=csr_stats(lamP, bulk_fraction=0.9)["r_mean"]))
    print(rows[-1])

def V(lam):
    s = np.sqrt(1 - abs(lam)**2)
    return np.array([[lam, -s], [s, np.conj(lam)]])
worst = 0.0
for _ in range(20000):
    l1 = 0.5*(rng.uniform(-1,1) + 1j*rng.uniform(-1,1))
    l2 = l1 + 0.01*(rng.standard_normal() + 1j*rng.standard_normal())
    if abs(l1) > 0.5 or abs(l2) > 0.5: continue
    worst = max(worst, np.linalg.norm(V(l1)-V(l2), 2)/abs(l1-l2))
print("BE Lipschitz empirical:", round(worst, 4))

(RESULTS / "barrier_matching.json").write_text(json.dumps(
    dict(matching=rows, be_lipschitz_empirical=float(worst),
         be_lipschitz_bound=float(1+1/np.sqrt(0.75))), indent=1))
print("SAVED results/barrier_matching.json")
