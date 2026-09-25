"""Girko-KPM baseline parameter scan at n=6 (validation refinement).

First validation run (eta=0.06, 17^2 grid, 64 moments) FAILED with s*_est=1.07
vs exact 2.01 -- the over-smoothed baseline pushed the slope toward fake chaos,
reproducing with the real KPM implementation exactly the eta-failure mode that
task7 identified synthetically (results/task7_endtoend.json: eta=0.1 -> 77%
fake-chaos misclassification). This scan tests whether task7's resource
requirement eta <= 0.02 rescues the estimator, before spending hours at n=8.

Output: ../results/a3_girko_scan.json
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
import json, time
import numpy as np
from pathlib import Path
from scipy.linalg import expm
from run_a3_sparse import (mixed_jump_chain_sparse, phi_hutchinson,
                           phi0_girko, s_late, TS, RESULTS)

L6 = mixed_jump_chain_sparse(6, True, np.random.default_rng(20260717))
D6 = L6.shape[0]
lam6 = np.linalg.eigvals(L6.toarray())
sh6 = lam6.real.max()
phi_ex = []
step = expm(TS[0] * L6.toarray()); E = None
for k in range(len(TS)):
    E = step if E is None else step @ E
    phi_ex.append(np.linalg.norm(E, "fro") ** 2 / D6 * np.exp(-2 * TS[k] * sh6))
phi_ex = np.array(phi_ex)
phi0_ex = np.array([np.mean(np.exp(2 * t * (lam6.real - sh6))) for t in TS])
s_exact = s_late(phi_ex / phi0_ex - 1)
phi_h, shH = phi_hutchinson(L6)

configs = [
    dict(eta=0.02, ngrid=25, n_mom=128, n_probe=8, pad=0.10),
    dict(eta=0.02, ngrid=33, n_mom=128, n_probe=8, pad=0.10),
    dict(eta=0.01, ngrid=33, n_mom=192, n_probe=8, pad=0.10),
]
out = dict(s_exact=s_exact, configs=[])
for cfg in configs:
    t0 = time.time()
    phi0_g, shG = phi0_girko(L6, **cfg)
    # shift-align both to exact shift for comparison
    p0 = phi0_g * np.exp(2 * TS * (shG - sh6))
    err0 = float(np.max(np.abs(p0 / phi0_ex - 1)))
    x_est = (phi_h * np.exp(2 * TS * (shH - sh6))) / p0 - 1
    row = dict(**cfg, err_phi0_max=err0, s_est=s_late(x_est),
               ds=abs(s_late(x_est) - s_exact), runtime=round(time.time() - t0))
    out["configs"].append(row)
    print(f"eta={cfg['eta']} grid={cfg['ngrid']} mom={cfg['n_mom']}: "
          f"max|dPhi0/Phi0|={err0:.3f} s*={row['s_est']:.3f} "
          f"(exact {s_exact:.3f}, ds={row['ds']:.3f}) [{row['runtime']}s]",
          flush=True)
    (RESULTS / "a3_girko_scan.json").write_text(json.dumps(out, indent=1))
print("SAVED")
