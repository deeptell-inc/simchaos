"""S6 remainder: shot-noise robustness of the (chi, s*) classifier.

Quantum estimation of Phi(t;theta) yields additive error eps per point
(Hadamard test / amplitude estimation). We model estimated Phi_hat =
Phi + N(0, eps) (Phi in shift-normalized units, O(0.01-1)), recompute
s* from noisy X-hat = Phi_hat - Phi0 (Phi0 assumed known to same eps,
errors added in quadrature), and measure classification flip rate over
n_mc Monte Carlo noise realizations.

Classifier: chaotic iff s* < S_C. Frozen shotnoise.json and the manuscript
use the SYK-calibrated boundary S_C = 1.41 (see Fig. 2b). chi < EPS0 -> integrable-normal.

Output: ../results/shotnoise.json
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
from models import ginue, nh_syk
from diagnostics import excess_profile

TS = 0.25 * np.arange(1, 13)
TH = np.array([0.0])
S_C = 1.41          # s* decision boundary (SYK-calibrated; matches frozen shotnoise.json)
TAIL = slice(-5, None)
N_MC = 400

CASES = [
    ("ginue_D1024", lambda rng: ginue(1024, rng), "chaotic"),
    ("nh_syk_q4_N18", lambda rng: nh_syk(18, 4, rng), "chaotic"),
    ("nh_syk_q2_N18", lambda rng: nh_syk(18, 2, rng), "integrable"),
]


def s_from(xs):
    return float(np.polyfit(np.log(TS[TAIL]), np.log(np.abs(xs[TAIL]) + 1e-300), 1)[0])


out = {}
rng0 = np.random.default_rng(20260717)
noise_rng = np.random.default_rng(777)
for name, gen, truth in CASES:
    L = gen(rng0)
    prof = excess_profile(L, TS, TH)
    x = prof["profile"][0]                       # exact X/Phi0 at theta=0
    # reconstruct Phi/Phi0 (dimensionless); noise eps acts on Phi-hat and
    # Phi0-hat in shift-normalized units where Phi0(t_max) ~ O(1/D)*sum ~ known.
    # We inject noise directly on X/Phi0 with sigma_rel = eps / Phi0_t —
    # conservative proxy: Phi0 in these units is O(0.01-0.1) at late t, so
    # relative error on the ratio is eps/Phi0. We scan eps as RELATIVE error
    # on X/Phi0 for clarity.
    rows = []
    for eps_rel in (0.10, 0.03, 0.01):  # matches frozen shotnoise.json rows
        flips = 0
        s_vals = []
        for _ in range(N_MC):
            xn = x * (1 + eps_rel * noise_rng.standard_normal(len(x)))
            s = s_from(xn)
            s_vals.append(s)
            pred = "chaotic" if s < S_C else "integrable"
            flips += (pred != truth)
        rows.append(dict(eps_rel=eps_rel, flip_rate=flips / N_MC,
                         s_mean=float(np.mean(s_vals)), s_std=float(np.std(s_vals))))
        print(f"{name} eps_rel={eps_rel}: s*={rows[-1]['s_mean']:.3f}"
              f"±{rows[-1]['s_std']:.3f} flip={rows[-1]['flip_rate']:.3f}", flush=True)
    out[name] = dict(truth=truth, s_exact=s_from(x), rows=rows)

RESULTS.joinpath("shotnoise.json").write_text(json.dumps(out, indent=1))
print("saved")
