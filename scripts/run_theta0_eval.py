"""Pre-registered theta=0-only decision rule, evaluated on unseen instances.

PRE-REGISTERED RULE (frozen 2026-09-09 BEFORE any evaluation run, calibrated
only from the frozen results/p3_results.json aggregates; no rule change after
this header is written):
  features at theta=0 only: chi0 = max_t X/Phi0, s* = late log-log slope
  (window t in [2.0, 3.0], same standardized window as the paper).
  1. chi0 < 1e-10          -> integrable-normal
  2. chi0 >= 2.0           -> chaotic          (GinUE-class magnitude rescue;
                                                frozen chi_theta0(GinUE)=2.21)
  3. else s* <= 1.30       -> chaotic
     s* >= 1.60            -> integrable-nonnormal
     1.30 < s* < 1.60      -> abstain
  Quasi-1D (Hatano-Nelson) is declared out-of-scope by the paper; it is run
  and reported separately, not counted in the primary table.

Expected from frozen data (recorded to keep us honest): random Liouvillians
have chi_theta0 = 0.79 +- 0.10 and s* near the abstention band, so the rule
is EXPECTED to abstain on them -- the evaluation quantifies the coverage cost
of the theta=0-only tier, which the manuscript now states qualitatively.

Evaluation set: fresh seeds 100-104 (never used in any prior run) for the six
in-family models + three out-of-family ensembles (GinOE, elliptic tau=0.15,
Gaussian-spectrum normal) + Hatano-Nelson (out-of-scope, reported apart).
Output: ../results/theta0_eval.json with per-instance rows and per-family
correct/incorrect/abstain/coverage.
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
from models import (ginue, poisson_normal, random_liouvillian,
                    dephasing_liouvillian, nh_syk, hatano_nelson, _rescale)
from diagnostics import excess_profile

TS = 0.25 * np.arange(1, 13)
TAIL = slice(-5, None)          # t in [2.0, 3.0], standardized window

def ginoe(D, rng):
    return _rescale(rng.standard_normal((D, D)) / np.sqrt(D))

def elliptic_ginue(D, rng, tau=0.15):
    A = (rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))) / np.sqrt(2 * D)
    return _rescale(A + tau * A.T)  # E[X_ij X_ji] = tau * E[|X_ij|^2]

def gaussian_normal(D, rng):
    lam = (rng.standard_normal(D) + 1j * rng.standard_normal(D)) / np.sqrt(2)
    Q = np.linalg.qr(rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D)))[0]
    return _rescale(Q @ np.diag(lam) @ Q.conj().T)

MODELS = {
    # name: (constructor, truth, family_group)
    "ginue":        (lambda rng: ginue(1024, rng),                 "chaotic",             "in-family"),
    "poisson_norm": (lambda rng: poisson_normal(1024, rng),        "integrable-normal",   "in-family"),
    "rand_liou":    (lambda rng: random_liouvillian(32, rng),      "chaotic",             "in-family"),
    "deph_liou":    (lambda rng: dephasing_liouvillian(32, rng),   "integrable-normal",   "in-family"),
    "syk_q4":       (lambda rng: nh_syk(18, 4, rng),               "chaotic",             "in-family"),
    "syk_q2":       (lambda rng: nh_syk(18, 2, rng),               "integrable-nonnormal","in-family"),
    "ginoe":        (lambda rng: ginoe(1024, rng),                 "chaotic",             "out-of-family"),
    "elliptic15":   (lambda rng: elliptic_ginue(1024, rng),        "chaotic",             "out-of-family"),
    "gauss_norm":   (lambda rng: gaussian_normal(1024, rng),       "integrable-normal",   "out-of-family"),
    "hatano":       (lambda rng: hatano_nelson(1024, rng),         "quasi-1D",            "out-of-scope"),
}

def classify(chi0, s):
    if chi0 < 1e-10: return "integrable-normal"
    if chi0 >= 2.0:  return "chaotic"
    if s <= 1.30:    return "chaotic"
    if s >= 1.60:    return "integrable-nonnormal"
    return "abstain"

rows = []
for name, (ctor, truth, group) in MODELS.items():
    for seed in range(100, 105):
        t0 = time.time()
        L = ctor(np.random.default_rng(seed))
        prof = excess_profile(L, TS, np.array([0.0]))
        x0 = np.abs(prof["profile"][0])
        chi0 = float(np.max(x0))
        s = float(np.polyfit(np.log(TS[TAIL]), np.log(x0[TAIL] + 1e-300), 1)[0])
        call = classify(chi0, s)
        rows.append(dict(model=name, seed=seed, group=group, truth=truth,
                         chi0=chi0, s_late=s, call=call,
                         correct=(call == truth), abstain=(call == "abstain"),
                         runtime=round(time.time() - t0, 1)))
        r = rows[-1]
        print(f"{name} s{seed}: chi0={chi0:.3g} s*={s:.3f} -> {call} "
              f"[truth {truth}] {'OK' if r['correct'] else ('ABSTAIN' if r['abstain'] else 'WRONG')}",
              flush=True)

# per-family summary
summary = {}
for name in MODELS:
    rs = [r for r in rows if r["model"] == name]
    n = len(rs)
    summary[name] = dict(
        group=rs[0]["group"], truth=rs[0]["truth"],
        correct=sum(r["correct"] for r in rs),
        wrong=sum((not r["correct"]) and (not r["abstain"]) for r in rs),
        abstain=sum(r["abstain"] for r in rs),
        coverage=round(1 - sum(r["abstain"] for r in rs) / n, 2), n=n)

(RESULTS / "theta0_eval.json").write_text(json.dumps(dict(rows=rows, summary=summary), indent=1))
print("\nSUMMARY"); [print(k, v) for k, v in summary.items()]
print("SAVED results/theta0_eval.json")
