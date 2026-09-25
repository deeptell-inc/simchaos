"""A1: circuit-level end-to-end demo. Born-rule sampling of actual circuit
acceptance probabilities (no injected Gaussian noise).
(A) SWAP test on two Choi-state copies (theta=0 tier).
(B) Unitary-dilation block encoding of truncated-polynomial propagator;
    flag-acceptance sampling -> Phi-hat -> (chi,s*) classification.
(C) Sample-cost table at eps_rel=3%.
-> ../results/circuit_demo.json
"""
import json, math
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.stats import unitary_group
from scipy.linalg import expm, sqrtm
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
from models import ginue, nh_syk

rngS = np.random.default_rng(999)
out = {}

# ---------- (A) SWAP test on Choi copies (dilation channel, d=4)
def choi_purity_swap(d, chaotic, rng):
    U = unitary_group.rvs(2*d, random_state=rng) if chaotic else \
        np.kron(unitary_group.rvs(d, random_state=rng), unitary_group.rvs(2, random_state=rng))
    Ks = [U[a*d:(a+1)*d, 0:d] for a in range(2)]
    # Choi (normalized): J = (1/d) sum_ij Lambda(|i><j|) x |i><j|
    J = sum(np.kron(K, np.eye(d)) @ (np.outer(np.eye(d).flatten(), np.eye(d).flatten())/d) @ np.kron(K, np.eye(d)).conj().T for K in Ks)
    purity = float(np.real(np.trace(J @ J)))
    p_swap = 0.5*(1 + purity)          # SWAP-test ancilla P(0)
    return purity, p_swap

rows = []
for chaotic in (True, False):
    purity, p = choi_purity_swap(4, chaotic, np.random.default_rng(20260717))
    for shots in (10**4, 10**5, 10**6):
        est = [2*(rngS.binomial(shots, p)/shots) - 1 for _ in range(200)]
        rows.append(dict(chaotic=chaotic, purity_exact=purity, shots=shots,
                         est_mean=float(np.mean(est)), est_std=float(np.std(est))))
    print(f"A SWAP chaotic={chaotic}: purity={purity:.5f} " +
          "; ".join(f"1e{int(np.log10(r['shots']))}:{r['est_std']:.2e}" for r in rows[-3:]), flush=True)
out["swap_test"] = rows

# ---------- (B) dilation block encoding of truncated propagator
TS = 0.25*np.arange(1, 13); TAIL = slice(-5, None)
def dilation_acceptance(L, t, K=30):
    """Truncated Taylor propagator M_K = sum (tL)^k/k!, exact unitary dilation,
    acceptance of flag on maximally mixed input = ||M/aF||_F^2 / D."""
    D = L.shape[0]
    M = np.zeros_like(L); T = np.eye(D, dtype=complex)
    for k in range(K+1):
        M += T
        T = T @ (t*L) / (k+1)
    aF = np.linalg.norm(M, 2) * 1.0000001
    Mn = M/aF
    p = float(np.linalg.norm(Mn, 'fro')**2 / D)   # acceptance probability
    return p, aF

def classify(chi, s):
    if chi < 1e-10: return "integrable"
    if s < 1.30 or chi > 2.0: return "chaotic"
    if s > 1.60 and chi < 1.0: return "integrable"
    return "ambiguous"

cases = [("nh_syk_q4", lambda r: nh_syk(14,4,r), "chaotic"),
         ("nh_syk_q2", lambda r: nh_syk(14,2,r), "integrable"),
         ("ginue64",  lambda r: ginue(64,r),   "chaotic")]
bres = []
for name, gen, truth in cases:
    L = gen(np.random.default_rng(20260717))
    lam = np.linalg.eigvals(L)
    ps, aFs, phi0s = [], [], []
    for t in TS:
        p, aF = dilation_acceptance(L, t)
        ps.append(p); aFs.append(aF)
        phi0s.append(float(np.sum(np.exp(2*t*lam.real))/L.shape[0]))
    ps, aFs, phi0s = map(np.array, (ps, aFs, phi0s))
    for shots in (10**5, 10**6, 10**7):
        flips = amb = 0; N_MC = 100
        for _ in range(N_MC):
            phat = rngS.binomial(shots, ps)/shots
            Phi_hat = phat*aFs**2/L.shape[0]*L.shape[0]  # Phi = p*aF^2 (unnorm Frobenius/D)
            x = np.clip(Phi_hat/phi0s - 1, 1e-12, None)
            s = float(np.polyfit(np.log(TS[TAIL]), np.log(x[TAIL]), 1)[0])
            pred = classify(float(x.max()), s)
            flips += (pred != truth and pred != "ambiguous"); amb += (pred == "ambiguous")
        bres.append(dict(model=name, shots=shots, flip=flips/N_MC, abstain=amb/N_MC,
                         p_t3=float(ps[-1]), shots_needed_3pct=float((1-ps[-1])/(ps[-1]*0.03**2))))
        print(f"B {name} shots=1e{int(np.log10(shots))}: flip={flips/N_MC:.2f} abstain={amb/N_MC:.2f} "
              f"p(t=3)={ps[-1]:.3g}", flush=True)
out["dilation"] = bres

# ---------- (C) sample-cost table at t=3, eps_rel=3%
out["cost_table"] = [dict(model=r["model"], p_acc_t3=r["p_t3"],
                          shots_needed=r["shots_needed_3pct"]) for r in bres if r["shots"]==10**5]
for c in out["cost_table"]:
    print(f"C {c['model']}: p_acc(t=3)={c['p_acc_t3']:.4g} shots(3%)={c['shots_needed']:.3g}", flush=True)

(RESULTS / "circuit_demo.json").write_text(json.dumps(out, indent=1))
print("SAVED")
