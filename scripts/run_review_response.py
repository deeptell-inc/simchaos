"""Panel-response data supplement: omega scans, alpha_LCU, SYK N=20, out-of-family holdout,
appendix precision reproduction. -> ../results/review_response.json"""
import json, math, time
import numpy as np
from itertools import combinations
from pathlib import Path
from scipy.special import iv
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
from models import MODELS, ginue, hatano_nelson, nh_syk, random_liouvillian, poisson_normal, _rescale
from diagnostics import excess_profile

R = RESULTS; out = {}
TS = 0.25*np.arange(1,13); TAIL = slice(-5,None)
def sstar(L, lam=None):
    p = excess_profile(L, TS, np.array([0.0]), lam=lam)
    x0 = p["profile"][0]
    return float(np.polyfit(np.log(TS[TAIL]), np.log(np.abs(x0[TAIL])+1e-300),1)[0]), float(p["chi"])
def omega(L, th=0.0):
    M = (np.exp(1j*th)*L + np.exp(-1j*th)*L.conj().T)/2
    return float(np.linalg.eigvalsh(M).max())

# (a) omega/specrad scans (specrad=1 by _rescale)
sc = {}
for name, sizes in (("ginue",[256,512,1024,2048]),("hatano_nelson",[256,512,1024,2048]),
                    ("random_liouvillian",[16,24,32]),("nh_syk_q4",[14,16,18])):
    rows=[]
    for sz in sizes:
        oms=[]
        for s in range(3):
            L = MODELS[name](sz, np.random.default_rng(20260717+s))
            oms.append(max(omega(L,th) for th in (0, np.pi/4, np.pi/2)))
        rows.append(dict(size=sz, D=int(L.shape[0]), omega_max=float(np.mean(oms)), std=float(np.std(oms))))
        print(f"omega {name} {sz}: {rows[-1]['omega_max']:.3f}±{rows[-1]['std']:.3f}", flush=True)
    sc[name]=rows
out["omega_scan"]=sc

# (b) alpha_LCU for SYK (couplings sum) — explicit
al=[]
for N in (14,16,18):
    rng = np.random.default_rng(20260717)
    sd = np.sqrt(math.factorial(3)/N**3); nt = math.comb(N,4)
    Js = sd*(rng.standard_normal(nt)+1j*0.4*rng.standard_normal(nt))
    L = nh_syk(N,4,np.random.default_rng(20260717))
    # alpha in rescaled units: alpha_LCU / (unrescaled spectral radius). Estimate radius ratio via norm:
    # reconstruct unrescaled H spectral radius from L: unavailable directly; report alpha_LCU/||H||2 via rebuild small
    al.append(dict(N=N, n_terms=int(nt), alpha_lcu_raw=float(np.abs(Js).sum())))
    print(f"alpha N={N}: {al[-1]}", flush=True)
out["alpha_lcu"]=al

# (c) SYK N=20 s* (D=512 sector)
for q in (4,2):
    t0=time.time()
    L = nh_syk(20,q,np.random.default_rng(20260717))
    s,chi = sstar(L)
    out[f"syk_N20_q{q}"]=dict(D=int(L.shape[0]), s_late=s, chi=chi, runtime=round(time.time()-t0))
    print(f"SYK N=20 q={q}: D={L.shape[0]} s*={s:.3f} chi={chi:.3g} ({out[f'syk_N20_q{q}']['runtime']}s)", flush=True)

# (d) out-of-family holdout: elliptic tau=0.3 (chaotic) + Gaussian-density normal (integrable)
def elliptic(D, tau, rng):
    H1 = rng.standard_normal((D,D))+1j*rng.standard_normal((D,D)); H1=(H1+H1.conj().T)/2
    H2 = rng.standard_normal((D,D))+1j*rng.standard_normal((D,D)); H2=(H2+H2.conj().T)/2
    G = (np.sqrt((1+tau)/2)*H1 + 1j*np.sqrt((1-tau)/2)*H2)*np.sqrt(2)/np.sqrt(2*D)
    return _rescale(G)
def gauss_normal(D, rng):
    lam = (rng.standard_normal(D)+1j*rng.standard_normal(D))/np.sqrt(2)
    Z = rng.standard_normal((D,D))+1j*rng.standard_normal((D,D))
    Q,Rr = np.linalg.qr(Z); Q=Q*(np.diagonal(Rr)/np.abs(np.diagonal(Rr)))
    return _rescale((Q*lam)@Q.conj().T)
def classify(chi,s):
    if chi < 1e-10: return "integrable"
    if s < 1.30 or chi > 2.0: return "chaotic"
    if s > 1.60 and chi < 1.0: return "integrable"
    return "ambiguous"
oof=[]; ok=amb=0
for kind, gen, truth in (("elliptic_tau0.3", lambda r: elliptic(1024,0.3,r), "chaotic"),
                          ("gauss_normal", lambda r: gauss_normal(1024,r), "integrable")):
    for i in range(3):
        L = gen(np.random.default_rng(20270201+i))
        s,chi = sstar(L)
        pred = classify(chi,s); ok += (pred==truth); amb += (pred=="ambiguous")
        oof.append(dict(kind=kind, seed=i, chi=chi, s_late=s, pred=pred, truth=truth))
        print(f"OOF {kind} s{i}: chi={chi:.3g} s*={s:.3f} -> {pred} [{'OK' if pred==truth else 'X'}]", flush=True)
out["oof_holdout"]=dict(rows=oof, correct=ok, total=len(oof), ambiguous=amb)

# (e) appendix finite-D reproduction (3 seeds, documented procedure)
ana = TS*iv(0,2*TS)/iv(1,2*TS)-1
fd=[]
for D in (512,2048):
    dm, db = [], []
    for s in range(3):
        r = np.random.default_rng(20260717+s)
        G = (r.standard_normal((D,D))+1j*r.standard_normal((D,D)))/np.sqrt(2*D)
        lam = np.linalg.eigvals(G)
        for norm, acc in ((np.abs(lam).max(), dm),(np.sqrt(2*np.mean(np.abs(lam)**2)), db)):
            x = excess_profile(G/norm, TS, np.array([0.0]))["profile"][0]
            acc.append(abs(x[-1]-ana[-1])/ana[-1])
    fd.append(dict(D=D, dev_max_pct=float(np.mean(dm)*100), dev_bulk_pct=float(np.mean(db)*100)))
    print(f"finiteD D={D}: max={fd[-1]['dev_max_pct']:.2f}% bulk={fd[-1]['dev_bulk_pct']:.2f}%", flush=True)
out["finiteD"]=fd
R.joinpath("review_response.json").write_text(json.dumps(out, indent=1))
print("SAVED")
