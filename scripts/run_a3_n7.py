import numpy as np, time, json
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
from models import _spin_ops, _vectorized_lindblad, _rescale
from diagnostics import csr_stats, excess_profile
def mixed_jump_chain(n, chaotic, rng, gamma=0.5, c=0.5, delta=1.0):
    ops = _spin_ops(n); d = 2**n
    H = np.zeros((d,d), dtype=complex)
    for i in range(n-1):
        H += ops["x"][i]@ops["x"][i+1] + ops["y"][i]@ops["y"][i+1]
    if chaotic:
        for i in range(n-1): H += delta*(ops["z"][i]@ops["z"][i+1])
        for i in range(n): H += (0.5*((-1)**i)+0.1*rng.uniform(-1,1))*ops["z"][i]
        for i in range(n-2): H += 0.6*(ops["z"][i]@ops["z"][i+2])
    sp = lambda i: ops["x"][i]+1j*ops["y"][i]
    sm = lambda i: ops["x"][i]-1j*ops["y"][i]
    jumps = [np.sqrt(gamma)*(sm(i)+c*sp(i)) for i in range(n)]
    return _rescale(_vectorized_lindblad(H, jumps))
TS = 0.25*np.arange(1,13); TAIL = slice(-5,None)
out=[]
for chaotic in (True, False):
    t0=time.time()
    L = mixed_jump_chain(7, chaotic, np.random.default_rng(20260717))
    print(f"built {chaotic} D={L.shape[0]} ({time.time()-t0:.0f}s)", flush=True)
    lam = np.linalg.eigvals(L)
    print(f"eig done ({time.time()-t0:.0f}s)", flush=True)
    csr = csr_stats(lam, drop_real_axis=1e-9, bulk_fraction=0.9)
    prof = excess_profile(L, TS, np.array([0.0]), lam=lam)
    x0 = prof["profile"][0]
    s = float(np.polyfit(np.log(TS[TAIL]), np.log(np.abs(x0[TAIL])+1e-300),1)[0])
    out.append(dict(chaotic=chaotic, D=int(L.shape[0]), r=csr["r_mean"], mcos=csr["mcos_mean"],
                    chi=prof["chi"], s_late=s, runtime=round(time.time()-t0)))
    print(f"RESULT chaotic={chaotic}: r={csr['r_mean']:.3f} -cos={csr['mcos_mean']:.3f} "
          f"chi={prof['chi']:.3g} s*={s:.3f}", flush=True)
    (RESULTS / "a3_n7.json").write_text(json.dumps(out, indent=1))
print("SAVED")
