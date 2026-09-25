"""Girko baseline v4: spectral deflation of the right edge.

v3 was accurate to ~1% for t <= 1 but failed at late t: the shift-normalized
Phi0(t) there is dominated by isolated right-edge eigenvalues (the stationary
state carries weight 1/D ~ 2e-4), which no smooth eta-resolved density can
represent to percent accuracy. Standard fix -- deflate:

  1. lam_top = K rightmost eigenvalues, exact, sparse Arnoldi ('LR').
  2. U_bulk(z) = U_KPM(z) - (1/2D) sum_k log(|z - lam_k|^2 + eta^2)
     (regularized log matched to the KPM eps^2 smoothing).
  3. Phi0 = (1/D) sum_k e^{2t(Re lam_k - x_ref)}  [exact head]
          + by-parts kernel integral of U_bulk    [smooth tail].

The bulk term is exponentially suppressed at late t, so v3-level (~few %)
accuracy on the smooth part suffices. Gate and stages as before.
Output: ../results/a3_girko_v4.json
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
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from run_a3_sparse import (mixed_jump_chain_sparse, phi_hutchinson,
                           s_late, TS, RESULTS, RNG)
from run_a3_girko_v3 import girko_u_grid, phi0_by_parts

ETA = 0.02
K_DEFLATE = 120

def rightmost(L, k=K_DEFLATE, sigma=0.02):
    """Rightmost eigenvalues via SHIFT-INVERT Arnoldi.

    Plain Arnoldi which='LR' fails spectacularly on this non-normal
    Lindbladian: Ritz values land deep in the pseudospectrum (k=20 already
    returns Re up to +2.0, k=120 up to +19, against a true max of 0) --
    the same classical non-normal pathology the paper's F4 documents for
    eigendecompositions. Shift-invert around sigma just right of the
    spectrum reproduces the dense truth to 4+ digits (validated at n=6).
    """
    w = eigs(L, k=k, sigma=sigma, which="LM", return_eigenvectors=False,
             maxiter=20000, tol=1e-10)
    return np.sort_complex(w)[::-1]

def phi0_deflated(L, x_ref, lam_top, eta=ETA, **grid_kw):
    D = L.shape[0]
    xs, ys, U = girko_u_grid(L, x_ref, eta=eta, **grid_kw)
    Z = xs[:, None] + 1j * ys[None, :]
    for lk in lam_top:
        U -= np.log(np.abs(Z - lk) ** 2 + eta ** 2) / (2 * D)
    bulk = phi0_by_parts(xs, ys, U, x_ref)
    head = np.array([np.sum(np.exp(2 * t * (lam_top.real - x_ref))) / D
                     for t in TS])
    return head + bulk, head, bulk

def main():
    out = {}
    print("=== v4 Stage 1: n=6 validation ===", flush=True)
    t0 = time.time()
    L6 = mixed_jump_chain_sparse(6, True, np.random.default_rng(20260717))
    D6 = L6.shape[0]
    lam6 = np.linalg.eigvals(L6.toarray())
    sh6 = lam6.real.max()
    from scipy.linalg import expm
    phi_ex, E = [], None
    step = expm(TS[0] * L6.toarray())
    for k in range(len(TS)):
        E = step if E is None else step @ E
        phi_ex.append(np.linalg.norm(E, "fro") ** 2 / D6 * np.exp(-2 * TS[k] * sh6))
    phi_ex = np.array(phi_ex)
    phi0_ex = np.array([np.mean(np.exp(2 * t * (lam6.real - sh6))) for t in TS])
    s_exact = s_late(phi_ex / phi0_ex - 1)

    lam_top = rightmost(L6)
    print(f"deflation head: {len(lam_top)} eigs, Re in "
          f"[{lam_top.real.min():.3f}, {lam_top.real.max():.3f}] "
          f"[{time.time()-t0:.0f}s]", flush=True)
    phi0_v4, head, bulk = phi0_deflated(L6, sh6, lam_top)
    err0 = float(np.max(np.abs(phi0_v4 / phi0_ex - 1)))
    phi_h, shH = phi_hutchinson(L6)
    x_est = (phi_h * np.exp(2 * TS * (shH - sh6))) / phi0_v4 - 1
    ds = abs(s_late(x_est) - s_exact)
    out["validation_n6"] = dict(K=len(lam_top), err_phi0_max=err0,
                                s_exact=s_exact, s_est=s_late(x_est), ds=ds,
                                phi0_ratio=[float(r) for r in phi0_v4 / phi0_ex],
                                head_frac=[float(h / p) for h, p in zip(head, phi0_v4)],
                                runtime=round(time.time() - t0))
    print(f"v4 n=6: max|dPhi0/Phi0|={err0:.3f} s*_exact={s_exact:.3f} "
          f"s*_est={s_late(x_est):.3f} (ds={ds:.3f}) [{time.time()-t0:.0f}s]",
          flush=True)
    (RESULTS / "a3_girko_v4.json").write_text(json.dumps(out, indent=1))
    if err0 > 0.05 or ds > 0.08:
        print("v4 VALIDATION FAILED - stopping before production", flush=True)
        raise SystemExit(1)

    print("=== v4 Stage 2: n=8 production ===", flush=True)
    prod = []
    for chaotic in (True, False):
        t0 = time.time()
        L = mixed_jump_chain_sparse(8, chaotic, np.random.default_rng(20260717))
        print(f"built n=8 chaotic={chaotic} D={L.shape[0]} [{time.time()-t0:.0f}s]",
              flush=True)
        phi, shH = phi_hutchinson(L)
        print(f"  Phi done [{time.time()-t0:.0f}s]", flush=True)
        lam_top = rightmost(L)
        print(f"  deflation head done [{time.time()-t0:.0f}s]", flush=True)
        phi0, head, bulk = phi0_deflated(L, shH, lam_top)
        x = phi / phi0 - 1
        prod.append(dict(chaotic=chaotic, D=int(L.shape[0]), K=len(lam_top),
                         chi0=float(np.max(np.abs(x))), s_late=s_late(x),
                         phi=[float(v) for v in phi],
                         phi0=[float(v) for v in phi0],
                         head_frac=[float(h / p) for h, p in zip(head, phi0)],
                         runtime=round(time.time() - t0)))
        print(f"RESULT n=8 chaotic={chaotic}: chi0={prod[-1]['chi0']:.3g} "
              f"s*={prod[-1]['s_late']:.3f} [{time.time()-t0:.0f}s]", flush=True)
        out["production_n8"] = prod
        (RESULTS / "a3_girko_v4.json").write_text(json.dumps(out, indent=1))
    print("SAVED results/a3_girko_v4.json")

if __name__ == "__main__":
    main()
