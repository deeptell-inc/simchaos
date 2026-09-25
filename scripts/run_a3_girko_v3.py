"""Girko baseline v3: by-parts kernel integration on an ASYMMETRIC box.

v2 failed by catastrophic cancellation: its box extended to x = +1.35 while
the Lindbladian spectrum obeys Re(lambda) <= x_ref ~ 0, so the kernel
K = e^{2t(x-x_ref)} reached e^{8.1} in a harmonic region whose volume/boundary
contributions must cancel exactly -- any stochastic noise there is amplified
by that factor (observed ratio ~1.5e4 at t=3).

v3 clips the box at x_max = x_ref + margin (kernel bounded by e^{2 t margin}),
keeps the full y extent, and uses second-order one-sided normal derivatives.
Everything else (Chebyshev log-det u(z), Hutchinson Phi) as before.

Stage 1: n=6 gate (err_phi0 <= 5%, ds <= 0.08). Stage 2: n=8 production.
Output: ../results/a3_girko_v3.json
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
from run_a3_sparse import (mixed_jump_chain_sparse, phi_hutchinson,
                           cheb_logdet, s_late, TS, RESULTS, RNG)

def girko_u_grid(L, x_ref, eta=0.02, pad=0.25, margin=0.10,
                 nx=25, ny=41, n_mom=128, n_probe=12, rng=RNG):
    D = L.shape[0]
    xs = np.linspace(-1 - pad, x_ref + margin, nx)
    ys = np.linspace(-1 - pad, 1 + pad, ny)
    U = np.zeros((nx, ny))
    for a, x in enumerate(xs):
        for b, y in enumerate(ys):
            z = x + 1j * y
            A = (L - z * sp.identity(D, format="csr", dtype=complex)).tocsr()
            Adag = A.conj().T.tocsr()
            Mop = lambda w: Adag @ (A @ w)
            v = rng.standard_normal(D) + 1j * rng.standard_normal(D)
            for _ in range(12):
                v = Mop(v); v = v / np.linalg.norm(v)
            scale = float(np.vdot(v, Mop(v)).real) * 1.2
            U[a, b] = 0.5 * cheb_logdet(Mop, D, scale, eta ** 2,
                                        n_mom=n_mom, n_probe=n_probe, rng=rng)
        print(f"  u-grid col {a+1}/{nx}", flush=True)
    return xs, ys, U

def dn2(U0, U1, U2, h):
    """2nd-order one-sided outward normal derivative from three inward rows."""
    return (3 * U0 - 4 * U1 + U2) / (2 * h)

def phi0_by_parts(xs, ys, U, x_ref):
    hx = xs[1] - xs[0]; hy = ys[1] - ys[0]
    Xg = xs[:, None] * np.ones((1, len(ys)))
    out = []
    for t in TS:
        K = np.exp(2 * t * (Xg - x_ref))
        vol = (U * 4 * t * t * K).sum() * hx * hy
        Kxmin = np.exp(2 * t * (xs[0] - x_ref))
        Kxmax = np.exp(2 * t * (xs[-1] - x_ref))
        Krow = np.exp(2 * t * (xs - x_ref))
        b_xmin = (Kxmin * dn2(U[0, :], U[1, :], U[2, :], hx)
                  - U[0, :] * (-2 * t * Kxmin)).sum() * hy
        b_xmax = (Kxmax * dn2(U[-1, :], U[-2, :], U[-3, :], hx)
                  - U[-1, :] * (+2 * t * Kxmax)).sum() * hy
        b_ymin = (Krow * dn2(U[:, 0], U[:, 1], U[:, 2], hy)).sum() * hx
        b_ymax = (Krow * dn2(U[:, -1], U[:, -2], U[:, -3], hy)).sum() * hx
        out.append((vol + b_xmin + b_xmax + b_ymin + b_ymax) / (2 * np.pi))
    return np.array(out)

def main():
    out = {}
    # ---- Stage 1: n=6 validation ----
    print("=== v3 Stage 1: n=6 validation ===", flush=True)
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

    xs, ys, U = girko_u_grid(L6, sh6)
    phi0_v3 = phi0_by_parts(xs, ys, U, sh6)
    err0 = float(np.max(np.abs(phi0_v3 / phi0_ex - 1)))
    phi_h, shH = phi_hutchinson(L6)
    x_est = (phi_h * np.exp(2 * TS * (shH - sh6))) / phi0_v3 - 1
    ds = abs(s_late(x_est) - s_exact)
    out["validation_n6"] = dict(err_phi0_max=err0, s_exact=s_exact,
                                s_est=s_late(x_est), ds=ds,
                                phi0_ratio=[float(r) for r in phi0_v3 / phi0_ex],
                                runtime=round(time.time() - t0))
    print(f"v3 n=6: max|dPhi0/Phi0|={err0:.3f} s*_exact={s_exact:.3f} "
          f"s*_est={s_late(x_est):.3f} (ds={ds:.3f}) [{time.time()-t0:.0f}s]",
          flush=True)
    (RESULTS / "a3_girko_v3.json").write_text(json.dumps(out, indent=1))
    if err0 > 0.05 or ds > 0.08:
        print("v3 VALIDATION FAILED - stopping before production", flush=True)
        raise SystemExit(1)

    # ---- Stage 2: n=8 production ----
    print("=== v3 Stage 2: n=8 production ===", flush=True)
    prod = []
    for chaotic in (True, False):
        t0 = time.time()
        L = mixed_jump_chain_sparse(8, chaotic, np.random.default_rng(20260717))
        print(f"built n=8 chaotic={chaotic} D={L.shape[0]} [{time.time()-t0:.0f}s]",
              flush=True)
        phi, shH = phi_hutchinson(L)
        print(f"  Phi done [{time.time()-t0:.0f}s]", flush=True)
        xs, ys, U = girko_u_grid(L, shH)
        phi0 = phi0_by_parts(xs, ys, U, shH)
        x = phi / phi0 - 1
        prod.append(dict(chaotic=chaotic, D=int(L.shape[0]),
                         chi0=float(np.max(np.abs(x))), s_late=s_late(x),
                         phi=[float(v) for v in phi],
                         phi0=[float(v) for v in phi0],
                         runtime=round(time.time() - t0)))
        print(f"RESULT n=8 chaotic={chaotic}: chi0={prod[-1]['chi0']:.3g} "
              f"s*={prod[-1]['s_late']:.3f} [{time.time()-t0:.0f}s]", flush=True)
        out["production_n8"] = prod
        (RESULTS / "a3_girko_v3.json").write_text(json.dumps(out, indent=1))
    print("SAVED results/a3_girko_v3.json")

if __name__ == "__main__":
    main()
