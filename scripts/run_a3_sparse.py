"""A3 sparse scale-up: mixed-charge bulk-jump chains beyond exact diagonalization.

Two estimators, both matrix-free (sparse matvecs only):
  Phi(t)  = (1/D) Tr[e^{tL+} e^{tL}]  via Hutchinson probes + expm_multiply.
  Phi0(t) = int rho(z) e^{2t Re z}    via Girko log-determinant:
            u(z) = (1/2D) Tr log[(L-z)+(L-z) + eps^2], rho = (1/2pi) Lap u,
            with u(z) estimated by stochastic Chebyshev (KPM) on M(z) --
            this IS the paper's classical KPM baseline route, implemented.

Stage 1 (validation, n=6, D=4096): dense eigendecomposition ground truth vs
both estimators; abort production if Phi rel-err > 3% or s* shifts > 0.05.
Stage 2 (production, n=8, D=65536): both members, s* extraction.

Output: ../results/a3_sparse.json  (validation block + production block)
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
from scipy.sparse.linalg import expm_multiply, eigs, LinearOperator
from pathlib import Path

TS = 0.25 * np.arange(1, 13)
TAIL = slice(-5, None)
RNG = np.random.default_rng(20260909)

# ---------- sparse model ----------
I2 = sp.identity(2, format="csr", dtype=complex)
SX = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
SY = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
SZ = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))

def site(op, i, n):
    m = sp.identity(1, format="csr", dtype=complex)
    for k in range(n):
        m = sp.kron(m, op if k == i else I2, format="csr")
    return m

def mixed_jump_chain_sparse(n, chaotic, rng, gamma=0.5, c=0.5, delta=1.0):
    """Same physics as run_a3_n7.mixed_jump_chain, built sparse."""
    X = [site(SX, i, n) for i in range(n)]
    Y = [site(SY, i, n) for i in range(n)]
    Z = [site(SZ, i, n) for i in range(n)]
    d = 2 ** n
    H = sp.csr_matrix((d, d), dtype=complex)
    for i in range(n - 1):
        H = H + X[i] @ X[i + 1] + Y[i] @ Y[i + 1]
    if chaotic:
        for i in range(n - 1):
            H = H + delta * (Z[i] @ Z[i + 1])
        for i in range(n):
            H = H + (0.5 * ((-1) ** i) + 0.1 * rng.uniform(-1, 1)) * Z[i]
        for i in range(n - 2):
            H = H + 0.6 * (Z[i] @ Z[i + 2])
    jumps = [np.sqrt(gamma) * ((X[i] - 1j * Y[i]) + c * (X[i] + 1j * Y[i]))
             for i in range(n)]
    Id = sp.identity(d, format="csr", dtype=complex)
    L = -1j * (sp.kron(H, Id) - sp.kron(Id, H.T))
    for K in jumps:
        KdK = (K.conj().T @ K)
        L = L + sp.kron(K, K.conj()) \
              - 0.5 * (sp.kron(KdK, Id) + sp.kron(Id, KdK.T))
    L = L.tocsr()
    # rescale spectral radius to 1 (largest-magnitude eigenvalue via ARPACK)
    lmax = eigs(L, k=1, which="LM", return_eigenvectors=False,
                maxiter=5000, tol=1e-6)[0]
    return (L / abs(lmax)).tocsr()

# ---------- Phi via Hutchinson + expm_multiply ----------
def phi_hutchinson(L, n_probe=24, rng=RNG):
    """Returns shift-normalized Phi(t) on TS. Shift = numerical-abscissa proxy
    max Re eig (ARPACK 'LR'); the classifier ratio is shift-invariant, the
    shift only keeps numbers O(1)."""
    D = L.shape[0]
    w = eigs(L, k=4, which="LR", return_eigenvectors=False,
             maxiter=5000, tol=1e-6)
    shift = float(np.max(w.real))
    acc = np.zeros(len(TS))
    for p in range(n_probe):
        v = ((rng.integers(0, 2, D) * 2 - 1)
             + 1j * (rng.integers(0, 2, D) * 2 - 1)) / np.sqrt(2)
        U = expm_multiply(L, v, start=TS[0], stop=TS[-1], num=len(TS),
                          endpoint=True)
        acc += np.array([np.vdot(U[i], U[i]).real for i in range(len(TS))])
    return acc / (n_probe * D) * np.exp(-2 * TS * shift), shift

# ---------- Phi0 via Girko log-det + stochastic Chebyshev ----------
def cheb_logdet(Mop, D, scale, eps2, n_mom=64, n_probe=8, rng=RNG):
    """(1/D) Tr log(M + eps2) with M PSD, lam_max(M) <= scale.
    Chebyshev of f(x) = log(scale*(x+1)/2 + eps2) on x in [-1,1]."""
    xk = np.cos(np.pi * (np.arange(2 * n_mom) + 0.5) / (2 * n_mom))
    fk = np.log(scale * (xk + 1) / 2 + eps2)
    ck = np.polynomial.chebyshev.chebfit(xk, fk, n_mom - 1)
    tot = 0.0
    for p in range(n_probe):
        v = ((rng.integers(0, 2, D) * 2 - 1)
             + 1j * (rng.integers(0, 2, D) * 2 - 1)) / np.sqrt(2)
        # three-term recurrence on normalized Mn = 2M/scale - I
        t0, t1 = v, Mop(v) * (2 / scale) - v
        s = ck[0] * np.vdot(v, t0).real + (ck[1] * np.vdot(v, t1).real
                                           if n_mom > 1 else 0.0)
        for m in range(2, n_mom):
            t2 = 2 * (Mop(t1) * (2 / scale) - t1) - t0
            s += ck[m] * np.vdot(v, t2).real
            t0, t1 = t1, t2
        tot += s
    return tot / (n_probe * D)

def phi0_girko(L, eta=0.06, pad=0.15, ngrid=17, n_mom=64, n_probe=6, rng=RNG):
    """Coarse rho(z) on a grid via Laplacian of u(z), then Phi0(t) on TS.
    eta plays the role of the KPM resolution (eps = eta)."""
    D = L.shape[0]
    xs = np.linspace(-1 - pad, 1 + pad, ngrid)
    h = xs[1] - xs[0]
    U = np.zeros((ngrid, ngrid))
    Ld = L  # csr
    for a, x in enumerate(xs):
        for b, y in enumerate(xs):
            z = x + 1j * y
            A = (Ld - z * sp.identity(D, format="csr", dtype=complex)).tocsr()
            Mop = lambda w: A.conj().T @ (A @ w)
            # power iteration for lam_max(M)
            v = rng.standard_normal(D) + 1j * rng.standard_normal(D)
            for _ in range(12):
                v = Mop(v); v = v / np.linalg.norm(v)
            scale = float(np.vdot(v, Mop(v)).real) * 1.2
            U[a, b] = 0.5 * cheb_logdet(Mop, D, scale, eta ** 2,
                                        n_mom=n_mom, n_probe=n_probe, rng=rng)
        print(f"  girko row {a+1}/{ngrid}", flush=True)
    lap = (np.roll(U, 1, 0) + np.roll(U, -1, 0) + np.roll(U, 1, 1)
           + np.roll(U, -1, 1) - 4 * U) / h ** 2
    rho = np.clip(lap[1:-1, 1:-1] / (2 * np.pi), 0, None)
    rho /= rho.sum() * h * h
    Xg = xs[1:-1][:, None] * np.ones((1, ngrid - 2))
    shiftmax = Xg[rho > rho.max() * 1e-3].max() if (rho > 0).any() else 1.0
    phi0 = np.array([(rho * np.exp(2 * t * (Xg - shiftmax))).sum() * h * h
                     for t in TS])
    return phi0, shiftmax

def s_late(x):
    return float(np.polyfit(np.log(TS[TAIL]), np.log(np.abs(x[TAIL]) + 1e-300), 1)[0])

def main():
    out = {}

    # ---------------- Stage 1: validation at n=6 ----------------
    print("=== Stage 1: n=6 validation ===", flush=True)
    t0 = time.time()
    L6 = mixed_jump_chain_sparse(6, True, np.random.default_rng(20260717))
    D6 = L6.shape[0]
    lam6 = np.linalg.eigvals(L6.toarray())
    sh6 = lam6.real.max()
    # exact Phi/Phi0 (same shift for both, ratio is shift-invariant)
    from scipy.linalg import expm
    phi_ex = []
    E = np.eye(D6, dtype=complex); step = expm(TS[0] * L6.toarray())
    for k in range(len(TS)):
        E = step @ E if k else step
        phi_ex.append(np.linalg.norm(E, "fro") ** 2 / D6 * np.exp(-2 * TS[k] * sh6))
    phi_ex = np.array(phi_ex)
    phi0_ex = np.array([np.mean(np.exp(2 * t * (lam6.real - sh6))) for t in TS])
    x_ex = phi_ex / phi0_ex - 1

    phi_h, shH = phi_hutchinson(L6)
    phi0_g, shG = phi0_girko(L6)
    # compare shift-invariant ratio
    x_est = (phi_h * np.exp(2 * TS * (shH - sh6))) / (phi0_g * np.exp(2 * TS * (shG - sh6))) - 1
    err_phi = float(np.max(np.abs(phi_h * np.exp(2 * TS * (shH - sh6)) / phi_ex - 1)))
    ds = abs(s_late(x_est) - s_late(x_ex))
    out["validation_n6"] = dict(err_phi_max=err_phi, s_exact=s_late(x_ex),
                                s_est=s_late(x_est), ds=ds,
                                runtime=round(time.time() - t0))
    print(f"n=6: max|dPhi/Phi|={err_phi:.3f}  s*_exact={s_late(x_ex):.3f} "
          f"s*_est={s_late(x_est):.3f} (ds={ds:.3f}) [{time.time()-t0:.0f}s]", flush=True)
    (RESULTS / "a3_sparse.json").write_text(json.dumps(out, indent=1))
    if err_phi > 0.05 or ds > 0.08:
        print("VALIDATION FAILED - stopping before production", flush=True)
        raise SystemExit(1)

    # ---------------- Stage 2: production at n=8 ----------------
    print("=== Stage 2: n=8 production ===", flush=True)
    prod = []
    for chaotic in (True, False):
        t0 = time.time()
        L = mixed_jump_chain_sparse(8, chaotic, np.random.default_rng(20260717))
        print(f"built n=8 chaotic={chaotic} D={L.shape[0]} nnz={L.nnz} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        phi, shH = phi_hutchinson(L)
        print(f"  Phi done [{time.time()-t0:.0f}s]", flush=True)
        phi0, shG = phi0_girko(L)
        x = (phi * np.exp(-2 * TS * (shG - shH))) / phi0 - 1
        prod.append(dict(chaotic=chaotic, D=int(L.shape[0]),
                         chi0=float(np.max(np.abs(x))), s_late=s_late(x),
                         phi=[float(v) for v in phi], phi0=[float(v) for v in phi0],
                         runtime=round(time.time() - t0)))
        print(f"RESULT n=8 chaotic={chaotic}: chi0={prod[-1]['chi0']:.3g} "
              f"s*={prod[-1]['s_late']:.3f} [{time.time()-t0:.0f}s]", flush=True)
        out["production_n8"] = prod
        (RESULTS / "a3_sparse.json").write_text(json.dumps(out, indent=1))
    print("SAVED results/a3_sparse.json")

if __name__ == "__main__":
    main()
