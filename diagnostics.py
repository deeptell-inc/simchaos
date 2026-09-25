"""Diagnostics for P3: CSR (ground truth), Phi / Phi0 / overlap excess X.

Key identity (theory-foundation.md, checked in claims-ledger):
    Phi(t, theta) = (1/D) ||exp(t e^{i theta} L)||_F^2
                  = (1/D) sum_{j,k} M_{jk} exp(t (e^{i theta} lam_k
                                               + e^{-i theta} conj(lam_j)))
with M_{jk} = (V^dag V)_{jk} (W W^dag)_{kj},  W = V^{-1}  (columns of V are
right eigenvectors).  M_{kk} = O_kk >= 1 are the Chalker-Mehlig diagonal
overlaps; for a normal matrix M = I and Phi == Phi0.

    Phi0(t, theta) = (1/D) sum_k exp(2 t Re(e^{i theta} lam_k))
    X = Phi - Phi0  (overlap excess; the proposed chaos signal)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- CSR

def csr_stats(ev: np.ndarray, drop_real_axis: float = 0.0,
              bulk_fraction: float = 1.0) -> dict:
    """Complex spacing ratio statistics <r>, -<cos theta>.

    ev: complex eigenvalues (symmetry-resolved sector, zero modes removed).
    drop_real_axis: drop |Im| < this (for spectra symmetric about R, e.g.
    Liouvillians, keep Im > cut only — pass cut > 0 and we keep one half).
    bulk_fraction: keep this central fraction by |lambda - mean| to reduce
    edge effects.
    """
    z = np.asarray(ev, dtype=complex)
    if drop_real_axis > 0:
        z = z[z.imag > drop_real_axis]
    if bulk_fraction < 1.0:
        c = z.mean()
        r = np.abs(z - c)
        z = z[r <= np.quantile(r, bulk_fraction)]
    if z.size < 8:
        return {"r_mean": np.nan, "mcos_mean": np.nan, "n_used": int(z.size)}
    pts = np.column_stack([z.real, z.imag])
    tree = cKDTree(pts)
    dist, idx = tree.query(pts, k=3)  # self, NN, NNN
    lam_nn = z[idx[:, 1]]
    lam_nnn = z[idx[:, 2]]
    zeta = (lam_nn - z) / (lam_nnn - z)
    return {
        "r_mean": float(np.abs(zeta).mean()),
        "mcos_mean": float(-np.cos(np.angle(zeta)).mean()),
        "n_used": int(z.size),
    }


# ------------------------------------------------- Phi / Phi0 / X

def eig_overlap_data(L: np.ndarray) -> dict:
    """Eigendecomposition + overlap matrix M_{jk} = (V+V)_{jk}(WW+)_{kj}."""
    lam, V = np.linalg.eig(L)
    W = np.linalg.inv(V)
    M = (V.conj().T @ V) * (W @ W.conj().T).T   # elementwise: (V+V)_{jk}(WW+)_{kj}
    return {"lam": lam, "M": M, "O_kk": np.real(np.diagonal(M))}


def phi_phi0(data: dict, t: float, theta: float) -> tuple[float, float]:
    """Exact Phi and Phi0 at (t, theta) from precomputed eig/overlap data.

    Both are shift-normalized by exp(2 t max Re(e^{i theta} lam)) so that the
    largest spectral weight is O(1) (X/Phi0 is invariant under this shift).
    """
    lam, M = data["lam"], data["M"]
    mu = np.exp(1j * theta) * lam
    shift = mu.real.max()
    ek = np.exp(t * (mu - shift))            # e^{t(e^{i th}lam_k - shift)}
    # Phi = (1/D) sum_{jk} M_jk conj(e_j) e_k  = (1/D) e^dag M e   (e_k column)
    D = lam.size
    phi = float(np.real(ek.conj() @ (M @ ek)) / D)
    phi0 = float(np.sum(np.exp(2 * t * (mu.real - shift))) / D)
    return phi, phi0


def phi_frobenius_direct(L: np.ndarray, t: float, theta: float) -> float:
    """Independent check: Phi via scipy expm (shift-normalized identically)."""
    from scipy.linalg import expm
    lam = np.linalg.eigvals(L)
    shift = (np.exp(1j * theta) * lam).real.max()
    A = t * (np.exp(1j * theta) * L - shift * np.eye(L.shape[0]))
    E = expm(A)
    return float(np.linalg.norm(E, "fro") ** 2 / L.shape[0])


def excess_profile(L: np.ndarray, ts: np.ndarray, thetas: np.ndarray,
                   lam: np.ndarray | None = None) -> dict:
    """X/Phi0 on a (t, theta) grid; scalar discriminant chi = max over grid.

    Numerically stable evaluation: Phi is computed by stepping
    E(t+dt) = E(dt) @ E(t) with one expm per theta (Schur-free, BLAS-bound).
    The eigenvector-overlap formula (phi_phi0) is exponentially ill-
    conditioned for strongly non-normal L (e.g. skin effect, kappa_V huge)
    and must NOT be used for computation there — interpretation only.

    Requires ts to be an increasing arithmetic grid t_k = k*dt, k>=1.
    Phi0 uses eigenvalues only (stable). Both shift-normalized by
    exp(2 t max Re(e^{i th} lam)); the ratio X/Phi0 is shift-invariant.
    """
    from scipy.linalg import expm
    D = L.shape[0]
    if lam is None:
        lam = np.linalg.eigvals(L)
    dt = ts[1] - ts[0] if len(ts) > 1 else ts[0]
    assert np.allclose(np.diff(ts), dt) and np.isclose(ts[0], dt), \
        "ts must be dt*[1..n]"
    prof = np.empty((len(thetas), len(ts)))
    for a, th in enumerate(thetas):
        mu = np.exp(1j * th) * lam
        shift = mu.real.max()
        Estep = expm(dt * (np.exp(1j * th) * L - shift * np.eye(D)))
        E = np.eye(D, dtype=complex)
        for b, t in enumerate(ts):
            E = Estep @ E
            phi = float(np.linalg.norm(E, "fro") ** 2 / D)
            phi0 = float(np.sum(np.exp(2 * t * (mu.real - shift))) / D)
            prof[a, b] = (phi - phi0) / phi0
    return {
        "profile": prof,
        "ts": ts,
        "thetas": thetas,
        "chi": float(prof.max()),          # scalar discriminant
        "chi_theta0": float(prof[np.isclose(thetas, 0).argmax()].max()),
    }
