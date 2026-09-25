"""Model generators for P3: chaotic vs integrable non-Hermitian generators.

All models return a dense complex matrix L (the generator), rescaled so that
the centered spectral radius is ~1 (normalization convention alpha ~ 1,
t = O(1) is then the physically meaningful window).

Models
------
ginue            : Ginibre unitary ensemble (chaotic reference, max non-normal)
poisson_normal   : normal matrix with iid complex eigenvalues (integrable
                   reference, X == 0 identically)
random_liouvillian : vectorized Lindbladian with random jump operator
                   (dissipative chaotic; Sa-Ribeiro-Prosen type)
dephasing_liouvillian : integrable Liouvillian (commuting dephasing jumps)
nh_spin_chain    : H_XXZ + i*gamma*(boundary loss) effective Hamiltonian;
                   integrable (delta only) vs chaotic (staggered field + NNN)
hatano_nelson    : disordered HN chain (single particle) — strongly non-normal
                   yet spectrally Poisson: false-positive stress test (U2)
"""
from __future__ import annotations

import numpy as np


def _rescale(L: np.ndarray) -> np.ndarray:
    """Center trace and rescale to unit spectral radius (via eigenvalues)."""
    D = L.shape[0]
    L = L - (np.trace(L) / D) * np.eye(D)
    ev = np.linalg.eigvals(L)
    r = np.abs(ev).max()
    return L / r


def ginue(D: int, rng: np.random.Generator) -> np.ndarray:
    G = (rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))) / np.sqrt(2 * D)
    return _rescale(G)


def poisson_normal(D: int, rng: np.random.Generator) -> np.ndarray:
    """Normal matrix, iid eigenvalues uniform in unit disk (2D Poisson)."""
    r = np.sqrt(rng.uniform(0, 1, D))
    phi = rng.uniform(0, 2 * np.pi, D)
    lam = r * np.exp(1j * phi)
    # Haar unitary via QR of complex Ginibre
    Z = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
    Q, R = np.linalg.qr(Z)
    Q = Q * (np.diagonal(R) / np.abs(np.diagonal(R)))
    return _rescale((Q * lam) @ Q.conj().T)


def _vectorized_lindblad(H: np.ndarray, jumps: list[np.ndarray]) -> np.ndarray:
    """L = -i(H x I - I x H^T) + sum_k [K x K*  - 1/2 (K'K x I + I x (K'K)^T)]."""
    d = H.shape[0]
    I = np.eye(d)
    L = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for K in jumps:
        KdK = K.conj().T @ K
        L += np.kron(K, K.conj()) - 0.5 * (np.kron(KdK, I) + np.kron(I, KdK.T))
    return L


def random_liouvillian(d: int, rng: np.random.Generator, n_jumps: int = 1,
                       g: float = 1.0) -> np.ndarray:
    """Random Lindbladian: GUE Hamiltonian + Ginibre jump(s). Chaotic (Ginibre CSR)."""
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = (A + A.conj().T) / np.sqrt(8 * d)
    jumps = []
    for _ in range(n_jumps):
        K = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
        K /= np.sqrt(2 * d)
        jumps.append(np.sqrt(g) * K)
    return _rescale(_vectorized_lindblad(H, jumps))


def dephasing_liouvillian(d: int, rng: np.random.Generator, g: float = 1.0) -> np.ndarray:
    """Integrable Liouvillian: random diagonal H + diagonal dephasing jump.

    All terms commute in the eigenbasis of H -> normal generator, Poisson spectrum.
    """
    e = rng.standard_normal(d)
    H = np.diag(e)
    K = np.diag(rng.standard_normal(d)) * np.sqrt(g)
    return _rescale(_vectorized_lindblad(H, [K]))


def _spin_ops(n: int):
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    ops = {"x": [], "y": [], "z": []}
    for i in range(n):
        for key, s in (("x", sx), ("y", sy), ("z", sz)):
            m = 1
            for j in range(n):
                m = np.kron(m, s if j == i else np.eye(2))
            ops[key].append(m)
    return ops


def nh_spin_chain(n: int, chaotic: bool, rng: np.random.Generator,
                  delta: float = 0.5, gamma: float = 0.4) -> np.ndarray:
    """H_eff = H_XXZ + perturbations + i*gamma*(loss on both edge sites).

    integrable (chaotic=False): pure XXZ + uniform field — Bethe-ansatz
    integrable backbone; complex spectrum from boundary loss.
    chaotic (chaotic=True): + staggered field + next-nearest ZZ, breaking
    integrability (standard recipe).
    NOTE: total S_z is conserved in both cases; we resolve the symmetry by
    projecting onto the half-filling sector before returning.
    """
    ops = _spin_ops(n)
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n - 1):
        H += ops["x"][i] @ ops["x"][i + 1] + ops["y"][i] @ ops["y"][i + 1]
        H += delta * (ops["z"][i] @ ops["z"][i + 1])
    H += 0.3 * sum(ops["z"])
    if chaotic:
        for i in range(n):
            H += 0.35 * ((-1) ** i) * ops["z"][i]
        for i in range(n - 2):
            H += 0.6 * (ops["z"][i] @ ops["z"][i + 2])
    n_op_0 = ops["z"][0] + 0.5 * np.eye(2 ** n)
    n_op_L = ops["z"][n - 1] + 0.5 * np.eye(2 ** n)
    Heff = H - 1j * gamma * (n_op_0 + 0.7 * n_op_L)
    # project to half-filling sector of total Sz
    sz_tot = np.diagonal(sum(ops["z"])).real.round(1)
    sector = np.isclose(sz_tot, 0.0 if n % 2 == 0 else 0.5)
    P = np.eye(2 ** n)[:, sector]
    return _rescale(P.T @ Heff @ P)


def hatano_nelson(D: int, rng: np.random.Generator, g: float = 0.5,
                  w: float = 1.0) -> np.ndarray:
    """Disordered Hatano-Nelson chain (OBC): non-normal (skin effect) but
    localized/Poisson-like statistics. False-positive stress test for X."""
    L = np.zeros((D, D), dtype=complex)
    for i in range(D - 1):
        L[i, i + 1] = np.exp(-g)
        L[i + 1, i] = np.exp(g)
    L += np.diag(rng.uniform(-w, w, D))
    return _rescale(L)


MODELS = {
    "ginue": lambda D, rng: ginue(D, rng),
    "poisson_normal": lambda D, rng: poisson_normal(D, rng),
    "random_liouvillian": lambda d, rng: random_liouvillian(d, rng),
    "dephasing_liouvillian": lambda d, rng: dephasing_liouvillian(d, rng),
    "nh_chain_chaotic": lambda n, rng: nh_spin_chain(n, True, rng),
    "nh_chain_integrable": lambda n, rng: nh_spin_chain(n, False, rng),
    "hatano_nelson": lambda D, rng: hatano_nelson(D, rng),
}


# ---------------------------------------------------------------- physical models v2

def _majoranas(n_maj: int) -> list[np.ndarray]:
    """Jordan-Wigner Majoranas: chi_{2k}=(prod Z)X_k, chi_{2k+1}=(prod Z)Y_k."""
    nq = n_maj // 2
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    chis = []
    for k in range(nq):
        for P in (X, Y):
            m = 1
            for j in range(nq):
                if j < k:
                    m = np.kron(m, Z)
                elif j == k:
                    m = np.kron(m, P)
                else:
                    m = np.kron(m, np.eye(2))
            chis.append(m / np.sqrt(2))
    return chis


def nh_syk(n_maj: int, q: int, rng: np.random.Generator,
           kappa: float = 0.4) -> np.ndarray:
    """Non-Hermitian SYK_q: H = sum_{i<j<k<l} (J + i kappa J') chi_i...chi_l.

    q=4: dissipative chaotic (Ginibre-class CSR).
    q=2: quadratic -> integrable (2D Poisson CSR after unfolding).
    """
    import math
    from itertools import combinations
    chis = _majoranas(n_maj)
    dim = chis[0].shape[0]
    H = np.zeros((dim, dim), dtype=complex)
    sd = np.sqrt(math.factorial(q - 1) / (n_maj ** (q - 1)))
    if q == 2:
        for (i, j) in combinations(range(n_maj), 2):
            J = sd * (rng.standard_normal() + 1j * kappa * rng.standard_normal())
            H += J * 1j * (chis[i] @ chis[j])
    elif q == 4:
        pair = {}
        for (i, j) in combinations(range(n_maj), 2):
            pair[(i, j)] = chis[i] @ chis[j]
        for (i, j, k, l) in combinations(range(n_maj), 4):
            J = sd * (rng.standard_normal() + 1j * kappa * rng.standard_normal())
            H += -J * (pair[(i, j)] @ pair[(k, l)])
    else:
        raise ValueError("q in {2,4}")
    # project onto even fermion-parity sector (H conserves parity; CSR must be
    # computed within one symmetry sector)
    nq = n_maj // 2
    Zs = np.ones(dim)
    for b in range(dim):
        Zs[b] = (-1) ** bin(b).count("1")
    sector = Zs > 0
    P = np.eye(dim)[:, sector]
    return _rescale(P.T @ H @ P)


def xxz_liouvillian(n: int, chaotic: bool, rng: np.random.Generator,
                    delta: float = 0.5, gp: float = 0.6, gm: float = 0.3) -> np.ndarray:
    """Boundary-driven XXZ Lindbladian (vectorized), d=2^n, D=4^n.

    integrable backbone: XXZ; chaotic: + staggered field + NNN ZZ.
    Jumps: sqrt(gp) sig+_1, sqrt(gm) sig-_1, sqrt(gp) sig-_n, sqrt(gm) sig+_n.
    """
    ops = _spin_ops(n)
    d = 2 ** n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n - 1):
        H += ops["x"][i] @ ops["x"][i + 1] + ops["y"][i] @ ops["y"][i + 1]
        H += delta * (ops["z"][i] @ ops["z"][i + 1])
    if chaotic:
        for i in range(n):
            H += 0.5 * ((-1) ** i) * ops["z"][i]
        for i in range(n - 2):
            H += 0.8 * (ops["z"][i] @ ops["z"][i + 2])
    sp = lambda i: ops["x"][i] + 1j * ops["y"][i]
    sm = lambda i: ops["x"][i] - 1j * ops["y"][i]
    jumps = [np.sqrt(gp) * sp(0), np.sqrt(gm) * sm(0),
             np.sqrt(gp) * sm(n - 1), np.sqrt(gm) * sp(n - 1)]
    L = _vectorized_lindblad(H, jumps)
    # weak U(1): jumps carry definite Sz charge -> delta_m = m_bra - m_ket is
    # conserved; restrict to the delta_m = 0 superoperator sector (contains
    # the steady state; CSR must be computed within one sector)
    pop = np.array([bin(i).count("1") for i in range(d)])
    bi, bj = np.divmod(np.arange(d * d), d)
    sector = pop[bi] == pop[bj]
    P = np.eye(d * d)[:, sector]
    return _rescale(P.T @ L @ P)


MODELS.update({
    "nh_syk_q4": lambda N, rng: nh_syk(N, 4, rng),
    "nh_syk_q2": lambda N, rng: nh_syk(N, 2, rng),
    "xxz_liou_chaotic": lambda n, rng: xxz_liouvillian(n, True, rng),
    "xxz_liou_integrable": lambda n, rng: xxz_liouvillian(n, False, rng),
})


def xx_dephasing_liouvillian(n: int, chaotic: bool, rng: np.random.Generator,
                             gamma: float = 1.0, delta: float = 1.0,
                             h_stag: float = 0.5, w_dis: float = 0.2) -> np.ndarray:
    """Bulk-dephasing spin chain Lindbladian, built directly in the
    (m_bra, m_ket) = (n//2, n//2) doubled sector.

    integrable (chaotic=False): pure XX chain + uniform dephasing sqrt(gamma) n_a
    -- Bethe-ansatz solvable (Medvedyeva-Essler-Prosen, PRL 117, 137202).
    chaotic (chaotic=True): + Delta ZZ + staggered field + weak disordered
    z-fields (seed-dependent), same dephasing.

    All terms conserve m_bra and m_ket separately (H conserves Sz; jumps n_a
    are diagonal), so each (m, m') block is invariant; we take m = m' = n//2,
    which contains the steady-state diagonal.
    """
    m = n // 2
    states = [s for s in range(2 ** n) if bin(s).count("1") == m]
    idx = {s: a for a, s in enumerate(states)}
    d = len(states)
    # sector Hamiltonian
    H = np.zeros((d, d))
    for a, s in enumerate(states):
        diag = 0.0
        occ = [(s >> b) & 1 for b in range(n)]
        if chaotic:
            for b in range(n - 1):
                diag += delta * (occ[b] - 0.5) * (occ[b + 1] - 0.5)
            for b in range(n):
                diag += (h_stag * ((-1) ** b) + w_dis * rng.uniform(-1, 1)) * (occ[b] - 0.5)
        H[a, a] = diag
        for b in range(n - 1):          # XX hopping
            if occ[b] != occ[b + 1]:
                s2 = s ^ (1 << b) ^ (1 << (b + 1))
                H[idx[s2], a] += 0.5
    # Liouvillian on sector x sector
    I = np.eye(d)
    L = -1j * (np.kron(H, I) - np.kron(I, H.T))
    occm = np.array([[(s >> b) & 1 for b in range(n)] for s in states], dtype=float)
    # dephasing: diagonal -gamma/2 sum_a (n_a(i)-n_a(j))^2
    ni = occm[:, None, :]
    nj = occm[None, :, :]
    deph = -0.5 * gamma * ((ni - nj) ** 2).sum(axis=2)   # (d,d) over (i,j)
    L += np.diag(deph.reshape(-1))
    return _rescale(L)


MODELS.update({
    "xx_deph_integrable": lambda n, rng: xx_dephasing_liouvillian(n, False, rng),
    "xx_deph_chaotic": lambda n, rng: xx_dephasing_liouvillian(n, True, rng),
})
