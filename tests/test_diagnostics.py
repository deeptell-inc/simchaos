"""Sanity tests (pytest). Conventions: rtol=0.01-ish, fixed seeds."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import ginue, poisson_normal, hatano_nelson  # noqa: E402
from diagnostics import (csr_stats, eig_overlap_data, phi_phi0,  # noqa: E402
                         phi_frobenius_direct)


RNG = np.random.default_rng(20260717)


def test_csr_ginue_benchmark():
    """GinUE: <r> ~ 0.738, -<cos> ~ 0.241 (PRX 10, 021019)."""
    vals = [csr_stats(np.linalg.eigvals(ginue(1500, RNG)), bulk_fraction=0.9)
            for _ in range(3)]
    r = np.mean([v["r_mean"] for v in vals])
    c = np.mean([v["mcos_mean"] for v in vals])
    assert abs(r - 0.738) < 0.02
    assert abs(c - 0.241) < 0.04


def test_csr_poisson_benchmark():
    """Poisson: <r> = 2/3, <cos> = 0."""
    vals = [csr_stats(np.linalg.eigvals(poisson_normal(1500, RNG)),
                      bulk_fraction=0.9) for _ in range(3)]
    r = np.mean([v["r_mean"] for v in vals])
    c = np.mean([v["mcos_mean"] for v in vals])
    assert abs(r - 2 / 3) < 0.02
    assert abs(c) < 0.04


def test_normal_matrix_zero_excess():
    """Normal matrix: M = I -> X == 0 to numerical precision."""
    L = poisson_normal(300, RNG)
    data = eig_overlap_data(L)
    assert np.allclose(data["O_kk"], 1.0, atol=1e-6)
    phi, phi0 = phi_phi0(data, t=1.3, theta=0.7)
    assert abs(phi - phi0) / phi0 < 1e-8


def test_phi_matches_expm_moderate_kappa():
    """Overlap-matrix formula == direct expm, for moderately non-normal L."""
    L = ginue(150, RNG)
    data = eig_overlap_data(L)
    for t, th in [(0.5, 0.0), (1.0, 1.1), (2.0, -0.6)]:
        phi, _ = phi_phi0(data, t, th)
        ref = phi_frobenius_direct(L, t, th)
        assert phi == pytest.approx(ref, rel=1e-6)


def test_stepping_matches_expm_skin_effect():
    """Stable stepping evaluator == expm even for skin-effect (kappa_V huge),
    where the eigenvector-overlap formula catastrophically cancels."""
    from diagnostics import excess_profile
    L = hatano_nelson(120, RNG)
    ts = np.linspace(0.4, 2.0, 5)
    res = excess_profile(L, ts, np.array([0.0, 1.1]))
    lam = np.linalg.eigvals(L)
    for b, t in enumerate(ts):
        ref = phi_frobenius_direct(L, t, 1.1)
        mu = np.exp(1j * 1.1) * lam
        phi0 = np.sum(np.exp(2 * t * (mu.real - mu.real.max()))) / 120
        assert res["profile"][1, b] == pytest.approx((ref - phi0) / phi0, rel=1e-5)


def test_okk_lower_bound():
    L = ginue(300, RNG)
    data = eig_overlap_data(L)
    assert (data["O_kk"] >= 1 - 1e-8).all()
