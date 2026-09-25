"""Verify the DQC1 generator-access reduction identities (Appendix D).

1. (1/D)Tr[LL+] = 3/2 + |Tr(U+V)/d|^2/2 - (2/D)Re Tr Lambda  for the
   two-unitary-jump Lindbladian L = sum_i K_i . K_i+ - id, K_1=U/sqrt2, K_2=V/sqrt2.
2. The t^2 coefficient of Phi(t;theta) at fixed theta is
   (1/D)(Tr[LL+] + Re[e^{2i theta} Tr L^2]); averaging the coefficients at
   theta=0 and theta=pi/2 cancels the L^2 term and isolates Tr[LL+]/D.

Added 2026-08-31 after the Codex adversarial panel flagged the Re Tr L^2
contamination; the theta-average extraction closes the reduction.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import expm

d = 8
D = d * d
U = unitary_group.rvs(d, random_state=1)
V = unitary_group.rvs(d, random_state=2)
Lam = 0.5 * (np.kron(U, U.conj()) + np.kron(V, V.conj()))
L = Lam - np.eye(D)

def phi(t, th):
    a = t * np.exp(1j * th)
    return np.trace(expm(a * L) @ expm(np.conj(a) * L.conj().T)).real / D

t = 1e-3
def c2(th):
    return (phi(t, th) - 2 * phi(0, th) + phi(-t, th)) / (2 * t * t)

avg = 0.5 * (c2(0.0) + c2(np.pi / 2))
target = np.trace(L @ L.conj().T).real / D
ident = 1.5 + 0.5 * abs(np.trace(U.conj().T @ V) / d) ** 2 - (2 / D) * np.real(np.trace(Lam))

print(f"theta-avg t^2 coeff  = {avg:.10f}")
print(f"Tr[LL+]/D            = {target:.10f}")
print(f"closed-form identity = {ident:.10f}")
print(f"theta=0 alone        = {c2(0.0):.10f}  (contains Re Tr L^2/D = {np.real(np.trace(L @ L))/D:.6f})")
assert abs(avg - target) < 1e-5, "theta-average extraction failed"
assert abs(target - ident) < 1e-10, "closed-form identity failed"
print("PASS: both identities verified")
