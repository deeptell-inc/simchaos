# simchaos — reproduction package

Companion code for *Tomography-free quantum diagnostics of dissipative quantum chaos*
(H. Wakaura and T. Tanimae, QIRI, 2026; submitted to Quantum Science and Technology).

## Install
    python -m venv venv && source venv/bin/activate
    pip install -e ".[dev]"        # numpy, scipy, matplotlib, pytest

Python >= 3.10. Tested with Python 3.13, numpy 2.5, scipy 1.18, matplotlib 3.11.

## Quick check (~1 min)
    python -m pytest tests/ -v          # CSR benchmarks, Lemma-1 identity, propagation stability
    python scripts/run_dqc1_check.py    # Choi-purity / DQC1 identities (Appendix D)
    python scripts/make_figures.py && python scripts/make_fig1.py && python scripts/make_fig4.py

Figure scripts read from `results/` if present, otherwise from the frozen `refdata/`,
so the paper figures can be regenerated without rerunning the simulations.
Scripts can be launched from any working directory; outputs go to `results/`
(git-ignored) next to `refdata/`.

## Full reproduction
| script | paper content | approx. time |
|---|---|---|
| `run_p3.py [--quick]` | main classification, 5 seeds, Table 1 | ~6 min |
| `run_s6.py` | D-scaling / anisotropy scans | ~8 min |
| `run_shotnoise.py` | shot-noise Monte Carlo | ~2 min |
| `run_theta0_eval.py` | pre-registered θ=0 evaluation, 45 held-out instances | ~10 min |
| `run_review_response.py` | precision numbers quoted in appendices | ~5 min |
| `run_circuit_demo.py` | circuit-level end-to-end estimate | ~5 min |
| `run_a2a_planar.py` | SYK planar reduction vs elliptic curve | ~10 min |
| `run_task5_n8.py`, `run_a3_n7.py`, `run_a3_*.py` | structured Liouvillians (dilution, mixed-charge jumps, Girko scans) | 10–60 min |
| `run_barrier_matching.py`, `run_sd_closed_form.py` | barrier constants, elliptic closed form | few min |

Seeds: 20260717+i (calibration), 20270101+i (holdout). Statistical benchmarks use rtol ≈ 1e-2;
exact identities are checked to ≤ 1e-6. Frozen outputs used in the paper live in `refdata/`.

## Contents
- `models.py`       GinUE, 2D-Poisson normal, random/dephasing Lindbladians, nH-SYK (q=2,4),
                    Hatano–Nelson, boundary-driven and bulk-dephasing spin-chain Liouvillians
- `diagnostics.py`  CSR statistics, stable propagation evaluator for Φ(t;θ), overlap matrices,
                    excess profiles and the (χ, s*, A) decision rule
- `scripts/`        paper drivers · `tests/` pytest · `refdata/` frozen JSON results

## License
MIT (see LICENSE).
