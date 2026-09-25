"""S11: Fig 1 concept figure. Message: 'the post-selection acceptance rate IS the signal'.
(a) complex spectrum + theta rotation; (b) protocol schematic; (c) X/Phi0 shape cartoon.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
RESULTS = _ROOT / 'results'; RESULTS.mkdir(exist_ok=True)
REFDATA = _ROOT / 'refdata'
_in = lambda n: RESULTS / n if (RESULTS / n).exists() else REFDATA / n  # frozen fallback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle, FancyArrowPatch
from pathlib import Path

R = RESULTS
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
fig = plt.figure(figsize=(7.0, 2.6))

# (a) spectrum + rotation
ax = fig.add_subplot(131)
rng = np.random.default_rng(7)
G = (rng.standard_normal((300, 300)) + 1j * rng.standard_normal((300, 300))) / np.sqrt(600)
ev = np.linalg.eigvals(G)
ax.scatter(ev.real, ev.imag, s=2.5, c="steelblue", alpha=0.6, lw=0)
th = np.pi / 5
ax.annotate("", xy=(1.25 * np.cos(th), 1.25 * np.sin(th)), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.6))
ax.plot([0, 1.25], [0, 0], color="gray", lw=1, ls=":")
ax.text(1.0, 0.28, r"$e^{i\theta}$", color="crimson", fontsize=10)
ax.text(-1.28, 1.05, r"spec$(L)$", fontsize=8, color="steelblue")
ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
ax.set_xlabel(r"Re$\,\lambda$"); ax.set_ylabel(r"Im$\,\lambda$")
ax.set_aspect("equal")
ax.set_title("(a) rotated generator", fontsize=9)

# (b) protocol schematic
ax = fig.add_subplot(132); ax.axis("off")
def box(x, y, w, h, txt, fc="#eef2f7"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="k", lw=0.8))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.5)
box(0.00, 0.60, 0.26, 0.26, "max. mixed\n$I/D$")
box(0.34, 0.55, 0.32, 0.38, "BE$\\left[e^{t e^{i\\theta}L}\\right]$\nQEVT/LCHS", fc="#fdeaea")
box(0.34, 0.06, 0.32, 0.28, "SWAP test,\n2 Choi copies\n($\\theta=0$, NISQ)", fc="#eafbea")
ax.add_patch(FancyArrowPatch((0.26, 0.73), (0.34, 0.73), arrowstyle="->", lw=1.2))
ax.add_patch(FancyArrowPatch((0.66, 0.73), (0.76, 0.73), arrowstyle="->", lw=1.2))
ax.text(0.78, 0.80, r"$P_{\rm acc}$", ha="left", va="center", fontsize=9, color="crimson")
ax.text(0.78, 0.66, r"$=\Phi(t;\theta)$", ha="left", va="center", fontsize=9, color="crimson")
ax.text(0.78, 0.46, "acceptance\nrate = the\nsignal", ha="left", va="center", fontsize=6.6,
        style="italic")
ax.add_patch(FancyArrowPatch((0.50, 0.34), (0.50, 0.55), arrowstyle="<->", lw=0.9, ls=":"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("(b) protocol", fontsize=9)

# (c) shape cartoon
ax = fig.add_subplot(133)
t = np.linspace(0.05, 3, 200)
ax.plot(t, 0.08 * t**2, "-", color="steelblue", lw=1.6, label="integrable non-normal: $t^2$")
ax.plot(t, 0.08 * t**2 / (1 + 0.55 * t), "-", color="crimson", lw=1.6,
        label="chaotic: bends to ramp")
ax.plot(t, 0 * t, "-", color="navy", lw=1.6, label="normal integrable: 0")
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$X/\Phi_0$")
ax.legend(fontsize=6.0, loc="upper left")
ax.set_title(r"(c) overlap excess $X=\Phi-\Phi_0$", fontsize=9)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(R / f"fig1.{ext}", dpi=200)
print("fig1 saved")
