"""
P_fig44_two_stage_model.py
===========================
Fig 4.4 — Two-stage stochastic model (F5 / Gate G3).

Panel A: Stage 1 — observed coincidence rate vs. Poisson null + logistic fit.
Panel B: Stage 2 — alpha(d) = alpha0*exp(-d/L) curve with pairwise diagnostics.

Reads:  results/gates/event_pairing_summary.parquet
        results/gates/f5_stage2_params.parquet
        results/gates/f5_stage2_pairwise_diagnostic.parquet
Writes: paper/figures/fig44_two_stage_model.pdf
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit   # logistic sigmoid

sys.path.insert(0, str(Path(__file__).parent))
from src import config as cfg
from src.paper_figures import save_publication_figure

ROOT   = Path(__file__).parent
EP_PQ  = cfg.DIRS["gates"] / "event_pairing_summary.parquet"
P2_PQ  = cfg.DIRS["gates"] / "f5_stage2_params.parquet"
DG_PQ  = cfg.DIRS["gates"] / "f5_stage2_pairwise_diagnostic.parquet"
OUT    = ROOT / "paper" / "figures" / "fig44_two_stage_model.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
})

C_OBS   = "#c0392b"
C_NULL  = "#7f8c8d"
C_FIT   = "#2c3e50"
C_DATA  = "#95a5a6"
C_CURVE = "#27ae60"
C_MED   = "#1a5276"

# ── Load ─────────────────────────────────────────────────────────────────────
ep   = pd.read_parquet(EP_PQ)
p2   = pd.read_parquet(P2_PQ).iloc[0]
diag = pd.read_parquet(DG_PQ)

ep["dist_km"] = ep["dist_ij_m"] / 1000.0

# Stage 1 model parameters (from FINDINGS.md / PRE_PAPER.md)
GAMMA0     =  3.68
GAMMA_DIST = -0.6511
GAMMA_X    = -0.1393  # not used for distance-only curve
MEDIAN_Xi  =  0.0     # Laplace-scale conditioning magnitude (median≈0 for plotting)

# Stage 2 parameters
ALPHA0 = float(p2["alpha0"])
L_KM   = float(p2["decay_param"])      # decay scale in km
BETA   = float(p2["beta"])
ALPHA0_CI_LOW  = float(p2["alpha0_ci_low"])
ALPHA0_CI_HIGH = float(p2["alpha0_ci_high"])
L_CI_LOW  = float(p2["decay_ci_low"])
L_CI_HIGH = float(p2["decay_ci_high"])
MED_DIST  = 3.151  # median matched-pair distance (km), from FINDINGS.md

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 72 / 25.4))

# ═══════════════════════════════════════════════════════════════════════════
# Panel A: Stage 1 — coincidence rate vs. null + logistic fit
# ═══════════════════════════════════════════════════════════════════════════
ax = axes[0]

# Bin pairs by distance
BIN_EDGES = np.array([0, 1, 2, 3, 4, 5])
BIN_MIDS  = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
ep_close  = ep[ep["dist_km"] < 5].copy()
ep_close["dist_bin"] = pd.cut(ep_close["dist_km"], bins=BIN_EDGES,
                              labels=BIN_MIDS, include_lowest=True)

grp_obs  = ep_close.groupby("dist_bin", observed=True)["frac_matched"].median()
grp_null = ep_close.groupby("dist_bin", observed=True)["p_null_coincidence"].median()
mids     = grp_obs.index.astype(float)

x = np.arange(len(mids))
w = 0.32
ax.bar(x - w/2, grp_obs.values * 100, width=w, color=C_OBS, alpha=0.8,
       label="Observed rate", edgecolor="white", lw=0.4)
ax.bar(x + w/2, grp_null.values * 100, width=w, color=C_NULL, alpha=0.8,
       label="Poisson null", edgecolor="white", lw=0.4)

# Logistic fit curve (evaluated at bin mid-points)
# logit(P) = logit(p_null) + γ₀ + γ_d*log(d) + γ_x*X_i
# For plotting at median Xi=0, use only γ₀ + γ_d*log(d) as excess over null
d_fine = np.linspace(0.1, 5.0, 100)
p_null_fine = 0.01 * (d_fine / 15)  # approximate null ≈ window/day ≈ (d/v)/1440
p_null_fine = np.clip(p_null_fine, 1e-6, 1 - 1e-6)
logit_null  = np.log(p_null_fine / (1 - p_null_fine))
logit_fit   = logit_null + GAMMA0 + GAMMA_DIST * np.log(d_fine)
p_fit       = expit(logit_fit)

# Convert to bin indices for overlay
d_to_idx = np.interp(d_fine, [0, 5], [0, len(mids) - 1])
ax.plot(d_to_idx, p_fit * 100, color=C_FIT, lw=1.4, ls="-",
        label="Logistic fit (Stage 1)", zorder=5)

ax.set_xticks(x)
labels = [f"{int(e)}\u2013{int(e+1)}" for e in BIN_EDGES[:-1]]
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_xlabel("Inter-plant distance (km)")
ax.set_ylabel("Coincidence rate (%)")
ax.set_title("(a) Stage 1: excess coincidence", pad=5)

excess_pct = 10.2
note_a = f"Excess over null: +{excess_pct} pp (global)"

# ═══════════════════════════════════════════════════════════════════════════
# Panel B: Stage 2 — alpha(d) curve
# ═══════════════════════════════════════════════════════════════════════════
ax2 = axes[1]

# Pairwise diagnostic points
diag_plot = diag[diag["n_matched"] >= 15].copy()
ax2.scatter(diag_plot["dist_km"], diag_plot["alpha_naive"],
            s=8, c=C_DATA, alpha=0.45, linewidths=0,
            label="Per-pair estimate ($n\\geq15$)")

# Fitted curve: alpha(d) = alpha0 * exp(-d/L)
d_curve = np.linspace(0, 5.5, 200)
alpha_curve = ALPHA0 * np.exp(-d_curve / L_KM)
ax2.plot(d_curve, alpha_curve, color=C_CURVE, lw=2.0,
         label=fr"$\hat{{\alpha}}(d)=\hat{{\alpha}}_0 e^{{-d/\hat{{L}}}}$")

# CI band
alpha_lo = ALPHA0_CI_LOW  * np.exp(-d_curve / L_CI_HIGH)
alpha_hi = ALPHA0_CI_HIGH * np.exp(-d_curve / L_CI_LOW)
ax2.fill_between(d_curve, alpha_lo, alpha_hi, alpha=0.15, color=C_CURVE)

# Median matched-pair distance
alpha_at_med = ALPHA0 * np.exp(-MED_DIST / L_KM)
ax2.axvline(MED_DIST, color="#aaa", lw=0.8, ls=":", zorder=0)
ax2.plot(MED_DIST, alpha_at_med, "D", color=C_MED, markersize=5, zorder=6,
         label=f"Median match dist. ({MED_DIST:.1f} km), $\\hat{{\\alpha}}={alpha_at_med:.3f}$")

# Independence reference
ax2.axhline(0, color=C_DATA, lw=0.7, ls="--", alpha=0.5)

ax2.set_xlabel("Inter-plant distance (km)")
ax2.set_ylabel(r"$\hat{\alpha}$ (conditional magnitude coupling)")
ax2.set_title("(b) Stage 2: conditional magnitude", pad=5)
ax2.set_xlim(-0.1, 5.5)
ax2.set_ylim(-0.1, 1.1)

note_b = (
    f"$\\hat{{\\alpha}}_0={ALPHA0:.3f}$ CI $[{ALPHA0_CI_LOW:.3f}, {ALPHA0_CI_HIGH:.3f}]$  ·  "
    f"$\\hat{{L}}={L_KM:.2f}$ km CI $[{L_CI_LOW:.2f}, {L_CI_HIGH:.2f}]$  ·  "
    f"$\\hat{{\\beta}}={BETA:.3f}$  ·  Gate G3: $\\rho=0.32$"
)

fig.tight_layout(pad=0.5, w_pad=1.2)

# Legends below x-axis labels (outside plot; bbox_inches='tight' on save keeps full axes)
ax.legend(
    framealpha=0.85, edgecolor="none",
    loc="upper center", bbox_to_anchor=(0.5, -0.22),
    ncol=3, columnspacing=1.0, handlelength=1.4,
    borderaxespad=0,
)
ax2.legend(
    fontsize=6.5, framealpha=0.85, edgecolor="none",
    loc="upper center", bbox_to_anchor=(0.5, -0.18),
    ncol=2, columnspacing=1.0, handlelength=1.4,
    borderaxespad=0,
)

# Notes below legends
ax.text(
    0.5, -0.38, note_a,
    transform=ax.transAxes, ha="center", va="top", fontsize=7,
    color=C_OBS, clip_on=False,
    bbox=dict(fc="white", ec=C_OBS, lw=0.5, alpha=0.9, pad=3),
)
ax2.text(
    0.5, -0.40, note_b,
    transform=ax2.transAxes, ha="center", va="top", fontsize=6.5,
    clip_on=False,
    bbox=dict(fc="white", ec="#ccc", lw=0.5, alpha=0.95, pad=4),
)
save_publication_figure(fig, OUT)
plt.close()
print(f"[OK] Fig 4.4 saved → {OUT}")
