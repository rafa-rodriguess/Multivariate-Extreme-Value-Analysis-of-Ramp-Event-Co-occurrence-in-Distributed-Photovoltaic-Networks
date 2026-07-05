"""
P_figA3_gpd_quality.py
=======================
Fig A.3 — GPD fit quality (Gate G2): representative QQ-plot + extremal index.

Reads:  results/gates/gpd_marginal_params.parquet
        data/interim/ramps_split.parquet
Writes: paper/figures/figA3_gpd_quality.pdf
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genpareto

sys.path.insert(0, str(Path(__file__).parent))
from src import config as cfg
from src.paper_figures import save_publication_figure

ROOT   = Path(__file__).parent
GPD    = cfg.DIRS["gates"] / "gpd_marginal_params.parquet"
RAMPS  = cfg.DIRS["interim"] / "ramps_split.parquet"
OUT    = ROOT / "paper" / "figures" / "figA3_gpd_quality.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
})

C_PASS = "#27ae60"
C_FAIL = "#e74c3c"
C_REF  = "#888888"

gpd = pd.read_parquet(GPD)
gpd_pass = gpd[gpd["gate2_pass"] == True].copy()
gpd_fail = gpd[gpd["gate2_pass"] == False].copy()

n_total = len(gpd)
n_pass  = len(gpd_pass)
n_fail  = len(gpd_fail)

# Representative series: down direction, median activity among passed
candidates = gpd_pass[gpd_pass["direction"] == "down"].copy()
candidates = candidates.sort_values("n_total_ramps")
rep = candidates.iloc[len(candidates) // 2]
rep_sid = rep["station_id"]
rep_dir = rep["direction"]

# Load exceedances for QQ (train split only, same logic as Gate G2)
ramps = pd.read_parquet(RAMPS, columns=["station_id", "direction", "delta_k", "split"])
sub = ramps[
    (ramps["station_id"] == rep_sid)
    & (ramps["direction"] == rep_dir)
    & (ramps["split"] == "train")
].copy()
sub["abs_mag"] = sub["delta_k"].abs()
u_thr = float(rep["threshold_u"])
y_exc = sub.loc[sub["abs_mag"] > u_thr, "abs_mag"].values - u_thr
y_exc = y_exc[np.isfinite(y_exc) & (y_exc > 0)]

xi = float(rep["xi"])
sigma = float(np.exp(rep["beta0"]))  # scale at reference covariates

fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 72 / 25.4))

# ── Panel A: QQ-plot (representative series) ────────────────────────────────
ax = axes[0]
if len(y_exc) >= 5 and np.isfinite(xi) and np.isfinite(sigma) and sigma > 0:
    u_theo = genpareto.cdf(np.sort(y_exc), xi, loc=0, scale=sigma)
    u_emp  = (np.arange(1, len(y_exc) + 1) - 0.5) / len(y_exc)
    ax.scatter(u_theo, u_emp, s=12, alpha=0.65, color="#2980b9", edgecolors="none")
    ax.plot([0, 1], [0, 1], color=C_FAIL, lw=1.0, ls="--", label="Perfect fit")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, framealpha=0.85, edgecolor="none")
else:
    ax.text(0.5, 0.5, "Insufficient exceedances\nfor QQ plot",
            ha="center", va="center", transform=ax.transAxes, fontsize=8)

ax.set_xlabel("GPD theoretical quantile")
ax.set_ylabel("Empirical quantile")
ax.set_title(
    f"(a) QQ-plot — {rep_sid} ({rep_dir}, $n={len(y_exc)}$ declustered ex.)",
    pad=5,
)

# ── Panel B: extremal index theta ───────────────────────────────────────────
ax2 = axes[1]
theta_vals = gpd_pass["theta_hat"].dropna()
theta_med  = theta_vals.median()
ax2.hist(theta_vals, bins=30, color=C_PASS, alpha=0.85,
         edgecolor="white", lw=0.3)
ax2.axvline(theta_med, color=C_FAIL, lw=1.2, ls="--",
            label=f"Median $\\hat{{\\theta}}={theta_med:.3f}$")
ax2.set_xlabel(r"Extremal index $\hat{\theta}$ (Ferro–Segers)")
ax2.set_ylabel("Count (station × direction)")
ax2.set_title("(b) Declustering index (passed series)", pad=5)
ax2.legend(fontsize=7, framealpha=0.85, edgecolor="none")
ax2.text(0.97, 0.97,
         f"Gate G2 pass rate:\n{n_pass}/{n_total} ({n_pass/n_total:.1%})",
         transform=ax2.transAxes, ha="right", va="top", fontsize=7,
         color=C_PASS,
         bbox=dict(fc="white", ec=C_PASS, lw=0.5, alpha=0.9, pad=3))

fig.tight_layout(pad=0.5, w_pad=1.2)
save_publication_figure(fig, OUT)
plt.close()
print(f"[OK] Fig A.3 saved → {OUT}")
print(f"  rep={rep_sid}/{rep_dir}, n_exc={len(y_exc)}, "
      f"n_pass={n_pass}, theta_med={theta_med:.3f}")
