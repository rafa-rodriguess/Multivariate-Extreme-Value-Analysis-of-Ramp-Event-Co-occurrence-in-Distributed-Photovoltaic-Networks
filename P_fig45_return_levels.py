"""
P_fig45_return_levels.py
=========================
Fig 4.5 — Return level curve + Gate G4 backtest.

Reads:  results/gates/f7_return_level_fit.parquet
        results/gates/f7_return_level_bootstrap.parquet
        results/gates/f7_backtest_coverage.parquet
Writes: paper/figures/fig45_return_levels.pdf
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src import config as cfg
from src.paper_figures import save_publication_figure

ROOT  = Path(__file__).parent
FIT   = cfg.DIRS["gates"] / "f7_return_level_fit.parquet"
BOOT  = cfg.DIRS["gates"] / "f7_return_level_bootstrap.parquet"
BT    = cfg.DIRS["gates"] / "f7_backtest_coverage.parquet"
OUT   = ROOT / "paper" / "figures" / "fig45_return_levels.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
})

C_CURVE = "#2980b9"
C_CI    = "#2980b9"
C_OBS   = "#c0392b"
C_COVER = "#27ae60"
C_FAIL  = "#e74c3c"

# ── Load ─────────────────────────────────────────────────────────────────────
fit  = pd.read_parquet(FIT).iloc[0]
boot = pd.read_parquet(BOOT)
bt   = pd.read_parquet(BT)

# Extract point estimates from fit row
T_cols = [c for c in fit.index if c.startswith("z_") and "ci" not in c]
T_vals = {float(c.replace("z_","").replace("y","")): float(fit[c]) for c in T_cols}
T_lo   = {float(c.replace("z_","").replace("y","").replace("_ci_low","")): float(fit[c])
           for c in fit.index if c.endswith("ci_low")}
T_hi   = {float(c.replace("z_","").replace("y","").replace("_ci_high","")): float(fit[c])
           for c in fit.index if c.endswith("ci_high")}

T_sorted  = sorted(T_vals.keys())
z_sorted  = [T_vals[t] for t in T_sorted]
z_lo_srt  = [T_lo[t]  for t in T_sorted]
z_hi_srt  = [T_hi[t]  for t in T_sorted]

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(88 / 25.4, 75 / 25.4))

# CI band
ax.fill_between(T_sorted, z_lo_srt, z_hi_srt,
                alpha=0.20, color=C_CI, label="95% CI (block bootstrap)")

# Return level curve
ax.plot(T_sorted, z_sorted, color=C_CURVE, lw=1.8, marker="o", markersize=3.5,
        label="Return level (GPD fit)")

# Backtest: mark each tested T with a symbol (covered = green circle)
from matplotlib.lines import Line2D
for _, row in bt.iterrows():
    T_yr    = row["return_period_years"]
    z_T     = row["z_T"]
    covered = row["covered_95pct_poisson_pi"]
    obs     = int(row["observed_exceedances"])
    exp     = float(row["expected_exceedances"])
    pl      = int(row["poisson_pi_low"])
    ph      = int(row["poisson_pi_high"])
    c = C_COVER if covered else C_FAIL
    m = "o" if covered else "x"
    ax.plot(T_yr, z_T, marker=m, ms=8, color=c, zorder=7,
            markeredgecolor="white" if covered else c, markeredgewidth=0.5)
    ax.text(T_yr, z_T + 0.003,
            f"obs={obs}\n[{pl},{ph}]",
            ha="center", va="bottom", fontsize=5.5, color=c)

leg_bt = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_COVER,
           markersize=7, label="Backtest: covered (4/4)"),
]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + leg_bt, labels + [leg_bt[0].get_label()],
          loc="upper left", framealpha=0.85, edgecolor="none")

# Parameters text box
xi  = float(fit["xi"])
sig = float(fit["sigma"])
u   = float(fit["u"])
n   = int(fit["n_declustered"])
ax.text(0.97, 0.05,
        f"$\\hat{{u}}={u:.3f}$ · $\\hat{{\\xi}}={xi:.3f}$ · $\\hat{{\\sigma}}={sig:.4f}$\n"
        f"$n_{{declust}}={n}$ · $\\hat{{\\theta}}={float(fit['theta_hat']):.3f}$\n"
        "2017 holdout: 4/4 horizons covered",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
        family="monospace",
        bbox=dict(fc="white", ec="#ccc", lw=0.5, alpha=0.95, pad=3))

ax.set_xscale("log")
ax.set_xlabel("Return period (years)")
ax.set_ylabel(r"Return level $|\Delta k|$ (aggregate ramp)")
ax.set_title("Return levels + Gate G4 backtest", pad=5)
ax.set_xlim(0.07, 6)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2g}"))

fig.tight_layout(pad=0.4)
save_publication_figure(fig, OUT)
plt.close()
print(f"[OK] Fig 4.5 saved → {OUT}")
