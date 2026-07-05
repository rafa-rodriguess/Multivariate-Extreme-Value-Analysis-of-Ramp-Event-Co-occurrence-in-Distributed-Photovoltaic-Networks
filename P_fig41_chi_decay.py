"""
P_fig41_chi_decay.py
====================
Fig 4.1 — Tail-dependence coefficient chi-hat(u=0.95) vs. inter-plant distance.

Reads:  results/gates/chi_estimates_raw.parquet
        results/gates/gate1_results.parquet
Writes: paper/figures/fig41_chi_decay.pdf
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src import config as cfg
from src.paper_figures import save_publication_figure

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
CHI_PQ  = cfg.DIRS["gates"] / "chi_estimates_raw.parquet"
G1_PQ   = cfg.DIRS["gates"] / "gate1_results.parquet"
OUT     = ROOT / "paper" / "figures" / "fig41_chi_decay.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Publication style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})
C_CLOSE   = "#c0392b"   # red for close pairs
C_ALL     = "#2980b9"   # blue for all pairs
C_THRESH  = "#888888"   # grey for threshold/reference
U_DECISION = 0.95

# ── Load data ────────────────────────────────────────────────────────────────
chi_raw = pd.read_parquet(CHI_PQ)
chi_95  = chi_raw[chi_raw["u"] == U_DECISION].copy()
chi_95["dist_km"] = chi_95["dist_ij_m"] / 1000.0

g1 = pd.read_parquet(G1_PQ)
g1_95  = g1[g1["u"] == U_DECISION].copy()
g1_95["dist_km"] = g1_95["dist_ij_m"] / 1000.0

# Merge significance flag into chi_raw
merged = chi_95.merge(
    g1_95[["station_i", "station_j", "significant"]],
    on=["station_i", "station_j"], how="left"
)

# ── Binned statistics ────────────────────────────────────────────────────────
BIN_EDGES = np.arange(0, 62, 4)
BIN_MIDS  = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
merged["dist_bin"] = pd.cut(merged["dist_km"], bins=BIN_EDGES, labels=BIN_MIDS)
grp = merged.groupby("dist_bin", observed=True)["chi_hat"]
med  = grp.median()
q25  = grp.quantile(0.25)
q75  = grp.quantile(0.75)
mids = med.index.astype(float)

# Close pairs (<5 km) with bootstrap CI from gate1_results
close   = merged[merged["dist_km"] < 5].copy()
n_close = len(close)
n_sig   = (close["significant"] == True).sum()

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(88 / 25.4, 70 / 25.4))

# Shaded IQR
ax.fill_between(mids, q25, q75, alpha=0.20, color=C_ALL, label="IQR (all pairs)")

# Median line — all pairs
ax.plot(mids, med, color=C_ALL, lw=1.4, marker="o", markersize=3,
        label=r"Median $\hat{\chi}$ (all pairs)")

# Independence reference
ax.axhline(1 - U_DECISION, color=C_THRESH, lw=0.8, ls="--",
           label=f"Independence baseline ($1-u={1-U_DECISION:.2f}$)")

# Vertical divider for close-pair zone
ax.axvspan(0, 5, alpha=0.07, color=C_CLOSE)
ax.axvline(5, color=C_CLOSE, lw=0.7, ls=":", alpha=0.7)

# Annotation for close pairs
frac_sig_pct = 100 * n_sig / n_close
ax.text(2.5, 0.01,
        f"{frac_sig_pct:.1f}%\nsignif.",
        ha="center", va="bottom", fontsize=6.5, color=C_CLOSE,
        style="italic", zorder=5,
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))

ax.set_xlabel("Inter-plant distance (km)")
ax.set_ylabel(r"Tail-dependence coefficient $\hat{\chi}$ ($u=0.95$)")
ax.set_xlim(0, 60)
ax.set_ylim(-0.01, 0.38)
ax.legend(
    loc="upper right", bbox_to_anchor=(0.995, 1.0),
    fontsize=7, framealpha=0.85, edgecolor="none",
)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

fig.tight_layout(pad=0.4)
save_publication_figure(fig, OUT)
plt.close()
print(f"[OK] Fig 4.1 saved → {OUT}")
