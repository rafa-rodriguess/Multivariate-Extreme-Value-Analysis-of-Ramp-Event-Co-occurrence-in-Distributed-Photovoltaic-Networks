"""
P_fig42_regime_asymmetry.py
============================
Fig 4.2 — Shared-regime vs. event-level dependence asymmetry (C1b result).

Panel A: observed coincidence rate (11.2%) vs. Poisson null (1.0%) by distance bin.
Panel B: chi_event (0.036) vs. chi_daily (0.124) per pair, with independence baseline.

Reads:  results/gates/event_pairing_summary.parquet
Writes: paper/figures/fig42_regime_asymmetry.pdf
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

ROOT   = Path(__file__).parent
EP_PQ  = cfg.DIRS["gates"] / "event_pairing_summary.parquet"
OUT    = ROOT / "paper" / "figures" / "fig42_regime_asymmetry.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
})

C_OBS   = "#c0392b"
C_NULL  = "#7f8c8d"
C_EVENT = "#e67e22"
C_DAILY = "#2980b9"
C_IND   = "#888888"

# ── Load ─────────────────────────────────────────────────────────────────────
ep = pd.read_parquet(EP_PQ)
ep["dist_km"] = ep["dist_ij_m"] / 1000.0

# ── Panel A: 1-km bins within 0-5 km ─────────────────────────────────────────
BIN_EDGES = np.array([0, 1, 2, 3, 4, 5])
BIN_MIDS  = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
ep["dist_bin"] = pd.cut(ep["dist_km"], bins=BIN_EDGES, labels=BIN_MIDS,
                        include_lowest=True)

grp_obs  = ep.groupby("dist_bin", observed=True)["frac_matched"].median()
grp_null = ep.groupby("dist_bin", observed=True)["p_null_coincidence"].median()
mids     = grp_obs.index.astype(float)

# ── Panel B scatter data ──────────────────────────────────────────────────────
close = ep[ep["dist_km"] < 5].copy()
chi_event_global = 0.0356   # from FINDINGS.md C1b
chi_daily_global = 0.1241   # from FINDINGS.md G1
chi_indep        = 0.05     # 1 - u = 1 - 0.95

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 72 / 25.4))

# ── Panel A ──────────────────────────────────────────────────────────────────
ax = axes[0]
x  = np.arange(len(mids))
w  = 0.35

ax.bar(x - w/2, grp_obs.values * 100, width=w, color=C_OBS,
       label="Observed", alpha=0.85, edgecolor="white", lw=0.5)
ax.bar(x + w/2, grp_null.values * 100, width=w, color=C_NULL,
       label="Poisson null", alpha=0.85, edgecolor="white", lw=0.5)

ax.set_xticks(x)
labels = [f"{int(e)}\u2013{int(e+1)}" for e in BIN_EDGES[:-1]]
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_xlabel("Inter-plant distance (km)")
ax.set_ylabel("Coincidence rate (%)")
ax.set_title("(a) Activity coincidence", pad=5)
ax.legend(framealpha=0.85, edgecolor="none", loc="upper left")

# ── Panel B ──────────────────────────────────────────────────────────────────
ax2 = axes[1]

# Scatter of per-pair chi_event vs chi_daily (close pairs only)
ax2.scatter(close["chi_hat_daily"], close["chi_event_hat"],
            s=6, c=C_DAILY, alpha=0.35, linewidths=0, label="Pair estimate")

# Reference lines
ax2.axhline(chi_indep, color=C_IND, lw=0.9, ls="--",
            label=f"Independence baseline ({chi_indep:.2f})")
ax2.axvline(chi_daily_global, color=C_DAILY, lw=0.8, ls=":",
            label=f"Median daily $\\hat{{\\chi}}$ ({chi_daily_global:.3f})")
ax2.axhline(chi_event_global, color=C_EVENT, lw=0.8, ls=":",
            label=f"Median event $\\hat{{\\chi}}$ ({chi_event_global:.3f})")

# 45-degree equality line for reference
lim = max(ax2.get_xlim()[1], 0.5)
ax2.plot([0, lim], [0, lim], color="#ccc", lw=0.7, ls="-", zorder=0)

ax2.set_xlabel(r"Daily-block $\hat{\chi}$ (Gate G1)")
ax2.set_ylabel(r"Event-level $\hat{\chi}$ (C1b)")
ax2.set_title("(b) Magnitude coupling", pad=5)
ax2.set_xlim(-0.02, 0.5)
ax2.set_ylim(-0.02, 0.5)
ax2.legend(fontsize=6.5, framealpha=0.85, edgecolor="none", loc="upper left")

fig.tight_layout(pad=0.5, w_pad=1.2)
save_publication_figure(fig, OUT)
plt.close()
print(f"[OK] Fig 4.2 saved → {OUT}")
