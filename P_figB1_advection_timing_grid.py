"""
P_figB1_advection_timing_grid.py
==================================
Fig B.1 — Advection timing grid: 4 heights × 2 seasons (F6b + S7).

Reads:  results/gates/s9_full_matrix.parquet (for rho values)
        results/gates/f6b_timing_bootstrap.parquet (for 10m)
        results/gates/f6b_timing_bootstrap_100m.parquet
        results/gates/f6b_timing_bootstrap_200m.parquet
        results/gates/f6b_timing_bootstrap_500m.parquet
Writes: paper/figures/figB1_advection_timing.pdf
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
S9_PQ = cfg.DIRS["gates"] / "s9_full_matrix.parquet"
OUT   = ROOT / "paper" / "figures" / "figB1_advection_timing.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.5,
})

THRESHOLD = 0.10
# Height × Season grid
HEIGHTS  = [10, 100, 200, 500]
H_LABELS = ["KNMI 10 m", "CERRA 100 m", "CERRA 200 m", "CERRA 500 m"]
SEASONS  = ["JJA", "REST"]
S_LABELS = ["JJA (primary)", "REST (primary)"]

s9 = pd.read_parquet(S9_PQ)
# Use Mode A (no filter) for this grid
s9a = s9[s9["mode"] == "A_nofilt"].copy()

fig, axes = plt.subplots(len(SEASONS), len(HEIGHTS),
                         figsize=(180 / 25.4, 100 / 25.4),
                         sharex=False, sharey=False)
fig.subplots_adjust(hspace=0.55, wspace=0.35,
                    left=0.08, right=0.97, top=0.82, bottom=0.14)

for row_i, (season, slabel) in enumerate(zip(SEASONS, S_LABELS)):
    for col_j, (alt, hlabel) in enumerate(zip(HEIGHTS, H_LABELS)):
        ax = axes[row_i, col_j]

        sub = s9a[(s9a["season_key"] == season) & (s9a["altitude_m"] == alt)]

        if len(sub) == 1:
            rho   = float(sub["spearman_r"].iloc[0])
            n     = int(sub["n_valid"].iloc[0])
            above = abs(rho) >= THRESHOLD
        else:
            rho = np.nan; n = 0; above = False

        # Show rho as a vertical bar on zero-centered axis
        c = "#c0392b" if above else "#2980b9"
        ax.barh(0, rho, color=c, height=0.5, alpha=0.8)
        ax.axvline(0, color="#888", lw=0.5)
        ax.axvline(THRESHOLD,  color="#e67e22", lw=0.6, ls="--", alpha=0.7)
        ax.axvline(-THRESHOLD, color="#e67e22", lw=0.6, ls="--", alpha=0.7)

        ax.set_xlim(-0.15, 0.15)
        ax.set_yticks([])
        ax.set_xlabel(r"$\rho_S$", labelpad=1)

        title = f"{hlabel}\n{slabel}"
        ax.set_title(title, fontsize=6.5, pad=3, color=c if above else "#333")

        ax.text(0.5, 0.12, f"$\\rho_S={rho:.3f}$\n$n={n:,}$",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=5.5, color=c)

# Super-title and legend note below grid
fig.text(0.5, 0.995,
         "Advection timing: observed vs. expected lag "
         f"(Mode A, threshold $|\\rho_S|={THRESHOLD}$, none exceeded in primary seasons)",
         ha="center", va="top", fontsize=7.5, color="#2c3e50")
fig.text(0.5, 0.04,
         "Orange dashed lines: pre-specified relevance threshold; "
         "blue = null, red = above threshold",
         ha="center", va="bottom", fontsize=6, color="#555", style="italic")

save_publication_figure(fig, OUT)
plt.close()
print(f"[OK] Fig B.1 saved → {OUT}")
