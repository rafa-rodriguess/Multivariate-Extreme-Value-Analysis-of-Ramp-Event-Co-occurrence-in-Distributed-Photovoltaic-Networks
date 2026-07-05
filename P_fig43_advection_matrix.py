"""
P_fig43_advection_matrix.py
============================
Fig 4.3 — Full advection-timing null result:
Heatmap of |Spearman rho| across 12 wind heights x 5 seasons x 2 modes.

Reads:  results/gates/s9_full_matrix.parquet
Writes: paper/figures/fig43_advection_matrix.pdf
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src import config as cfg
from src.paper_figures import save_publication_figure

ROOT  = Path(__file__).parent
S9_PQ = cfg.DIRS["gates"] / "s9_full_matrix.parquet"
OUT   = ROOT / "paper" / "figures" / "fig43_advection_matrix.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7,
    "figure.dpi": 300,
})

THRESHOLD = 0.10  # pre-specified relevance threshold

# ── Load and sort ────────────────────────────────────────────────────────────
s9 = pd.read_parquet(S9_PQ)

# Season order and labels
SEASON_ORDER  = ["JJA", "REST", "DJF", "MAM", "SON"]
SEASON_LABELS = {
    "JJA":  "JJA\n(primary)",
    "REST": "REST\n(primary)",
    "DJF":  "DJF\n(expl.)",
    "MAM":  "MAM\n(expl.)",
    "SON":  "SON\n(expl.)",
}
MODE_LABELS = {"A_nofilt": "Mode A\n(all events)", "B_coloc30": "Mode B\n(±30 min)"}

# Altitude order (low to high)
alt_order = sorted(s9["altitude_m"].unique())

# Construct label for each altitude
def alt_label(a):
    if a < 600:
        return f"{a} m"
    km = a / 1000.0
    return f"~{km:.1f} km"

alt_labels = [alt_label(a) for a in alt_order]

# ── Build pivot per mode ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 105 / 25.4))
fig.subplots_adjust(wspace=0.03, left=0.20, right=0.91, top=0.80, bottom=0.16)

cmap = plt.cm.YlOrRd
vmin, vmax = 0, 0.15

for ax_idx, mode_key in enumerate(["A_nofilt", "B_coloc30"]):
    ax   = axes[ax_idx]
    sub  = s9[s9["mode"] == mode_key].copy()

    # Build matrix: rows=altitude (low→high), cols=season
    mat = np.full((len(alt_order), len(SEASON_ORDER)), np.nan)
    is_meaningful = np.zeros((len(alt_order), len(SEASON_ORDER)), dtype=bool)

    for i, alt in enumerate(alt_order):
        for j, sea in enumerate(SEASON_ORDER):
            row = sub[(sub["altitude_m"] == alt) & (sub["season_key"] == sea)]
            if len(row) == 1:
                mat[i, j]          = abs(row["spearman_r"].iloc[0])
                is_meaningful[i, j] = row["meaningful"].iloc[0]

    im = ax.imshow(mat, aspect="auto", origin="lower",
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")

    # Mark cells above threshold with an X
    for i in range(len(alt_order)):
        for j in range(len(SEASON_ORDER)):
            if not np.isnan(mat[i, j]):
                txt = f"{mat[i,j]:.3f}"
                fs  = 5.5
                col = "black" if mat[i, j] < 0.07 else "white"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=fs, color=col)
                if is_meaningful[i, j]:
                    ax.text(j, i + 0.32, "!", ha="center", va="center",
                            fontsize=7, color="#e63946", fontweight="bold")

    # Primary vs exploratory divider (after column 1 = REST)
    ax.axvline(1.5, color="white", lw=1.5, ls="-")
    ax.axvline(1.5, color="#333", lw=0.7, ls="--", alpha=0.6)

    ax.set_title(MODE_LABELS[mode_key], pad=12)
    ax.set_xticks(range(len(SEASON_ORDER)))
    ax.set_xticklabels([SEASON_LABELS[s] for s in SEASON_ORDER],
                       fontsize=6.5)
    ax.tick_params(axis="x", pad=6)

    # Y ticks — set AFTER imshow
    ax.set_yticks(range(len(alt_order)))
    if ax_idx == 0:
        ax.set_yticklabels(alt_labels, fontsize=6.5)
        ax.set_ylabel("Wind altitude")
    else:
        ax.set_yticklabels([])

# Shared colorbar (flush with heatmaps; threshold label offset to avoid tick overlap)
cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.78])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label(
    r"$|\rho_S|$ (observed vs. expected advection lag)",
    fontsize=7, labelpad=8,
)
cb.ax.tick_params(labelsize=6.5)
cb.ax.axhline(THRESHOLD, color="#c0392b", lw=1.2, ls="--")
THRESHOLD_LABEL_Y = THRESHOLD - 0.028  # manual offset below the dashed line
cb.ax.text(
    1.35, THRESHOLD_LABEL_Y, f"Threshold\n({THRESHOLD})",
    transform=cb.ax.get_yaxis_transform(),
    va="center", ha="left", fontsize=5.5, color="#c0392b", clip_on=False,
)

# Super-title with divider annotation
fig.text(0.5, 0.995, "Advection timing null result — 12 altitudes × 5 seasons × 2 modes",
         ha="center", va="top", fontsize=8, color="#2c3e50")
fig.text(0.5, 0.02,
         "Primary groups: JJA, REST  |  Exploratory: DJF, MAM, SON  |  "
         "! = |ρ| > 0.10 (none survive Bonferroni, N=120)",
         ha="center", va="bottom", fontsize=6, color="#555", style="italic")

save_publication_figure(fig, OUT)
plt.close()
print(f"[OK] Fig 4.3 saved → {OUT}")
