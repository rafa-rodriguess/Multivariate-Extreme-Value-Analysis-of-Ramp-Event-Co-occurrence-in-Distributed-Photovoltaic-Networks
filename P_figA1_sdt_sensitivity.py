"""
P_figA1_sdt_sensitivity.py
===========================
Fig A.1 — Sharp vs. gradual ramp coherence vs. magnitude threshold (S3 sensitivity).

Reads:  results/gates/s3_f2_threshold_grid.parquet
Writes: paper/figures/figA1_sdt_sensitivity.pdf
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
S3_PQ = cfg.DIRS["gates"] / "s3_f2_threshold_grid.parquet"
OUT   = ROOT / "paper" / "figures" / "figA1_sdt_sensitivity.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
})

C_SHARP = "#c0392b"
C_GRAD  = "#7f8c8d"

s3 = pd.read_parquet(S3_PQ)

fig, ax = plt.subplots(figsize=(88 / 25.4, 70 / 25.4))

ax.plot(s3["sharp_delta_thresh"], s3["r_pooled"], color=C_SHARP, lw=1.6,
        marker="o", markersize=3.5, label="Sharp ramps (pooled $r$)")
ax.plot(s3["sharp_delta_thresh"], s3["r_one_per_block"], color=C_SHARP, lw=1.0,
        ls="--", marker="s", markersize=2.5, alpha=0.6,
        label="Sharp ramps (1 per block)")

# The adopted threshold
adopted = 0.15
ax.axvline(adopted, color="#333", lw=0.8, ls=":", alpha=0.7)
ax.text(adopted + 0.003, 0.13, f"Adopted\n$|\\Delta k|\\geq{adopted}$",
        fontsize=6.5, color="#333", va="bottom")

# Gradual reference (r=0.047 from FINDINGS.md)
ax.axhline(0.047, color=C_GRAD, lw=0.9, ls="--",
           label="Gradual ramps (all thresholds, $r\\approx0.047$)")

ax.set_xlabel(r"Magnitude threshold $|\Delta k|_\mathrm{sharp}$")
ax.set_ylabel("Spatial coherence $r$ (Pearson, across pairs)")
ax.set_title("Robustness: sharp/gradual contrast vs. threshold", pad=5)
ax.set_ylim(0, 0.42)

contrast_note = "Contrast ratio $\\approx$4.5 stable across threshold choices"

fig.tight_layout(pad=0.4)

# Legend below plot (outside axes)
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.16),
    ncol=2, framealpha=0.85, edgecolor="none", columnspacing=1.2,
)
ax.text(
    0.5, -0.32, contrast_note,
    transform=ax.transAxes, ha="center", va="top", fontsize=7,
    bbox=dict(fc="white", ec="#333", lw=0.5, alpha=0.9, pad=3),
    clip_on=False,
)
save_publication_figure(fig, OUT)
plt.close()
print(f"[OK] Fig A.1 saved → {OUT}")
