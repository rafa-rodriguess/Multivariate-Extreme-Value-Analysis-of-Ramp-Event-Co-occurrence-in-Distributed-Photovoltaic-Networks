"""
P_figB2_spatial_coherence.py
=============================
Fig B.2 — Spatial coherence: sharp vs. gradual ramps (F2 result).

Reads:  results/gates/s3_f2_threshold_grid.parquet  (for r values)
Writes: paper/figures/figB2_spatial_coherence.pdf
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
OUT   = ROOT / "paper" / "figures" / "figB2_spatial_coherence.pdf"
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
# The adopted sharp threshold
sharp_row = s3[s3["sharp_delta_thresh"].round(2) == 0.15]
if len(sharp_row) == 0:
    sharp_row = s3.iloc[[abs(s3["sharp_delta_thresh"] - 0.15).argmin()]]
r_sharp = float(sharp_row["r_pooled"].iloc[0])

# Gradual: from FINDINGS.md (r ≈ 0.047 for |Δk| < 0.10)
r_gradual = 0.047

fig, axes = plt.subplots(1, 2, figsize=(140 / 25.4, 70 / 25.4))

# ── Panel A: simple bar comparison ────────────────────────────────────────────
ax = axes[0]
bars = ax.bar(["Sharp\n($|\\Delta k|\\geq0.15$)", "Gradual\n($|\\Delta k|<0.10$)"],
              [r_sharp, r_gradual],
              color=[C_SHARP, C_GRAD], alpha=0.85, width=0.5,
              edgecolor="white", lw=0.5)
for bar, val in zip(bars, [r_sharp, r_gradual]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
            f"$r={val:.3f}$", ha="center", va="bottom", fontsize=8,
            fontweight="bold")
ax.axhline(0, color="#ccc", lw=0.5)
ax.set_ylabel("Spatial coherence $r$ (Pearson)")
ax.set_title("(a) Sharp vs. gradual ramps", pad=5)
ax.set_ylim(0, 0.28)

ratio = r_sharp / r_gradual
ax.text(0.5, 0.92, f"Contrast ratio: {ratio:.1f}$\\times$",
        transform=ax.transAxes, ha="center", va="top", fontsize=9,
        fontweight="bold", color=C_SHARP)

# ── Panel B: r vs. threshold from S3 ─────────────────────────────────────────
ax2 = axes[1]
ax2.plot(s3["sharp_delta_thresh"], s3["r_pooled"], color=C_SHARP, lw=1.6,
         marker="o", markersize=3, label="Sharp pooled $r$")
ax2.axhline(r_gradual, color=C_GRAD, lw=1.0, ls="--",
            label=f"Gradual $r={r_gradual}$ (reference)")
ax2.axvline(0.15, color="#333", lw=0.7, ls=":", alpha=0.6,
            label="Adopted threshold")
ax2.set_xlabel(r"Sharp magnitude threshold $|\Delta k|$")
ax2.set_ylabel("Spatial coherence $r$")
ax2.set_title("(b) Coherence vs. threshold (all values)", pad=5)
ax2.legend(fontsize=7, framealpha=0.85, edgecolor="none")

fig.tight_layout(pad=0.5, w_pad=1.5)
save_publication_figure(fig, OUT)
plt.close()
print(f"[OK] Fig B.2 saved → {OUT}")
print(f"  r_sharp={r_sharp:.3f}, r_gradual={r_gradual:.3f}, ratio={r_sharp/r_gradual:.1f}x")
