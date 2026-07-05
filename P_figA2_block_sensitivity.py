"""
P_figA2_block_sensitivity.py
=============================
Fig A.2 — Reserve ratio sensitivity to bootstrap block length (S1).

Reads:  results/gates/s1_block_length_sensitivity.parquet
Writes: paper/figures/figA2_block_sensitivity.pdf
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
S1_PQ = cfg.DIRS["gates"] / "s1_block_length_sensitivity.parquet"
OUT   = ROOT / "paper" / "figures" / "figA2_block_sensitivity.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
})

C_RATIO = "#2980b9"
C_REF   = "#888888"

s1 = pd.read_parquet(S1_PQ)
blocks = s1["block_length_days"].values
ratios = s1["ratio_median"].values
ci_lo  = s1["ratio_ci_low"].values
ci_hi  = s1["ratio_ci_high"].values

fig, ax = plt.subplots(figsize=(88 / 25.4, 65 / 25.4))

ax.fill_between(blocks, ci_lo, ci_hi, alpha=0.20, color=C_RATIO,
                label="95% CI")
ax.plot(blocks, ratios, color=C_RATIO, lw=1.8, marker="o", markersize=5,
        label="Ratio (real / independence)")

ax.axhline(1.0, color=C_REF, lw=0.8, ls="--", alpha=0.6,
           label="Independence (ratio = 1.0)")

# Annotate adopted block length
adopted = 10
ax.axvline(adopted, color="#333", lw=0.8, ls=":", alpha=0.7)
ax.text(adopted + 0.5, ax.get_ylim()[0] + 0.05,
        f"Adopted: {adopted}-day blocks", fontsize=6.5, color="#333", va="bottom")

ax.set_xlabel("Bootstrap block length (days)")
ax.set_ylabel("Reserve ratio (real / independence)")
ax.set_title("Robustness: ratio vs. bootstrap block length", pad=5)
ax.legend(framealpha=0.85, edgecolor="none")
ax.set_xticks(blocks)
ax.set_xticklabels([str(b) for b in blocks])

# Annotate stability
ratio_range = ratios.max() - ratios.min()
ax.text(0.97, 0.10,
        f"Ratio range across blocks:\n${ratios.min():.2f}$ to ${ratios.max():.2f}$ "
        f"(spread $\\Delta={ratio_range:.3f}$)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
        bbox=dict(fc="white", ec=C_RATIO, lw=0.5, alpha=0.9, pad=3))

fig.tight_layout(pad=0.4)
save_publication_figure(fig, OUT)
plt.close()
print(f"[OK] Fig A.2 saved → {OUT}")
