"""
P_figA4_censoring.py
======================
Fig A.4 — Informative censoring check (D2): coverage vs. ramp activity correlation.

Reads:  results/gates/informative_censoring_results.parquet
Writes: paper/figures/figA4_censoring.pdf
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
D2_PQ = cfg.DIRS["gates"] / "informative_censoring_results.parquet"
OUT   = ROOT / "paper" / "figures" / "figA4_censoring.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
})

C_SIG   = "#e74c3c"
C_NSIG  = "#2980b9"
C_ZERO  = "#888888"

d2 = pd.read_parquet(D2_PQ)
n_sig_inf = d2["significant_informative"].sum()
n_sig_rob = d2["significant_informative_robust"].sum()
n_total   = len(d2)

fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 72 / 25.4))

# Panel A: coverage_median vs. rho_extreme_ramp
ax = axes[0]
colors = d2["significant_informative"].map({True: C_SIG, False: C_NSIG})
ax.scatter(d2["coverage_median"], d2["rho_extreme_ramp"],
           c=colors, s=18, alpha=0.7, linewidths=0.3, edgecolors="white")
ax.axhline(0, color=C_ZERO, lw=0.7, ls="--", alpha=0.6)

from matplotlib.patches import Patch
legend_el = [Patch(fc=C_SIG, label=f"Informative (BH, $n={n_sig_inf}$)"),
             Patch(fc=C_NSIG, label=f"Non-informative ($n={n_total - n_sig_inf}$)")]
ax.legend(handles=legend_el, fontsize=7.5, framealpha=0.85, edgecolor="none")
ax.set_xlabel("Station data coverage (fraction)")
ax.set_ylabel(r"$\rho_S$ (coverage vs. extreme ramp activity)")
ax.set_title("(a) Coverage vs. extreme-ramp correlation", pad=5)

# Panel B: histogram of rho_extreme_ramp
ax2 = axes[1]
bins = np.linspace(-0.5, 0.5, 22)
ax2.hist(d2.loc[~d2["significant_informative"], "rho_extreme_ramp"],
         bins=bins, color=C_NSIG, alpha=0.8, label="Non-informative",
         edgecolor="white", lw=0.3)
ax2.hist(d2.loc[d2["significant_informative"], "rho_extreme_ramp"],
         bins=bins, color=C_SIG, alpha=0.9, label="Informative",
         edgecolor="white", lw=0.3)
ax2.axvline(0, color=C_ZERO, lw=0.7, ls="--", alpha=0.6)
ax2.set_xlabel(r"$\rho_S$ (coverage vs. extreme ramp activity)")
ax2.set_ylabel("Count (stations)")
ax2.set_title("(b) Distribution of $\\rho_S$", pad=5)
ax2.legend(fontsize=7.5, framealpha=0.85, edgecolor="none")
ax2.text(0.97, 0.97,
         f"{n_sig_inf}/{n_total} stations\ninformative (BH)\n"
         f"{n_sig_rob}/{n_total} robust test",
         transform=ax2.transAxes, ha="right", va="top", fontsize=7,
         bbox=dict(fc="white", ec=C_SIG, lw=0.5, alpha=0.9, pad=3))

fig.tight_layout(pad=0.5, w_pad=1.2)
save_publication_figure(fig, OUT)
plt.close()
print(f"[OK] Fig A.4 saved → {OUT}")
print(f"  n_total={n_total}, n_sig_informative={n_sig_inf}, n_sig_robust={n_sig_rob}")
