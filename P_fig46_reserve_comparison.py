"""
P_fig46_reserve_comparison.py
==============================
Fig 4.6 (CENTRAL) — Reserve comparison: 5 scenarios, T=1 year.

Reads:  results/gates/f8_rq3_ratio.parquet
        results/gates/f8b_rq3b_ratio.parquet
        results/gates/f7_return_level_fit.parquet (for closed-form z)
        results/gates/f7_backtest_coverage.parquet
Writes: paper/figures/fig46_reserve_comparison.pdf
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
F8    = cfg.DIRS["gates"] / "f8_rq3_ratio.parquet"
F8B   = cfg.DIRS["gates"] / "f8b_rq3b_ratio.parquet"
FIT   = cfg.DIRS["gates"] / "f7_return_level_fit.parquet"
OUT   = ROOT / "paper" / "figures" / "fig46_reserve_comparison.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8.5,
    "legend.fontsize": 7.5, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
})

# ── Load at T=1 year ─────────────────────────────────────────────────────────
f8  = pd.read_parquet(F8)
f8b = pd.read_parquet(F8B)
f8_1y  = f8[f8["return_period_years"].round(2) == 1.00].iloc[0]
f8b_1y = f8b[f8b["return_period_years"].round(2) == 1.00].iloc[0]

# Return levels at T=1 year (absolute)
z_real  = float(f8_1y["z_real"])
z_ind   = float(f8_1y["z_independence"])
z_model = float(f8_1y["z_model_implied"])
z_cop   = float(f8b_1y["z_copula_gaussian"])
z_clsd  = float(f8b_1y["z_closed_form_gaussian"])

# Ratios relative to independence scenario (normalise to independence=1)
norm = z_ind
vals = {
    "Real\n(observed)":                      z_real  / norm,
    "Model-implied\n(H-T + Stage 1)":        z_model / norm,
    "Gaussian copula\n(+ Stage 1)":          z_cop   / norm,
    "Closed-form\n(no coincidence model)":   z_clsd  / norm,
    "Independence\n(naive baseline)":        z_ind   / norm,
}

# CI for real (from F8)
ci_lo = float(f8_1y["ratio_ci_low"])
ci_hi = float(f8_1y["ratio_ci_high"])

COLORS = {
    "Real\n(observed)":                    "#c0392b",
    "Model-implied\n(H-T + Stage 1)":      "#27ae60",
    "Gaussian copula\n(+ Stage 1)":        "#2ecc71",
    "Closed-form\n(no coincidence model)": "#e67e22",
    "Independence\n(naive baseline)":      "#7f8c8d",
}

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(140 / 25.4, 90 / 25.4))

labels  = list(vals.keys())
values  = list(vals.values())
colors  = [COLORS[l] for l in labels]
y_pos   = np.arange(len(labels))

bars = ax.barh(y_pos, values, color=colors, height=0.55,
               edgecolor="white", linewidth=0.8, zorder=3)

# CI for Real scenario (index 0) — single error bar + label (no duplicate bracket)
ax.errorbar(values[0], y_pos[0],
            xerr=[[values[0] - ci_lo], [ci_hi - values[0]]],
            fmt="none", color="#922b21", capsize=4, lw=1.5, zorder=6)
ax.text(
    ci_hi + 0.05, y_pos[0],
    f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]",
    ha="left", va="center", fontsize=6.5, color="#922b21", style="italic",
    bbox=dict(fc="white", ec="#922b21", lw=0.35, alpha=0.92, pad=2),
)

# Independence reference line
ax.axvline(1.0, color="#7f8c8d", lw=0.9, ls="--", zorder=1)
ax.text(
    0.992, 1.03, "Baseline (independence = 1.0)",
    transform=ax.get_xaxis_transform(),
    va="bottom", ha="right", fontsize=6.5, color="#555", style="italic",
    clip_on=False,
)

# Value labels on bars (per-bar vertical tweak)
label_y_offset = {0: -0.12, 3: -0.12}  # 2.40× and 1.83× — nudge upward
for i, (bar, val) in enumerate(zip(bars, values)):
    y_lbl = bar.get_y() + bar.get_height() / 2 + label_y_offset.get(i, 0.0)
    ax.text(val + 0.03, y_lbl,
            f"{val:.2f}×", va="center", ha="left",
            fontsize=8, fontweight="bold", color="#2c3e50")

# Annotation: Real vs Independence — bracket above bar (arrow was hidden inside red bar)
ratio_real = values[0]
pct_real_vs_ind = (ratio_real - 1.0) * 100.0
real_ann_y = y_pos[0] - 0.36
ax.annotate("",
            xy=(ratio_real, real_ann_y), xytext=(1.0, real_ann_y),
            arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.8))
for x_end in (1.0, ratio_real):
    ax.plot([x_end, x_end], [real_ann_y - 0.06, real_ann_y + 0.06],
            color="#c0392b", lw=1.2, zorder=7, clip_on=False)
ax.text((1.0 + ratio_real) / 2, real_ann_y - 0.10,
        f"+{pct_real_vs_ind:.0f}% (real vs. independence)",
        ha="center", va="bottom", fontsize=7, color="#c0392b", fontweight="bold",
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

# Annotation: Real vs Closed-form (computed from data)
ratio_clsd_norm = values[3]
pct_real_vs_clsd = (ratio_real / ratio_clsd_norm - 1.0) * 100.0
ax.annotate("",
            xy=(ratio_real, y_pos[3]), xytext=(ratio_clsd_norm, y_pos[3]),
            arrowprops=dict(arrowstyle="<->", color="#e67e22", lw=1.2))
ax.text((ratio_clsd_norm + ratio_real) / 2 + 0.06, y_pos[3] + 0.22,
        f"+{pct_real_vs_clsd:.0f}% (real vs. closed-form)",
        ha="center", va="center", fontsize=7, color="#e67e22")

# Note: model vs copula difference (Stage 1 shared)
copula_pct_diff = abs(values[1] - values[2]) / values[1] * 100.0
ax.text(values[1] + 0.05, (y_pos[1] + y_pos[2]) / 2,
        f"Stage 1 shared\n(copula choice:\n{copula_pct_diff:.1f}% difference)",
        va="center", ha="left", fontsize=6, color="#555", style="italic",
        bbox=dict(fc="white", ec="#ccc", lw=0.4, alpha=0.9, pad=2))

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Operational reserve (normalised, independence = 1.0)")
ax.set_title(
    "Reserve requirement by scenario — $T=1$ year",
    pad=16,
)
ax.set_xlim(0.7, max(ratio_real + 0.5, ci_hi + 0.55))
ax.invert_yaxis()  # Real at top

fig.tight_layout(pad=0.5)
save_publication_figure(fig, OUT)
plt.close()
print(f"[OK] Fig 4.6 saved → {OUT}")
print(f"  z_real={z_real:.4f}  z_ind={z_ind:.4f}  ratio={z_real/z_ind:.3f}x  "
      f"z_closed={z_clsd:.4f}  real/closed={(z_real/z_clsd):.3f}x")
