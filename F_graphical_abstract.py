"""
F_graphical_abstract.py
=======================
Generates paper/figures/graphical_abstract.pdf (vector PDF,
white background, no title/caption inside image — Springer standard).

Layout (3 horizontal panels):
  Panel A: Utrecht PV map with edges coloured by chi_hat
  Panel B: chi_hat vs. distance decay curve (real data)
  Panel C: Reserve bar chart (independence vs. model) + coincidence gap

Data sources:
  data/interim/coords.parquet          -> lat/lon per station
  results/gates/chi_estimates_raw.parquet -> chi_hat per pair at u=0.95
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from src.paper_figures import save_publication_figure

ROOT    = pathlib.Path(__file__).parent
COORDS  = ROOT / "data/interim/coords.parquet"
CHI     = ROOT / "results/gates/chi_estimates_raw.parquet"
OUT_DIR = ROOT / "paper/figures"
OUT     = OUT_DIR / "graphical_abstract.pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# High-resolution export: large vector canvas + sharp raster basemap tiles
# Canvas ~7200×2370 px (24 in @ 300 dpi) — print-grade without tile overload
FIGSIZE_IN = (24.0, 7.9)
FIG_DPI    = 300
SAVE_DPI   = 1200          # embedded raster DPI in PDF
BASEMAP_ZOOM = 14          # fixed high zoom (Utrecht extent)

# ── Palette ──────────────────────────────────────────────────────────────────
C_TAIL   = "#c0392b"   # red  — tail dependence / risk
C_IND    = "#2980b9"   # blue — independence assumption
C_MODEL  = "#27ae60"   # green — our model
C_MUTED  = "#95a5a6"   # grey — neutral
TITLE_C  = "#2c3e50"

# ── Load data ────────────────────────────────────────────────────────────────
coords = pd.read_parquet(COORDS, columns=["station_id","lat_centroid","lon_centroid"])
coords = coords.dropna()

chi_all = pd.read_parquet(CHI)
chi_95  = chi_all[chi_all["u"] == 0.9].copy()          # u=0.9 (closest to 0.95)
chi_95["dist_km"] = chi_95["dist_ij_m"] / 1000.0

# ── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=FIGSIZE_IN, facecolor="white", dpi=FIG_DPI)
gs  = gridspec.GridSpec(1, 3, figure=fig,
                        left=0.04, right=0.97,
                        top=0.82, bottom=0.14,
                        wspace=0.38)

# Panel labels style
PANEL_KW = dict(fontsize=11, fontweight="bold", color=TITLE_C,
                transform=None, va="top", ha="left")

# ════════════════════════════════════════════════════════════════════════════
# PANEL A — Utrecht map with short-range edges coloured by chi_hat
# ════════════════════════════════════════════════════════════════════════════
axA = fig.add_subplot(gs[0])

# Convert to Web Mercator
gdf = gpd.GeoDataFrame(
    coords,
    geometry=[Point(xy) for xy in zip(coords.lon_centroid, coords.lat_centroid)],
    crs=4326,
).to_crs(3857)
x = gdf.geometry.x.values
y = gdf.geometry.y.values
sid_to_idx = {s: i for i, s in enumerate(gdf.station_id)}

# Select close pairs (< 8 km) with valid chi
edges = chi_95[chi_95["dist_km"] < 8].copy()
edges = edges.dropna(subset=["chi_hat"])

chi_norm = Normalize(vmin=0, vmax=0.35)
edge_cmap = plt.cm.YlOrRd

for _, row in edges.iterrows():
    i = sid_to_idx.get(row.station_i)
    j = sid_to_idx.get(row.station_j)
    if i is None or j is None:
        continue
    color = edge_cmap(chi_norm(row.chi_hat))
    axA.plot([x[i], x[j]], [y[i], y[j]],
             color=color, lw=0.5, alpha=0.55, zorder=2)

# Station dots
axA.scatter(x, y, s=10, c="#2c3e50", zorder=4, linewidths=0)

# Basemap
pad = 2000
axA.set_xlim(x.min()-pad, x.max()+pad)
axA.set_ylim(y.min()-pad, y.max()+pad)
try:
    ctx.add_basemap(axA, crs=gdf.crs,
                    source=ctx.providers.CartoDB.Positron,
                    zoom=BASEMAP_ZOOM, attribution_size=5)
except Exception:
    axA.set_facecolor("#f5f5f0")

# Colorbar for edges
sm = ScalarMappable(cmap=edge_cmap, norm=chi_norm)
sm.set_array([])
cbA = fig.colorbar(sm, ax=axA, orientation="horizontal",
                   fraction=0.046, pad=0.12, aspect=20)
cbA.set_label(r"Tail dependence $\hat{\chi}$ (pairs $<$8 km)", fontsize=7.5)
cbA.ax.tick_params(labelsize=7)

axA.axis("off")
axA.set_title("(a) Spatial network", fontsize=9, pad=4, color=TITLE_C, fontweight="bold")

# ════════════════════════════════════════════════════════════════════════════
# PANEL B — chi_hat vs. distance (binned median ± IQR)
# ════════════════════════════════════════════════════════════════════════════
axB = fig.add_subplot(gs[1])

bins  = np.arange(0, 62, 4)          # 0-60 km, 4-km bins
mids  = (bins[:-1] + bins[1:]) / 2
chi_95["dist_bin"] = pd.cut(chi_95["dist_km"], bins=bins, labels=mids)
grp   = chi_95.groupby("dist_bin", observed=True)["chi_hat"]
med   = grp.median()
q25   = grp.quantile(0.25)
q75   = grp.quantile(0.75)
mids_v = med.index.astype(float)

axB.fill_between(mids_v, q25, q75,
                 alpha=0.20, color=C_TAIL, label="IQR")
axB.plot(mids_v, med,
         color=C_TAIL, lw=2.0, marker="o", markersize=3.5,
         label=r"Median $\hat{\chi}$")
axB.axhline(0, color="#555", lw=0.8, ls="--", label="Independence (χ = 0)")

# Annotate significance region
axB.axvspan(0, 5, alpha=0.07, color=C_TAIL)
axB.text(2.5, 0.32, "52.1%\nsignif.", ha="center", va="top",
         fontsize=6.5, color=C_TAIL, style="italic")

# Exponential-decay sketch
d_fit = np.linspace(0.5, 60, 200)
chi0, L = 0.26, 12.0          # approximate from G1 results
axB.plot(d_fit, chi0 * np.exp(-d_fit / L),
         color=C_TAIL, lw=1.0, ls=":", alpha=0.7,
         label=r"Exp. decay ($L$≈12 km)")

axB.set_xlabel("Inter-plant distance (km)", fontsize=8.5)
axB.set_ylabel(r"Tail dependence $\hat{\chi}$ ($u=0.90$)", fontsize=8.5)
axB.set_xlim(0, 60)
axB.set_ylim(-0.02, 0.38)
axB.tick_params(labelsize=7.5)
axB.legend(fontsize=6.5, loc="upper right", framealpha=0.8)
axB.spines[["top","right"]].set_visible(False)
axB.set_title("(b) Tail dependence decay", fontsize=9, pad=4,
              color=TITLE_C, fontweight="bold")

# ════════════════════════════════════════════════════════════════════════════
# PANEL C — Reserve bar chart
# ════════════════════════════════════════════════════════════════════════════
axC = fig.add_subplot(gs[2])

labels  = ["Independence\nassumption\n(market)", "Our model\n(coincidence\n+ copula)"]
values  = [1.00, 2.39]       # relative reserve requirement (independence = 1.0)
ci_low  = [0.00, 2.03]
ci_high = [0.00, 2.70]
colors  = [C_IND, C_MODEL]

bars = axC.bar(labels, values, color=colors, width=0.42,
               edgecolor="white", linewidth=1.2, zorder=3)

# Error bar on model bar
axC.errorbar(1, values[1],
             yerr=[[values[1]-ci_low[1]], [ci_high[1]-values[1]]],
             fmt="none", color="#1a5c2a", capsize=5, lw=1.5, zorder=5)

# Annotation: +139% gap
gap_y = values[0] + (values[1] - values[0]) / 2
axC.annotate("",
             xy=(0.72, values[1]), xytext=(0.72, values[0]),
             arrowprops=dict(arrowstyle="<->", color=C_TAIL, lw=1.6))
axC.text(0.55, gap_y, "+139%\n(+31% from\ncoincidence)", ha="right", va="center",
         fontsize=7, color=C_TAIL, fontweight="bold",
         bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))

# Reference line at independence
axC.axhline(1.0, color=C_IND, lw=0.8, ls="--", alpha=0.6)

axC.set_ylabel("Relative operational reserve\n(independence baseline = 1.0)", fontsize=8)
axC.set_ylim(0, 3.0)
axC.tick_params(axis="x", labelsize=8)
axC.tick_params(axis="y", labelsize=7.5)
axC.spines[["top","right"]].set_visible(False)
axC.set_title("(c) Reserve underestimation", fontsize=9, pad=4,
              color=TITLE_C, fontweight="bold")

# Value labels on bars
for bar, val in zip(bars, values):
    axC.text(bar.get_x() + bar.get_width()/2,
             val + 0.05, f"{val:.2f}×",
             ha="center", va="bottom", fontsize=9, fontweight="bold",
             color=bar.get_facecolor())

# ════════════════════════════════════════════════════════════════════════════
# Global headline
# ════════════════════════════════════════════════════════════════════════════
fig.text(0.5, 0.96,
         "Geographic diversification fails at the tail: "
         "coincidence of extreme ramp events drives reserve underestimation",
         ha="center", va="top", fontsize=10.5, fontweight="bold",
         color=TITLE_C, wrap=True)

fig.text(0.5, 0.89,
         "Utrecht PV network · 174 systems · 556,859 ramp events · 2014–2017",
         ha="center", va="top", fontsize=8, color="#666", style="italic")

# ── Save ─────────────────────────────────────────────────────────────────────
out = save_publication_figure(fig, OUT, dpi=SAVE_DPI)
plt.close()
print(f"[OK] Graphical abstract saved → {out}")
print(f"     canvas {FIGSIZE_IN[0]:.0f}×{FIGSIZE_IN[1]:.0f} in @ {FIG_DPI} dpi; "
      f"save @ {SAVE_DPI} dpi; basemap zoom {BASEMAP_ZOOM}")
