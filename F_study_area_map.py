"""
F_study_area_map.py
===================
Generates two versions of the study-area map (Fig. 1):

  1. paper/figures/fig01_study_area.html  -- interactive Leaflet/Folium
  2. paper/figures/fig01_study_area.pdf   -- static vector PDF (for LaTeX)

Data source: data/interim/coords.parquet
  columns: lat_centroid, lon_centroid, capacity_dc_kwp, station_id
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import folium
from folium.plugins import MiniMap
from shapely.geometry import Point, box as shpbox
import pyproj
import geodatasets

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from src.paper_figures import save_publication_figure

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).parent
COORDS   = ROOT / "data/interim/coords.parquet"
OUT_DIR  = ROOT / "paper/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_OUT = OUT_DIR / "fig01_study_area.html"
PDF_OUT  = OUT_DIR / "fig01_study_area.pdf"

# ── Load coordinates ─────────────────────────────────────────────────────────
df = pd.read_parquet(COORDS, columns=["station_id", "lat_centroid",
                                       "lon_centroid", "capacity_dc_kwp"])
df = df.dropna(subset=["lat_centroid", "lon_centroid"])
n  = len(df)
print(f"Loaded {n} stations | "
      f"lat [{df.lat_centroid.min():.3f}, {df.lat_centroid.max():.3f}] "
      f"lon [{df.lon_centroid.min():.3f}, {df.lon_centroid.max():.3f}]")

center_lat = df.lat_centroid.mean()
center_lon = df.lon_centroid.mean()

# ── Colour scale by capacity ─────────────────────────────────────────────────
cap     = df.capacity_dc_kwp
cap_min = cap.quantile(0.02)
cap_max = cap.quantile(0.98)
norm    = plt.Normalize(vmin=cap_min, vmax=cap_max)
cmap    = plt.cm.plasma

def cap_to_hex(c):
    rgba = cmap(norm(np.clip(c, cap_min, cap_max)))
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

# ═══════════════════════════════════════════════════════════════════════════════
# 1. INTERACTIVE HTML (Folium / Leaflet)
# ═══════════════════════════════════════════════════════════════════════════════
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="OpenStreetMap",
    control_scale=True,
)
MiniMap(tile_layer="OpenStreetMap", toggle_display=True,
        zoom_level_offset=-6).add_to(m)
folium.TileLayer("CartoDB positron", name="CartoDB Light").add_to(m)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
folium.LayerControl().add_to(m)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row.lat_centroid, row.lon_centroid],
        radius=5,
        color="white", weight=0.5,
        fill=True, fill_color=cap_to_hex(row.capacity_dc_kwp),
        fill_opacity=0.85,
        tooltip=folium.Tooltip(
            f"<b>{row.station_id}</b><br>"
            f"Lat: {row.lat_centroid:.4f} | Lon: {row.lon_centroid:.4f}<br>"
            f"Capacity: {row.capacity_dc_kwp:.2f} kWp"
        ),
    ).add_to(m)

bounds = [[df.lat_centroid.min(), df.lon_centroid.min()],
          [df.lat_centroid.max(), df.lon_centroid.max()]]
folium.Rectangle(bounds=bounds, color="#e63946", weight=1.5,
                 fill=False, dash_array="6 4",
                 tooltip="Study area extent").add_to(m)

legend_html = """
<div style="position:fixed;bottom:30px;left:30px;z-index:9999;
            background:rgba(255,255,255,0.92);padding:10px 14px;
            border-radius:6px;box-shadow:2px 2px 6px rgba(0,0,0,.3);
            font-family:Arial,sans-serif;font-size:12px;">
  <b>Installed capacity (DC)</b><br>
  <svg width="160" height="14">
    <defs>
      <linearGradient id="lg" x1="0" x2="1" y1="0" y2="0">
        <stop offset="0%"   stop-color="#0d0887"/>
        <stop offset="50%"  stop-color="#cc4778"/>
        <stop offset="100%" stop-color="#f0f921"/>
      </linearGradient>
    </defs>
    <rect width="160" height="14" fill="url(#lg)" rx="2"/>
  </svg><br>
  <span style="float:left">{:.1f} kWp</span>
  <span style="float:right">{:.1f} kWp</span>
  <br style="clear:both">
  <hr style="margin:6px 0">
  <span>&#9679; n = {:d} PV systems</span>
</div>
""".format(cap_min, cap_max, n)
m.get_root().html.add_child(folium.Element(legend_html))
m.save(str(HTML_OUT))
print(f"[OK] HTML saved → {HTML_OUT}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. STATIC PNG (contextily + geopandas, 300 dpi)
# ═══════════════════════════════════════════════════════════════════════════════
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df.lon_centroid, df.lat_centroid)],
    crs="EPSG:4326",
).to_crs(epsg=3857)

x_vals = gdf.geometry.x
y_vals = gdf.geometry.y
pad    = 1500  # metres

fig, axes = plt.subplots(
    1, 2,
    figsize=(11, 6),
    gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
)

# ── Left panel: main map ────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(x_vals.min() - pad, x_vals.max() + pad)
ax.set_ylim(y_vals.min() - pad, y_vals.max() + pad)

try:
    ctx.add_basemap(ax, crs=gdf.crs,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    zoom="auto", attribution_size=6)
except Exception as e:
    print(f"  [warn] basemap fetch failed ({e})")
    ax.set_facecolor("#f0f0f0")

sc = ax.scatter(
    x_vals, y_vals,
    c=df.capacity_dc_kwp.values,
    cmap="plasma", norm=norm,
    s=28, edgecolors="white", linewidths=0.4, zorder=5,
)

# Scale bar (~5 km)
x0 = float(x_vals.min()) - pad + 1200
y0 = float(y_vals.min()) - pad + 1200
ax.plot([x0, x0 + 5000], [y0, y0], "k-", lw=2, zorder=8)
ax.text(x0 + 2500, y0 + 400, "5 km",
        ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=8)

# North arrow
ax.annotate("N", xy=(0.97, 0.12), xytext=(0.97, 0.05),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=10, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color="k"))

ax.set_title(f"Utrecht PV network — {n} systems, 2014–2017",
             fontsize=10, pad=8)
ax.set_xlabel("Longitude", fontsize=8)
ax.set_ylabel("Latitude", fontsize=8)
ax.tick_params(labelsize=7)

cbar = fig.colorbar(sc, ax=ax, orientation="vertical",
                    fraction=0.03, pad=0.02, aspect=30)
cbar.set_label("Installed capacity (DC, kWp)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# ── Right panel: Western Europe inset (Natural Earth land) ──────────────────
ax2 = axes[1]

# Load Natural Earth land polygons and crop to Western Europe
land = gpd.read_file(geodatasets.get_path("naturalearth.land")).to_crs(3857)
cities_all = gpd.read_file(
    geodatasets.get_path("naturalearth.cities")).to_crs(3857)

inset_xlim = (-300_000, 1_600_000)
inset_ylim = (6_050_000, 7_800_000)

# Clip land to inset extent
clip_geom = gpd.GeoDataFrame(
    geometry=[shpbox(*inset_xlim, *inset_ylim)], crs=3857)
land_clip = gpd.clip(land, clip_geom)
land_clip.plot(ax=ax2, color="#dde8d8", edgecolor="#aaa",
               linewidth=0.4, zorder=1)

# Netherlands bounding box (approx) highlighted
transformer = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
nl_w, nl_s = transformer.transform(3.36, 50.75)
nl_e, nl_n = transformer.transform(7.23, 53.55)
nl_box = gpd.GeoDataFrame(
    geometry=[shpbox(nl_w, nl_s, nl_e, nl_n)], crs=3857)
nl_box.plot(ax=ax2, color="#b8d4ea", edgecolor="#4477aa",
            linewidth=0.9, alpha=0.7, zorder=2)

# Study-area star marker
study_cx = float(x_vals.mean())
study_cy = float(y_vals.mean())
ax2.plot(study_cx, study_cy, marker="*", color="#e63946",
         markersize=12, zorder=6,
         markeredgecolor="white", markeredgewidth=0.5)
ax2.text(study_cx + 30_000, study_cy - 25_000, "Utrecht",
         fontsize=6.5, color="#e63946", fontweight="bold",
         va="top", zorder=7)

# Reference city dots
ref_names = ["Amsterdam", "Brussels", "Berlin", "Paris", "London", "Copenhagen"]
ref_cities = cities_all[cities_all["name"].isin(ref_names)]
for _, city in ref_cities.iterrows():
    cx, cy = city.geometry.x, city.geometry.y
    if inset_xlim[0] < cx < inset_xlim[1] and inset_ylim[0] < cy < inset_ylim[1]:
        ax2.plot(cx, cy, "o", color="#333", markersize=3, zorder=5)
        ax2.text(cx + 22_000, cy, city["name"],
                 fontsize=5.5, color="#222", va="center", zorder=5)

# Sea background (set axes background to light blue)
ax2.set_facecolor("#d0e8f5")
ax2.set_xlim(*inset_xlim)
ax2.set_ylim(*inset_ylim)
ax2.set_title("Location\n(Western Europe)", fontsize=8, pad=4)
ax2.axis("off")
# Thin frame
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.6)
    spine.set_color("#999")

fig.suptitle(
    "Figure 1. Geographic distribution of the 174 distributed PV systems "
    "in the Utrecht study area, the Netherlands.",
    fontsize=8.5, y=0.01, style="italic"
)

out = save_publication_figure(fig, PDF_OUT)
plt.close()
print(f"[OK] PDF saved → {out}")
