"""
BR-PVGen dataset loader.

Dataset structure (discovered from actual files):
  solar_station/<ps_id>.json  — 15-min meteorological + irradiance data (PRIMARY)
  inverter/<ps_id>.json       — 15-min per-inverter power data (aggregated → plant power)
  power_station_metadata.csv  — plant specs (state, nominal power, panel type) — NO lat/lon

Coordinates are NOT included in BR-PVGen.
Supply a coordinates lookup CSV (plant_coordinates.csv) or use INMET station
centroids as geographic proxies after spatial matching.

coordinate lookup format: ps_id, latitude, longitude
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
from loguru import logger

# ── Column rename maps ────────────────────────────────────────────────────────

# solar_station JSON → standard names
SOLAR_COL_RENAME: dict[str, str] = {
    "datetime":                      "timestamp",
    "ps_id":                         "plant_id",
    "poa_irradiance_wm2":            "poa_irradiance_wm2",
    "ghi_irradiance_wm2":            "ghi_wm2",
    "gri_irradiance_wm2":            "gri_wm2",
    "wind_speed_ms":                 "wind_speed_ms",
    "wind_direction_degrees":        "wind_direction_deg",
    "panel_temperature_celsius":     "panel_temp_c",
    "ambient_temperature_celsius":   "ambient_temp_c",
    "precipitation_accumulated_mm":  "precipitation_mm",
    "tracker_albedo_index":          "tracker_albedo",
}

# inverter JSON → standard names (before aggregation)
INVERTER_COL_RENAME: dict[str, str] = {
    "datetime":              "timestamp",
    "ps_id":                 "plant_id",
    "inverter_id":           "inverter_id",
    "total_active_power_w":  "active_power_w",
    "total_dc_power_w":      "dc_power_w",
    "total_reactive_power_var": "reactive_power_var",
    "internal_temperature_celsius": "inverter_temp_c",
}

# Columns to keep after loading solar_station
SOLAR_KEEP = [
    "timestamp", "plant_id",
    "poa_irradiance_wm2", "ghi_wm2", "gri_wm2",
    "wind_speed_ms", "wind_direction_deg",
    "panel_temp_c", "ambient_temp_c",
    "precipitation_mm", "tracker_albedo",
]


# ── File discovery ────────────────────────────────────────────────────────────

def list_solar_station_jsons(raw_dir: str | Path) -> list[Path]:
    """
    Return sorted list of solar_station JSON files.
    These contain the primary meteorological + irradiance 15-min data.
    """
    raw_dir = Path(raw_dir)
    # Try both possible subdirectory structures
    candidates = list(raw_dir.rglob("solar_station/*.json"))
    logger.info(f"BR-PVGen: {len(candidates)} solar_station JSON(s) found in {raw_dir}")
    return sorted(candidates)


def list_inverter_jsons(raw_dir: str | Path) -> list[Path]:
    """Return sorted list of inverter JSON files."""
    raw_dir = Path(raw_dir)
    candidates = list(raw_dir.rglob("inverter/*.json"))
    logger.info(f"BR-PVGen: {len(candidates)} inverter JSON(s) found in {raw_dir}")
    return sorted(candidates)


# kept for backward compat — returns CSVs (inverter-level)
def list_plant_files(raw_dir: str | Path) -> list[Path]:
    """Return sorted list of CSV files (inverter level). Prefer list_solar_station_jsons()."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.rglob("*.csv"))
    # Exclude metadata CSV
    files = [f for f in files if "metadata" not in f.name.lower()]
    logger.info(f"BR-PVGen: {len(files)} inverter CSV(s) found in {raw_dir}")
    return files


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_solar_station(path: str | Path) -> pd.DataFrame:
    """
    Load a single solar_station JSON file.

    Returns a DataFrame with standardised column names, a UTC-parsed
    'timestamp' column, and numeric meteorological variables.
    Rows where all irradiance/weather values are null are dropped.
    """
    path = Path(path)
    with open(path) as f:
        records = json.load(f)

    df = pd.DataFrame(records)

    # Drop the nested document_count dict column if present
    df = df.drop(columns=[c for c in df.columns if c == "document_count"], errors="ignore")

    df = df.rename(columns=SOLAR_COL_RENAME)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)   # strip UTC, keep naive
    else:
        logger.warning(f"No 'datetime' column in {path.name}")

    # Cast numeric columns
    numeric_cols = [c for c in SOLAR_KEEP if c not in ("timestamp", "plant_id")]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only standard columns that exist
    keep = [c for c in SOLAR_KEEP if c in df.columns]
    df = df[keep].copy()

    logger.debug(f"Loaded solar_station: {path.name} — {len(df):,} rows")
    return df


def load_inverter_power(path: str | Path) -> pd.DataFrame:
    """
    Load a single inverter JSON file and aggregate to plant level.

    Multiple inverters per timestamp are summed → one row per (timestamp, plant_id).
    Returns columns: timestamp, plant_id, active_power_kw, dc_power_kw, reactive_power_kvar
    """
    path = Path(path)
    with open(path) as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df = df.drop(columns=[c for c in df.columns if c == "document_count"], errors="ignore")
    df = df.rename(columns=INVERTER_COL_RENAME)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    for col in ["active_power_w", "dc_power_w", "reactive_power_var"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate inverters → plant level
    agg = (
        df.groupby(["timestamp", "plant_id"], sort=False)
        .agg(
            active_power_kw=("active_power_w",   lambda x: x.sum(min_count=1) / 1000),
            dc_power_kw=    ("dc_power_w",        lambda x: x.sum(min_count=1) / 1000),
            reactive_power_kvar=("reactive_power_var", lambda x: x.sum(min_count=1) / 1000),
        )
        .reset_index()
        .sort_values("timestamp")
    )
    logger.debug(f"Loaded inverter: {path.name} — {len(agg):,} plant-level rows")
    return agg


def load_plant(plant_id: str, raw_dir: str | Path) -> pd.DataFrame:
    """
    Load and merge solar_station (weather/irradiance) + inverter (power)
    data for a single plant.

    Returns a merged 15-min DataFrame.
    """
    raw_dir = Path(raw_dir)

    # Find the matching files
    solar_files = list(raw_dir.rglob(f"solar_station/{plant_id}.json"))
    inv_files   = list(raw_dir.rglob(f"inverter/{plant_id}.json"))

    if not solar_files:
        raise FileNotFoundError(f"solar_station/{plant_id}.json not found in {raw_dir}")

    solar_df = load_solar_station(solar_files[0])

    if inv_files:
        inv_df = load_inverter_power(inv_files[0])
        df = solar_df.merge(inv_df, on=["timestamp", "plant_id"], how="left")
    else:
        logger.warning(f"{plant_id}: no inverter JSON found — power columns will be missing")
        df = solar_df

    return df.sort_values("timestamp").reset_index(drop=True)


def load_all_plants(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load all solar_station + inverter data for every plant found.

    Returns
    -------
    dict mapping plant_id → merged DataFrame
    """
    raw_dir = Path(raw_dir)
    solar_files = list_solar_station_jsons(raw_dir)

    if not solar_files:
        raise FileNotFoundError(
            f"No solar_station JSONs found in {raw_dir.resolve()}.\n"
            "Download from https://www.kaggle.com/datasets/tecsci/brazilian-pv-dataset\n"
            "and extract into data/raw/brpvgen/"
        )

    plants: dict[str, pd.DataFrame] = {}
    for sf in solar_files:
        plant_id = sf.stem   # e.g. "PS_001"
        try:
            plants[plant_id] = load_plant(plant_id, raw_dir)
        except Exception as e:
            logger.error(f"Failed to load {plant_id}: {e}")

    logger.info(f"BR-PVGen: {len(plants)} plant(s) loaded")
    return plants


# ── Metadata ──────────────────────────────────────────────────────────────────

def load_station_metadata(raw_dir: str | Path) -> pd.DataFrame:
    """
    Load power_station_metadata.csv.

    NOTE: The BR-PVGen dataset does NOT include geographic coordinates.
    Latitude/longitude columns will be missing (all NaN) unless a
    coordinates lookup file is provided via enrich_with_coordinates().
    """
    raw_dir = Path(raw_dir)
    meta_files = list(raw_dir.rglob("power_station_metadata.csv"))
    if not meta_files:
        raise FileNotFoundError("power_station_metadata.csv not found.")

    meta = pd.read_csv(meta_files[0]).rename(columns={"id": "plant_id"})
    # Coordinates are not in the dataset — add empty columns for downstream code
    meta["latitude"]  = None
    meta["longitude"] = None
    logger.info(
        f"BR-PVGen metadata: {len(meta)} plants loaded. "
        "⚠ Coordinates NOT available in dataset — use enrich_with_coordinates()."
    )
    return meta


def enrich_with_coordinates(meta: pd.DataFrame,
                             coords_path: str | Path) -> pd.DataFrame:
    """
    Merge plant metadata with an external coordinates lookup CSV.

    The lookup file must have columns: plant_id, latitude, longitude
    Example path: data/interim/plant_coordinates.csv

    This file must be created manually (or via geocoding) since the
    BR-PVGen dataset does not publish exact plant locations.
    """
    coords_path = Path(coords_path)
    if not coords_path.exists():
        logger.warning(
            f"Coordinates file not found: {coords_path}\n"
            "  Plants will have lat=None, lon=None.\n"
            "  Create the file with columns: plant_id, latitude, longitude"
        )
        return meta

    coords = pd.read_csv(coords_path)[["plant_id", "latitude", "longitude"]]
    meta = meta.drop(columns=["latitude", "longitude"], errors="ignore")
    enriched = meta.merge(coords, on="plant_id", how="left")
    n_with = enriched["latitude"].notna().sum()
    logger.info(f"Coordinates enriched: {n_with}/{len(enriched)} plants have lat/lon")
    return enriched


def extract_plants_metadata(plants: dict[str, pd.DataFrame],
                             raw_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Build a one-row-per-plant metadata DataFrame.

    If raw_dir is provided, also loads power_station_metadata.csv specs.
    Latitude/longitude will be None unless enrich_with_coordinates() is called.
    """
    records = []
    for plant_id, df in plants.items():
        rec = {
            "plant_id":  plant_id,
            "latitude":  None,
            "longitude": None,
            "n_rows":    len(df),
            "period_start": df["timestamp"].min() if "timestamp" in df.columns else None,
            "period_end":   df["timestamp"].max() if "timestamp" in df.columns else None,
        }
        records.append(rec)

    meta = pd.DataFrame(records)

    if raw_dir is not None:
        try:
            station_meta = load_station_metadata(raw_dir)
            meta = meta.merge(station_meta.drop(columns=["latitude", "longitude"]),
                              on="plant_id", how="left")
        except FileNotFoundError as e:
            logger.warning(str(e))

    n_missing = meta["latitude"].isna().sum()
    if n_missing:
        logger.warning(
            f"{n_missing}/{len(meta)} plants have no coordinates. "
            "Create data/interim/plant_coordinates.csv and call enrich_with_coordinates()."
        )
    return meta


def inspect_plant(df: pd.DataFrame) -> dict:
    """Return a diagnostic summary of a plant DataFrame (for EDA)."""
    return {
        "n_rows":        len(df),
        "columns":       df.columns.tolist(),
        "period_start":  df["timestamp"].min() if "timestamp" in df.columns else None,
        "period_end":    df["timestamp"].max() if "timestamp" in df.columns else None,
        "freq_inferred": (
            pd.infer_freq(df["timestamp"].dropna()) if "timestamp" in df.columns else None
        ),
        "missing_pct":   (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "dtypes":        df.dtypes.astype(str).to_dict(),
    }
