"""
Geographic matching between BR-PVGen plants and INMET weather stations.

Responsibilities:
- Compute Haversine distances between every (plant, station) pair
- Return a full mapping and the nearest (primary) station per plant
- Persist and reload the mapping as CSV files
"""
from pathlib import Path

import pandas as pd
from haversine import haversine, Unit
from loguru import logger


def match_plants_to_stations(
    plants_meta: pd.DataFrame,
    stations_meta: pd.DataFrame,
    radius_km: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Associate each plant with every INMET station within ``radius_km``.

    Parameters
    ----------
    plants_meta   : DataFrame with columns plant_id, latitude, longitude
    stations_meta : DataFrame with columns station_code, latitude, longitude
    radius_km     : maximum association radius in kilometres

    Returns
    -------
    mapping : all (plant, station) pairs within the radius, sorted by distance
    primary : one row per plant — the closest station (primary association)
    """
    # Drop plants or stations with missing coordinates before matching
    plants_valid = plants_meta.dropna(subset=["latitude", "longitude"])
    stations_valid = stations_meta.dropna(subset=["latitude", "longitude"])

    n_skipped_plants = len(plants_meta) - len(plants_valid)
    n_skipped_sta = len(stations_meta) - len(stations_valid)
    if n_skipped_plants:
        logger.warning(f"Skipping {n_skipped_plants} plant(s) with missing coordinates")
    if n_skipped_sta:
        logger.warning(f"Skipping {n_skipped_sta} station(s) with missing coordinates")

    records = []
    for _, plant in plants_valid.iterrows():
        plant_coord = (float(plant["latitude"]), float(plant["longitude"]))
        for _, sta in stations_valid.iterrows():
            sta_coord = (float(sta["latitude"]), float(sta["longitude"]))
            dist = haversine(plant_coord, sta_coord, unit=Unit.KILOMETERS)
            if dist <= radius_km:
                records.append({
                    "plant_id":     plant["plant_id"],
                    "station_code": sta["station_code"],
                    "station_name": sta.get("station_name", ""),
                    "distance_km":  round(dist, 2),
                })

    if not records:
        logger.warning(
            f"No INMET station found within {radius_km} km of any plant. "
            "Consider increasing the radius."
        )
        return pd.DataFrame(), pd.DataFrame()

    mapping = (
        pd.DataFrame(records)
        .sort_values(["plant_id", "distance_km"])
        .reset_index(drop=True)
    )
    primary = (
        mapping.groupby("plant_id")
        .first()
        .reset_index()
        .rename(columns={
            "station_code": "primary_station",
            "distance_km":  "primary_dist_km",
        })
    )

    covered = primary["plant_id"].nunique()
    total   = plants_meta["plant_id"].nunique()
    logger.info(
        f"Matching: {covered}/{total} plants have at least one INMET station "
        f"within {radius_km} km"
    )
    return mapping, primary


def save_mapping(mapping: pd.DataFrame, primary: pd.DataFrame,
                  interim_dir: str | Path) -> None:
    """Save both mapping DataFrames as CSV files."""
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(interim_dir / "plant_station_mapping_all.csv",     index=False)
    primary.to_csv(interim_dir / "plant_station_mapping_primary.csv", index=False)
    logger.info(f"Mappings saved to {interim_dir}")


def load_primary_mapping(interim_dir: str | Path) -> pd.DataFrame:
    """Load the primary (closest station) mapping from disk."""
    path = Path(interim_dir) / "plant_station_mapping_primary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Mapping not found: {path}\n"
            "Run match_plants_to_stations() and save_mapping() first."
        )
    return pd.read_csv(path)


def coverage_report(plants_meta: pd.DataFrame,
                     primary: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-plant coverage report showing whether each plant has
    an associated INMET station. Useful for displaying in the notebook.

    Handles gracefully when `primary` is empty (no matches found).
    """
    base = plants_meta[["plant_id", "latitude", "longitude"]].copy()

    expected_cols = {"plant_id", "primary_station", "primary_dist_km"}
    if primary.empty or not expected_cols.issubset(primary.columns):
        # No matches — all plants are uncovered
        base["primary_station"] = None
        base["primary_dist_km"] = None
        base["has_inmet"] = False
        return base

    report = base.merge(
        primary[["plant_id", "primary_station", "primary_dist_km"]],
        on="plant_id",
        how="left",
    )
    report["has_inmet"] = report["primary_station"].notna()
    return report
