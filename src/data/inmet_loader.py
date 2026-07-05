"""
INMET (BDMEP) station data loader and downloader.

Responsibilities:
- List automatic weather stations via the BDMEP REST API
- Download hourly data for one or many stations over a date range
- Load already-downloaded Parquet/CSV files
- Standardise column names and parse timestamps
"""
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

BDMEP_BASE = "https://apitempo.inmet.gov.br"

# Raw INMET column names → standardised names
INMET_COL_RENAME = {
    "DT_MEDICAO": "date",
    "HR_MEDICAO": "hour",
    "VEN_VEL":    "wind_speed_ms",
    "VEN_DIR":    "wind_direction_deg",
    "RAD_GLO":    "ghi_wm2",
    "TEM_INS":    "temp_c",
    "UMD_INS":    "humidity_pct",
    "CD_ESTACAO": "station_code",
    "DC_NOME":    "station_name",
    "VL_LATITUDE": "latitude",
    "VL_LONGITUDE": "longitude",
    "VL_ALTITUDE": "altitude_m",
}


# ---------------------------------------------------------------------------
# BDMEP API
# ---------------------------------------------------------------------------

def get_stations(station_type: str = "T") -> pd.DataFrame:
    """
    Return a DataFrame listing all stations of the given type.

    Parameters
    ----------
    station_type : 'T' for automatic, 'M' for conventional
    """
    url = f"{BDMEP_BASE}/estacoes/{station_type}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"INMET API request failed: {e}") from e

    stations = pd.DataFrame(resp.json()).rename(columns={
        "CD_ESTACAO":   "station_code",
        "DC_NOME":      "station_name",
        "SG_ESTADO":    "state",
        "VL_LATITUDE":  "latitude",
        "VL_LONGITUDE": "longitude",
        "VL_ALTITUDE":  "altitude_m",
    })
    for col in ["latitude", "longitude", "altitude_m"]:
        if col in stations.columns:
            stations[col] = pd.to_numeric(stations[col], errors="coerce")

    logger.info(f"INMET: {len(stations)} type-{station_type} stations available")
    return stations


def download_station(station_code: str, start: str, end: str,
                      out_dir: str | Path) -> Path:
    """
    Download hourly data for a single station and cache it as Parquet.

    Parameters
    ----------
    station_code : station identifier (e.g. 'A001')
    start / end  : date strings in 'YYYY-MM-DD' format
    out_dir      : output directory

    Returns
    -------
    Path to the saved Parquet file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{station_code}.parquet"

    if out_path.exists():
        logger.debug(f"Cache hit: {out_path} — skipping download")
        return out_path

    url = f"{BDMEP_BASE}/estacao/{start}/{end}/{station_code}"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download station {station_code}: {e}") from e

    data = resp.json()
    if not data:
        logger.warning(f"Station {station_code}: empty response for {start}→{end}")
        return out_path

    pd.DataFrame(data).to_parquet(out_path, index=False)
    logger.info(f"Saved: {out_path} ({len(data):,} rows)")
    return out_path


def download_all_stations(station_codes: list[str], start: str, end: str,
                           out_dir: str | Path) -> list[Path]:
    """Download data for a list of stations with a progress bar."""
    from tqdm import tqdm

    paths = []
    for code in tqdm(station_codes, desc="Downloading INMET stations"):
        try:
            paths.append(download_station(code, start, end, out_dir))
        except RuntimeError as e:
            logger.error(str(e))
    return paths


# ---------------------------------------------------------------------------
# Loading and standardisation
# ---------------------------------------------------------------------------

def load_station(path: str | Path) -> pd.DataFrame:
    """
    Load a Parquet or CSV station file and standardise columns.

    Returns
    -------
    DataFrame with a parsed 'timestamp' column (hourly, timezone-naive)
    and numeric meteorological variables.
    """
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df = df.rename(columns=INMET_COL_RENAME)

    # Build timestamp from separate date and hour columns
    if "date" in df.columns and "hour" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " "
            + df["hour"].astype(str).str.zfill(4),
            format="%Y-%m-%d %H%M",
            errors="coerce",
        )
        df = df.drop(columns=["date", "hour"], errors="ignore")
    elif "timestamp" not in df.columns:
        logger.warning(f"No time column identified in {path.name}")

    for col in ["wind_speed_ms", "wind_direction_deg", "ghi_wm2",
                "temp_c", "humidity_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("timestamp").reset_index(drop=True)


def load_all_stations(inmet_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load all Parquet/CSV files in the INMET directory.

    Returns
    -------
    Dictionary mapping station_code → DataFrame.
    """
    inmet_dir = Path(inmet_dir)
    files = list(inmet_dir.glob("*.parquet")) + list(inmet_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No files found in {inmet_dir.resolve()}.\n"
            "Run download_all_stations() first."
        )

    stations: dict[str, pd.DataFrame] = {}
    for f in sorted(files):
        stations[f.stem] = load_station(f)

    logger.info(f"INMET: {len(stations)} station(s) loaded")
    return stations


def inspect_station(df: pd.DataFrame) -> dict:
    """Return a diagnostic summary for one INMET station (used in EDA notebook)."""
    return {
        "n_rows": len(df),
        "columns": df.columns.tolist(),
        "period_start": df["timestamp"].min() if "timestamp" in df.columns else None,
        "period_end":   df["timestamp"].max() if "timestamp" in df.columns else None,
        "missing_pct":  (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }
