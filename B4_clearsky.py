"""
B4_clearsky.py — Índice de céu claro k_i(t) via Ineichen-Perez
===============================================================
Para cada usina i, calcula:

    k_i(t) = p*_i(t) / p_cs_i(t)

onde p_cs_i(t) é a irradiância de céu claro plano inclinado (POA clearsky),
normalizada pela capacidade DC, estimada via pvlib (Ineichen-Perez).

k_i(t) é definido apenas quando p_cs_i(t) > CLEARSKY_MIN_THRESHOLD.
Valores de k_i fora de [0, K_MAX] são marcados NaN.

Entradas:
  data/interim/power_norm.parquet
  data/interim/coords.parquet

Saídas:
  data/interim/clearsky_index.parquet   — k_i(t) wide (timestamps × usinas)
  data/interim/clearsky_report.csv      — cobertura de k_i por usina

Executar:
    python B4_clearsky.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg

IN_NORM   = cfg.DIRS["interim"] / "power_norm.parquet"
COORDS    = cfg.DIRS["interim"] / "coords.parquet"
OUT_PQ    = cfg.DIRS["interim"] / "clearsky_index.parquet"
OUT_REP   = cfg.DIRS["interim"] / "clearsky_report.csv"

# Limiar mínimo de irradiância clearsky (fração da DC_cap) para definir k
CLEARSKY_MIN = cfg.QC.get("clearsky_min_threshold", 0.005)
K_MAX        = cfg.QC.get("k_max", 1.5)

# Altitude de Utrecht (metros acima do nível do mar)
ALTITUDE_M = 5.0

SEP = "─" * 60


def _clearsky_poa_norm(lat: float, lon: float, tilt: float, azimuth: float,
                       times: pd.DatetimeIndex) -> pd.Series:
    """
    Calcula GHI clearsky via Ineichen com Linke Turbidity fixo.

    Passa linke_turbidity=3.0 diretamente para evitar o lookup geográfico
    (que falha quando lat/lon contém NaN).  TL=3.0 é valor típico para
    Utrecht (clima marítimo temperado — verão ≈ 3.5, inverno ≈ 2.5).
    """
    loc = Location(lat, lon, tz="UTC", altitude=ALTITUDE_M)
    cs  = loc.get_clearsky(times, model="ineichen", linke_turbidity=3.0)
    return cs["ghi"]   # W/m²


def main() -> None:
    print(SEP)
    print("B4 — ÍNDICE DE CÉU CLARO k_i(t)  [Ineichen-Perez]")
    print(SEP)

    for p in [IN_NORM, COORDS]:
        if not p.exists():
            print(f"ERRO: {p} não encontrado.")
            sys.exit(1)

    df_norm = pd.read_parquet(IN_NORM)
    coords  = pd.read_parquet(COORDS).set_index("station_id")
    times   = df_norm.index

    print(f"  Dados: {df_norm.shape}  (timestamps × usinas)")
    print(f"  Calculando clearsky GHI para {len(coords)} usinas…")
    print("  (Ineichen-Perez, altitude=5 m, UTC)")

    # Como as usinas estão em um raio de ~10 km, a variação do clearsky
    # entre elas é mínima. Calculamos por usina mesmo assim (lat/lon próprias).
    df_k = pd.DataFrame(index=times, columns=df_norm.columns, dtype=float)
    report_rows = []

    for i, sid in enumerate(df_norm.columns, 1):
        if sid not in coords.index:
            df_k[sid] = np.nan
            continue

        lat = coords.loc[sid, "lat_centroid"]
        lon = coords.loc[sid, "lon_centroid"]

        if np.isnan(lat) or np.isnan(lon):
            print(f"  AVISO: {sid} com lat/lon NaN — pulando")
            df_k[sid] = np.nan
            continue

        # GHI clearsky em W/m²
        ghi_cs = _clearsky_poa_norm(lat, lon, tilt=None, azimuth=None, times=times)

        # p_cs normalizado: GHI_clearsky / 1000 W/m² (STC) como proxy de p*
        p_cs   = ghi_cs.values / 1000.0    # adimensional, mesma escala de p*
        p_star = df_norm[sid].values

        # k_i — suppress divide-by-zero warning (np.where evaluates both branches)
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(p_cs > CLEARSKY_MIN, p_star / p_cs, np.nan)
        k = np.where((k < 0) | (k > K_MAX), np.nan, k)
        df_k[sid] = k

        n_valid = int(np.isfinite(k).sum())
        report_rows.append({
            "station_id": sid,
            "n_valid_k":  n_valid,
            "k_median":   float(np.nanmedian(k)),
            "k_p95":      float(np.nanpercentile(k, 95)),
        })

        if i % 25 == 0 or i == len(df_norm.columns):
            print(f"  [{i:3d}/{len(df_norm.columns)}]  {sid}  "
                  f"k_median={report_rows[-1]['k_median']:.3f}")

    df_k = df_k.astype(float)
    report = pd.DataFrame(report_rows)

    print(f"\n  k_i global: mediana={float(np.nanmedian(df_k.values)):.3f}  "
          f"P95={float(np.nanpercentile(df_k.values[np.isfinite(df_k.values)], 95)):.3f}")

    df_k.to_parquet(OUT_PQ)
    print(f"  Salvo: {OUT_PQ.relative_to(cfg.ROOT)}  ({OUT_PQ.stat().st_size/1e6:.0f} MB)")

    report.to_csv(OUT_REP, index=False)
    print(f"  Salvo: {OUT_REP.relative_to(cfg.ROOT)}")

    print(f"\n{SEP}")
    print("B4 concluído — clearsky_index.parquet disponível.")


if __name__ == "__main__":
    main()
