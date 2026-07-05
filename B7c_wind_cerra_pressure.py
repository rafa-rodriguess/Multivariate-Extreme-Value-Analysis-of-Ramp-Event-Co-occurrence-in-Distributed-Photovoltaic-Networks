"""
B7c_wind_cerra_pressure.py — Vento em altura de NUVEM (CERRA, níveis de pressão)
================================================================================
Complementa `B7b_wind_cerra.py` (CERRA níveis de altura, 100/200/500m — cobre só
a camada limite) com vento em alturas que realmente cobrem a base de nuvens
BAIXAS/MÉDIAS/ALTAS, usando o dataset `reanalysis-cerra-pressure-levels`
(29 níveis de 1000 a 1 hPa, mesma reanálise de 5,5km já usada em B7b).

Motivação (ver ROADMAP, discussão F6d): F6/F6b já testaram advecção com vento de
10-500m e deram resultado nulo em todas as alturas. Mas essa faixa cobre só a
camada limite — nem chega perto da base real da maioria das nuvens:

    Categoria       Altitude da base   Exemplos
    Nuvens baixas   até 2.000 m        Stratus, Stratocumulus, Nimbostratus
    Nuvens médias   2.000-6.000 m      Altocumulus, Altostratus
    Nuvens altas    acima de 6.000 m   Cirrus, Cirrocumulus, Cirrostratus

Níveis baixados (hPa -> altitude aprox., atmosfera padrão):
    950->540m, 900->990m, 850->1460m, 800->1950m   (baixas)
    700->3010m, 600->4200m, 500->5570m              (médias)
    400->7190m                                       (alta, limite inferior)

Diferença técnica vs. B7b: o dataset de níveis de PRESSÃO fornece as
componentes u/v do vento (não velocidade/direção diretas como o de níveis de
ALTURA) — este script calcula speed=sqrt(u²+v²) e
dir=(180+atan2(u,v))°%360 (convenção meteorológica: direção DE ONDE vem o vento).

Mesmo cadastro/credencial CDS já usado por B7b -- requer aceitar a licença do
dataset `reanalysis-cerra-pressure-levels` separadamente (mesma conta, licença
por-dataset): https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-pressure-levels

Entradas:
  data/interim/wind_joined.parquet   (gerado por B7_wind_join.py / B7b_wind_cerra.py)

Saída:
  data/raw/wind/cerra_pressure_by_year/cerra_p_<ano>_<mm>-<mm>.nc  (cache trimestral)
  data/interim/wind_joined.parquet   (colunas cerra_wind_speed_ms_{P}hPa /
                                       cerra_wind_dir_deg_{P}hPa adicionadas)

Executar:
    python B7c_wind_cerra_pressure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result

LEVELS_HPA = cfg.F6D["cerra_pressure_hpa"]          # [950,900,850,800,700,600,500,400]
ALT_BY_HPA = cfg.F6D["cerra_pressure_alt_m"]
YEARS = ["2013", "2014", "2015", "2016", "2017", "2018"]   # margem de 1 ano nas pontas

AREA = [52.45, 4.70, 51.75, 5.70]                    # (N, W, S, E), igual B7b
NETWORK_CENTROID = (52.0927, 5.2114)
GRID_DEG = [0.05, 0.05]

CDSAPIRC = Path.home() / ".cdsapirc"
CACHE_DIR = cfg.DIRS["raw_wind"] / "cerra_pressure_by_year"
WIND_PQ  = cfg.DIRS["interim"] / "wind_joined.parquet"

MERGE_TOLERANCE = pd.Timedelta("100min")   # mesma tolerância de B7b (passo 3h)

# Um pedido por MÊS (não por trimestre como em B7b): com 8 níveis x 2 variáveis
# (vs. 3 alturas x 2 variáveis em B7b), um trimestre inteiro excede o limite de
# custo da API CDS (confirmado empiricamente: 403 "cost limits exceeded" com
# 3*31*8*8*2=11.904 campos/pedido). Por mês: 1*31*8*8*2=3.968 campos -- dentro
# do teto confirmado em B7b (3*31*8*3*2=4.464 funcionou).
MONTHS_PER_YEAR = [f"{m:02d}" for m in range(1, 13)]
QUARTERS: list[tuple[str, list[str]]] = [
    (year, [month])
    for year in YEARS
    for month in MONTHS_PER_YEAR
]

SEP = "─" * 60


def _cerra_request_for_quarter(year: str, months: list[str]) -> dict:
    """Um pedido por MÊS (ver nota acima sobre o particionamento)."""
    return {
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": [str(p) for p in LEVELS_HPA],
        "product_type": ["analysis"],
        "data_type": ["reanalysis"],
        "year": [year],
        "month": months,
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
        "area": AREA,
        "grid": GRID_DEG,
        "data_format": "netcdf",
    }


def _print_registration_instructions() -> None:
    print(f"""
  AVISO: {CDSAPIRC} não encontrado — cadastro CDS pendente.

  Este é um APRIMORAMENTO OPCIONAL (vento em altura de nuvem, F6d). Requer a
  MESMA conta CDS já usada por B7b, mas com a licença de um dataset adicional:

    1. Login em https://cds.climate.copernicus.eu/
    2. Abrir https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-pressure-levels
       e ACEITAR A LICENÇA do dataset (separada da de níveis de altura)
    3. Confirmar que {CDSAPIRC} já existe com a chave (criado em B7b)
    4. Rodar novamente: python B7c_wind_cerra_pressure.py

  B7c será pulado por enquanto (saída sem erro).
""")


def _download_cerra_pressure() -> list[Path] | None:
    try:
        import cdsapi
    except ImportError:
        print("  AVISO: pacote 'cdsapi' não instalado (pip install cdsapi). Pulando.")
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    paths = []
    for i, (year, months) in enumerate(QUARTERS, 1):
        tag = f"{year}-{months[0]}"
        out_path = CACHE_DIR / f"cerra_p_{year}_{months[0]}.nc"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  [{i}/{len(QUARTERS)}] {tag}: cache já existe, pulando download.")
            paths.append(out_path)
            continue
        print(f"  [{i}/{len(QUARTERS)}] Requisição CDS {tag} (meses {months}, níveis {LEVELS_HPA} hPa)...")
        print("  AVISO: pode ficar em fila ('Accepted') por até ~1h no CDS -- normal, idempotente entre execuções.")
        try:
            client.retrieve("reanalysis-cerra-pressure-levels", _cerra_request_for_quarter(year, months), str(out_path))
        except Exception as e:
            print(f"  ERRO no download CERRA pressão ({tag}): {type(e).__name__}: {e}")
            print("  Se for erro de parâmetro inválido, regenere a requisição via a interface web "
                  "do dataset (botão 'Show API request') e ajuste este script.")
            if paths:
                print(f"  Prosseguindo com os {len(paths)} trimestre(s) já baixado(s); rode de novo "
                      "mais tarde para retomar de onde parou.")
                break
            return None
        print(f"  Cache salvo: {out_path.relative_to(cfg.ROOT)}")
        paths.append(out_path)
    return paths if paths else None


def _extract_nearest_point_series(nc_paths: list[Path]) -> pd.DataFrame:
    """Abre os NetCDF (um por trimestre), concatena no tempo, extrai a célula de
    grade mais próxima do centroide da rede para cada nível de pressão, calcula
    speed/direction a partir de u/v, e retorna colunas
    cerra_wind_{speed,dir}_{P}hPa indexadas por timestamp UTC."""
    import xarray as xr

    if len(nc_paths) > 1:
        datasets = [xr.open_dataset(str(p)) for p in nc_paths]
        time_dim = "valid_time" if "valid_time" in datasets[0].dims else "time"
        ds = xr.concat(datasets, dim=time_dim).sortby(time_dim)
    else:
        ds = xr.open_dataset(nc_paths[0])
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    point = ds.sel({lat_name: NETWORK_CENTROID[0], lon_name: NETWORK_CENTROID[1]}, method="nearest")

    time_name = "valid_time" if "valid_time" in point.coords else "time"
    level_dim = next((d for d in ("isobaricInhPa", "pressure_level", "level") if d in point.dims), None)
    if level_dim is None:
        raise RuntimeError(f"Dimensão de nível de pressão não encontrada. Dims disponíveis: {list(point.dims)}")

    u_var = next((v for v in point.data_vars if str(v).lower() in ("u", "u_component_of_wind") or str(v).lower().startswith("u1")), None)
    v_var = next((v for v in point.data_vars if str(v).lower() in ("v", "v_component_of_wind") or str(v).lower().startswith("v1")), None)
    if u_var is None or v_var is None:
        raise RuntimeError(f"Variáveis u/v não encontradas. Disponíveis: {list(point.data_vars)} "
                            f"-- inspecione manualmente e ajuste este script.")

    frames = []
    for p_hpa in LEVELS_HPA:
        try:
            sub = point.sel({level_dim: p_hpa})
        except Exception as e:
            print(f"  AVISO: não foi possível extrair {p_hpa}hPa ({e}) — verifique os valores de "
                  f"{level_dim} disponíveis: {point[level_dim].to_numpy()}")
            continue
        u = sub[u_var].to_numpy()
        v = sub[v_var].to_numpy()
        speed = np.sqrt(u ** 2 + v ** 2)
        direction = (180.0 + np.degrees(np.arctan2(u, v))) % 360.0   # convenção meteorológica (de onde vem)
        alt_m = ALT_BY_HPA.get(p_hpa)
        df_h = pd.DataFrame({
            f"cerra_wind_speed_ms_{p_hpa}hPa": speed,
            f"cerra_wind_dir_deg_{p_hpa}hPa": direction,
        }, index=pd.to_datetime(sub[time_name].to_numpy(), utc=True))
        frames.append(df_h)
        print(f"  {p_hpa}hPa (~{alt_m}m): {len(df_h):,} timestamps extraídos.")

    if not frames:
        raise RuntimeError("Nenhum nível extraído com sucesso — ver avisos acima.")
    out = pd.concat(frames, axis=1)
    return out[~out.index.duplicated(keep="first")].sort_index()


def main() -> None:
    print(SEP)
    print("B7c — VENTO EM ALTURA DE NUVEM (CERRA, níveis de pressão, 8 alturas)")
    print(SEP)

    if not WIND_PQ.exists():
        print(f"ERRO: {WIND_PQ} não encontrado. Execute B7_wind_join.py primeiro.")
        sys.exit(1)

    if not CDSAPIRC.exists():
        _print_registration_instructions()
        print(f"\n{SEP}")
        print("B7c PULADO (aprimoramento opcional, pipeline principal não afetado).")
        log_result(
            script="B7c_wind_cerra_pressure.py", gate="", phase="F6d",
            params={"dataset": "reanalysis-cerra-pressure-levels", "levels_hpa": LEVELS_HPA},
            results={"status": "skipped_no_registration"},
            decision="N/A — data acquisition step, pending user action",
            action="Skipped: ~/.cdsapirc not found.",
            interpretation="Cloud-height wind (8 pressure levels spanning low/mid/high clouds, "
                           "~540-7190m) not yet available; F6d network-implied vector can still be "
                           "computed from power data alone in the meantime.",
            paper_ref="Section 8 (F6 spatial structure) -- F6d cloud-height wind, pending",
        )
        return

    nc_paths = _download_cerra_pressure()
    if not nc_paths:
        print(f"\n{SEP}")
        print("B7c PULADO (download indisponível — ver erro acima).")
        return

    print("\n  Extraindo série no ponto de grade mais próximo do centroide da rede...")
    cerra = _extract_nearest_point_series(nc_paths)
    print(f"  Série CERRA (pressão): {cerra.shape}  [{cerra.index[0]} → {cerra.index[-1]}]")

    wind = pd.read_parquet(WIND_PQ)
    wind["start_ts"] = pd.to_datetime(wind["start_ts"], utc=True)
    cerra_cols = cerra.columns.tolist()
    drop_existing = [c for c in cerra_cols if c in wind.columns]
    if drop_existing:
        wind = wind.drop(columns=drop_existing)

    df_joined = pd.merge_asof(
        wind.sort_values("start_ts"),
        cerra.rename_axis("start_ts").reset_index(),
        on="start_ts",
        tolerance=MERGE_TOLERANCE,
        direction="nearest",
    )
    matched_counts = {}
    for p_hpa in LEVELS_HPA:
        col = f"cerra_wind_speed_ms_{p_hpa}hPa"
        if col in df_joined.columns:
            n_matched = df_joined[col].notna().sum()
            matched_counts[p_hpa] = int(n_matched)
            print(f"  {p_hpa}hPa (~{ALT_BY_HPA[p_hpa]}m): {n_matched:,} / {len(df_joined):,} eventos casados "
                  f"({n_matched/len(df_joined):.1%})")

    df_joined.to_parquet(WIND_PQ, index=False)
    print(f"\n  Atualizado: {WIND_PQ.relative_to(cfg.ROOT)} (+{len(cerra_cols)} colunas CERRA pressão)")

    print(f"\n{SEP}")
    print("B7c concluído — 8 alturas de vento cobrindo nuvens baixas/médias/altas disponíveis para F6d.")

    log_result(
        script="B7c_wind_cerra_pressure.py", gate="", phase="F6d",
        params={
            "dataset": "reanalysis-cerra-pressure-levels", "resolution_km": 5.5,
            "levels_hpa": LEVELS_HPA, "alt_m_approx": ALT_BY_HPA,
            "cloud_categories": cfg.F6D["cloud_category_by_hpa"],
            "temporal_resolution": "3-hourly (analysis)",
            "merge_tolerance_min": MERGE_TOLERANCE.total_seconds() / 60,
            "grid_point": "nearest to network centroid (single-point time series)",
        },
        results={f"n_matched_{p}hPa": n for p, n in matched_counts.items()},
        decision="N/A — data acquisition step",
        action=(
            "Added cloud-height wind (8 CERRA pressure levels, 950-400 hPa, ~540-7190m) spanning "
            "low/mid/high cloud categories, as a complement to B7/B7b's boundary-layer wind "
            "(10-500m), motivated by F6/F6b's null result at all boundary-layer heights and the "
            "observation that most shading-relevant clouds (stratus/stratocumulus) have bases "
            "above the previously tested 500m ceiling."
        ),
        interpretation=(
            "8-height cloud-level wind profile now available for F6d's network-implied cloud "
            "vector comparison; which height (if any) best matches the network-implied advection "
            "vector should be reported as an explicit finding, not assumed a priori."
        ),
        paper_ref="Section 8 (F6 spatial structure) -- F6d cloud-height wind acquisition",
    )


if __name__ == "__main__":
    main()
