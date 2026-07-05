"""
B7b_wind_cerra.py — Vento em altura (CERRA, 5,5 km) para análise de anisotropia
================================================================================
Complementa `B7_wind_join.py` (vento de SUPERFÍCIE, 10 m, KNMI De Bilt) com
vento em ALTURAS REAIS via CERRA (Copernicus European Regional ReAnalysis) —
a reanálise europeia de maior resolução horizontal disponível abertamente.

Por que CERRA e não ERA5:
  A rede Utrecht tem ~44 km (lat) × 50 km (lon) de extensão. A resolução do
  ERA5 (31 km) cobriria essa área em apenas ~1,4 × 1,6 células de grade —
  essencialmente o MESMO ponto de reanálise para toda a rede, insuficiente
  até para uma checagem básica de robustez espacial. A CERRA (5,5 km) cobre
  a mesma área em ~8,0 × 9,2 células — uma melhoria de ~6× na resolução.

Alturas baixadas: **100 m, 200 m, 500 m** (dataset `reanalysis-cerra-height-levels`,
alturas disponíveis: 15-500 m, desenhado para aplicações de energia eólica).
Combinadas com os 10 m de superfície já obtidos em B7, dão um PERFIL VERTICAL
de 4 alturas — a comparação entre elas (qual explica melhor a anisotropia
observada em F6) deve ser reportada no paper como achado metodológico, não
assumida a priori.

Trade-off honesto (documentar no paper):
  CERRA ganha ~6× em resolução espacial mas é publicada a cada 3 HORAS (não
  horária como KNMI/ERA5). Para o teste de anisotropia (direção do vento
  AMBIENTE que advecta nuvens) isso é aceitável — a direção do vento sinótico
  raramente inverte em menos de 3h, exceto passagem de frente — mas o
  casamento usa tolerância mais larga (100 min) e isso deve ser reportado.

Pré-requisito — cadastro CDS (gratuito, único, mesmo usado por qualquer
dataset Copernicus, ex. ERA5):
  1. Criar conta em https://cds.climate.copernicus.eu/
  2. Login → clicar no nome (canto superior direito) → copiar o
     "Personal Access Token"
  3. Acessar a página do dataset e ACEITAR A LICENÇA (obrigatório, é uma
     aceitação por dataset, separada da conta):
     https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-height-levels
  4. Criar o arquivo `~/.cdsapirc` com:
       url: https://cds.climate.copernicus.eu/api
       key: <SEU_TOKEN_AQUI>
  5. `pip install cdsapi xarray netcdf4` (já incluído em requirements.txt)

Se `~/.cdsapirc` não existir, este script imprime estas instruções e
encerra sem erro (não bloqueia o pipeline — é um aprimoramento opcional).

Nota sobre nomes de campo da API: a CDS recomenda fortemente construir a
requisição pela interface web do dataset (botão "Show API request") em vez
de adivinhar nomes de campo, pois eles mudam entre datasets/versões. Os
valores abaixo (`variable`, `height_level`, `product_type`) foram
confirmados em exemplo oficial da documentação CERRA no momento da escrita
deste script — se a requisição falhar por parâmetro inválido, regenere via
essa interface e ajuste a função `_cerra_request_for_year` abaixo.

Entradas:
  data/interim/ramps.parquet
  data/interim/wind_joined.parquet   (gerado por B7_wind_join.py — será estendido)

Saída:
  data/raw/wind/cerra_utrecht_100_200_500m.nc   (cache do download bruto)
  data/interim/wind_joined.parquet   (colunas cerra_wind_speed_ms_{H}m /
                                       cerra_wind_dir_deg_{H}m adicionadas)

Executar:
    python B7b_wind_cerra.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result

HEIGHTS_M = [100, 200, 500]
YEARS = ["2013", "2014", "2015", "2016", "2017", "2018"]   # margem de 1 ano nas pontas

# Bounding box com margem generosa em torno da rede Utrecht (N, W, S, E)
AREA = [52.45, 4.70, 51.75, 5.70]
NETWORK_CENTROID = (52.0927, 5.2114)   # (lat, lon) — ponto único extraído para o covariável principal

CDSAPIRC = Path.home() / ".cdsapirc"
CACHE_DIR = cfg.DIRS["raw_wind"] / "cerra_by_year"
RAMPS_PQ = cfg.DIRS["interim"] / "ramps.parquet"
WIND_PQ  = cfg.DIRS["interim"] / "wind_joined.parquet"

MERGE_TOLERANCE = pd.Timedelta("100min")   # metade do passo de 3h + folga


GRID_DEG = [0.05, 0.05]   # regrade para lat/lon regular (ver nota abaixo)

# Trimestres (ano, [meses]) cobrindo 2013-2018 (margem de 1 ano em cada ponta de 2014-2017).
QUARTERS: list[tuple[str, list[str]]] = [
    (year, months)
    for year in YEARS
    for months in (["01", "02", "03"], ["04", "05", "06"], ["07", "08", "09"], ["10", "11", "12"])
]


def _cerra_request_for_quarter(year: str, months: list[str]) -> dict:
    """Uma requisição por TRIMESTRE. Duas descobertas do CDS motivam este
    desenho (2026-07-01):
    (1) o limite de 'custo' da API rejeita (HTTP 403 'cost limits exceeded')
        requisições grandes demais em número de campos (ano*mes*dia*hora*
        altura*variável) -- um ANO inteiro (12*31*8*3*2 ~= 17,9 mil campos)
        excede o teto; um TRIMESTRE (3*31*8*3*2 ~= 4,5 mil campos) fica
        dentro do limite (confirmado empiricamente).
    (2) a grade nativa da CERRA é uma projeção cônica de Lambert -- pedir só
        `area` (sem `grid`) faz o MIR (motor de interpolação do CDS) falhar
        com 'Serious bug: Representation::croppedRepresentation() not
        implemented for RegularGrid'. Adicionar `grid` (regrade para
        lat/lon regular ANTES do recorte) contorna o bug."""
    return {
        "variable": ["wind_speed", "wind_direction"],
        "height_level": [f"{h}_m" for h in HEIGHTS_M],
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

SEP = "─" * 60


def _print_registration_instructions() -> None:
    print(f"""
  AVISO: {CDSAPIRC} não encontrado — cadastro CDS pendente.

  Este é um APRIMORAMENTO OPCIONAL (vento em altura). O pipeline principal
  já funciona com o vento de superfície de B7_wind_join.py. Para habilitar
  este script:

    1. Criar conta gratuita em https://cds.climate.copernicus.eu/
    2. Login → clicar no nome (canto superior direito) → copiar o
       "Personal Access Token"
    3. Abrir https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-height-levels
       e ACEITAR A LICENÇA do dataset (obrigatório, separado da conta)
    4. Criar o arquivo {CDSAPIRC} com o conteúdo:
           url: https://cds.climate.copernicus.eu/api
           key: <SEU_TOKEN_AQUI>
    5. Rodar novamente: python B7b_wind_cerra.py

  B7b será pulado por enquanto (saída sem erro).
""")


def _download_cerra() -> list[Path] | None:
    """Baixa um arquivo NetCDF POR TRIMESTRE (ver `_cerra_request_for_quarter`
    para o motivo do particionamento e do parâmetro `grid`). Reaproveita
    trimestres já baixados em execuções anteriores (idempotente) -- útil
    porque cada requisição pode levar dezenas de minutos na fila do CDS."""
    try:
        import cdsapi
    except ImportError:
        print("  AVISO: pacote 'cdsapi' não instalado (pip install cdsapi). Pulando.")
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    paths = []
    for i, (year, months) in enumerate(QUARTERS, 1):
        tag = f"{year}Q{months[0][:2]}"
        out_path = CACHE_DIR / f"cerra_{year}_{months[0]}-{months[-1]}.nc"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  [{i}/{len(QUARTERS)}] {tag}: cache já existe, pulando download.")
            paths.append(out_path)
            continue
        print(f"  [{i}/{len(QUARTERS)}] Requisição CDS {tag} (meses {months})...")
        print("  AVISO: cada requisição pode ficar em fila ('Accepted') por até ~1h no lado "
              "do CDS antes de começarem a rodar — normal, aguardar (idempotente entre execuções).")
        try:
            client.retrieve("reanalysis-cerra-height-levels", _cerra_request_for_quarter(year, months), str(out_path))
        except Exception as e:
            print(f"  ERRO no download CERRA ({tag}): {type(e).__name__}: {e}")
            print("  Se for erro de parâmetro inválido, regenere a requisição via a interface "
                  "web do dataset (botão 'Show API request') e ajuste este script.")
            if paths:
                print(f"  Prosseguindo com os {len(paths)} trimestre(s) já baixado(s) com sucesso; "
                      "rode o script de novo mais tarde para retomar de onde parou.")
                break
            return None
        print(f"  Cache salvo: {out_path.relative_to(cfg.ROOT)}")
        paths.append(out_path)
    return paths if paths else None


def _extract_nearest_point_series(nc_paths: list[Path]) -> pd.DataFrame:
    """Abre os NetCDF (um por trimestre), concatena no tempo, extrai a célula de
    grade mais próxima do centroide da rede Utrecht para cada altura, e
    retorna colunas cerra_wind_{speed,dir}_{H}m indexadas por timestamp UTC."""
    import xarray as xr

    # Evita xr.open_mfdataset (exige o pacote opcional `dask` para combinar arquivos em modo
    # lazy) — os 24 arquivos trimestrais são pequenos (~3MB cada, ~72MB total), então abrir
    # cada um eager e concatenar na memória com xr.concat é mais simples e não adiciona uma
    # dependência nova só para isto.
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
    # Nome confirmado empiricamente no NetCDF real da CERRA (2026-07-01): a variável de
    # velocidade vem nomeada "si100" (herdado do shortName GRIB do primeiro nível pedido) MESMO
    # contendo as 3 alturas na dimensão `heightAboveGround` -- NÃO existem "si200"/"si500"
    # separados. Por isso buscamos por prefixo "si" em vez de montar o nome com a altura.
    speed_var = next((v for v in point.data_vars if str(v).startswith("si")), None)
    dir_var = next((v for v in point.data_vars if "wdir" in str(v) or "wind_dir" in str(v)), None)
    frames = []
    for h in HEIGHTS_M:
        try:
            sub = point.sel(heightAboveGround=h) if "heightAboveGround" in point.dims else point
            speed = sub[speed_var] if speed_var else None
            direction = sub[dir_var] if dir_var else None
        except Exception as e:
            print(f"  AVISO: não foi possível extrair altura {h}m ({e}) — verifique nomes de "
                  "variável no NetCDF baixado (ds.data_vars).")
            continue
        if speed is None or direction is None:
            print(f"  AVISO: variáveis de vento não encontradas para {h}m — inspecione "
                  f"manualmente: xr.open_dataset(caminho).data_vars")
            continue
        df_h = pd.DataFrame({
            f"cerra_wind_speed_ms_{h}m": speed.to_numpy(),
            f"cerra_wind_dir_deg_{h}m": direction.to_numpy(),
        }, index=pd.to_datetime(sub[time_name].to_numpy(), utc=True))
        frames.append(df_h)

    if not frames:
        raise RuntimeError("Nenhuma altura extraída com sucesso — ver avisos acima.")
    out = pd.concat(frames, axis=1)
    return out[~out.index.duplicated(keep="first")].sort_index()


def main() -> None:
    print(SEP)
    print("B7b — VENTO EM ALTURA (CERRA, 5,5 km, 100/200/500 m)")
    print(SEP)

    if not WIND_PQ.exists():
        print(f"ERRO: {WIND_PQ} não encontrado. Execute B7_wind_join.py primeiro.")
        sys.exit(1)

    if not CDSAPIRC.exists():
        _print_registration_instructions()
        print(f"\n{SEP}")
        print("B7b PULADO (aprimoramento opcional, pipeline principal não afetado).")
        log_result(
            script="B7b_wind_cerra.py", gate="", phase="F0/B7b",
            params={"dataset": "reanalysis-cerra-height-levels", "heights_m": HEIGHTS_M},
            results={"status": "skipped_no_registration"},
            decision="N/A — data acquisition step, pending user action",
            action="Skipped: ~/.cdsapirc not found. Requires free CDS account + personal "
                   "access token (instructions printed to stdout and logged in ROADMAP F6).",
            interpretation="Height-resolved wind (100/200/500m via CERRA, 5.5km) not yet "
                           "available; F6 anisotropy analysis can proceed with the 10m surface "
                           "wind from B7 in the meantime, height comparison deferred.",
            paper_ref="Section 8 (F6 spatial structure) -- wind height sensitivity, pending",
        )
        return

    nc_paths = _download_cerra()
    if not nc_paths:
        print(f"\n{SEP}")
        print("B7b PULADO (download indisponível — ver erro acima).")
        return

    print("\n  Extraindo série no ponto de grade mais próximo do centroide da rede...")
    cerra = _extract_nearest_point_series(nc_paths)
    print(f"  Série CERRA: {cerra.shape}  [{cerra.index[0]} → {cerra.index[-1]}]")

    wind = pd.read_parquet(WIND_PQ)
    wind["start_ts"] = pd.to_datetime(wind["start_ts"], utc=True)
    cerra_cols = cerra.columns.tolist()
    drop_existing = [c for c in cerra_cols if c in wind.columns]
    if drop_existing:
        wind = wind.drop(columns=drop_existing)   # idempotente em re-runs

    df_joined = pd.merge_asof(
        wind.sort_values("start_ts"),
        cerra.rename_axis("start_ts").reset_index(),
        on="start_ts",
        tolerance=MERGE_TOLERANCE,
        direction="nearest",
    )
    for h in HEIGHTS_M:
        col = f"cerra_wind_speed_ms_{h}m"
        if col in df_joined.columns:
            n_matched = df_joined[col].notna().sum()
            print(f"  {h}m: {n_matched:,} / {len(df_joined):,} eventos casados "
                  f"({n_matched/len(df_joined):.1%})")

    df_joined.to_parquet(WIND_PQ, index=False)
    print(f"\n  Atualizado: {WIND_PQ.relative_to(cfg.ROOT)} (+{len(cerra_cols)} colunas CERRA)")

    print(f"\n{SEP}")
    print("B7b concluído — perfil vertical de vento (10/100/200/500m) disponível para F6.")

    log_result(
        script="B7b_wind_cerra.py", gate="", phase="F0/B7b",
        params={
            "dataset": "reanalysis-cerra-height-levels", "resolution_km": 5.5,
            "heights_m": HEIGHTS_M, "temporal_resolution": "3-hourly (analysis)",
            "merge_tolerance_min": MERGE_TOLERANCE.total_seconds() / 60,
            "grid_point": "nearest to network centroid (single-point time series)",
        },
        results={
            f"n_matched_{h}m": int(df_joined[f"cerra_wind_speed_ms_{h}m"].notna().sum())
            for h in HEIGHTS_M if f"cerra_wind_speed_ms_{h}m" in df_joined.columns
        },
        decision="N/A — data acquisition step",
        action=(
            "Added height-resolved wind (100/200/500m) via CERRA regional reanalysis (5.5km) "
            "as a complement to B7's 10m KNMI surface wind, after determining ERA5 pressure-level "
            "wind (31km resolution) was too coarse: the Utrecht network (~44x50km) spans only "
            "~1.4x1.6 ERA5 grid cells (essentially one point), vs ~8.0x9.2 CERRA cells -- a ~6x "
            "improvement, at the cost of 3-hourly (not hourly) temporal resolution."
        ),
        interpretation=(
            "Four-height wind profile (10/100/200/500m) now available for F6; which height best "
            "explains the observed anisotropy (if any) should be reported as an explicit "
            "methodological finding in the paper, not assumed a priori."
        ),
        paper_ref="Section 8 (F6 spatial structure) -- wind height sensitivity analysis",
    )


if __name__ == "__main__":
    main()
