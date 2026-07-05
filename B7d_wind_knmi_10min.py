"""
B7d_wind_knmi_10min.py — Vento KNMI De Bilt, resolução de 10 minutos (Risco C)
================================================================================
Motivação (ROADMAP Seção 0.2, Risco C / resolucao_gaps.md Seção 1): o teste de advecção
F6b usa vento HORÁRIO (KNMI, `B7_wind_join.py`) e CERRA de 3h, ambos mais grosseiros que a
escala do fenômeno (rampa mediana=14min, atraso mediano entre estações=14min). A KNMI Data
Platform tem um dataset HISTÓRICO de observações de 10 minutos
(`10-minute-in-situ-meteorological-observations`, desde 2012-01-01) que resolve a dimensão
temporal para o canal de vento de SUPERFÍCIE (não resolve CERRA, que segue limitado a 3h por
ser produto de reanálise — ver ressalva no ROADMAP).

Estratégia de escopo (evita baixar 5-6 anos contínuos, ~262 mil arquivos país-inteiro):
F6b só usa vento no instante `t_ext` de cada evento CASADO (`matched=True`) de
`aligned_pairs.parquet` — ao arredondar para o múltiplo de 10min mais próximo, isso reduz a
necessidade a ~6.571 timestamps ÚNICOS (não a série contínua), tornando o download tratável
com a chave anônima pública da KNMI (50 req/min compartilhado), sem precisar de cadastro
nem de chave "bulk" (que exigiria e-mail a opendata@knmi.nl e aprovação manual).

Fonte: KNMI Data Platform, Open Data API (`api.dataplatform.knmi.nl`), dataset
`10-minute-in-situ-meteorological-observations` v1.0, arquivos NetCDF (1 arquivo = 1
timestamp de 10min, ~64 estações automáticas do país). Variáveis usadas para a estação
De Bilt (WMO 06260, mesma estação de `B7_wind_join.py`):
  - `dd` — direção do vento (°), média do intervalo de 10min ("Wind Direction Mean with MD")
  - `ff` — velocidade do vento a 10m (m/s), média do intervalo ("Wind Speed at 10 m Mean with MD")
Timestamp do arquivo / coordenada `time` = UTC, FIM do intervalo de 10min — sem correção de
fuso necessária (ao contrário do endpoint horário legado usado em B7, que publica em UTC+1
fixo "wintertijd").

Chave de API: usa a chave ANÔNIMA pública documentada em
https://developer.dataplatform.knmi.nl/open-data-api ("Anonymous key" — acesso não
cadastrado, compartilhado, 50 req/min / 3000 req/hora). Não é segredo do projeto — é uma
chave de demonstração publicada pela própria KNMI para uso não-autenticado.

Cache: arquivos NetCDF brutos salvos em `data/raw/wind/knmi_10min_cache/<timestamp>.nc`
(idempotente — timestamps já baixados são pulados em reexecuções; permite retomar se
interrompido sem perder trabalho já feito).

Não compete com o download de CERRA em andamento (`B7c_wind_cerra_pressure.py`) — API/host
completamente diferentes (api.dataplatform.knmi.nl vs. cds.climate.copernicus.eu),
throttling conservador, arquivos pequenos (~150-180KB cada).

Saída:
  data/interim/wind_knmi_10min.parquet   — timestamp (UTC, fim do intervalo), wind_speed_ms,
                                            wind_dir_deg, station_wmo="06260"

Executar:
    python B7d_wind_knmi_10min.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg

ALIGNED_PQ = cfg.DIRS["processed"] / "aligned_pairs.parquet"
CACHE_DIR  = cfg.DIRS["raw_wind"] / "knmi_10min_cache"
OUT_PQ     = cfg.DIRS["interim"] / "wind_knmi_10min.parquet"

ANON_API_KEY = ("eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjUzYTg1ZDBhMmQ5YzRk"
                "YzJiYWNlNzQ4NTQ2Zjk4ODExIiwiaCI6Im11cm11cjEyOCJ9")   # chave anônima pública KNMI
DATASET = "10-minute-in-situ-meteorological-observations"
VERSION = "1.0"
API_BASE = f"https://api.dataplatform.knmi.nl/open-data/v1/datasets/{DATASET}/versions/{VERSION}/files"
STATION_WMO = "06260"   # De Bilt — mesma estação de B7_wind_join.py (KNMI 260)

SLEEP_BETWEEN_REQ_S = 1.3   # throttling conservador (anônimo: 50 req/min compartilhado)
MAX_RETRIES = 3
CHECKPOINT_EVERY = 200

SEP = "─" * 60


def _fname_for(ts: pd.Timestamp) -> str:
    return f"KMDS__OPER_P___10M_OBS_L2_{ts.strftime('%Y%m%d%H%M')}.nc"


def _fetch_one(ts: pd.Timestamp, session: requests.Session) -> Path | None:
    """Baixa (ou reaproveita do cache) o arquivo NetCDF de um timestamp. Retorna o path
    local, ou None se o arquivo não existir na fonte (ex.: lacuna operacional)."""
    fname = _fname_for(ts)
    local_path = CACHE_DIR / fname
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(f"{API_BASE}/{fname}/url",
                             headers={"Authorization": ANON_API_KEY}, timeout=20)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(5.0 * (attempt + 1))
                continue
            r.raise_for_status()
            url = r.json()["temporaryDownloadUrl"]
            time.sleep(SLEEP_BETWEEN_REQ_S)

            r2 = session.get(url, timeout=30)
            r2.raise_for_status()
            local_path.write_bytes(r2.content)
            return local_path
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                print(f"    AVISO: falha ao baixar {fname} após {MAX_RETRIES} tentativas ({e}).")
                return None
            time.sleep(3.0 * (attempt + 1))
    return None


def _parse_one(path: Path, ts: pd.Timestamp) -> dict | None:
    try:
        with xr.open_dataset(path) as ds:
            if STATION_WMO not in ds["station"].values:
                return None
            sub = ds.sel(station=STATION_WMO)
            speed = float(sub["ff"].values[0]) if "ff" in ds else np.nan
            direction = float(sub["dd"].values[0]) if "dd" in ds else np.nan
    except Exception as e:
        print(f"    AVISO: falha ao ler {path.name} ({type(e).__name__}: {e}).")
        return None
    if not (np.isfinite(speed) and np.isfinite(direction)):
        return None
    return {"timestamp_utc": ts, "wind_speed_ms": speed, "wind_dir_deg": direction,
            "station_wmo": STATION_WMO}


def main() -> None:
    print(SEP)
    print("B7d — VENTO KNMI DE BILT, RESOLUÇÃO DE 10 MINUTOS (Risco C, ROADMAP Seção 0.2)")
    print(SEP)

    if not ALIGNED_PQ.exists():
        print(f"ERRO: {ALIGNED_PQ} não encontrado. Execute C1b_event_pairing.py primeiro.")
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Determinando timestamps únicos necessários (eventos casados de F6b)...")
    aligned = pd.read_parquet(ALIGNED_PQ)
    matched = aligned[aligned["matched"] & aligned["dt_min"].notna()]
    t_ext = pd.to_datetime(matched["t_ext"], utc=True)
    t_rounded = t_ext.dt.round("10min")
    unique_ts = pd.Series(t_rounded.unique()).sort_values().reset_index(drop=True)
    print(f"  Eventos casados: {len(matched):,}  →  timestamps únicos (10min): {len(unique_ts):,}")
    print(f"  Faixa: {unique_ts.min()} .. {unique_ts.max()}")
    est_hours = len(unique_ts) * 2 * SLEEP_BETWEEN_REQ_S / 3600.0
    print(f"  Estimativa de tempo (throttling {SLEEP_BETWEEN_REQ_S}s/req, 2 req/arquivo): "
          f"~{est_hours:.1f}h (chave anônima pública, compartilhada — pode variar)")

    already_cached = sum(1 for ts in unique_ts if (CACHE_DIR / _fname_for(pd.Timestamp(ts))).exists())
    print(f"  Já em cache local: {already_cached:,}/{len(unique_ts):,} (retomando de execução anterior, se houver)")

    print(f"\n[2/3] Baixando (ou reaproveitando cache) — {len(unique_ts):,} arquivos...")
    session = requests.Session()
    rows = []
    t0 = time.time()
    n_missing_source = 0
    n_parse_fail = 0
    for i, ts_raw in enumerate(unique_ts):
        ts = pd.Timestamp(ts_raw)
        path = _fetch_one(ts, session)
        if path is None:
            n_missing_source += 1
            continue
        row = _parse_one(path, ts)
        if row is None:
            n_parse_fail += 1
            continue
        rows.append(row)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta_min = (len(unique_ts) - (i + 1)) / rate / 60.0 if rate > 0 else float("nan")
            print(f"  ... {i+1:,}/{len(unique_ts):,} processados ({len(rows):,} válidos, "
                  f"{n_missing_source} ausentes na fonte, {n_parse_fail} falha de parsing) "
                  f"[{elapsed/60:.1f}min decorridos, ETA ~{eta_min:.0f}min]")
            pd.DataFrame(rows).to_parquet(OUT_PQ, index=False)   # checkpoint incremental

    print(f"\n  Concluído: {len(rows):,}/{len(unique_ts):,} timestamps com dado válido "
          f"({n_missing_source} ausentes na fonte, {n_parse_fail} falha de parsing) "
          f"em {(time.time()-t0)/60:.1f} min")

    if not rows:
        print("ERRO: nenhum timestamp resultou em dado válido — abortando.")
        sys.exit(1)

    print("\n[3/3] Salvando parquet final...")
    out_df = pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)
    out_df.to_parquet(OUT_PQ, index=False)
    print(f"  Salvo: {OUT_PQ.relative_to(cfg.ROOT)}  ({len(out_df):,} linhas)")
    print(f"  Cobertura: {len(out_df)}/{len(unique_ts)} = {len(out_df)/len(unique_ts):.1%} dos timestamps necessários")

    print(f"\n{SEP}")
    print("B7d concluído — wind_knmi_10min.parquet disponível para S4_f6b_wind_10min.py.")


if __name__ == "__main__":
    main()
