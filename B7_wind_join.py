"""
B7_wind_join.py — Junção de dados de vento KNMI (De Bilt, estação 260)
========================================================================
Associa covariáveis de vento (velocidade, direção) aos timestamps dos ramp
events, para uso na análise de anisotropia (F6).

Fonte: KNMI — dados horários históricos da estação De Bilt (260), obtidos
automaticamente via HTTP POST ao endpoint público (SEM chave/cadastro):

    https://www.daggegevens.knmi.nl/klimatologie/uurgegevens

Este é o endpoint de dados HISTÓRICOS (não confundir com o feed de tempo
quase-real `opendata.knmi.nl/Actuele10min...`, que só cobre os últimos 10
dias e é inadequado para 2014-2017).

Limitações conhecidas (documentar no paper):
  1. **Altura de medição**: vento de SUPERFÍCIE (10 m), não necessariamente
     representativo do vento em altura de base de nuvem (tipicamente
     600-1500 m no clima da Holanda). Aprimoramento em `B7b_wind_cerra.py`
     (CERRA, 5,5 km, alturas reais até 500 m) — ver ROADMAP F6. (ERA5 por
     nível de pressão foi avaliado e descartado: resolução de 31 km cobre a
     rede Utrecht em só ~1,4×1,6 células, insuficiente até para robustez
     espacial; CERRA resolve em ~8×9 células. Torre Cabauw também avaliada
     e descartada: vive num portal de pesquisa separado do KNMI Data
     Platform, exigiria outro cadastro não verificado, e mesmo assim só
     alcança 200 m, abaixo do teto de 500 m da CERRA.)
  2. **Convenção de tempo**: a KNMI publica em referência fixa UTC+1 o ano
     todo ("wintertijd", nunca ajustada por horário de verão). Convertido
     aqui para UTC verdadeiro (subtrai 1h) e ancorado no PONTO MÉDIO da
     hora (não no fim), para um casamento mais robusto por `merge_asof`.
  3. **Estação única (De Bilt)**: representa o vento sinótico regional, não
     variação espacial dentro da rede (~50-70 km de extensão). Isso é
     adequado para o teste de anisotropia pretendido (bearing do par vs.
     direção do vento AMBIENTE no momento do evento é uma pergunta
     temporal, não espacial — a escala de coerência do vento sinótico é
     muito maior que a extensão da rede).

Fallback (se o download falhar — ex. sem rede):
  1. Usa cache local em `data/raw/wind/knmi_debilt_wind.csv`, se existir.
  2. Caso contrário, gera vento sintético de placeholder (não bloqueia o
     pipeline) e emite AVISO claro — NÃO usar para conclusões científicas.

Entradas:
  data/interim/ramps.parquet

Saída:
  data/interim/wind_joined.parquet
  data/raw/wind/knmi_debilt_wind.csv   (cache do download, para reprodutibilidade)

Executar:
    python B7_wind_join.py
"""

from __future__ import annotations

import sys
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result

STATION_ID = "260"   # De Bilt
KNMI_URL   = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"

WIND_CSV   = cfg.DIRS["raw_wind"]  / "knmi_debilt_wind.csv"
RAMPS_PQ   = cfg.DIRS["interim"]   / "ramps.parquet"
OUT_PQ     = cfg.DIRS["interim"]   / "wind_joined.parquet"

MERGE_TOLERANCE = pd.Timedelta("45min")

SEP = "─" * 60


def _download_knmi_hourly(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """
    Baixa vento horário real da estação De Bilt (260) via POST público, sem
    chave/cadastro. Retorna None em qualquer falha (rede, formato inesperado),
    para acionar o fallback — nunca lança exceção para o chamador.
    """
    try:
        import requests
    except ImportError:
        print("  AVISO: pacote 'requests' não instalado — pulando download.")
        return None

    # ARMADILHA DA API: a parte HH de `start`/`end` não define um intervalo de tempo
    # contínuo — define quais HORAS DO DIA são retornadas em TODO o período (ex.
    # start=...06, end=...08 devolve só as horas 6,7,8 de CADA dia do intervalo). Para
    # obter as 24h de cada dia, forçamos HH=01 no início e HH=24 no fim.
    try:
        resp = requests.post(
            KNMI_URL,
            data={
                "stns": STATION_ID,
                "vars": "WIND",   # DD (direção), FH/FF (velocidade horária/10min-média), FX (rajada)
                "start": start.strftime("%Y%m%d") + "01",
                "end": end.strftime("%Y%m%d") + "24",
            },
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  AVISO: download KNMI falhou ({type(e).__name__}: {e}).")
        return None

    lines = resp.text.splitlines()
    # NOTA: há duas linhas de comentário começando com "# STN" — a de METADADOS da
    # estação ("# STN         LON(east)   LAT(north) ...", sem vírgulas) e a linha de
    # CABEÇALHO real ("# STN,YYYYMMDD,HH,...", com vírgulas). Precisamos da segunda.
    header_line = next((l for l in lines if l.lstrip().startswith("# STN,")), None)
    if header_line is None:
        print("  AVISO: resposta da KNMI em formato inesperado (sem cabeçalho '# STN,...').")
        return None

    columns = [c.strip().lstrip("#").strip() for c in header_line.split(",")]
    data_str = "\n".join(l for l in lines if l.strip() and not l.strip().startswith("#"))
    if not data_str:
        print("  AVISO: resposta da KNMI sem linhas de dados.")
        return None

    try:
        df = pd.read_csv(StringIO(data_str), names=columns, skipinitialspace=True)
    except Exception as e:
        print(f"  AVISO: falha ao parsear CSV da KNMI ({type(e).__name__}: {e}).")
        return None

    return df


def _knmi_hour_to_utc_mid(yyyymmdd: pd.Series, hh: pd.Series) -> pd.Series:
    """
    KNMI publica HH em {1..24} como fim-de-hora em referência fixa UTC+1
    (nunca ajustada por DST). HH=24 vira 00:00 do dia seguinte (tratado
    corretamente por soma de timedelta em horas). Convertemos para o PONTO
    MÉDIO da hora em UTC verdadeiro: -60 min (offset UTC+1→UTC) -30 min
    (fim-de-hora→meio-de-hora) = -90 min.
    """
    date = pd.to_datetime(yyyymmdd.astype(int).astype(str), format="%Y%m%d")
    hour_end_local = date + pd.to_timedelta(hh.astype(int), unit="h")
    return (hour_end_local - pd.Timedelta(minutes=90)).dt.tz_localize("UTC")


def _parse_knmi_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Converte o DataFrame cru (colunas STN, YYYYMMDD, HH, DD, FH, FF, FX) em
    uma série indexada por timestamp UTC com wind_speed_ms e wind_dir_deg."""
    df = df.rename(columns=str.strip)
    ts = _knmi_hour_to_utc_mid(df["YYYYMMDD"], df["HH"])
    speed_col = "FH" if "FH" in df.columns else "FF"   # FH = média horária; fallback FF
    wind_speed_ms = pd.to_numeric(df[speed_col], errors="coerce") / 10.0   # 0.1 m/s -> m/s
    wind_dir_deg = pd.to_numeric(df["DD"], errors="coerce").astype(float)
    # Convenção KNMI: DD=0 é "calmaria" (sem direção definida, não Norte real) e
    # DD=990 é "direção variável" — ambos não são bearings válidos, viram NaN.
    wind_dir_deg = wind_dir_deg.where(~wind_dir_deg.isin([0, 990]), np.nan)
    # IMPORTANTE: usar .to_numpy() — passar Series (índice posicional 0..n) junto com
    # `index=ts` (DatetimeIndex) faria o pandas REALINHAR por índice, sem nenhuma
    # correspondência entre RangeIndex e timestamps, zerando tudo para NaN.
    out = pd.DataFrame(
        {"wind_speed_ms": wind_speed_ms.to_numpy(), "wind_dir_deg": wind_dir_deg.to_numpy()},
        index=ts,
    )
    return out[~out.index.duplicated(keep="first")].sort_index()


def _load_cached_csv(path: Path) -> pd.DataFrame:
    """Carrega o CSV já processado (índice = timestamp UTC, colunas wind_speed_ms/wind_dir_deg)."""
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["wind_speed_ms", "wind_dir_deg"]]


def _synthetic_wind(ramps: pd.DataFrame) -> pd.DataFrame:
    """Vento sintético de placeholder (ruído uniforme) — só para não travar o
    pipeline se download E cache falharem. NÃO usar para conclusões científicas."""
    warnings.warn(
        "VENTO SINTÉTICO EM USO — download KNMI e cache local ambos indisponíveis. "
        "Resultados de anisotropia (F6) NÃO devem ser interpretados cientificamente "
        "até que isso seja corrigido.",
        UserWarning,
        stacklevel=2,
    )
    rng = np.random.default_rng(cfg.SEED)
    n = len(ramps)
    return ramps.assign(
        wind_speed_ms=rng.uniform(0, 12, n).round(2),
        wind_dir_deg=rng.uniform(0, 360, n).round(1),
        wind_synthetic=True,
    )


def main() -> None:
    print(SEP)
    print("B7 — JUNÇÃO DE DADOS DE VENTO (KNMI De Bilt, estação 260)")
    print(SEP)

    if not RAMPS_PQ.exists():
        print(f"ERRO: {RAMPS_PQ} não encontrado. Execute B5_ramp_detection.py primeiro.")
        sys.exit(1)

    ramps = pd.read_parquet(RAMPS_PQ)
    ramps["start_ts"] = pd.to_datetime(ramps["start_ts"], utc=True)
    print(f"  Ramp events: {len(ramps):,}")
    print(f"  Período: {ramps['start_ts'].min()} → {ramps['start_ts'].max()}")

    wind = None
    if WIND_CSV.exists():
        print(f"\n[1/2] Cache local encontrado: {WIND_CSV.relative_to(cfg.ROOT)}")
        try:
            wind = _load_cached_csv(WIND_CSV)
            print(f"  Vento (cache): {wind.shape}  [{wind.index[0]} → {wind.index[-1]}]")
        except Exception as e:
            print(f"  AVISO: cache corrompido ({e}); tentando download.")
            wind = None

    if wind is None:
        print(f"\n[1/2] Baixando vento horário real da KNMI (estação {STATION_ID}, De Bilt)...")
        margin = pd.Timedelta(days=1)
        raw = _download_knmi_hourly(ramps["start_ts"].min() - margin, ramps["start_ts"].max() + margin)
        if raw is not None:
            wind = _parse_knmi_raw(raw)
            cfg.DIRS["raw_wind"].mkdir(parents=True, exist_ok=True)
            wind.to_csv(WIND_CSV)
            print(f"  Download OK: {wind.shape}  [{wind.index[0]} → {wind.index[-1]}]")
            print(f"  Cache salvo: {WIND_CSV.relative_to(cfg.ROOT)}")

    print("\n[2/2] Casando vento com eventos de rampa...")
    if wind is not None:
        df_joined = pd.merge_asof(
            ramps.sort_values("start_ts"),
            wind.rename_axis("start_ts").reset_index(),
            on="start_ts",
            tolerance=MERGE_TOLERANCE,
            direction="nearest",
        )
        df_joined["wind_synthetic"] = False
        n_matched = df_joined["wind_speed_ms"].notna().sum()
        print(f"  Merge (tolerância {MERGE_TOLERANCE}): {n_matched:,} / {len(df_joined):,} "
              f"eventos com vento real ({n_matched/len(df_joined):.1%})")
        if n_matched < len(df_joined):
            print(f"  AVISO: {len(df_joined)-n_matched:,} eventos sem vento casado "
                  "(provável lacuna na série KNMI) — ficam NaN, não sintéticos.")
    else:
        print("  Download e cache indisponíveis — usando vento sintético (ver AVISO acima).")
        df_joined = _synthetic_wind(ramps)

    df_joined.to_parquet(OUT_PQ, index=False)
    print(f"\n  Salvo: {OUT_PQ.relative_to(cfg.ROOT)}  ({OUT_PQ.stat().st_size/1e6:.1f} MB)")

    print(f"\n{SEP}")
    is_real = wind is not None
    if is_real:
        print("B7 concluído — vento REAL (KNMI De Bilt, 10m) disponível em wind_joined.parquet.")
        print("LIMITAÇÃO: vento de superfície (10m), não de altura de base de nuvem.")
        print("Aprimoramento em B7b_wind_cerra.py: vento em altura real (100/200/500m, CERRA "
              "5,5km) — requer cadastro CDS opcional (ver ROADMAP F6).")
    else:
        print("B7 concluído — AÇÃO NECESSÁRIA: vento ainda sintético, ver AVISO acima.")

    log_result(
        script="B7_wind_join.py",
        gate="",
        phase="F0/B7",
        params={
            "source": "KNMI De Bilt (station 260), historical hourly, no-auth POST endpoint" if is_real
                       else "synthetic placeholder (uniform noise)",
            "merge_tolerance_min": MERGE_TOLERANCE.total_seconds() / 60,
            "height_m": 10,
        },
        results={
            "n_ramp_events": len(df_joined),
            "n_matched_real_wind": int(n_matched) if is_real else 0,
            "frac_matched": round(n_matched / len(df_joined), 4) if is_real else 0.0,
            "is_synthetic": not is_real,
        },
        decision="N/A — data acquisition step",
        action=(
            "Rewrote from a synthetic-placeholder-only design to a real automated download "
            "(no registration needed) from KNMI's historical hourly endpoint. Fixed two bugs "
            "found during this session: (1) constructing the DataFrame with a Series (not "
            "array) alongside an explicit DatetimeIndex caused silent index-alignment "
            "reindexing to all-NaN; (2) the KNMI API's HH portion of start/end does NOT define "
            "a continuous time range -- it restricts WHICH HOURS OF EACH DAY are returned across "
            "the whole period (e.g. start=...07,end=...15 returns only hours 7-15 of every day), "
            "which silently limited the first successful download to 48.1% matched (only ramps "
            "within that window); fixed by forcing HH=01/HH=24 to request full days."
        ),
        interpretation=(
            f"Real KNMI De Bilt (station 260, 10m) hourly wind now backs {n_matched:,}/{len(df_joined):,} "
            f"({n_matched/len(df_joined):.1%}) ramp events" if is_real else
            "Wind remains synthetic -- anisotropy analysis (F6) must not proceed until this is real."
        ),
        paper_ref="Section 3 (Data) / Section 8 (F6 spatial structure) -- wind covariate provenance",
    )


if __name__ == "__main__":
    main()
