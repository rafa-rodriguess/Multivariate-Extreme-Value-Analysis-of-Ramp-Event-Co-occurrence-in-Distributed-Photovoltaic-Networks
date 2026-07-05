"""
B2_qc.py — Controle de qualidade e censura por usina
======================================================
Carrega o dataset pré-filtrado (filtered_pv_power_measurements_ac.csv)
que já passou pela rotina qcpv de Lanzilao & Meyer (2022).
Documenta a censura: % de timestamps removidos por usina ao longo
de cada período de operação declarado em metadata.csv.

Entradas:
  data/raw/utrecht/filtered_pv_power_measurements_ac.csv
  data/interim/coords.parquet

Saídas:
  data/interim/power_qc.parquet   — dados wide (DateTime × 175 colunas)
  data/interim/qc_report.csv      — relatório de censura por usina

Executar:
    python B2_qc.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg

RAW_CSV   = cfg.DIRS["raw_utrecht"] / "filtered_pv_power_measurements_ac.csv"
COORDS    = cfg.DIRS["interim"]     / "coords.parquet"
OUT_PQ    = cfg.DIRS["interim"]     / "power_qc.parquet"
OUT_REP   = cfg.DIRS["interim"]     / "qc_report.csv"

SEP = "─" * 60


def _check_inputs() -> None:
    """Verifica presença dos arquivos de entrada."""
    for p in [RAW_CSV, COORDS]:
        if not p.exists():
            print(f"ERRO: arquivo não encontrado: {p}")
            print("       Execute B1_download_utrecht.py primeiro.")
            sys.exit(1)


def _load_power(path: Path) -> pd.DataFrame:
    """
    Carrega CSV wide com DatetimeIndex UTC.
    Aceita separador ',' e ';'.
    Colunas de usina: ID001 … ID175 (valores em Watts, float).
    """
    print(f"  Carregando {path.name} ({path.stat().st_size/1e9:.2f} GB)…")
    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
        low_memory=False,
    )
    # Garantir DatetimeIndex UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "datetime"

    # Converter colunas para float (qcpv pode ter strings vazias)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def _clip_to_operational(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada usina, seta NaN fora do intervalo [begin_ts, end_ts]
    declarado em metadata.csv (alguns sistemas têm período mais curto que 2014-2017).
    """
    coords_idx = coords.set_index("station_id")
    for sid in df.columns:
        if sid not in coords_idx.index:
            continue
        begin = coords_idx.loc[sid, "begin_ts"]
        end   = coords_idx.loc[sid, "end_ts"]
        mask_out = (df.index < begin) | (df.index > end)
        df.loc[mask_out, sid] = np.nan
    return df


def _build_qc_report(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula para cada usina:
      - n_total:     timestamps dentro do período operacional
      - n_valid:     timestamps com valor não-NaN
      - pct_valid:   percentagem válida
      - pct_censored: percentagem censurada
    """
    coords_idx = coords.set_index("station_id")
    rows = []
    for sid in df.columns:
        if sid in coords_idx.index:
            begin = coords_idx.loc[sid, "begin_ts"]
            end   = coords_idx.loc[sid, "end_ts"]
            mask = (df.index >= begin) & (df.index <= end)
        else:
            mask = pd.Series(True, index=df.index)

        total   = int(mask.sum())
        valid   = int(df.loc[mask, sid].notna().sum())
        rows.append({
            "station_id":   sid,
            "n_total":      total,
            "n_valid":      valid,
            "pct_valid":    round(valid / total * 100, 2) if total > 0 else 0.0,
            "pct_censored": round((1 - valid / total) * 100, 2) if total > 0 else 100.0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    print(SEP)
    print("B2 — CONTROLE DE QUALIDADE (qcpv — Lanzilao & Meyer 2022)")
    print(SEP)

    _check_inputs()

    # 1. Carregar dados
    df     = _load_power(RAW_CSV)
    coords = pd.read_parquet(COORDS)
    print(f"  Shape bruto: {df.shape}  (timestamps × usinas)")

    # 2. Checar cobertura temporal
    print(f"  Período: {df.index[0]}  →  {df.index[-1]}")
    dt_diff = df.index.to_series().diff().dropna()
    mode_dt = dt_diff.mode()[0]
    print(f"  Resolução modal: {mode_dt}")

    # 3. Clipar ao período operacional de cada usina
    print("  Clipping ao período operacional por usina…")
    df = _clip_to_operational(df, coords)

    # 4. Relatório de censura
    print("  Calculando relatório de censura…")
    report = _build_qc_report(df, coords)

    pct_med  = report["pct_valid"].median()
    pct_p10  = report["pct_valid"].quantile(0.10)
    n_low    = (report["pct_valid"] < 50).sum()
    print(f"\n  Cobertura mediana:     {pct_med:.1f}%")
    print(f"  Percentil 10%:         {pct_p10:.1f}%")
    print(f"  Usinas com <50% dados: {n_low} de {len(report)}")

    if n_low > 0:
        print("\n  Usinas com cobertura crítica (<50%):")
        low = report[report["pct_valid"] < 50][["station_id", "pct_valid", "n_total", "n_valid"]]
        print(low.to_string(index=False))

    # 5. Salvar
    print(f"\n  Salvando power_qc.parquet…")
    df.to_parquet(OUT_PQ)
    print(f"  Salvo: {OUT_PQ.relative_to(cfg.ROOT)}  ({OUT_PQ.stat().st_size/1e6:.0f} MB)")

    report.to_csv(OUT_REP, index=False)
    print(f"  Salvo: {OUT_REP.relative_to(cfg.ROOT)}")

    print(f"\n{SEP}")
    print("B2 concluído — power_qc.parquet e qc_report.csv disponíveis.")


if __name__ == "__main__":
    main()
