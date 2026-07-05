"""
B8_temporal_split.py — Partição temporal train / test
======================================================
Define uma coluna 'split' nos dados de séries temporais e no catálogo
de ramp events, separando os períodos de treinamento e teste.

Critério (conforme ROADMAP — Gate G1 e além):
  train : 2014-01-01 → 2016-12-31 UTC  (~3 anos, ~75% do período)
  test  : 2017-01-01 → 2017-12-31 UTC  (~1 ano, ~25%)

Importante: a partição é puramente temporal (block split) para evitar
vazamento de dependência temporal.

Entradas:
  data/interim/clearsky_index.parquet
  data/interim/ramps.parquet
  data/interim/wind_joined.parquet

Saídas:
  data/interim/clearsky_index.parquet   — sobrescrito com coluna 'split'
  data/interim/ramps_split.parquet      — ramps com coluna 'split'
  data/interim/split_report.csv         — contagem por split e usina

Executar:
    python B8_temporal_split.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg

IN_K         = cfg.DIRS["interim"] / "clearsky_index.parquet"
IN_RAMPS     = cfg.DIRS["interim"] / "ramps.parquet"
IN_WIND      = cfg.DIRS["interim"] / "wind_joined.parquet"
OUT_RAMPS    = cfg.DIRS["interim"] / "ramps_split.parquet"
OUT_REP      = cfg.DIRS["interim"] / "split_report.csv"

TRAIN_END = pd.Timestamp(cfg.UTRECHT.get("train_end", "2016-12-31 23:59"), tz="UTC")
TEST_START= pd.Timestamp(cfg.UTRECHT.get("test_start","2017-01-01 00:00"), tz="UTC")

SEP = "─" * 60


def _assign_split(ts: pd.Series) -> pd.Series:
    """Retorna 'train' ou 'test' com base no timestamp."""
    split = pd.Series("train", index=ts.index)
    split.loc[ts >= TEST_START] = "test"
    return split


def main() -> None:
    print(SEP)
    print("B8 — PARTIÇÃO TEMPORAL TRAIN / TEST")
    print(f"  Train: até {TRAIN_END.date()}")
    print(f"  Test:  a partir de {TEST_START.date()}")
    print(SEP)

    # ── 1. clearsky_index.parquet ───────────────────────────────────────────
    if not IN_K.exists():
        print(f"ERRO: {IN_K} não encontrado. Execute B4_clearsky.py primeiro.")
        sys.exit(1)

    df_k = pd.read_parquet(IN_K)
    if "split" in df_k.columns:
        df_k = df_k.drop(columns=["split"])   # idempotente — evita duplicar em reruns
    split_ts = _assign_split(df_k.index.to_series())
    df_k.insert(0, "split", split_ts.values)
    df_k.to_parquet(IN_K)   # sobrescreve com coluna adicional

    n_train = (split_ts == "train").sum()
    n_test  = (split_ts == "test").sum()
    print(f"  clearsky_index: {n_train:,} timestamps train / {n_test:,} test")

    # ── 2. ramps_split.parquet ──────────────────────────────────────────────
    if not IN_RAMPS.exists():
        print(f"AVISO: {IN_RAMPS} não encontrado — pulando ramps.")
    else:
        ramps = pd.read_parquet(IN_RAMPS)
        ramps["start_ts"] = pd.to_datetime(ramps["start_ts"], utc=True)
        ramps["split"] = _assign_split(ramps["start_ts"]).values

        ramps.to_parquet(OUT_RAMPS, index=False)
        n_r_train = (ramps["split"] == "train").sum()
        n_r_test  = (ramps["split"] == "test").sum()
        print(f"  ramps_split:    {n_r_train:,} train / {n_r_test:,} test")
        print(f"  Salvo: {OUT_RAMPS.relative_to(cfg.ROOT)}")

    # ── 3. wind_joined.parquet — também adicionar split ─────────────────────
    if IN_WIND.exists():
        wind = pd.read_parquet(IN_WIND)
        if "start_ts" in wind.columns:
            wind["start_ts"] = pd.to_datetime(wind["start_ts"], utc=True)
            wind["split"] = _assign_split(wind["start_ts"]).values
            wind.to_parquet(IN_WIND, index=False)
            print(f"  wind_joined:    split adicionado")

    # ── 4. Relatório por usina ───────────────────────────────────────────────
    if IN_RAMPS.exists():
        report = (
            ramps
            .groupby(["station_id", "split"])
            .size()
            .unstack(fill_value=0)
            .rename(columns={"train": "n_train", "test": "n_test"})
            .reset_index()
        )
        report.to_csv(OUT_REP, index=False)
        print(f"  Salvo: {OUT_REP.relative_to(cfg.ROOT)}")

        n_ok = ((report.get("n_train", pd.Series([0])) >= cfg.G1.get("min_events", 30)) &
                (report.get("n_test",  pd.Series([0])) >= 5)).sum()
        print(f"\n  Usinas com ≥{cfg.G1.get('min_events',30)} ramps no train: {n_ok}/{len(report)}")

    print(f"\n{SEP}")
    print("B8 concluído — partição temporal definida.")
    print("Próximo passo: Bloco C — Gate G0 real com coordenadas Utrecht.")


if __name__ == "__main__":
    main()
