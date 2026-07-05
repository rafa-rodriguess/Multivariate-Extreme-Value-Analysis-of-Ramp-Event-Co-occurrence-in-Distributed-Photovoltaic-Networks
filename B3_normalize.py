"""
B3_normalize.py — Normalização pela capacidade DC instalada
============================================================
Converte medições em Watts para potência normalizada p* (adimensional):

    p*_i(t) = P_i(t) [W]  /  DC_capacity_i [W]

Critérios de sanidade após normalização:
  - p* deve estar em [0, p_max_norm]  (p_max_norm = cfg.QC["p_norm_max"])
  - Valores fora do intervalo → NaN (possível defeito não filtrado pelo qcpv)

Entradas:
  data/interim/power_qc.parquet
  data/interim/coords.parquet

Saídas:
  data/interim/power_norm.parquet   — p*_i(t) por usina
  data/interim/norm_report.csv      — contagem de clipping por usina

Executar:
    python B3_normalize.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg

IN_PQ    = cfg.DIRS["interim"] / "power_qc.parquet"
COORDS   = cfg.DIRS["interim"] / "coords.parquet"
OUT_PQ   = cfg.DIRS["interim"] / "power_norm.parquet"
OUT_REP  = cfg.DIRS["interim"] / "norm_report.csv"

P_MAX    = cfg.QC.get("p_norm_max", 1.1)   # limiar superior (tolerância de 10%)

SEP = "─" * 60


def main() -> None:
    print(SEP)
    print("B3 — NORMALIZAÇÃO PELA CAPACIDADE DC")
    print(SEP)

    for p in [IN_PQ, COORDS]:
        if not p.exists():
            print(f"ERRO: {p} não encontrado. Execute B2_qc.py primeiro.")
            sys.exit(1)

    # Carregar
    df     = pd.read_parquet(IN_PQ)
    coords = pd.read_parquet(COORDS).set_index("station_id")
    print(f"  Dados: {df.shape}  (timestamps × usinas)")

    # Normalizar coluna a coluna
    df_norm = df.copy()
    report_rows = []

    for sid in df.columns:
        if sid not in coords.index:
            print(f"  AVISO: {sid} sem metadados — pulando")
            df_norm[sid] = np.nan
            continue

        dc_cap_w = coords.loc[sid, "capacity_dc_wp"]
        if dc_cap_w <= 0 or np.isnan(dc_cap_w):
            print(f"  AVISO: {sid} com capacidade DC inválida ({dc_cap_w}) — pulando")
            df_norm[sid] = np.nan
            continue

        # Normalizar
        df_norm[sid] = df[sid] / dc_cap_w

        # Clipar negativos → NaN (valores sub-limiar remanescentes)
        n_neg = (df_norm[sid] < 0).sum()
        df_norm.loc[df_norm[sid] < 0, sid] = np.nan

        # Clipar acima do máximo → NaN (overirradiance extremo / erro sensor)
        n_high = (df_norm[sid] > P_MAX).sum()
        df_norm.loc[df_norm[sid] > P_MAX, sid] = np.nan

        report_rows.append({
            "station_id":  sid,
            "dc_cap_wp":   dc_cap_w,
            "n_neg":       int(n_neg),
            "n_high":      int(n_high),
            "n_clipped":   int(n_neg + n_high),
        })

    report = pd.DataFrame(report_rows)

    # Sumário
    total_clipped = report["n_clipped"].sum()
    total_obs     = df.notna().sum().sum()
    print(f"  Total de observações válidas (pré-clip): {total_obs:,}")
    print(f"  Observações clippadas → NaN:             {total_clipped:,}  ({total_clipped/total_obs*100:.3f}%)")
    print(f"  Usinas com clip:  {(report['n_clipped'] > 0).sum()}")
    print(f"\n  p* range (pós-clip): [{df_norm.min().min():.4f}, {df_norm.max().max():.4f}]")

    # Salvar
    df_norm.to_parquet(OUT_PQ)
    print(f"\n  Salvo: {OUT_PQ.relative_to(cfg.ROOT)}  ({OUT_PQ.stat().st_size/1e6:.0f} MB)")

    report.to_csv(OUT_REP, index=False)
    print(f"  Salvo: {OUT_REP.relative_to(cfg.ROOT)}")

    print(f"\n{SEP}")
    print("B3 concluído — power_norm.parquet disponível.")


if __name__ == "__main__":
    main()
