"""
F7a_aggregate_series.py — Rampa agregada da rede (usina virtual)
==================================================================
Constrói a série de índice de céu-claro AGREGADO da rede, tratando as 175
usinas como uma única "usina virtual" ponderada por capacidade instalada:

    k_agg(t) = Σ_i w_i · k_i(t),   w_i = capacidade_dc_kwp_i / Σ_j capacidade_dc_kwp_j

Reaproveita 100% o `clearsky_index.parquet` já calculado em B4 (não recalcula
céu-claro) — a variação do modelo de céu-claro entre usinas já foi mostrada
desprezível em B4 (todas dentro de ~10km de Utrecht), então a média ponderada
dos k_i individuais é equivalente, na prática, a normalizar a soma de potência
pelo céu-claro agregado, sem precisar reconstruir esse denominador.

Média ponderada é "NaN-aware": se uma usina específica tem gap pontual, o peso
é redistribuído entre as usinas disponíveis naquele instante (em vez de anular
o instante inteiro). Timestamps noturnos (todas as usinas NaN) permanecem NaN.

A série k_agg(t) resultante passa pelo MESMO detector SDT (B5, idênticos ε/Δ/
duração mínima) — a rede é tratada como uma usina qualquer, sem lógica de
detecção nova. Isso evita ter que reconciliar rampas individuais que começam/
terminam em instantes diferentes por usina.

Entradas:
  data/interim/clearsky_index.parquet   (com coluna 'split', de B8)
  data/interim/coords.parquet

Saídas:
  data/interim/aggregate_clearsky_index.parquet   — k_agg(t), split
  data/interim/aggregate_ramps.parquet            — eventos de rampa da rede agregada
  results/figures/f7a_aggregate_series_example.png

Executar:
    python F7a_aggregate_series.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result
from B5_ramp_detection import swinging_door, _sdt_compress_core

IN_K      = cfg.DIRS["interim"] / "clearsky_index.parquet"
IN_COORDS = cfg.DIRS["interim"] / "coords.parquet"
OUT_K_AGG = cfg.DIRS["interim"] / "aggregate_clearsky_index.parquet"
OUT_RAMPS = cfg.DIRS["interim"] / "aggregate_ramps.parquet"
OUT_FIG   = cfg.DIRS["figures"] / "f7a_aggregate_series_example.png"

EPS   = cfg.RAMP["compression_eps"]
DELTA = cfg.RAMP["delta"]
MIN_DURATION_MIN = cfg.RAMP["min_duration_min"]

SEP = "─" * 60


def build_capacity_weights(coords: pd.DataFrame, station_cols: list[str]) -> np.ndarray:
    cap = coords.set_index("station_id")["capacity_dc_kwp"].reindex(station_cols)
    if cap.isna().any():
        missing = cap[cap.isna()].index.tolist()
        print(f"  AVISO: {len(missing)} usinas sem capacidade cadastrada (peso=0): {missing[:5]}...")
        cap = cap.fillna(0.0)
    w = cap.to_numpy(dtype=float)
    total = w.sum()
    return w / total


def weighted_k_agg(K: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Média ponderada NaN-aware: peso redistribuído entre usinas disponíveis a cada t."""
    valid = np.isfinite(K)
    Kz = np.where(valid, K, 0.0)
    weighted_sum = Kz @ w
    weight_total = valid @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        k_agg = np.where(weight_total > 1e-9, weighted_sum / weight_total, np.nan)
    return k_agg


def main() -> None:
    print(SEP)
    print("F7a — SÉRIE AGREGADA DA REDE (usina virtual ponderada por capacidade)")
    print(SEP)

    for p in (IN_K, IN_COORDS):
        if not p.exists():
            print(f"ERRO: {p} não encontrado.")
            sys.exit(1)

    df_k = pd.read_parquet(IN_K)
    coords = pd.read_parquet(IN_COORDS)
    station_cols = [c for c in df_k.columns if c.startswith("ID")]
    split_col = df_k["split"] if "split" in df_k.columns else None
    print(f"  Usinas: {len(station_cols)}  |  Timestamps: {len(df_k):,}")

    w = build_capacity_weights(coords, station_cols)
    print(f"  Pesos de capacidade: soma={w.sum():.4f}  min={w.min():.5f}  max={w.max():.5f}")

    print("\n[1/3] Calculando k_agg(t) = Σ w_i·k_i(t)  (média ponderada NaN-aware)...")
    K = df_k[station_cols].to_numpy(dtype=float)
    k_agg = weighted_k_agg(K, w)
    n_valid = int(np.isfinite(k_agg).sum())
    print(f"  k_agg válido: {n_valid:,}/{len(k_agg):,} timestamps "
          f"({n_valid/len(k_agg):.1%})")
    print(f"  k_agg: mediana={np.nanmedian(k_agg):.4f}  P95={np.nanpercentile(k_agg, 95):.4f}  "
          f"max={np.nanmax(k_agg):.4f}")

    df_agg = pd.DataFrame({"k_agg": k_agg}, index=df_k.index)
    if split_col is not None:
        df_agg["split"] = split_col.values
    df_agg.to_parquet(OUT_K_AGG)
    print(f"  Salvo: {OUT_K_AGG.relative_to(cfg.ROOT)}")

    # ── Detecção de rampa na série agregada (mesmo SDT de B5) ────────────────
    print(f"\n[2/3] Detectando rampas na série agregada (SDT: ε={EPS}, Δ={DELTA}, "
          f"duração_mín={MIN_DURATION_MIN}min)...")
    times = df_k.index
    dt_minutes = pd.Series(times).diff().dropna().mode()[0].total_seconds() / 60
    min_duration_samples = int(round(MIN_DURATION_MIN / dt_minutes))

    _sdt_compress_core(np.array([0.0, 0.1, 0.05]), 0.02)   # warm-up JIT
    t0 = time.time()
    events = swinging_door(k_agg, epsilon=EPS, delta=DELTA,
                            min_duration_samples=min_duration_samples)
    print(f"  {len(events):,} eventos detectados em {time.time()-t0:.1f}s")

    rows = [{
        "station_id": "AGG",
        "start_ts": times[ev["idx_start"]],
        "end_ts": times[ev["idx_end"]],
        "delta_k": ev["delta_k"],
        "direction": ev["direction"],
        "duration_min": (ev["idx_end"] - ev["idx_start"]) * dt_minutes,
    } for ev in events]
    df_ramps = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["station_id", "start_ts", "end_ts", "delta_k", "direction", "duration_min"])

    train_end = pd.Timestamp(cfg.UTRECHT["train_end"], tz="UTC")
    test_start = pd.Timestamp(cfg.UTRECHT["test_start"], tz="UTC")
    df_ramps["split"] = np.where(df_ramps["start_ts"] >= test_start, "test", "train")

    n_train, n_test = (df_ramps["split"] == "train").sum(), (df_ramps["split"] == "test").sum()
    n_days_train = (train_end - times[0]).days
    n_days_test = (times[-1] - test_start).days
    print(f"  Rampas train: {n_train:,} ({n_train/max(n_days_train,1):.2f}/dia)  "
          f"test: {n_test:,} ({n_test/max(n_days_test,1):.2f}/dia)")

    df_ramps.to_parquet(OUT_RAMPS, index=False)
    print(f"  Salvo: {OUT_RAMPS.relative_to(cfg.ROOT)}")

    # ── Comparação: magnitude agregada vs. magnitude típica por usina ────────
    print("\n[3/3] Comparando magnitude agregada vs. individual (efeito-portfólio, checagem crua)...")
    ramps_indiv = pd.read_parquet(cfg.DIRS["interim"] / "ramps_split.parquet")
    ramps_indiv = ramps_indiv[ramps_indiv["split"] == "train"]
    df_ramps_train = df_ramps[df_ramps["split"] == "train"]

    for direction in ("down", "up"):
        agg_mag = df_ramps_train.loc[df_ramps_train["direction"] == direction, "delta_k"].abs()
        ind_mag = ramps_indiv.loc[ramps_indiv["direction"] == direction, "delta_k"].abs()
        print(f"  [{direction}] agregado: mediana={agg_mag.median():.4f} P95={agg_mag.quantile(0.95):.4f} "
              f"max={agg_mag.max():.4f}  (n={len(agg_mag)})")
        print(f"  [{direction}] individual (por usina): mediana={ind_mag.median():.4f} "
              f"P95={ind_mag.quantile(0.95):.4f} max={ind_mag.max():.4f}  (n={len(ind_mag)})")
        print(f"  [{direction}] razão das medianas (agregado/individual): "
              f"{agg_mag.median()/ind_mag.median():.3f}  "
              "(< 1 esperado sob qualquer nível de diversificação; quanto mais perto de 1, "
              "menos a rede se beneficia da diversificação geográfica)")

    # ── Figura ilustrativa: uma semana de k_agg com rampas destacadas ────────
    week = df_agg.loc["2016-06-01":"2016-06-07", "k_agg"]
    week_ramps = df_ramps_train[(df_ramps_train["start_ts"] >= week.index.min()) &
                                 (df_ramps_train["start_ts"] <= week.index.max())]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(week.index, week.values, color="#2c7bb6", lw=0.8)
    for _, r in week_ramps.iterrows():
        color = "crimson" if r["direction"] == "down" else "seagreen"
        ax.axvspan(r["start_ts"], r["end_ts"], color=color, alpha=0.25)
    ax.set_ylabel("k_agg(t)")
    ax.set_title("F7a — Série agregada da rede (exemplo: 1 semana), rampas destacadas\n"
                  "(vermelho=descida, verde=subida)")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()
    print(f"\n  Figura: {OUT_FIG.relative_to(cfg.ROOT)}")

    log_result(
        script="F7a_aggregate_series.py",
        gate="",
        phase="F7",
        params={
            "aggregation": "capacity-weighted mean of per-station k_i(t), NaN-aware "
                            "(weight redistributed among available stations per timestamp)",
            "n_stations": len(station_cols),
            "sdt_epsilon": EPS, "sdt_delta": DELTA, "min_duration_min": MIN_DURATION_MIN,
            "rationale": "network treated as a single virtual plant; reuses B4's clearsky_index "
                         "(cross-station clearsky variation already shown negligible in B4) and "
                         "B5's exact SDT detector — avoids reconciling per-station ramps with "
                         "different start/end times into an ad hoc network-level event list",
        },
        results={
            "n_ramps_train": int(n_train), "n_ramps_test": int(n_test),
            "rate_per_day_train": round(float(n_train/max(n_days_train,1)), 3),
            "median_agg_down_mag": round(float(df_ramps_train.loc[df_ramps_train['direction']=='down','delta_k'].abs().median()), 4),
            "median_indiv_down_mag": round(float(ramps_indiv.loc[ramps_indiv['direction']=='down','delta_k'].abs().median()), 4),
        },
        decision="N/A — feature extraction step",
        action="Aggregate series feeds F7 (Gate G4: return levels + backtest) and F8 (RQ3 central result).",
        interpretation=(
            "The network is treated as a single capacity-weighted virtual plant: k_agg(t) is the "
            "capacity-weighted mean of the already-computed per-station clearsky index k_i(t), and "
            "the exact same SDT ramp detector (B5, unchanged epsilon/delta/duration) is applied to "
            "this single series. This sidesteps having to reconcile per-station ramp events (which "
            "start/end at different times) into an ad hoc network-level event definition, and reuses "
            "100% of already-validated detection machinery. The aggregate/individual median ramp "
            "magnitude ratio (reported per direction) is a crude, purely descriptive first look at "
            "the portfolio effect -- the formal test (with proper independence counterfactual and "
            "GPD tail comparison) is done in F8."
        ),
        paper_ref="Section 9 — Aggregate Network Ramp (F7/F8); Fig. 8-9; Tab. 4-5",
    )

    print(f"\n{SEP}")
    print("F7a concluído — aggregate_ramps.parquet / aggregate_clearsky_index.parquet disponíveis.")


if __name__ == "__main__":
    main()
