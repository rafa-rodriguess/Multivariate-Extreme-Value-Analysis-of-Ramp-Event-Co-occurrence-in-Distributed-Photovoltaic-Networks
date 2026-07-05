"""
B5_ramp_detection.py — Detecção de ramp events via Swinging-Door (SDT)
========================================================================
Implementa o algoritmo Swinging-Door Trending (SDT) verdadeiro (Bristol, 1990),
usado na literatura de detecção de rampa solar/eólica (Florita, Hodge & Orwig,
2013 — "Identifying Wind and Solar Ramping Events") para comprimir k_i(t) em
segmentos lineares de tendência, dos quais os eventos de rampa são extraídos.

Algoritmo (duas fases):

  FASE 1 — Compressão (parâmetro: ε, tolerância de compressão):
    Mantém um ponto "arquivado" e uma "porta" (par de limites de inclinação)
    que se estreita a cada novo ponto recebido. Quando a porta fecha
    (upper_slope < lower_slope), o último ponto compatível vira um novo
    ponto arquivado — um segmento de tendência real foi encontrado.
    Resultado: uma sequência de breakpoints (t_j, k_j) que aproxima a série
    original dentro da tolerância ε.

  FASE 2 — Extração de rampa (parâmetro: Δ, magnitude mínima):
    Cada segmento entre dois breakpoints consecutivos é candidato a rampa.
    Se |k_{j+1} - k_j| ≥ Δ, é registrado como evento.

Por que isso é melhor que um limiar em janela fixa:
  - As fronteiras de segmento são pontos REAIS de mudança de tendência na
    série, não artefatos de uma janela arbitrária.
  - ε filtra ruído de sensor/nuvem residual antes de qualquer decisão de
    rampa — Δ não precisa fazer esse trabalho sozinho.
  - Duração do evento é uma saída natural do algoritmo, não uma gambiarra
    de merge de janelas sobrepostas (versão anterior deste script).

Gaps noturnos (NaN em k_i) quebram a compressão — cada trecho contíguo de
dados válidos é comprimido independentemente; rampas nunca "atravessam" a
noite artificialmente.

Por padrão usa os parâmetros de cfg:
  ε = cfg.RAMP["compression_eps"]  (tolerância de compressão SDT)
  Δ = cfg.RAMP["delta"]            (magnitude mínima de rampa)

Saídas:
  data/interim/ramps.parquet       — evento por (station_id, start_ts)
  data/interim/ramp_report.csv     — resumo por usina

Formato de ramps.parquet:
  station_id | start_ts | end_ts | delta_k | direction | duration_min

Executar:
    python B5_ramp_detection.py [--epsilon FLOAT] [--delta FLOAT]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result

IN_K    = cfg.DIRS["interim"] / "clearsky_index.parquet"
OUT_PQ  = cfg.DIRS["interim"] / "ramps.parquet"
OUT_REP = cfg.DIRS["interim"] / "ramp_report.csv"

SEP = "─" * 60


# ── Fase 1: Compressão Swinging-Door (SDT), JIT-compilada ────────────────────

@njit(cache=True)
def _sdt_compress_core(v: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Compressão Swinging-Door Trending sobre uma série SEM gaps (sem NaN).
    Assume amostragem regular — usa índice inteiro como eixo temporal.

    Retorna array de índices (em `v`) dos pontos arquivados (breakpoints),
    sempre incluindo o primeiro e o último ponto da série.
    """
    n = len(v)
    breakpoints = np.empty(n, dtype=np.int64)
    if n == 0:
        return breakpoints[:0]
    if n == 1:
        breakpoints[0] = 0
        return breakpoints[:1]

    n_bp = 1
    breakpoints[0] = 0
    archived_idx = 0
    upper_slope = np.inf
    lower_slope = -np.inf
    snap_idx = 1

    for i in range(1, n):
        dt = i - archived_idx
        test_upper = ((v[i] + epsilon) - v[archived_idx]) / dt
        test_lower = ((v[i] - epsilon) - v[archived_idx]) / dt

        if i == archived_idx + 1:
            upper_slope = test_upper
            lower_slope = test_lower
            snap_idx = i
            continue

        if test_upper < lower_slope or test_lower > upper_slope:
            # Porta fechou: arquivar o último ponto compatível (snap)
            breakpoints[n_bp] = snap_idx
            n_bp += 1
            archived_idx = snap_idx
            dt2 = i - archived_idx
            upper_slope = ((v[i] + epsilon) - v[archived_idx]) / dt2
            lower_slope = ((v[i] - epsilon) - v[archived_idx]) / dt2
            snap_idx = i
        else:
            if test_upper < upper_slope:
                upper_slope = test_upper
            if test_lower > lower_slope:
                lower_slope = test_lower
            snap_idx = i

    if breakpoints[n_bp - 1] != snap_idx:
        breakpoints[n_bp] = snap_idx
        n_bp += 1

    return breakpoints[:n_bp]


def _find_valid_runs(k: np.ndarray, min_len: int = 2):
    """Gera (start, end) [exclusivo] de trechos contíguos sem NaN."""
    is_valid = ~np.isnan(k)
    if not is_valid.any():
        return
    padded = np.concatenate(([False], is_valid, [False]))
    edges  = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]
    for s, e in zip(starts, ends):
        if e - s >= min_len:
            yield s, e


# ── Fase 2: Extração de rampas a partir da compressão ─────────────────────────

def swinging_door(k: np.ndarray, epsilon: float, delta: float,
                   min_duration_samples: int = 0) -> list[dict]:
    """
    Detecta eventos de rampa via SDT verdadeiro.

    Uma rampa exige DOIS critérios simultâneos, como na literatura de ramp
    events de rede (Florita, Hodge & Orwig 2013; práticas usuais de estudos
    de integração eólica/solar):
      1. Magnitude:  |k[end] - k[start]| ≥ delta
      2. Persistência: duração do segmento ≥ min_duration_samples

    Sem o critério 2, transições de 1-2 amostras causadas por variabilidade
    de nuvem fragmentada (não ruído de sensor, mas variabilidade física de
    altíssima frequência) dominam a contagem e inflam artificialmente a
    taxa de eventos — ver ROADMAP B.5 para a análise que motivou este filtro.

    Parâmetros
    ----------
    k                    : array 1-D de float (k_i, pode conter NaN — gaps noturnos)
    epsilon              : tolerância de compressão SDT (filtra ruído de sensor)
    delta                : magnitude mínima de rampa sobre o segmento comprimido
    min_duration_samples : duração mínima do segmento, em amostras

    Retorna
    -------
    Lista de dicts com chaves: idx_start, idx_end, delta_k, direction
    """
    events = []
    for s, e in _find_valid_runs(k, min_len=2):
        seg = k[s:e].astype(np.float64)
        bp  = _sdt_compress_core(seg, epsilon)
        for j in range(len(bp) - 1):
            i0, i1 = bp[j], bp[j + 1]
            if (i1 - i0) < min_duration_samples:
                continue
            v0, v1 = seg[i0], seg[i1]
            dv = v1 - v0
            if abs(dv) >= delta:
                events.append({
                    "idx_start": s + i0,
                    "idx_end":   s + i1,
                    "delta_k":   dv,
                    "direction": "up" if dv > 0 else "down",
                })
    return events


# ── Main ──────────────────────────────────────────────────────────────────────

def main(epsilon: float, delta: float, min_duration_min: float) -> None:
    print(SEP)
    print(f"B5 — DETECÇÃO DE RAMPS  [SDT verdadeiro: ε={epsilon:.3f}, Δ={delta:.3f}, "
          f"duração_mín={min_duration_min:.0f}min]")
    print(SEP)

    if not IN_K.exists():
        print(f"ERRO: {IN_K} não encontrado. Execute B4_clearsky.py primeiro.")
        sys.exit(1)

    df_k = pd.read_parquet(IN_K)
    times = df_k.index

    # clearsky_index.parquet pode já conter a coluna 'split' (adicionada por B8
    # em execuções anteriores) — manter apenas colunas de usina (formato IDxxx)
    station_cols = [c for c in df_k.columns if c.startswith("ID")]
    df_k = df_k[station_cols]
    print(f"  Dados: {df_k.shape}  (timestamps × usinas)")

    dt_minutes = pd.Series(times).diff().dropna().mode()[0].total_seconds() / 60
    print(f"  Resolução temporal: {dt_minutes:.0f} min")
    min_duration_samples = int(round(min_duration_min / dt_minutes))
    print(f"  Duração mínima: {min_duration_min:.0f} min = {min_duration_samples} amostras")

    # Warm-up do JIT (primeira chamada compila; medir tempo à parte)
    import time
    t0 = time.time()
    _sdt_compress_core(np.array([0.0, 0.1, 0.05], dtype=np.float64), 0.02)
    print(f"  JIT warm-up: {time.time() - t0:.2f}s")

    all_events = []
    report_rows = []
    t_start = time.time()

    for i, sid in enumerate(df_k.columns, 1):
        k_arr = df_k[sid].values
        events = swinging_door(k_arr, epsilon=epsilon, delta=delta,
                                min_duration_samples=min_duration_samples)

        rows_station = []
        for ev in events:
            rows_station.append({
                "station_id":    sid,
                "start_ts":      times[ev["idx_start"]],
                "end_ts":        times[ev["idx_end"]],
                "delta_k":       ev["delta_k"],
                "direction":     ev["direction"],
                "duration_min":  (ev["idx_end"] - ev["idx_start"]) * dt_minutes,
            })
        all_events.extend(rows_station)

        n_ev = len(rows_station)
        n_up = sum(1 for e in rows_station if e["direction"] == "up")
        report_rows.append({
            "station_id":   sid,
            "n_ramps":      n_ev,
            "n_ramps_up":   n_up,
            "n_ramps_down": n_ev - n_up,
            "rate_per_day": round(n_ev / max((df_k[sid].notna().sum() * dt_minutes / 1440), 1), 3),
        })

        if i % 25 == 0 or i == len(df_k.columns):
            elapsed = time.time() - t_start
            print(f"  [{i:3d}/{len(df_k.columns)}]  {sid}  eventos_acum={len(all_events):,}  "
                  f"({elapsed:.1f}s decorridos)")

    df_ramps  = pd.DataFrame(all_events) if all_events else pd.DataFrame(
        columns=["station_id", "start_ts", "end_ts", "delta_k", "direction", "duration_min"])
    df_report = pd.DataFrame(report_rows)

    total_ramps = len(df_ramps)
    n_days_total = (times[-1] - times[0]).days
    rate_per_day_per_station = total_ramps / max(len(df_k.columns) * n_days_total, 1)
    print(f"\n  Total de ramp events: {total_ramps:,}")
    print(f"  Usinas com ≥1 evento:  {(df_report['n_ramps'] > 0).sum()}")
    print(f"  Média por usina:       {df_report['n_ramps'].mean():.1f}")
    print(f"  Taxa média:            {rate_per_day_per_station:.2f} rampas/dia/usina")
    print(f"  Período coberto:       {n_days_total} dias")

    if len(df_ramps) > 0:
        dur = df_ramps["duration_min"]
        print(f"\n  Duração dos eventos (min): "
              f"mediana={dur.median():.1f}  P90={dur.quantile(0.90):.1f}  max={dur.max():.1f}")

    df_ramps.to_parquet(OUT_PQ, index=False)
    print(f"\n  Salvo: {OUT_PQ.relative_to(cfg.ROOT)}  ({OUT_PQ.stat().st_size/1e6:.1f} MB)")

    df_report.to_csv(OUT_REP, index=False)
    print(f"  Salvo: {OUT_REP.relative_to(cfg.ROOT)}")

    log_result(
        script = "B5_ramp_detection.py",
        gate   = "",
        phase  = "F0/B5",
        params = {"epsilon": epsilon, "delta": delta, "min_duration_min": min_duration_min,
                  "n_stations": len(df_k.columns),
                  "algorithm": "true SDT (Bristol 1990) via numba JIT + duration filter"},
        results = {
            "total_ramps": total_ramps,
            "mean_ramps_per_station": round(float(df_report["n_ramps"].mean()), 1),
            "rate_per_day_per_station": round(rate_per_day_per_station, 2),
            "duration_median_min": round(float(df_ramps["duration_min"].median()), 1) if total_ramps else None,
            "duration_p90_min": round(float(df_ramps["duration_min"].quantile(0.90)), 1) if total_ramps else None,
            "duration_max_min": round(float(df_ramps["duration_min"].max()), 1) if total_ramps else None,
        },
        decision = "N/A — feature extraction step",
        action   = (
            "Ramp events extracted via true Swinging-Door Trending (SDT) compression + magnitude "
            "threshold + minimum duration filter (10 min), added after diagnosing that the SDT-only "
            "definition (magnitude threshold alone) was dominated by 1-2 minute cloud-edge "
            "variability (63% of events had duration=1 sample), not sustained ramps."
        ),
        interpretation = (
            f"Using the genuine SDT algorithm (Bristol 1990; compression tolerance epsilon={epsilon}) "
            f"with a two-criterion ramp definition — magnitude delta={delta} AND minimum duration "
            f"{min_duration_min} min — {total_ramps:,} ramp events were detected across "
            f"{len(df_k.columns)} stations, averaging {rate_per_day_per_station:.2f} "
            "ramps/day/station. This two-criterion definition (magnitude + persistence) follows "
            "standard practice in the grid-integration ramp literature (e.g., Florita, Hodge & "
            "Orwig 2013) and excludes high-frequency single-minute cloud-edge variability that "
            "an SDT-magnitude-only rule would otherwise misclassify as 'ramps'. A diagnostic run "
            "without the duration filter produced 59.93 ramps/day/station with 63% of events "
            "lasting exactly 1 sample — physically implausible as sustained ramps, motivating "
            "this fix."
        ),
        paper_ref = "Section 3.2 — Ramp Event Detection (Swinging-Door Trending)",
    )

    print(f"\n{SEP}")
    print("B5 concluído — ramps.parquet disponível.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Detecção de ramps via Swinging-Door Trending (SDT)")
    p.add_argument("--epsilon", type=float, default=cfg.RAMP["compression_eps"],
                   help=f"Tolerância de compressão SDT (default: {cfg.RAMP['compression_eps']})")
    p.add_argument("--delta",   type=float, default=cfg.RAMP["delta"],
                   help=f"Magnitude mínima de rampa (default: {cfg.RAMP['delta']})")
    p.add_argument("--min-duration", type=float, default=cfg.RAMP["min_duration_min"],
                   help=f"Duração mínima em minutos (default: {cfg.RAMP['min_duration_min']})")
    args = p.parse_args()
    main(epsilon=args.epsilon, delta=args.delta, min_duration_min=args.min_duration)
