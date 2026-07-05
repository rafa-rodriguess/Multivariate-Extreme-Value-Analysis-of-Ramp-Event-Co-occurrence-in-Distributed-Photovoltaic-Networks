"""
S5_event_pairing_test.py — Pareamento evento-a-evento restrito ao holdout 2017
(ROADMAP.md Risco D, Seção 0.2)
================================================================================
Motivação: `F8_portfolio_effect.py`'s cenário "IMPLICADO PELO MODELO" injeta o
modelo de dois estágios (`F5_two_stage.py`, ajustado sobre `aligned_pairs.parquet`
train-only, n=84.776 casados) NOS MESMOS eventos que o ajustaram — o "desvio
mediano de 4,5%" reportado é qualidade de ajuste IN-SAMPLE, não validação
OUT-OF-SAMPLE genuína (ROADMAP Risco D).

Este script replica EXATAMENTE o desenho de pareamento de `C1b_event_pairing.py`
(mesma janela de coincidência, mesma definição de evento extremo, mesmo
pareamento simetrizado nearest-in-time), mas:
  1. Evento extremo condicionante: rampa de 2017 (`split=="test"`) com |Δk| acima
     do limiar u=0.95 aprendido em TREINO (2014-2016) — o limiar não usa nenhuma
     informação do próprio 2017, exatamente como o limiar da GPD marginal em
     Gate G2/G4 é ajustado no treino e aplicado ao teste.
  2. Pareamento com a rampa mais próxima no tempo do vizinho <5km, buscando
     SOMENTE no catálogo de rampas de 2017 do vizinho (não mistura com 2014-2016)
     — garante que a variável-resposta usada na validação (S6) também é
     genuinamente holdout.

Saída: `results/processed/aligned_pairs_test.parquet` — mesmo schema de
`aligned_pairs.parquet`, mas 100% construído a partir de eventos nunca vistos
por `F5_two_stage.py` (que só usa `split=="train"`).

Não reabre/modifica `C1b_event_pairing.py` (script já aprovado, produz o
artefato de treino usado por F5/F8) — duplica a lógica de pareamento aqui.

Executar:
    python S5_event_pairing_test.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result
from C1b_event_pairing import match_nearest

RAMPS_PQ  = cfg.DIRS["interim"] / "ramps_split.parquet"
GATE1_PQ  = cfg.DIRS["gates"]   / "gate1_results.parquet"
OUT_ALIGNED_TEST = cfg.DIRS["processed"] / "aligned_pairs_test.parquet"
OUT_DEC = cfg.DIRS["gates"] / "s5_event_pairing_test.md"

U_DECISION     = 0.95
CLOUD_SPEED    = cfg.G1["cloud_speed_ms"]
WINDOW_CAP_MIN = cfg.G1["coincidence_window_max_min"]
SEP = "─" * 60


def main() -> None:
    print(SEP)
    print("S5 — PAREAMENTO EVENTO-A-EVENTO RESTRITO AO HOLDOUT 2017 (Risco D)")
    print(SEP)

    for p in (RAMPS_PQ, GATE1_PQ):
        if not p.exists():
            print(f"ERRO: {p} não encontrado.")
            sys.exit(1)

    ramps_all = pd.read_parquet(RAMPS_PQ)
    ramps_all["abs_mag"] = ramps_all["delta_k"].abs()
    ramps_all["start_ts"] = pd.to_datetime(ramps_all["start_ts"])
    ramps_all["direction"] = np.where(ramps_all["delta_k"] >= 0, "up", "down")

    ramps_train = ramps_all[ramps_all["split"] == "train"].copy()
    ramps_test = ramps_all[ramps_all["split"] == "test"].copy()
    gate1_results = pd.read_parquet(GATE1_PQ)
    print(f"\n  Rampas de treino (só para DEFINIR o limiar u={U_DECISION}): {len(ramps_train):,}")
    print(f"  Rampas de teste/holdout 2017 (evento condicionante E catálogo de vizinhos): {len(ramps_test):,}")
    print(f"  Pares < {cfg.G1['max_pair_dist_km']}km (do Gate G1): {len(gate1_results):,}")

    # ── Limiar por usina, aprendido EXCLUSIVAMENTE em treino ──────────────────
    print(f"\n[1/2] Limiar u={U_DECISION} por usina (treino, NUNCA usa 2017)...")
    threshold_by_station = {
        sid: float(np.quantile(g["abs_mag"].to_numpy(), U_DECISION))
        for sid, g in ramps_train.groupby("station_id")
    }
    print(f"  Limiar mediano (|Δk|): {np.median(list(threshold_by_station.values())):.4f}")

    # ── Catálogo de rampas de 2017 por usina (para casamento) ─────────────────
    by_station_test = {}
    for sid, g in ramps_test.groupby("station_id"):
        g = g.sort_values("start_ts")
        by_station_test[sid] = {
            "times_ns": g["start_ts"].to_numpy().astype("datetime64[ns]").astype(np.int64),
            "mags": g["abs_mag"].to_numpy(),
            "dirs": g["direction"].to_numpy(),
        }

    print(f"\n[2/2] Pareando eventos extremos de 2017 (v={CLOUD_SPEED} m/s, teto {WINDOW_CAP_MIN} min)...")
    aligned_records = []
    n_ext_total = 0
    for _, row in gate1_results.iterrows():
        si, sj, dist_m = row["station_i"], row["station_j"], row["dist_ij_m"]
        if si not in by_station_test or sj not in by_station_test or \
           si not in threshold_by_station or sj not in threshold_by_station:
            continue
        Si, Sj = by_station_test[si], by_station_test[sj]
        ui, uj = threshold_by_station[si], threshold_by_station[sj]
        dt_window_min = min(dist_m / CLOUD_SPEED / 60.0, WINDOW_CAP_MIN)

        for src_key, dst_key, Ssrc, Sdst, u_src in ((si, sj, Si, Sj, ui), (sj, si, Sj, Si, uj)):
            ext_mask = Ssrc["mags"] > u_src
            q_times = Ssrc["times_ns"][ext_mask]
            q_mags = Ssrc["mags"][ext_mask]
            q_dirs = Ssrc["dirs"][ext_mask]
            if len(q_times) == 0:
                continue
            m_mag, m_dir, m_t, m_dt, found = match_nearest(
                q_times, Sdst["times_ns"], Sdst["mags"], Sdst["dirs"], dt_window_min)
            n_ext_total += len(q_times)
            for k in range(len(q_times)):
                aligned_records.append((
                    src_key, dst_key, pd.Timestamp(q_times[k]), float(q_mags[k]), q_dirs[k],
                    pd.Timestamp(m_t[k]) if found[k] else pd.NaT,
                    float(m_mag[k]) if found[k] else np.nan,
                    m_dir[k] if found[k] else "",
                    float(m_dt[k]) if found[k] else np.nan,
                    True, bool(found[k]),
                ))

    aligned_test = pd.DataFrame(aligned_records, columns=[
        "station_ext", "station_partner", "t_ext", "mag_ext", "dir_ext",
        "t_partner", "mag_partner", "dir_partner", "dt_min",
        "ext_is_extreme", "matched",
    ])
    cfg.DIRS["processed"].mkdir(parents=True, exist_ok=True)
    aligned_test.to_parquet(OUT_ALIGNED_TEST, index=False)
    n_matched = int(aligned_test["matched"].sum())
    print(f"\n  Eventos extremos condicionantes (2017): {len(aligned_test):,}")
    print(f"  ...dos quais casados com vizinho <5km (2017): {n_matched:,} ({n_matched/len(aligned_test):.1%})")
    print(f"  Salvo: {OUT_ALIGNED_TEST.relative_to(cfg.ROOT)}")

    decision = "S5 CONCLUÍDO — catálogo de pareamento 2017 genuinamente out-of-sample construído"
    action = (f"{len(aligned_test):,} eventos extremos condicionantes de 2017 pareados "
              f"({n_matched:,} casados, {n_matched/len(aligned_test):.1%}), usando limiares de "
              f"extremidade aprendidos EXCLUSIVAMENTE em treino (2014-2016) e catálogo de vizinho "
              f"restrito a 2017 (não mistura com o catálogo de treino). Nenhum destes eventos foi "
              f"usado para ajustar α₀/L/β/γ em `F5_two_stage.py` — pronto para uso em "
              f"`S6_f8_oos_validation.py` como validação genuinamente out-of-sample.")

    OUT_DEC.write_text(f"""# S5 — Pareamento Evento-a-Evento Restrito ao Holdout 2017 (Risco D)

**Data:** {date.today().isoformat()}
**Decisão:** {decision}

## Motivação
`F8_portfolio_effect.py`'s cenário "implicado pelo modelo" injeta o modelo de dois estágios
(ajustado sobre eventos casados de TREINO) nos MESMOS eventos que o ajustaram — o "desvio
mediano de 4,5%" é qualidade de ajuste in-sample, não validação out-of-sample genuína
(ROADMAP.md, Risco D). Este script constrói o catálogo de pareamento necessário para uma
validação out-of-sample genuína, restrito ao holdout 2017.

## Desenho (idêntico a `C1b_event_pairing.py`, restrito a 2017)
- Evento extremo condicionante: rampa de 2017 com \\|Δk\\| > limiar u={U_DECISION} **aprendido em
  treino** (não usa nenhuma informação da distribuição de 2017).
- Pareamento com a rampa mais próxima no tempo do vizinho <{cfg.G1['max_pair_dist_km']}km, buscando
  SOMENTE no catálogo de rampas de 2017 do vizinho — variável-resposta também genuinamente holdout.
- Janela de coincidência Δt_ij = min(dist_ij/v, {WINDOW_CAP_MIN} min), v={CLOUD_SPEED} m/s (idêntico
  ao Gate G1/C1b).

## Resultados
| Métrica | Valor |
|---|---|
| Eventos extremos condicionantes (2017) | {len(aligned_test):,} |
| ...casados com vizinho <5km (2017) | {n_matched:,} ({n_matched/len(aligned_test):.1%}) |
| Limiar u mediano (aprendido em treino) | {np.median(list(threshold_by_station.values())):.4f} |

## Decisão
**{decision}**

{action}

## Referência cruzada
- Saída: `results/processed/aligned_pairs_test.parquet`
- Consumido por: `S6_f8_oos_validation.py`
- Contraste: `results/processed/aligned_pairs.parquet` (treino, usado por F5/F8)
""")
    print(f"  Salvo: {OUT_DEC.relative_to(cfg.ROOT)}")

    log_result(
        script="S5_event_pairing_test.py",
        gate="",
        phase="S5_riscoD",
        params={
            "u_decision": U_DECISION, "cloud_speed_ms": CLOUD_SPEED,
            "coincidence_window_max_min": WINDOW_CAP_MIN,
            "threshold_source": "train-only (2014-2016), applied to 2017 magnitudes",
            "neighbor_catalog_source": "test-only (2017), not mixed with train",
        },
        results={
            "n_conditioning_events_2017": len(aligned_test),
            "n_matched_2017": n_matched,
            "frac_matched_2017": round(n_matched / len(aligned_test), 4) if len(aligned_test) else np.nan,
        },
        decision=decision,
        action=action,
        interpretation=(
            "S5 builds a genuinely out-of-sample event-pairing catalog restricted to the 2017 "
            "holdout, mirroring C1b_event_pairing.py's exact matching design but with the extremity "
            "threshold learned purely on 2014-2016 training data (never touching 2017's own "
            "distribution) and the neighbor-matching catalog restricted to 2017 events only (not "
            "mixed with training-period neighbor ramps). This produces aligned_pairs_test.parquet, "
            "consumed by S6_f8_oos_validation.py to test whether F5_two_stage.py's already-fitted "
            "two-stage model (Stage 1 coincidence GLM + Stage 2 Heffernan-Tawn, both fit exclusively "
            "on training-period matched events) has genuine predictive value on conditioning events "
            "it has never seen, addressing ROADMAP Risco D (the F8 'model-implied' scenario's 4.5% "
            "deviation being an in-sample fit-quality metric, not an out-of-sample validation)."
        ),
        paper_ref="Section 9 (F8 model-implied scenario) -- out-of-sample validation input (Risco D)",
    )

    print(f"\n{SEP}")
    print(f"S5 — {decision}")


if __name__ == "__main__":
    main()
