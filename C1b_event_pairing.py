"""
C1b_event_pairing.py — Refinamento evento-a-evento do Gate G1 (ROADMAP Bloco E.2 completo)
============================================================================================
INVESTIGA A RESSALVA do Gate G1: o desenho original (`C1_gate1.py`) pareou rampas
por MÁXIMO DIÁRIO por usina, o que mistura dependência de "regime compartilhado"
(dois dias nublados coincidentes) com dependência "evento-a-evento" (mesma
borda de nuvem passando pelas duas usinas com poucos minutos de diferença).
A checagem física (E.2) mostrou que só 15,4% dos dias de excedência conjunta
tinham horários compatíveis com advecção — logo F5 (Heffernan-Tawn) e F6
(anisotropia/direção) não podem usar o pareamento diário sem investigação
adicional. RESULTADO (ver seção de resultados abaixo): o refinamento não apenas
confirma, mas REFINA e AGRAVA a ressalva — a taxa de coincidência de atividade
(qualquer magnitude) excede fortemente o acaso, mas a dependência de MAGNITUDE
ao nível de evento pareado fica no nível do baseline de independência. Ou seja,
a dependência de cauda do Gate G1 aparenta ser de regime compartilhado, não de
evento-a-evento — achado que precisa ser lido antes de prosseguir para F5/F6.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESENHO METODOLÓGICO — pareamento evento-a-evento
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Para cada par de usinas (i, j) com dist_ij < 5 km (mesmo conjunto de "pares
próximos" do Gate G1):

  1. "Evento extremo" na usina i = rampa com |Δk| acima do quantil u=0.95 da
     PRÓPRIA usina i (calculado sobre TODAS as rampas de treino de i, ambas
     direções). Mesmo quantil de decisão usado no Gate G1 (U_DECISION=0.95).
  2. Janela de coincidência Δt_ij = min(dist_ij / v_nuvem, 30 min) — igual à
     do Gate G1 (E.2), com grade de sensibilidade em v ∈ {10,15,20} m/s.
  3. Para cada evento extremo em i, busca-se a rampa de j mais próxima no
     tempo (qualquer magnitude) dentro de ±Δt_ij. Repete-se simetricamente
     (extremos de j buscando a rampa mais próxima em i) para não enviesar o
     pareamento a favor do catálogo de eventos de uma única estação.
  4. χ_evento(u) := P(evento pareado em j também é extremo | evento extremo
     em i, pareado dentro da janela) — definição CLÁSSICA de χ(u) (Coles
     2001), agora aplicada a pares casados no tempo por evento, não por dia.
     Sob independência, χ_evento(u) → (1-u) = 0.05 (não 0): dado que um
     evento de j caiu por acaso dentro da janela, a chance de ele estar no
     top-(1-u) de j é exatamente 1-u se o pareamento for descorrelacionado
     da magnitude. Este é o baseline correto de comparação (não zero).
  5. Taxa de coincidência (qualquer magnitude) vs. baseline de independência:
     sob processos de Poisson independentes com taxa própria λ_i, λ_j
     (estimada por usina), a probabilidade de achar por acaso >=1 evento de
     j numa janela de largura 2·Δt_ij é p_null = 1 - exp(-λ_j · 2·Δt_ij).
     Comparar a taxa de coincidência OBSERVADA com p_null opera o E.2
     corretamente — com baseline nula — em vez do número descritivo cru
     (15,4%) reportado no Gate G1 original.

Saídas:
  - aligned_pairs.parquet : par a par, cada evento extremo casado (ou não)
    com o evento mais próximo da usina vizinha — ESTE é o artefato que F5
    (Heffernan-Tawn) precisa (X_i > u pareado com Y_j correspondente no tempo).
  - event_pairing_summary.parquet : métricas agregadas por par (χ_evento,
    taxa de coincidência observada vs. nula) para os ~3.070 pares < 5 km.
  - Comparação χ_evento (pooled) vs. χ_hat diário (Gate G1) vs. baseline de
    independência (1-u=0.05) → atualiza a ressalva do Gate G1.

Executar:
    python C1b_event_pairing.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result

RAMPS_PQ    = cfg.DIRS["interim"] / "ramps_split.parquet"
COORDS_PQ   = cfg.DIRS["interim"] / "coords.parquet"
GATE1_PQ    = cfg.DIRS["gates"]   / "gate1_results.parquet"

OUT_ALIGNED = cfg.DIRS["processed"] / "aligned_pairs.parquet"
OUT_SUMMARY = cfg.DIRS["gates"]     / "event_pairing_summary.parquet"
OUT_DEC     = cfg.DIRS["gates"]     / "gate1_event_refinement.md"
OUT_FIG     = cfg.DIRS["figures"]   / "gate1b_chi_event_vs_daily.png"
OUT_FIG2    = cfg.DIRS["figures"]   / "gate1b_coincidence_vs_null.png"

U_DECISION    = 0.95
CLOUD_SPEED   = cfg.G1["cloud_speed_ms"]              # 15.0 m/s
SPEED_GRID    = cfg.G1["cloud_speed_grid"]            # [10,15,20]
WINDOW_CAP_MIN= cfg.G1["coincidence_window_max_min"]  # 30

SEP = "─" * 60


# ─────────────────────────────────────────────────────────────────────────────
# Pareamento evento-a-evento por busca do vizinho mais próximo no tempo
# ─────────────────────────────────────────────────────────────────────────────

def match_nearest(query_ns: np.ndarray, target_ns: np.ndarray, target_mag: np.ndarray,
                   target_dir: np.ndarray, window_min: float):
    """
    Para cada tempo em query_ns (int64, ns), acha o mais próximo em target_ns
    (ordenado, int64 ns) dentro de +/- window_min minutos.
    Retorna (matched_mag, matched_dir, matched_t_ns, dt_min_signed, found)
    alinhados com query_ns. dt_min_signed = (t_partner - t_query) em minutos
    (positivo = parceiro depois; NaN se não houver match).
    """
    n = len(query_ns)
    if len(target_ns) == 0:
        return (np.full(n, np.nan), np.full(n, "", dtype=object),
                np.full(n, 0, dtype=np.int64), np.full(n, np.nan), np.zeros(n, dtype=bool))

    window_ns = window_min * 60e9
    idx = np.searchsorted(target_ns, query_ns)
    idx_lo = np.clip(idx - 1, 0, len(target_ns) - 1)
    idx_hi = np.clip(idx, 0, len(target_ns) - 1)
    dt_lo = np.abs(query_ns - target_ns[idx_lo]).astype(np.float64)
    dt_hi = np.abs(query_ns - target_ns[idx_hi]).astype(np.float64)
    use_hi = dt_hi < dt_lo
    best_idx = np.where(use_hi, idx_hi, idx_lo)
    best_dt_abs = np.where(use_hi, dt_hi, dt_lo)
    found = best_dt_abs <= window_ns
    matched_mag = np.where(found, target_mag[best_idx], np.nan)
    matched_dir = np.where(found, target_dir[best_idx], "")
    matched_t_ns = target_ns[best_idx]
    dt_min_signed = (matched_t_ns.astype(np.float64) - query_ns.astype(np.float64)) / 60e9
    dt_min_signed = np.where(found, dt_min_signed, np.nan)
    return matched_mag, matched_dir, matched_t_ns, dt_min_signed, found


def any_extreme_in_window(query_ns: np.ndarray, target_ns: np.ndarray, target_mag: np.ndarray,
                           threshold: float, window_min: float) -> np.ndarray:
    """
    Para cada tempo em query_ns, verifica se EXISTE (não necessariamente o
    mais próximo) algum evento em target_ns dentro de +/- window_min minutos
    com magnitude > threshold. Complementa `match_nearest`: imune à diluição
    por fragmentação do SDT (uma rampa grande cortada em vários segmentos
    pequenos pode fazer o "vizinho mais próximo" ser um fragmento pequeno,
    mesmo havendo um fragmento grande a poucos minutos de distância).
    """
    n = len(query_ns)
    if len(target_ns) == 0:
        return np.zeros(n, dtype=bool)
    window_ns = window_min * 60e9
    lo = np.searchsorted(target_ns, query_ns - window_ns, side="left")
    hi = np.searchsorted(target_ns, query_ns + window_ns, side="right")
    result = np.zeros(n, dtype=bool)
    has_any = hi > lo
    idx_has = np.where(has_any)[0]
    for k in idx_has:
        result[k] = target_mag[lo[k]:hi[k]].max() > threshold
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP)
    print("C1b — REFINAMENTO EVENTO-A-EVENTO DO GATE G1 (E.2 completo)")
    print(SEP)

    for p in (RAMPS_PQ, COORDS_PQ, GATE1_PQ):
        if not p.exists():
            print(f"\nERRO: {p} não encontrado. Execute C1_gate1.py primeiro.")
            sys.exit(1)

    ramps_all = pd.read_parquet(RAMPS_PQ)
    ramps = ramps_all[ramps_all["split"] == "train"].copy()
    ramps["abs_mag"] = ramps["delta_k"].abs()
    ramps["start_ts"] = pd.to_datetime(ramps["start_ts"])
    ramps["direction"] = np.where(ramps["delta_k"] >= 0, "up", "down")

    gate1_results = pd.read_parquet(GATE1_PQ)
    print(f"\n  Rampas de treino: {len(ramps):,}")
    print(f"  Pares < 5 km (do Gate G1): {len(gate1_results):,}")

    # ── Estruturas por usina: todas as rampas (para casamento) + extremos ────
    print("\n[1/3] Construindo catálogo de eventos por usina (todas as rampas + limiar u=0.95)...")
    stations = sorted(ramps["station_id"].unique())
    by_station = {}
    threshold_by_station = {}
    for sid, g in ramps.groupby("station_id"):
        g = g.sort_values("start_ts")
        times_ns = g["start_ts"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        mags = g["abs_mag"].to_numpy()
        dirs = g["direction"].to_numpy()
        u_thresh = float(np.quantile(mags, U_DECISION))
        span_min = (times_ns[-1] - times_ns[0]) / 60e9 if len(times_ns) > 1 else np.nan
        by_station[sid] = {
            "times_ns": times_ns, "mags": mags, "dirs": dirs,
            "n_total": len(g), "span_min": span_min,
        }
        threshold_by_station[sid] = u_thresh

    print(f"  Usinas: {len(stations)}   Limiar u=0.95 mediano (|Δk|): "
          f"{np.median(list(threshold_by_station.values())):.4f}")

    # ── Pareamento por par próximo ────────────────────────────────────────────
    print(f"\n[2/3] Pareando eventos extremos dentro da janela de coincidência "
          f"(v={CLOUD_SPEED} m/s, teto {WINDOW_CAP_MIN} min)...")

    aligned_records = []
    summary_records = []

    for _, row in gate1_results.iterrows():
        si, sj, dist_m = row["station_i"], row["station_j"], row["dist_ij_m"]
        if si not in by_station or sj not in by_station:
            continue
        Si, Sj = by_station[si], by_station[sj]
        ui, uj = threshold_by_station[si], threshold_by_station[sj]
        dt_window_min = min(dist_m / CLOUD_SPEED / 60.0, WINDOW_CAP_MIN)

        # -- extremos de i casados com evento mais próximo (qualquer mag) de j --
        ext_i_mask = Si["mags"] > ui
        q_times = Si["times_ns"][ext_i_mask]
        q_mags  = Si["mags"][ext_i_mask]
        q_dirs  = Si["dirs"][ext_i_mask]
        m_mag_j, m_dir_j, m_t_j, m_dt_j, found_j = match_nearest(
            q_times, Sj["times_ns"], Sj["mags"], Sj["dirs"], dt_window_min)
        j_extreme = found_j & (m_mag_j > uj)
        j_extreme_any = any_extreme_in_window(q_times, Sj["times_ns"], Sj["mags"], uj, dt_window_min)

        for k in range(len(q_times)):
            aligned_records.append((
                si, sj, pd.Timestamp(q_times[k]), float(q_mags[k]), q_dirs[k],
                pd.Timestamp(m_t_j[k]) if found_j[k] else pd.NaT,
                float(m_mag_j[k]) if found_j[k] else np.nan,
                m_dir_j[k] if found_j[k] else "",
                float(m_dt_j[k]) if found_j[k] else np.nan,
                True, bool(found_j[k]), bool(j_extreme[k]), bool(j_extreme_any[k]),
            ))

        # -- extremos de j casados com evento mais próximo (qualquer mag) de i --
        ext_j_mask = Sj["mags"] > uj
        q_times2 = Sj["times_ns"][ext_j_mask]
        q_mags2  = Sj["mags"][ext_j_mask]
        q_dirs2  = Sj["dirs"][ext_j_mask]
        m_mag_i, m_dir_i, m_t_i, m_dt_i, found_i = match_nearest(
            q_times2, Si["times_ns"], Si["mags"], Si["dirs"], dt_window_min)
        i_extreme = found_i & (m_mag_i > ui)
        i_extreme_any = any_extreme_in_window(q_times2, Si["times_ns"], Si["mags"], ui, dt_window_min)

        for k in range(len(q_times2)):
            aligned_records.append((
                sj, si, pd.Timestamp(q_times2[k]), float(q_mags2[k]), q_dirs2[k],
                pd.Timestamp(m_t_i[k]) if found_i[k] else pd.NaT,
                float(m_mag_i[k]) if found_i[k] else np.nan,
                m_dir_i[k] if found_i[k] else "",
                float(m_dt_i[k]) if found_i[k] else np.nan,
                True, bool(found_i[k]), bool(i_extreme[k]), bool(i_extreme_any[k]),
            ))

        n_ext_i, n_ext_j = len(q_times), len(q_times2)
        n_match_i, n_match_j = int(found_j.sum()), int(found_i.sum())
        n_both_i, n_both_j = int(j_extreme.sum()), int(i_extreme.sum())
        n_both_any_i, n_both_any_j = int(j_extreme_any.sum()), int(i_extreme_any.sum())

        denom = n_ext_i + n_ext_j
        chi_event = (n_both_i + n_both_j) / denom if denom > 0 else np.nan
        chi_event_any = (n_both_any_i + n_both_any_j) / denom if denom > 0 else np.nan
        frac_matched = (n_match_i + n_match_j) / denom if denom > 0 else np.nan

        lam_i = Si["n_total"] / Si["span_min"] if Si["span_min"] else np.nan
        lam_j = Sj["n_total"] / Sj["span_min"] if Sj["span_min"] else np.nan
        p_null_i = 1 - np.exp(-lam_j * 2 * dt_window_min) if np.isfinite(lam_j) else np.nan  # match i->j
        p_null_j = 1 - np.exp(-lam_i * 2 * dt_window_min) if np.isfinite(lam_i) else np.nan  # match j->i
        p_null = (p_null_i * n_ext_i + p_null_j * n_ext_j) / denom if denom > 0 else np.nan

        summary_records.append({
            "station_i": si, "station_j": sj, "dist_ij_m": dist_m,
            "dt_window_min": dt_window_min,
            "n_extreme_i": n_ext_i, "n_extreme_j": n_ext_j,
            "n_matched": n_match_i + n_match_j, "frac_matched": frac_matched,
            "p_null_coincidence": p_null,
            "n_both_extreme": n_both_i + n_both_j,
            "chi_event_hat": chi_event,
            "n_both_extreme_any": n_both_any_i + n_both_any_j,
            "chi_event_any_hat": chi_event_any,
            "chi_hat_daily": row["chi_hat"],
            "significant_daily": row["significant"],
        })

    aligned_pairs = pd.DataFrame(aligned_records, columns=[
        "station_ext", "station_partner", "t_ext", "mag_ext", "dir_ext",
        "t_partner", "mag_partner", "dir_partner", "dt_min",
        "ext_is_extreme", "matched", "partner_is_extreme", "partner_extreme_anywhere_in_window",
    ])
    summary = pd.DataFrame(summary_records)

    cfg.DIRS["processed"].mkdir(parents=True, exist_ok=True)
    cfg.DIRS["gates"].mkdir(parents=True, exist_ok=True)
    aligned_pairs.to_parquet(OUT_ALIGNED, index=False)
    summary.to_parquet(OUT_SUMMARY, index=False)
    print(f"  Salvo: {OUT_ALIGNED.relative_to(cfg.ROOT)}  ({len(aligned_pairs):,} eventos extremos avaliados)")
    print(f"  Salvo: {OUT_SUMMARY.relative_to(cfg.ROOT)}  ({len(summary):,} pares)")

    # ── Comparação agregada ────────────────────────────────────────────────────
    print("\n[3/3] Comparando χ_evento com χ diário (Gate G1) e baseline de independência...")
    valid = summary.dropna(subset=["chi_event_hat"])
    n_ext_total = (valid["n_extreme_i"] + valid["n_extreme_j"]).sum()
    chi_event_pooled = valid["n_both_extreme"].sum() / n_ext_total
    chi_event_any_pooled = valid["n_both_extreme_any"].sum() / n_ext_total
    chi_event_median = valid["chi_event_hat"].median()
    chi_event_any_median = valid["chi_event_any_hat"].median()
    chi_daily_median = valid["chi_hat_daily"].median()
    chi_null = 1 - U_DECISION

    frac_matched_mean = valid["frac_matched"].mean()
    p_null_mean = valid["p_null_coincidence"].mean()
    excess_coincidence = frac_matched_mean - p_null_mean

    corr_event_daily = valid["chi_event_hat"].corr(valid["chi_hat_daily"], method="spearman")
    corr_event_any_daily = valid["chi_event_any_hat"].corr(valid["chi_hat_daily"], method="spearman")

    print(f"  χ_evento(u=0.95) pooled — vizinho MAIS PRÓXIMO no tempo:  {chi_event_pooled:.4f}")
    print(f"  χ_evento(u=0.95) pooled — QUALQUER extremo na janela:     {chi_event_any_pooled:.4f}")
    print(f"  χ_hat diário (Gate G1) mediano — mesmos pares:            {chi_daily_median:.4f}")
    print(f"  Baseline de independência (1-u):                          {chi_null:.4f}")
    print(f"  Corr. Spearman χ_evento(nearest) vs χ_diário:             {corr_event_daily:.3f}")
    print(f"  Corr. Spearman χ_evento(any-in-window) vs χ_diário:       {corr_event_any_daily:.3f}")
    print(f"\n  Taxa de coincidência observada (qualquer magnitude): {frac_matched_mean:.3f}")
    print(f"  Taxa de coincidência esperada sob independência:      {p_null_mean:.3f}")
    print(f"  Excesso de coincidência (observado - nulo):           {excess_coincidence:.3f}")

    # ── Decisão / atualização da ressalva do Gate G1 ─────────────────────────
    # Duas métricas de chi_evento foram calculadas: "nearest" (vizinho mais próximo
    # no tempo) e "any-in-window" (existe algum evento extremo do vizinho na
    # janela). Elas saíram QUASE IDÊNTICAS (26.227 vs 26.233 em ~736k avaliações)
    # — ou seja, a hipótese de que o casamento por vizinho-mais-próximo seria
    # diluído por fragmentação do SDT NÃO se confirmou na prática: dentro da
    # janela de coincidência (tipicamente pequena, escalada pela distância), quase
    # sempre há no máximo um candidato relevante. O achado real é outro (ver abaixo).
    chi_event_signal = chi_event_any_pooled > chi_null
    activity_excess = excess_coincidence > 0.03
    corroborates = chi_event_any_pooled > 2 * chi_null and activity_excess

    if corroborates:
        decision = "REFINAMENTO CORROBORA G1"
        action = ("χ_evento >> baseline de independência E taxa de coincidência >> nula: a dependência de "
                   "cauda detectada no Gate G1 tem componente evento-a-evento genuína. F5 (Heffernan-Tawn) "
                   "pode usar aligned_pairs.parquet diretamente como pareamento evento-a-evento.")
    elif activity_excess and not chi_event_signal:
        decision = "REFINAMENTO PARCIAL — DEPENDÊNCIA É DE REGIME, NÃO DE EVENTO"
        action = ("ACHADO IMPORTANTE: a taxa de coincidência de atividade (qualquer magnitude) excede "
                   f"fortemente o esperado por acaso (+{excess_coincidence:.1%}), confirmando que dias/"
                   "períodos de muita atividade de rampa são compartilhados entre usinas vizinhas. PORÉM "
                   f"χ_evento ({chi_event_any_pooled:.3f}) fica NO NÍVEL (ou abaixo) do baseline de "
                   f"independência ({chi_null:.2f}): dado que uma rampa extrema ocorre em i e existe uma "
                   "rampa qualquer em j dentro da janela de coincidência, essa rampa de j NÃO tem chance "
                   "maior que o acaso de também ser extrema. Isso reforça — e agora quantifica melhor — a "
                   "ressalva original do Gate G1: a dependência de cauda diária é predominantemente "
                   "'regime compartilhado' (mais rampas, de todos os tamanhos, em dias/períodos "
                   "meteorologicamente mais variáveis nos dois locais) e NÃO 'mesma borda de nuvem, "
                   "magnitude comparável, poucos minutos de diferença'. RECOMENDAÇÃO para F5/F6: (a) não "
                   "assumir que o pareamento evento-a-evento por proximidade temporal produzirá sinal de "
                   "dependência de magnitude tão forte quanto o bloco diário sugeriu — testar isso "
                   "explicitamente antes de investir no ajuste completo de Heffernan-Tawn; (b) considerar "
                   "modelar a dependência via uma covariável de 'atividade regional' (regime) em vez de, "
                   "ou além de, pareamento evento-a-evento estrito; (c) F6 (anisotropia/velocidade de "
                   "propagação) fica ENFRAQUECIDO por este achado — sem acoplamento de magnitude evento-a-"
                   "evento, estimar direção/velocidade de propagação a partir de defasagens temporais pode "
                   "não ser identificável com este desenho.")
    else:
        decision = "REFINAMENTO PARCIAL"
        action = ("Nem o sinal de atividade nem o de magnitude evento-a-evento foram claros. Investigar "
                   "antes de F5: revisar janela de coincidência, detecção SDT, ou tratar a dependência "
                   "como predominantemente de regime compartilhado em vez de pareamento evento-a-evento "
                   "estrito.")

    print(f"\n  DECISÃO: {decision}")
    print(f"  Ação: {action}")

    # ── Documento de decisão ──────────────────────────────────────────────────
    decision_md = f"""# Gate G1 — Refinamento Evento-a-Evento (C1b, resolve ressalva original)

**Data:** {date.today().isoformat()}
**Decisão:** {decision}

## Motivação
O Gate G1 original (`C1_gate1.py`) pareou rampas por MÁXIMO DIÁRIO, o que mistura
dependência de "regime compartilhado" com dependência evento-a-evento. A checagem
física (E.2) mostrou que só 15,4% dos dias de excedência conjunta eram compatíveis
com advecção — insuficiente para embasar F5/F6 sem refinamento.

## Desenho
- "Evento extremo" na usina = rampa com \\|Δk\\| > quantil u={U_DECISION} da PRÓPRIA
  usina (todas as rampas de treino, ambas direções).
- Cada evento extremo é casado com a rampa mais próxima no tempo da usina vizinha,
  dentro da janela de coincidência Δt_ij = min(dist_ij/v, {WINDOW_CAP_MIN} min),
  v={CLOUD_SPEED} m/s. Pareamento feito nos DOIS sentidos (i→j e j→i) e agregado.
- χ_evento(u) := P(evento pareado também extremo \\| evento extremo, pareado dentro
  da janela) — definição clássica de χ(u), agora em pares casados por evento.
  Baseline de independência = 1-u = {chi_null:.2f} (não zero).
  Duas variantes: **nearest** (vizinho mais próximo no tempo) e **any-in-window**
  (existe algum evento extremo do vizinho na janela, imune à fragmentação do SDT).
- Taxa de coincidência (qualquer magnitude) comparada a um baseline nulo de
  processos de Poisson independentes com taxa própria por usina.

## Resultados
| Métrica | Valor |
|---|---|
| Pares avaliados | {len(valid):,} |
| χ_evento(u={U_DECISION}) pooled — nearest | {chi_event_pooled:.4f} |
| χ_evento(u={U_DECISION}) pooled — any-in-window | {chi_event_any_pooled:.4f} |
| χ_evento(u={U_DECISION}) mediano por par — nearest | {chi_event_median:.4f} |
| χ_evento(u={U_DECISION}) mediano por par — any-in-window | {chi_event_any_median:.4f} |
| χ_hat diário (Gate G1), mesmos pares | {chi_daily_median:.4f} |
| Baseline de independência (1-u) | {chi_null:.4f} |
| Correlação Spearman χ_evento(nearest) × χ_diário | {corr_event_daily:.3f} |
| Correlação Spearman χ_evento(any-in-window) × χ_diário | {corr_event_any_daily:.3f} |
| Taxa de coincidência observada | {frac_matched_mean:.3f} |
| Taxa de coincidência esperada (nula) | {p_null_mean:.3f} |
| Excesso de coincidência | {excess_coincidence:.3f} |

**Nota:** as variantes nearest/any-in-window de χ_evento saíram quase idênticas
({int(valid['n_both_extreme'].sum()):,} vs {int(valid['n_both_extreme_any'].sum()):,}
casamentos extremos, de {len(aligned_pairs):,} eventos avaliados) — a hipótese de
diluição por fragmentação do SDT NÃO se confirmou; dentro da janela de coincidência
há tipicamente no máximo um candidato relevante do vizinho.

## Decisão
**{decision}**

{action}

## Referência cruzada
- Fig.: `results/figures/gate1b_chi_event_vs_daily.png`
- Fig.: `results/figures/gate1b_coincidence_vs_null.png`
- Dados: `results/processed/aligned_pairs.parquet` (usar em F5),
  `results/gates/event_pairing_summary.parquet`
"""
    OUT_DEC.write_text(decision_md)
    print(f"  Salvo: {OUT_DEC.relative_to(cfg.ROOT)}")

    # ── Logging estruturado ───────────────────────────────────────────────────
    log_result(
        script="C1b_event_pairing.py",
        gate="G1",
        params={
            "u_decision": U_DECISION,
            "cloud_speed_ms": CLOUD_SPEED,
            "coincidence_window_max_min": WINDOW_CAP_MIN,
            "extreme_definition": "per-station 95th pctile of |delta_k|, both directions pooled",
            "matching": "nearest-in-time, symmetrized (i->j and j->i), within coincidence window",
        },
        results={
            "n_pairs_evaluated": len(valid),
            "chi_event_nearest_pooled": round(float(chi_event_pooled), 4),
            "chi_event_any_in_window_pooled": round(float(chi_event_any_pooled), 4),
            "chi_daily_median_same_pairs": round(float(chi_daily_median), 4),
            "chi_independence_baseline": round(float(chi_null), 4),
            "spearman_corr_nearest_vs_daily": round(float(corr_event_daily), 3),
            "spearman_corr_any_vs_daily": round(float(corr_event_any_daily), 3),
            "frac_matched_observed": round(float(frac_matched_mean), 3),
            "frac_matched_null": round(float(p_null_mean), 3),
            "excess_coincidence": round(float(excess_coincidence), 3),
        },
        decision=decision,
        action=action,
        interpretation=(
            f"This analysis attempts to resolve the Gate G1 caveat by rebuilding the paired series at the "
            f"EVENT level within the physically-justified coincidence window, instead of daily block "
            f"maxima. Two matching rules were tested -- nearest-in-time and any-extreme-in-window (does "
            "the neighbour have ANY extreme event within the window, not just the temporally closest one) "
            "-- as a check against SDT fragmentation possibly diluting nearest-in-time matches (B5b). They "
            f"turned out to be nearly identical ({int(valid['n_both_extreme'].sum())} vs "
            f"{int(valid['n_both_extreme_any'].sum())} matches out of {len(aligned_pairs):,} extreme "
            "events evaluated) -- fragmentation is NOT the dominant effect here; within the (typically "
            "small, distance-scaled) coincidence window there is essentially at most one relevant "
            f"candidate event at the neighbour. chi_event(u={U_DECISION}) pooled across {len(valid):,} "
            f"close pairs is {chi_event_any_pooled:.3f} -- AT OR BELOW the independence baseline of "
            f"{chi_null:.2f} (=1-u): conditional on i having an extreme ramp and finding a temporally "
            "coincident ramp at neighbouring j, that matched ramp is NOT more likely than chance to also "
            f"be extreme. This contrasts sharply with the any-magnitude coincidence RATE, which strongly "
            f"exceeds its independence-Poisson null ({frac_matched_mean:.1%} observed vs "
            f"{p_null_mean:.1%} expected, excess={excess_coincidence:.1%}) -- confirming genuine "
            "excess co-occurrence of ramp ACTIVITY between nearby stations (consistent with shared "
            "cloud-variable weather regimes), even though this activity coupling does NOT translate into "
            "magnitude-conditional tail dependence at the matched-event level. TOGETHER, this refines "
            "(rather than simply confirms) the original Gate G1 caveat with a sharper, quantified "
            "diagnosis: the daily-block chi>0 finding from Gate G1 is predominantly a SHARED-REGIME effect "
            "(more ramps of all sizes on the same variable-weather days/periods at both stations), not "
            "'same cloud edge, comparable magnitude, minutes apart' event coupling. PRACTICAL CONSEQUENCE "
            "FOR F5/F6: naive event-level Heffernan-Tawn conditioning (X_i>u paired with the nearest-in-"
            "time Y_j) should NOT be assumed to reproduce the daily-block dependence strength -- this "
            "should be tested explicitly before committing to the full HT fit; F6 (anisotropy/propagation-"
            "speed regression from event-level time lags) is weakened by this finding, since without "
            "magnitude coupling at the event level, lag-based direction/speed estimates may not be well "
            "identified. A regime-level covariate (e.g. regional ramp activity in a given window) may be "
            "a more productive way to carry the G1 dependence signal into the modelling phases than strict "
            "event-to-event pairing. aligned_pairs.parquet is nonetheless saved as the full event-level "
            "matched dataset (with both 'partner_is_extreme' and 'partner_extreme_anywhere_in_window' "
            "flags) for these follow-up tests. The daily-block gate1_results.parquet remains the record of "
            "the original G1 go/no-go decision, which stands unaffected -- this is an additional diagnostic "
            "for the NEXT phases, not a re-litigation of G1 itself."
        ),
        paper_ref=(
            "Section 6 — Gate G1 Tail Dependence Diagnostic, Event-Level Refinement; "
            "gate1_event_refinement.md; aligned_pairs.parquet (input to F5/F6)"
        ),
    )

    # ── Figuras ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    lims = [0, max(valid["chi_hat_daily"].max(), valid["chi_event_any_hat"].max()) * 1.05]
    for ax, col, corr, label in [
        (axes[0], "chi_event_hat", corr_event_daily, "nearest-in-time"),
        (axes[1], "chi_event_any_hat", corr_event_any_daily, "any-in-window"),
    ]:
        ax.scatter(valid["chi_hat_daily"], valid[col], s=8, alpha=0.3, color="#2c7bb6")
        ax.plot(lims, lims, color="grey", lw=1, linestyle="--", label="y = x")
        ax.axhline(chi_null, color="crimson", lw=1, linestyle=":",
                   label=f"baseline independência (1-u={chi_null:.2f})")
        ax.set_xlabel("χ̂ diário (Gate G1, bloco de máximo diário)")
        ax.set_ylabel(f"χ̂ evento ({label})")
        ax.set_title(f"{label} — Spearman ρ={corr:.2f}")
        ax.legend(fontsize=8)
        ax.set_xlim(lims); ax.set_ylim(lims)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()
    print(f"  Figura: {OUT_FIG.relative_to(cfg.ROOT)}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(valid["dist_ij_m"] / 1000, valid["frac_matched"], s=8, alpha=0.3,
               color="#2c7bb6", label="taxa de coincidência observada")
    ax.scatter(valid["dist_ij_m"] / 1000, valid["p_null_coincidence"], s=8, alpha=0.3,
               color="grey", label="taxa esperada sob independência (nula)")
    ax.set_xlabel("Distância entre usinas (km)")
    ax.set_ylabel("Taxa de coincidência (qualquer magnitude)")
    ax.set_title("Coincidência observada vs. baseline nulo (Poisson independente)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG2, dpi=150)
    plt.close()
    print(f"  Figura: {OUT_FIG2.relative_to(cfg.ROOT)}")

    print(f"\n{SEP}")
    print(f"Refinamento G1 (evento-a-evento) — {decision}")
    print("Próximo passo: F4/F5 usando aligned_pairs.parquet para pareamento evento-a-evento")


if __name__ == "__main__":
    main()
