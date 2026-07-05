"""
S7_f6b_seasonal.py — Estratificação sazonal do teste de advecção F6b (Aprimoramento G)
=======================================================================================
Replica a análise central de `F6b_wind_timing.py` (correlação atraso_observado vs.
atraso_esperado por advecção pura) estratificada por estação do ano, para verificar
se o resultado nulo agregado esconde heterogeneidade sazonal.

Hipótese física pré-especificada (Passo 1 do resolucao_gaps.md):
  Convecção de verão (JJA) pode gerar nuvens convectivas de menor escala transportadas
  mais visivelmente pelo vento do que frentes estratiformes de inverno → possível sinal
  de advecção cronometrável APENAS em JJA que o agregado anual dilui.

Dois tipos de análise:
  (a) PRINCIPAL pré-especificada: JJA vs. Resto do ano (DJF+MAM+SON)
  (b) EXPLORATÓRIA: 4 estações (DJF / MAM / JJA / SON)

Nota sobre o sinal negativo de CERRA em JJA (resultado rodada anterior, esperado):
  F6_anisotropy.py já havia documentado sinal invertido em CERRA 200/500m: coincidência
  MAIOR quando o vento sopra CONTRA o par, mais forte com a altura — confundimento provável
  com padrão sinótico/sazonal (ventos de oeste dominantes em episódios de inverno capturam
  covariância com regime, não transporte). Em JJA o ρ negativo de CERRA é a versão sazonal
  desse mesmo confundimento: o grupo JJA segmenta justamente o período de menor frequência
  de ventos de oeste, amplificando o desvio do padrão de confundimento sem introduzir sinal
  de advecção real.

Filtro de colocação temporal CERRA (CERRA_COLOCATION_WIN_MIN, decisão 2026-07-02):
  CERRA tem resolução de 3h — o vento usado para uma rampa pode ter sido observado até
  100 min antes/depois (tolerância atual de wind_joined.parquet). Para mitigar o efeito de
  temporal desalinhamento, especialmente em JJA onde rampas são mais curtas (convectivas),
  os resultados são rodados em DOIS modos:
    (A) SEM filtro (colocation_win=None) → replica modo original de F6b (todos os eventos)
    (B) COM filtro ±30 min em torno do snapshot CERRA → apenas rampas temporalmente
        colocadas com a observação de vento (decisão de design documentada em ROADMAP.md)
  Para KNMI horário (passo=60min), ±30min cobre a janela inteira → filtro sem efeito.

NÃO reabre/modifica F6b_wind_timing.py (já aprovado).

Saídas:
  results/gates/s7_timing_{season}_{height}.md              — sem filtro
  results/gates/s7_timing_{season}_{height}_coloc30.md      — com filtro ±30min CERRA
  results/gates/s7_seasonal_comparison.md                   — tabela resumo (ambos modos)
  results/figures/s7_seasonal_comparison.png                — heatmap ρ (ambos modos)

Executar:
    python S7_f6b_seasonal.py
"""

from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result
from F6_anisotropy import bearing_deg, circular_alignment

# ── Entradas (mesmas de F6b) ──────────────────────────────────────────────────
ALIGNED_PQ   = cfg.DIRS["processed"] / "aligned_pairs.parquet"
EVENTPAIR_PQ = cfg.DIRS["gates"]     / "event_pairing_summary.parquet"
COORDS_PQ    = cfg.DIRS["interim"]   / "coords.parquet"
WIND_PQ      = cfg.DIRS["interim"]   / "wind_joined.parquet"

# ── Parâmetros (mesmos de F6b) ────────────────────────────────────────────────
N_BOOT        = cfg.F6["n_bootstrap_pairs"]    # 150
MIN_VALONG_MS = cfg.F6["min_valong_ms"]        # 1.0 m/s
MIN_CORR_MEAN = cfg.F6["min_corr_meaningful"]  # 0.10
SEED          = cfg.SEED

# ── Filtro de colocação temporal CERRA (decisão 2026-07-02) ──────────────────
# Passo do CERRA: 3h = 180 min. None = sem filtro (modo original F6b).
CERRA_STEP_MIN   = 180
CERRA_STEP_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]  # UTC
CERRA_COLOCATION_WIN_MIN = 30    # ±30 min em torno de cada snapshot CERRA
KNMI_IS_CERRA = False            # KNMI (10m/horário) não precisa do filtro


def _dist_to_cerra_snapshot(t_utc: pd.Series) -> pd.Series:
    """Minutos entre cada timestamp e o snapshot CERRA 3h mais próximo."""
    mins_since_midnight = t_utc.dt.hour * 60 + t_utc.dt.minute
    anchors = np.array([h * 60 for h in CERRA_STEP_HOURS])
    dists = np.abs(mins_since_midnight.to_numpy()[:, None] - anchors[None, :])
    return pd.Series(dists.min(axis=1), index=t_utc.index)

SEP = "─" * 60

# ── Fontes de vento disponíveis (mesmas de F6b) ───────────────────────────────
WIND_SOURCES = [
    ("wind_speed_ms",            "wind_dir_deg",            "KNMI De Bilt, 10m", "10m"),
    ("cerra_wind_speed_ms_100m", "cerra_wind_dir_deg_100m", "CERRA 100m",         "100m"),
    ("cerra_wind_speed_ms_200m", "cerra_wind_dir_deg_200m", "CERRA 200m",         "200m"),
    ("cerra_wind_speed_ms_500m", "cerra_wind_dir_deg_500m", "CERRA 500m",         "500m"),
]

# ── Grupos sazonais ───────────────────────────────────────────────────────────
# (a) Principal pré-especificada
JJA_MONTHS  = {6, 7, 8}
REST_MONTHS = {1, 2, 3, 4, 5, 9, 10, 11, 12}

# (b) Exploratória (4 estações)
SEASON_GROUPS_EXPL = [
    ("DJF", {12, 1, 2},  "Inverno (DJF)"),
    ("MAM", {3, 4, 5},   "Primavera (MAM)"),
    ("JJA", {6, 7, 8},   "Verão (JJA)"),
    ("SON", {9, 10, 11}, "Outono (SON)"),
]

# Grupos finais: principal + exploratórios (sem duplicar JJA)
ALL_SEASON_GROUPS: list[tuple[str, set, str, bool]] = [
    ("JJA",  JJA_MONTHS,  "Verão (JJA) — principal",           True),   # is_primary
    ("REST", REST_MONTHS, "Resto do ano (DJF+MAM+SON) — principal", True),
    ("DJF",  {12, 1, 2},  "Inverno (DJF) — exploratório",      False),
    ("MAM",  {3, 4, 5},   "Primavera (MAM) — exploratório",    False),
    ("SON",  {9, 10, 11}, "Outono (SON) — exploratório",       False),
]


def _run_one(matched_season: pd.DataFrame, wind: pd.DataFrame,
             speed_col: str, dir_col: str, wind_label: str, wind_suffix: str,
             season_key: str, season_label: str, is_primary: bool,
             is_cerra: bool = False, coloc_win: int | None = None) -> dict | None:
    """
    Núcleo da análise de timing de advecção para um subgrupo sazonal × fonte de vento.
    Replica a lógica essencial de F6b_wind_timing.run_for_source com nomes de arquivo
    prefixados por 's7_{season_key}_{wind_suffix}'.

    is_cerra: True para fontes CERRA (sujeitas ao filtro de colocação temporal).
    coloc_win: se não None e is_cerra=True, restringe eventos a ±coloc_win min do
               snapshot CERRA mais próximo (implementa decisão de design 2026-07-02).
    """
    coloc_tag = f"_coloc{coloc_win}" if (is_cerra and coloc_win is not None) else ""
    tag = f"{season_key}_{wind_suffix}{coloc_tag}"
    OUT_MD  = cfg.DIRS["gates"]   / f"s7_timing_{tag}.md"
    OUT_PQ  = cfg.DIRS["gates"]   / f"s7_timing_{tag}.parquet"
    OUT_FIG = cfg.DIRS["figures"] / f"s7_scatter_{tag}.png"

    if speed_col not in wind.columns or dir_col not in wind.columns:
        print(f"  AVISO: '{speed_col}'/'{dir_col}' não disponível — pulando.")
        return None

    matched = matched_season.copy()

    # Filtro de colocação temporal CERRA (±coloc_win min do snapshot mais próximo)
    coloc_applied = False
    n_before_coloc = len(matched)
    if is_cerra and coloc_win is not None:
        dist_min = _dist_to_cerra_snapshot(matched["t_ext"])
        matched = matched[dist_min <= coloc_win].copy()
        coloc_applied = True
        print(f"  Filtro colocação temporal ±{coloc_win}min (CERRA): "
              f"{len(matched):,}/{n_before_coloc:,} eventos retidos "
              f"({len(matched)/n_before_coloc:.1%})")

    n_season = len(matched)
    if n_season < 50:
        print(f"  AVISO: amostra muito pequena ({n_season} eventos) — pulando {tag}.")
        return None

    # Casar vento no instante do evento condicionante
    wind_lookup = wind[["station_id", "start_ts", speed_col, dir_col]].copy()
    wind_lookup["start_ts"] = pd.to_datetime(wind_lookup["start_ts"], utc=True)
    matched = matched.merge(
        wind_lookup, left_on=["station_ext", "t_ext"], right_on=["station_id", "start_ts"], how="left",
    )
    matched["travel_dir_deg"] = (matched[dir_col] + 180.0) % 360.0
    matched["alignment"]  = circular_alignment(matched["travel_dir_deg"].to_numpy(), matched["bearing_deg"].to_numpy())
    matched["v_along_ms"] = matched[speed_col] * matched["alignment"]

    valid = (
        np.isfinite(matched["dist_km"]) & np.isfinite(matched["v_along_ms"]) &
        (matched["v_along_ms"] > MIN_VALONG_MS) & np.isfinite(matched["dt_min"])
    )
    d = matched.loc[valid].copy()
    if len(d) < 30:
        print(f"  AVISO: apenas {len(d)} eventos com v_along>{MIN_VALONG_MS} m/s — pulando {tag}.")
        return None

    d["expected_lag_min"] = (d["dist_km"] * 1000.0 / d["v_along_ms"]) / 60.0
    n_censored = int((d["expected_lag_min"] > d["dt_window_min"]).sum())

    pearson_r, _ = st.pearsonr(d["dt_min"], d["expected_lag_min"])
    spearman_r, spearman_p = st.spearmanr(d["dt_min"], d["expected_lag_min"])

    X = sm.add_constant(d["expected_lag_min"].to_numpy())
    ols = sm.OLS(d["dt_min"].to_numpy(), X).fit()
    intercept, slope = ols.params
    se_slope = ols.bse[1]

    pair_id = (d["station_ext"] + "__" + d["station_partner"]).to_numpy()
    unique_pairs = np.unique(pair_id)
    pair_to_idx = {p: np.where(pair_id == p)[0] for p in unique_pairs}
    rng = np.random.default_rng(SEED)
    dt_arr, exp_arr = d["dt_min"].to_numpy(), d["expected_lag_min"].to_numpy()
    boot_rhos = []
    for _ in range(N_BOOT):
        sp = rng.choice(unique_pairs, size=len(unique_pairs), replace=True)
        idx_b = np.concatenate([pair_to_idx[p] for p in sp])
        try:
            rho_b, _ = st.spearmanr(dt_arr[idx_b], exp_arr[idx_b])
            if np.isfinite(rho_b):
                boot_rhos.append(rho_b)
        except Exception:
            continue
    boot_rhos = np.array(boot_rhos)
    ci_low, ci_high = (np.percentile(boot_rhos, [2.5, 97.5]) if len(boot_rhos) >= 10
                       else (np.nan, np.nan))
    pd.DataFrame({"boot_spearman_rho": boot_rhos}).to_parquet(OUT_PQ, index=False)

    significant = (spearman_p < 0.05) and not (np.isnan(ci_low) or ci_low < 0 < ci_high)
    meaningful   = abs(spearman_r) > MIN_CORR_MEAN
    primary_tag  = "principal" if is_primary else "exploratório"

    if significant and meaningful:
        decision = f"SINAL DE ADVECÇÃO ({season_label}, {wind_label}) — ρ relevante"
    elif significant:
        decision = f"SEM RELEVÂNCIA PRÁTICA ({season_label}, {wind_label}) — significante mas trivial"
    else:
        decision = f"SEM SINAL ({season_label}, {wind_label})"

    # Figura
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(d["expected_lag_min"], d["dt_min"], s=5, alpha=0.08, color="#2c7bb6")
    lim = min(float(d["expected_lag_min"].quantile(0.99)), 30)
    ax.plot([0, lim], [0, lim], "k--", lw=1.5, label="advecção perfeita")
    grid = np.linspace(0, lim, 50)
    ax.plot(grid, intercept + slope * grid, color="crimson", lw=2,
            label=f"OLS (slope={slope:.3f})")
    ax.set_xlim(0, lim)
    ax.set_xlabel("Atraso esperado por advecção (dist/v_along, min)")
    ax.set_ylabel("Atraso observado dt_min (min)")
    ax.set_title(f"S7 — {season_label} | {wind_label}\nρ={spearman_r:.3f} IC95%[{ci_low:.3f},{ci_high:.3f}]")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()

    # Markdown
    OUT_MD.write_text(f"""# S7 — Advecção Sazonal: {season_label} × {wind_label}

**Data:** {date.today().isoformat()}  
**Tipo de análise:** {primary_tag}  
**Estação:** {season_label}  
**Fonte de vento:** {wind_label}

## Amostra
{n_season:,} eventos casados nesta estação; com v_along > {MIN_VALONG_MS} m/s: **{len(d):,} eventos,
{len(unique_pairs):,} pares direcionais únicos**.  
Censura por desenho: {n_censored:,}/{len(d):,} ({n_censored/len(d):.1%}).

## Resultados
| Métrica | Valor |
|---|---|
| Pearson r | {pearson_r:.4f} |
| Spearman ρ | {spearman_r:.4f} (p={spearman_p:.4f}) |
| IC95% Spearman ρ (bootstrap B={len(boot_rhos)}) | ({ci_low:.4f}, {ci_high:.4f}) |
| OLS slope | {slope:.4f} (SE {se_slope:.4f}) |
| OLS R² | {ols.rsquared:.4f} |

Limiar de relevância prática: |ρ| > {MIN_CORR_MEAN:.2f}.

## Decisão
**{decision}**

{"**NOTA:** resultado exploratório — interpretar com correção de Bonferroni (4 comparações sazonais)." if not is_primary else ""}

## Referência cruzada
- Fig.: `results/figures/s7_scatter_{tag}.png`
- Resumo geral: `results/gates/s7_seasonal_comparison.md`
- F6b original (agregado): `results/gates/f6b_timing_comparison.md`
""")

    return {
        "season_key": season_key, "season_label": season_label,
        "wind_label": wind_label, "wind_suffix": wind_suffix,
        "is_primary": is_primary,
        "coloc_applied": coloc_applied, "coloc_win": coloc_win,
        "n_season": n_season, "n_valid": len(d), "n_pairs": len(unique_pairs),
        "pearson_r": round(float(pearson_r), 4),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 4),
        "ci_low": round(float(ci_low), 4) if np.isfinite(ci_low) else None,
        "ci_high": round(float(ci_high), 4) if np.isfinite(ci_high) else None,
        "slope": round(float(slope), 4),
        "r2": round(float(ols.rsquared), 4),
        "significant": bool(significant),
        "meaningful": bool(meaningful),
        "decision": decision,
    }


def main() -> None:
    print(SEP)
    print("S7 — ESTRATIFICAÇÃO SAZONAL DO TESTE DE ADVECÇÃO F6b (Aprimoramento G)")
    print(SEP)

    for p in (ALIGNED_PQ, EVENTPAIR_PQ, COORDS_PQ, WIND_PQ):
        if not p.exists():
            print(f"\nERRO: {p} não encontrado. Certifique-se de rodar F6b primeiro.")
            sys.exit(1)

    aligned      = pd.read_parquet(ALIGNED_PQ)
    event_summary = pd.read_parquet(EVENTPAIR_PQ)
    coords        = pd.read_parquet(COORDS_PQ).dropna(subset=["lat_centroid", "lon_centroid"]).set_index("station_id")
    wind          = pd.read_parquet(WIND_PQ)

    # Pré-processamento idêntico ao de F6b_wind_timing.main()
    matched = aligned[aligned["matched"] & aligned["dt_min"].notna()].copy()
    have_coords = matched["station_ext"].isin(coords.index) & matched["station_partner"].isin(coords.index)
    matched = matched.loc[have_coords].copy()
    print(f"\n  Eventos casados com coords válidas: {len(matched):,}")

    lat1 = coords.loc[matched["station_ext"], "lat_centroid"].to_numpy()
    lon1 = coords.loc[matched["station_ext"], "lon_centroid"].to_numpy()
    lat2 = coords.loc[matched["station_partner"], "lat_centroid"].to_numpy()
    lon2 = coords.loc[matched["station_partner"], "lon_centroid"].to_numpy()
    matched["bearing_deg"] = bearing_deg(lat1, lon1, lat2, lon2)

    dist_lookup, dtwin_lookup = {}, {}
    for _, row in event_summary.iterrows():
        d_km = row["dist_ij_m"] / 1000.0
        dist_lookup[(row["station_i"], row["station_j"])] = d_km
        dist_lookup[(row["station_j"], row["station_i"])] = d_km
        dtwin_lookup[(row["station_i"], row["station_j"])] = row["dt_window_min"]
        dtwin_lookup[(row["station_j"], row["station_i"])] = row["dt_window_min"]
    matched["dist_km"]       = [dist_lookup.get((a, b), np.nan)   for a, b in zip(matched["station_ext"], matched["station_partner"])]
    matched["dt_window_min"] = [dtwin_lookup.get((a, b), np.nan)  for a, b in zip(matched["station_ext"], matched["station_partner"])]
    matched["t_ext"]         = pd.to_datetime(matched["t_ext"], utc=True)
    matched["_month"]        = matched["t_ext"].dt.month

    # ── Distribuição sazonal da amostra ───────────────────────────────────────
    season_map = {m: s for s, ms, *_ in SEASON_GROUPS_EXPL for m in ms}
    matched["_season4"] = matched["_month"].map(season_map)
    print("\n  Distribuição sazonal dos eventos casados:")
    for s, ms, slabel in SEASON_GROUPS_EXPL:
        n = int((matched["_season4"] == s).sum())
        pct = n / len(matched)
        print(f"    {slabel:30s}: {n:,} ({pct:.1%})")

    # ── Colocação temporal: distância ao snapshot CERRA mais próximo ─────────
    matched["_cerra_dist_min"] = _dist_to_cerra_snapshot(matched["t_ext"])
    print(f"\n  Distribuição de distância ao snapshot CERRA mais próximo:")
    for thr in [15, 30, 45, 60, 90]:
        pct = (matched["_cerra_dist_min"] <= thr).mean()
        print(f"    ≤{thr:3d} min: {pct:.1%} dos eventos")

    # ── Rodar análise por estação × fonte de vento (dois modos) ──────────────
    # Modo A: sem filtro (replica F6b original); Modo B: ±30 min para CERRA
    RUN_MODES = [
        ("A_nofilt", None,                      "sem filtro (replica F6b)"),
        ("B_coloc30", CERRA_COLOCATION_WIN_MIN,  f"±{CERRA_COLOCATION_WIN_MIN}min colocação CERRA"),
    ]

    all_results = []
    for mode_key, coloc_win, mode_label in RUN_MODES:
        print(f"\n{'═'*60}")
        print(f"  MODO {mode_key}: {mode_label}")
        print(f"{'═'*60}")
        for season_key, season_months, season_label, is_primary in ALL_SEASON_GROUPS:
            subset = matched[matched["_month"].isin(season_months)].copy()
            print(f"\n{SEP}")
            print(f"  Estação: {season_label}  (n={len(subset):,} eventos)")
            print(SEP)
            for speed_col, dir_col, wind_label, wind_suffix in WIND_SOURCES:
                is_cerra = wind_suffix != "10m"
                print(f"\n  Fonte: {wind_label}  [{wind_suffix}]")
                r = _run_one(subset, wind, speed_col, dir_col, wind_label, wind_suffix,
                             season_key, season_label, is_primary,
                             is_cerra=is_cerra, coloc_win=coloc_win)
                if r is not None:
                    r["mode"] = mode_key
                    all_results.append(r)
                    flag = "*** RELEVANTE ***" if (r["significant"] and r["meaningful"]) else ""
                    print(f"    ρ={r['spearman_r']:.4f}  IC95%[{r['ci_low']},{r['ci_high']}]  "
                          f"p={r['spearman_p']:.4f}  slope={r['slope']:.4f}  {flag}")

    if not all_results:
        print("\nERRO: nenhum resultado gerado.")
        sys.exit(1)

    # ── Tabela-resumo e figura de comparação ─────────────────────────────────
    comp_df = pd.DataFrame(all_results)

    OUT_COMP_MD  = cfg.DIRS["gates"]   / "s7_seasonal_comparison.md"
    OUT_COMP_FIG = cfg.DIRS["figures"] / "s7_seasonal_comparison.png"

    # Heatmap de ρ: linhas = estação, colunas = altura de vento
    season_order = ["JJA", "REST", "DJF", "MAM", "SON"]
    wind_order   = ["10m", "100m", "200m", "500m"]
    seasons_avail = [s for s in season_order if s in comp_df["season_key"].values]
    winds_avail   = [w for w in wind_order   if w in comp_df["wind_suffix"].values]

    hmap = pd.DataFrame(index=seasons_avail, columns=winds_avail, dtype=float)
    for _, row in comp_df.iterrows():
        if row["season_key"] in hmap.index and row["wind_suffix"] in hmap.columns:
            hmap.loc[row["season_key"], row["wind_suffix"]] = row["spearman_r"]

    fig, ax = plt.subplots(figsize=(len(winds_avail) * 2 + 1, len(seasons_avail) + 1))
    im = ax.imshow(hmap.values.astype(float), cmap="RdBu_r", vmin=-0.20, vmax=0.20, aspect="auto")
    ax.set_xticks(range(len(winds_avail)))
    ax.set_xticklabels(winds_avail)
    ax.set_yticks(range(len(seasons_avail)))
    ax.set_yticklabels(seasons_avail)
    for i, skey in enumerate(seasons_avail):
        for j, wkey in enumerate(winds_avail):
            val = hmap.loc[skey, wkey]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(val) > 0.12 else "black")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("S7 — ρ (atraso observado vs. esperado) por estação × altura de vento\n"
                 "(verde tracejado = limiar |ρ|>0.10; JJA/REST = principal; DJF/MAM/SON = exploratório)")
    ax.axhline(1.5, color="white", lw=2)  # separador principal vs. exploratório
    plt.tight_layout()
    plt.savefig(OUT_COMP_FIG, dpi=150)
    plt.close()

    # Markdown de comparação
    any_meaningful = any(r["significant"] and r["meaningful"] for r in all_results)
    primary_any_meaningful = any(r["significant"] and r["meaningful"] and r["is_primary"] for r in all_results)

    rows_md = []
    for r in all_results:
        ci = f"[{r['ci_low']},{r['ci_high']}]" if r["ci_low"] is not None else "n/d"
        flag = " ***" if (r["significant"] and r["meaningful"]) else ""
        tipo = "principal" if r["is_primary"] else "exploratório"
        coloc_str = f"±{r['coloc_win']}min" if r["coloc_applied"] else "—"
        rows_md.append(f"| {r.get('mode','?')} | {r['season_label']} | {r['wind_label']} | "
                        f"{tipo} | {coloc_str} | {r['n_valid']:,} | {r['spearman_r']:.4f} | "
                        f"{ci} | {r['spearman_p']:.4f} | {r['slope']:.4f} | "
                        f"{'Sim' if r['meaningful'] else 'Não'} | {r['decision']}{flag} |")

    if primary_any_meaningful:
        conclusion = ("Pelo menos um grupo pré-especificado (JJA ou REST) mostrou sinal de advecção "
                      "estatisticamente significativo E praticamente relevante (|ρ|>0.10). "
                      "Ver tabela acima para detalhe por altura.")
    elif any_meaningful:
        conclusion = ("Nenhum grupo PRINCIPAL mostrou sinal relevante. Algum grupo exploratório mostrou "
                      "sinal, mas deve ser interpretado com correção de Bonferroni (4 comparações) e "
                      "considerado gerador de hipótese, não confirmatório.")
    else:
        conclusion = ("NENHUMA estação × altura mostrou correlação praticamente relevante (|ρ|>0.10) "
                      "entre atraso observado e esperado. Resultado nulo se mantém em todas as estratificações "
                      "sazonais testadas — reforça a conclusão de regime compartilhado, não advecção física "
                      "cronometrável. A heterogeneidade sazonal (se existir) não é suficientemente forte "
                      "para inverter o nulo agregado de F6b.")

    OUT_COMP_MD.write_text(f"""# S7 — Estratificação Sazonal do Teste de Advecção F6b

**Data:** {date.today().isoformat()}  
**Hipótese principal (pré-especificada):** JJA vs. Resto do ano  
**Análise exploratória:** DJF / MAM / JJA / SON (interpretar com Bonferroni, 4 comparações)

## Nota sobre o sinal negativo em CERRA (esperado)
O sinal invertido (ρ < 0) em CERRA — especialmente em JJA — é **esperado e já documentado**
em `F6_anisotropy.py` (ROADMAP, linha ~872): F6 encontrou coincidência MAIOR quando o vento
sopra CONTRA o par em CERRA 200/500m, efeito crescente com a altura, interpretado como
confundimento sazonal/sinótico (ventos de oeste dominantes em episódios de instabilidade de
inverno capturados pelo alinhamento `alignment`). O segmento JJA expõe exatamente esse
confundimento, não um sinal de advecção real.

## Filtro de colocação temporal CERRA (decisão de design 2026-07-02)
Modo A (sem filtro): todos os eventos, tolerância de 100min herdada de `wind_joined.parquet`.  
Modo B (±{CERRA_COLOCATION_WIN_MIN}min): apenas rampas dentro de ±{CERRA_COLOCATION_WIN_MIN}min
do snapshot CERRA mais próximo — garante que o vento é contemporâneo à rampa (≤ 2× duração
mediana de 14min). Para KNMI horário (passo=60min), o filtro não tem efeito prático.

## Tabela de resultados

| Modo | Estação | Fonte de vento | Tipo | Colocação | n válido | Spearman ρ | IC95% | p | OLS slope | Relevante? | Decisão |
|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows_md)}

_*** = significante E praticamente relevante (|ρ|>{MIN_CORR_MEAN:.2f})_

## Conclusão
{conclusion}

## Notas metodológicas
- Limiar de relevância prática: |ρ| > {MIN_CORR_MEAN:.2f} (mesmo critério de F6b)
- Bootstrap por pares direcionais, B={N_BOOT}
- Análise exploratória com 4 estações: aplicar correção de Bonferroni antes de declarar
  qualquer resultado "significativo" isoladamente (threshold ajustado: p < 0.0125)
- Período de análise restrito ao conjunto de treinamento (2014-2016, split="train")
  herdado de aligned_pairs.parquet

## Referência cruzada
- F6b agregado: `results/gates/f6b_timing_comparison.md`
- Fig. heatmap: `results/figures/s7_seasonal_comparison.png`
- Figs. por grupo: `results/figures/s7_scatter_{{season}}_{{height}}.png`
""")

    print(f"\n{SEP}")
    print("S7 — RESUMO FINAL")
    print(SEP)
    print(f"  Grupos × alturas analisados: {len(all_results)}")
    print(f"  Com sinal relevante (|ρ|>{MIN_CORR_MEAN}): "
          f"{sum(r['significant'] and r['meaningful'] for r in all_results)}")
    print(f"\n  Conclusão: {conclusion[:120]}...")
    print(f"\n  Salvo: {OUT_COMP_MD.relative_to(cfg.ROOT)}")
    print(f"  Fig:   {OUT_COMP_FIG.relative_to(cfg.ROOT)}")

    log_result(
        script="S7_f6b_seasonal.py",
        gate="",
        phase="S7_AprimoramentoG",
        params={
            "seasons_primary": ["JJA", "REST"],
            "seasons_exploratory": ["DJF", "MAM", "SON"],
            "wind_sources": [ws[3] for ws in WIND_SOURCES],
            "min_valong_ms": MIN_VALONG_MS,
            "min_corr_meaningful": MIN_CORR_MEAN,
            "n_bootstrap": N_BOOT,
        },
        results={
            "n_groups_analyzed": len(all_results),
            "n_significant_and_meaningful": int(sum(r["significant"] and r["meaningful"] for r in all_results)),
            "primary_any_meaningful": primary_any_meaningful,
            "summary_by_season": {
                r["season_key"] + "_" + r["wind_suffix"]: r["spearman_r"]
                for r in all_results
            },
        },
        decision=conclusion[:300],
        action="Seasonal stratification of advection timing test (Aprimoramento G, resolucao_gaps.md).",
        interpretation=conclusion,
        paper_ref="Section 8 (F6 spatial structure) — seasonal robustness check of advection null result.",
    )


if __name__ == "__main__":
    main()
