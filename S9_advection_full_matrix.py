"""
S9_advection_full_matrix.py — Matriz definitiva: advecção × TODAS as alturas × todas as estações
=================================================================================================
Motivação: B7c (CERRA pressure levels) agora completo (72/72 meses). `wind_joined.parquet`
contém 10 fontes de vento em diferentes alturas:

  KNMI De Bilt 10m (superfície, horário)
  CERRA 100m, 200m, 500m     ← altitude geométrica (já em S7)
  CERRA 950hPa (~540m), 900hPa (~990m), 850hPa (~1460m), 800hPa (~1950m)
  CERRA 700hPa (~3010m), 600hPa (~4200m), 500hPa (~5570m), 400hPa (~7190m)

Cobre a coluna atmosférica de 10m a ~7.2km — nuvens baixas (stratus, stratocumulus,
~200-600m), médias (altocumulus, ~2-4km), altas (cirrus, ~6-7km) e extremamente altas.

Pergunta central (hipótese da advecção):
  "Existe ALGUMA altura de vento, em ALGUMA estação do ano, em que o atraso OBSERVADO
   entre eventos extremos em pares de usinas correlaciona com o atraso ESPERADO por
   transporte advectivo puro (distância / velocidade de vento na direção do par)?"

  Se Spearman ρ > 0.10 (limiar de relevância prática de F6b) em qualquer célula da
  matriz → evidência de transporte físico mensurável nessa altitude/estação.
  Se nulo em todas as 60 células (10 alturas × 5 épocas) → confirmação definitiva de
  que o mecanismo não é advecção física cronometrável em NENHUMA parte da coluna
  atmosférica, reforçando a interpretação de REGIME COMPARTILHADO (sincronismo
  meteorológico de grande escala, não transporte de nuvem individual).

Reutiliza _run_one / _dist_to_cerra_snapshot de S7_f6b_seasonal.py (sem modificar
o script aprovado) e respeita os mesmos critérios de filtragem:
  - v_along > 1.0 m/s (MIN_VALONG_MS de cfg)
  - n_valid ≥ 30 por célula
  - coloc ±30min para todas as fontes CERRA (altura ou pressão)

NÃO reabre/modifica S7_f6b_seasonal.py, F6b_wind_timing.py, F6_anisotropy.py.

Saídas:
  results/gates/s9_full_matrix.parquet       — tabela de resultados (60 × 2 modos)
  results/gates/s9_full_matrix_decision.md   — decisão e tabela resumo
  results/figures/s9_full_matrix_heatmap.png — heatmap ρ: estação × altura (ambos modos)
  results/figures/s9_full_matrix_profile.png — perfil vertical de ρ por estação

Executar:
    python S9_advection_full_matrix.py
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
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result
from F6_anisotropy import bearing_deg, circular_alignment
from S7_f6b_seasonal import (
    _run_one,
    _dist_to_cerra_snapshot,
    ALL_SEASON_GROUPS,
    N_BOOT,
    MIN_VALONG_MS,
    MIN_CORR_MEAN,
    CERRA_COLOCATION_WIN_MIN,
    SEED,
)

ALIGNED_PQ   = cfg.DIRS["processed"] / "aligned_pairs.parquet"
EVENTPAIR_PQ = cfg.DIRS["gates"]     / "event_pairing_summary.parquet"
COORDS_PQ    = cfg.DIRS["interim"]   / "coords.parquet"
WIND_PQ      = cfg.DIRS["interim"]   / "wind_joined.parquet"

OUT_PQ   = cfg.DIRS["gates"]   / "s9_full_matrix.parquet"
OUT_DEC  = cfg.DIRS["gates"]   / "s9_full_matrix_decision.md"
OUT_FIG1 = cfg.DIRS["figures"] / "s9_full_matrix_heatmap.png"
OUT_FIG2 = cfg.DIRS["figures"] / "s9_full_matrix_profile.png"

SEP = "─" * 60

# ── Todas as fontes de vento (superfície + alturas CERRA + níveis de pressão) ──────────
# Ordem crescente de altitude para o perfil vertical
WIND_SOURCES_FULL = [
    # col_speed,                       col_dir,                          label,               suffix,    altitude_m, is_cerra
    ("wind_speed_ms",                  "wind_dir_deg",                   "KNMI 10m (sup)",     "10m",     10,        False),
    ("cerra_wind_speed_ms_950hPa",     "cerra_wind_dir_deg_950hPa",     "CERRA 950hPa~540m",  "950hPa",  540,       True),
    ("cerra_wind_speed_ms_100m",       "cerra_wind_dir_deg_100m",       "CERRA 100m",         "100m",    100,       True),
    ("cerra_wind_speed_ms_900hPa",     "cerra_wind_dir_deg_900hPa",     "CERRA 900hPa~990m",  "900hPa",  990,       True),
    ("cerra_wind_speed_ms_850hPa",     "cerra_wind_dir_deg_850hPa",     "CERRA 850hPa~1460m", "850hPa",  1460,      True),
    ("cerra_wind_speed_ms_200m",       "cerra_wind_dir_deg_200m",       "CERRA 200m",         "200m",    200,       True),
    ("cerra_wind_speed_ms_800hPa",     "cerra_wind_dir_deg_800hPa",     "CERRA 800hPa~1950m", "800hPa",  1950,      True),
    ("cerra_wind_speed_ms_700hPa",     "cerra_wind_dir_deg_700hPa",     "CERRA 700hPa~3010m", "700hPa",  3010,      True),
    ("cerra_wind_speed_ms_500m",       "cerra_wind_dir_deg_500m",       "CERRA 500m",         "500m",    500,       True),
    ("cerra_wind_speed_ms_600hPa",     "cerra_wind_dir_deg_600hPa",     "CERRA 600hPa~4200m", "600hPa",  4200,      True),
    ("cerra_wind_speed_ms_500hPa",     "cerra_wind_dir_deg_500hPa",     "CERRA 500hPa~5570m", "500hPa",  5570,      True),
    ("cerra_wind_speed_ms_400hPa",     "cerra_wind_dir_deg_400hPa",     "CERRA 400hPa~7190m", "400hPa",  7190,      True),
]
# Ordenar por altitude para o perfil vertical
WIND_SOURCES_FULL_SORTED = sorted(WIND_SOURCES_FULL, key=lambda x: x[4])

SEASON_ORDER = ["JJA", "REST", "DJF", "MAM", "SON"]


def main() -> None:
    print(SEP)
    print("S9 — MATRIZ DEFINITIVA: ADVECÇÃO × TODAS AS ALTURAS × TODAS AS ESTAÇÕES")
    print(f"     {len(WIND_SOURCES_FULL)} fontes de vento × {len(ALL_SEASON_GROUPS)} épocas × 2 modos")
    print(SEP)

    for p in (ALIGNED_PQ, EVENTPAIR_PQ, COORDS_PQ, WIND_PQ):
        if not p.exists():
            print(f"\nERRO: {p} não encontrado.")
            sys.exit(1)

    # Verificar quais colunas estão disponíveis
    wind = pd.read_parquet(WIND_PQ)
    available = set(wind.columns)
    missing = [(s, d, lbl, sfx) for s, d, lbl, sfx, *_ in WIND_SOURCES_FULL
               if s not in available or d not in available]
    if missing:
        print(f"\n  AVISO: {len(missing)} fontes de vento não disponíveis no parquet:")
        for s, d, lbl, sfx in missing:
            print(f"    {lbl}: {s}, {d}")

    aligned       = pd.read_parquet(ALIGNED_PQ)
    event_summary = pd.read_parquet(EVENTPAIR_PQ)
    coords        = pd.read_parquet(COORDS_PQ).dropna(subset=["lat_centroid","lon_centroid"]).set_index("station_id")

    matched = aligned[aligned["matched"] & aligned["dt_min"].notna()].copy()
    have_coords = (matched["station_ext"].isin(coords.index) &
                   matched["station_partner"].isin(coords.index))
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

    matched["dist_km"]       = [dist_lookup.get((a, b), np.nan)  for a, b in zip(matched["station_ext"], matched["station_partner"])]
    matched["dt_window_min"] = [dtwin_lookup.get((a, b), np.nan) for a, b in zip(matched["station_ext"], matched["station_partner"])]
    matched["t_ext"]         = pd.to_datetime(matched["t_ext"], utc=True)
    matched["_month"]        = matched["t_ext"].dt.month
    matched["_cerra_dist_min"] = _dist_to_cerra_snapshot(matched["t_ext"])

    RUN_MODES = [
        ("A_nofilt",  None,                     "sem filtro"),
        ("B_coloc30", CERRA_COLOCATION_WIN_MIN,  f"±{CERRA_COLOCATION_WIN_MIN}min colocação CERRA"),
    ]

    all_results = []
    total_cells = len(ALL_SEASON_GROUPS) * len(WIND_SOURCES_FULL) * len(RUN_MODES)
    done = 0

    for mode_key, coloc_win, mode_label in RUN_MODES:
        print(f"\n{'═'*60}")
        print(f"  MODO {mode_key}: {mode_label}")
        print(f"{'═'*60}")
        for season_key, season_months, season_label, is_primary in ALL_SEASON_GROUPS:
            subset = matched[matched["_month"].isin(season_months)].copy()
            print(f"\n{SEP}")
            print(f"  {season_label}  (n={len(subset):,})")
            print(SEP)
            for speed_col, dir_col, wind_label, wind_suffix, altitude_m, is_cerra in WIND_SOURCES_FULL:
                done += 1
                print(f"  [{done}/{total_cells}] {wind_label}  (alt={altitude_m}m)")
                r = _run_one(
                    subset, wind, speed_col, dir_col, wind_label, wind_suffix,
                    season_key, season_label, is_primary,
                    is_cerra=is_cerra, coloc_win=coloc_win,
                )
                if r is not None:
                    r["mode"] = mode_key
                    r["altitude_m"] = altitude_m
                    all_results.append(r)
                    flag = "*** RELEVANTE ***" if (r["significant"] and r["meaningful"]) else ""
                    print(f"      ρ={r['spearman_r']:+.4f}  IC95%[{r['ci_low']},{r['ci_high']}]  "
                          f"p={r['spearman_p']:.4f}  {flag}")

    if not all_results:
        print("\nERRO: nenhum resultado gerado.")
        sys.exit(1)

    comp_df = pd.DataFrame(all_results)
    comp_df.to_parquet(OUT_PQ, index=False)
    print(f"\n  Salvo: {OUT_PQ.relative_to(cfg.ROOT)}")

    # ── Decisão global ────────────────────────────────────────────────────────
    any_meaningful      = any(r["significant"] and r["meaningful"] for r in all_results)
    primary_meaningful  = any(r["significant"] and r["meaningful"] and r["is_primary"]
                              for r in all_results)
    n_cells = len(all_results)
    n_mode_a = sum(1 for r in all_results if r["mode"] == "A_nofilt")
    n_mode_b = sum(1 for r in all_results if r["mode"] == "B_coloc30")
    max_rho  = max(abs(r["spearman_r"]) for r in all_results)
    max_rho_cell = max(all_results, key=lambda r: abs(r["spearman_r"]))

    if primary_meaningful:
        decision = ("SINAL DE ADVECÇÃO DETECTADO — pelo menos um grupo pré-especificado "
                    "(JJA ou REST) em alguma altitude mostrou |ρ| > 0.10 estatisticamente significante.")
        interpretation = "Evidência de transporte físico mensurável — reportar com detalhe de altitude/estação."
    elif any_meaningful:
        decision = ("SINAL EXPLORATÓRIO APENAS — nenhum grupo PRINCIPAL mostrou |ρ| > 0.10; "
                    "algum grupo exploratório mostrou, requer correção de múltiplas comparações.")
        interpretation = ("Sugestivo mas não confirmatório — interpretar como gerador de hipótese "
                          "para investigação futura em estudo dedicado.")
    else:
        decision = (f"NULO DEFINITIVO — nenhuma das {n_cells} células "
                    f"({len(WIND_SOURCES_FULL)} alturas × {len(ALL_SEASON_GROUPS)} épocas × 2 modos) "
                    f"mostrou |ρ| > {MIN_CORR_MEAN:.2f}. ρ máximo observado: "
                    f"|ρ|={max_rho:.4f} ({max_rho_cell['wind_label']}, {max_rho_cell['season_label']}, "
                    f"modo {max_rho_cell['mode']}).")
        interpretation = (
            f"O resultado nulo de F6b (KNMI 10m, agregado) é robusto em TODA a coluna "
            f"atmosférica de 10m a ~7.2km e em todas as cinco épocas sazonais testadas. "
            f"Nuvens individuais — quer estejam na camada limite (~10-100m), em nuvens "
            f"baixas (~500-1000m, 950-900hPa), médias (~1.5-3km, 850-700hPa) ou altas "
            f"(~4-7km, 600-400hPa) — não geram um sinal de advecção CRONOMETRÁVEL entre "
            f"usinas ao nível de rede. Isso é evidência positiva (não ausência de dado) "
            f"de que a dependência espacial observada (Gate G1, χ>0 em pares <5km) é "
            f"causada por REGIME COMPARTILHADO (sistemas meteorológicos de meso/macro-escala "
            f"que afetam simultaneamente múltiplas usinas), não pelo transporte físico de "
            f"nuvens individuais de uma usina para a outra. Esse mecanismo é, paradoxalmente, "
            f"MAIS adverso para a gestão de reserva (porque não há defasagem útil entre "
            f"usinas — o impacto chega simultâneo), reforçando o resultado central do paper "
            f"(razão RQ3 = 2.39×)."
        )

    print(f"\n  DECISÃO: {decision}")

    # ── Figura 1: Heatmap ρ (estação × altura), dois modos lado a lado ────────
    # Ordenar alturas para o heatmap (crescente)
    wind_order_sorted = [ws[3] for ws in WIND_SOURCES_FULL_SORTED]
    wind_labels_sorted = [ws[2] for ws in WIND_SOURCES_FULL_SORTED]

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(wind_order_sorted) * 1.4 + 2), 5))
    for ax_i, (mode_key, _, mode_label) in enumerate(RUN_MODES):
        ax = axes[ax_i]
        sub = comp_df[comp_df["mode"] == mode_key]
        seasons_avail = [s for s in SEASON_ORDER if s in sub["season_key"].values]
        winds_avail   = [w for w in wind_order_sorted if w in sub["wind_suffix"].values]
        wind_labels_avail = [wind_labels_sorted[wind_order_sorted.index(w)]
                             for w in winds_avail]

        hmap = np.full((len(seasons_avail), len(winds_avail)), np.nan)
        sig_mask = np.zeros_like(hmap, dtype=bool)
        for _, row in sub.iterrows():
            r_i = seasons_avail.index(row["season_key"]) if row["season_key"] in seasons_avail else -1
            r_j = winds_avail.index(row["wind_suffix"])  if row["wind_suffix"]  in winds_avail  else -1
            if r_i >= 0 and r_j >= 0:
                hmap[r_i, r_j]    = row["spearman_r"]
                sig_mask[r_i, r_j] = row["significant"] and row["meaningful"]

        im = ax.imshow(hmap, cmap="RdBu_r", vmin=-0.20, vmax=0.20, aspect="auto")
        ax.set_xticks(range(len(winds_avail)))
        ax.set_xticklabels(wind_labels_avail, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(seasons_avail)))
        ax.set_yticklabels(seasons_avail)
        for i in range(len(seasons_avail)):
            for j in range(len(winds_avail)):
                if np.isfinite(hmap[i, j]):
                    txt = f"{hmap[i,j]:+.3f}"
                    border = "**" if sig_mask[i, j] else ""
                    ax.text(j, i, border + txt + border, ha="center", va="center",
                            fontsize=7, color="white" if abs(hmap[i, j]) > 0.12 else "black",
                            fontweight="bold" if sig_mask[i, j] else "normal")
        ax.axhline(1.5, color="white", lw=2)  # sep principal / exploratório
        ax.set_title(f"Modo {mode_key}: {mode_label}\nρ (atraso obs. vs. esperado)")
        plt.colorbar(im, ax=ax, label="Spearman ρ")

    fig.suptitle(
        f"S9 — Matriz definitiva: advecção × {len(WIND_SOURCES_FULL)} alturas × épocas\n"
        f"Limiar relevância prática: |ρ|>{MIN_CORR_MEAN:.2f}  |  JJA/REST=principal, DJF/MAM/SON=exploratório",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(OUT_FIG1, dpi=150)
    plt.close()
    print(f"  Figura 1 (heatmap): {OUT_FIG1.relative_to(cfg.ROOT)}")

    # ── Figura 2: Perfil vertical de ρ por estação (modo A) ──────────────────
    mode_a = comp_df[comp_df["mode"] == "A_nofilt"].copy()
    seasons_plot = [s for _, _, s, _ in ALL_SEASON_GROUPS]
    season_keys  = [k for k, *_ in ALL_SEASON_GROUPS]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    for (sk, _, sl, _), color in zip(ALL_SEASON_GROUPS, colors):
        sub_s = mode_a[mode_a["season_key"] == sk].copy()
        if "altitude_m" not in sub_s.columns:
            alt_map = {ws[3]: ws[4] for ws in WIND_SOURCES_FULL}
            sub_s["altitude_m"] = sub_s["wind_suffix"].map(alt_map)
        sub_s = sub_s.sort_values("altitude_m")
        if len(sub_s) == 0:
            continue
        ax2.plot(sub_s["spearman_r"], sub_s["altitude_m"] / 1000.0,
                 "o-", color=color, label=sl, lw=1.5, markersize=5)
        # IC sombreado onde disponível
        has_ci = sub_s["ci_low"].notna() & sub_s["ci_high"].notna()
        if has_ci.any():
            sub_ci = sub_s[has_ci]
            ax2.fill_betweenx(sub_ci["altitude_m"] / 1000.0,
                               sub_ci["ci_low"], sub_ci["ci_high"],
                               color=color, alpha=0.10)

    ax2.axvline(0, color="black", lw=0.8, linestyle="--")
    ax2.axvline(MIN_CORR_MEAN, color="grey", lw=1, linestyle=":", label=f"|ρ|={MIN_CORR_MEAN:.2f} (relevância)")
    ax2.axvline(-MIN_CORR_MEAN, color="grey", lw=1, linestyle=":")
    ax2.set_xlabel("Spearman ρ (atraso observado vs. esperado por advecção)")
    ax2.set_ylabel("Altitude aproximada (km)")
    ax2.set_title("S9 — Perfil vertical de ρ por estação (modo sem filtro)\nLinha tracejada = limiar de relevância prática")
    ax2.legend(fontsize=8, loc="center right")
    cloud_layers = [(0.0, 0.6, "nuvens baixas\n(stratus, ~0-600m)", 0.85),
                    (0.6, 3.0, "nuvens médias\n(alto-, ~0.6-3km)", 0.65),
                    (3.0, 7.5, "nuvens altas\n(cirrus, ~3-7km)", 0.45)]
    for y0, y1, lbl, alpha in cloud_layers:
        ax2.axhspan(y0, y1, color="lightyellow", alpha=0.3)
        ax2.text(ax2.get_xlim()[0] + 0.001, (y0 + y1) / 2, lbl,
                 fontsize=6, color="goldenrod", va="center")
    plt.tight_layout()
    plt.savefig(OUT_FIG2, dpi=150)
    plt.close()
    print(f"  Figura 2 (perfil vertical): {OUT_FIG2.relative_to(cfg.ROOT)}")

    # ── Tabela markdown ────────────────────────────────────────────────────────
    # Agregar: para cada (season, wind_suffix), mostrar ρ modo A + ρ modo B
    rows_md = []
    pivot = comp_df.groupby(["season_key", "wind_suffix", "mode"])[
        ["spearman_r", "ci_low", "ci_high", "spearman_p", "meaningful", "significant", "altitude_m"]
    ].first().unstack("mode")
    for (sk, ws), row in pivot.iterrows():
        rho_a = row.get(("spearman_r", "A_nofilt"), np.nan)
        rho_b = row.get(("spearman_r", "B_coloc30"), np.nan)
        ci_a  = (f"[{row.get(('ci_low','A_nofilt'),np.nan):.3f},"
                 f"{row.get(('ci_high','A_nofilt'),np.nan):.3f}]")
        ci_b  = (f"[{row.get(('ci_low','B_coloc30'),np.nan):.3f},"
                 f"{row.get(('ci_high','B_coloc30'),np.nan):.3f}]")
        rel   = "SIM ***" if row.get(("meaningful", "A_nofilt"), False) else "Não"
        alt   = int(row.get(("altitude_m", "A_nofilt"), 0) or 0)
        rows_md.append(f"| {sk} | {ws} | ~{alt}m | {rho_a:+.4f} | {ci_a} | "
                       f"{rho_b:+.4f} | {ci_b} | {rel} |")

    # Resumo por altitude (modo A, todas as estações — range de ρ)
    rows_alt = []
    for speed_col, dir_col, lbl, sfx, alt_m, _ in WIND_SOURCES_FULL_SORTED:
        sub_alt = comp_df[(comp_df["wind_suffix"] == sfx) & (comp_df["mode"] == "A_nofilt")]
        if len(sub_alt) == 0:
            continue
        rho_min = sub_alt["spearman_r"].min()
        rho_max = sub_alt["spearman_r"].max()
        rho_abs_max = sub_alt["spearman_r"].abs().max()
        any_rel = (sub_alt["meaningful"] & sub_alt["significant"]).any()
        rows_alt.append(f"| {lbl} | ~{alt_m}m | {rho_min:+.4f} | {rho_max:+.4f} | "
                        f"{rho_abs_max:.4f} | {'SIM ***' if any_rel else 'Não'} |")

    OUT_DEC.write_text(f"""# S9 — Matriz Definitiva: Advecção × Todas as Alturas × Todas as Estações

**Data:** {date.today().isoformat()}  
**Fontes de vento:** {len(WIND_SOURCES_FULL)} (KNMI 10m + CERRA 100/200/500m + CERRA 950/900/850/800/700/600/500/400hPa)  
**Épocas:** {len(ALL_SEASON_GROUPS)} (JJA, REST, DJF, MAM, SON)  
**Modos:** 2 (sem filtro de colocação temporal; ±30min do snapshot CERRA)  
**Células avaliadas:** {n_cells} ({n_mode_a} modo A + {n_mode_b} modo B)

## Pergunta central
Existe ALGUMA altitude de vento, em ALGUMA estação do ano, em que o atraso OBSERVADO
entre eventos extremos em pares de usinas correlaciona com o atraso ESPERADO por
transporte advectivo puro (distância / velocidade de vento na direção do par), acima
do limiar de relevância prática |ρ| > {MIN_CORR_MEAN:.2f}?

## Decisão global
**{decision}**

## Interpretação física
{interpretation}

## Resumo por altitude (modo A — sem filtro, todas as estações)
| Fonte | Altitude aprox. | ρ mín | ρ máx | |ρ| máx | Relevante? |
|---|---|---|---|---|---|
{chr(10).join(rows_alt)}

## Tabela completa (todas as células)
| Época | Altura | Alt. | ρ modo A | IC95% A | ρ modo B | IC95% B | Relevante? |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows_md)}

## Figuras
- Heatmap (estação × altura, ambos modos): `results/figures/s9_full_matrix_heatmap.png`
- Perfil vertical de ρ por estação (modo A): `results/figures/s9_full_matrix_profile.png`

## Referência cruzada
- F6b (agregado, KNMI 10m): `results/gates/f6b_timing_model_10m.md`
- S4 (resolução temporal, KNMI 10min nativo): `results/gates/s4_f6b_resolution_comparison.md`
- S7 (sazonal, 4 alturas): `results/gates/s7_seasonal_comparison.md`
- S8 (χ(u) vs. gaussiana): `results/gates/s8_chi_vs_gaussian_decision.md`
- Gate G1 (dependência de cauda): `results/gates/gate1_decision.md`
""")
    print(f"  Salvo: {OUT_DEC.relative_to(cfg.ROOT)}")

    log_result(
        script="S9_advection_full_matrix.py",
        gate="",
        phase="S9_AdveccaoDefinitiva",
        params={
            "wind_sources": [ws[2] for ws in WIND_SOURCES_FULL],
            "n_wind_sources": len(WIND_SOURCES_FULL),
            "altitude_range_m": [10, 7190],
            "seasons": [k for k, *_ in ALL_SEASON_GROUPS],
            "n_seasons": len(ALL_SEASON_GROUPS),
            "n_modes": 2,
            "min_valong_ms": MIN_VALONG_MS,
            "min_corr_meaningful": MIN_CORR_MEAN,
            "cerra_coloc_win_min": CERRA_COLOCATION_WIN_MIN,
            "n_bootstrap": N_BOOT,
        },
        results={
            "n_cells_evaluated": n_cells,
            "any_meaningful": bool(any_meaningful),
            "primary_meaningful": bool(primary_meaningful),
            "max_abs_rho": round(float(max_rho), 4),
            "max_abs_rho_cell": {
                "wind": max_rho_cell["wind_label"],
                "season": max_rho_cell["season_label"],
                "mode": max_rho_cell["mode"],
                "rho": round(float(max_rho_cell["spearman_r"]), 4),
            },
        },
        decision=decision,
        action=(
            "Comprehensive advection timing test across full atmospheric column (10m to ~7.2km) "
            "and all seasons, using all available wind sources from wind_joined.parquet. "
            "Closes the B7c-dependent analysis blocked since download started."
        ),
        interpretation=interpretation,
        paper_ref=(
            "Section 8 (spatial structure / advection null result) — definitive multi-height, "
            "multi-season robustness check for F6b null conclusion. Supports shared-regime "
            "interpretation (Section 9, RQ3 mechanism)."
        ),
    )

    print(f"\n{SEP}")
    print(f"S9 — {decision}")
    print(SEP)


if __name__ == "__main__":
    main()
