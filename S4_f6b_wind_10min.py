"""
S4_f6b_wind_10min.py — Re-teste de advecção F6b com vento KNMI de RESOLUÇÃO NATIVA de 10min
================================================================================================
Motivação (Risco C, ROADMAP Seção 0.2 / resolucao_gaps.md Seção 1): `F6b_wind_timing.py`
usa vento de superfície HORÁRIO (KNMI De Bilt, `wind_joined.parquet`, produzido por
`B7_wind_join.py`), mais grosseiro que a escala do fenômeno (rampa mediana=14min, atraso
mediano entre estações também ~14min). Isso levanta a dúvida: será que o resultado nulo de
F6b (nenhuma correlação praticamente relevante entre atraso observado e esperado por
advecção) é um artefato de baixa resolução temporal do vento, e não uma ausência real de
sinal físico?

Este script resolve essa dúvida repetindo EXATAMENTE a mesma análise central de F6b (mesmos
filtros, mesmo `MIN_VALONG_MS`, mesmo bootstrap por par direcional, mesma fonte física —
De Bilt, mas agora casada ao vento REAL de 10min mais próximo do instante `t_ext` de cada
evento condicionante (`B7d_wind_knmi_10min.py`, arredondamento ao múltiplo de 10min mais
próximo — desalinhamento residual máximo de ±5min, vs. até ±30min no casamento horário).

NÃO reabre/modifica F6b_wind_timing.py (já aprovado) — script auxiliar independente que
reusa `bearing_deg`/`circular_alignment` de F6_anisotropy.py.

Saídas:
  results/gates/s4_f6b_timing_model_10min_hires.md
  results/gates/s4_f6b_timing_bootstrap_10min_hires.parquet
  results/figures/s4_f6b_timing_scatter_10min_hires.png
  results/gates/s4_f6b_decision_10min_hires.md
  results/gates/s4_f6b_resolution_comparison.md   — comparação horário vs. 10min

Executar:
    python S4_f6b_wind_10min.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st
from statsmodels.stats.proportion import proportions_ztest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result
from F6_anisotropy import bearing_deg, circular_alignment

ALIGNED_PQ    = cfg.DIRS["processed"] / "aligned_pairs.parquet"
EVENTPAIR_PQ  = cfg.DIRS["gates"]     / "event_pairing_summary.parquet"
COORDS_PQ     = cfg.DIRS["interim"]   / "coords.parquet"
WIND_10MIN_PQ = cfg.DIRS["interim"]   / "wind_knmi_10min.parquet"

OUT_MODEL = cfg.DIRS["gates"]   / "s4_f6b_timing_model_10min_hires.md"
OUT_BOOT  = cfg.DIRS["gates"]   / "s4_f6b_timing_bootstrap_10min_hires.parquet"
OUT_FIG   = cfg.DIRS["figures"] / "s4_f6b_timing_scatter_10min_hires.png"
OUT_DEC   = cfg.DIRS["gates"]   / "s4_f6b_decision_10min_hires.md"
OUT_COMP  = cfg.DIRS["gates"]   / "s4_f6b_resolution_comparison.md"

N_BOOT        = cfg.F6["n_bootstrap_pairs"]     # 150
MIN_VALONG_MS = cfg.F6["min_valong_ms"]         # 1.0 m/s
MIN_CORR_MEAN = cfg.F6["min_corr_meaningful"]   # 0.10
SEED          = cfg.SEED

SEP = "─" * 60


def main() -> None:
    print(SEP)
    print("S4 — RE-TESTE DE ADVECÇÃO F6b COM VENTO KNMI DE 10MIN (Risco C)")
    print(SEP)

    for p in (ALIGNED_PQ, EVENTPAIR_PQ, COORDS_PQ, WIND_10MIN_PQ):
        if not p.exists():
            print(f"\nERRO: {p} não encontrado.")
            if p == WIND_10MIN_PQ:
                print("  Execute B7d_wind_knmi_10min.py primeiro.")
            sys.exit(1)

    aligned = pd.read_parquet(ALIGNED_PQ)
    event_summary = pd.read_parquet(EVENTPAIR_PQ)
    coords = pd.read_parquet(COORDS_PQ).dropna(subset=["lat_centroid", "lon_centroid"]).set_index("station_id")
    wind10 = pd.read_parquet(WIND_10MIN_PQ)

    print("\n[1/6] Restringindo ao subconjunto casado (dt_min observado) — idêntico a F6b...")
    matched = aligned[aligned["matched"] & aligned["dt_min"].notna()].copy()
    have_coords = matched["station_ext"].isin(coords.index) & matched["station_partner"].isin(coords.index)
    matched = matched.loc[have_coords].copy()
    print(f"  Eventos casados com coords válidas: {len(matched):,}")

    print("\n[2/6] Calculando bearing geodésico e distância por par (idêntico a F6b)...")
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
    matched["dist_km"] = [dist_lookup.get((a, b), np.nan) for a, b in zip(matched["station_ext"], matched["station_partner"])]
    matched["dt_window_min"] = [dtwin_lookup.get((a, b), np.nan) for a, b in zip(matched["station_ext"], matched["station_partner"])]
    matched["t_ext"] = pd.to_datetime(matched["t_ext"], utc=True)

    print("\n[3/6] Casando vento KNMI de 10min (arredondado ao múltiplo de 10min mais próximo "
          "de t_ext, desalinhamento residual máx. ±5min)...")
    matched["t_ext_round10"] = matched["t_ext"].dt.round("10min")
    wind_lookup = wind10.rename(columns={"timestamp_utc": "t_ext_round10"})[
        ["t_ext_round10", "wind_speed_ms", "wind_dir_deg"]
    ].copy()
    n_before = len(matched)
    matched = matched.merge(wind_lookup, on="t_ext_round10", how="left")
    n_wind_ok = int(matched["wind_speed_ms"].notna().sum())
    print(f"  Vento 10min casado: {n_wind_ok:,}/{n_before:,} ({n_wind_ok/n_before:.1%}) eventos casados "
          f"(mesma estação De Bilt para todas as estações-alvo, como em F6b/B7).")

    matched["travel_dir_deg"] = (matched["wind_dir_deg"] + 180.0) % 360.0
    matched["alignment"] = circular_alignment(matched["travel_dir_deg"].to_numpy(), matched["bearing_deg"].to_numpy())
    matched["v_along_ms"] = matched["wind_speed_ms"] * matched["alignment"]
    print(f"  Distribuição de v_along (m/s): mediana={matched['v_along_ms'].median():.2f}, "
          f"% apontando para o parceiro (v_along>0) = {(matched['v_along_ms'] > 0).mean():.1%}")

    print(f"\n[4/6] Calculando atraso esperado (dist/v_along) e comparando com dt_min observado "
          f"(restrito a v_along > {MIN_VALONG_MS} m/s) — idêntico a F6b...")
    valid = (
        np.isfinite(matched["dist_km"]) & np.isfinite(matched["v_along_ms"]) &
        (matched["v_along_ms"] > MIN_VALONG_MS) & np.isfinite(matched["dt_min"])
    )
    d = matched.loc[valid].copy()
    d["expected_lag_min"] = (d["dist_km"] * 1000.0 / d["v_along_ms"]) / 60.0

    n_censored = int((d["expected_lag_min"] > d["dt_window_min"]).sum())
    print(f"  Amostra válida (v_along > {MIN_VALONG_MS} m/s): {len(d):,}/{len(matched):,} eventos casados")
    print(f"  AVISO (censura por desenho): {n_censored:,}/{len(d):,} ({n_censored/len(d):.1%}) têm atraso "
          f"esperado maior que a janela de busca do C1b.")

    pearson_r, pearson_p = st.pearsonr(d["dt_min"], d["expected_lag_min"])
    spearman_r, spearman_p = st.spearmanr(d["dt_min"], d["expected_lag_min"])
    print(f"  Pearson r = {pearson_r:.4f} (p={pearson_p:.4f})")
    print(f"  Spearman ρ = {spearman_r:.4f} (p={spearman_p:.4f})")

    X = sm.add_constant(d["expected_lag_min"].to_numpy())
    y = d["dt_min"].to_numpy()
    ols = sm.OLS(y, X).fit()
    intercept, slope = ols.params
    se_slope = ols.bse[1]
    print(f"  OLS: dt_min = {intercept:.3f} + {slope:.4f}·atraso_esperado (SE={se_slope:.4f}; R²={ols.rsquared:.4f})")

    print(f"\n[5/6] Cluster bootstrap (por par direcional, B={N_BOOT}) do ρ de Spearman...")
    pair_id = (d["station_ext"] + "__" + d["station_partner"]).to_numpy()
    unique_pairs = np.unique(pair_id)
    pair_to_idx = {p: np.where(pair_id == p)[0] for p in unique_pairs}
    rng = np.random.default_rng(SEED)
    dt_arr, exp_arr = d["dt_min"].to_numpy(), d["expected_lag_min"].to_numpy()
    boot_rhos = []
    for _ in range(N_BOOT):
        sampled_pairs = rng.choice(unique_pairs, size=len(unique_pairs), replace=True)
        idx_b = np.concatenate([pair_to_idx[p] for p in sampled_pairs])
        try:
            rho_b, _ = st.spearmanr(dt_arr[idx_b], exp_arr[idx_b])
            if np.isfinite(rho_b):
                boot_rhos.append(rho_b)
        except Exception:
            continue
    boot_rhos = np.array(boot_rhos)
    ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5])
    print(f"  Bootstrap: {len(boot_rhos)}/{N_BOOT} válidos. IC95%: ({ci_low:.4f}, {ci_high:.4f})")
    pd.DataFrame({"boot_spearman_rho": boot_rhos}).to_parquet(OUT_BOOT, index=False)

    print("\n  Teste complementar (consistência de sinal, amostra completa de casados)...")
    toward = matched["v_along_ms"] > 0
    away = matched["v_along_ms"] <= 0
    frac_pos_toward = float((matched.loc[toward, "dt_min"] > 0).mean())
    frac_pos_away = float((matched.loc[away, "dt_min"] > 0).mean())
    n_toward, n_away = int(toward.sum()), int(away.sum())
    count = np.array([int((matched.loc[toward, "dt_min"] > 0).sum()), int((matched.loc[away, "dt_min"] > 0).sum())])
    nobs = np.array([n_toward, n_away])
    z_prop, p_prop = proportions_ztest(count, nobs)
    print(f"  P(dt_min>0 | vento aponta p/ parceiro, n={n_toward:,}) = {frac_pos_toward:.4f}")
    print(f"  P(dt_min>0 | vento NÃO aponta p/ parceiro, n={n_away:,}) = {frac_pos_away:.4f}")
    print(f"  Diferença = {frac_pos_toward - frac_pos_away:+.4f} (z={z_prop:.2f}, p={p_prop:.4f})")

    print("\n[6/6] Gerando figura e comparando com a versão horária (F6b original)...")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(d["expected_lag_min"], d["dt_min"], s=6, alpha=0.08, color="#2c7bb6")
    lims = [0, min(float(d["expected_lag_min"].quantile(0.99)), 30)]
    ax.plot(lims, lims, "k--", lw=1.5, label="advecção perfeita (y=x)")
    grid = np.linspace(0, lims[1], 50)
    ax.plot(grid, intercept + slope * grid, color="crimson", lw=2, label=f"OLS ajustado (slope={slope:.3f})")
    ax.axhline(0, color="grey", lw=0.8, linestyle=":")
    ax.set_xlim(0, lims[1])
    ax.set_ylim(d["dt_min"].min() - 0.5, d["dt_min"].max() + 0.5)
    ax.set_xlabel("Atraso ESPERADO por advecção pura (min) = dist/v_along")
    ax.set_ylabel("Atraso OBSERVADO entre eventos casados (dt_min, min)")
    ax.set_title(f"S4/F6b — atraso observado vs. esperado (KNMI 10min nativo)\n"
                 f"Spearman ρ={spearman_r:.3f} (IC95% [{ci_low:.3f}, {ci_high:.3f}])")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()

    significant = (spearman_p < 0.05) and not (ci_low < 0 < ci_high)
    meaningful = abs(spearman_r) > MIN_CORR_MEAN
    if significant and meaningful:
        decision = "SINAL DE ADVECÇÃO DETECTADO (KNMI 10min nativo) — atraso observado correlaciona com o esperado"
    elif significant and not meaningful:
        decision = "SEM SINAL PRATICAMENTE RELEVANTE (KNMI 10min nativo) — correlação detectável mas trivial"
    else:
        decision = "SEM SINAL DE ADVECÇÃO DETECTÁVEL (KNMI 10min nativo)"
    print(f"\n  DECISÃO: {decision}")

    # ── Comparação direta com a versão horária original (F6b, suffix "10m") ─────
    hourly_dec_path = cfg.DIRS["gates"] / "f6b_decision_10m.md"
    hourly_model_path = cfg.DIRS["gates"] / "f6b_timing_model_10m.md"
    hourly_rho, hourly_ci, hourly_p = None, None, None
    if hourly_model_path.exists():
        import re
        txt = hourly_model_path.read_text()
        m_rho = re.search(r"Spearman ρ \| (-?\d+\.\d+) \(p=(\d+\.\d+)\)", txt)
        m_ci = re.search(r"IC95% Spearman.*\| \((-?\d+\.\d+), (-?\d+\.\d+)\)", txt)
        if m_rho:
            hourly_rho, hourly_p = float(m_rho.group(1)), float(m_rho.group(2))
        if m_ci:
            hourly_ci = (float(m_ci.group(1)), float(m_ci.group(2)))

    OUT_MODEL.write_text(f"""# S4 — Re-teste de Advecção F6b com Vento KNMI de Resolução Nativa de 10min

**Data:** {date.today().isoformat()}
**Fonte de vento:** KNMI De Bilt, 10min nativo (`B7d_wind_knmi_10min.py`), casado por
arredondamento ao múltiplo de 10min mais próximo de `t_ext` (desalinhamento residual máx. ±5min,
vs. até ±30min no casamento horário original de F6b).

## Amostra
{len(matched):,} eventos casados com coordenadas válidas; vento de 10min disponível para
{n_wind_ok:,}/{len(matched):,} ({n_wind_ok/len(matched):.1%}). Restrito a
`v_along > {MIN_VALONG_MS} m/s`: **{len(d):,} eventos, {len(unique_pairs):,} pares direcionais únicos**.
Censura por desenho: {n_censored:,}/{len(d):,} ({n_censored/len(d):.1%}) com atraso esperado
maior que a janela de busca do C1b.

## Resultados — correlação atraso observado vs. esperado
| Métrica | Valor |
|---|---|
| Pearson r | {pearson_r:.4f} (p={pearson_p:.4f}) |
| Spearman ρ | {spearman_r:.4f} (p={spearman_p:.4f}) |
| IC95% Spearman ρ (bootstrap, B={len(boot_rhos)}/{N_BOOT}) | ({ci_low:.4f}, {ci_high:.4f}) |
| OLS slope | {slope:.4f} (SE {se_slope:.4f}) |
| OLS R² | {ols.rsquared:.4f} |

Limiar de relevância prática: |ρ| > {MIN_CORR_MEAN:.2f}.

## Teste complementar — consistência de sinal (n={len(matched):,})
| Condição | n | P(dt_min>0) |
|---|---|---|
| Vento aponta para o parceiro | {n_toward:,} | {frac_pos_toward:.4f} |
| Vento NÃO aponta para o parceiro | {n_away:,} | {frac_pos_away:.4f} |

Diferença: {frac_pos_toward - frac_pos_away:+.4f} (z={z_prop:.2f}, p={p_prop:.4f}).

## Decisão
**{decision}**

## Referência cruzada
- Fig.: `results/figures/s4_f6b_timing_scatter_10min_hires.png`
- Comparação com versão horária: `results/gates/s4_f6b_resolution_comparison.md`
- Versão horária original: `results/gates/f6b_timing_model_10m.md`
""")

    OUT_DEC.write_text(f"""# S4/F6b — Re-teste de Advecção com Vento de 10min

**Data:** {date.today().isoformat()}
**Decisão:** {decision}

Spearman ρ = {spearman_r:.4f} (IC95% [{ci_low:.4f}, {ci_high:.4f}]), p={spearman_p:.4f}.
Limiar de relevância prática: |ρ| > {MIN_CORR_MEAN:.2f} — {'atingido' if meaningful else 'NÃO atingido'}.
""")

    if hourly_rho is not None:
        delta_rho = spearman_r - hourly_rho
        conclusion = (
            f"A resolução temporal do vento (horário → 10min nativo) alterou ρ de "
            f"{hourly_rho:.4f} para {spearman_r:.4f} (Δ={delta_rho:+.4f}). "
            + ("A mudança inverteu a conclusão prática (agora relevante)." if (meaningful and not (abs(hourly_rho) > MIN_CORR_MEAN))
               else "A mudança NÃO alterou a conclusão prática — resolução temporal grosseira do vento horário "
                    "NÃO era a causa do resultado nulo de F6b. O resultado nulo (ausência de sinal de advecção "
                    "cronometrável) é robusto à granularidade temporal do vento de superfície, reforçando a "
                    "interpretação de regime compartilhado em vez de transporte físico direto mensurável por "
                    "este desenho.")
        )
        comp_rows = f"""| Resolução | n válido | Spearman ρ | IC95% | p | Relevante? |
|---|---|---|---|---|---|
| Horário (F6b original) | — | {hourly_rho:.4f} | {f'[{hourly_ci[0]:.4f}, {hourly_ci[1]:.4f}]' if hourly_ci else 'n/d'} | {hourly_p if hourly_p is not None else 'n/d'} | {'Sim' if abs(hourly_rho) > MIN_CORR_MEAN else 'Não'} |
| 10min nativo (S4, este script) | {len(d):,} | {spearman_r:.4f} | [{ci_low:.4f}, {ci_high:.4f}] | {spearman_p:.4f} | {'Sim' if meaningful else 'Não'} |"""
    else:
        conclusion = "Versão horária de referência (f6b_timing_model_10m.md) não encontrada para comparação direta."
        comp_rows = f"| 10min nativo (S4, este script) | {len(d):,} | {spearman_r:.4f} | [{ci_low:.4f}, {ci_high:.4f}] | {spearman_p:.4f} | {'Sim' if meaningful else 'Não'} |"

    OUT_COMP.write_text(f"""# S4 — Comparação: Vento Horário vs. 10min Nativo (Re-teste de Advecção, Risco C)

**Data:** {date.today().isoformat()}

## Motivação
`F6b_wind_timing.py` original usa vento horário (KNMI De Bilt). O atraso esperado por
advecção pode ficar mal estimado se o vento observado estiver defasado até ±30min do
instante real do evento condicionante. Este script repete a mesma análise central com vento
de resolução nativa de 10min (desalinhamento residual máx. ±5min), para verificar se essa
granularidade era responsável pelo resultado nulo agregado de F6b.

## Tabela de comparação
{comp_rows}

## Conclusão
{conclusion}

## Referência cruzada
- Modelo detalhado (10min): `results/gates/s4_f6b_timing_model_10min_hires.md`
- Modelo detalhado (horário, original): `results/gates/f6b_timing_model_10m.md`
- Fig. (10min): `results/figures/s4_f6b_timing_scatter_10min_hires.png`
""")
    print(f"\n  Salvo: {OUT_COMP.relative_to(cfg.ROOT)}")

    log_result(
        script="S4_f6b_wind_10min.py",
        gate="",
        phase="RiscoC",
        params={
            "model": "dt_min (observed lag) ~ expected_lag_min (dist_km*1000/v_along/60)",
            "wind_source": "KNMI De Bilt, 10min nativo (B7d)",
            "min_valong_ms": MIN_VALONG_MS,
            "min_corr_meaningful": MIN_CORR_MEAN,
            "n_bootstrap_pairs": N_BOOT,
        },
        results={
            "n_matched_total": len(matched),
            "n_wind_10min_matched": n_wind_ok,
            "n_valid_valong_filtered": len(d),
            "n_unique_directional_pairs": len(unique_pairs),
            "pearson_r": round(float(pearson_r), 4),
            "spearman_rho": round(float(spearman_r), 4),
            "spearman_p": round(float(spearman_p), 4),
            "spearman_ci_low": round(float(ci_low), 4),
            "spearman_ci_high": round(float(ci_high), 4),
            "ols_slope": round(float(slope), 4),
            "ols_r2": round(float(ols.rsquared), 4),
            "sign_consistency_frac_pos_toward": round(frac_pos_toward, 4),
            "sign_consistency_frac_pos_away": round(frac_pos_away, 4),
            "sign_consistency_p": round(float(p_prop), 4),
            "effect_size_meaningful": bool(meaningful),
            "hourly_spearman_rho_reference": hourly_rho,
        },
        decision=decision,
        action="Advection timing test re-run with native 10min KNMI wind resolution (Risco C, ROADMAP Secao 0.2), "
               "to test whether the F6b null result was an artifact of hourly wind resolution.",
        interpretation=(
            f"Spearman rho={spearman_r:.4f} (p={spearman_p:.4f}, CI [{ci_low:.4f},{ci_high:.4f}]), "
            f"OLS slope={slope:.4f}, R2={ols.rsquared:.4f}. {decision}. {conclusion}"
        ),
        paper_ref="Section 8 (F6 spatial structure) -- advection timing test, temporal-resolution robustness check (Risco C).",
    )

    print(f"\n{SEP}")
    print("S4 concluído.")
    print(f"  Decisão: {decision}")
    print(SEP)


if __name__ == "__main__":
    main()
