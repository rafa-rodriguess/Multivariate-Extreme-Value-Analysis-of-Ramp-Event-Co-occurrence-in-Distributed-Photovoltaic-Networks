"""
F6b_wind_timing.py — Teste de advecção: atraso OBSERVADO vs. atraso ESPERADO (dist/vento)
================================================================================
`F6_anisotropy.py` testou só a DIREÇÃO do vento (o excesso de coincidência é maior
quando o vento aponta para o parceiro?) e encontrou efeito estatisticamente
detectável mas praticamente trivial com vento de superfície (e efeito relevante
mas de SINAL INVERTIDO com CERRA em 200/500m — ver `f6_anisotropy_comparison.md`).
Este script implementa um teste mais forte e mais direto do mecanismo físico de
advecção, combinando distância + velocidade REAL do vento + tempo, em vez de só
direção:

  Se uma nuvem realmente viaja de i para j carregada pelo vento, o atraso OBSERVADO
  entre o evento extremo em i e o evento casado em j (`dt_min`, já calculado em
  `C1b_event_pairing.py`) deveria se aproximar do atraso ESPERADO por advecção pura:

      atraso_esperado (min) = distância_ij (m) / v_along (m/s) / 60

  onde v_along é a componente da velocidade real do vento na direção geodésica de
  i para j (mesmo ângulo de alinhamento do F6_anisotropy.py, agora multiplicado
  pela velocidade real, não só o cosseno do ângulo).

Restrito ao subconjunto já CASADO (`matched=True`, 84.776 eventos de
`aligned_pairs.parquet`) com v_along > `cfg.F6['min_valong_ms']` (abaixo disso a
previsão de atraso explode/não tem sentido físico — vento fraco ou não apontando
para o parceiro não permite prever um tempo de chegada).

Roda automaticamente para TODAS as fontes de vento disponíveis (KNMI 10m +
CERRA 100/200/500m), como checagem de robustez por altura — nenhuma assumida a
priori como a correta.

Saídas (uma família por altura, sufixo `_<altura>`, mais comparação):
  - `results/gates/f6b_timing_model_<altura>.md`
  - `results/gates/f6b_timing_bootstrap_<altura>.parquet`
  - `results/figures/f6b_timing_scatter_<altura>.png`
  - `results/gates/f6b_decision_<altura>.md`
  - `results/gates/f6b_timing_comparison.md`
  - `results/figures/f6b_timing_comparison.png`

Executar:
    python F6b_wind_timing.py
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

ALIGNED_PQ   = cfg.DIRS["processed"] / "aligned_pairs.parquet"
EVENTPAIR_PQ = cfg.DIRS["gates"]     / "event_pairing_summary.parquet"
COORDS_PQ    = cfg.DIRS["interim"]   / "coords.parquet"
WIND_PQ      = cfg.DIRS["interim"]   / "wind_joined.parquet"

OUT_COMPARISON_MD  = cfg.DIRS["gates"]   / "f6b_timing_comparison.md"
OUT_COMPARISON_FIG = cfg.DIRS["figures"] / "f6b_timing_comparison.png"

N_BOOT          = cfg.F6["n_bootstrap_pairs"]      # 150
MIN_VALONG_MS   = cfg.F6["min_valong_ms"]          # 1.0
MIN_CORR_MEAN   = cfg.F6["min_corr_meaningful"]    # 0.10
SEED            = cfg.SEED

SEP = "─" * 60

WIND_SOURCES = [
    ("wind_speed_ms",            "wind_dir_deg",            "KNMI De Bilt, 10m (superfície)", "10m",  True),
    ("cerra_wind_speed_ms_100m", "cerra_wind_dir_deg_100m", "CERRA, 100m",                     "100m", False),
    ("cerra_wind_speed_ms_200m", "cerra_wind_dir_deg_200m", "CERRA, 200m",                     "200m", False),
    ("cerra_wind_speed_ms_500m", "cerra_wind_dir_deg_500m", "CERRA, 500m",                     "500m", False),
]


def run_for_source(matched_base: pd.DataFrame, wind: pd.DataFrame,
                    speed_col: str, dir_col: str, label: str, suffix: str, has_synth_flag: bool) -> dict | None:
    print(f"\n{SEP}")
    print(f"  Fonte de vento: {label}  ({speed_col} / {dir_col})")
    print(SEP)

    if speed_col not in wind.columns or dir_col not in wind.columns:
        print(f"  AVISO: colunas '{speed_col}'/'{dir_col}' não encontradas — pulando esta altura.")
        return None

    OUT_MODEL = cfg.DIRS["gates"]   / f"f6b_timing_model_{suffix}.md"
    OUT_BOOT  = cfg.DIRS["gates"]   / f"f6b_timing_bootstrap_{suffix}.parquet"
    OUT_FIG   = cfg.DIRS["figures"] / f"f6b_timing_scatter_{suffix}.png"
    OUT_DEC   = cfg.DIRS["gates"]   / f"f6b_decision_{suffix}.md"

    matched = matched_base.copy()

    # ── Vento real no instante do evento condicionante + componente na direção do par ─
    print(f"\n[3/5] Casando vento real ({label}) no instante t_ext e projetando na direção do par...")
    wind_cols = ["station_id", "start_ts", speed_col, dir_col] + (["wind_synthetic"] if has_synth_flag else [])
    wind_lookup = wind[wind_cols].copy()
    wind_lookup["start_ts"] = pd.to_datetime(wind_lookup["start_ts"], utc=True)
    matched = matched.merge(
        wind_lookup, left_on=["station_ext", "t_ext"], right_on=["station_id", "start_ts"], how="left",
    )
    matched["travel_dir_deg"] = (matched[dir_col] + 180.0) % 360.0
    matched["alignment"] = circular_alignment(matched["travel_dir_deg"].to_numpy(), matched["bearing_deg"].to_numpy())
    matched["v_along_ms"] = matched[speed_col] * matched["alignment"]

    if has_synth_flag:
        n_real_wind = int((~matched["wind_synthetic"].fillna(True)).sum())
    else:
        n_real_wind = int(matched[speed_col].notna().sum())
    print(f"  Vento casado: {matched[speed_col].notna().sum():,}/{len(matched):,} "
          f"({n_real_wind:,} reais). Distribuição de v_along (m/s): "
          f"mediana={matched['v_along_ms'].median():.2f}, "
          f"% apontando para o parceiro (v_along>0) = {(matched['v_along_ms'] > 0).mean():.1%}")

    # ── Atraso esperado por advecção pura, e comparação com o observado ─
    print(f"\n[4/5] Calculando atraso esperado (dist/v_along) e comparando com dt_min observado "
          f"(restrito a v_along > {MIN_VALONG_MS} m/s)...")
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

    print("\n  Checagem por direção da rampa (down=sombra chegando vs. up=recuperação)...")
    same_dir_frac = float((matched["dir_ext"] == matched["dir_partner"]).mean())
    dir_results = {}
    for dlabel, key in [("down", "down"), ("up", "up")]:
        dsub = d[(d["dir_ext"] == key) & (d["dir_partner"] == key)]
        if len(dsub) >= 30:
            rho_d, p_d = st.spearmanr(dsub["dt_min"], dsub["expected_lag_min"])
        else:
            rho_d, p_d = np.nan, np.nan
        dir_results[dlabel] = {"n": len(dsub), "rho": rho_d, "p": p_d}
        print(f"  Só '{dlabel}': n={len(dsub):,}  Spearman ρ={rho_d:.4f}  p={p_d:.4f}")

    X = sm.add_constant(d["expected_lag_min"].to_numpy())
    y = d["dt_min"].to_numpy()
    ols = sm.OLS(y, X).fit()
    intercept, slope = ols.params
    se_intercept, se_slope = ols.bse
    print(f"  OLS: dt_min = {intercept:.3f} + {slope:.4f}·atraso_esperado (SE={se_slope:.4f}; R²={ols.rsquared:.4f})")

    print(f"  Cluster bootstrap (por par direcional, B={N_BOOT}) do ρ de Spearman...")
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
    from statsmodels.stats.proportion import proportions_ztest
    z_prop, p_prop = proportions_ztest(count, nobs)
    print(f"  P(dt_min>0 | vento aponta p/ parceiro, n={n_toward:,}) = {frac_pos_toward:.4f}")
    print(f"  P(dt_min>0 | vento NÃO aponta p/ parceiro, n={n_away:,}) = {frac_pos_away:.4f}")
    print(f"  Diferença = {frac_pos_toward - frac_pos_away:+.4f} (z={z_prop:.2f}, p={p_prop:.4f})")

    print("\n[5/5] Gerando figura de dispersão (atraso observado vs. esperado)...")
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
    ax.set_title(f"F6b — atraso observado vs. esperado ({label})\nSpearman ρ={spearman_r:.3f} (IC95% [{ci_low:.3f}, {ci_high:.3f}])")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()

    significant = (spearman_p < 0.05) and not (ci_low < 0 < ci_high)
    meaningful = abs(spearman_r) > MIN_CORR_MEAN
    if significant and meaningful:
        decision = f"SINAL DE ADVECÇÃO DETECTADO ({label}) — atraso observado correlaciona com o esperado"
    elif significant and not meaningful:
        decision = f"SEM SINAL PRATICAMENTE RELEVANTE ({label}) — correlação detectável mas trivial"
    else:
        decision = f"SEM SINAL DE ADVECÇÃO DETECTÁVEL ({label})"
    print(f"\n  DECISÃO ({label}): {decision}")

    OUT_MODEL.write_text(f"""# F6b — Teste de Advecção: Atraso Observado vs. Esperado ({label})

**Data:** {date.today().isoformat()}
**Fonte de vento:** {label}

## Amostra
{len(matched):,} eventos casados com coordenadas válidas; restrito a
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

## Checagem por direção da rampa
{same_dir_frac:.1%} dos casados têm mesma direção em ambas as pontas (emergente, não imposto).

| Direção | n | Spearman ρ | p |
|---|---|---|---|
| Só "down" | {dir_results['down']['n']:,} | {dir_results['down']['rho']:.4f} | {dir_results['down']['p']:.4f} |
| Só "up" | {dir_results['up']['n']:,} | {dir_results['up']['rho']:.4f} | {dir_results['up']['p']:.4f} |

## Interpretação
{'Correlação estatisticamente detectável E de tamanho de efeito relevante (|ρ|>' + f'{MIN_CORR_MEAN:.2f})' + ' -- evidência de sinal de advecção nesta altura.' if (significant and meaningful) else f'Correlação {"detectável (p<0.05) mas" if significant else "não distinguível de zero e"} de tamanho de efeito abaixo do limiar de relevância prática ({MIN_CORR_MEAN:.2f}).'}
**Decisão: {decision}**

## Referência cruzada
- Fig.: `results/figures/f6b_timing_scatter_{suffix}.png`
- Comparação entre alturas: `results/gates/f6b_timing_comparison.md`
""")

    OUT_DEC.write_text(f"""# F6b — Teste de Advecção ({label})

**Data:** {date.today().isoformat()}
**Decisão:** {decision}

Spearman ρ = {spearman_r:.4f} (IC95% [{ci_low:.4f}, {ci_high:.4f}]), p={spearman_p:.4f}.
Limiar de relevância prática: |ρ| > {MIN_CORR_MEAN:.2f} — {'atingido' if meaningful else 'NÃO atingido'}.
""")

    log_result(
        script="F6b_wind_timing.py",
        gate="",
        phase="F6b",
        params={
            "model": "dt_min (observed lag) ~ expected_lag_min (dist_km*1000/v_along/60)",
            "wind_source": label,
            "wind_height": suffix,
            "min_valong_ms": MIN_VALONG_MS,
            "min_corr_meaningful": MIN_CORR_MEAN,
            "n_bootstrap_pairs": N_BOOT,
        },
        results={
            "n_matched_total": len(matched),
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
            "same_direction_match_frac": round(same_dir_frac, 4),
            "spearman_rho_down_only": round(float(dir_results["down"]["rho"]), 4) if np.isfinite(dir_results["down"]["rho"]) else None,
        },
        decision=decision,
        action=f"Advection timing test re-run for wind height {suffix} ({label}) as part of the multi-height robustness comparison.",
        interpretation=(
            f"Wind source: {label}. Spearman rho={spearman_r:.4f} (p={spearman_p:.4f}, CI "
            f"[{ci_low:.4f},{ci_high:.4f}]), OLS slope={slope:.4f}, R2={ols.rsquared:.4f}. "
            f"Sign-consistency test difference={frac_pos_toward-frac_pos_away:+.4f} (p={p_prop:.4f}). "
            f"{decision}. See f6b_timing_comparison.md for the full cross-height comparison."
        ),
        paper_ref="Section 8 (F6 spatial structure) -- advection timing test, multi-height robustness check",
    )

    return {
        "label": label, "suffix": suffix, "n": len(d), "n_pairs": len(unique_pairs),
        "spearman_r": spearman_r, "ci_low": ci_low, "ci_high": ci_high, "p": spearman_p,
        "slope": slope, "meaningful": meaningful, "significant": significant, "decision": decision,
    }


def main() -> None:
    print(SEP)
    print("F6b — TESTE DE ADVECÇÃO: ATRASO OBSERVADO vs. ESPERADO (DIST/VENTO)")
    print("      (rodando para todas as alturas de vento disponíveis)")
    print(SEP)

    for p in (ALIGNED_PQ, EVENTPAIR_PQ, COORDS_PQ, WIND_PQ):
        if not p.exists():
            print(f"\nERRO: {p} não encontrado.")
            sys.exit(1)

    aligned = pd.read_parquet(ALIGNED_PQ)
    event_summary = pd.read_parquet(EVENTPAIR_PQ)
    coords = pd.read_parquet(COORDS_PQ).dropna(subset=["lat_centroid", "lon_centroid"]).set_index("station_id")
    wind = pd.read_parquet(WIND_PQ)

    print("\n[1/5] Restringindo ao subconjunto casado (dt_min observado)...")
    matched = aligned[aligned["matched"] & aligned["dt_min"].notna()].copy()
    have_coords = matched["station_ext"].isin(coords.index) & matched["station_partner"].isin(coords.index)
    matched = matched.loc[have_coords].copy()
    print(f"  Eventos casados com coords válidas: {len(matched):,}")

    print("\n[2/5] Calculando bearing geodésico e distância por par...")
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

    results = []
    for speed_col, dir_col, label, suffix, has_synth in WIND_SOURCES:
        r = run_for_source(matched, wind, speed_col, dir_col, label, suffix, has_synth)
        if r is not None:
            results.append(r)

    if not results:
        print("\nERRO: nenhuma fonte de vento pôde ser processada.")
        sys.exit(1)

    print(f"\n{SEP}\nCOMPARAÇÃO ENTRE ALTURAS\n{SEP}")
    comp_df = pd.DataFrame(results)
    print(comp_df[["label", "n", "spearman_r", "p", "slope", "meaningful"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(results))
    rhos = [r["spearman_r"] for r in results]
    errs_low = [r["spearman_r"] - r["ci_low"] for r in results]
    errs_high = [r["ci_high"] - r["spearman_r"] for r in results]
    colors = ["crimson" if r["significant"] and r["meaningful"] else "#2c7bb6" for r in results]
    ax.errorbar(rhos, y_pos, xerr=[errs_low, errs_high], fmt="o", capsize=4, color="black", ecolor="grey")
    for i, r in enumerate(results):
        ax.scatter([r["spearman_r"]], [i], color=colors[i], s=80, zorder=3)
    ax.axvline(0, color="grey", lw=1, linestyle=":")
    ax.axvline(MIN_CORR_MEAN, color="green", lw=1, linestyle="--", alpha=0.5)
    ax.axvline(-MIN_CORR_MEAN, color="green", lw=1, linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in results])
    ax.set_xlabel("Spearman ρ (atraso observado vs. esperado) — IC95% bootstrap")
    ax.set_title("F6b — correlação de timing de advecção por altura de vento")
    plt.tight_layout()
    plt.savefig(OUT_COMPARISON_FIG, dpi=150)
    plt.close()
    print(f"\n  Figura comparativa: {OUT_COMPARISON_FIG.relative_to(cfg.ROOT)}")

    any_meaningful = any(r["significant"] and r["meaningful"] for r in results)
    OUT_COMPARISON_MD.write_text(f"""# F6b — Comparação do Teste de Advecção (Timing) entre Alturas de Vento

**Data:** {date.today().isoformat()}

| Fonte de vento | n eventos | Spearman ρ | IC95% | p | OLS slope | Relevante? | Decisão |
|---|---|---|---|---|---|---|---|
{chr(10).join(f"| {r['label']} | {r['n']:,} | {r['spearman_r']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['p']:.4f} | {r['slope']:.4f} | {'Sim' if r['meaningful'] else 'Não'} | {r['decision']} |" for r in results)}

## Conclusão
{'Pelo menos uma altura mostrou sinal de advecção (timing) estatisticamente significativo E praticamente relevante -- reportar no paper qual altura e com que magnitude.' if any_meaningful else 'NENHUMA altura testada (superfície nem CERRA 100/200/500m) mostrou correlação praticamente relevante entre o atraso OBSERVADO e o atraso ESPERADO por advecção pura (distância/velocidade). Isso é um resultado mais forte que o teste de mera direção em `f6_anisotropy_comparison.md`: mesmo nas alturas onde a DIREÇÃO do vento mostrou um efeito (sinal invertido) sobre a PROBABILIDADE de coincidência, a MAGNITUDE do atraso não é prevista pela velocidade real do vento em nenhuma altura testada. Reforça a conclusão de que a dependência de cauda é de regime compartilhado, não de advecção física cronometrável por este desenho -- e que o efeito de sinal invertido em F6 provavelmente reflete confundimento (ex.: direção do vento correlacionada com estação do ano/padrão sinótico), não um mecanismo de transporte direto.'}

## Referência cruzada
- Figs.: `results/figures/f6b_timing_scatter_<altura>.png`, `results/figures/f6b_timing_comparison.png`
- Modelos individuais: `results/gates/f6b_timing_model_<altura>.md`
- Ver também: `results/gates/f6_anisotropy_comparison.md` (teste de direção apenas)
""")
    print(f"  Salvo: {OUT_COMPARISON_MD.relative_to(cfg.ROOT)}")

    print(f"\n{SEP}")
    print("F6b (todas as alturas) — resumo:")
    for r in results:
        print(f"  {r['label']:35s} → {r['decision']}")


if __name__ == "__main__":
    main()
