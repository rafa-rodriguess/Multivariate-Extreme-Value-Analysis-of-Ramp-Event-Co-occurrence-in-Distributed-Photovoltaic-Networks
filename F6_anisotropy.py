"""
F6_anisotropy.py — Estrutura espacial: anisotropia vs. direção real do vento
================================================================================
Segunda (e última pendente) pergunta do F6 do ROADMAP. A primeira ("χ̂ vs.
distância: exponencial ou potência?") já foi respondida por um canal
independente em `F5_two_stage.py` (Estágio 2, α(dist) — exponencial venceu por
AIC). Este script cobre a parte de ANISOTROPIA: a dependência de cauda entre
pares é mais forte quando o vento ambiente "empurra" de uma usina para a
outra, do que quando é perpendicular ou contrário?

Desenho adaptado (ver ROADMAP, nota pré-F6 "achado importante" do C1b):
O achado do C1b mostrou que o acoplamento evento-a-evento (mesma magnitude,
poucos minutos de diferença) é fraco — a dependência detectada no Gate G1 é
majoritariamente efeito de REGIME COMPARTILHADO. Isso invalida a abordagem
ingênua de estimar velocidade/direção de propagação a partir da defasagem
temporal entre eventos pareados (o desenho original do F6 pressupunha isso
implicitamente). Em vez disso, testamos aqui se o ALINHAMENTO entre a direção
do vento no momento do evento condicionante e o bearing geográfico do par
explica variação na força do regime compartilhado já estabelecido (Gate G1
aprovado, 52,1% dos pares significativos) — reaproveitando exatamente o
desenho do Estágio 1 de `F5_two_stage.py` (regressão logística do EXCESSO de
coincidência sobre a nula de Poisson, com offset), com uma covariável nova:

    alignment_ij = cos(travel_dir_vento − bearing_ij)

onde travel_dir_vento = wind_dir_deg + 180° (a direção CONVENCIONAL do vento é
"de onde vem"; o ar viaja na direção oposta) e bearing_ij é o rumo geodésico
inicial de station_ext para station_partner. alignment=+1 significa "vento
soprando de ext para partner" (a favor); -1 significa "vento soprando de
partner para ext" (contra); 0 significa perpendicular.

Fontes de vento testadas (roda automaticamente para TODAS, uma checagem de
robustez por altura, nenhuma assumida a priori como "a correta"):
  - KNMI De Bilt, 10 m, superfície (B7_wind_join.py)
  - CERRA, 100 / 200 / 500 m (B7b_wind_cerra.py) — mais próximo da altura real
    de base de nuvem que o vento de superfície.

Saídas (uma família de arquivos por altura, sufixo `_<altura>`, mais um resumo
comparativo entre alturas):
  - `results/gates/f6_anisotropy_model_<altura>.md`
  - `results/gates/f6_anisotropy_bootstrap_<altura>.parquet`
  - `results/figures/f6_anisotropy_bins_<altura>.png`
  - `results/gates/f6_decision_<altura>.md`
  - `results/gates/f6_anisotropy_comparison.md` — tabela comparativa entre alturas
  - `results/figures/f6_anisotropy_comparison.png` — coeficiente de alinhamento por altura (forest plot)

Executar:
    python F6_anisotropy.py
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

RAMPS_PQ     = cfg.DIRS["interim"]   / "ramps_split.parquet"
ALIGNED_PQ   = cfg.DIRS["processed"] / "aligned_pairs.parquet"
EVENTPAIR_PQ = cfg.DIRS["gates"]     / "event_pairing_summary.parquet"
COORDS_PQ    = cfg.DIRS["interim"]   / "coords.parquet"
WIND_PQ      = cfg.DIRS["interim"]   / "wind_joined.parquet"

OUT_COMPARISON_MD  = cfg.DIRS["gates"]   / "f6_anisotropy_comparison.md"
OUT_COMPARISON_FIG = cfg.DIRS["figures"] / "f6_anisotropy_comparison.png"

N_BOOT   = cfg.F6["n_bootstrap_pairs"]   # 150
N_BINS   = cfg.F6["alignment_bins"]      # 5
MIN_EFFECT_REL = cfg.F6["min_effect_size_rel"]  # 0.05
SEED     = cfg.SEED
DIST_FLOOR_KM = 0.1

SEP = "─" * 60

# (speed_col, dir_col, label, suffix, is_synthetic_flagged)
WIND_SOURCES = [
    ("wind_speed_ms",          "wind_dir_deg",          "KNMI De Bilt, 10m (superfície)", "10m",  True),
    ("cerra_wind_speed_ms_100m", "cerra_wind_dir_deg_100m", "CERRA, 100m",                 "100m", False),
    ("cerra_wind_speed_ms_200m", "cerra_wind_dir_deg_200m", "CERRA, 200m",                 "200m", False),
    ("cerra_wind_speed_ms_500m", "cerra_wind_dir_deg_500m", "CERRA, 500m",                 "500m", False),
]


def bearing_deg(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Rumo geodésico inicial (graus, 0-360, 0=Norte) de (lat1,lon1) para (lat2,lon2)."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    theta = np.degrees(np.arctan2(x, y))
    return (theta + 360.0) % 360.0


def circular_alignment(travel_dir_deg: np.ndarray, bearing_deg_: np.ndarray) -> np.ndarray:
    """cos(diferença angular) entre a direção de viagem do ar e o bearing do par.
    +1 = vento a favor (ext->partner); -1 = vento contra; 0 = perpendicular."""
    diff = np.radians(travel_dir_deg - bearing_deg_)
    return np.cos(diff)


def run_for_source(aligned_base: pd.DataFrame, ramps: pd.DataFrame, event_summary: pd.DataFrame,
                    coords: pd.DataFrame, wind: pd.DataFrame,
                    speed_col: str, dir_col: str, label: str, suffix: str, has_synth_flag: bool) -> dict | None:
    print(f"\n{SEP}")
    print(f"  Fonte de vento: {label}  ({speed_col} / {dir_col})")
    print(SEP)

    if speed_col not in wind.columns or dir_col not in wind.columns:
        print(f"  AVISO: colunas '{speed_col}'/'{dir_col}' não encontradas em wind_joined.parquet — pulando esta altura.")
        return None

    OUT_MODEL = cfg.DIRS["gates"]   / f"f6_anisotropy_model_{suffix}.md"
    OUT_BOOT  = cfg.DIRS["gates"]   / f"f6_anisotropy_bootstrap_{suffix}.parquet"
    OUT_FIG   = cfg.DIRS["figures"] / f"f6_anisotropy_bins_{suffix}.png"
    OUT_DEC   = cfg.DIRS["gates"]   / f"f6_decision_{suffix}.md"

    aligned = aligned_base.copy()

    # ── [2/5] Vento real no momento do evento condicionante ──────
    print(f"\n[2/5] Casando vento real ({label}) no instante t_ext...")
    wind_cols = ["station_id", "start_ts", speed_col, dir_col] + (["wind_synthetic"] if has_synth_flag else [])
    wind_lookup = wind[wind_cols].copy()
    wind_lookup["start_ts"] = pd.to_datetime(wind_lookup["start_ts"], utc=True)
    aligned = aligned.merge(
        wind_lookup, left_on=["station_ext", "t_ext"], right_on=["station_id", "start_ts"], how="left",
    )
    if has_synth_flag:
        n_real_wind = int((~aligned["wind_synthetic"].fillna(True)).sum())
    else:
        n_real_wind = int(aligned[speed_col].notna().sum())
    n_valid_dir = int(aligned[dir_col].notna().sum())
    print(f"  Vento casado: {aligned[speed_col].notna().sum():,}/{len(aligned):,} "
          f"({n_real_wind:,} reais, {n_valid_dir:,} com direção válida)")

    aligned["travel_dir_deg"] = (aligned[dir_col] + 180.0) % 360.0
    aligned["alignment"] = circular_alignment(aligned["travel_dir_deg"].to_numpy(), aligned["bearing_deg"].to_numpy())

    # ── [3/5] Distância e p_null direcional (mesmo desenho do Estágio 1 de F5) ─
    dist_lookup, dtwin_lookup = {}, {}
    for _, row in event_summary.iterrows():
        d_km = row["dist_ij_m"] / 1000.0
        dist_lookup[(row["station_i"], row["station_j"])] = d_km
        dist_lookup[(row["station_j"], row["station_i"])] = d_km
        dtwin_lookup[(row["station_i"], row["station_j"])] = row["dt_window_min"]
        dtwin_lookup[(row["station_j"], row["station_i"])] = row["dt_window_min"]

    lam_by_station = {}
    for sid, g in ramps.groupby("station_id"):
        ts = pd.to_datetime(g["start_ts"])
        span_min = (ts.max() - ts.min()).total_seconds() / 60.0 if len(g) > 1 else np.nan
        lam_by_station[sid] = len(g) / span_min if span_min and span_min > 0 else np.nan

    def p_null_directional(ext_i: str, partner_j: str) -> float:
        lam_j = lam_by_station.get(partner_j, np.nan)
        dtw = dtwin_lookup.get((ext_i, partner_j), np.nan)
        if not (np.isfinite(lam_j) and np.isfinite(dtw)):
            return np.nan
        return 1.0 - np.exp(-lam_j * 2.0 * dtw)

    aligned["dist_km"] = [dist_lookup.get((a, b), np.nan) for a, b in zip(aligned["station_ext"], aligned["station_partner"])]
    aligned["p_null"] = [p_null_directional(a, b) for a, b in zip(aligned["station_ext"], aligned["station_partner"])]

    # ── [4/5] Regressão logística: excesso de coincidência COM vs. SEM alinhamento ─
    print("\n[4/5] Ajustando modelo (excesso de coincidência ~ offset(p_null) + dist + alinhamento)...")
    valid = (np.isfinite(aligned["dist_km"]) & np.isfinite(aligned["p_null"]) &
             (aligned["p_null"] > 0) & (aligned["p_null"] < 1) & np.isfinite(aligned["alignment"]))
    d = aligned.loc[valid].copy()
    print(f"  Amostra válida (dist + p_null + vento com direção): {len(d):,}/{len(aligned):,}")

    log_dist = np.log(np.maximum(d["dist_km"].to_numpy(), DIST_FLOOR_KM))
    align = d["alignment"].to_numpy()
    y = d["matched"].to_numpy().astype(float)
    p_null_clip = np.clip(d["p_null"].to_numpy(), 1e-4, 1 - 1e-4)
    offset = np.log(p_null_clip / (1 - p_null_clip))

    def fit_logit(X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset).fit()
            null_dev = m.null_deviance
        return m, null_dev

    X_base = sm.add_constant(log_dist)
    X_full = sm.add_constant(np.column_stack([log_dist, align]))
    model_base, nulldev_base = fit_logit(X_base)
    model_full, nulldev_full = fit_logit(X_full)

    g0, g_dist, g_align = model_full.params
    se0, se_dist, se_align = model_full.bse
    z_align = g_align / se_align
    p_align = 2 * (1 - st.norm.cdf(abs(z_align)))

    aic_base = model_base.aic
    aic_full = model_full.aic
    pseudo_r2_full = 1 - model_full.deviance / nulldev_full

    print(f"  Sem alinhamento: AIC={aic_base:.1f}")
    print(f"  Com alinhamento: AIC={aic_full:.1f}  (ΔAIC={aic_base-aic_full:+.2f}, negativo=sem-alinhamento melhor)")
    print(f"  logit(P(match)) = logit(p_null) + {g0:.3f} + {g_dist:.3f}·log(dist_km) + {g_align:.3f}·alignment")
    print(f"  Coeficiente de alinhamento: {g_align:.4f} (erro-padrão {se_align:.4f}, z={z_align:.2f}, p={p_align:.4f})")
    print(f"  Pseudo-R² (McFadden, modelo completo): {pseudo_r2_full:.4f}")

    # ── Tamanho de efeito (não só significância) ──────────────────────────────
    med_logdist = float(np.median(log_dist))
    med_offset = float(np.median(offset))
    logit_minus1 = g0 + g_dist * med_logdist + g_align * (-1.0) + med_offset
    logit_plus1  = g0 + g_dist * med_logdist + g_align * (+1.0) + med_offset
    p_minus1 = 1.0 / (1.0 + np.exp(-logit_minus1))
    p_plus1  = 1.0 / (1.0 + np.exp(-logit_plus1))
    p_base   = (p_minus1 + p_plus1) / 2.0
    effect_abs = p_plus1 - p_minus1
    effect_rel = effect_abs / p_base if p_base > 0 else np.nan
    print(f"  Tamanho de efeito: P(match) vai de {p_minus1:.4f} (contra) a {p_plus1:.4f} (a favor) "
          f"-- Δ={effect_abs:+.4f} ({effect_rel:+.1%} relativo)")

    # ── Cluster bootstrap (por par direcional) do coeficiente de alinhamento ──
    print(f"  Cluster bootstrap (por par direcional, B={N_BOOT}) do coeficiente de alinhamento...")
    pair_id = (d["station_ext"] + "__" + d["station_partner"]).to_numpy()
    unique_pairs = np.unique(pair_id)
    pair_to_idx = {p: np.where(pair_id == p)[0] for p in unique_pairs}
    rng = np.random.default_rng(SEED)
    boot_coefs = []
    for _ in range(N_BOOT):
        sampled_pairs = rng.choice(unique_pairs, size=len(unique_pairs), replace=True)
        idx_b = np.concatenate([pair_to_idx[p] for p in sampled_pairs])
        Xb = sm.add_constant(np.column_stack([log_dist[idx_b], align[idx_b]]))
        yb = y[idx_b]
        offb = offset[idx_b]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mb = sm.GLM(yb, Xb, family=sm.families.Binomial(), offset=offb).fit()
            boot_coefs.append(mb.params[2])
        except Exception:
            continue
    boot_coefs = np.array(boot_coefs)
    ci_low, ci_high = np.percentile(boot_coefs, [2.5, 97.5])
    print(f"  Bootstrap: {len(boot_coefs)}/{N_BOOT} convergiram. IC95%: ({ci_low:.4f}, {ci_high:.4f})")
    pd.DataFrame({"boot_align_coef": boot_coefs}).to_parquet(OUT_BOOT, index=False)

    # ── [5/5] Figura ───────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p_pred_full = model_full.predict(X_full, offset=offset)
    d = d.assign(p_pred=p_pred_full, p_null_col=p_null_clip)
    d["align_bin"] = pd.qcut(d["alignment"], N_BINS, duplicates="drop")
    binned = d.groupby("align_bin", observed=True).agg(
        align_mid=("alignment", "mean"), obs=("matched", "mean"),
        pred=("p_pred", "mean"), null_=("p_null_col", "mean"), n=("matched", "size"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(binned["align_mid"], binned["obs"], "o-", label="observado", color="#2c7bb6")
    ax.plot(binned["align_mid"], binned["pred"], "s--", label="previsto (logit + alinhamento)", color="crimson")
    ax.plot(binned["align_mid"], binned["null_"], "^:", label="nula (independência)", color="grey")
    ax.set_xlabel("Alinhamento vento-bearing (+1=a favor, -1=contra, 0=perpendicular)")
    ax.set_ylabel("P(coincidência)")
    ax.set_title(f"F6 — coincidência por faixa de alinhamento ({label})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()

    # ── Decisão ────────────────────────────────────────────────────────────
    significant = (p_align < 0.05) and not (ci_low < 0 < ci_high)
    meaningful = abs(effect_rel) > MIN_EFFECT_REL
    aligned_helps = g_align > 0
    if significant and meaningful and aligned_helps:
        decision = f"ANISOTROPIA DETECTADA ({label}) — efeito significativo E relevante"
    elif significant and meaningful and not aligned_helps:
        decision = f"ANISOTROPIA DETECTADA ({label}, sinal invertido) — investigar antes de interpretar"
    elif significant and not meaningful:
        decision = f"SEM ANISOTROPIA PRATICAMENTE RELEVANTE ({label}) — significativo mas efeito trivial"
    else:
        decision = f"SEM ANISOTROPIA DETECTÁVEL ({label})"
    print(f"\n  DECISÃO ({label}): {decision}")

    OUT_MODEL.write_text(f"""# F6 — Modelo de Anisotropia (alinhamento com vento real)

**Data:** {date.today().isoformat()}
**Fonte de vento:** {label}

## Especificação
logit(P(matched)) = offset(logit(p_null)) + γ₀ + γ_dist·log(dist_km) + γ_align·alignment

`alignment = cos(travel_dir_vento - bearing_geodésico(ext->partner))`, onde
`travel_dir_vento = {dir_col} + 180°`. +1 = vento a favor do par, -1 = contra, 0 = perpendicular.
Amostrado no instante do evento condicionante (t_ext). Mesmo desenho de offset do Estágio 1
de `F5_two_stage.py`.

## Amostra
{len(d):,} eventos extremos avaliados, {len(unique_pairs):,} pares direcionais únicos.

## Resultados
| Modelo | AIC |
|---|---|
| Sem alinhamento | {aic_base:.1f} |
| Com alinhamento | {aic_full:.1f} |

| Parâmetro | Estimativa | Erro-padrão | z | p |
|---|---|---|---|---|
| γ₀ | {g0:.4f} | {se0:.4f} | — | — |
| γ_dist | {g_dist:.4f} | {se_dist:.4f} | — | — |
| γ_align | {g_align:.4f} | {se_align:.4f} | {z_align:.2f} | {p_align:.4f} |

IC95% bootstrap (B={len(boot_coefs)}/{N_BOOT}) de γ_align: **({ci_low:.4f}, {ci_high:.4f})**
Pseudo-R² (McFadden): {pseudo_r2_full:.4f}

## Tamanho de efeito
P(match): {p_minus1:.4f} (contra) → {p_plus1:.4f} (a favor). Δ={effect_abs:+.4f}
({effect_rel:+.1%} relativo). Limiar de relevância: {MIN_EFFECT_REL:.0%}.
Efeito **{'praticamente relevante' if meaningful else 'trivial'}**.

## Coincidência por faixa de alinhamento
{binned[["align_mid", "obs", "pred", "null_", "n"]].to_markdown(index=False)}

## Interpretação
γ_align {'positivo' if g_align > 0 else 'negativo'} e {'estatisticamente significativo' if significant else 'não distinguível de zero'}
(p={p_align:.4f}), efeito {'relevante' if meaningful else f'TRIVIAL ({effect_rel:+.1%})'}.
**Decisão: {decision}**

## Referência cruzada
- Fig.: `results/figures/f6_anisotropy_bins_{suffix}.png`
- Comparação entre alturas: `results/gates/f6_anisotropy_comparison.md`
""")

    OUT_DEC.write_text(f"""# F6 — Anisotropia vs. Direção do Vento ({label})

**Data:** {date.today().isoformat()}
**Decisão:** {decision}

γ_align = {g_align:.4f} (IC95% [{ci_low:.4f}, {ci_high:.4f}]), p={p_align:.4f}.
Tamanho de efeito: {effect_rel:+.1%} relativo (limiar {MIN_EFFECT_REL:.0%}).
""")

    log_result(
        script="F6_anisotropy.py",
        gate="",
        phase="F6",
        params={
            "model": "logit(matched) ~ offset(logit(p_null)) + log(dist_km) + alignment",
            "wind_source": label,
            "wind_height": suffix,
            "n_bootstrap_pairs": N_BOOT,
        },
        results={
            "n_events": len(d),
            "n_unique_directional_pairs": len(unique_pairs),
            "gamma_align": round(float(g_align), 4),
            "gamma_align_p": round(float(p_align), 4),
            "gamma_align_ci_low": round(float(ci_low), 4),
            "gamma_align_ci_high": round(float(ci_high), 4),
            "effect_size_rel": round(float(effect_rel), 4),
            "effect_size_meaningful": bool(meaningful),
            "aic_base": round(float(aic_base), 2),
            "aic_full": round(float(aic_full), 2),
        },
        decision=decision,
        action=f"Anisotropy test (wind-alignment covariate on F5 Stage 1 design) re-run for wind height {suffix} ({label}) as part of the multi-height robustness comparison.",
        interpretation=(
            f"Wind source: {label}. gamma_align={g_align:.4f} (p={p_align:.4f}, CI [{ci_low:.4f},{ci_high:.4f}]), "
            f"effect size {effect_rel:+.1%} relative ({'meaningful' if meaningful else 'trivial'}, threshold {MIN_EFFECT_REL:.0%}). "
            f"{decision}. See f6_anisotropy_comparison.md for the full cross-height comparison."
        ),
        paper_ref="Section 8 (F6 spatial structure) -- anisotropy vs. real wind direction, multi-height robustness check",
    )

    return {
        "label": label, "suffix": suffix, "n": len(d), "n_pairs": len(unique_pairs),
        "gamma_align": g_align, "se_align": se_align, "ci_low": ci_low, "ci_high": ci_high,
        "p_align": p_align, "effect_rel": effect_rel, "meaningful": meaningful,
        "significant": significant, "decision": decision,
    }


def main() -> None:
    print(SEP)
    print("F6 — ANISOTROPIA: ALINHAMENTO COM VENTO REAL vs. EXCESSO DE COINCIDÊNCIA")
    print("     (rodando para todas as alturas de vento disponíveis)")
    print(SEP)

    for p in (RAMPS_PQ, ALIGNED_PQ, EVENTPAIR_PQ, COORDS_PQ, WIND_PQ):
        if not p.exists():
            print(f"\nERRO: {p} não encontrado.")
            sys.exit(1)

    ramps_all = pd.read_parquet(RAMPS_PQ)
    ramps = ramps_all[ramps_all["split"] == "train"].copy()
    aligned = pd.read_parquet(ALIGNED_PQ)
    event_summary = pd.read_parquet(EVENTPAIR_PQ)
    coords = pd.read_parquet(COORDS_PQ).dropna(subset=["lat_centroid", "lon_centroid"]).set_index("station_id")
    wind = pd.read_parquet(WIND_PQ)

    print(f"\n  Eventos extremos avaliados (aligned_pairs): {len(aligned):,}")
    print(f"  ...dos quais casados: {int(aligned['matched'].sum()):,} ({aligned['matched'].mean():.1%})")

    print("\n[1/5] Calculando bearing geodésico (station_ext -> station_partner)...")
    have_coords = aligned["station_ext"].isin(coords.index) & aligned["station_partner"].isin(coords.index)
    aligned = aligned.loc[have_coords].copy()
    lat1 = coords.loc[aligned["station_ext"], "lat_centroid"].to_numpy()
    lon1 = coords.loc[aligned["station_ext"], "lon_centroid"].to_numpy()
    lat2 = coords.loc[aligned["station_partner"], "lat_centroid"].to_numpy()
    lon2 = coords.loc[aligned["station_partner"], "lon_centroid"].to_numpy()
    aligned["bearing_deg"] = bearing_deg(lat1, lon1, lat2, lon2)
    aligned["t_ext"] = pd.to_datetime(aligned["t_ext"], utc=True)
    print(f"  Bearing calculado para {len(aligned):,} eventos.")

    results = []
    for speed_col, dir_col, label, suffix, has_synth in WIND_SOURCES:
        r = run_for_source(aligned, ramps, event_summary, coords, wind, speed_col, dir_col, label, suffix, has_synth)
        if r is not None:
            results.append(r)

    if not results:
        print("\nERRO: nenhuma fonte de vento pôde ser processada.")
        sys.exit(1)

    # ── Comparação entre alturas ───────────────────────────────────────────
    print(f"\n{SEP}\nCOMPARAÇÃO ENTRE ALTURAS\n{SEP}")
    comp_df = pd.DataFrame(results)
    print(comp_df[["label", "n", "gamma_align", "p_align", "effect_rel", "meaningful"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(results))
    gammas = [r["gamma_align"] for r in results]
    errs_low = [r["gamma_align"] - r["ci_low"] for r in results]
    errs_high = [r["ci_high"] - r["gamma_align"] for r in results]
    colors = ["crimson" if r["significant"] and r["meaningful"] else "#2c7bb6" for r in results]
    ax.errorbar(gammas, y_pos, xerr=[errs_low, errs_high], fmt="o", capsize=4, color="black", ecolor="grey")
    for i, r in enumerate(results):
        ax.scatter([r["gamma_align"]], [i], color=colors[i], s=80, zorder=3)
    ax.axvline(0, color="grey", lw=1, linestyle=":")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in results])
    ax.set_xlabel("γ_align (coeficiente de alinhamento com vento) — IC95% bootstrap")
    ax.set_title("F6 — coeficiente de anisotropia por altura de vento")
    plt.tight_layout()
    plt.savefig(OUT_COMPARISON_FIG, dpi=150)
    plt.close()
    print(f"\n  Figura comparativa: {OUT_COMPARISON_FIG.relative_to(cfg.ROOT)}")

    any_meaningful = any(r["significant"] and r["meaningful"] for r in results)
    OUT_COMPARISON_MD.write_text(f"""# F6 — Comparação de Anisotropia entre Alturas de Vento

**Data:** {date.today().isoformat()}

Mesmo teste de anisotropia (alinhamento vento-bearing sobre o desenho de offset do
Estágio 1 de F5), repetido para cada altura de vento disponível, sem assumir a priori
qual altura é a fisicamente correta.

| Fonte de vento | n eventos | γ_align | IC95% | p | Δ relativo | Relevante? | Decisão |
|---|---|---|---|---|---|---|---|
{chr(10).join(f"| {r['label']} | {r['n']:,} | {r['gamma_align']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['p_align']:.4f} | {r['effect_rel']:+.1%} | {'Sim' if r['meaningful'] else 'Não'} | {r['decision']} |" for r in results)}

## Conclusão
{'Pelo menos uma altura mostrou anisotropia estatisticamente significativa E praticamente relevante -- reportar no paper qual altura e com que magnitude, como achado explícito (não assumido a priori).' if any_meaningful else 'NENHUMA altura testada (superfície nem CERRA 100/200/500m) mostrou anisotropia praticamente relevante por alinhamento com o vento. Isso fortalece a conclusão de que a dependência de cauda detectada no Gate G1 é predominantemente de REGIME COMPARTILHADO (atividade meteorológica correlacionada na escala da rede), não de propagação/advecção direcional local capturável por este desenho. Ver também `f6b_timing_model.md` para o teste complementar de tempo (distância/velocidade), que testa o mesmo mecanismo de forma mais direta.'}

## Referência cruzada
- Figs.: `results/figures/f6_anisotropy_bins_<altura>.png`, `results/figures/f6_anisotropy_comparison.png`
- Modelos individuais: `results/gates/f6_anisotropy_model_<altura>.md`
- Ver também: `results/gates/f6b_timing_model.md` (teste de tempo, distância/velocidade)
""")
    print(f"  Salvo: {OUT_COMPARISON_MD.relative_to(cfg.ROOT)}")

    print(f"\n{SEP}")
    print("F6 (todas as alturas) — resumo:")
    for r in results:
        print(f"  {r['label']:35s} → {r['decision']}")


if __name__ == "__main__":
    main()
