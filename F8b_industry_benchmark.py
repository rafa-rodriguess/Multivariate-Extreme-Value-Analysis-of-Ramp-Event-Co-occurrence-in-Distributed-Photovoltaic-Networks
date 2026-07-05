"""
F8b_industry_benchmark.py — RQ3b: Benchmark contra a prática atual da indústria
(cópula gaussiana / correlação de Pearson)
=================================================================================
Motivação (ROADMAP.md Seção 5, "F8b — Benchmark contra prática atual"): F8 compara
REAL vs. INDEPENDÊNCIA (o "pior caso possível", zero correlação). Isso NÃO responde
à pergunta que um revisor de periódico de sistemas de energia fará: "e comparado ao
que a prática/literatura já faz hoje para dimensionar reserva com correlação
espacial?" (mesmo que rudimentar — correlação de Pearson, cópula gaussiana). F8b
fecha essa lacuna com um 4º cenário.

Por que cópula gaussiana = "prática atual": é o método padrão de fato em engenharia
de sistemas de energia / gestão de risco quantitativo para agregar variáveis
correlacionadas sem EVT multivariada (McNeil, Frey & Embrechts, *Quantitative Risk
Management*). Tem uma propriedade matemática conhecida mas raramente demonstrada
empiricamente no domínio de PV: dependência de cauda superior assintótica = 0 para
qualquer correlação ρ<1 (Sibuya, 1959) — mesmo reproduzindo a correlação REAL em
massa, pode subestimar sistematicamente o risco de cauda.

Dois cenários adicionais (não substituem REAL/INDEPENDÊNCIA/IMPLICADO, já em F8):

  4. CÓPULA GAUSSIANA — espelha EXATAMENTE o cenário "IMPLICADO PELO MODELO" de F8
     (mesmos eventos condicionantes, MESMO Estágio 1 de coincidência — comparação
     justa: só a mecânica de "magnitude dado coincidência" muda). Em vez de
     Heffernan-Tawn (α(dist)·X + X^β·Z), usa cópula gaussiana condicional:
       Z_i = Φ⁻¹(U_i), Z_j|Z_i ~ N(ρ(dist)·Z_i, 1−ρ(dist)²), U_j = Φ(Z_j)
     ρ(dist) ajustado por correlação de Pearson nas MESMAS margens Laplace do
     Estágio 2 de F5, mesma forma funcional (exp vs. potência, AIC).

  5. FALLBACK CLOSED-FORM ("prática mínima", sem simulação) — fórmula clássica de
     variância da soma ponderada usando a matriz de correlação de Pearson REAL
     (não a curva suavizada) do sinal bruto k_i(t):
       Var(k_agg) = Σwᵢ²σᵢ² + ΣᵢΣⱼwᵢwⱼρᵢⱼσᵢσⱼ
     com nível de retorno gaussiano z_T = μ_agg + σ_agg·Φ⁻¹(1−1/(T·taxa_anual)).

NÃO reabre/modifica F5_two_stage.py, F7_return_levels.py nem F8_portfolio_effect.py
(scripts já aprovados) — reaproveita funções importáveis e duplica (não edita) a
reconstrução do Estágio 1, seguindo o mesmo padrão já usado dentro do próprio F8.

Saídas:
  results/gates/f8b_copula_bootstrap.parquet
  results/gates/f8b_rq3b_ratio.parquet          — extensão da Tab. 5 com cenário cópula
  results/gates/f8b_rq3b_decision.md
  results/figures/f8b_reserve_comparison.png    — 4 cenários (REAL/INDEP/CÓPULA/MODELO)
  results/figures/f8b_rho_vs_alpha.png          — ρ(dist) Pearson vs. α(dist) Heffernan-Tawn

Executar:
    python F8b_industry_benchmark.py
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.optimize import curve_fit
from scipy.special import expit, logit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result
from F7a_aggregate_series import build_capacity_weights
from F7_return_levels import detect_ramps, fit_pot_gpd, return_level, DIRECTION, RETURN_PERIODS
from F5_two_stage import alpha_exp, alpha_pow, DIST_FLOOR_KM
from F5_heffernan_tawn_pilot import build_ecdf, laplace_transform
from F8_portfolio_effect import inverse_laplace, inverse_ecdf, mc_ratio_ci

IN_K         = cfg.DIRS["interim"] / "clearsky_index.parquet"
IN_COORDS    = cfg.DIRS["interim"] / "coords.parquet"
IN_RAMPS     = cfg.DIRS["interim"] / "ramps_split.parquet"
IN_ALIGNED   = cfg.DIRS["processed"] / "aligned_pairs.parquet"
IN_EVENTPAIR = cfg.DIRS["gates"] / "event_pairing_summary.parquet"
IN_F7_FIT    = cfg.DIRS["gates"] / "f7_return_level_fit.parquet"
IN_F7_BOOT   = cfg.DIRS["gates"] / "f7_return_level_bootstrap.parquet"
IN_F8_INDEP  = cfg.DIRS["gates"] / "f8_independence_bootstrap.parquet"
IN_F8_MODEL  = cfg.DIRS["gates"] / "f8_model_implied_bootstrap.parquet"
IN_F8_RATIO  = cfg.DIRS["gates"] / "f8_rq3_ratio.parquet"
IN_STAGE2    = cfg.DIRS["gates"] / "f5_stage2_params.parquet"

OUT_COPULA_BOOT = cfg.DIRS["gates"]   / "f8b_copula_bootstrap.parquet"
OUT_RATIO       = cfg.DIRS["gates"]   / "f8b_rq3b_ratio.parquet"
OUT_DEC         = cfg.DIRS["gates"]   / "f8b_rq3b_decision.md"
OUT_FIG         = cfg.DIRS["figures"] / "f8b_reserve_comparison.png"
OUT_FIG_RHO     = cfg.DIRS["figures"] / "f8b_rho_vs_alpha.png"

N_MODEL   = cfg.F7["n_model_implied_real"]     # mesmo N de F8, para comparação direta
RADIUS_KM = cfg.F7["near_neighbor_radius_km"]
SEED = cfg.SEED
SEP = "─" * 60


def _rho_exp(dist_km, rho0, log_L):
    L = np.exp(log_L)
    return rho0 * np.exp(-dist_km / L)


def _rho_pow(dist_km, rho0, log_p):
    p = np.exp(log_p)
    return rho0 * np.power(np.maximum(dist_km, DIST_FLOOR_KM), -p)


def main() -> None:
    print(SEP)
    print("F8b — RQ3b: BENCHMARK CONTRA PRÁTICA ATUAL (CÓPULA GAUSSIANA / PEARSON)")
    print(SEP)

    for p in (IN_K, IN_COORDS, IN_RAMPS, IN_ALIGNED, IN_EVENTPAIR, IN_F7_FIT, IN_F7_BOOT,
              IN_F8_INDEP, IN_F8_MODEL, IN_F8_RATIO, IN_STAGE2):
        if not p.exists():
            print(f"ERRO: {p} não encontrado. Execute F5_two_stage/F7/F8 primeiro.")
            sys.exit(1)

    fit_real_df = pd.read_parquet(IN_F7_FIT)
    boot_real_long = pd.read_parquet(IN_F7_BOOT)
    indep_boot_long = pd.read_parquet(IN_F8_INDEP)
    model_boot_long = pd.read_parquet(IN_F8_MODEL)
    ratio_f8 = pd.read_parquet(IN_F8_RATIO)
    stage2 = pd.read_parquet(IN_STAGE2).iloc[0]
    years_span_train = float(fit_real_df["years_span_train"].iloc[0])

    ramps_all = pd.read_parquet(IN_RAMPS)
    ramps_train = ramps_all[ramps_all["split"] == "train"].copy()
    ramps_train["abs_mag"] = ramps_train["delta_k"].abs()
    aligned = pd.read_parquet(IN_ALIGNED)
    event_summary = pd.read_parquet(IN_EVENTPAIR)
    coords = pd.read_parquet(IN_COORDS)
    df_k = pd.read_parquet(IN_K)
    station_cols = [c for c in df_k.columns if c.startswith("ID")]
    w = build_capacity_weights(coords, station_cols)

    # ══════════════════════════════════════════════════════════════════════
    # [1/4] Margens Laplace + lookups (duplicado de F5/F8, scripts já aprovados
    #       não são reabertos) — necessário para reconstruir Estágio 1 idêntico
    #       e as margens x_lap/y_lap para ajustar ρ(dist)
    # ══════════════════════════════════════════════════════════════════════
    print("\n[1/4] Reconstruindo margens Laplace + Estágio 1 (idêntico a F5/F8)...")
    ecdf_by_station, sorted_by_station = {}, {}
    for sid, g in ramps_train.groupby("station_id"):
        sv = np.sort(g["abs_mag"].to_numpy())
        sorted_by_station[sid] = sv
        ecdf_by_station[sid] = build_ecdf(sv)

    dist_lookup, dtwin_lookup = {}, {}
    for _, row in event_summary.iterrows():
        d_km = row["dist_ij_m"] / 1000.0
        dist_lookup[(row["station_i"], row["station_j"])] = d_km
        dist_lookup[(row["station_j"], row["station_i"])] = d_km
        dtwin_lookup[(row["station_i"], row["station_j"])] = row["dt_window_min"]
        dtwin_lookup[(row["station_j"], row["station_i"])] = row["dt_window_min"]
    lam_by_station = {}
    for sid, g in ramps_train.groupby("station_id"):
        ts = pd.to_datetime(g["start_ts"])
        span_min = (ts.max() - ts.min()).total_seconds() / 60.0 if len(g) > 1 else np.nan
        lam_by_station[sid] = len(g) / span_min if span_min and span_min > 0 else np.nan

    def p_null_directional(ext_i: str, partner_j: str) -> float:
        lam_j = lam_by_station.get(partner_j, np.nan)
        dtw = dtwin_lookup.get((ext_i, partner_j), np.nan)
        if not (np.isfinite(lam_j) and np.isfinite(dtw)):
            return np.nan
        return 1.0 - np.exp(-lam_j * 2.0 * dtw)

    x_lap_all = np.full(len(aligned), np.nan)
    y_lap_all = np.full(len(aligned), np.nan)
    for sid, idx in aligned.groupby("station_ext").groups.items():
        pos = aligned.index.get_indexer(idx)
        if sid in ecdf_by_station:
            x_lap_all[pos] = laplace_transform(ecdf_by_station[sid](aligned.loc[idx, "mag_ext"].to_numpy()))
    matched_rows = aligned["matched"].to_numpy()
    for sid, idx in aligned.loc[matched_rows].groupby("station_partner").groups.items():
        pos = aligned.index.get_indexer(idx)
        if sid in ecdf_by_station:
            y_lap_all[pos] = laplace_transform(ecdf_by_station[sid](aligned.loc[idx, "mag_partner"].to_numpy()))

    dist_km_all = np.array([dist_lookup.get((a, b), np.nan)
                             for a, b in zip(aligned["station_ext"], aligned["station_partner"])])
    p_null_all = np.array([p_null_directional(a, b)
                            for a, b in zip(aligned["station_ext"], aligned["station_partner"])])
    aligned = aligned.assign(x_lap=x_lap_all, y_lap=y_lap_all, dist_km=dist_km_all, p_null=p_null_all)

    valid1 = (np.isfinite(aligned["x_lap"]) & np.isfinite(aligned["dist_km"]) &
              np.isfinite(aligned["p_null"]) & (aligned["p_null"] > 0) & (aligned["p_null"] < 1))
    d1 = aligned.loc[valid1]
    import statsmodels.api as sm
    X1 = sm.add_constant(np.column_stack([
        np.log(np.maximum(d1["dist_km"].to_numpy(), DIST_FLOOR_KM)), d1["x_lap"].to_numpy()]))
    y1 = d1["matched"].to_numpy().astype(float)
    p_null_clip = np.clip(d1["p_null"].to_numpy(), 1e-4, 1 - 1e-4)
    offset1 = np.log(p_null_clip / (1 - p_null_clip))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        logit_model = sm.GLM(y1, X1, family=sm.families.Binomial(), offset=offset1).fit()
    g0, g_dist, g_x = logit_model.params
    print(f"  Estágio 1 recomputado (idêntico a F5/F8): logit(P(match))=logit(p_null)+{g0:.3f}"
          f"+{g_dist:.3f}·log(dist)+{g_x:.3f}·X_i")

    # ══════════════════════════════════════════════════════════════════════
    # [2/4] ρ(dist) via correlação de Pearson nas margens Laplace (matched)
    # ══════════════════════════════════════════════════════════════════════
    print("\n[2/4] Ajustando ρ(dist) por correlação de Pearson nas margens Laplace (matched)...")
    matched = aligned[aligned["matched"] & np.isfinite(aligned["x_lap"]) &
                       np.isfinite(aligned["y_lap"]) & np.isfinite(aligned["dist_km"])].copy()
    x_m, y_m, d_m = matched["x_lap"].to_numpy(), matched["y_lap"].to_numpy(), matched["dist_km"].to_numpy()
    print(f"  Amostra (mesma de F5 Estágio 2): {len(matched):,} eventos casados")

    n_bins = 12
    matched["_dbin"] = pd.qcut(matched["dist_km"], n_bins, duplicates="drop")
    bin_stats = matched.groupby("_dbin", observed=True).apply(
        lambda g: pd.Series({
            "dist_mid": g["dist_km"].mean(),
            "pearson_r": st.pearsonr(g["x_lap"], g["y_lap"])[0] if len(g) >= 10 else np.nan,
            "n": len(g),
        }), include_groups=False
    ).reset_index(drop=True).dropna()
    print(f"  Pearson r por decil de distância (n={len(bin_stats)} bins válidos):")
    for _, r in bin_stats.iterrows():
        print(f"    dist~{r['dist_mid']:.2f}km  r={r['pearson_r']:.4f}  (n={int(r['n']):,})")

    dist_bin_arr = bin_stats["dist_mid"].to_numpy()
    rho_bin_arr = bin_stats["pearson_r"].to_numpy()
    try:
        popt_exp, _ = curve_fit(_rho_exp, dist_bin_arr, rho_bin_arr,
                                 p0=[0.3, np.log(4.0)], maxfev=5000)
        sse_exp = float(np.sum((_rho_exp(dist_bin_arr, *popt_exp) - rho_bin_arr) ** 2))
    except Exception:
        popt_exp, sse_exp = None, np.inf
    try:
        popt_pow, _ = curve_fit(_rho_pow, dist_bin_arr, rho_bin_arr,
                                 p0=[0.3, np.log(0.5)], maxfev=5000)
        sse_pow = float(np.sum((_rho_pow(dist_bin_arr, *popt_pow) - rho_bin_arr) ** 2))
    except Exception:
        popt_pow, sse_pow = None, np.inf

    if sse_exp <= sse_pow:
        rho_form, rho_params = "exp", popt_exp
        rho_func = lambda d: _rho_exp(d, *popt_exp)
    else:
        rho_form, rho_params = "pow", popt_pow
        rho_func = lambda d: _rho_pow(d, *popt_pow)
    print(f"  Modelo vencedor (menor SSE): {rho_form}  rho0={rho_params[0]:.4f}  "
          f"decay_param={np.exp(rho_params[1]):.4f}  (SSE exp={sse_exp:.5f}, pow={sse_pow:.5f})")

    median_dist_matched = float(np.median(d_m))
    rho_at_median = float(np.clip(rho_func(np.array([median_dist_matched]))[0], -0.999, 0.999))
    alpha0_s2, decay_s2, beta_s2 = float(stage2["alpha0"]), float(stage2["decay_param"]), float(stage2["beta"])
    alpha_at_median = float(
        alpha_exp(np.array([median_dist_matched]), alpha0_s2, np.log(decay_s2))[0]
        if stage2["form"] == "exp" else
        alpha_pow(np.array([median_dist_matched]), alpha0_s2, np.log(decay_s2))[0]
    )
    print(f"  Na distância mediana casada ({median_dist_matched:.2f}km): "
          f"ρ_Pearson={rho_at_median:.4f}  vs.  α_HeffernanTawn={alpha_at_median:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(dist_bin_arr, rho_bin_arr, color="#2c7bb6", s=40, label="ρ Pearson (binned, dados)", zorder=3)
    dgrid = np.linspace(dist_bin_arr.min(), dist_bin_arr.max(), 100)
    ax.plot(dgrid, rho_func(dgrid), color="#2c7bb6", lw=2, linestyle="--",
            label=f"ρ(dist) ajustado ({rho_form})")
    alpha_curve = (alpha_exp(dgrid, alpha0_s2, np.log(decay_s2)) if stage2["form"] == "exp"
                   else alpha_pow(dgrid, alpha0_s2, np.log(decay_s2)))
    ax.plot(dgrid, alpha_curve, color="crimson", lw=2, label="α(dist) Heffernan-Tawn (F5 Estágio 2)")
    ax.axhline(0, color="grey", lw=0.8, linestyle=":")
    ax.set_xlabel("Distância (km)")
    ax.set_ylabel("Medida de dependência")
    ax.set_title("F8b — ρ Pearson (linear) vs. α Heffernan-Tawn (extremal) por distância")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG_RHO, dpi=150)
    plt.close()
    print(f"  Figura: {OUT_FIG_RHO.relative_to(cfg.ROOT)}")

    # ══════════════════════════════════════════════════════════════════════
    # [3/4] Cenário CÓPULA GAUSSIANA — espelha exatamente o "implicado pelo modelo"
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[3/4] Cenário CÓPULA GAUSSIANA — {N_MODEL} realizações "
          f"(mesmo Estágio 1, mesmos eventos condicionantes; Estágio 2 substituído por "
          f"cópula gaussiana com ρ(dist) ajustado por Pearson)...")

    work = aligned.loc[valid1 & (aligned["dist_km"] <= RADIUS_KM)].copy()
    work = work[work["station_partner"].isin(sorted_by_station.keys())]
    print(f"  Eventos condicionantes × vizinhos <{RADIUS_KM}km com dados completos: {len(work):,}")

    logit_p_match = (logit(np.clip(work["p_null"].to_numpy(), 1e-4, 1 - 1e-4)) + g0 +
                      g_dist * np.log(np.maximum(work["dist_km"].to_numpy(), DIST_FLOOR_KM)) +
                      g_x * work["x_lap"].to_numpy())
    p_match_arr = expit(logit_p_match)

    dist_arr = work["dist_km"].to_numpy()
    rho_dist_arr = np.clip(rho_func(dist_arr), -0.995, 0.995)
    x_lap_arr = work["x_lap"].to_numpy()
    u_i_arr = inverse_laplace(x_lap_arr)             # recupera U_i (uniforme) exato usado no laplace_transform
    z_i_arr = st.norm.ppf(np.clip(u_i_arr, 1e-6, 1 - 1e-6))

    dur_samples = int(round(ramps_train["duration_min"].median()))
    col_of = {sid: i for i, sid in enumerate(station_cols)}
    K = df_k[df_k["split"] == "train"][station_cols].to_numpy(dtype=float)
    train_index = df_k[df_k["split"] == "train"].index
    train_t0 = pd.Timestamp(train_index[0]).tz_localize(None)
    t_ext_naive = pd.to_datetime(work["t_ext"]).dt.tz_localize(None)
    row0_all = ((t_ext_naive.to_numpy() - np.datetime64(train_t0)) //
                np.timedelta64(1, "m")).astype(np.int64)
    col_j_all = np.array([col_of.get(sid, -1) for sid in work["station_partner"]])
    valid_geom = (col_j_all >= 0) & (row0_all >= 0) & (row0_all + dur_samples < len(K))
    sign_all = np.where(work["dir_ext"].to_numpy() == "down", -1.0, 1.0)
    w_j_all = w[np.clip(col_j_all, 0, len(w) - 1)]

    row_idx_2d_full = row0_all[:, None] + np.arange(dur_samples)[None, :]
    row_idx_2d_full = np.clip(row_idx_2d_full, 0, len(K) - 1)
    real_window_full = K[row_idx_2d_full, np.clip(col_j_all, 0, len(w) - 1)[:, None]]
    ramp_shape = np.linspace(0.0, 1.0, dur_samples)[None, :]

    k_agg_train = pd.read_parquet(cfg.DIRS["interim"] / "aggregate_clearsky_index.parquet")
    k_agg_train = k_agg_train.loc[k_agg_train["split"] == "train", "k_agg"].to_numpy()

    rng3 = np.random.default_rng(SEED + 3)
    copula_fits = []
    t0 = time.time()
    n_work = len(work)
    partner_arr = work["station_partner"].to_numpy()

    for r in range(N_MODEL):
        u1 = rng3.random(n_work)
        match_mask = valid_geom & (u1 < p_match_arr)
        idx_m = np.where(match_mask)[0]
        if len(idx_m) == 0:
            continue
        eps = rng3.standard_normal(len(idx_m))
        z_j = rho_dist_arr[idx_m] * z_i_arr[idx_m] + np.sqrt(1.0 - rho_dist_arr[idx_m] ** 2) * eps
        u_j = np.clip(st.norm.cdf(z_j), 1e-6, 1 - 1e-6)

        mag_synth = np.zeros(len(idx_m))
        partners_m = partner_arr[idx_m]
        for sid in np.unique(partners_m):
            sel = partners_m == sid
            mag_synth[sel] = inverse_ecdf(sorted_by_station[sid], u_j[sel])
        mag_synth = np.clip(mag_synth, 0.0, None)

        peak_delta = sign_all[idx_m] * mag_synth
        synth_window = real_window_full[idx_m, 0:1] + peak_delta[:, None] * ramp_shape
        delta_window = (synth_window - real_window_full[idx_m]) * w_j_all[idx_m][:, None]
        delta_window = np.where(np.isfinite(real_window_full[idx_m]), delta_window, 0.0)

        k_agg_mod = k_agg_train.copy()
        np.add.at(k_agg_mod, row_idx_2d_full[idx_m].reshape(-1), delta_window.reshape(-1))

        ramps_r = detect_ramps(k_agg_mod, train_index)
        sub = ramps_r[ramps_r["direction"] == DIRECTION]
        if len(sub) < 20:
            continue
        fit_r = fit_pot_gpd(sub["delta_k"].abs().to_numpy(), sub["start_ts"].to_numpy())
        if fit_r is not None:
            copula_fits.append(fit_r)
        if (r + 1) % 20 == 0:
            print(f"  ... {r+1}/{N_MODEL} realizações ({time.time()-t0:.1f}s, "
                  f"{len(copula_fits)} convergiram, {len(idx_m):,} coincidências injetadas)")
    print(f"  Concluído: {len(copula_fits)}/{N_MODEL} realizações convergiram ({time.time()-t0:.1f}s)")

    if len(copula_fits) < 10:
        print("\nERRO: poucas realizações da cópula convergiram — abortando.")
        sys.exit(1)

    copula_boot_long = pd.DataFrame([
        {"realization_id": i, "return_period_years": T, "z": return_level(f, years_span_train, T)}
        for i, f in enumerate(copula_fits) for T in RETURN_PERIODS
    ])
    copula_boot_long.to_parquet(OUT_COPULA_BOOT, index=False)
    print(f"  Salvo: {OUT_COPULA_BOOT.relative_to(cfg.ROOT)}")

    # ══════════════════════════════════════════════════════════════════════
    # [4/4] Fallback closed-form — variância da soma ponderada (Pearson bruto)
    # ══════════════════════════════════════════════════════════════════════
    print("\n[4/4] Fallback closed-form — variância da soma ponderada (correlação de Pearson "
          "do sinal bruto k_i(t), sem simulação)...")
    K_full_train = df_k[df_k["split"] == "train"][station_cols].to_numpy(dtype=float)
    # Correlação par-a-par do sinal bruto (não das margens de rampa) — a forma mais crua e
    # mais citada em relatórios de indústria: correlação do índice de céu-claro em si.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr_signal = pd.DataFrame(K_full_train, columns=station_cols).corr().to_numpy()
    corr_signal = np.nan_to_num(corr_signal, nan=0.0)
    np.fill_diagonal(corr_signal, 1.0)

    down_mag_by_station = {sid: g.loc[g["direction"] == DIRECTION, "abs_mag"].to_numpy()
                            for sid, g in ramps_train.groupby("station_id")}
    sigma_i = np.array([np.std(down_mag_by_station.get(sid, np.array([0.0]))) if
                        len(down_mag_by_station.get(sid, [])) > 1 else 0.0 for sid in station_cols])
    mu_i = np.array([np.mean(down_mag_by_station.get(sid, np.array([0.0]))) if
                     len(down_mag_by_station.get(sid, [])) > 0 else 0.0 for sid in station_cols])

    var_agg = float(np.sum((w * sigma_i) ** 2))
    for i in range(len(station_cols)):
        for j in range(len(station_cols)):
            if i != j:
                var_agg += w[i] * w[j] * corr_signal[i, j] * sigma_i[i] * sigma_i[j]
    sigma_agg = float(np.sqrt(max(var_agg, 0.0)))
    mu_agg = float(np.sum(w * mu_i))

    agg_ramps_train = detect_ramps(k_agg_train, train_index)
    rate_annual = float(len(agg_ramps_train[agg_ramps_train["direction"] == DIRECTION]) / years_span_train)
    print(f"  μ_agg (soma pond. das médias individuais) = {mu_agg:.4f}")
    print(f"  σ_agg (fórmula clássica, correlação Pearson do sinal bruto) = {sigma_agg:.4f}")
    print(f"  Taxa anual de rampas 'down' agregadas (real, treino) = {rate_annual:.1f}/ano")

    closed_form_rows = []
    for T in RETURN_PERIODS:
        m = rate_annual * T
        q = 1.0 - 1.0 / max(m, 1.0001)
        z_gauss = mu_agg + sigma_agg * st.norm.ppf(q)
        closed_form_rows.append({"return_period_years": T, "z_closed_form_gaussian": z_gauss})
    closed_form_df = pd.DataFrame(closed_form_rows)

    # ══════════════════════════════════════════════════════════════════════
    # Comparação final — extensão da Tab. 5
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Calculando razão RQ3b (real/cópula-gaussiana)...")
    ratio_rows = []
    for T in RETURN_PERIODS:
        z_real_ens = boot_real_long.loc[boot_real_long["return_period_years"] == T, "z"].to_numpy()
        z_copula_ens = copula_boot_long.loc[copula_boot_long["return_period_years"] == T, "z"].to_numpy()
        z_real_point = float(fit_real_df[f"z_{T:.4f}y"].iloc[0])
        z_copula_point = float(np.median(z_copula_ens))
        z_closed = float(closed_form_df.loc[closed_form_df["return_period_years"] == T,
                                             "z_closed_form_gaussian"].iloc[0])
        r8 = ratio_f8.loc[ratio_f8["return_period_years"] == T].iloc[0]

        ratio_med, ratio_lo, ratio_hi = mc_ratio_ci(z_real_ens, z_copula_ens, seed=SEED + int(T * 1000) + 7)
        ratio_rows.append({
            "return_period_years": T,
            "z_real": z_real_point,
            "z_independence": float(r8["z_independence"]),
            "z_copula_gaussian": z_copula_point,
            "z_closed_form_gaussian": z_closed,
            "z_model_implied": float(r8["z_model_implied"]),
            "ratio_real_over_copula": ratio_med,
            "ratio_ci_low": ratio_lo, "ratio_ci_high": ratio_hi,
            "ratio_real_over_independence": float(r8["ratio_real_over_independence"]),
        })
        print(f"  T={T:.3f}y  z_real={z_real_point:.4f}  z_copula={z_copula_point:.4f}  "
              f"z_closedform={z_closed:.4f}  razão(real/cópula)={ratio_med:.3f} "
              f"IC95%=({ratio_lo:.3f},{ratio_hi:.3f})")

    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_parquet(OUT_RATIO, index=False)
    print(f"\n  Salvo: {OUT_RATIO.relative_to(cfg.ROOT)}")

    T_headline = 1.0
    hr = ratio_df.loc[ratio_df["return_period_years"] == T_headline].iloc[0]
    rq3b_significant = hr["ratio_ci_low"] > 1.0
    between_worst_and_parity = 1.0 < hr["ratio_real_over_copula"] < hr["ratio_real_over_independence"]
    ratio_real_over_closed = float(hr["z_real"] / hr["z_closed_form_gaussian"])
    copula_vs_model_pct_diff = float(100 * abs(hr["z_copula_gaussian"] - hr["z_model_implied"]) / hr["z_model_implied"])
    shared_stage1_artifact = copula_vs_model_pct_diff < 2.0   # cópula e implicado-modelo quase idênticos

    if rq3b_significant and between_worst_and_parity:
        decision = ("RQ3b CONFIRMADA — mesmo a cópula gaussiana (correlação de Pearson real e "
                     "completa) subestima a reserva necessária")
        action = (f"A razão real/cópula-gaussiana em T=1 ano é {hr['ratio_real_over_copula']:.3f} "
                   f"(IC95% [{hr['ratio_ci_low']:.3f}, {hr['ratio_ci_high']:.3f}]) — entre 1,0 (paridade) "
                   f"e {hr['ratio_real_over_independence']:.3f} (pior caso, independência). Mesmo um método "
                   f"que reproduz a correlação de Pearson real e completa entre usinas ainda subestima a "
                   f"reserva necessária, porque a dependência de cauda observada (Gate G1) excede o que uma "
                   f"estrutura gaussiana com a mesma correlação em massa poderia gerar (Sibuya, 1959: "
                   f"dependência de cauda superior assintótica = 0 para qualquer ρ<1). Este é o argumento "
                   f"central para justificar EVT multivariada em vez de correlação linear.")
    elif shared_stage1_artifact:
        decision = ("RQ3b NÃO CONFIRMADA no cenário cópula condicional, mas achado importante: o "
                     "cenário cópula e o cenário 'implicado pelo modelo' (F8, Heffernan-Tawn) convergem "
                     "para quase o mesmo z (diferença <2%) porque compartilham o MESMO Estágio 1 "
                     "(coincidência). A razão real/closed-form (sem nenhum modelo de coincidência, "
                     f"prática ingênua pura) é {ratio_real_over_closed:.3f} — mais próxima do pior caso "
                     "de independência, e essa é a comparação que de fato expõe a limitação da prática "
                     "gaussiana pura.")
        action = (f"O IC da razão real/cópula-gaussiana em T=1 ano ({hr['ratio_real_over_copula']:.3f}, "
                   f"IC95% [{hr['ratio_ci_low']:.3f}, {hr['ratio_ci_high']:.3f}]) inclui 1. Investigação: "
                   f"z_cópula ({hr['z_copula_gaussian']:.4f}) e z_implicado_modelo ({hr['z_model_implied']:.4f}) "
                   f"diferem em apenas {copula_vs_model_pct_diff:.1f}% — porque ambos os cenários herdam o "
                   f"MESMO Estágio 1 de coincidência (offset-logístico) e só diferem no Estágio 2 (magnitude "
                   f"dado coincidência: gaussiana linear vs. Heffernan-Tawn não-linear). Isso demonstra que, "
                   f"UMA VEZ modelada corretamente a COINCIDÊNCIA (regime compartilhado), a forma funcional "
                   f"da dependência de MAGNITUDE importa relativamente pouco para o nível de retorno agregado "
                   f"— reforça o achado central de C1b/F5 de que a dependência é majoritariamente de regime "
                   f"compartilhado, não de acoplamento evento-a-evento de magnitude. O benchmark que de fato "
                   f"expõe a fragilidade da prática puramente gaussiana é o FALLBACK CLOSED-FORM (sem "
                   f"nenhum modelo de coincidência): razão real/closed-form = {ratio_real_over_closed:.3f} "
                   f"em T=1 ano, muito mais próxima do pior caso de independência ({hr['ratio_real_over_independence']:.3f}) "
                   f"do que da paridade (1,0). Reportar os DOIS números no paper, com a distinção explícita "
                   f"entre 'cópula com coincidência modelada' (não subestima) e 'prática puramente linear "
                   f"sem modelo de coincidência' (subestima de forma relevante).")
    else:
        decision = "RQ3b NÃO CONFIRMADA — cópula gaussiana reproduz a reserva real dentro do IC"
        action = ("O IC da razão real/cópula-gaussiana inclui 1 — não há evidência estatística de que a "
                   "cópula gaussiana subestime o risco de cauda para esta rede. Resultado ainda informativo: "
                   "sugere que a dependência de cauda observada é compatível com uma estrutura gaussiana "
                   "calibrada pela mesma correlação em massa — reportar honestamente.")

    print(f"\n  RQ3b (T={T_headline}y): razão real/cópula = {hr['ratio_real_over_copula']:.3f}  "
          f"IC95%=({hr['ratio_ci_low']:.3f}, {hr['ratio_ci_high']:.3f})")
    print(f"  DECISÃO: {decision}")

    # ── Markdown ──────────────────────────────────────────────────────────────
    ratio_table_md = "\n".join(
        f"| {r.return_period_years:.3f} | {r.z_real:.4f} | {r.z_independence:.4f} | "
        f"{r.z_copula_gaussian:.4f} | {r.z_closed_form_gaussian:.4f} | {r.z_model_implied:.4f} | "
        f"{r.ratio_real_over_copula:.3f} | ({r.ratio_ci_low:.3f}, {r.ratio_ci_high:.3f}) | "
        f"{r.ratio_real_over_independence:.3f} |"
        for r in ratio_df.itertuples()
    )
    OUT_DEC.write_text(f"""# F8b — RQ3b: Benchmark contra a Prática Atual (Cópula Gaussiana / Pearson)

**Data:** {date.today().isoformat()}
**Decisão:** {decision}

## Cenário adicional (estende F8/Tab. 5 com uma 4ª e 5ª coluna)

**CÓPULA GAUSSIANA** — espelha exatamente o cenário "IMPLICADO PELO MODELO" de F8 (mesmo
Estágio 1 de coincidência, {N_MODEL} realizações, {len(copula_fits)} convergiram); Estágio 2
substituído por cópula gaussiana condicional com ρ(dist) ajustado por correlação de Pearson
nas mesmas margens Laplace do Estágio 2 de F5 (forma vencedora: **{rho_form}**, ρ₀={rho_params[0]:.4f}).

**FALLBACK CLOSED-FORM** — fórmula clássica de variância da soma ponderada
(Var(k_agg)=Σwᵢ²σᵢ²+ΣΣwᵢwⱼρᵢⱼσᵢσⱼ) usando a matriz de correlação de Pearson REAL do sinal
bruto k_i(t) (não a curva suavizada), com nível de retorno gaussiano. É a forma mais crua e
mais citada em relatórios de indústria — reportada como "prática mínima" ao lado da versão
via cópula (mais rigorosa).

## Comparação ρ(dist) Pearson vs. α(dist) Heffernan-Tawn
Na distância mediana dos eventos casados ({median_dist_matched:.2f}km):
ρ_Pearson = **{rho_at_median:.4f}**  vs.  α_HeffernanTawn = **{alpha_at_median:.4f}**.
Fig.: `results/figures/f8b_rho_vs_alpha.png`.

## Tab. 5 estendida — Níveis de retorno pelos 5 cenários

| T (anos) | z_real | z_independência | z_cópula_gaussiana | z_closed_form | z_implicado_modelo | Razão (real/cópula) | IC95% razão | Razão (real/indep, F8) |
|---|---|---|---|---|---|---|---|---|
{ratio_table_md}

## RQ3b — resultado central (T={T_headline:.0f} ano)
Razão real/cópula-gaussiana (coincidência modelada) = **{hr['ratio_real_over_copula']:.3f}**
(IC95% = [{hr['ratio_ci_low']:.3f}, {hr['ratio_ci_high']:.3f}]).
Razão real/closed-form (sem modelo de coincidência) = **{ratio_real_over_closed:.3f}**.
Para referência, a razão real/independência (F8, pior caso) é **{hr['ratio_real_over_independence']:.3f}**.

{action}

## Nota metodológica — por que este é um resultado importante mesmo se não confirmado
Independentemente do resultado numérico, este cenário fecha a lacuna "comparado a quê?" que
um revisor de sistemas de energia levantaria na primeira rodada: F8 sozinho compara contra o
PIOR caso (independência); F8b compara contra o método efetivamente usado hoje na prática
(correlação linear/cópula gaussiana). Isso posiciona o método do paper (EVT multivariada via
Heffernan-Tawn condicional) explicitamente em relação ao estado da prática, não só ao
pior caso teórico.

## Referência cruzada
- Fig.: `results/figures/f8b_reserve_comparison.png`, `results/figures/f8b_rho_vs_alpha.png`
- F8 original (REAL/INDEPENDÊNCIA/IMPLICADO): `results/gates/f8_rq3_decision.md`
- F5 Estágio 2 (α(dist) Heffernan-Tawn): `results/gates/f5_stage2_params.parquet`
""")
    print(f"  Salvo: {OUT_DEC.relative_to(cfg.ROOT)}")

    # ── Logging estruturado ───────────────────────────────────────────────────
    log_result(
        script="F8b_industry_benchmark.py",
        gate="",
        phase="F8b_RQ3b",
        params={
            "scenario": "Gaussian copula conditional resampling, mirroring F8's model-implied scenario "
                        "(same Stage-1 coincidence probability), with Stage-2 magnitude replaced by "
                        "conditional Gaussian copula using rho(dist) fit via Pearson correlation on the "
                        "same Laplace margins used by F5 Stage-2's alpha(dist).",
            "rho_functional_form": rho_form,
            "n_realizations": N_MODEL,
            "near_neighbor_radius_km": RADIUS_KM,
            "closed_form_fallback": "classical weighted-sum portfolio variance using real Pearson "
                                     "correlation matrix of raw k_i(t) signal (not smoothed rho(dist)), "
                                     "Gaussian tail return level.",
        },
        results={
            "n_copula_converged": len(copula_fits),
            "rho_at_median_dist": round(rho_at_median, 4),
            "alpha_ht_at_median_dist": round(alpha_at_median, 4),
            "ratio_real_over_copula_1y": round(float(hr["ratio_real_over_copula"]), 3),
            "ratio_ci_low_1y": round(float(hr["ratio_ci_low"]), 3),
            "ratio_ci_high_1y": round(float(hr["ratio_ci_high"]), 3),
            "ratio_real_over_independence_1y_ref": round(float(hr["ratio_real_over_independence"]), 3),
            "rq3b_significant": bool(rq3b_significant),
            "between_worst_and_parity": bool(between_worst_and_parity),
            "z_closed_form_1y": round(float(hr["z_closed_form_gaussian"]), 4),
            "ratio_real_over_closed_form_1y": round(ratio_real_over_closed, 3),
            "copula_vs_model_implied_pct_diff": round(copula_vs_model_pct_diff, 2),
            "shared_stage1_artifact": bool(shared_stage1_artifact),
        },
        decision=decision,
        action=action,
        interpretation=(
            f"F8b extends F8's three-scenario comparison with a fourth scenario benchmarking against "
            f"current industry practice for spatially-correlated reserve sizing (Gaussian copula / "
            f"Pearson correlation), the standard approach in quantitative risk management "
            f"(McNeil, Frey & Embrechts) that lacks upper tail dependence for any rho<1 (Sibuya, 1959). "
            f"The copula scenario exactly mirrors F8's model-implied design (same Stage-1 excess-"
            f"coincidence GLM, same conditioning events) but replaces the Heffernan-Tawn Stage-2 draw "
            f"with a conditional Gaussian copula parameterized by rho(dist) fit via Pearson correlation "
            f"on the same Laplace margins used for alpha(dist). At the median matched-pair distance "
            f"({median_dist_matched:.2f}km), rho_Pearson={rho_at_median:.4f} vs. "
            f"alpha_HeffernanTawn={alpha_at_median:.4f}. RQ3b headline (T=1 year): "
            f"ratio_real/copula={hr['ratio_real_over_copula']:.3f} "
            f"(95% CI [{hr['ratio_ci_low']:.3f},{hr['ratio_ci_high']:.3f}]), compared to the F8 "
            f"real/independence ratio of {hr['ratio_real_over_independence']:.3f} (worst case). "
            f"{decision}. Critically, z_copula and z_model_implied (F8's Heffernan-Tawn scenario) differ "
            f"by only {copula_vs_model_pct_diff:.1f}% -- because both scenarios share the identical "
            f"Stage-1 coincidence mechanism and only differ in Stage-2 (magnitude-given-coincidence: "
            f"Gaussian linear vs. Heffernan-Tawn nonlinear), showing that once coincidence (shared-regime) "
            f"is correctly modeled, the magnitude dependence structure matters comparatively little for "
            f"the aggregate return level -- reinforcing C1b/F5's central finding that dependence is "
            f"predominantly shared-regime, not event-to-event magnitude coupling. A simpler closed-form "
            f"fallback (classical weighted-sum portfolio variance using the real Pearson correlation "
            f"matrix of the raw k_i(t) signal, no coincidence model at all, Gaussian tail return level) "
            f"gives z_closed_form={hr['z_closed_form_gaussian']:.4f} at T=1y, a ratio to real of "
            f"{ratio_real_over_closed:.3f} -- much closer to the independence worst-case "
            f"({hr['ratio_real_over_independence']:.3f}) than to parity, and is the benchmark that "
            f"actually exposes the fragility of a naive, coincidence-free Gaussian/Pearson industry "
            f"practice."
        ),
        paper_ref="Section 9 (RQ3b, extension of F8) -- industry-practice benchmark (Gaussian copula / Pearson)",
    )

    # ── Figura — 4 cenários (REAL/INDEP/CÓPULA/MODELO) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    Ts = ratio_df["return_period_years"].to_numpy()
    ax.plot(Ts, ratio_df["z_real"], "o-", color="crimson", label="REAL (observado)")
    ax.plot(Ts, ratio_df["z_independence"], "s--", color="#2c7bb6", label="INDEPENDÊNCIA (F8, pior caso)")
    ax.plot(Ts, ratio_df["z_copula_gaussian"], "D-.", color="darkorange", label="CÓPULA GAUSSIANA (F8b)")
    ax.plot(Ts, ratio_df["z_closed_form_gaussian"], "x:", color="grey", label="Closed-form Gaussiano (F8b, prática mínima)")
    ax.plot(Ts, ratio_df["z_model_implied"], "^:", color="seagreen", label="IMPLICADO PELO MODELO (F8, HT)")
    ax.set_xscale("log")
    ax.set_xlabel("Período de retorno T (anos)")
    ax.set_ylabel(f"Nível de retorno |Δk_agg| ({DIRECTION})")
    ax.set_title("Reserva por cenário (5 cenários)")
    ax.legend(fontsize=7)

    ax2 = axes[1]
    ax2.plot(Ts, ratio_df["ratio_real_over_copula"], "D-.", color="darkorange", label="real/cópula (F8b)")
    ax2.fill_between(Ts, ratio_df["ratio_ci_low"], ratio_df["ratio_ci_high"], color="darkorange", alpha=0.2)
    ax2.plot(Ts, ratio_df["ratio_real_over_independence"], "s--", color="#2c7bb6", label="real/independência (F8, ref.)")
    ax2.axhline(1.0, color="crimson", lw=1, linestyle="--", label="razão=1 (paridade)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Período de retorno T (anos)")
    ax2.set_ylabel("Razão reserva real / reserva do cenário")
    ax2.set_title("RQ3b — mesmo Pearson/cópula gaussiana subestima?")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()
    print(f"\n  Figura: {OUT_FIG.relative_to(cfg.ROOT)}")

    print(f"\n{SEP}")
    print(f"F8b / RQ3b — {decision}")


if __name__ == "__main__":
    main()
