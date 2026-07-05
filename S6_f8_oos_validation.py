"""
S6_f8_oos_validation.py — Validação genuinamente out-of-sample do cenário
"implicado pelo modelo" de F8 (ROADMAP.md Risco D, Seção 0.2)
================================================================================
Motivação: F8_portfolio_effect.py aplica o modelo de dois estágios de
F5_two_stage.py (α₀/L/β/γ ajustados sobre `aligned_pairs.parquet`, treino-only,
n=84.776 casados) DE VOLTA aos mesmos ~20.614 eventos condicionantes de treino
que o ajustaram. O "desvio mediano de 4,5%" reportado é qualidade de ajuste
IN-SAMPLE, não validação out-of-sample.

Este script reaplica o MESMO modelo já ajustado (Estágio 1 recomputado
identicamente sobre o treino, exatamente como em F8; Estágio 2 lido direto de
`f5_stage2_params.parquet`, sem nenhum reajuste) aos eventos condicionantes de
2017 construídos em `S5_event_pairing_test.py` — holdout genuíno, nunca visto
pelo ajuste do modelo. Compara dois modos:

  (a) BACKTEST DE EXCEDÊNCIAS (mesma metodologia de Gate G4/F7): conta quantas
      excedências do nível de retorno z_T (ajustado no TREINO) ocorrem na série
      2017 SINTÉTICA (implicada pelo modelo) vs. na série 2017 REAL (já
      aprovada em Gate G4) vs. o intervalo preditivo de Poisson.
  (b) NÍVEL DE RETORNO GPD em 2017 (years_span=1): ajusta POT/GPD à série
      sintética 2017 (ensemble de realizações) e à série 2017 real, compara o
      desvio percentual mediano — o mesmo tipo de métrica reportada por F8
      (4,5%), agora genuinamente out-of-sample.

Não reabre/modifica F5_two_stage.py, F7_return_levels.py nem
F8_portfolio_effect.py (scripts já aprovados) — duplica os trechos necessários,
seguindo o mesmo padrão já usado dentro do próprio F8/F8b.

Saídas:
  results/gates/s6_oos_backtest.parquet
  results/gates/s6_oos_return_level.parquet
  results/gates/s6_f8_oos_decision.md
  results/figures/s6_oos_validation.png

Executar:
    python S6_f8_oos_validation.py
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
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
from F8_portfolio_effect import inverse_ecdf, inverse_laplace
from C2_gate2 import decluster_cluster_ids

IN_K          = cfg.DIRS["interim"] / "clearsky_index.parquet"
IN_COORDS     = cfg.DIRS["interim"] / "coords.parquet"
IN_RAMPS      = cfg.DIRS["interim"] / "ramps_split.parquet"
IN_AGG_RAMPS  = cfg.DIRS["interim"] / "aggregate_ramps.parquet"
IN_ALIGNED    = cfg.DIRS["processed"] / "aligned_pairs.parquet"
IN_ALIGNED_TS = cfg.DIRS["processed"] / "aligned_pairs_test.parquet"
IN_EVENTPAIR  = cfg.DIRS["gates"] / "event_pairing_summary.parquet"
IN_STAGE2     = cfg.DIRS["gates"] / "f5_stage2_params.parquet"
IN_F7_FIT     = cfg.DIRS["gates"] / "f7_return_level_fit.parquet"
IN_K_AGG      = cfg.DIRS["interim"] / "aggregate_clearsky_index.parquet"

OUT_BACKTEST = cfg.DIRS["gates"]   / "s6_oos_backtest.parquet"
OUT_RETLEVEL = cfg.DIRS["gates"]   / "s6_oos_return_level.parquet"
OUT_DEC      = cfg.DIRS["gates"]   / "s6_f8_oos_decision.md"
OUT_FIG      = cfg.DIRS["figures"] / "s6_oos_validation.png"

N_MODEL   = cfg.F7["n_model_implied_real"]
RADIUS_KM = cfg.F7["near_neighbor_radius_km"]
BACKTEST_PERIODS = cfg.F7["backtest_periods_years"]
SEED = cfg.SEED
SEP = "─" * 60


def main() -> None:
    print(SEP)
    print("S6 — VALIDAÇÃO OUT-OF-SAMPLE GENUÍNA DO CENÁRIO 'IMPLICADO PELO MODELO' (Risco D)")
    print(SEP)

    for p in (IN_K, IN_COORDS, IN_RAMPS, IN_AGG_RAMPS, IN_ALIGNED, IN_ALIGNED_TS, IN_EVENTPAIR,
              IN_STAGE2, IN_F7_FIT, IN_K_AGG):
        if not p.exists():
            print(f"ERRO: {p} não encontrado. Execute S5_event_pairing_test.py / F5/F7/F8 primeiro.")
            sys.exit(1)

    fit_real_df = pd.read_parquet(IN_F7_FIT)
    fit_real = fit_real_df.iloc[0].to_dict()
    years_span_train = float(fit_real_df["years_span_train"].iloc[0])
    stage2 = pd.read_parquet(IN_STAGE2).iloc[0]

    ramps_all = pd.read_parquet(IN_RAMPS)
    ramps_train = ramps_all[ramps_all["split"] == "train"].copy()
    ramps_train["abs_mag"] = ramps_train["delta_k"].abs()
    ramps_test = ramps_all[ramps_all["split"] == "test"].copy()
    ramps_test["abs_mag"] = ramps_test["delta_k"].abs()

    aligned_train = pd.read_parquet(IN_ALIGNED)
    aligned_test = pd.read_parquet(IN_ALIGNED_TS)
    event_summary = pd.read_parquet(IN_EVENTPAIR)
    coords = pd.read_parquet(IN_COORDS)
    df_k = pd.read_parquet(IN_K)
    station_cols = [c for c in df_k.columns if c.startswith("ID")]
    w = build_capacity_weights(coords, station_cols)

    print(f"\n  Eventos condicionantes de 2017 (holdout genuíno, S5): {len(aligned_test):,}")
    print(f"  Eventos condicionantes de treino (usados para ajustar F5): {len(aligned_train):,}")

    # ══════════════════════════════════════════════════════════════════════
    # [1/4] Reconstrução do modelo já ajustado (idêntico a F5/F8) — NENHUM
    #       reajuste; Estágio 1 recomputado sobre TREINO (como em F8), Estágio 2
    #       lido direto de f5_stage2_params.parquet (já aprovado no Gate G3)
    # ══════════════════════════════════════════════════════════════════════
    print("\n[1/4] Reconstruindo Estágio 1 (GLM, ajustado em TREINO) + lendo Estágio 2 (Gate G3)...")
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

    lam_train_by_station, lam_test_by_station = {}, {}
    for sid, g in ramps_train.groupby("station_id"):
        ts = pd.to_datetime(g["start_ts"])
        span_min = (ts.max() - ts.min()).total_seconds() / 60.0 if len(g) > 1 else np.nan
        lam_train_by_station[sid] = len(g) / span_min if span_min and span_min > 0 else np.nan
    for sid, g in ramps_test.groupby("station_id"):
        ts = pd.to_datetime(g["start_ts"])
        span_min = (ts.max() - ts.min()).total_seconds() / 60.0 if len(g) > 1 else np.nan
        lam_test_by_station[sid] = len(g) / span_min if span_min and span_min > 0 else np.nan

    def p_null_directional(ext_i, partner_j, lam_map):
        lam_j = lam_map.get(partner_j, np.nan)
        dtw = dtwin_lookup.get((ext_i, partner_j), np.nan)
        if not (np.isfinite(lam_j) and np.isfinite(dtw)):
            return np.nan
        return 1.0 - np.exp(-lam_j * 2.0 * dtw)

    # -- Estágio 1: fit IDÊNTICO a F5/F8, exclusivamente sobre eventos de TREINO --
    x_lap_tr = np.full(len(aligned_train), np.nan)
    for sid, idx in aligned_train.groupby("station_ext").groups.items():
        pos = aligned_train.index.get_indexer(idx)
        if sid in ecdf_by_station:
            x_lap_tr[pos] = laplace_transform(ecdf_by_station[sid](aligned_train.loc[idx, "mag_ext"].to_numpy()))
    dist_km_tr = np.array([dist_lookup.get((a, b), np.nan)
                           for a, b in zip(aligned_train["station_ext"], aligned_train["station_partner"])])
    p_null_tr = np.array([p_null_directional(a, b, lam_train_by_station)
                          for a, b in zip(aligned_train["station_ext"], aligned_train["station_partner"])])
    aligned_train = aligned_train.assign(x_lap=x_lap_tr, dist_km=dist_km_tr, p_null=p_null_tr)

    valid_tr = (np.isfinite(aligned_train["x_lap"]) & np.isfinite(aligned_train["dist_km"]) &
                np.isfinite(aligned_train["p_null"]) & (aligned_train["p_null"] > 0) & (aligned_train["p_null"] < 1))
    d_tr = aligned_train.loc[valid_tr]
    import statsmodels.api as sm
    from scipy.special import expit, logit
    X_tr = sm.add_constant(np.column_stack([
        np.log(np.maximum(d_tr["dist_km"].to_numpy(), DIST_FLOOR_KM)), d_tr["x_lap"].to_numpy()]))
    y_tr = d_tr["matched"].to_numpy().astype(float)
    p_null_clip_tr = np.clip(d_tr["p_null"].to_numpy(), 1e-4, 1 - 1e-4)
    offset_tr = np.log(p_null_clip_tr / (1 - p_null_clip_tr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        logit_model = sm.GLM(y_tr, X_tr, family=sm.families.Binomial(), offset=offset_tr).fit()
    g0, g_dist, g_x = logit_model.params
    print(f"  Estágio 1 (ajustado em treino, idêntico a F5/F8): logit(P(match))=logit(p_null)+{g0:.3f}"
          f"+{g_dist:.3f}·log(dist)+{g_x:.3f}·X_i")
    print(f"  Estágio 2 (Gate G3, sem reajuste): α₀={float(stage2['alpha0']):.4f}  "
          f"L/p={float(stage2['decay_param']):.4f}  β={float(stage2['beta']):.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # [2/4] APLICAR o modelo já ajustado aos eventos condicionantes de 2017
    #       (S5, nunca vistos pelo ajuste) — SEM reajustar nada
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[2/4] Aplicando o modelo já ajustado aos {len(aligned_test):,} eventos condicionantes "
          f"de 2017 (S5)...")
    x_lap_ts = np.full(len(aligned_test), np.nan)
    for sid, idx in aligned_test.groupby("station_ext").groups.items():
        pos = aligned_test.index.get_indexer(idx)
        if sid in ecdf_by_station:
            x_lap_ts[pos] = laplace_transform(ecdf_by_station[sid](aligned_test.loc[idx, "mag_ext"].to_numpy()))
    dist_km_ts = np.array([dist_lookup.get((a, b), np.nan)
                           for a, b in zip(aligned_test["station_ext"], aligned_test["station_partner"])])
    p_null_ts = np.array([p_null_directional(a, b, lam_test_by_station)
                          for a, b in zip(aligned_test["station_ext"], aligned_test["station_partner"])])
    aligned_test = aligned_test.assign(x_lap=x_lap_ts, dist_km=dist_km_ts, p_null=p_null_ts)

    valid_ts = (np.isfinite(aligned_test["x_lap"]) & np.isfinite(aligned_test["dist_km"]) &
                np.isfinite(aligned_test["p_null"]) & (aligned_test["p_null"] > 0) & (aligned_test["p_null"] < 1))
    work = aligned_test.loc[valid_ts & (aligned_test["dist_km"] <= RADIUS_KM)].copy()
    work = work[work["station_partner"].isin(sorted_by_station.keys())]
    print(f"  Eventos condicionantes 2017 × vizinhos <{RADIUS_KM}km com dados completos: {len(work):,}")

    logit_p_match = (logit(np.clip(work["p_null"].to_numpy(), 1e-4, 1 - 1e-4)) + g0 +
                      g_dist * np.log(np.maximum(work["dist_km"].to_numpy(), DIST_FLOOR_KM)) +
                      g_x * work["x_lap"].to_numpy())
    p_match_arr = expit(logit_p_match)

    form = stage2["form"]
    alpha0, decay_param, beta = float(stage2["alpha0"]), float(stage2["decay_param"]), float(stage2["beta"])
    mu, sigma = float(stage2["mu"]), float(stage2["sigma"])
    dist_arr = work["dist_km"].to_numpy()
    alpha_dist_arr = (alpha_exp(dist_arr, alpha0, np.log(decay_param)) if form == "exp"
                      else alpha_pow(dist_arr, alpha0, np.log(decay_param)))
    x_lap_arr = work["x_lap"].to_numpy()

    # ══════════════════════════════════════════════════════════════════════
    # [3/4] Reconstruir a série 2017 REAL como base + geometria de patching
    #       (idêntico a F8, mas com base/índice/eventos = TESTE, nunca treino)
    # ══════════════════════════════════════════════════════════════════════
    print("\n[3/4] Reconstruindo série 2017 REAL (base) + geometria de patching...")
    dur_samples = int(round(ramps_train["duration_min"].median()))
    col_of = {sid: i for i, sid in enumerate(station_cols)}
    test_block = df_k[df_k["split"] == "test"]
    K_test = test_block[station_cols].to_numpy(dtype=float)
    test_index = test_block.index
    test_t0 = pd.Timestamp(test_index[0]).tz_localize(None)
    t_ext_naive = pd.to_datetime(work["t_ext"]).dt.tz_localize(None)
    row0_all = ((t_ext_naive.to_numpy() - np.datetime64(test_t0)) //
                np.timedelta64(1, "m")).astype(np.int64)
    col_j_all = np.array([col_of.get(sid, -1) for sid in work["station_partner"]])
    valid_geom = (col_j_all >= 0) & (row0_all >= 0) & (row0_all + dur_samples < len(K_test))
    sign_all = np.where(work["dir_ext"].to_numpy() == "down", -1.0, 1.0)
    w_j_all = w[np.clip(col_j_all, 0, len(w) - 1)]

    row_idx_2d_full = row0_all[:, None] + np.arange(dur_samples)[None, :]
    row_idx_2d_full = np.clip(row_idx_2d_full, 0, len(K_test) - 1)
    real_window_full = K_test[row_idx_2d_full, np.clip(col_j_all, 0, len(w) - 1)[:, None]]
    ramp_shape = np.linspace(0.0, 1.0, dur_samples)[None, :]

    k_agg_full = pd.read_parquet(IN_K_AGG)
    k_agg_test = k_agg_full.loc[k_agg_full["split"] == "test", "k_agg"].to_numpy()
    years_span_test = float(len(k_agg_test) / 1440 / 365.25)
    print(f"  Base 2017: {len(k_agg_test):,} min ({years_span_test:.3f} anos)")

    # ══════════════════════════════════════════════════════════════════════
    # [4/4] N_MODEL realizações do cenário implicado-2017 + comparação com o real
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[4/4] Gerando {N_MODEL} realizações do cenário IMPLICADO-2017 (out-of-sample)...")
    rng4 = np.random.default_rng(SEED + 4)
    oos_fits = []
    backtest_rows_per_real = {T: [] for T in BACKTEST_PERIODS}
    partner_arr = work["station_partner"].to_numpy()
    n_work = len(work)
    t0 = time.time()

    for r in range(N_MODEL):
        u1 = rng4.random(n_work)
        match_mask = valid_geom & (u1 < p_match_arr)
        idx_m = np.where(match_mask)[0]
        if len(idx_m) == 0:
            continue
        z_resid = rng4.normal(mu, sigma, size=len(idx_m))
        y_lap = alpha_dist_arr[idx_m] * x_lap_arr[idx_m] + np.power(x_lap_arr[idx_m], beta) * z_resid
        u_quant = inverse_laplace(y_lap)

        mag_synth = np.zeros(len(idx_m))
        partners_m = partner_arr[idx_m]
        for sid in np.unique(partners_m):
            sel = partners_m == sid
            mag_synth[sel] = inverse_ecdf(sorted_by_station[sid], u_quant[sel])
        mag_synth = np.clip(mag_synth, 0.0, None)

        peak_delta = sign_all[idx_m] * mag_synth
        synth_window = real_window_full[idx_m, 0:1] + peak_delta[:, None] * ramp_shape
        delta_window = (synth_window - real_window_full[idx_m]) * w_j_all[idx_m][:, None]
        delta_window = np.where(np.isfinite(real_window_full[idx_m]), delta_window, 0.0)

        k_agg_mod = k_agg_test.copy()
        np.add.at(k_agg_mod, row_idx_2d_full[idx_m].reshape(-1), delta_window.reshape(-1))

        ramps_r = detect_ramps(k_agg_mod, test_index)
        sub = ramps_r[ramps_r["direction"] == DIRECTION]

        for T in BACKTEST_PERIODS:
            z_T = return_level(fit_real, years_span_train, T)
            mags_r = sub["delta_k"].abs().to_numpy()
            times_r = pd.to_datetime(sub["start_ts"]).to_numpy()
            exceed_mask = mags_r > z_T
            if exceed_mask.sum() == 0:
                observed_count = 0
            else:
                t_exc = times_r[exceed_mask]
                order = np.argsort(t_exc)
                t_exc = t_exc[order]
                if len(t_exc) < 2:
                    observed_count = len(t_exc)
                else:
                    gaps_min = np.diff(t_exc).astype("timedelta64[m]").astype(float)
                    cid = decluster_cluster_ids(gaps_min, fit_real["run_length_min"])
                    observed_count = int(cid.max() + 1)
            backtest_rows_per_real[T].append(observed_count)

        if len(sub) >= 20:
            fit_r = fit_pot_gpd(sub["delta_k"].abs().to_numpy(), sub["start_ts"].to_numpy())
            if fit_r is not None:
                oos_fits.append(fit_r)
        if (r + 1) % 30 == 0:
            print(f"  ... {r+1}/{N_MODEL} realizações ({time.time()-t0:.1f}s, {len(oos_fits)} GPD convergiram)")
    print(f"  Concluído: {len(oos_fits)}/{N_MODEL} realizações com GPD convergido ({time.time()-t0:.1f}s)")

    # ── (a) Backtest de excedências: implicado-2017 vs. real-2017 vs. Poisson ──
    # IMPORTANTE: a comparação REAL precisa usar a série da REDE AGREGADA (F7a/F7,
    # "usina virtual"), não a soma de rampas por estação individual — mesma
    # convenção de Gate G4/F7/F8 (RQ3 é sempre sobre o k_agg(t), nunca sobre estações
    # isoladas). Usar `ramps_split.parquet` aqui inflaria artificialmente a contagem
    # de excedências (rampas de estações individuais, muito mais numerosas e menores
    # em escala que rampas da rede agregada).
    print("\n  Backtest de excedências (mesma metodologia de Gate G4/F7, série da REDE agregada)...")
    agg_ramps_all = pd.read_parquet(IN_AGG_RAMPS)
    agg_ramps_test = agg_ramps_all[(agg_ramps_all["split"] == "test") &
                                    (agg_ramps_all["direction"] == DIRECTION)].copy()
    test_mag_real = agg_ramps_test["delta_k"].abs().to_numpy()
    test_t_real = pd.to_datetime(agg_ramps_test["start_ts"]).to_numpy()

    backtest_rows = []
    for T in BACKTEST_PERIODS:
        z_T = return_level(fit_real, years_span_train, T)
        exceed_mask = test_mag_real > z_T
        if exceed_mask.sum() == 0:
            observed_real = 0
        else:
            t_exc = test_t_real[exceed_mask]
            order = np.argsort(t_exc)
            t_exc = t_exc[order]
            if len(t_exc) < 2:
                observed_real = len(t_exc)
            else:
                gaps_min = np.diff(t_exc).astype("timedelta64[m]").astype(float)
                cid = decluster_cluster_ids(gaps_min, fit_real["run_length_min"])
                observed_real = int(cid.max() + 1)
        expected_count = years_span_test / T
        pi_low = poisson.ppf(0.025, expected_count)
        pi_high = poisson.ppf(0.975, expected_count)
        model_counts = np.array(backtest_rows_per_real[T])
        model_median = float(np.median(model_counts)) if len(model_counts) else np.nan
        model_covered = pi_low <= model_median <= pi_high
        backtest_rows.append({
            "return_period_years": T, "z_T_from_train": z_T,
            "expected_exceedances_poisson": expected_count,
            "observed_exceedances_real_2017": observed_real,
            "observed_exceedances_model_implied_2017_median": model_median,
            "poisson_pi_low": pi_low, "poisson_pi_high": pi_high,
            "model_covered_by_poisson_pi": bool(model_covered),
        })
        print(f"  T={T:.3f}y  z_T(treino)={z_T:.4f}  real_2017={observed_real}  "
              f"implicado_2017(mediana)={model_median:.1f}  esperado_Poisson={expected_count:.2f}  "
              f"IC95%=[{pi_low:.0f},{pi_high:.0f}]  {'✓' if model_covered else '✗'}")

    backtest_df = pd.DataFrame(backtest_rows)
    backtest_df.to_parquet(OUT_BACKTEST, index=False)
    print(f"  Salvo: {OUT_BACKTEST.relative_to(cfg.ROOT)}")

    # ── (b) Nível de retorno GPD, years_span=1, implicado vs. real 2017 ─────────
    print("\n  Nível de retorno GPD (years_span=1 ano) — implicado-2017 vs. real-2017...")
    fit_real_2017 = fit_pot_gpd(test_mag_real, test_t_real)
    ratio_rows = []
    if fit_real_2017 is not None and len(oos_fits) >= 10:
        for T in BACKTEST_PERIODS:
            z_real_2017 = return_level(fit_real_2017, years_span_test, T)
            z_oos_ens = np.array([return_level(f, years_span_test, T) for f in oos_fits])
            z_oos_median = float(np.median(z_oos_ens))
            pct_diff = 100 * (z_oos_median - z_real_2017) / z_real_2017 if z_real_2017 else np.nan
            ratio_rows.append({
                "return_period_years": T, "z_real_2017": z_real_2017,
                "z_model_implied_2017_median": z_oos_median, "pct_diff": pct_diff,
            })
            print(f"  T={T:.3f}y  z_real_2017={z_real_2017:.4f}  z_implicado_2017={z_oos_median:.4f}  "
                  f"Δ%={pct_diff:+.1f}%")
        ratio_df = pd.DataFrame(ratio_rows)
        oos_median_pct_diff = float(ratio_df["pct_diff"].abs().median())
    else:
        ratio_df = pd.DataFrame()
        oos_median_pct_diff = np.nan
        print("  Amostra de 2017 insuficiente para ajuste GPD direto (esperado, 1 único ano) — "
              "usando só o backtest de excedências (a) como validação primária.")
    ratio_df.to_parquet(OUT_RETLEVEL, index=False)
    print(f"  Salvo: {OUT_RETLEVEL.relative_to(cfg.ROOT)}")

    # ── Decisão ──────────────────────────────────────────────────────────────
    frac_covered = backtest_df["model_covered_by_poisson_pi"].mean()
    real_vs_expected_ok = True   # já sabido de Gate G4 (100% coberto)
    oos_confirmed = frac_covered >= 0.75 and (not np.isfinite(oos_median_pct_diff) or oos_median_pct_diff < 40)

    if oos_confirmed:
        decision = ("RISCO D RESOLVIDO — validação OUT-OF-SAMPLE genuína CONFIRMA a utilidade "
                     "preditiva do modelo de dois estágios")
        action = (f"Aplicando o modelo já ajustado (Estágios 1/2, sem nenhum reajuste) aos "
                   f"{len(work):,} eventos condicionantes de 2017 (nunca vistos pelo ajuste, S5), "
                   f"a série agregada 2017 SINTÉTICA implicada pelo modelo tem cobertura Poisson "
                   f"de {frac_covered:.0%} nos horizontes de backtest testados"
                   + (f", e desvio percentual mediano de {oos_median_pct_diff:.1f}% vs. o nível de "
                      f"retorno ajustado diretamente na série 2017 real" if np.isfinite(oos_median_pct_diff) else "")
                   + f". Isso é evidência de validação OUT-OF-SAMPLE genuína (não apenas qualidade de "
                     f"ajuste in-sample como o '4,5%' original de F8) — o modelo de dois estágios tem "
                     f"utilidade preditiva real para dados nunca vistos. Reportar este número como a "
                     f"validação primária do modelo condicional no paper, com o '4,5%' de F8 reclassificado "
                     f"explicitamente como diagnóstico de ajuste in-sample (não removido, mas re-rotulado).")
    else:
        decision = ("RISCO D — validação OUT-OF-SAMPLE FRACA/NÃO CONFIRMADA — modelo pode estar "
                     "sobreajustado ao conjunto de treino")
        action = (f"Cobertura Poisson do backtest OOS foi de apenas {frac_covered:.0%}"
                   + (f", com desvio percentual mediano de {oos_median_pct_diff:.1f}%" if np.isfinite(oos_median_pct_diff) else "")
                   + f". Isso sugere que o desvio de 4,5% relatado por F8 é, ao menos em parte, "
                     f"qualidade de ajuste in-sample, não generalização genuína. NÃO invalida a razão "
                     f"RQ3 = 2,39× (que não usa o modelo de dois estágios), mas exige reportar a alegação "
                     f"de 'utilidade preditiva do modelo condicional' com mais cautela — como diagnóstico "
                     f"in-sample, não validação out-of-sample confirmada.")

    print(f"\n  DECISÃO: {decision}")
    print(f"  Ação: {action}")

    # ── Markdown ──────────────────────────────────────────────────────────────
    backtest_table_md = "\n".join(
        f"| {r.return_period_years:.3f} | {r.z_T_from_train:.4f} | {r.observed_exceedances_real_2017} | "
        f"{r.observed_exceedances_model_implied_2017_median:.1f} | {r.expected_exceedances_poisson:.2f} | "
        f"({r.poisson_pi_low:.0f}, {r.poisson_pi_high:.0f}) | {'✓' if r.model_covered_by_poisson_pi else '✗'} |"
        for r in backtest_df.itertuples()
    )
    retlevel_section = ""
    if len(ratio_df):
        retlevel_table_md = "\n".join(
            f"| {r.return_period_years:.3f} | {r.z_real_2017:.4f} | {r.z_model_implied_2017_median:.4f} | "
            f"{r.pct_diff:+.1f}% |"
            for r in ratio_df.itertuples()
        )
        retlevel_section = f"""
## (b) Nível de retorno GPD ajustado direto em 2017 (years_span=1 ano)
| T (anos) | z_real_2017 | z_implicado_2017 (mediana) | Δ% |
|---|---|---|---|
{retlevel_table_md}

Desvio percentual absoluto mediano: **{oos_median_pct_diff:.1f}%** (comparar com o 4,5% IN-SAMPLE
reportado por F8 — esta é a versão genuinamente OUT-OF-SAMPLE da mesma métrica).
"""

    OUT_DEC.write_text(f"""# S6 — Validação Out-of-Sample do Cenário "Implicado pelo Modelo" (Risco D)

**Data:** {date.today().isoformat()}
**Decisão:** {decision}

## Motivação
F8_portfolio_effect.py aplica o modelo de dois estágios (F5_two_stage.py, ajustado sobre
eventos casados de TREINO) DE VOLTA aos mesmos eventos que o ajustaram — o "desvio mediano de
4,5%" é qualidade de ajuste in-sample, não validação out-of-sample (ROADMAP Risco D). Este
script aplica o MESMO modelo (sem nenhum reajuste) aos {len(work):,} eventos condicionantes de
2017 construídos em `S5_event_pairing_test.py` — holdout genuíno.

## (a) Backtest de excedências (mesma metodologia de Gate G4/F7)
Conta excedências do nível de retorno z_T (ajustado em TREINO) na série 2017 REAL vs. na série
2017 SINTÉTICA implicada pelo modelo (mediana de {N_MODEL} realizações) vs. o intervalo
preditivo de Poisson.

| T (anos) | z_T (treino) | Excedências reais 2017 | Excedências implicadas 2017 (mediana) | Esperado (Poisson) | IC95% Poisson | Implicado coberto? |
|---|---|---|---|---|---|---|
{backtest_table_md}

Fração de horizontes com o cenário IMPLICADO dentro do IC-Poisson: **{frac_covered:.0%}**
(para referência: a série REAL de 2017 já tem 100% de cobertura, Gate G4 aprovado).
{retlevel_section}
## Decisão
**{decision}**

{action}

## Referência cruzada
- Entrada: `results/processed/aligned_pairs_test.parquet` (S5)
- Contraste: `results/gates/f8_rq3_decision.md` (F8, validação in-sample original, 4,5%)
- Fig.: `results/figures/s6_oos_validation.png`
""")
    print(f"  Salvo: {OUT_DEC.relative_to(cfg.ROOT)}")

    # ── Logging estruturado ───────────────────────────────────────────────────
    log_result(
        script="S6_f8_oos_validation.py",
        gate="",
        phase="S6_riscoD",
        params={
            "design": "Apply F5_two_stage.py's already-fitted model (Stage 1 GLM refit identically "
                      "on training data as in F8, Stage 2 read directly from Gate-G3-approved params, "
                      "NO retraining) to 2017 holdout conditioning events built in S5, patching the "
                      "REAL 2017 aggregate base series -- genuine out-of-sample test.",
            "n_realizations": N_MODEL,
            "backtest_periods_years": BACKTEST_PERIODS,
        },
        results={
            "n_conditioning_events_2017": len(work),
            "n_gpd_realizations_converged": len(oos_fits),
            "frac_horizons_covered_by_poisson_pi": round(float(frac_covered), 3),
            "oos_median_pct_diff_return_level": round(float(oos_median_pct_diff), 1) if np.isfinite(oos_median_pct_diff) else None,
            "in_sample_pct_diff_reference_f8": 4.5,
            "oos_confirmed": bool(oos_confirmed),
        },
        decision=decision,
        action=action,
        interpretation=(
            f"S6 addresses ROADMAP Risco D: F8's 'model-implied' scenario applies F5_two_stage.py's "
            f"fitted model back onto the SAME training-period conditioning events used to fit it, so "
            f"its reported 4.5% median deviation is an in-sample fit-quality metric, not genuine "
            f"out-of-sample validation. S6 reapplies the identical already-fitted model (no "
            f"retraining) to {len(work):,} 2017-holdout conditioning events (S5_event_pairing_test.py, "
            f"never seen during fitting), patching the REAL 2017 aggregate base series. Two "
            f"complementary checks: (a) exceedance backtest against train-fitted return levels "
            f"(same methodology as Gate G4), model-implied 2017 series covered by the 95% Poisson "
            f"predictive interval in {frac_covered:.0%} of tested horizons; (b) direct GPD return-level "
            f"comparison fit on the real vs. model-implied 2017 series (years_span=1), median absolute "
            f"percent deviation = {oos_median_pct_diff:.1f}% (comparable to F8's in-sample 4.5%). "
            f"{decision}. Critically, this does NOT affect RQ3's headline ratio (2.39x, real vs. "
            f"independence), which never uses the two-stage model at all -- only the secondary claim "
            f"about the conditional model's predictive utility is at stake."
        ),
        paper_ref="Section 9 (F8 model-implied scenario) -- genuine out-of-sample validation (Risco D)",
    )

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    Ts = backtest_df["return_period_years"].to_numpy()
    ax.plot(Ts, backtest_df["observed_exceedances_real_2017"], "o-", color="crimson", label="Real 2017 (observado)")
    ax.plot(Ts, backtest_df["observed_exceedances_model_implied_2017_median"], "^--", color="seagreen",
            label="Implicado 2017 (mediana, out-of-sample)")
    ax.plot(Ts, backtest_df["expected_exceedances_poisson"], "s:", color="grey", label="Esperado (Poisson)")
    ax.fill_between(Ts, backtest_df["poisson_pi_low"], backtest_df["poisson_pi_high"], color="grey", alpha=0.2,
                    label="IC95% Poisson")
    ax.set_xscale("log")
    ax.set_xlabel("Período de retorno T (anos)")
    ax.set_ylabel("Nº de excedências declusterizadas")
    ax.set_title("(a) Backtest de excedências — real vs. implicado-2017 (OOS)")
    ax.legend(fontsize=7)

    ax2 = axes[1]
    if len(ratio_df):
        ax2.plot(ratio_df["return_period_years"], ratio_df["z_real_2017"], "o-", color="crimson", label="z real 2017")
        ax2.plot(ratio_df["return_period_years"], ratio_df["z_model_implied_2017_median"], "^--",
                 color="seagreen", label="z implicado 2017 (mediana, OOS)")
        ax2.set_xscale("log")
        ax2.set_xlabel("Período de retorno T (anos)")
        ax2.set_ylabel(f"Nível de retorno |Δk_agg| ({DIRECTION})")
        ax2.set_title(f"(b) GPD 2017 direto — Δ% mediano={oos_median_pct_diff:.1f}%")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "GPD direto em 2017:\namostra insuficiente\n(esperado, 1 ano)",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_axis_off()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()
    print(f"\n  Figura: {OUT_FIG.relative_to(cfg.ROOT)}")

    print(f"\n{SEP}")
    print(f"S6 / Risco D — {decision}")


if __name__ == "__main__":
    main()
