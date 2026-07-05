"""
F8_portfolio_effect.py — RQ3 (resultado central): efeito-portfólio sob independência
vs. dependência real/implicada pelo modelo
========================================================================================
Pergunta (RQ3, ROADMAP Seção 2): o efeito-portfólio geográfico assumido no
dimensionamento de reserva se sustenta no extremo? Compara três cenários para
a rampa agregada da rede (mesma "usina virtual" de F7a, mesma metodologia
POT/GPD de F7/Gate G4):

  1. REAL (observado) — já ajustado em F7_return_levels.py (Gate G4). Ground
     truth: dependência espacial tal como ela realmente é.
  2. INDEPENDÊNCIA (contrafactual empírico) — cada usina recebe um deslocamento
     circular de blocos de dias INDEPENDENTE (preserva exatamente a distribuição
     marginal/GPD de cada usina; destrói a sincronia entre usinas). Não depende
     do modelo de dois estágios — é o "efeito-portfólio" assumido na prática de
     dimensionamento de reserva, operacionalizado sem premissas paramétricas.
  3. IMPLICADO PELO MODELO (validação) — parte da série REAL de treino e, para
     cada evento extremo histórico real (catalogado em `aligned_pairs.parquet`,
     C1b), SUBSTITUI a resposta dos vizinhos próximos (<5km) pela previsão do
     modelo de dois estágios já ajustado e aprovado no Gate G3 (`F5_two_stage.py`:
     Estágio 1 decide coincidência, Estágio 2 sorteia magnitude). Se a cauda
     resultante bater com a do cenário REAL, isso valida que o Heffernan-Tawn
     condicional (o aparato multivariado central do paper) de fato reproduz o
     risco de cauda do portfólio — não é só um ajuste post-hoc.

RQ3 central: razão entre os níveis de retorno REAL e INDEPENDÊNCIA, com IC
(combinação Monte Carlo dos dois ensembles bootstrap — real de F7, independência
gerado aqui).

Saídas:
  results/gates/f8_independence_bootstrap.parquet
  results/gates/f8_model_implied_bootstrap.parquet
  results/gates/f8_rq3_ratio.parquet                — Tab. 5
  results/gates/f8_rq3_decision.md
  results/figures/f8_reserve_comparison.png         — Fig. 9 (central)

Executar:
    python F8_portfolio_effect.py
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
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
from C1_gate1 import make_block_index_arrays
from B5_ramp_detection import _sdt_compress_core

IN_K         = cfg.DIRS["interim"] / "clearsky_index.parquet"
IN_COORDS    = cfg.DIRS["interim"] / "coords.parquet"
IN_RAMPS     = cfg.DIRS["interim"] / "ramps_split.parquet"
IN_ALIGNED   = cfg.DIRS["processed"] / "aligned_pairs.parquet"
IN_EVENTPAIR = cfg.DIRS["gates"] / "event_pairing_summary.parquet"
IN_STAGE2    = cfg.DIRS["gates"] / "f5_stage2_params.parquet"
IN_F7_FIT    = cfg.DIRS["gates"] / "f7_return_level_fit.parquet"
IN_F7_BOOT   = cfg.DIRS["gates"] / "f7_return_level_bootstrap.parquet"

OUT_INDEP_BOOT = cfg.DIRS["gates"] / "f8_independence_bootstrap.parquet"
OUT_MODEL_BOOT = cfg.DIRS["gates"] / "f8_model_implied_bootstrap.parquet"
OUT_RATIO      = cfg.DIRS["gates"] / "f8_rq3_ratio.parquet"
OUT_DEC        = cfg.DIRS["gates"] / "f8_rq3_decision.md"
OUT_FIG9       = cfg.DIRS["figures"] / "f8_reserve_comparison.png"

N_INDEP   = cfg.F7["n_independence_real"]
N_MODEL   = cfg.F7["n_model_implied_real"]
RADIUS_KM = cfg.F7["near_neighbor_radius_km"]
SEED = cfg.SEED
SEP = "─" * 60


def inverse_laplace(y: np.ndarray) -> np.ndarray:
    u = np.where(y >= 0, 1 - np.exp(-y) / 2, np.exp(y) / 2)
    return np.clip(u, 1e-6, 1 - 1e-6)


def inverse_ecdf(sorted_vals: np.ndarray, u_query: np.ndarray) -> np.ndarray:
    """Inversa (função quantil) da ECDF empírica — plotting position de Hazen, consistente
    o bastante com `build_ecdf` para uso GENERATIVO (simulação), não para reestimação."""
    n = len(sorted_vals)
    pp = (np.arange(1, n + 1) - 0.5) / n
    return np.interp(u_query, pp, sorted_vals, left=sorted_vals[0], right=sorted_vals[-1])


def mc_ratio_ci(real_ensemble: np.ndarray, other_ensemble: np.ndarray, n_mc: int = 5000, seed: int = SEED):
    """IC da razão real/outro combinando dois ensembles bootstrap INDEPENDENTES via
    reamostragem Monte Carlo pareada aleatoriamente (nenhum dos dois ensembles tem
    correspondência natural par-a-par — a combinação MC é a forma padrão de propagar
    duas fontes de incerteza independentes para uma razão)."""
    rng = np.random.default_rng(seed)
    a = rng.choice(real_ensemble, size=n_mc, replace=True)
    b = rng.choice(other_ensemble, size=n_mc, replace=True)
    ratio = a / b
    return float(np.median(ratio)), float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))


def main() -> None:
    print(SEP)
    print("F8 — RQ3: EFEITO-PORTFÓLIO (INDEPENDÊNCIA vs. REAL vs. IMPLICADO PELO MODELO)")
    print(SEP)

    for p in (IN_K, IN_COORDS, IN_RAMPS, IN_ALIGNED, IN_EVENTPAIR, IN_STAGE2, IN_F7_FIT, IN_F7_BOOT):
        if not p.exists():
            print(f"ERRO: {p} não encontrado. Execute F7a/F7/F5_two_stage primeiro.")
            sys.exit(1)

    fit_real_df = pd.read_parquet(IN_F7_FIT)
    boot_real_long = pd.read_parquet(IN_F7_BOOT)
    block_len = int(fit_real_df["block_length_days"].iloc[0])
    years_span_train = float(fit_real_df["years_span_train"].iloc[0])
    print(f"\n  Cenário REAL (F7/Gate G4) carregado — block_length={block_len}d, "
          f"years_span={years_span_train:.3f}")

    _sdt_compress_core(np.array([0.0, 0.1, 0.05]), 0.02)   # warm-up JIT

    # ══════════════════════════════════════════════════════════════════════
    # [1/3] CENÁRIO INDEPENDÊNCIA — deslocamento circular de dias, por usina
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[1/3] Cenário INDEPENDÊNCIA — {N_INDEP} realizações "
          "(cada usina com deslocamento de blocos de dias independente)...")
    df_k = pd.read_parquet(IN_K)
    coords = pd.read_parquet(IN_COORDS)
    station_cols = [c for c in df_k.columns if c.startswith("ID")]
    w = build_capacity_weights(coords, station_cols)

    train = df_k[df_k["split"] == "train"]
    K = train[station_cols].to_numpy(dtype=float)
    n_days = len(K) // 1440
    assert len(K) == n_days * 1440
    K3 = K.reshape(n_days, 1440, len(station_cols))
    synth_dt_index = pd.date_range("2000-01-01", periods=n_days * 1440, freq="1min")

    rng = np.random.default_rng(SEED + 1)
    indep_fits = []
    t0 = time.time()
    for r in range(N_INDEP):
        station_day_idx = make_block_index_arrays(n_days, block_len, len(station_cols), rng)  # (n_stations, n_days)
        sum_acc = np.zeros((n_days, 1440))
        weight_acc = np.zeros((n_days, 1440))
        for s in range(len(station_cols)):
            sl = K3[station_day_idx[s], :, s]          # (n_days, 1440), dia embaralhado p/ ESTA usina
            valid = np.isfinite(sl)
            sum_acc += w[s] * np.where(valid, sl, 0.0)
            weight_acc += w[s] * valid
        with np.errstate(invalid="ignore", divide="ignore"):
            k_indep = np.where(weight_acc > 1e-9, sum_acc / weight_acc, np.nan).reshape(-1)
        ramps_r = detect_ramps(k_indep, synth_dt_index)
        sub = ramps_r[ramps_r["direction"] == DIRECTION]
        if len(sub) < 20:
            continue
        fit_r = fit_pot_gpd(sub["delta_k"].abs().to_numpy(), sub["start_ts"].to_numpy())
        if fit_r is not None:
            indep_fits.append(fit_r)
        if (r + 1) % 20 == 0:
            print(f"  ... {r+1}/{N_INDEP} realizações ({time.time()-t0:.1f}s, {len(indep_fits)} convergiram)")
    print(f"  Concluído: {len(indep_fits)}/{N_INDEP} realizações convergiram ({time.time()-t0:.1f}s)")

    indep_boot_long = pd.DataFrame([
        {"realization_id": i, "return_period_years": T, "z": return_level(f, years_span_train, T)}
        for i, f in enumerate(indep_fits) for T in RETURN_PERIODS
    ])
    indep_boot_long.to_parquet(OUT_INDEP_BOOT, index=False)
    print(f"  Salvo: {OUT_INDEP_BOOT.relative_to(cfg.ROOT)}")

    # ══════════════════════════════════════════════════════════════════════
    # [2/3] CENÁRIO IMPLICADO PELO MODELO — F5 Estágio 1/2 sobre eventos reais
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[2/3] Cenário IMPLICADO PELO MODELO — {N_MODEL} realizações "
          f"(perturbando vizinhos <{RADIUS_KM}km de eventos extremos reais via F5 Estágio 1/2)...")

    ramps_all = pd.read_parquet(IN_RAMPS)
    ramps_train = ramps_all[ramps_all["split"] == "train"].copy()
    ramps_train["abs_mag"] = ramps_train["delta_k"].abs()
    aligned = pd.read_parquet(IN_ALIGNED)
    event_summary = pd.read_parquet(IN_EVENTPAIR)
    stage2 = pd.read_parquet(IN_STAGE2).iloc[0]

    ecdf_by_station = {}
    sorted_by_station = {}
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

    # Recomputa o Estágio 1 (GLM com offset) — mesma especificação exata de F5_two_stage.py.
    # Duplicado (não importado) para não reabrir/alterar o script já aprovado no Gate G3.
    import statsmodels.api as sm
    x_lap_all = np.full(len(aligned), np.nan)
    for sid, idx in aligned.groupby("station_ext").groups.items():
        pos = aligned.index.get_indexer(idx)
        if sid in ecdf_by_station:
            x_lap_all[pos] = laplace_transform(ecdf_by_station[sid](aligned.loc[idx, "mag_ext"].to_numpy()))
    dist_km_all = np.array([dist_lookup.get((a, b), np.nan)
                             for a, b in zip(aligned["station_ext"], aligned["station_partner"])])
    p_null_all = np.array([p_null_directional(a, b)
                            for a, b in zip(aligned["station_ext"], aligned["station_partner"])])
    aligned = aligned.assign(x_lap=x_lap_all, dist_km=dist_km_all, p_null=p_null_all)

    valid1 = (np.isfinite(aligned["x_lap"]) & np.isfinite(aligned["dist_km"]) &
              np.isfinite(aligned["p_null"]) & (aligned["p_null"] > 0) & (aligned["p_null"] < 1))
    d1 = aligned.loc[valid1]
    X1 = sm.add_constant(np.column_stack([
        np.log(np.maximum(d1["dist_km"].to_numpy(), DIST_FLOOR_KM)), d1["x_lap"].to_numpy()]))
    y1 = d1["matched"].to_numpy().astype(float)
    p_null_clip = np.clip(d1["p_null"].to_numpy(), 1e-4, 1 - 1e-4)
    offset1 = np.log(p_null_clip / (1 - p_null_clip))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        logit_model = sm.GLM(y1, X1, family=sm.families.Binomial(), offset=offset1).fit()
    g0, g_dist, g_x = logit_model.params
    print(f"  Estágio 1 recomputado (idêntico a F5): logit(P(match))=logit(p_null)+{g0:.3f}"
          f"+{g_dist:.3f}·log(dist)+{g_x:.3f}·X_i")

    # Restringir a vizinhos dentro do raio validado e com todas as covariáveis finitas
    work = aligned.loc[valid1 & (aligned["dist_km"] <= RADIUS_KM)].copy()
    work = work[work["station_partner"].isin(sorted_by_station.keys())]
    print(f"  Eventos condicionantes × vizinhos <{RADIUS_KM}km com dados completos: {len(work):,} "
          f"({work.drop_duplicates(['station_ext','t_ext']).shape[0]:,} eventos únicos)")

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

    # Índices/baseline REPLICATE-INVARIANTES (dados reais — pré-computados uma única vez)
    dur_samples = int(round(ramps_train["duration_min"].median()))
    col_of = {sid: i for i, sid in enumerate(station_cols)}
    train_t0 = pd.Timestamp(train.index[0]).tz_localize(None)
    t_ext_naive = pd.to_datetime(work["t_ext"]).dt.tz_localize(None)
    row0_all = ((t_ext_naive.to_numpy() - np.datetime64(train_t0)) //
                np.timedelta64(1, "m")).astype(np.int64)
    col_j_all = np.array([col_of.get(sid, -1) for sid in work["station_partner"]])
    valid_geom = (col_j_all >= 0) & (row0_all >= 0) & (row0_all + dur_samples < len(K))
    sign_all = np.where(work["dir_ext"].to_numpy() == "down", -1.0, 1.0)
    w_j_all = w[np.clip(col_j_all, 0, len(w) - 1)]

    row_idx_2d_full = row0_all[:, None] + np.arange(dur_samples)[None, :]
    row_idx_2d_full = np.clip(row_idx_2d_full, 0, len(K) - 1)
    real_window_full = K[row_idx_2d_full, np.clip(col_j_all, 0, len(w) - 1)[:, None]]   # (n_work, dur)
    ramp_shape = np.linspace(0.0, 1.0, dur_samples)[None, :]                            # (1, dur)

    # k_agg REAL de treino já foi salvo por F7a — reaproveita direto em vez de recalcular
    k_agg_train = pd.read_parquet(cfg.DIRS["interim"] / "aggregate_clearsky_index.parquet")
    k_agg_train = k_agg_train.loc[k_agg_train["split"] == "train", "k_agg"].to_numpy()

    rng2 = np.random.default_rng(SEED + 2)
    model_fits = []
    t0 = time.time()
    n_work = len(work)
    unique_partners = np.unique(work["station_partner"].to_numpy()[valid_geom])
    partner_arr = work["station_partner"].to_numpy()

    for r in range(N_MODEL):
        u1 = rng2.random(n_work)
        match_mask = valid_geom & (u1 < p_match_arr)
        idx_m = np.where(match_mask)[0]
        if len(idx_m) == 0:
            continue
        z_resid = rng2.normal(mu, sigma, size=len(idx_m))
        y_lap = alpha_dist_arr[idx_m] * x_lap_arr[idx_m] + np.power(x_lap_arr[idx_m], beta) * z_resid
        u_quant = inverse_laplace(y_lap)

        mag_synth = np.zeros(len(idx_m))
        partners_m = partner_arr[idx_m]
        for sid in np.unique(partners_m):
            sel = partners_m == sid
            mag_synth[sel] = inverse_ecdf(sorted_by_station[sid], u_quant[sel])
        mag_synth = np.clip(mag_synth, 0.0, None)

        peak_delta = sign_all[idx_m] * mag_synth                     # (n_m,)
        synth_window = real_window_full[idx_m, 0:1] + peak_delta[:, None] * ramp_shape   # (n_m, dur)
        delta_window = (synth_window - real_window_full[idx_m]) * w_j_all[idx_m][:, None]
        delta_window = np.where(np.isfinite(real_window_full[idx_m]), delta_window, 0.0)

        k_agg_mod = k_agg_train.copy()
        np.add.at(k_agg_mod, row_idx_2d_full[idx_m].reshape(-1), delta_window.reshape(-1))

        ramps_r = detect_ramps(k_agg_mod, train.index)
        sub = ramps_r[ramps_r["direction"] == DIRECTION]
        if len(sub) < 20:
            continue
        fit_r = fit_pot_gpd(sub["delta_k"].abs().to_numpy(), sub["start_ts"].to_numpy())
        if fit_r is not None:
            model_fits.append(fit_r)
        if (r + 1) % 20 == 0:
            print(f"  ... {r+1}/{N_MODEL} realizações ({time.time()-t0:.1f}s, {len(model_fits)} convergiram, "
                  f"{len(idx_m):,} coincidências injetadas)")
    print(f"  Concluído: {len(model_fits)}/{N_MODEL} realizações convergiram ({time.time()-t0:.1f}s)")

    model_boot_long = pd.DataFrame([
        {"realization_id": i, "return_period_years": T, "z": return_level(f, years_span_train, T)}
        for i, f in enumerate(model_fits) for T in RETURN_PERIODS
    ])
    model_boot_long.to_parquet(OUT_MODEL_BOOT, index=False)
    print(f"  Salvo: {OUT_MODEL_BOOT.relative_to(cfg.ROOT)}")

    # ══════════════════════════════════════════════════════════════════════
    # [3/3] RQ3 — razão REAL/INDEPENDÊNCIA (+ checagem REAL vs. IMPLICADO)
    # ══════════════════════════════════════════════════════════════════════
    print("\n[3/3] Calculando razão RQ3 (real/independência) e validação (real vs. implicado)...")
    ratio_rows = []
    for T in RETURN_PERIODS:
        z_real_ens = boot_real_long.loc[boot_real_long["return_period_years"] == T, "z"].to_numpy()
        z_indep_ens = indep_boot_long.loc[indep_boot_long["return_period_years"] == T, "z"].to_numpy() \
            if len(indep_fits) else np.array([np.nan])
        z_model_ens = model_boot_long.loc[model_boot_long["return_period_years"] == T, "z"].to_numpy() \
            if len(model_fits) else np.array([np.nan])

        z_real_point = float(fit_real_df[f"z_{T:.4f}y"].iloc[0])
        z_indep_point = float(np.median(z_indep_ens)) if len(indep_fits) else np.nan
        z_model_point = float(np.median(z_model_ens)) if len(model_fits) else np.nan

        if len(indep_fits):
            ratio_med, ratio_lo, ratio_hi = mc_ratio_ci(z_real_ens, z_indep_ens, seed=SEED + int(T * 1000))
        else:
            ratio_med = ratio_lo = ratio_hi = np.nan

        ratio_rows.append({
            "return_period_years": T,
            "z_real": z_real_point, "z_independence": z_indep_point, "z_model_implied": z_model_point,
            "ratio_real_over_independence": ratio_med,
            "ratio_ci_low": ratio_lo, "ratio_ci_high": ratio_hi,
            "model_vs_real_pct_diff": (100 * (z_model_point - z_real_point) / z_real_point
                                       if np.isfinite(z_model_point) and z_real_point else np.nan),
        })
        print(f"  T={T:.3f}y  z_real={z_real_point:.4f}  z_indep={z_indep_point:.4f}  "
              f"z_modelo={z_model_point:.4f}  razão(real/indep)={ratio_med:.3f} "
              f"IC95%=({ratio_lo:.3f},{ratio_hi:.3f})")

    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_parquet(OUT_RATIO, index=False)
    print(f"\n  Salvo: {OUT_RATIO.relative_to(cfg.ROOT)}")

    # ── Decisão RQ3 ───────────────────────────────────────────────────────────
    T_headline = 1.0
    hr = ratio_df.loc[ratio_df["return_period_years"] == T_headline].iloc[0]
    rq3_significant = hr["ratio_ci_low"] > 1.0
    print(f"\n  RQ3 (T={T_headline}y): razão real/independência = {hr['ratio_real_over_independence']:.3f}  "
          f"IC95%=({hr['ratio_ci_low']:.3f}, {hr['ratio_ci_high']:.3f})")

    if rq3_significant:
        decision = "RQ3 CONFIRMADA — EFEITO-PORTFÓLIO SOB INDEPENDÊNCIA SUBESTIMA A RESERVA REAL"
        action = (f"A reserva 'implícita por independência' subestima a reserva real necessária em "
                   f"{100*(hr['ratio_real_over_independence']-1):.0f}% (IC95% "
                   f"[{100*(hr['ratio_ci_low']-1):.0f}%, {100*(hr['ratio_ci_high']-1):.0f}%]) no horizonte "
                   f"de {T_headline:.0f} ano. Este é o número central do abstract (RQ3/F8).")
    else:
        decision = "RQ3 NÃO CONFIRMADA — SEM DIFERENÇA ESTATISTICAMENTE SIGNIFICATIVA"
        action = ("O IC da razão real/independência inclui 1 — não há evidência estatística de que a "
                   "dependência de cauda estimada exija reserva adicional além da heurística de "
                   "portfólio sob independência. Resultado ainda publicável (ver ROADMAP Seção 8: "
                   "'se RQ1 for refutada, confirma que diversificação protege até no extremo').")

    model_check = ratio_df["model_vs_real_pct_diff"].abs().median()
    model_validates = model_check < 25   # critério adotado: desvio mediano < 25% = modelo reproduz a cauda real
    print(f"\n  Validação do modelo (Estágio 1/2 vs. real): desvio percentual mediano = {model_check:.1f}%  "
          f"({'modelo reproduz a cauda real' if model_validates else 'modelo diverge do real — ver ressalva'})")

    print(f"\n  DECISÃO F8/RQ3: {decision}")
    print(f"  Ação: {action}")

    # ── Markdown ──────────────────────────────────────────────────────────────
    ratio_table_md = "\n".join(
        f"| {r.return_period_years:.3f} | {r.z_real:.4f} | {r.z_independence:.4f} | {r.z_model_implied:.4f} | "
        f"{r.ratio_real_over_independence:.3f} | ({r.ratio_ci_low:.3f}, {r.ratio_ci_high:.3f}) | "
        f"{r.model_vs_real_pct_diff:+.1f}% |"
        for r in ratio_df.itertuples()
    )
    decision_md = f"""# F8 — RQ3: Efeito-Portfólio sob Independência vs. Dependência Real

**Data:** {date.today().isoformat()}
**Decisão:** {decision}

## Três cenários comparados (mesma metodologia POT/GPD de F7/Gate G4)
1. **REAL** — série agregada observada (F7a/F7, Gate G4 aprovado).
2. **INDEPENDÊNCIA** — cada usina com deslocamento circular de blocos de dias
   INDEPENDENTE ({N_INDEP} realizações, {len(indep_fits)} convergiram; block_length={block_len} dias,
   mesmo valor de F7). Preserva exatamente a margem/GPD de cada usina, destrói a sincronia
   espacial — é o "efeito-portfólio" assumido na prática de dimensionamento de reserva,
   sem premissas paramétricas de dependência.
3. **IMPLICADO PELO MODELO** — parte da série real de treino; para cada um dos
   {work.drop_duplicates(['station_ext','t_ext']).shape[0]:,} eventos extremos históricos reais,
   substitui a resposta dos vizinhos <{RADIUS_KM}km pela previsão do modelo de dois estágios
   já aprovado no Gate G3 (`F5_two_stage.py`): Estágio 1 (offset-logístico) decide coincidência,
   Estágio 2 (Heffernan-Tawn, α(dist)/β/μ/σ já ajustados) sorteia a magnitude. {N_MODEL} realizações,
   {len(model_fits)} convergiram. Serve de VALIDAÇÃO do modelo condicional, não é o resultado
   central de RQ3 (esse é REAL vs. INDEPENDÊNCIA).

## Tab. 5 — Níveis de retorno por cenário e razão RQ3
| T (anos) | z_real | z_independência | z_implicado_modelo | Razão (real/indep) | IC95% razão | Δ modelo vs. real |
|---|---|---|---|---|---|---|
{ratio_table_md}

## RQ3 — resultado central (T={T_headline:.0f} ano)
Razão real/independência = **{hr['ratio_real_over_independence']:.3f}**
(IC95% = [{hr['ratio_ci_low']:.3f}, {hr['ratio_ci_high']:.3f}]).

{'IC exclui 1 — a dependência real exige reserva adicional, estatisticamente significativa.' if rq3_significant else 'IC inclui 1 — sem evidência estatística de reserva adicional além da heurística de independência.'}

## Validação do modelo condicional (Estágio 1/2)
Desvio percentual mediano entre o nível de retorno IMPLICADO PELO MODELO e o REAL, através
dos horizontes testados: **{model_check:.1f}%**. {'O modelo de dois estágios (Gate G3) reproduz a cauda real do portfólio dentro de uma margem razoável — evidência de que o aparato multivariado (Heffernan-Tawn condicional) tem utilidade preditiva direta para o risco agregado, não é só um ajuste post-hoc de pares.' if model_validates else 'O modelo diverge de forma não-trivial da cauda real — consistente com o achado de F6/F6b de que boa parte da dependência é de REGIME COMPARTILHADO (não capturado pelo mecanismo par-a-par do Estágio 1/2, que só perturba vizinhos <5km de eventos JÁ extremos); reportar como limitação, não invalida a razão real/independência (que não depende deste modelo).'}

## Decisão
**{decision}**

{action}

## Referência cruzada
- Fig. 9 (central): `results/figures/f8_reserve_comparison.png`
- Fig. 8: `results/figures/f7_return_level_curve.png`
- Tab. 5: `results/gates/f8_rq3_ratio.parquet`
- Ver também: `results/gates/gate4_decision.md` (cenário REAL), `results/gates/gate3_decision.md`
  (modelo de dois estágios reaproveitado no cenário IMPLICADO)
"""
    OUT_DEC.write_text(decision_md)
    print(f"  Salvo: {OUT_DEC.relative_to(cfg.ROOT)}")

    # ── Logging estruturado ───────────────────────────────────────────────────
    log_result(
        script="F8_portfolio_effect.py",
        gate="",
        phase="F8",
        params={
            "scenarios": "real (F7/G4) vs independence (per-station independent circular day-block shift) "
                         "vs model-implied (F5 two-stage Stage1/2 perturbation of real near-neighbor responses)",
            "n_independence_realizations": N_INDEP, "n_model_implied_realizations": N_MODEL,
            "near_neighbor_radius_km": RADIUS_KM,
            "ratio_ci_method": "Monte Carlo combination of two independent bootstrap ensembles (5000 paired draws)",
        },
        results={
            "n_independence_converged": len(indep_fits), "n_model_implied_converged": len(model_fits),
            "ratio_real_over_independence_1y": round(float(hr["ratio_real_over_independence"]), 3),
            "ratio_ci_low_1y": round(float(hr["ratio_ci_low"]), 3),
            "ratio_ci_high_1y": round(float(hr["ratio_ci_high"]), 3),
            "rq3_significant": bool(rq3_significant),
            "model_vs_real_median_pct_diff": round(float(model_check), 1),
        },
        decision=decision,
        action=action,
        interpretation=(
            f"RQ3 (central result) is tested via three scenarios for the capacity-weighted aggregate "
            f"network ramp series (F7a's 'virtual plant'), all fit with the exact same POT/GPD + "
            f"moving-block-bootstrap methodology as Gate G4: (1) REAL -- the observed series (Gate G4's "
            f"already-approved fit); (2) INDEPENDENCE -- an empirical counterfactual built by giving each "
            f"station an INDEPENDENT circular day-block shift (preserves each station's exact marginal/GPD "
            f"distribution, destroys cross-station synchrony), requiring no parametric dependence model at "
            f"all -- this operationalizes the 'geographic portfolio effect' assumption used in reserve "
            f"sizing practice; (3) MODEL-IMPLIED -- starting from the real training series, near-neighbor "
            f"(<{RADIUS_KM}km) responses to the "
            f"{work.drop_duplicates(['station_ext','t_ext']).shape[0]:,} real historical extreme "
            f"conditioning events are REPLACED by draws from the already-fitted and Gate-G3-approved "
            f"two-stage Heffernan-Tawn model (Stage 1 excess-coincidence, Stage 2 conditional magnitude), "
            f"serving as a VALIDATION of whether the pairwise conditional-extremes apparatus (the paper's "
            f"main multivariate EVT contribution) actually reproduces real portfolio tail risk, rather "
            f"than being a purely pairwise curiosity. RQ3 headline (T=1 year): the real/independence "
            f"return-level ratio is {hr['ratio_real_over_independence']:.3f} "
            f"(95% CI [{hr['ratio_ci_low']:.3f}, {hr['ratio_ci_high']:.3f}], via Monte Carlo combination "
            f"of the two scenarios' independent bootstrap ensembles) -- "
            f"{'the CI excludes 1, confirming that the independence-based reserve heuristic significantly UNDERESTIMATES the real tail risk' if rq3_significant else 'the CI includes 1, providing no significant evidence that real tail dependence requires additional reserve beyond the independence heuristic'}. "
            f"The model-implied scenario's median absolute deviation from the real scenario's return "
            f"levels across tested horizons is {model_check:.1f}% -- "
            f"{'within a defensible range, evidencing that the fitted two-stage conditional extremes model has direct predictive value for aggregate portfolio tail risk' if model_validates else 'a non-trivial gap, consistent with F6/F6b evidence that a substantial share of the dependence is SHARED-REGIME (not captured by the pairwise Stage1/2 mechanism, which only perturbs <5km neighbors of already-extreme events) -- reported as a limitation, though it does not affect the real-vs-independence ratio (which is model-free)'}."
        ),
        paper_ref="Section 9 — RQ3 Central Result (F8); Fig. 9; Tab. 5; f8_rq3_decision.md",
    )

    # ── Figura 9 (central) ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    Ts = ratio_df["return_period_years"].to_numpy()
    ax.plot(Ts, ratio_df["z_real"], "o-", color="crimson", label="REAL (observado)")
    ax.plot(Ts, ratio_df["z_independence"], "s--", color="#2c7bb6", label="INDEPENDÊNCIA (contrafactual)")
    ax.plot(Ts, ratio_df["z_model_implied"], "^:", color="seagreen", label="IMPLICADO PELO MODELO (F5)")
    ax.set_xscale("log")
    ax.set_xlabel("Período de retorno T (anos)")
    ax.set_ylabel(f"Nível de retorno |Δk_agg| ({DIRECTION})")
    ax.set_title("Reserva por cenário")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(Ts, ratio_df["ratio_real_over_independence"], "o-", color="black")
    ax2.fill_between(Ts, ratio_df["ratio_ci_low"], ratio_df["ratio_ci_high"], color="grey", alpha=0.25,
                      label="IC 95% (combinação Monte Carlo)")
    ax2.axhline(1.0, color="crimson", lw=1, linestyle="--", label="razão=1 (sem efeito adicional)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Período de retorno T (anos)")
    ax2.set_ylabel("Razão reserva real / reserva independência")
    ax2.set_title("Fig. 9 (central) — RQ3: efeito-portfólio falha na cauda?")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG9, dpi=150)
    plt.close()
    print(f"\n  Figura: {OUT_FIG9.relative_to(cfg.ROOT)}")

    print(f"\n{SEP}")
    print(f"F8 / RQ3 — {decision}")


if __name__ == "__main__":
    main()
