"""
S1_block_length_sensitivity.py — Sensibilidade do bootstrap ao block_length (Risco E)
========================================================================================
Motivação (ROADMAP Seção 0.2, Risco E): `estimate_block_length()` (C1_gate1.py) tem um
`np.clip(L, 2, 10)` HARD-CODED — estruturalmente incapaz de escolher block_length > 10
dias, não importa o que a autocorrelação real diga. Essa função é reaproveitada nos
bootstraps de Gate G1 (chi), Gate G4/F7 (níveis de retorno REAIS) e F8 (cenário
INDEPENDÊNCIA + razão RQ3). Se a persistência do "regime compartilhado" (achado central
de C1b) se estender além de 10 dias, os IC atuais podem ser anti-conservadores (estreitos
demais) — incluindo o IC da razão RQ3 = 2,39×, o número mais citável do paper.

Este script NÃO reestima block_length — FORÇA manualmente block_length em {10, 20, 30}
(10 = valor de produção atual, incluído como referência) e re-roda, para cada valor:
  1. Cenário REAL (Gate G4): bootstrap por blocos móveis da série agregada de treino
     (reaproveita `analyze_series`/`fit_pot_gpd`/`return_level` de F7_return_levels.py).
  2. Cenário INDEPENDÊNCIA (F8): deslocamento circular de blocos de dias, independente
     por usina (reaproveita a mesma lógica de F8_portfolio_effect.py).
  3. Razão RQ3 (real/independência) em T=1 ano: reaproveita `mc_ratio_ci` de F8.

Critério de leitura: se o IC95% da razão a T=1 ano permanecer > 1 mesmo com
block_length=30 (3x o valor de produção), a conclusão de RQ3 fica confirmada como robusta
a essa escolha metodológica. Se o IC passar a incluir 1 em algum block_length maior, é um
achado importante a resolver antes de reportar RQ3 como "confirmada" sem ressalva.

Saídas:
  results/gates/s1_block_length_sensitivity.parquet
  results/gates/s1_block_length_sensitivity.md
  results/figures/s1_block_length_sensitivity.png

Executar:
    python S1_block_length_sensitivity.py
"""

from __future__ import annotations

import sys
import time
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
from B5_ramp_detection import _sdt_compress_core
from C1_gate1 import make_block_index_arrays
from F7_return_levels import detect_ramps, fit_pot_gpd, return_level, DIRECTION, RETURN_PERIODS
from F7a_aggregate_series import build_capacity_weights
from F8_portfolio_effect import mc_ratio_ci

IN_K_AGG = cfg.DIRS["interim"] / "aggregate_clearsky_index.parquet"
IN_K     = cfg.DIRS["interim"] / "clearsky_index.parquet"
IN_COORDS = cfg.DIRS["interim"] / "coords.parquet"

OUT_PARQUET = cfg.DIRS["gates"] / "s1_block_length_sensitivity.parquet"
OUT_MD      = cfg.DIRS["gates"] / "s1_block_length_sensitivity.md"
OUT_FIG     = cfg.DIRS["figures"] / "s1_block_length_sensitivity.png"

N_BOOT  = cfg.F7["n_bootstrap_real"]
N_INDEP = cfg.F7["n_independence_real"]
BLOCK_LENGTHS = [10, 20, 30]   # 10 = valor de produção atual (referência); 20/30 = sensibilidade
T_HEADLINE = 1.0
SEED = cfg.SEED
SEP = "─" * 60


def real_scenario_ensemble(k_train: np.ndarray, n_days_train: int, years_span_train: float,
                            block_len: int, seed: int) -> list[dict]:
    """Bootstrap por blocos móveis da série REAL agregada — mesma lógica de F7_return_levels.py,
    mas com block_length FORÇADO em vez de estimado via ACF."""
    K_days = k_train.reshape(n_days_train, 1440)
    rng = np.random.default_rng(seed)
    boot_idx = make_block_index_arrays(n_days_train, block_len, N_BOOT, rng)
    synth_dt_index = pd.date_range("2000-01-01", periods=n_days_train * 1440, freq="1min")
    fits = []
    for b in range(N_BOOT):
        k_boot = K_days[boot_idx[b]].reshape(-1)
        ramps = detect_ramps(k_boot, synth_dt_index)
        sub = ramps[ramps["direction"] == DIRECTION]
        if len(sub) < 20:
            continue
        fit = fit_pot_gpd(sub["delta_k"].abs().to_numpy(), sub["start_ts"].to_numpy())
        if fit is not None:
            fits.append(fit)
    return fits


def independence_scenario_ensemble(K3: np.ndarray, w: np.ndarray, n_days: int, station_cols: list[str],
                                    block_len: int, seed: int) -> list[dict]:
    """Cenário INDEPENDÊNCIA — mesma lógica de F8_portfolio_effect.py, block_length FORÇADO."""
    synth_dt_index = pd.date_range("2000-01-01", periods=n_days * 1440, freq="1min")
    rng = np.random.default_rng(seed)
    fits = []
    for r in range(N_INDEP):
        station_day_idx = make_block_index_arrays(n_days, block_len, len(station_cols), rng)
        sum_acc = np.zeros((n_days, 1440))
        weight_acc = np.zeros((n_days, 1440))
        for s in range(len(station_cols)):
            sl = K3[station_day_idx[s], :, s]
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
            fits.append(fit_r)
    return fits


def main() -> None:
    print(SEP)
    print("S1 — SENSIBILIDADE DO BOOTSTRAP AO BLOCK_LENGTH (Risco E, ROADMAP Seção 0.2)")
    print(SEP)

    for p in (IN_K_AGG, IN_K, IN_COORDS):
        if not p.exists():
            print(f"ERRO: {p} não encontrado. Execute F7a/F7/F8 primeiro.")
            sys.exit(1)

    df_k_agg = pd.read_parquet(IN_K_AGG)
    train_agg = df_k_agg[df_k_agg["split"] == "train"]
    k_train = train_agg["k_agg"].to_numpy()
    n_days_train = len(k_train) // 1440
    assert len(k_train) == n_days_train * 1440
    years_span_train = n_days_train / 365.25
    print(f"\n  Treino: {n_days_train} dias ({years_span_train:.3f} anos)")

    df_k = pd.read_parquet(IN_K)
    coords = pd.read_parquet(IN_COORDS)
    station_cols = [c for c in df_k.columns if c.startswith("ID")]
    w = build_capacity_weights(coords, station_cols)
    train = df_k[df_k["split"] == "train"]
    K = train[station_cols].to_numpy(dtype=float)
    n_days = len(K) // 1440
    assert len(K) == n_days * 1440 == n_days_train * 1440
    K3 = K.reshape(n_days, 1440, len(station_cols))

    _sdt_compress_core(np.array([0.0, 0.1, 0.05]), 0.02)   # warm-up JIT

    rows = []
    t_all0 = time.time()
    for block_len in BLOCK_LENGTHS:
        print(f"\n{'='*60}\nblock_length = {block_len} dias\n{'='*60}")

        t0 = time.time()
        real_fits = real_scenario_ensemble(k_train, n_days_train, years_span_train, block_len, SEED)
        print(f"  [REAL] {len(real_fits)}/{N_BOOT} réplicas convergiram ({time.time()-t0:.1f}s)")

        t0 = time.time()
        indep_fits = independence_scenario_ensemble(K3, w, n_days, station_cols, block_len, SEED + 1)
        print(f"  [INDEPENDÊNCIA] {len(indep_fits)}/{N_INDEP} réplicas convergiram ({time.time()-t0:.1f}s)")

        if not real_fits or not indep_fits:
            print(f"  AVISO: ensemble vazio para block_length={block_len} — pulando.")
            continue

        z_real_ens = np.array([return_level(f, years_span_train, T_HEADLINE) for f in real_fits])
        z_indep_ens = np.array([return_level(f, years_span_train, T_HEADLINE) for f in indep_fits])
        ratio_med, ratio_lo, ratio_hi = mc_ratio_ci(z_real_ens, z_indep_ens, seed=SEED + block_len)

        z_real_ci = (float(np.percentile(z_real_ens, 2.5)), float(np.percentile(z_real_ens, 97.5)))
        z_indep_ci = (float(np.percentile(z_indep_ens, 2.5)), float(np.percentile(z_indep_ens, 97.5)))
        real_ci_width = z_real_ci[1] - z_real_ci[0]

        row = {
            "block_length_days": block_len,
            "n_real_converged": len(real_fits), "n_indep_converged": len(indep_fits),
            "z_real_median": float(np.median(z_real_ens)),
            "z_real_ci_low": z_real_ci[0], "z_real_ci_high": z_real_ci[1],
            "real_ci_width": real_ci_width,
            "z_indep_median": float(np.median(z_indep_ens)),
            "z_indep_ci_low": z_indep_ci[0], "z_indep_ci_high": z_indep_ci[1],
            "ratio_median": ratio_med, "ratio_ci_low": ratio_lo, "ratio_ci_high": ratio_hi,
            "ratio_significant": bool(ratio_lo > 1.0),
        }
        rows.append(row)
        print(f"  z_real(1y)   = {row['z_real_median']:.4f}  IC95%=({z_real_ci[0]:.4f},{z_real_ci[1]:.4f})  largura={real_ci_width:.4f}")
        print(f"  z_indep(1y)  = {row['z_indep_median']:.4f}  IC95%=({z_indep_ci[0]:.4f},{z_indep_ci[1]:.4f})")
        print(f"  razão(1y)    = {ratio_med:.3f}  IC95%=({ratio_lo:.3f},{ratio_hi:.3f})  "
              f"{'SIGNIFICATIVA (>1)' if row['ratio_significant'] else 'NÃO significativa (IC inclui 1)'}")

    print(f"\n  Tempo total: {time.time()-t_all0:.1f}s")

    if not rows:
        print("ERRO: nenhum block_length produziu ensemble válido.")
        sys.exit(1)

    result_df = pd.DataFrame(rows)
    result_df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n  Salvo: {OUT_PARQUET.relative_to(cfg.ROOT)}")

    # ── Decisão ────────────────────────────────────────────────────────────────
    baseline = result_df.loc[result_df["block_length_days"] == 10].iloc[0] if 10 in result_df["block_length_days"].values else result_df.iloc[0]
    all_significant = bool(result_df["ratio_significant"].all())
    max_block = int(result_df["block_length_days"].max())
    ratio_at_max = result_df.loc[result_df["block_length_days"] == max_block].iloc[0]

    if all_significant:
        decision = f"ROBUSTO — RAZÃO RQ3 PERMANECE SIGNIFICATIVA ATÉ block_length={max_block} DIAS"
        action = (f"O IC95% da razão real/independência a T=1 ano exclui 1 em TODOS os block_length "
                   f"testados ({', '.join(str(b) for b in BLOCK_LENGTHS)} dias), inclusive {max_block} dias "
                   f"(3x o valor de produção). O cap estrutural de estimate_block_length() em 10 dias "
                   f"(Risco E) NÃO ameaça a conclusão de RQ3 — a razão permanece estatisticamente "
                   f"significativa mesmo assumindo uma persistência de regime muito mais longa do que a "
                   f"função de produção jamais conseguiria escolher sozinha.")
    else:
        decision = "ATENÇÃO — RAZÃO RQ3 DEIXA DE SER SIGNIFICATIVA EM ALGUM block_length TESTADO"
        action = (f"O IC95% da razão real/independência a T=1 ano passa a incluir 1 em pelo menos um "
                   f"block_length testado acima do valor de produção (10 dias). Isso indica que o cap "
                   f"estrutural de estimate_block_length() (Risco E) pode estar produzindo um IC "
                   f"anti-conservador na produção — revisar block_length antes de reportar RQ3 como "
                   f"'confirmada' sem essa ressalva explícita.")

    print(f"\n  DECISÃO: {decision}")
    print(f"  Ação: {action}")

    # ── Markdown ──────────────────────────────────────────────────────────────
    table_md = "\n".join(
        f"| {r.block_length_days} | {r.n_real_converged}/{N_BOOT} | {r.z_real_median:.4f} "
        f"({r.z_real_ci_low:.4f}, {r.z_real_ci_high:.4f}) | {r.real_ci_width:.4f} | "
        f"{r.n_indep_converged}/{N_INDEP} | {r.z_indep_median:.4f} "
        f"({r.z_indep_ci_low:.4f}, {r.z_indep_ci_high:.4f}) | {r.ratio_median:.3f} "
        f"({r.ratio_ci_low:.3f}, {r.ratio_ci_high:.3f}) | {'✓' if r.ratio_significant else '✗'} |"
        for r in result_df.itertuples()
    )
    decision_md = f"""# S1 — Sensibilidade do Bootstrap ao Block Length (Risco E)

**Data:** {date.today().isoformat()}
**Decisão:** {decision}

## Motivação
`estimate_block_length()` (`C1_gate1.py`) tem um `np.clip(L, 2, 10)` hard-coded — não importa
o que a autocorrelação real diga, a função é estruturalmente incapaz de escolher block_length
acima de 10 dias. Essa função é reaproveitada nos bootstraps de Gate G1, Gate G4/F7 (níveis de
retorno REAIS) e F8 (cenário INDEPENDÊNCIA + razão RQ3 = 2,39×). Este script força
block_length ∈ {{{', '.join(str(b) for b in BLOCK_LENGTHS)}}} dias manualmente (10 = valor de
produção, incluído como referência) e recomputa os cenários REAL e INDEPENDÊNCIA do zero para
cada valor, checando se a razão RQ3 (T=1 ano) permanece estatisticamente significativa.

## Resultado
| block_length (dias) | REAL convergiu | z_real(1y) (IC95%) | largura IC real | INDEP convergiu | z_indep(1y) (IC95%) | razão (IC95%) | Significativa? |
|---|---|---|---|---|---|---|---|
{table_md}

## Decisão
**{decision}**

{action}

## Referência cruzada
- ROADMAP Seção 0.2, Risco E
- Ver também: `results/gates/gate4_decision.md` (cenário REAL, block_length=10 de produção),
  `results/gates/f8_rq3_decision.md` (razão RQ3 original, block_length=10 de produção)
- Fig.: `results/figures/s1_block_length_sensitivity.png`
"""
    OUT_MD.write_text(decision_md)
    print(f"  Salvo: {OUT_MD.relative_to(cfg.ROOT)}")

    # ── Logging estruturado ───────────────────────────────────────────────────
    log_result(
        script="S1_block_length_sensitivity.py",
        gate="",
        phase="S1",
        params={
            "block_lengths_tested": BLOCK_LENGTHS,
            "production_block_length": 10,
            "n_bootstrap_real": N_BOOT, "n_independence": N_INDEP,
            "return_period_headline": T_HEADLINE,
            "motivation": "estimate_block_length() in C1_gate1.py has a hard-coded np.clip(L,2,10) "
                          "cap, structurally unable to select block_length > 10 days regardless of "
                          "the true autocorrelation of the shared-regime dependence found in C1b",
        },
        results={
            f"ratio_median_bl{int(r.block_length_days)}": round(float(r.ratio_median), 3)
            for r in result_df.itertuples()
        } | {
            f"ratio_ci_low_bl{int(r.block_length_days)}": round(float(r.ratio_ci_low), 3)
            for r in result_df.itertuples()
        } | {"all_block_lengths_significant": all_significant},
        decision=decision,
        action=action,
        interpretation=(
            f"Sensitivity check for Risk E (ROADMAP Section 0.2): estimate_block_length() has a "
            f"hard-coded np.clip(L,2,10), structurally incapable of returning a value above 10 days "
            f"regardless of the true autocorrelation of the network's shared-regime dependence (C1b's "
            f"central finding). This script forces block_length in {BLOCK_LENGTHS} (10 = current "
            f"production value, included as reference) and fully re-runs the REAL and INDEPENDENCE "
            f"bootstrap ensembles from scratch for each value, re-computing the RQ3 headline ratio "
            f"(T=1 year) each time. "
            f"{'The ratio remains statistically significant (CI excludes 1) at every tested block_length, including 3x the production value -- the 10-day cap does NOT threaten the RQ3 conclusion.' if all_significant else 'The ratio CI includes 1 at at least one tested block_length above the production value -- the 10-day cap may be producing an anti-conservative (too narrow) CI in production; RQ3 should not be reported as unconditionally confirmed without this caveat.'} "
            f"At block_length={max_block} (3x production): ratio={ratio_at_max['ratio_median']:.3f}, "
            f"95% CI=[{ratio_at_max['ratio_ci_low']:.3f}, {ratio_at_max['ratio_ci_high']:.3f}]."
        ),
        paper_ref="ROADMAP Section 0.2 (Risk E); Section 9 (RQ3 robustness)",
    )

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    bl = result_df["block_length_days"].to_numpy()
    ax.errorbar(bl, result_df["z_real_median"],
                yerr=[result_df["z_real_median"] - result_df["z_real_ci_low"],
                      result_df["z_real_ci_high"] - result_df["z_real_median"]],
                fmt="o-", color="crimson", capsize=4, label="REAL")
    ax.errorbar(bl, result_df["z_indep_median"],
                yerr=[result_df["z_indep_median"] - result_df["z_indep_ci_low"],
                      result_df["z_indep_ci_high"] - result_df["z_indep_median"]],
                fmt="s--", color="#2c7bb6", capsize=4, label="INDEPENDÊNCIA")
    ax.set_xlabel("block_length (dias)")
    ax.set_ylabel(f"Nível de retorno |Δk_agg| a T=1 ano ({DIRECTION})")
    ax.set_title("Níveis de retorno por block_length")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.errorbar(bl, result_df["ratio_median"],
                 yerr=[result_df["ratio_median"] - result_df["ratio_ci_low"],
                       result_df["ratio_ci_high"] - result_df["ratio_median"]],
                 fmt="o-", color="black", capsize=4)
    ax2.axhline(1.0, color="crimson", lw=1, linestyle="--", label="razão=1")
    ax2.axvline(10, color="grey", lw=1, linestyle=":", label="block_length de produção")
    ax2.set_xlabel("block_length (dias)")
    ax2.set_ylabel("Razão reserva real / independência (T=1 ano)")
    ax2.set_title("S1 — Sensibilidade da razão RQ3 ao block_length")
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()
    print(f"\n  Figura: {OUT_FIG.relative_to(cfg.ROOT)}")

    print(f"\n{SEP}")
    print(f"S1 — {decision}")


if __name__ == "__main__":
    main()
