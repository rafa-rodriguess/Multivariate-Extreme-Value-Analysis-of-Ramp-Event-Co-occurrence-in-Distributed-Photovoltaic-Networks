"""
S2_f8_convergence_diagnosis.py — Diagnóstico das não-convergências do cenário
INDEPENDÊNCIA em F8 (Risco F)
========================================================================================
Motivação (ROADMAP Seção 0.2, Risco F): `F8_portfolio_effect.py` reporta apenas a
contagem agregada de realizações convergidas no cenário INDEPENDÊNCIA (ex.: 143/150),
sem registrar QUAL realização falhou, POR QUAL motivo, ou qualquer covariável da
realização descartada. Se as falhas não forem aleatórias — por exemplo, se réplicas com
atividade extrema mais concentrada/dispersa tendem a falhar mais — a razão RQ3 poderia
estar sutilmente viesada (as réplicas sobreviventes não seriam uma amostra representativa
do espaço de reamostragem).

Este script REPRODUZ EXATAMENTE o cenário INDEPENDÊNCIA de F8 (mesma seed, mesma lógica de
deslocamento circular por usina) mas instrumenta cada uma das 150 realizações com:
  - motivo específico de falha (< 20 rampas brutas | < 30 excessos p/ candidato de limiar |
    < 5 excedências acima do limiar escolhido | < min_declustered_agg declusterizadas |
    GPD não convergiu | |xi| >= 0.5 | convergiu)
  - covariáveis descritivas da réplica (nº de rampas "down" detectadas, magnitude máxima,
    magnitude mediana) — para comparar convergidas vs. não-convergidas.

Não modifica nem reabre `F8_portfolio_effect.py` (script já aprovado) — duplica a lógica
necessária, mesmo princípio já usado por F8 ao duplicar o Estágio 1 de F5.

Saídas:
  results/gates/s2_f8_convergence_diagnosis.parquet   — 1 linha por realização (150)
  results/gates/s2_f8_convergence_diagnosis.md
  results/figures/s2_f8_convergence_diagnosis.png

Executar:
    python S2_f8_convergence_diagnosis.py
"""

from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result
from B5_ramp_detection import _sdt_compress_core
from C1_gate1 import make_block_index_arrays
from C2_gate2 import select_threshold_adaptive, decluster, fit_gpd_plain, CANDIDATE_QUANTILES, MIN_RAW_FOR_CANDIDATE
from F7_return_levels import detect_ramps, DIRECTION
from F7a_aggregate_series import build_capacity_weights

IN_K      = cfg.DIRS["interim"] / "clearsky_index.parquet"
IN_COORDS = cfg.DIRS["interim"] / "coords.parquet"

OUT_PARQUET = cfg.DIRS["gates"] / "s2_f8_convergence_diagnosis.parquet"
OUT_MD      = cfg.DIRS["gates"] / "s2_f8_convergence_diagnosis.md"
OUT_FIG     = cfg.DIRS["figures"] / "s2_f8_convergence_diagnosis.png"

N_INDEP = cfg.F7["n_independence_real"]
MIN_DECLUSTERED_AGG = cfg.F7["min_declustered_agg"]
SEED = cfg.SEED
SEP = "─" * 60


def fit_pot_gpd_diag(mags: np.ndarray, times: np.ndarray) -> dict:
    """Réplica instrumentada de `fit_pot_gpd` (F7_return_levels.py) — idêntica lógica,
    mas retorna o motivo específico de falha em vez de None silencioso."""
    if len(mags) < MIN_RAW_FOR_CANDIDATE * 2:
        return {"converged": False, "fail_reason": "raw_ramps_below_2x_min_candidate", "xi": np.nan}
    chosen_k, thresholds, xi_arr, se_arr, n_arr, mrl_arr, fallback = select_threshold_adaptive(
        mags, CANDIDATE_QUANTILES)
    u = thresholds[chosen_k]
    order = np.argsort(times)
    t_sorted, m_sorted = times[order], mags[order]
    mask = m_sorted > u
    t_exc, m_exc = t_sorted[mask], m_sorted[mask]
    if len(t_exc) < 5:
        return {"converged": False, "fail_reason": "exceedances_above_threshold_below_5", "xi": np.nan}
    idx_max, theta_hat, run_length = decluster(t_exc, m_exc)
    y_final = m_exc[idx_max] - u
    if len(y_final) < MIN_DECLUSTERED_AGG:
        return {"converged": False, "fail_reason": "declustered_below_min_declustered_agg", "xi": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_gpd_plain(y_final)
    if not fit["converged"]:
        return {"converged": False, "fail_reason": "gpd_mle_did_not_converge", "xi": float(fit.get("xi", np.nan))}
    if not (np.isfinite(fit["xi"]) and abs(fit["xi"]) < 0.5):
        return {"converged": False, "fail_reason": "xi_out_of_bounds_or_nonfinite", "xi": float(fit["xi"])}
    return {"converged": True, "fail_reason": "", "xi": float(fit["xi"]),
            "n_declustered": len(y_final), "threshold_quantile": float(CANDIDATE_QUANTILES[chosen_k])}


def main() -> None:
    print(SEP)
    print("S2 — DIAGNÓSTICO DAS NÃO-CONVERGÊNCIAS DO CENÁRIO INDEPENDÊNCIA (Risco F, ROADMAP Seção 0.2)")
    print(SEP)

    for p in (IN_K, IN_COORDS):
        if not p.exists():
            print(f"ERRO: {p} não encontrado. Execute B4/B8 primeiro.")
            sys.exit(1)

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
    block_len = 10   # mesmo valor de produção de F7/F8 (Gate G4)

    _sdt_compress_core(np.array([0.0, 0.1, 0.05]), 0.02)   # warm-up JIT

    print(f"\n  Reproduzindo EXATAMENTE o cenário INDEPENDÊNCIA de F8 (seed={SEED+1}, "
          f"block_length={block_len}d, N={N_INDEP})...")
    rng = np.random.default_rng(SEED + 1)   # idêntico a F8_portfolio_effect.py (rng = default_rng(SEED+1))
    rows = []
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

        row = {"realization_id": r, "n_ramps_down": len(sub)}
        if len(sub) < 20:
            row.update({"converged": False, "fail_reason": "fewer_than_20_down_ramps",
                        "max_mag": np.nan, "median_mag": np.nan, "xi": np.nan})
        else:
            mags = sub["delta_k"].abs().to_numpy()
            diag = fit_pot_gpd_diag(mags, sub["start_ts"].to_numpy())
            row.update({
                "converged": diag["converged"], "fail_reason": diag["fail_reason"],
                "max_mag": float(mags.max()), "median_mag": float(np.median(mags)),
                "xi": diag.get("xi", np.nan),
            })
        rows.append(row)
        if (r + 1) % 30 == 0:
            n_ok = sum(x["converged"] for x in rows)
            print(f"  ... {r+1}/{N_INDEP} realizações ({n_ok} convergiram)")

    diag_df = pd.DataFrame(rows)
    diag_df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n  Salvo: {OUT_PARQUET.relative_to(cfg.ROOT)}")

    n_converged = int(diag_df["converged"].sum())
    n_failed = N_INDEP - n_converged
    print(f"\n  Convergiram: {n_converged}/{N_INDEP}  |  Falharam: {n_failed}/{N_INDEP}")
    print(f"  Confere com F8_portfolio_effect.py original? esperado 143/150 convergidas "
          f"(f8_rq3_decision.md) — obtido aqui: {n_converged}/{N_INDEP}.")

    if n_failed > 0:
        print("\n  Motivos de falha:")
        for reason, cnt in diag_df.loc[~diag_df["converged"], "fail_reason"].value_counts().items():
            print(f"    {reason}: {cnt}")

    # ── Teste de aleatoriedade das falhas: convergidas vs. não-convergidas ──────
    ok = diag_df[diag_df["converged"]]
    bad = diag_df[~diag_df["converged"]]
    print(f"\n  Comparação convergidas (n={len(ok)}) vs. não-convergidas (n={len(bad)}):")
    print(f"    n_ramps_down   — convergidas: mediana={ok['n_ramps_down'].median():.0f}  "
          f"| não-convergidas: mediana={bad['n_ramps_down'].median() if len(bad) else float('nan'):.0f}")

    mw_stat = mw_p = np.nan
    if len(bad) >= 3 and len(ok) >= 3:
        mw_stat, mw_p = stats.mannwhitneyu(ok["n_ramps_down"], bad["n_ramps_down"], alternative="two-sided")
        print(f"    Mann-Whitney U (n_ramps_down, convergidas vs. não): U={mw_stat:.1f}  p={mw_p:.4f}  "
              f"{'(diferença significativa — falhas NÃO parecem aleatórias)' if mw_p < 0.05 else '(sem diferença significativa — consistente com falhas aleatórias/idiossincráticas)'}")
    else:
        print("    Amostra de falhas pequena demais para teste formal (Mann-Whitney) — reportar apenas descritivo.")

    # ── Decisão ────────────────────────────────────────────────────────────────
    frac_failed = n_failed / N_INDEP
    systematic = bool(np.isfinite(mw_p) and mw_p < 0.05)
    if n_failed == 0:
        decision = "SEM FALHAS — TODAS AS REALIZAÇÕES CONVERGIRAM NESTA REPRODUÇÃO"
        action = "Nenhuma ação necessária."
    elif not systematic:
        decision = f"FALHAS CONSISTENTES COM ALEATORIEDADE — {n_failed}/{N_INDEP} ({frac_failed:.1%})"
        action = (f"As {n_failed} realizações não-convergidas não diferem significativamente das "
                   f"convergidas em nº de rampas 'down' detectadas (Mann-Whitney p={mw_p:.3f} se "
                   f"aplicável, ou amostra pequena demais para teste formal). Não há evidência de "
                   f"viés sistemático na direção de reservas mais/menos extremas — a razão RQ3 "
                   f"(2,39×) não parece inflada/deflacionada por descarte seletivo de réplicas.")
    else:
        decision = f"⚠ FALHAS POTENCIALMENTE SISTEMÁTICAS — {n_failed}/{N_INDEP} ({frac_failed:.1%})"
        action = (f"As realizações não-convergidas diferem significativamente das convergidas em "
                   f"nº de rampas 'down' detectadas (Mann-Whitney p={mw_p:.3f}) — possível viés de "
                   f"seleção na razão RQ3. Investigar se o motivo de falha dominante "
                   f"({diag_df.loc[~diag_df['converged'],'fail_reason'].mode().iloc[0] if n_failed else 'N/A'}) "
                   f"tende a ocorrer em réplicas com atividade sistematicamente maior ou menor, e "
                   f"reportar como limitação explícita se confirmado.")

    print(f"\n  DECISÃO: {decision}")
    print(f"  Ação: {action}")

    # ── Markdown ──────────────────────────────────────────────────────────────
    reason_table = diag_df.loc[~diag_df["converged"], "fail_reason"].value_counts()
    reason_md = "\n".join(f"| {reason} | {cnt} |" for reason, cnt in reason_table.items()) if n_failed else "| (nenhuma falha) | 0 |"
    decision_md = f"""# S2 — Diagnóstico das Não-Convergências do Cenário INDEPENDÊNCIA (F8, Risco F)

**Data:** {date.today().isoformat()}
**Decisão:** {decision}

## Motivação
`F8_portfolio_effect.py` reportava apenas a contagem agregada de realizações convergidas no
cenário INDEPENDÊNCIA (143/150), sem registrar qual realização falhou, por qual motivo, ou
qualquer covariável da réplica descartada. Este script reproduz exatamente a mesma sequência de
realizações (mesma seed) e instrumenta cada uma com o motivo específico de falha.

## Resultado agregado
- Convergiram: **{n_converged}/{N_INDEP}**
- Falharam: **{n_failed}/{N_INDEP}** ({frac_failed:.1%})

## Motivos de falha
| Motivo | Contagem |
|---|---|
{reason_md}

## Teste de aleatoriedade (convergidas vs. não-convergidas, nº de rampas 'down' detectadas)
Mann-Whitney U = {(f'{mw_stat:.1f}' if np.isfinite(mw_stat) else 'N/A (amostra pequena demais)')}, p = {(f'{mw_p:.4f}' if np.isfinite(mw_p) else 'N/A')}

## Decisão
**{decision}**

{action}

## Referência cruzada
- ROADMAP Seção 0.2, Risco F
- Ver também: `results/gates/f8_rq3_decision.md` (cenário INDEPENDÊNCIA original, mesma seed)
- Dados completos por realização: `results/gates/s2_f8_convergence_diagnosis.parquet`
"""
    OUT_MD.write_text(decision_md)
    print(f"  Salvo: {OUT_MD.relative_to(cfg.ROOT)}")

    # ── Logging estruturado ───────────────────────────────────────────────────
    log_result(
        script="S2_f8_convergence_diagnosis.py",
        gate="",
        phase="S2",
        params={
            "reproduces": "F8_portfolio_effect.py's INDEPENDENCE scenario, identical seed (SEED+1) "
                          "and block_length=10 (production value)",
            "n_realizations": N_INDEP,
            "instrumentation": "per-realization failure reason (5 possible stages) + descriptive "
                                "covariates (n_ramps_down, max/median magnitude)",
        },
        results={
            "n_converged": n_converged, "n_failed": n_failed,
            "frac_failed": round(frac_failed, 4),
            "mann_whitney_p": round(float(mw_p), 4) if np.isfinite(mw_p) else None,
            "failures_systematic": systematic,
            "fail_reasons": reason_table.to_dict() if n_failed else {},
        },
        decision=decision,
        action=action,
        interpretation=(
            f"Diagnostic re-run of F8's INDEPENDENCE scenario (Risk F, ROADMAP Section 0.2), using "
            f"the identical RNG seed and block_length to reproduce the exact same 150 realizations, "
            f"now instrumented to record WHY each non-converging realization failed (5 possible "
            f"pipeline stages: raw ramp count, threshold-candidate exceedance count, post-threshold "
            f"exceedance count, post-declustering count, or GPD MLE convergence/xi bounds) and "
            f"descriptive covariates (n_ramps_down, max/median magnitude) for every realization. "
            f"Result: {n_converged}/{N_INDEP} converged, {n_failed} failed "
            f"({frac_failed:.1%}), dominant failure reason(s): "
            f"{', '.join(reason_table.index.tolist()) if n_failed else 'N/A'}. "
            f"{'A Mann-Whitney test on n_ramps_down finds no significant difference between converged and non-converged realizations, consistent with failures being idiosyncratic/random rather than systematically biased toward more or less extreme replicas -- the RQ3 ratio (2.39x) does not appear to be inflated or deflated by selective replica attrition.' if not systematic else 'A Mann-Whitney test finds a SIGNIFICANT difference in n_ramps_down between converged and non-converged realizations -- possible selection bias in the RQ3 ratio, flagged as a limitation pending further investigation.'}"
        ),
        paper_ref="ROADMAP Section 0.2 (Risk F); Section 9 (RQ3 robustness)",
    )

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    ax.hist(ok["n_ramps_down"], bins=20, alpha=0.6, color="#2c7bb6", label=f"convergiram (n={len(ok)})")
    if len(bad):
        ax.hist(bad["n_ramps_down"], bins=20, alpha=0.7, color="crimson", label=f"não convergiram (n={len(bad)})")
    ax.set_xlabel("Nº de rampas 'down' detectadas na réplica")
    ax.set_ylabel("Contagem de realizações")
    ax.set_title("S2 — Distribuição de atividade: convergidas vs. não")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    if n_failed:
        reason_table.plot(kind="barh", ax=ax2, color="crimson")
    else:
        ax2.text(0.5, 0.5, "Nenhuma falha", ha="center", va="center")
    ax2.set_xlabel("Contagem")
    ax2.set_title("Motivos de falha")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()
    print(f"\n  Figura: {OUT_FIG.relative_to(cfg.ROOT)}")

    print(f"\n{SEP}")
    print(f"S2 — {decision}")


if __name__ == "__main__":
    main()
