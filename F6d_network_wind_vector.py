"""
F6d_network_wind_vector.py — Vetor de deslocamento de nuvem IMPLÍCITO pela rede
================================================================================
F6 testou DIREÇÃO do vento (alinhamento vento-bearing) e F6b testou TEMPO
(atraso observado vs. dist/velocidade real), ambos par-a-par contra uma altura
de vento EXTERNA fixa por vez -- e deram nulo em todas as 4 alturas de camada
limite (10/100/200/500m). Aqui invertemos a pergunta: em vez de assumir uma
altura de vento e testar se ela explica os atrasos, deixamos os PRÓPRIOS DADOS
de potência revelarem que velocidade (direção + módulo) de deslocamento de
nuvem melhor explica TODOS os atrasos observados numa mesma janela de tempo,
simultaneamente -- a técnica clássica de "cloud motion vector a partir de rede
terrestre de sensores" (Bosch & Kleissl, Solar Energy, 2013).

Método (inversão por mínimos quadrados, por janela de tempo):
  Para cada par casado (i=station_ext, j=station_partner) numa mesma janela,
  com deslocamento planar (Δx_ij, Δy_ij) [m, aprox. equiretangular] e atraso
  observado dt_ij [s] (`dt_min` de `C1b_event_pairing.py`, já restrito a pares
  <5km):
      Δx_ij ≈ vx · dt_ij  +  erro
      Δy_ij ≈ vy · dt_ij  +  erro
  Resolvido por OLS-pela-origem, empilhando TODOS os pares ativos na janela:
      vx = Σ(dt_ij · Δx_ij) / Σ(dt_ij²)      vy = Σ(dt_ij · Δy_ij) / Σ(dt_ij²)
  Isso dá UM vetor de velocidade "que mais se encaixa" em todos os pares da
  janela ao mesmo tempo -- muito mais robusto a ruído de qualquer par
  individual do que a correlação par-a-par de F6b, e não depende de nenhuma
  fonte de vento externa. Janela: dia civil de t_ext (ver `cfg.F6D`), mínimo de
  5 pares casados por janela.

Comparação (Fase 2, depende de B7c/B7b): o vetor implícito (100% derivado dos
dados de potência) é comparado contra o vento REAL médio do dia em cada altura
disponível -- as 4 já testadas em F6/F6b (10/100/200/500m, camada limite) MAIS
as 8 novas de `B7c_wind_cerra_pressure.py` cobrindo nuvens baixas/médias/altas
(950-400 hPa, ~540-7190m). Roda para todas automaticamente, nenhuma assumida a
priori. A Fase 1 (cálculo do vetor implícito) não depende de nenhum vento
externo e roda mesmo antes do download de B7c terminar.

Saídas:
  - `data/processed/f6d_network_vector_windows.parquet` (Fase 1: um vetor implícito por dia)
  - `results/gates/f6d_network_vector_report.md` (Fase 1: estatísticas descritivas do vetor implícito)
  - `results/gates/f6d_wind_comparison_<altura>.md` (Fase 2: comparação com cada altura de vento)
  - `results/gates/f6d_wind_comparison.md` (Fase 2: tabela comparativa entre todas as alturas)
  - `results/figures/f6d_network_vector_examples.png`, `results/figures/f6d_wind_comparison.png`

Executar:
    python F6d_network_wind_vector.py
"""

from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result

ALIGNED_PQ = cfg.DIRS["processed"] / "aligned_pairs.parquet"
COORDS_PQ  = cfg.DIRS["interim"]   / "coords.parquet"
WIND_PQ    = cfg.DIRS["interim"]   / "wind_joined.parquet"

OUT_WINDOWS_PQ    = cfg.DIRS["processed"] / "f6d_network_vector_windows.parquet"
OUT_REPORT_MD     = cfg.DIRS["gates"]     / "f6d_network_vector_report.md"
OUT_EXAMPLES_FIG  = cfg.DIRS["figures"]   / "f6d_network_vector_examples.png"
OUT_COMPARISON_MD  = cfg.DIRS["gates"]   / "f6d_wind_comparison.md"
OUT_COMPARISON_FIG = cfg.DIRS["figures"] / "f6d_wind_comparison.png"

MIN_PAIRS   = cfg.F6D["min_pairs_per_window"]   # 5
MIN_SPEED   = cfg.F6D["min_speed_ms"]           # 1.0
N_BOOT      = cfg.F6D["n_bootstrap_windows"]    # 150
MIN_CORR    = cfg.F6D["min_corr_meaningful"]    # 0.10
CLOUD_CAT_BY_HPA = cfg.F6D["cloud_category_by_hpa"]
ALT_BY_HPA       = cfg.F6D["cerra_pressure_alt_m"]
SEED        = cfg.SEED

SEP = "─" * 60

# (speed_col, dir_col, label, suffix, altitude_m, category)
WIND_SOURCES = [
    ("wind_speed_ms",              "wind_dir_deg",              "KNMI De Bilt, 10m (superfície)", "10m",  10,   "camada limite"),
    ("cerra_wind_speed_ms_100m",   "cerra_wind_dir_deg_100m",   "CERRA, 100m",                     "100m", 100,  "camada limite"),
    ("cerra_wind_speed_ms_200m",   "cerra_wind_dir_deg_200m",   "CERRA, 200m",                     "200m", 200,  "camada limite"),
    ("cerra_wind_speed_ms_500m",   "cerra_wind_dir_deg_500m",   "CERRA, 500m",                     "500m", 500,  "camada limite"),
] + [
    (f"cerra_wind_speed_ms_{p}hPa", f"cerra_wind_dir_deg_{p}hPa",
     f"CERRA, {p}hPa (~{ALT_BY_HPA[p]}m)", f"{p}hPa", ALT_BY_HPA[p], CLOUD_CAT_BY_HPA[p])
    for p in cfg.F6D["cerra_pressure_hpa"]
]


def planar_xy(lat: np.ndarray, lon: np.ndarray, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    """Projeção equiretangular simples (m), válida para a extensão pequena
    (~44x50km) da rede Utrecht -- mesma aproximação usada em outras partes do
    pipeline para bearing/distância."""
    x = np.radians(lon) * 111_320.0 * np.cos(np.radians(lat0))
    y = np.radians(lat) * 110_540.0
    return x, y


def circular_mean_deg(deg: np.ndarray) -> float:
    rad = np.radians(deg)
    return float(np.degrees(np.arctan2(np.nanmean(np.sin(rad)), np.nanmean(np.cos(rad)))) % 360.0)


def circular_corr(a_deg: np.ndarray, b_deg: np.ndarray) -> float:
    """Correlação circular-circular de Jammalamadaka & SenGupta (1988)."""
    a, b = np.radians(a_deg), np.radians(b_deg)
    a_bar, b_bar = np.arctan2(np.sin(a).mean(), np.cos(a).mean()), np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    sa, sb = np.sin(a - a_bar), np.sin(b - b_bar)
    denom = np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
    return float((sa * sb).sum() / denom) if denom > 0 else np.nan


def compute_implied_vectors(aligned: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Fase 1 -- inversão por mínimos quadrados, uma janela (dia civil) por vez."""
    d = aligned[aligned["matched"] & aligned["dt_min"].notna()].copy()
    have_coords = d["station_ext"].isin(coords.index) & d["station_partner"].isin(coords.index)
    d = d.loc[have_coords].copy()
    d["t_ext"] = pd.to_datetime(d["t_ext"], utc=True)
    print(f"  Pares casados com coordenadas válidas: {len(d):,}")

    lat0 = float(coords["lat_centroid"].mean())
    x1, y1 = planar_xy(coords.loc[d["station_ext"], "lat_centroid"].to_numpy(),
                        coords.loc[d["station_ext"], "lon_centroid"].to_numpy(), lat0)
    x2, y2 = planar_xy(coords.loc[d["station_partner"], "lat_centroid"].to_numpy(),
                        coords.loc[d["station_partner"], "lon_centroid"].to_numpy(), lat0)
    d["dx_m"] = x2 - x1
    d["dy_m"] = y2 - y1
    d["dt_sec"] = d["dt_min"].to_numpy() * 60.0
    d["date"] = d["t_ext"].dt.date

    rows = []
    for day, g in d.groupby("date"):
        n = len(g)
        if n < MIN_PAIRS:
            continue
        dt = g["dt_sec"].to_numpy()
        dx = g["dx_m"].to_numpy()
        dy = g["dy_m"].to_numpy()
        denom = float(np.sum(dt ** 2))
        if denom <= 0:
            continue
        vx = float(np.sum(dt * dx) / denom)
        vy = float(np.sum(dt * dy) / denom)
        dx_hat, dy_hat = vx * dt, vy * dt
        ss_res = float(np.sum((dx - dx_hat) ** 2) + np.sum((dy - dy_hat) ** 2))
        ss_tot = float(np.sum((dx - dx.mean()) ** 2) + np.sum((dy - dy.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        speed = float(np.hypot(vx, vy))
        travel_to_deg = float(np.degrees(np.arctan2(vx, vy)) % 360.0)   # p/ onde o "ar" viaja
        dir_from_deg = (travel_to_deg + 180.0) % 360.0                   # convenção meteorológica
        rows.append({
            "date": day, "n_pairs": n, "vx_ms": vx, "vy_ms": vy,
            "speed_implied_ms": speed, "dir_implied_from_deg": dir_from_deg, "r2": r2,
        })
    out = pd.DataFrame(rows)
    return out


def build_daily_wind(wind: pd.DataFrame, speed_col: str, dir_col: str) -> pd.DataFrame | None:
    """Vento representativo do dia (fonte regional de ponto único, reamostrada
    em cada evento de rampa) -- média circular de direção + média de módulo,
    a partir de todas as amostras disponíveis naquele dia civil em
    `wind_joined.parquet` (reaproveita o casamento já feito por B7/B7b/B7c)."""
    if speed_col not in wind.columns or dir_col not in wind.columns:
        return None
    w = wind[["start_ts", speed_col, dir_col]].dropna()
    if w.empty:
        return None
    w = w.copy()
    w["start_ts"] = pd.to_datetime(w["start_ts"], utc=True)
    w["date"] = w["start_ts"].dt.date
    out = w.groupby("date").agg(
        speed_wind_ms=(speed_col, "mean"),
        dir_wind_deg=(dir_col, circular_mean_deg),
        n_samples=(speed_col, "size"),
    ).reset_index()
    return out


def run_for_source(windows: pd.DataFrame, wind: pd.DataFrame,
                    speed_col: str, dir_col: str, label: str, suffix: str,
                    altitude_m: float, category: str) -> dict | None:
    daily_wind = build_daily_wind(wind, speed_col, dir_col)
    if daily_wind is None:
        print(f"  AVISO: '{speed_col}'/'{dir_col}' não encontradas ou vazias — pulando {label}.")
        return None

    m = windows.merge(daily_wind, on="date", how="inner")
    m = m[m["speed_implied_ms"] >= MIN_SPEED].copy()
    n = len(m)
    if n < 10:
        print(f"  AVISO: apenas {n} janelas utilizáveis para {label} (< 10) — pulando.")
        return None

    circ_r = circular_corr(m["dir_implied_from_deg"].to_numpy(), m["dir_wind_deg"].to_numpy())
    cos_align = np.cos(np.radians(m["dir_implied_from_deg"].to_numpy() - m["dir_wind_deg"].to_numpy()))
    mean_align = float(np.mean(cos_align))
    pearson_r, pearson_p = st.pearsonr(m["speed_implied_ms"], m["speed_wind_ms"])

    rng = np.random.default_rng(SEED)
    boot_circ, boot_align = [], []
    idx_all = np.arange(n)
    for _ in range(N_BOOT):
        idx_b = rng.choice(idx_all, size=n, replace=True)
        boot_circ.append(circular_corr(m["dir_implied_from_deg"].to_numpy()[idx_b], m["dir_wind_deg"].to_numpy()[idx_b]))
        boot_align.append(float(np.mean(cos_align[idx_b])))
    boot_circ = np.array(boot_circ)[np.isfinite(boot_circ)]
    boot_align = np.array(boot_align)
    ci_circ = np.percentile(boot_circ, [2.5, 97.5]) if len(boot_circ) else (np.nan, np.nan)
    ci_align = np.percentile(boot_align, [2.5, 97.5])

    significant = not (ci_align[0] < 0 < ci_align[1])
    meaningful = abs(mean_align) > MIN_CORR
    if significant and meaningful:
        decision = f"SINAL DE ADVECÇÃO DETECTADO ({label}) — vetor implícito alinha com o vento real"
    elif significant and not meaningful:
        decision = f"SEM SINAL PRATICAMENTE RELEVANTE ({label}) — alinhamento detectável mas trivial"
    else:
        decision = f"SEM SINAL DE ADVECÇÃO DETECTÁVEL ({label})"
    print(f"  {label:35s} n={n:4d}  align={mean_align:+.4f} [{ci_align[0]:+.4f},{ci_align[1]:+.4f}]  "
          f"circ_r={circ_r:+.4f}  speed_r={pearson_r:+.4f}  -> {decision}")

    OUT_MODEL = cfg.DIRS["gates"] / f"f6d_wind_comparison_{suffix}.md"
    OUT_MODEL.write_text(f"""# F6d — Vetor Implícito vs. Vento Real ({label})

**Data:** {date.today().isoformat()}
**Categoria de nuvem:** {category} (altitude aprox. {altitude_m}m)

## Amostra
{n:,} janelas (dias) com vetor implícito não-degenerado (speed >= {MIN_SPEED} m/s) e vento
médio do dia disponível nesta altura, de {len(windows):,} janelas totais com >= {MIN_PAIRS}
pares casados.

## Resultados
| Métrica | Valor |
|---|---|
| Correlação circular-circular (direção) | {circ_r:.4f} (IC95% [{ci_circ[0]:.4f}, {ci_circ[1]:.4f}]) |
| Alinhamento médio cos(Δdireção) | {mean_align:+.4f} (IC95% [{ci_align[0]:+.4f}, {ci_align[1]:+.4f}]) |
| Correlação de módulo (Pearson) | {pearson_r:.4f} (p={pearson_p:.4f}) |

Limiar de relevância prática: |alinhamento médio| > {MIN_CORR:.2f}.

## Interpretação
{'Alinhamento estatisticamente diferente de zero E de tamanho de efeito relevante -- evidência de que o vetor de nuvem implícito pela rede (100% derivado dos dados de potência) é consistente com o vento real nesta altura.' if (significant and meaningful) else 'Alinhamento não distinguível de zero ou de tamanho de efeito trivial nesta altura.'}
**Decisão: {decision}**

## Referência cruzada
- Comparação entre todas as alturas: `results/gates/f6d_wind_comparison.md`
- Vetor implícito (Fase 1, sem vento): `results/gates/f6d_network_vector_report.md`
""")

    log_result(
        script="F6d_network_wind_vector.py", gate="", phase="F6d",
        params={"wind_source": label, "wind_height": suffix, "altitude_m": altitude_m,
                "cloud_category": category, "min_pairs_per_window": MIN_PAIRS,
                "min_speed_ms": MIN_SPEED, "n_bootstrap_windows": N_BOOT},
        results={"n_windows": n, "circular_corr": round(circ_r, 4) if np.isfinite(circ_r) else None,
                 "mean_alignment": round(mean_align, 4), "align_ci_low": round(float(ci_align[0]), 4),
                 "align_ci_high": round(float(ci_align[1]), 4), "speed_pearson_r": round(float(pearson_r), 4),
                 "speed_pearson_p": round(float(pearson_p), 4), "meaningful": bool(meaningful)},
        decision=decision,
        action=f"Network-implied cloud velocity vector (LSQ inversion over matched pairs per day) "
               f"compared against real wind at height {suffix} ({label}, category: {category}).",
        interpretation=(
            f"Wind source: {label} ({category}, ~{altitude_m}m). Circular corr={circ_r:.4f}, "
            f"mean alignment={mean_align:+.4f} (CI [{ci_align[0]:+.4f},{ci_align[1]:+.4f}]), "
            f"speed Pearson r={pearson_r:.4f} (p={pearson_p:.4f}). {decision}."
        ),
        paper_ref="Section 8 (F6 spatial structure) -- F6d network-implied cloud vector vs. real wind",
    )

    return {
        "label": label, "suffix": suffix, "altitude_m": altitude_m, "category": category,
        "n": n, "circ_r": circ_r, "mean_align": mean_align, "ci_low": ci_align[0], "ci_high": ci_align[1],
        "speed_r": pearson_r, "speed_p": pearson_p, "significant": significant, "meaningful": meaningful,
        "decision": decision,
    }


def main() -> None:
    print(SEP)
    print("F6d — VETOR DE DESLOCAMENTO DE NUVEM IMPLÍCITO PELA REDE (LSQ, Bosch & Kleissl)")
    print(SEP)

    for p in (ALIGNED_PQ, COORDS_PQ):
        if not p.exists():
            print(f"\nERRO: {p} não encontrado.")
            sys.exit(1)

    aligned = pd.read_parquet(ALIGNED_PQ)
    coords = pd.read_parquet(COORDS_PQ).dropna(subset=["lat_centroid", "lon_centroid"]).set_index("station_id")

    print("\n[Fase 1] Inversão por mínimos quadrados, por janela (dia civil)...")
    windows = compute_implied_vectors(aligned, coords)
    windows.to_parquet(OUT_WINDOWS_PQ, index=False)
    n_total_days = windows["date"].nunique() if len(windows) else 0
    n_usable = int((windows["speed_implied_ms"] >= MIN_SPEED).sum()) if len(windows) else 0
    print(f"  Janelas com >= {MIN_PAIRS} pares casados: {n_total_days:,}")
    print(f"  ...das quais com vetor não-degenerado (speed >= {MIN_SPEED} m/s): {n_usable:,} "
          f"({n_usable/n_total_days:.1%})" if n_total_days else "  Nenhuma janela utilizável.")
    if len(windows):
        print(f"  Velocidade implícita: mediana={windows['speed_implied_ms'].median():.2f} m/s, "
              f"R² mediano do ajuste={windows['r2'].median():.3f}")

    OUT_REPORT_MD.write_text(f"""# F6d (Fase 1) — Vetor de Deslocamento de Nuvem Implícito pela Rede

**Data:** {date.today().isoformat()}

## Método
Para cada dia civil com >= {MIN_PAIRS} pares de estações casados (`aligned_pairs.parquet`,
restrito a pares <5km, mesmo conjunto do Gate G1), resolve-se por mínimos quadrados (OLS
pela origem) um único vetor de velocidade (vx, vy) que melhor explica TODOS os
deslocamentos (Δx,Δy) vs. atrasos observados (dt) da janela simultaneamente -- técnica de
"cloud motion vector" a partir de rede terrestre de sensores (Bosch & Kleissl, 2013). Não
depende de nenhuma fonte de vento externa.

## Amostra
- Dias com >= {MIN_PAIRS} pares casados: **{n_total_days:,}**
- ...dos quais com vetor não-degenerado (módulo >= {MIN_SPEED} m/s): **{n_usable:,}** ({n_usable/n_total_days:.1%} se houver dias)
- Velocidade implícita mediana: {windows['speed_implied_ms'].median():.2f} m/s (dias utilizáveis)
- R² mediano do ajuste por janela: {windows['r2'].median():.3f}

## Próximo passo
Ver `results/gates/f6d_wind_comparison.md` para a comparação deste vetor implícito contra
o vento real em cada altura disponível (camada limite + níveis de pressão de nuvem).

## Referência cruzada
- Dados: `data/processed/f6d_network_vector_windows.parquet`
""" if n_total_days else "# F6d (Fase 1) — sem janelas utilizáveis (ver aligned_pairs.parquet).\n")

    if len(windows) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].hist(windows["speed_implied_ms"], bins=40, color="#2c7bb6")
        axes[0].axvline(MIN_SPEED, color="crimson", linestyle="--", label=f"limiar não-degenerado ({MIN_SPEED} m/s)")
        axes[0].set_xlabel("Velocidade implícita (m/s)")
        axes[0].set_ylabel("Nº de janelas (dias)")
        axes[0].legend(fontsize=8)
        usable = windows[windows["speed_implied_ms"] >= MIN_SPEED]
        if len(usable):
            theta = np.radians(usable["dir_implied_from_deg"])
            ax_polar = plt.subplot(1, 2, 2, projection="polar")
            ax_polar.hist(theta, bins=24, color="#2c7bb6")
            ax_polar.set_theta_zero_location("N")
            ax_polar.set_theta_direction(-1)
            ax_polar.set_title("Direção implícita (de onde 'vem')", fontsize=9)
        plt.tight_layout()
        plt.savefig(OUT_EXAMPLES_FIG, dpi=150)
        plt.close()
        print(f"  Figura: {OUT_EXAMPLES_FIG.relative_to(cfg.ROOT)}")

    log_result(
        script="F6d_network_wind_vector.py", gate="", phase="F6d",
        params={"window": cfg.F6D["window"], "min_pairs_per_window": MIN_PAIRS, "min_speed_ms": MIN_SPEED},
        results={"n_windows_total": n_total_days, "n_windows_usable": n_usable,
                 "median_speed_implied_ms": round(float(windows["speed_implied_ms"].median()), 3) if n_total_days else None,
                 "median_r2": round(float(windows["r2"].median()), 3) if n_total_days else None},
        decision="N/A — Fase 1 é descritiva (cálculo do vetor implícito), decisão vem da Fase 2 (comparação com vento)",
        action="Computed network-implied cloud displacement vector via per-day LSQ inversion over "
               "all matched close pairs (<5km), independent of any external wind source.",
        interpretation="See f6d_wind_comparison.md (Phase 2) for the comparison against real wind "
                       "at each available height.",
        paper_ref="Section 8 (F6 spatial structure) -- F6d network-implied cloud vector, Phase 1",
    )

    if not WIND_PQ.exists():
        print(f"\n{SEP}\nFase 2 pulada: {WIND_PQ} não encontrado.")
        return
    wind = pd.read_parquet(WIND_PQ)

    print(f"\n{SEP}\n[Fase 2] Comparando vetor implícito com vento real, por altura...\n{SEP}")
    results = []
    for speed_col, dir_col, label, suffix, altitude_m, category in WIND_SOURCES:
        r = run_for_source(windows, wind, speed_col, dir_col, label, suffix, altitude_m, category)
        if r is not None:
            results.append(r)

    if not results:
        print("\nNenhuma altura de vento pôde ser comparada ainda (B7c pode não ter terminado).")
        return

    comp_df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(results) + 2))
    y_pos = np.arange(len(results))
    aligns = [r["mean_align"] for r in results]
    errs_low = [r["mean_align"] - r["ci_low"] for r in results]
    errs_high = [r["ci_high"] - r["mean_align"] for r in results]
    colors = ["crimson" if r["significant"] and r["meaningful"] else "#2c7bb6" for r in results]
    ax.errorbar(aligns, y_pos, xerr=[errs_low, errs_high], fmt="o", capsize=4, color="black", ecolor="grey")
    for i, r in enumerate(results):
        ax.scatter([r["mean_align"]], [i], color=colors[i], s=80, zorder=3)
    ax.axvline(0, color="grey", lw=1, linestyle=":")
    ax.axvline(MIN_CORR, color="green", lw=1, linestyle="--", alpha=0.5)
    ax.axvline(-MIN_CORR, color="green", lw=1, linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['label']} ({r['category']})" for r in results])
    ax.set_xlabel("Alinhamento médio cos(Δdireção) vetor implícito vs. vento real — IC95% bootstrap")
    ax.set_title("F6d — vetor implícito pela rede vs. vento real, por altura")
    plt.tight_layout()
    plt.savefig(OUT_COMPARISON_FIG, dpi=150)
    plt.close()
    print(f"\n  Figura comparativa: {OUT_COMPARISON_FIG.relative_to(cfg.ROOT)}")

    any_meaningful = any(r["significant"] and r["meaningful"] for r in results)
    cats = {"baixa": "Nuvens baixas (<2000m)", "média": "Nuvens médias (2000-6000m)",
            "alta": "Nuvens altas (>6000m)", "camada limite": "Camada limite (10-500m, já testado em F6/F6b)"}
    rows_md = "\n".join(
        f"| {r['label']} | {r['category']} | {r['n']:,} | {r['circ_r']:.4f} | {r['mean_align']:+.4f} | "
        f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['speed_r']:.4f} | {'Sim' if r['meaningful'] else 'Não'} | {r['decision']} |"
        for r in results
    )
    OUT_COMPARISON_MD.write_text(f"""# F6d — Comparação do Vetor Implícito pela Rede vs. Vento Real, por Altura

**Data:** {date.today().isoformat()}

Vetor de deslocamento de nuvem estimado 100% a partir dos dados de potência (inversão por
mínimos quadrados, Fase 1: `f6d_network_vector_report.md`), comparado contra o vento real
médio do dia em cada altura disponível -- camada limite (10-500m, já testado em F6/F6b com
metodologia par-a-par) MAIS níveis de pressão cobrindo nuvens baixas/médias/altas
(950-400 hPa, ~540-7190m, `B7c_wind_cerra_pressure.py`).

| Fonte de vento | Categoria | n janelas | Corr. circular | Alinhamento médio | IC95% | Corr. módulo | Relevante? | Decisão |
|---|---|---|---|---|---|---|---|---|
{rows_md}

## Conclusão
{'Pelo menos uma altura mostrou alinhamento estatisticamente significativo E praticamente relevante entre o vetor implícito pela rede e o vento real -- evidência NOVA e mais forte de advecção física, reportar no paper qual altura/categoria de nuvem e com que magnitude.' if any_meaningful else 'NENHUMA altura testada (camada limite nem níveis de pressão de nuvens baixas/médias/altas) mostrou alinhamento praticamente relevante entre o vetor de deslocamento implícito pela rede e o vento real. Isso é o teste mais forte e mais direto tentado até agora -- não depende de assumir uma altura a priori, nem de comparações par-a-par ruidosas -- e reforça ainda mais a conclusão de regime compartilhado (Gate G1, C1b, F6, F6b): a dependência de cauda observada não parece decorrer de um mecanismo de advecção física cronometrável e direcionalmente coerente com o vento ambiente, em nenhuma altura ou categoria de nuvem disponível publicamente para a região.'}

## Referência cruzada
- Fase 1 (vetor implícito, sem vento): `results/gates/f6d_network_vector_report.md`
- Modelos individuais: `results/gates/f6d_wind_comparison_<altura>.md`
- Fig.: `results/figures/f6d_wind_comparison.png`
- Ver também: `results/gates/f6_anisotropy_comparison.md`, `results/gates/f6b_timing_comparison.md`
  (testes par-a-par anteriores, mesma conclusão)
""")
    print(f"  Salvo: {OUT_COMPARISON_MD.relative_to(cfg.ROOT)}")

    print(f"\n{SEP}\nF6d (todas as alturas) — resumo:")
    for r in results:
        print(f"  {r['label']:35s} ({r['category']:14s}) → {r['decision']}")


if __name__ == "__main__":
    main()
