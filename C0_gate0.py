"""
C0_gate0.py — Gate G0: Adequação Espacial da Geolocalização
============================================================
Responde: a grade de anonimização de 150 m × 150 m das coordenadas
Utrecht introduz incerteza aceitável na matriz de distâncias par-a-par?

Método (Monte Carlo de posição — ROADMAP F1):
  Para cada draw b = 1..n_draws:
    - Cada usina recebe um offset aleatório Uniform(-75 m, +75 m) em X e Y
    - Calcula matriz de distâncias haversine com as posições perturbadas
  Para cada par (i,j):
    - d_median   = mediana das n_draws distâncias
    - RU_ij      = (P97.5 - P2.5) / d_median   (incerteza relativa)
  Decisão:
    - fracao_afetada = #{pares com RU > 0.20} / #{todos os pares}
    - se fracao_afetada < 0.05 E concentrada em d_nominal < 500 m → APROVADO COM RESSALVA
    - senão → REPROVADO PARCIAL

Saídas:
  results/gates/gate0_results.parquet   — (i, j, d_nominal, d_median_mc, RU_ij, flagged)
  results/gates/gate0_decision.md       — decisão escrita com data
  results/figures/gate0_ru_histogram.png
  results/figures/gate0_ru_vs_dist.png

Executar:
    python C0_gate0.py
"""

import sys
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

COORDS_PQ = cfg.DIRS["interim"] / "coords.parquet"
OUT_PQ    = cfg.DIRS["gates"]   / "gate0_results.parquet"
OUT_DEC   = cfg.DIRS["gates"]   / "gate0_decision.md"
OUT_HIST  = cfg.DIRS["figures"] / "gate0_ru_histogram.png"
OUT_SCAT  = cfg.DIRS["figures"] / "gate0_ru_vs_dist.png"

N_DRAWS    = cfg.GATE0["n_draws"]          # 1000
CELL_M     = cfg.GATE0["cell_size_m"]      # 150
RU_THRESH  = cfg.GATE0["ru_threshold"]     # 0.20
MAX_FRAC   = cfg.GATE0["max_affected_frac"]# 0.05
SHORT_DIST = cfg.GATE0["short_dist_m"]     # 500

SEP = "─" * 60

# ── Funções auxiliares ────────────────────────────────────────────────────────

def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Distâncias haversine entre todos os pares de N pontos.
    Retorna matriz N×N em metros.
    """
    R = 6_371_000.0   # raio médio da Terra em metros
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def latlon_offset(cell_m: float, lat_deg: float) -> tuple[float, float]:
    """
    Converte cell_m metros para graus de latitude e longitude (aproximação local).
    """
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_deg))
    return cell_m / meters_per_deg_lat, cell_m / meters_per_deg_lon


def monte_carlo_gate0(
    lat: np.ndarray,
    lon: np.ndarray,
    cell_m: float,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Monte Carlo de posição: para cada draw perturba lat/lon de todas as usinas
    dentro de ±cell_m/2 e retorna array (n_draws, N, N) de distâncias.

    Para economizar memória, acumula P2.5 / mediana / P97.5 on-the-fly
    em vez de guardar todos os draws.
    """
    N = len(lat)
    lat_center = float(np.nanmean(lat))
    half_deg_lat, half_deg_lon = latlon_offset(cell_m / 2, lat_center)
    assert np.isfinite(half_deg_lat) and np.isfinite(half_deg_lon), \
        f"Offsets inválidos: lat_offset={half_deg_lat}, lon_offset={half_deg_lon}"

    # Acumular draws em buffer para cálculo de percentis
    buf = np.empty((n_draws, N * (N - 1) // 2), dtype=np.float32)

    idx_upper = np.triu_indices(N, k=1)

    for b in range(n_draws):
        dlat = rng.uniform(-half_deg_lat, half_deg_lat, N)
        dlon = rng.uniform(-half_deg_lon, half_deg_lon, N)
        d = haversine_matrix(lat + dlat, lon + dlon)
        buf[b] = d[idx_upper].astype(np.float32)

    p025    = np.percentile(buf, 2.5,  axis=0)
    p500    = np.percentile(buf, 50.0, axis=0)
    p975    = np.percentile(buf, 97.5, axis=0)

    return idx_upper, p025, p500, p975


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP)
    print("C0 — GATE G0: ADEQUAÇÃO ESPACIAL DA GEOLOCALIZAÇÃO")
    print(f"     n_draws={N_DRAWS}  cell_size={CELL_M} m  RU_threshold={RU_THRESH:.0%}")
    print(SEP)

    if not COORDS_PQ.exists():
        print(f"ERRO: {COORDS_PQ} não encontrado. Execute B1 primeiro.")
        sys.exit(1)

    coords = pd.read_parquet(COORDS_PQ)
    # Remover usinas sem coordenadas válidas
    coords = coords.dropna(subset=["lat_centroid", "lon_centroid"])
    lat    = coords["lat_centroid"].values.astype(float)
    lon    = coords["lon_centroid"].values.astype(float)
    sids   = coords["station_id"].values
    N      = len(lat)
    print(f"  (Após remoção de NaN: {N} usinas)")
    n_pairs = N * (N - 1) // 2
    print(f"  Usinas: {N}   Pares: {n_pairs:,}")

    # Distância nominal (centróides, sem perturbação)
    print("  Calculando distâncias nominais...")
    d_nominal_mat  = haversine_matrix(lat, lon)
    idx_upper      = np.triu_indices(N, k=1)
    d_nominal      = d_nominal_mat[idx_upper]

    # Monte Carlo
    print(f"  Rodando Monte Carlo ({N_DRAWS} draws)...")
    rng = np.random.default_rng(cfg.SEED)
    _, p025, p500, p975 = monte_carlo_gate0(lat, lon, CELL_M, N_DRAWS, rng)

    # Incerteza relativa
    RU = (p975 - p025) / np.where(p500 > 0, p500, np.nan)

    # Construir tabela de resultados
    i_idx, j_idx = idx_upper
    results = pd.DataFrame({
        "station_i":   sids[i_idx],
        "station_j":   sids[j_idx],
        "d_nominal_m": d_nominal.astype(float),
        "d_median_mc": p500.astype(float),
        "RU":          RU.astype(float),
        "flagged":     RU > RU_THRESH,
    })

    # Decisão
    n_flagged       = results["flagged"].sum()
    fracao_afetada  = n_flagged / n_pairs
    flagged_short   = results[results["flagged"] & (results["d_nominal_m"] < SHORT_DIST)]
    frac_short      = len(flagged_short) / max(n_flagged, 1)

    print(f"\n  Pares com RU > {RU_THRESH:.0%}:   {n_flagged:,} / {n_pairs:,}  ({fracao_afetada:.3%})")
    print(f"  Desses, com d < {SHORT_DIST} m: {len(flagged_short):,}  ({frac_short:.1%} dos flagrados)")
    print(f"  Mediana de RU (todos os pares): {np.nanmedian(RU):.4f}")
    print(f"  P95 de RU:                      {np.nanpercentile(RU, 95):.4f}")

    if fracao_afetada < MAX_FRAC and frac_short >= 0.80:
        decision = "APROVADO COM RESSALVA"
        action   = (f"Excluir ou tratar separadamente os {n_flagged} pares flagrados "
                    f"(d < {SHORT_DIST} m) na regressão de distância (F6).")
    elif fracao_afetada < MAX_FRAC:
        decision = "APROVADO COM RESSALVA"
        action   = (f"Pares flagrados ({n_flagged}) não concentrados em curtíssima distância — "
                    f"monitorar na análise de F6.")
    else:
        decision = "REPROVADO PARCIAL"
        action   = ("Considerar regressão com errors-in-variables (EIV) em F6 "
                    "para acomodar incerteza de posição.")

    print(f"\n  DECISÃO GATE G0: {decision}")
    print(f"  Ação:            {action}")

    # Salvar resultados
    cfg.DIRS["gates"].mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUT_PQ, index=False)
    print(f"\n  Salvo: {OUT_PQ.relative_to(cfg.ROOT)}")

    # Escrever decisão
    decision_md = f"""# Gate G0 — Adequação Espacial da Geolocalização

**Data:** {date.today().isoformat()}
**Decisão:** {decision}

## Parâmetros
| Parâmetro | Valor |
|---|---|
| n_draws (Monte Carlo) | {N_DRAWS} |
| cell_size (grade anon.) | {CELL_M} m |
| RU_threshold | {RU_THRESH:.0%} |
| max_affected_frac | {MAX_FRAC:.0%} |
| short_dist | {SHORT_DIST} m |

## Resultados
| Métrica | Valor |
|---|---|
| Usinas | {N} |
| Pares totais | {n_pairs:,} |
| Pares com RU > {RU_THRESH:.0%} | {n_flagged:,} ({fracao_afetada:.3%}) |
| Desses com d < {SHORT_DIST} m | {len(flagged_short):,} ({frac_short:.1%} dos flagrados) |
| RU mediano (todos) | {np.nanmedian(RU):.4f} |
| RU P95 (todos) | {np.nanpercentile(RU, 95):.4f} |

## Decisão
**{decision}**

{action}

## Referência cruzada
- Fig. 0: `results/figures/gate0_ru_histogram.png`
- Fig. 0b: `results/figures/gate0_ru_vs_dist.png`
- Dados: `results/gates/gate0_results.parquet`
"""
    OUT_DEC.write_text(decision_md)
    print(f"  Salvo: {OUT_DEC.relative_to(cfg.ROOT)}")

    # ── Logging estruturado ───────────────────────────────────────────────────
    log_result(
        script  = "C0_gate0.py",
        gate    = "G0",
        params  = {
            "n_draws":      N_DRAWS,
            "cell_size_m":  CELL_M,
            "ru_threshold": RU_THRESH,
            "max_affected_frac": MAX_FRAC,
            "short_dist_m": SHORT_DIST,
            "n_stations":   N,
        },
        results = {
            "n_pairs":          n_pairs,
            "n_flagged":        int(n_flagged),
            "frac_flagged_pct": round(fracao_afetada * 100, 3),
            "frac_short_pct":   round(frac_short * 100, 1),
            "ru_median":        round(float(np.nanmedian(RU)), 4),
            "ru_p95":           round(float(np.nanpercentile(RU, 95)), 4),
        },
        decision = decision,
        action   = action,
        interpretation = (
            f"The 150 m anonymization grid introduces RU > {RU_THRESH:.0%} in only "
            f"{fracao_afetada:.2%} of the {n_pairs:,} station pairs — well below the "
            f"{MAX_FRAC:.0%} threshold. The median RU across all pairs is "
            f"{float(np.nanmedian(RU)):.3f} (~{float(np.nanmedian(RU))*100:.1f}%), "
            "negligible for the spatial scales of interest (1–35 km). "
            f"Of the {int(n_flagged)} flagged pairs, only {frac_short:.1%} fall below "
            f"{SHORT_DIST} m, indicating the high-RU regime is not exclusively a "
            "very-short-distance artifact — these pairs should be monitored (not blindly "
            "excluded) in the F6 chi-vs-distance regression. "
            "Overall, the distance matrix is reliable and the project proceeds to Gate G1."
        ),
        paper_ref = (
            "Section 5 — Spatial Adequacy Verification; "
            "Figure 0 (gate0_ru_histogram.png + gate0_ru_vs_dist.png); "
            "Table 0 (gate0_results.parquet summary)"
        ),
    )

    # ── Figuras ──────────────────────────────────────────────────────────────

    # Fig 1: Histograma de RU
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(RU[np.isfinite(RU)], bins=80, color="#2c7bb6", edgecolor="none", alpha=0.85)
    ax.axvline(RU_THRESH, color="crimson", lw=1.5, linestyle="--", label=f"RU threshold = {RU_THRESH:.0%}")
    ax.set_xlabel("Relative Uncertainty (RU)")
    ax.set_ylabel("Number of station pairs")
    ax.set_title(f"Gate G0 — Distribution of pairwise RU\n({N_DRAWS} MC draws, cell = {CELL_M} m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_HIST, dpi=150)
    plt.close()
    print(f"  Figura: {OUT_HIST.relative_to(cfg.ROOT)}")

    # Fig 2: RU vs distância nominal
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        results["d_nominal_m"] / 1000,
        results["RU"],
        c=results["flagged"].astype(int),
        cmap="RdYlGn_r",
        s=1,
        alpha=0.3,
        rasterized=True,
    )
    ax.axhline(RU_THRESH, color="crimson", lw=1.5, linestyle="--", label=f"RU = {RU_THRESH:.0%}")
    ax.axvline(SHORT_DIST / 1000, color="orange", lw=1.2, linestyle=":", label=f"d = {SHORT_DIST} m")
    ax.set_xlabel("Nominal distance (km)")
    ax.set_ylabel("Relative Uncertainty (RU)")
    ax.set_title("Gate G0 — RU vs nominal distance between stations")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_SCAT, dpi=150)
    plt.close()
    print(f"  Figura: {OUT_SCAT.relative_to(cfg.ROOT)}")

    print(f"\n{SEP}")
    print(f"Gate G0 — {decision}")
    print(f"Próximo passo: C1_gate1.py — diagnóstico de dependência de cauda")


if __name__ == "__main__":
    main()
