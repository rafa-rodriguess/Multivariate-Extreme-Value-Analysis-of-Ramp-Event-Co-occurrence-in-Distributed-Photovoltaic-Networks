"""
A0_config.py — Validar e exibir configuração central
=====================================================
Passo zero do pipeline: importa src/config.py, verifica que todos os valores
estão dentro dos limites esperados e imprime um sumário legível.

Executar:
    python A0_config.py

Critério de pronto:
    - Nenhum AssertionError
    - Saída impressa sem erros
    - chi teórico de Gumbel θ=2 exibido corretamente (≈ 0.5858)
"""

import sys
from pathlib import Path

# Garante que src/ está no path mesmo rodando da raiz do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg


def validate_config() -> None:
    """Verifica limites e coerência interna dos parâmetros."""

    # Seed
    assert isinstance(cfg.RANDOM_SEED, int) and cfg.RANDOM_SEED >= 0, \
        "RANDOM_SEED deve ser inteiro não-negativo"

    # Utrecht
    assert cfg.UTRECHT["n_stations"] == 175
    assert cfg.UTRECHT["grid_cell_m"] == 150
    assert cfg.UTRECHT["resolution_min"] == 1

    # Ramp
    assert cfg.RAMP["delta"] in cfg.RAMP["sensitivity_deltas"], \
        "delta base deve estar na grade de sensibilidade"
    assert cfg.RAMP["compression_eps"] in cfg.RAMP["sensitivity_eps"], \
        "compression_eps base deve estar na grade de sensibilidade"

    # Gate G0
    assert 0 < cfg.GATE0["ru_threshold"] < 1
    assert 0 < cfg.GATE0["max_affected_frac"] < 1
    assert cfg.GATE0["n_draws"] >= 500, "n_draws G0 deve ser >= 500"

    # Gate G1
    assert cfg.GATE1["go_threshold"] > 0
    assert cfg.GATE1["fdr_alpha"] < 0.10
    assert cfg.GATE1["n_bootstrap_pilot"] >= 50

    # Sintéticos
    assert cfg.SYNTHETIC["n_samples"] >= 10_000
    assert 1.0 < cfg.SYNTHETIC["gumbel_theta"]
    # χ teórico de Gumbel: 2 - 2^(1/θ)
    import math
    chi_expected = 2 - 2 ** (1 / cfg.SYNTHETIC["gumbel_theta"])
    assert abs(cfg.SYNTHETIC["gumbel_chi_theory"] - chi_expected) < 1e-10, \
        f"gumbel_chi_theory incorreto: {cfg.SYNTHETIC['gumbel_chi_theory']:.6f} ≠ {chi_expected:.6f}"

    # Gate G0 analítico
    import math
    ru_expected = math.sqrt(2) * cfg.SYNTHETIC["gate0_cell_size_m"] / cfg.SYNTHETIC["gate0_d_nominal_m"]
    assert abs(cfg.SYNTHETIC["gate0_ru_analytic"] - ru_expected) < 1e-10, \
        f"gate0_ru_analytic incorreto: {cfg.SYNTHETIC['gate0_ru_analytic']:.6f} ≠ {ru_expected:.6f}"


def print_summary() -> None:
    """Imprime sumário legível da configuração."""
    sep = "─" * 60

    print(sep)
    print("CONFIGURAÇÃO DO PROJETO")
    print(sep)
    print(f"  Raiz         : {cfg.ROOT}")
    print(f"  Seed global  : {cfg.RANDOM_SEED}")
    print()

    print("  UTRECHT")
    print(f"    Estações     : {cfg.UTRECHT['n_stations']}")
    print(f"    Resolução    : {cfg.UTRECHT['resolution_min']} min")
    print(f"    Treino       : {cfg.UTRECHT['period_train'][0]} → {cfg.UTRECHT['period_train'][1]}")
    print(f"    Teste        : {cfg.UTRECHT['period_test'][0]} → {cfg.UTRECHT['period_test'][1]}")
    print(f"    Grade anon.  : {cfg.UTRECHT['grid_cell_m']} m × {cfg.UTRECHT['grid_cell_m']} m")
    print()

    print("  DETECÇÃO DE RAMPA (base) — Swinging-Door Trending")
    print(f"    epsilon (compressão) : {cfg.RAMP['compression_eps']}")
    print(f"    delta (magnitude)    : {cfg.RAMP['delta']}")
    print(f"    Grade epsilons       : {cfg.RAMP['sensitivity_eps']}")
    print(f"    Grade deltas         : {cfg.RAMP['sensitivity_deltas']}")
    print()

    print("  GATE G0 — adequação espacial")
    print(f"    n_draws Monte Carlo : {cfg.GATE0['n_draws']}")
    print(f"    Limiar RU           : {cfg.GATE0['ru_threshold']:.0%}")
    print(f"    Fração afetada max  : {cfg.GATE0['max_affected_frac']:.0%}")
    print()

    print("  GATE G1 — dependência de cauda (piloto)")
    print(f"    Quantis             : {cfg.GATE1['quantile_pilot']}")
    print(f"    Bootstrap (piloto)  : {cfg.GATE1['n_bootstrap_pilot']} sorteios")
    print(f"    Pares próximos      : < {cfg.GATE1['max_pair_dist_km']} km")
    print(f"    Limiar go/no-go     : {cfg.GATE1['go_threshold']:.0%} pares significativos")
    print(f"    FDR α               : {cfg.GATE1['fdr_alpha']}")
    print()

    import math
    chi_t = cfg.SYNTHETIC["gumbel_chi_theory"]
    print("  TESTES SINTÉTICOS")
    print(f"    n amostras          : {cfg.SYNTHETIC['n_samples']:,}")
    print(f"    AR(1) ρ             : {cfg.SYNTHETIC['ar1_rho']}")
    print(f"    Gumbel θ            : {cfg.SYNTHETIC['gumbel_theta']}")
    print(f"    χ teórico Gumbel    : {chi_t:.6f}  (= 2 − √2)")
    print(f"    Tolerância T1       : chi_hat < {cfg.SYNTHETIC['tol_independence']}")
    print(f"    Tolerância T2       : |chi_hat − {chi_t:.4f}| < {cfg.SYNTHETIC['tol_gumbel']}")
    print(f"    G0 d_nominal        : {cfg.SYNTHETIC['gate0_d_nominal_m']:.0f} m")
    print(f"    G0 RU analítico     : {cfg.SYNTHETIC['gate0_ru_analytic']:.4f}  (= √2 × 150/300)")
    print()

    print("  DIRETÓRIOS")
    for name, path in cfg.DIRS.items():
        exists = "✓" if path.exists() else "✗ (não existe ainda)"
        print(f"    {name:<16}: {path.relative_to(cfg.ROOT)}  {exists}")
    print(sep)


if __name__ == "__main__":
    print("Validando configuração...", end=" ")
    validate_config()
    print("OK\n")
    print_summary()
    print("\nA0 concluído — configuração válida.")
