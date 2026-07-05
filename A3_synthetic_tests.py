"""
A3_synthetic_tests.py — Testes sintéticos de validação do estimador χ/χ̄
=========================================================================
Implementa os 4 testes do ROADMAP (Seção 10.4). Deve passar ANTES de
qualquer análise com dados reais.

Executar como script:    python A3_synthetic_tests.py
Executar via pytest:     pytest A3_synthetic_tests.py -v

Critério de pronto: todos os 4 testes passam sem AssertionError.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Teste 1 — Independência
  Dois processos AR(1) independentes (ρ=0.3), série longa.
  chi_hat(0.95) deve ser << 1 — confirma que o estimador não infla
  dependência quando os processos são realmente independentes.

Teste 2 — Dependência assintótica conhecida (cópula de Gumbel θ=2)
  χ teórico = 2 − √2 ≈ 0.5858.
  Simula via método frailty (Lévy positivo estável).
  Confirma que chi_hat(0.95) está dentro de tol=0.05 do valor teórico.

Teste 3 — Cross-check com texmex (R)
  Repete Testes 1 e 2 via rpy2 → texmex::chi.
  Diferença máxima tolerada: 1% entre Python e R.
  Pulado automaticamente se rpy2 não está disponível.

Teste 4 — Gate G0 analítico
  Para d_nominal=300m e cell_size=150m, a incerteza relativa esperada
  analiticamente é RU ≈ √2 × 150/300 ≈ 0.707 (IC 95% sobre posição
  uniforme dentro da célula).
  Valida a função de Monte Carlo do Gate G0 contra esse valor.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None  # noqa: N816 — permite importar utilitários (empirical_chi etc.)
                    # deste módulo em outros scripts (C1, D, ...) sem exigir pytest.

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg

# ─────────────────────────────────────────────────────────────────────────────
# Utilitário: detecta rpy2 em runtime (definido antes da classe para o decorator)
# ─────────────────────────────────────────────────────────────────────────────

def _rpy2_available() -> bool:
    try:
        import rpy2  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Estimador empírico de χ (reutilizado em C/D/E)
# ─────────────────────────────────────────────────────────────────────────────

def empirical_chi(u1: np.ndarray, u2: np.ndarray, q: float) -> float:
    """
    Estimador empírico do coeficiente de dependência de cauda superior χ.

    χ̂(q) = P(U₂ > q | U₁ > q) ≈ #{U₁>q ∧ U₂>q} / (n × (1−q))

    Parâmetros
    ----------
    u1, u2 : arrays em [0,1] (margens uniformes via ECDF ou cópula)
    q      : quantil de corte (ex: 0.95)

    Retorna
    -------
    chi_hat : float ≥ 0
    """
    assert len(u1) == len(u2), "u1 e u2 devem ter o mesmo comprimento"
    n = len(u1)
    joint = np.sum((u1 > q) & (u2 > q))
    return float(joint) / (n * (1.0 - q))


def to_uniform(x: np.ndarray) -> np.ndarray:
    """Transforma série para escala uniforme via posto empírico (rank-based)."""
    n = len(x)
    ranks = np.argsort(np.argsort(x)) + 1   # postos 1..n
    return ranks / (n + 1)                   # em (0,1), evita 0 e 1 exatos


def empirical_chibar(u1: np.ndarray, u2: np.ndarray, q: float) -> float:
    """
    Estimador empírico do coeficiente de Ledford-Tawn χ̄ (chi-bar).

    χ̄(q) = 2·log(1−q) / log(Ĉ(q,q)) − 1,   Ĉ(q,q) = #{U₁>q ∧ U₂>q} / n

    χ̄ = 1  → dependência assintótica (mesma classe de χ > 0)
    χ̄ < 1  → independência assintótica; mede a *taxa* de decaimento da
              probabilidade conjunta mesmo quando χ → 0.

    Retorna -1.0 (piso, independência assintótica forte) se não houver
    nenhuma coocorrência acima de q na amostra (log(0) indefinido).
    """
    assert len(u1) == len(u2), "u1 e u2 devem ter o mesmo comprimento"
    n = len(u1)
    joint = np.sum((u1 > q) & (u2 > q))
    c_hat = joint / n
    if c_hat <= 0:
        return -1.0
    return float(2 * np.log(1 - q) / np.log(c_hat) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Simulações
# ─────────────────────────────────────────────────────────────────────────────

def simulate_ar1_pair(
    n: int,
    rho: float = cfg.SYNTHETIC["ar1_rho"],
    seed: int  = cfg.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dois processos AR(1) INDEPENDENTES entre si, cada um com autocorrelação ρ.
    Retorna (U1, U2) em escala uniforme via ECDF.
    """
    rng = np.random.default_rng(seed)
    noise1 = rng.standard_normal(n)
    noise2 = rng.standard_normal(n)     # série separada → independente

    x1 = np.zeros(n)
    x2 = np.zeros(n)
    for t in range(1, n):
        x1[t] = rho * x1[t-1] + np.sqrt(1 - rho**2) * noise1[t]
        x2[t] = rho * x2[t-1] + np.sqrt(1 - rho**2) * noise2[t]

    return to_uniform(x1), to_uniform(x2)


def simulate_gumbel_copula(
    n: int,
    theta: float = cfg.SYNTHETIC["gumbel_theta"],
    seed: int    = cfg.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simula da cópula de Gumbel com parâmetro θ usando método frailty.

    Algoritmo (Joe 1997, Nelsen 2006):
      1. V ~ Lévy(loc=0, scale=0.5) — variável frailty positivo-estável PS(1/θ)
         para θ=2: LT E[e^{-sV}] = exp(-√(2×0.5×s)) = exp(-√s) = exp(-s^{1/2}) ✓
      2. E₁, E₂ ~ Exp(1) i.i.d., independentes de V
      3. Uₖ = exp(-(Eₖ/V)^{1/θ})

    Verificação das marginais:
      P(Uₖ ≤ u) = E_V[exp(-V(-ln u)^θ)] = exp(-(-ln u)) = u  → Uniform(0,1) ✓

    Verificação da cópula:
      P(U₁≤u₁, U₂≤u₂) = E_V[exp(-V((-ln u₁)^θ + (-ln u₂)^θ))]
                       = exp(-[(-ln u₁)^θ + (-ln u₂)^θ]^{1/θ})  = C_Gumbel ✓
    """
    from scipy.stats import levy

    rng = np.random.default_rng(seed)

    # Passo 1: V ~ PS(1/θ)
    # Para θ=2: scale = 0.5 (Lévy(0,0.5) tem LT exp(-√s) ≡ exp(-s^{1/2}))
    # Para θ geral: implementar Chambers-Mallows-Stuck (só θ=2 no projeto)
    if abs(theta - 2.0) < 1e-10:
        V = levy.rvs(loc=0, scale=0.5, size=n, random_state=rng)
    else:
        raise NotImplementedError(
            f"Gumbel frailty implementado só para θ=2; recebido θ={theta}. "
            "Para θ geral, implementar Chambers-Mallows-Stuck."
        )

    # Passo 2: exponenciais independentes
    E1 = rng.exponential(1.0, n)
    E2 = rng.exponential(1.0, n)

    # Passo 3: transformação frailty
    inv_theta = 1.0 / theta
    U1 = np.exp(-(E1 / V) ** inv_theta)
    U2 = np.exp(-(E2 / V) ** inv_theta)

    return U1, U2


def monte_carlo_ru(
    d_nominal_m: float,
    cell_size_m: float,
    n_draws: int,
    seed: int = cfg.RANDOM_SEED,
) -> float:
    """
    Calcula a incerteza relativa (RU) de distância entre dois pontos fixos
    quando cada posição é perturbada aleatoriamente dentro de uma célula
    quadrada de lado cell_size_m.

    RU = (p97.5 - p2.5) / mediana  sobre os n_draws sorteios.
    """
    rng = np.random.default_rng(seed)

    # Dois pontos a d_nominal_m de distância ao longo do eixo x
    x1, y1 = 0.0, 0.0
    x2, y2 = d_nominal_m, 0.0

    half = cell_size_m / 2.0
    dx1 = rng.uniform(-half, half, n_draws)
    dy1 = rng.uniform(-half, half, n_draws)
    dx2 = rng.uniform(-half, half, n_draws)
    dy2 = rng.uniform(-half, half, n_draws)

    d_perturbed = np.sqrt((x2 + dx2 - x1 - dx1)**2 + (y2 + dy2 - y1 - dy1)**2)

    p025 = np.percentile(d_perturbed, 2.5)
    p975 = np.percentile(d_perturbed, 97.5)
    med  = np.median(d_perturbed)

    return float((p975 - p025) / med)


# ─────────────────────────────────────────────────────────────────────────────
# Testes pytest
# ─────────────────────────────────────────────────────────────────────────────

class TestSynthetic:

    def test_1_independence(self) -> None:
        """
        T1: AR(1) independentes → chi_hat(0.95) deve ser pequeno.

        AR(1) com ρ=0.3 tem autocorrelação serial mas as duas séries são
        INDEPENDENTES entre si → χ → 0. Para q=0.95 e n=50.000, o valor
        esperado é ≈ (1-q) = 0.05, com ruído amostral < 0.05.
        """
        q   = cfg.SYNTHETIC["test_quantile"]
        tol = cfg.SYNTHETIC["tol_independence"]
        n   = cfg.SYNTHETIC["n_samples"]

        U1, U2 = simulate_ar1_pair(n=n)
        chi    = empirical_chi(U1, U2, q)

        print(f"\n  T1: chi_hat({q}) = {chi:.4f}  (limite: < {tol})")
        assert chi < tol, (
            f"T1 FALHOU: chi_hat={chi:.4f} ≥ {tol} para AR(1) independentes. "
            f"O estimador pode estar inflando dependência espúria."
        )

    def test_2_gumbel_copula(self) -> None:
        """
        T2: cópula de Gumbel θ=2 → chi_hat(0.95) ≈ 2 − √2 ≈ 0.5858.

        Tolerância: ±0.05 (cobrindo variação amostral com n=50.000).
        Se falhar, o estimador tem bug — não prosseguir para dados reais.
        """
        q      = cfg.SYNTHETIC["test_quantile"]
        tol    = cfg.SYNTHETIC["tol_gumbel"]
        chi_th = cfg.SYNTHETIC["gumbel_chi_theory"]
        n      = cfg.SYNTHETIC["n_samples"]

        U1, U2 = simulate_gumbel_copula(n=n)
        chi    = empirical_chi(U1, U2, q)
        delta  = abs(chi - chi_th)

        print(f"\n  T2: chi_hat({q}) = {chi:.4f}  |  teórico = {chi_th:.4f}  |  Δ = {delta:.4f}  (limite: < {tol})")
        assert delta < tol, (
            f"T2 FALHOU: |chi_hat − chi_teórico| = {delta:.4f} ≥ {tol}. "
            f"Estimador ou simulação da cópula com bug."
        )

    @(pytest.mark.skipif(
        not _rpy2_available(),
        reason="rpy2 não instalado — Teste 3 pulado (instalar: pip install rpy2)",
    ) if pytest is not None else (lambda f: f))
    def test_3_rpy2_crosscheck(self) -> None:
        """
        T3: cross-check dos Testes 1 e 2 via rpy2 → texmex::chi.
        Diferença máxima tolerada: 1% entre estimativas Python e R.
        """
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        texmex = importr("texmex")

        q  = cfg.SYNTHETIC["test_quantile"]
        n  = cfg.SYNTHETIC["n_samples"]

        for label, (U1, U2), chi_py in [
            ("independência", simulate_ar1_pair(n=n),         empirical_chi(*simulate_ar1_pair(n=n), q)),
            ("Gumbel θ=2",   simulate_gumbel_copula(n=n),     empirical_chi(*simulate_gumbel_copula(n=n), q)),
        ]:
            r_data = ro.r.cbind(ro.FloatVector(U1), ro.FloatVector(U2))
            r_chi  = texmex.chi(r_data, nq=1, qlim=ro.FloatVector([q, q]))
            chi_r  = float(np.array(r_chi.rx2("chi"))[0])

            delta_pct = abs(chi_py - chi_r) / max(abs(chi_r), 1e-6) * 100
            print(f"\n  T3 [{label}]: Python={chi_py:.4f}, R={chi_r:.4f}, Δ={delta_pct:.2f}%")
            assert delta_pct < 1.0, (
                f"T3 FALHOU [{label}]: diferença Python/R = {delta_pct:.2f}% ≥ 1%. "
                f"Checar implementação."
            )

    def test_4_gate0_analytic(self) -> None:
        """
        T4: Gate G0 — RU do Monte Carlo deve bater com a aproximação analítica.

        Para dois pontos a d_nominal=300m, com célula de 150m:
          RU_analítico = √2 × 150 / 300 ≈ 0.707
        Tolerância: ±10% do valor analítico.
        """
        d_nom    = cfg.SYNTHETIC["gate0_d_nominal_m"]
        cell     = cfg.SYNTHETIC["gate0_cell_size_m"]
        ru_anal  = cfg.SYNTHETIC["gate0_ru_analytic"]
        tol_rel  = cfg.SYNTHETIC["tol_gate0_ru"]
        n_draws  = cfg.SYNTHETIC["gate0_n_draws"]

        ru_mc    = monte_carlo_ru(d_nom, cell, n_draws)
        delta    = abs(ru_mc - ru_anal) / ru_anal

        print(f"\n  T4: RU_mc = {ru_mc:.4f}  |  RU_analítico = {ru_anal:.4f}  |  Δ_rel = {delta:.2%}  (limite: < {tol_rel:.0%})")
        assert delta < tol_rel, (
            f"T4 FALHOU: RU Monte Carlo ({ru_mc:.4f}) difere do analítico ({ru_anal:.4f}) "
            f"em {delta:.2%} ≥ {tol_rel:.0%}. Checar função monte_carlo_ru()."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Execução standalone (sem pytest)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    suite = TestSynthetic()
    tests = [
        ("T1 — Independência (AR1)",          suite.test_1_independence),
        ("T2 — Gumbel θ=2 (dependência)",     suite.test_2_gumbel_copula),
        ("T3 — Cross-check R/texmex",         suite.test_3_rpy2_crosscheck),
        ("T4 — Gate G0 analítico",            suite.test_4_gate0_analytic),
    ]

    sep = "─" * 60
    print(sep)
    print("A3 — TESTES SINTÉTICOS")
    print(sep)

    results = {}
    for name, fn in tests:
        # Pular T3 se rpy2 indisponível
        if "Cross-check" in name and not _rpy2_available():
            print(f"\n  {name}")
            print("    PULADO — rpy2 não instalado")
            results[name] = "SKIP"
            continue

        print(f"\n  {name}")
        try:
            fn()
            print(f"    PASSOU ✓")
            results[name] = "PASS"
        except AssertionError as e:
            print(f"    FALHOU ✗  {e}")
            results[name] = "FAIL"
        except Exception as e:
            print(f"    ERRO ✗  {e}")
            traceback.print_exc()
            results[name] = "ERROR"

    print(f"\n{sep}")
    print("RESULTADO FINAL")
    print(sep)
    for name, status in results.items():
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "~", "ERROR": "!"}.get(status, "?")
        print(f"  {icon} {status:<6}  {name}")

    n_fail = sum(1 for s in results.values() if s in ("FAIL", "ERROR"))
    print(sep)

    if n_fail:
        print(f"\nA3 REPROVADO: {n_fail} teste(s) falharam. Não prosseguir para dados reais.")
        sys.exit(1)
    else:
        print("\nA3 concluído — todos os testes passaram. Estimador validado.")
