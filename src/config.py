"""
config.py — Configuração central do projeto
============================================
Importar em todos os scripts de pipeline:

    from src.config import cfg

Todos os parâmetros do ROADMAP (Seção 10.2) estão aqui.
Nenhum script define parâmetros locais — eles vêm daqui.
"""

from pathlib import Path

# ── Raiz do projeto ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# ── Seed global (fixo para toda análise) ─────────────────────────────────────
RANDOM_SEED = 42

# ── Estrutura de diretórios ───────────────────────────────────────────────────
DIRS = {
    "raw":          ROOT / "data" / "raw",
    "raw_utrecht":  ROOT / "data" / "raw" / "utrecht",
    "raw_wind":     ROOT / "data" / "raw" / "wind",
    "interim":      ROOT / "data" / "interim",
    "processed":    ROOT / "data" / "processed",
    "gates":        ROOT / "results" / "gates",
    "figures":      ROOT / "results" / "figures",
    "src":          ROOT / "src",
    "tests":        ROOT / "tests",
}

# ── Dataset Utrecht ───────────────────────────────────────────────────────────
UTRECHT = {
    "zenodo_doi":   "10.5281/zenodo.6906504",
    "n_stations":   175,
    "period_train": ("2014-01-01", "2016-12-31"),
    "period_test":  ("2017-01-01", "2017-12-31"),
    "train_end":    "2016-12-31 23:59",
    "test_start":   "2017-01-01 00:00",
    "resolution_min": 1,
    "grid_cell_m":  150,      # resolução da grade de anonimização (metros)
    "lat_center":   52.09,    # Utrecht, NL (aprox.)
    "lon_center":   5.12,
}

# ── Parâmetros de detecção de rampa — Swinging-Door Trending (Bristol 1990) ───
# Algoritmo real de 2 fases (ROADMAP 10.2, B.5):
#   Fase 1 (compressão): compression_eps — tolerância que filtra ruído e define
#     as fronteiras de segmento de tendência real na série k_i(t).
#   Fase 2 (rampa): delta — magnitude mínima sobre um segmento comprimido para
#     ser classificado como evento de rampa.
RAMP = {
    "compression_eps":   0.02,   # tolerância de compressão SDT (adimensional)
    "delta":             0.10,   # magnitude mínima de rampa (adimensional)
    "min_duration_min":  10,     # duração mínima (min) — filtra ruído de 1-2 min
                                  # sob nuvens fragmentadas (ver ROADMAP B.5, decisão pós-SDT)
    # Grades de sensibilidade (Tab. 6 do paper) — ε × Δ (min_duration_min fixo)
    "sensitivity_eps":    [0.01, 0.02, 0.03, 0.05, 0.08],
    "sensitivity_deltas": [0.10, 0.15, 0.20, 0.25, 0.30],
}

# ── Modelo de céu-claro ───────────────────────────────────────────────────────
CLEARSKY = {
    "model":              "ineichen",   # pvlib.clearsky.ineichen
    "nocturnal_threshold_w_kwp": 5.0,  # p_clearsky < 5 W/kWp → k = NaN
}

# ── Controle de qualidade ─────────────────────────────────────────────────────
QC = {
    "p_norm_max":            1.1,    # máximo de p* (tolerância 10% acima de STC)
    "clearsky_min_threshold":0.005,  # fração mínima de GHI_cs/GHI_STC para definir k
    "k_max":                 1.5,    # máximo aceitável de k_i
}

# ── Gate G0: adequação espacial ───────────────────────────────────────────────
G0 = GATE0 = {
    "n_draws":          1000,   # sorteios Monte Carlo
    "cell_size_m":      150,    # tamanho da célula de anonimização (metros)
    "ru_threshold":     0.20,   # incerteza relativa máxima aceitável (20%)
    "max_affected_frac":0.05,   # fração máxima de pares afetados para G0 aprovado
    "short_dist_m":     500,    # limiar de "curtíssima distância" (metros)
}

# ── Gate G1: dependência de cauda ─────────────────────────────────────────────
G1 = GATE1 = {
    "quantile_pilot":       [0.90, 0.95],       # piloto
    "quantile_full":        [0.90, 0.95, 0.98, 0.99, 0.995],  # paper final
    "max_pair_dist_km":     5.0,    # pares "próximos" para decisão de go/no-go
    "n_bootstrap_pilot":    100,    # sorteios bootstrap no piloto
    "n_bootstrap_final":    500,    # sorteios bootstrap no paper final
    "fdr_alpha":            0.05,   # taxa de falsa descoberta (Benjamini-Hochberg)
    "go_threshold":         0.30,   # fração mínima de pares significativos para GO
    "min_events":           30,     # exceedances mínimas por usina (Gate G1)
    # Velocidade de deslocamento de nuvem para janela de coincidência
    "cloud_speed_ms":       15.0,   # m/s — valor base
    "cloud_speed_grid":     [10.0, 15.0, 20.0],  # sensibilidade
    "coincidence_window_max_min": 30,  # teto em minutos
}

# ── Gate G3: dependência condicional (Heffernan-Tawn, dois estágios) ─────────
G3 = GATE3 = {
    "min_matched_diagnostic": 15,   # mínimo de eventos casados p/ diagnóstico por par (não gating)
    "n_bootstrap_pairs": 150,       # sorteios do cluster bootstrap (por par direcional)
    "dist_ref_km": 1.0,             # distância de referência p/ evitar singularidade no modelo de potência
    "alpha_coherence_tol": 0.30,    # tolerância p/ checagem de coerência Gate G3 vs χ do Gate G1
}

# ── F6: estrutura espacial (anisotropia vs. vento) ───────────────────────────
F6 = {
    "n_bootstrap_pairs": 150,        # mesmo desenho do G3 (cluster bootstrap por par direcional)
    "min_matched_diagnostic": 15,    # mínimo de eventos casados p/ diagnóstico por par
    "alignment_bins": 5,             # nº de faixas p/ o gráfico de coincidência vs. alinhamento
    "min_effect_size_rel": 0.05,     # mudança mínima RELATIVA em P(match) (alinhamento -1->+1)
                                      # para considerar o efeito PRATICAMENTE relevante, além de
                                      # estatisticamente significativo (amostras de ~700k inflam
                                      # significância de efeitos triviais — ver F6_anisotropy.py)
    "min_valong_ms": 1.0,            # componente mínima do vento na direção do par (m/s) para
                                      # calcular um atraso advectivo esperado (distância/veloc.) —
                                      # abaixo disso a previsão explode/não tem sentido físico
    "min_corr_meaningful": 0.10,     # |correlação| mínima (Spearman) entre atraso observado e
                                      # esperado para considerar sinal de advecção PRATICAMENTE
                                      # relevante — mesma lógica de tamanho de efeito do F6_anisotropy.py
}

# ── F7/F8: rampa agregada da rede — níveis de retorno, backtesting, RQ3 ──────
# Rede tratada como "usina virtual": k_agg(t) = soma ponderada por capacidade dos
# k_i(t) já calculados em B4 (clearsky_index.parquet) — reaproveita 100% do
# pipeline de detecção (mesmo SDT) e de modelagem marginal (mesma seleção
# adaptativa de limiar + declustering de C2_gate2.py) aplicados a essa série única.
F7 = {
    "reserve_direction":      "down",  # rampas DOWN = risco de falta de suprimento
                                         # (o que exige reserva operacional); "up" é
                                         # reportado como robustez, não é o resultado central
    "near_neighbor_radius_km": 5.0,     # mesmo raio de G1/F5 (max_pair_dist_km) — fora
                                         # desse raio, alpha(dist) do F5 já é ~0
    "n_bootstrap_real":        150,     # bootstrap por blocos móveis, cenário REAL (Gate G4)
    "n_independence_real":     150,     # realizações do cenário INDEPENDÊNCIA (F8)
    "n_model_implied_real":    150,     # realizações do cenário IMPLICADO PELO MODELO (F8)
    # Horizontes de retorno reportados (anos) — Fig. 8/9, Tab. 4/5
    "return_periods_years":   [1/12, 0.25, 0.5, 1.0, 2.0, 5.0],
    # Horizontes usados no backtest de cobertura (Gate G4) — restritos a períodos
    # curtos o bastante para ter exceedances esperadas dentro do único ano de teste (2017)
    "backtest_periods_years": [1/12, 1/6, 0.25, 0.5],
    "gate4_coverage_low":     0.90,     # cobertura empírica mínima aceitável do IC nominal 95%
    "gate4_coverage_high":    0.98,     # cobertura empírica máxima aceitável
    "min_declustered_agg":    30,       # piso de excedências declusterizadas p/ ajuste GPD da série agregada
                                         # (mais baixo que o de G2/série individual — só 1 série agregada)
}

# ── F6d: vetor de deslocamento de nuvem implícito pela rede + vento em altura ─
# Em vez de comparar par-a-par (F6/F6b) contra vento de uma altura fixa, inverte
# a pergunta: que velocidade (direção+módulo) melhor explica TODOS os atrasos
# observados numa mesma janela de tempo, simultaneamente (mínimos quadrados,
# técnica de "cloud motion vector" a partir de rede terrestre de sensores,
# Bosch & Kleissl 2013)? Depois compara esse vetor IMPLÍCITO (100% derivado dos
# dados de potência, sem depender de nenhuma fonte de vento externa) contra o
# vento real em altura de nuvem via CERRA (níveis de pressão).
F6D = {
    "window": "day",              # agrupamento das janelas de inversão (por dia civil de t_ext)
    "min_pairs_per_window": 5,    # mínimo de pares casados na janela p/ tentar a inversão
    "min_speed_ms": 1.0,          # módulo mínimo do vetor implícito p/ considerá-lo não-degenerado
    "n_bootstrap_windows": 150,   # bootstrap por janela (não por par) do alinhamento circular
    # Níveis de pressão CERRA (hPa) cobrindo as 3 categorias de nuvem (baixa/média/alta)
    "cerra_pressure_hpa":      [950, 900, 850, 800, 700, 600, 500, 400],
    "cerra_pressure_alt_m":    {950: 540, 900: 990, 850: 1460, 800: 1950,
                                 700: 3010, 600: 4200, 500: 5570, 400: 7190},
    "cloud_category_by_hpa": {
        950: "baixa", 900: "baixa", 850: "baixa", 800: "baixa",
        700: "média", 600: "média", 500: "média", 400: "alta",
    },
    "min_corr_meaningful": 0.10,  # mesmo limiar de relevância prática do F6b
}

# ── Testes sintéticos (ROADMAP 10.4) ─────────────────────────────────────────
SYNTHETIC = {
    "n_samples":          50_000,
    "ar1_rho":            0.3,        # autocorrelação dos AR(1) independentes
    "gumbel_theta":       2.0,        # parâmetro da cópula de Gumbel (Teste 2)
    "gumbel_chi_theory":  2 - 2**0.5, # χ teórico ≈ 0.5858
    "test_quantile":      0.95,       # quantil de referência nos testes
    # Tolerâncias
    "tol_independence":   0.10,       # chi_hat < tol em Teste 1 (independência)
    "tol_gumbel":         0.05,       # |chi_hat - chi_theory| < tol em Teste 2
    "tol_gate0_ru":       0.10,       # |RU_mc - RU_analytic| / RU_analytic < tol
    # Gate G0 analítico (Teste 4)
    "gate0_d_nominal_m":  300.0,
    "gate0_cell_size_m":  150.0,
    # RU analítico ≈ √2 × cell_size / d_nominal
    "gate0_ru_analytic":  (2**0.5) * 150 / 300,  # ≈ 0.707
    "gate0_n_draws":      2000,
}


# ── Atalho: objeto simples para acesso por atributo ───────────────────────────
class _Config:
    ROOT          = ROOT
    SEED          = RANDOM_SEED   # alias curto
    RANDOM_SEED   = RANDOM_SEED
    DIRS          = DIRS
    UTRECHT       = UTRECHT
    RAMP          = RAMP
    CLEARSKY      = CLEARSKY
    QC            = QC
    GATE0         = GATE0
    G0            = GATE0         # alias curto
    GATE1         = GATE1
    G1            = GATE1         # alias curto
    GATE3         = GATE3
    G3            = GATE3         # alias curto
    F6            = F6
    F6D           = F6D
    F7            = F7
    SYNTHETIC     = SYNTHETIC

cfg = _Config()
