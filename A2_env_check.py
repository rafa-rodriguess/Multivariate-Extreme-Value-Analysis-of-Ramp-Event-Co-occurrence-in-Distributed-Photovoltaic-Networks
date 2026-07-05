"""
A2_env_check.py — Verificar ambiente e versões de pacotes
==========================================================
Importa todos os pacotes necessários, imprime versões e reporta
o que está faltando. Verifica também a disponibilidade da ponte
rpy2 → R e dos pacotes R necessários (texmex, evd, POT).

Executar:
    python A2_env_check.py

Critério de pronto:
    - Todos os pacotes Python listados em REQUIRED importam sem erro
    - Versões impressas para auditoria
    - R/rpy2 verificado (falha com aviso, não erro — é opcional no piloto)
"""

import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg

# ── Pacotes Python obrigatórios ───────────────────────────────────────────────
REQUIRED_PYTHON = [
    ("numpy",       "np"),
    ("pandas",      "pd"),
    ("scipy",       "scipy"),
    ("pvlib",       "pvlib"),
    ("pyproj",      "pyproj"),
    ("statsmodels", "statsmodels"),
    ("joblib",      "joblib"),
    ("pyarrow",     "pyarrow"),
    ("matplotlib",  "matplotlib"),
    ("seaborn",     "seaborn"),
    ("pytest",      "pytest"),
]

# ── Pacotes R verificados via rpy2 ────────────────────────────────────────────
REQUIRED_R = ["texmex", "evd", "POT"]


def check_python_packages() -> list[str]:
    """Importa cada pacote e coleta versões. Retorna lista de pacotes faltando."""
    missing = []
    print(f"\n{'─'*60}")
    print("PACOTES PYTHON")
    print(f"{'─'*60}")
    print(f"  {'Pacote':<20} {'Versão':<15} {'Status'}")
    print(f"  {'─'*20} {'─'*15} {'─'*10}")

    for pkg_name, _ in REQUIRED_PYTHON:
        try:
            mod = importlib.import_module(pkg_name)
            version = getattr(mod, "__version__", "n/a")
            print(f"  {pkg_name:<20} {version:<15} ✓")
        except ImportError as e:
            print(f"  {pkg_name:<20} {'N/A':<15} ✗  ({e})")
            missing.append(pkg_name)

    return missing


def check_r_packages() -> tuple[bool, list[str]]:
    """Verifica rpy2 e pacotes R. Retorna (rpy2_ok, r_pkgs_missing)."""
    print(f"\n{'─'*60}")
    print("R / rpy2 (necessário para validação dos estimadores χ/χ̄)")
    print(f"{'─'*60}")

    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr, isinstalled
        print(f"  rpy2          ✓  (versão: {importlib.import_module('rpy2').__version__})")

        r_missing = []
        for pkg in REQUIRED_R:
            if isinstalled(pkg):
                print(f"  R::{pkg:<15} ✓")
            else:
                print(f"  R::{pkg:<15} ✗  (instalar no R: install.packages('{pkg}'))")
                r_missing.append(pkg)

        return True, r_missing

    except ImportError:
        print("  rpy2          ✗  (não instalado — install: pip install rpy2)")
        print("  AVISO: rpy2 é necessário para o Teste 3 (cross-check com texmex).")
        print("  Os testes 1, 2 e 4 rodam sem rpy2.")
        return False, REQUIRED_R


def check_pvlib_clearsky() -> None:
    """Verifica que pvlib consegue calcular índice de céu-claro para Utrecht."""
    print(f"\n{'─'*60}")
    print("VERIFICAÇÃO FUNCIONAL: pvlib clearsky (Utrecht)")
    print(f"{'─'*60}")
    try:
        import pvlib
        import pandas as pd

        lat, lon = cfg.UTRECHT["lat_center"], cfg.UTRECHT["lon_center"]
        times = pd.date_range("2015-06-21 06:00", "2015-06-21 20:00",
                              freq="1min", tz="UTC")
        loc   = pvlib.location.Location(lat, lon, tz="UTC")
        cs    = loc.get_clearsky(times, model="ineichen")
        ghi_peak = cs["ghi"].max()
        print(f"  Location     : lat={lat}, lon={lon}")
        print(f"  Test date    : 2015-06-21 (solstício de verão)")
        print(f"  GHI máx      : {ghi_peak:.1f} W/m²")
        ok = ghi_peak > 600
        print(f"  GHI peak > 600 W/m²: {'✓' if ok else '✗'}")
        if not ok:
            print("  AVISO: GHI anormalmente baixo — checar turbidez de Linke.")
    except Exception as e:
        print(f"  ✗ Falha: {e}")


def check_seed() -> None:
    """Verifica que o seed global produz resultados reproduzíveis."""
    print(f"\n{'─'*60}")
    print("REPRODUTIBILIDADE: seed global")
    print(f"{'─'*60}")
    import numpy as np
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    val = rng.uniform()
    print(f"  RANDOM_SEED  : {cfg.RANDOM_SEED}")
    print(f"  np.rng(seed).uniform() : {val:.10f}  (deve ser estável entre execuções)")


def main() -> None:
    print("=" * 60)
    print("A2 — VERIFICAÇÃO DE AMBIENTE")
    print(f"Python {sys.version}")
    print("=" * 60)

    py_missing  = check_python_packages()
    rpy2_ok, r_missing = check_r_packages()
    check_pvlib_clearsky()
    check_seed()

    print(f"\n{'─'*60}")
    print("RESUMO")
    print(f"{'─'*60}")

    if py_missing:
        print(f"  ✗ Pacotes Python faltando: {py_missing}")
        print(f"    → pip install {' '.join(py_missing)}")
    else:
        print("  ✓ Todos os pacotes Python presentes")

    if not rpy2_ok:
        print("  ~ rpy2 ausente — Teste 3 (cross-check R) será pulado")
    elif r_missing:
        print(f"  ~ Pacotes R faltando: {r_missing}")
        print(f"    → No R: install.packages({r_missing})")
    else:
        print("  ✓ rpy2 + pacotes R presentes")

    print(f"{'─'*60}")

    if py_missing:
        sys.exit(1)

    print("\nA2 concluído — ambiente verificado.")


if __name__ == "__main__":
    main()
