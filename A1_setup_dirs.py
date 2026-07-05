"""
A1_setup_dirs.py — Criar estrutura de diretórios do projeto
============================================================
Cria todos os diretórios definidos em cfg.DIRS (idempotente — seguro
de rodar múltiplas vezes sem apagar arquivos existentes).

Adiciona um .gitkeep em cada diretório vazio para que o Git rastreie
a estrutura sem precisar commitar dados.

Executar:
    python A1_setup_dirs.py

Critério de pronto:
    - Todos os diretórios em cfg.DIRS existem após a execução
    - Saída lista cada diretório criado ou já existente
    - Nenhum arquivo existente é removido
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg

EXTRA_DIRS = [
    cfg.ROOT / "tests",
    cfg.ROOT / "data" / "raw" / "utrecht",
    cfg.ROOT / "data" / "raw" / "wind",
    cfg.ROOT / "results" / "gates",
    cfg.ROOT / "results" / "figures",
]


def setup_dirs() -> None:
    all_dirs = list(cfg.DIRS.values()) + EXTRA_DIRS

    print(f"{'─'*60}")
    print("CRIANDO ESTRUTURA DE DIRETÓRIOS")
    print(f"{'─'*60}")

    for d in sorted(set(all_dirs)):
        existed = d.exists()
        d.mkdir(parents=True, exist_ok=True)

        # .gitkeep em diretórios vazios (não sobrescreve arquivos existentes)
        gitkeep = d / ".gitkeep"
        if not any(p for p in d.iterdir() if p.name != ".gitkeep"):
            gitkeep.touch()

        rel = d.relative_to(cfg.ROOT)
        status = "já existia" if existed else "criado"
        print(f"  {status:<12}  {rel}/")

    print(f"{'─'*60}")

    # Verificação final
    missing = [d for d in all_dirs if not d.exists()]
    if missing:
        raise RuntimeError(f"Diretórios não criados: {missing}")

    print(f"  Total: {len(set(all_dirs))} diretórios presentes.")


if __name__ == "__main__":
    setup_dirs()
    print("\nA1 concluído — estrutura de diretórios pronta.")
