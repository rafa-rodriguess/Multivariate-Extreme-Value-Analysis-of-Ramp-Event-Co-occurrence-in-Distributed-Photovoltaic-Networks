"""
src/logger.py — Logging estruturado de resultados e decisões
=============================================================
Dois destinos simultâneos:

1. results/log/run_log.jsonl   — audit trail machine-readable
   Uma linha JSON por (script, gate, phase) — o registro mais recente substitui o
   anterior da mesma chave (mesma semântica de deduplicação do FINDINGS.md abaixo),
   de modo que ambos os destinos refletem sempre o estado atual e nunca divergem.
   Contém: timestamp, script, gate/phase, seed, params, results, decision, paper_ref.

2. results/FINDINGS.md         — narrativa acumulativa human-readable
   Seções por gate/fase, com data, estatísticas, interpretação e
   mapeamento para seção/figura/tabela do paper.

Uso nos scripts de pipeline:
    from src.logger import log_result

    log_result(
        script      = "C0_gate0.py",
        gate        = "G0",
        params      = {"n_draws": 1000, "cell_m": 150, "ru_threshold": 0.20},
        results     = {"n_pairs": 15051, "n_flagged": 420, "frac_flagged": 0.0279,
                       "ru_median": 0.0199, "ru_p95": 0.1329},
        decision    = "APPROVED WITH CAVEAT",
        action      = "Monitor 420 flagged pairs in F6 distance regression.",
        interpretation = (
            "The 150 m anonymization grid introduces RU > 20% in only 2.79% of pairs. "
            "The median RU is 0.020 — negligible for pairs separated by > 1 km. "
            "Flagged pairs are spread across multiple distance bins, not concentrated "
            "exclusively at < 500 m, so blanket exclusion is unwarranted. "
            "The distance matrix is reliable for the tail dependence analysis."
        ),
        paper_ref   = "Section 5 — Spatial Adequacy; Figure 0; Table 0",
    )
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import cfg

LOG_JSONL  = cfg.DIRS["gates"].parent / "log" / "run_log.jsonl"
FINDINGS   = cfg.DIRS["gates"].parent / "FINDINGS.md"


def log_result(
    script:         str,
    gate:           str,
    params:         dict[str, Any],
    results:        dict[str, Any],
    decision:       str,
    action:         str,
    interpretation: str,
    paper_ref:      str = "",
    phase:          str = "",
) -> None:
    """
    Registra um resultado de gate/fase nos dois destinos.

    Parâmetros
    ----------
    script          : nome do arquivo .py que gerou o resultado
    gate            : identificador do gate (ex: "G0", "G1") ou vazio para fase
    phase           : identificador de fase (ex: "F2") se não for gate
    params          : dicionário de parâmetros usados
    results         : dicionário de métricas numéricas
    decision        : string de decisão (ex: "APPROVED WITH CAVEAT")
    action          : ação concreta derivada da decisão
    interpretation  : parágrafo de interpretação para o paper
    paper_ref       : mapeamento para seções/figuras/tabelas do paper
    """
    LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()
    label = gate if gate else phase

    # ── 1. run_log.jsonl ─────────────────────────────────────────────────────
    record = {
        "timestamp":      ts,
        "script":         script,
        "gate":           gate,
        "phase":          phase,
        "seed":           cfg.SEED,
        "params":         params,
        "results":        results,
        "decision":       decision,
        "action":         action,
        "interpretation": interpretation,
        "paper_ref":      paper_ref,
    }
    # Deduplicar por (script, gate, phase): remove qualquer registro anterior da mesma
    # chave antes de anexar o novo, mantendo o arquivo como reflexo do estado ATUAL
    # (consistente com o FINDINGS.md, que substitui por cabeçalho). Preserva a ordem de
    # primeira aparição de cada chave para estabilidade do diff.
    key = (script, gate, phase)
    existing_records = []
    if LOG_JSONL.exists():
        with open(LOG_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (rec.get("script"), rec.get("gate", ""), rec.get("phase", "")) != key:
                    existing_records.append(rec)
    existing_records.append(record)
    with open(LOG_JSONL, "w") as f:
        for rec in existing_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── 2. FINDINGS.md ───────────────────────────────────────────────────────
    date_str  = datetime.now().strftime("%Y-%m-%d")
    title     = f"Gate {gate}" if gate else f"Phase {phase}"
    header    = f"## {title} — {script}  ·  {date_str}"

    params_md  = "\n".join(f"| `{k}` | `{v}` |" for k, v in params.items())
    results_md = "\n".join(f"| `{k}` | **{v}** |" for k, v in results.items())

    block = f"""
{header}

**Decision:** {decision}  
**Action:** {action}  
**Paper ref:** {paper_ref}

### Parameters
| Parameter | Value |
|---|---|
{params_md}

### Results
| Metric | Value |
|---|---|
{results_md}

### Interpretation
{interpretation}

---
"""

    # Verificar se o cabeçalho do arquivo existe; criar se necessário
    if not FINDINGS.exists():
        FINDINGS.write_text(
            "# Research Findings Log\n\n"
            "> Auto-generated by `src/logger.py`. "
            "Each section corresponds to a gate or analysis phase.\n\n"
            "---\n"
        )

    # Verificar se este gate/fase já foi registrado (evita duplicatas em reruns)
    existing = FINDINGS.read_text()
    if header in existing:
        # Substituir bloco existente
        import re
        pattern = re.escape(header) + r".*?(?=\n## |\Z)"
        updated = re.sub(pattern, block.strip(), existing, flags=re.DOTALL)
        FINDINGS.write_text(updated)
    else:
        with open(FINDINGS, "a") as f:
            f.write(block)

    print(f"  [logger] → {LOG_JSONL.relative_to(cfg.ROOT)}")
    print(f"  [logger] → {FINDINGS.relative_to(cfg.ROOT)}")
