# Multivariate Extreme-Value Analysis of Ramp-Event Co-occurrence in Distributed Photovoltaic Networks

Reproducible analysis pipeline accompanying the study **"Geographic Diversification and Tail Dependence in PV Networks"**. The code implements a gated, two-stage conditional-extremes framework applied to 175 rooftop PV systems in the Utrecht region (Netherlands, 2014–2017), including quality gates, tail-dependence diagnostics, stochastic coincidence modelling, return-level estimation, and portfolio reserve comparisons.

---

## Overview

This repository contains the **executable pipeline** used to produce the paper's quantitative results. Design principles:

- **Single source of configuration** — all parameters live in `src/config.py`; scripts do not hard-code analysis choices.
- **Gate-based workflow** — pre-specified quality gates (G0–G4) must pass before downstream modelling proceeds.
- **Script-per-step** — each pipeline stage is a standalone Python file at the repository root, callable from the command line or from the orchestrating notebook.
- **Reproducibility** — fixed random seed (`RANDOM_SEED = 42`), logged outputs under `results/gates/`, and synthetic validation tests before real-data analysis.

### Research questions addressed

| ID | Question |
|----|------------|
| **RQ1** | Is there statistically significant tail dependence between nearby PV plants during extreme ramp events? |
| **RQ2** | What is the spatial and meteorological structure of that dependence (regime, advection timing, wind height)? |
| **RQ3** | How much does observed tail dependence inflate operational reserve requirements relative to an independence baseline? |

---

## Dataset

| Item | Detail |
|------|--------|
| **Source** | Utrecht PV Systems open dataset (Visser et al., 2022) |
| **DOI** | [10.5281/zenodo.6906504](https://doi.org/10.5281/zenodo.6906504) |
| **Stations** | 175 systems, 1-minute resolution |
| **Period** | 2014-01-01 → 2017-12-31 |
| **Train / test split** | 2014–2016 (training) · 2017 (hold-out validation) |

Wind reanalysis (KNMI 10 m, optional CERRA multi-level) is fetched by dedicated download scripts when mechanism diagnostics are run.

---

## Repository structure

```
.
├── 00_pipeline.ipynb          # End-to-end orchestration notebook
├── requirements.txt           # Python dependencies
├── README.md
│
├── A*.py                      # Block A — setup & validation
├── B*.py                      # Block B — data ingestion & ramp detection
├── C*.py                      # Block C — quality gates G0–G2
├── D*.py                      # Block D — data-quality diagnostics
├── F*.py                      # Block F — modelling & portfolio analysis
├── P_*.py                     # Publication figures & tables (optional)
├── S*.py                      # Sensitivity & robustness checks
│
└── src/
    ├── config.py              # Central configuration (import as `from src.config import cfg`)
    ├── logger.py              # Structured run logging
    ├── paper_figures.py       # Matplotlib export helpers
    └── data/                  # Loaders and spatial utilities
```

After running `A1_setup_dirs.py`, the pipeline creates (locally, not tracked in this repo):

```
data/
├── raw/utrecht/               # Zenodo download
├── raw/wind/                  # KNMI / CERRA wind files
├── interim/                   # Intermediate parquet files
└── processed/                 # Normalised power, ramps, clearsky

results/
├── gates/                     # Gate decisions, model fits, parquet outputs
└── figures/                   # Diagnostic plots (if figure scripts are run)
```

---

## Requirements

### Python

- **Python 3.11+** recommended (tested with 3.13)
- Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** `pyarrow` must be **≥ 19.0.1** (avoid 19.0.0 — known parquet read bug).

### R (optional but recommended for GPD / return levels)

Several stages call R via `rpy2`. Install in an R console:

```r
install.packages(c("texmex", "evd", "POT"))
```

`A2_env_check.py` verifies Python packages and, when `rpy2` is present, R package availability.

### Optional external services

| Service | Used by | Purpose |
|---------|---------|---------|
| [Zenodo](https://zenodo.org) | `B1_download_utrecht.py` | PV generation data download |
| [KNMI Data Platform](https://dataplatform.knmi.nl) | `B7_wind_join.py` | 10 m wind observations |
| [Copernicus CDS](https://cds.climate.copernicus.eu) | `B7b_wind_cerra.py` | CERRA reanalysis (multi-level wind) |

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/rafa-rodriguess/Multivariate-Extreme-Value-Analysis-of-Ramp-Event-Co-occurrence-in-Distributed-Photovoltaic-Networks.git
cd Multivariate-Extreme-Value-Analysis-of-Ramp-Event-Co-occurrence-in-Distributed-Photovoltaic-Networks
pip install -r requirements.txt
```

### 2. Validate environment

```bash
python A0_config.py      # Print and validate central configuration
python A1_setup_dirs.py  # Create data/ and results/ directory tree
python A2_env_check.py   # Check packages (and R, if installed)
python A3_synthetic_tests.py  # Statistical estimator sanity checks
```

All four should complete without errors before downloading data.

### 3. Run the full pipeline

**Option A — Jupyter notebook (recommended for first run)**

```bash
jupyter notebook 00_pipeline.ipynb
```

Run cells sequentially from the project root. The notebook calls each script in dependency order and streams stdout live.

**Option B — Command line**

Execute scripts in the order listed in `00_pipeline.ipynb`. Minimal core path:

```bash
# Data
python B1_download_utrecht.py
python B2_qc.py
python B3_normalize.py
python B4_clearsky.py
python B5_ramp_detection.py
python B8_temporal_split.py

# Gates
python C0_gate0.py
python C1_gate1.py
python C2_gate2.py

# Two-stage model & portfolio
python F5_two_stage.py
python F7_return_levels.py
python F8_portfolio_effect.py
```

Wind-mechanism and robustness scripts (`B7*`, `F6*`, `S*`, `D2*`) extend the core path; see the notebook for the complete ordered list.

---

## Script naming convention

| Prefix | Block | Description |
|--------|-------|-------------|
| **A** | Foundation | Configuration, directories, environment, synthetic tests |
| **B** | Data | Download, QC, normalisation, clearsky, ramp detection, wind join, temporal split |
| **C** | Gates | G0 (spatial adequacy), G1 (tail dependence), G2 (GPD quality), event pairing |
| **D** | Data quality | Informative censoring diagnostic and sensitivity |
| **F** | Findings | Spatial coherence, Heffernan–Tawn / two-stage model, anisotropy, return levels, portfolio effect |
| **P** | Publication | Figure and table generation for the manuscript (requires prior gate outputs) |
| **S** | Sensitivity | Block length, threshold grid, advection matrix, OOS validation, subsampling |

Each script is self-documented: read the module docstring for inputs, outputs, and pass/fail criteria.

---

## Configuration

All analysis parameters are defined in **`src/config.py`** and exposed as the `cfg` object:

```python
from src.config import cfg

print(cfg.RANDOM_SEED)          # 42
print(cfg.UTRECHT["n_stations"]) # 175
print(cfg.GATE1["go_threshold"]) # 0.30
```

Key parameter groups: `UTRECHT`, `RAMP`, `GATE0`–`GATE2`, `G1`, `G2`, `F5`, `F7`, `F8`, `DIRS`.

**Do not edit parameters inside individual pipeline scripts** — change `src/config.py` (or override in a fork) to ensure reproducibility across the full workflow.

---

## Outputs

Primary artefacts are written as **Parquet** files under `results/gates/`, for example:

| File | Content |
|------|---------|
| `gate1_results.parquet` | Pairwise χ̂, significance flags (Gate G1) |
| `f5_stage2_params.parquet` | Two-stage model parameters (Gate G3) |
| `f8_rq3_ratio.parquet` | Real vs. independence reserve ratios (RQ3) |
| `s6_oos_return_level.parquet` | 2017 out-of-sample validation |

Each major script also appends a JSON-lines entry to `results/log/run_log.jsonl` with timestamp, parameters, and gate decision.

---

## Reproducibility checklist

- [ ] Run `A0`–`A3` successfully
- [ ] Confirm `cfg.RANDOM_SEED == 42` in `src/config.py`
- [ ] Use the same Python/R package versions as `requirements.txt`
- [ ] Execute scripts in notebook order (dependencies matter)
- [ ] Keep train/test split at 2016-12-31 / 2017-01-01 (do not re-fit on 2017 before OOS evaluation)

---

## Publication figure scripts

Scripts prefixed with `P_` regenerate individual manuscript figures when gate outputs already exist. They read from `results/gates/` and are **not** required for the statistical pipeline itself. Example:

```bash
python P_fig41_chi_decay.py
python P_fig46_reserve_comparison.py
```

---

## Citation

If you use this code, please cite the accompanying paper and the Utrecht PV dataset:

```bibtex
@dataset{visser2022utrecht,
  author  = {Visser, L. and Laar, T. and Waal, B. and Sark, W.},
  title   = {Open-source quality control routine and multi-year power generation
             data of 175 PV systems},
  year    = {2022},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.6906504}
}
```

*(Paper bibliographic entry to be added upon publication.)*

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: src` | Not running from project root | `cd` to repository root; scripts add root to `sys.path` automatically |
| `rpy2` / R package errors | R not installed or packages missing | Install R + `texmex`, `evd`, `POT`; or skip R-dependent steps temporarily |
| Parquet read error | `pyarrow==19.0.0` | Upgrade: `pip install "pyarrow>=19.0.1"` |
| Gate script fails on missing input | Upstream step not run | Follow order in `00_pipeline.ipynb` |
| CDS download fails | No Copernicus account / API key | Register at CDS; configure `~/.cdsapirc` for `B7b_wind_cerra.py` |

---

## Contact

**Rafael Rodrigues** — [github.com/rafa-rodriguess](https://github.com/rafa-rodriguess)

For bugs or questions about reproduction, open an [issue](https://github.com/rafa-rodriguess/Multivariate-Extreme-Value-Analysis-of-Ramp-Event-Co-occurrence-in-Distributed-Photovoltaic-Networks/issues) on this repository.
