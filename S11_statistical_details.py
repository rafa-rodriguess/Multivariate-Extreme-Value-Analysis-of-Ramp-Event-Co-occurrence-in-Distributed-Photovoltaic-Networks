"""
S11_statistical_details.py — Paper-ready statistical detail table
==================================================================
Extracts precomputed pipeline statistics into a long-format CSV for
supplementary tables / Overleaf ingestion.

Sources:
  - Stage 1 logistic regression (F5_two_stage.py design, refit via statsmodels GLM)
  - Gate G1 BH-adjusted p-values (gate1_results.parquet)
  - System counts (coords.parquet + cfg.UTRECHT)

Output:
  results/gates/s11_statistical_details.csv

Execute:
    python S11_statistical_details.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result
from F5_heffernan_tawn_pilot import build_ecdf, laplace_transform
from F5_two_stage import DIST_FLOOR_KM

RAMPS_PQ = cfg.DIRS["interim"] / "ramps_split.parquet"
ALIGNED_PQ = cfg.DIRS["processed"] / "aligned_pairs.parquet"
EVENTPAIR_PQ = cfg.DIRS["gates"] / "event_pairing_summary.parquet"
GATE1_PQ = cfg.DIRS["gates"] / "gate1_results.parquet"
COORDS_PQ = cfg.DIRS["interim"] / "coords.parquet"

OUT_CSV = cfg.DIRS["gates"] / "s11_statistical_details.csv"

MAX_DIST_KM = cfg.G1["max_pair_dist_km"]
G1_PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

STAGE1_REF = {
    "stage1_gamma0_estimate": 3.68,
    "stage1_gammad_estimate": -0.651,
    "stage1_gammax_estimate": -0.139,
    "stage1_pseudo_r2": 0.93,
}

SEP = "─" * 60


def _build_stage1_dataset(
    aligned: pd.DataFrame,
    ramps: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Replicate F5 Stage 1 data preparation (offset + covariates)."""
    dist_lookup: dict[tuple[str, str], float] = {}
    dtwin_lookup: dict[tuple[str, str], float] = {}
    for _, row in event_summary.iterrows():
        d_km = row["dist_ij_m"] / 1000.0
        dist_lookup[(row["station_i"], row["station_j"])] = d_km
        dist_lookup[(row["station_j"], row["station_i"])] = d_km
        dtwin_lookup[(row["station_i"], row["station_j"])] = row["dt_window_min"]
        dtwin_lookup[(row["station_j"], row["station_i"])] = row["dt_window_min"]

    lam_by_station: dict[str, float] = {}
    for sid, g in ramps.groupby("station_id"):
        ts = pd.to_datetime(g["start_ts"])
        span_min = (ts.max() - ts.min()).total_seconds() / 60.0 if len(g) > 1 else np.nan
        lam_by_station[sid] = len(g) / span_min if span_min and span_min > 0 else np.nan

    def p_null_directional(ext_i: str, partner_j: str) -> float:
        lam_j = lam_by_station.get(partner_j, np.nan)
        dtw = dtwin_lookup.get((ext_i, partner_j), np.nan)
        if not (np.isfinite(lam_j) and np.isfinite(dtw)):
            return np.nan
        return 1.0 - np.exp(-lam_j * 2.0 * dtw)

    ecdf_by_station = {
        sid: build_ecdf(np.sort(g["abs_mag"].to_numpy()))
        for sid, g in ramps.groupby("station_id")
    }

    x_lap = np.full(len(aligned), np.nan)
    for sid, idx in aligned.groupby("station_ext").groups.items():
        pos = aligned.index.get_indexer(idx)
        x_lap[pos] = laplace_transform(
            ecdf_by_station[sid](aligned.loc[idx, "mag_ext"].to_numpy())
        )

    dist_km = np.array([
        dist_lookup.get((a, b), np.nan)
        for a, b in zip(aligned["station_ext"], aligned["station_partner"])
    ])
    p_null = np.array([
        p_null_directional(a, b)
        for a, b in zip(aligned["station_ext"], aligned["station_partner"])
    ])
    d1 = aligned.assign(x_lap=x_lap, dist_km=dist_km, p_null=p_null)

    valid = (
        np.isfinite(d1["x_lap"])
        & np.isfinite(d1["dist_km"])
        & np.isfinite(d1["p_null"])
        & (d1["p_null"] > 0)
        & (d1["p_null"] < 1)
    )
    d1 = d1.loc[valid]

    x_design = sm.add_constant(np.column_stack([
        np.log(np.maximum(d1["dist_km"].to_numpy(), DIST_FLOOR_KM)),
        d1["x_lap"].to_numpy(),
    ]))
    y = d1["matched"].to_numpy().astype(float)
    p_null_clip = np.clip(d1["p_null"].to_numpy(), 1e-4, 1 - 1e-4)
    offset = np.log(p_null_clip / (1 - p_null_clip))
    return d1, x_design, y, offset


def fit_stage1_statistics(
    aligned: pd.DataFrame,
    ramps: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> dict[str, float]:
    """Fit Stage 1 GLM and return estimate / SE / z / p / CI for each coefficient."""
    _, x_design, y, offset = _build_stage1_dataset(aligned, ramps, event_summary)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model = sm.GLM(y, x_design, family=sm.families.Binomial(), offset=offset).fit()
        pseudo_r2 = 1 - model.deviance / model.null_deviance

    names = ["gamma0", "gammad", "gammax"]
    out: dict[str, float] = {"stage1_pseudo_r2": round(float(pseudo_r2), 4)}
    ci = np.asarray(model.conf_int(alpha=0.05))

    for i, name in enumerate(names):
        out[f"stage1_{name}_estimate"] = round(float(model.params[i]), 4)
        out[f"stage1_{name}_se"] = round(float(model.bse[i]), 4)
        out[f"stage1_{name}_z"] = round(float(model.tvalues[i]), 4)
        out[f"stage1_{name}_pvalue"] = float(model.pvalues[i])
        out[f"stage1_{name}_ci_lo"] = round(float(ci[i, 0]), 4)
        out[f"stage1_{name}_ci_hi"] = round(float(ci[i, 1]), 4)

    return out


def extract_g1_pval_percentiles(gate1: pd.DataFrame) -> dict[str, float]:
    """Percentiles of BH-adjusted p-values among significant close pairs."""
    close = gate1[gate1["dist_ij_m"] <= MAX_DIST_KM * 1000]
    sig = close.loc[close["significant"], "p_adjusted"]
    if sig.empty:
        raise ValueError("No significant Gate G1 pairs found — run C1_gate1.py first.")

    pct = sig.quantile(G1_PERCENTILES)
    labels = ["p10", "p25", "p50", "p75", "p90"]
    return {
        f"g1_pval_adj_{lab}": round(float(val), 6)
        for lab, val in zip(labels, pct)
    }


def count_systems(coords: pd.DataFrame) -> dict[str, int]:
    """System counts for the paper (total, valid coordinates, spatial analysis)."""
    valid = coords["lat_centroid"].notna() & coords["lon_centroid"].notna()
    n_valid = int(valid.sum())
    return {
        "n_systems_total": int(cfg.UTRECHT["n_stations"]),
        "n_systems_valid_coords": n_valid,
        "n_systems_spatial_analysis": n_valid,
    }


def to_long_csv(metrics: dict[str, float | int]) -> pd.DataFrame:
    order = [
        "stage1_gamma0_estimate", "stage1_gamma0_se", "stage1_gamma0_z",
        "stage1_gamma0_pvalue", "stage1_gamma0_ci_lo", "stage1_gamma0_ci_hi",
        "stage1_gammad_estimate", "stage1_gammad_se", "stage1_gammad_z",
        "stage1_gammad_pvalue", "stage1_gammad_ci_lo", "stage1_gammad_ci_hi",
        "stage1_gammax_estimate", "stage1_gammax_se", "stage1_gammax_z",
        "stage1_gammax_pvalue", "stage1_gammax_ci_lo", "stage1_gammax_ci_hi",
        "g1_pval_adj_p10", "g1_pval_adj_p25", "g1_pval_adj_p50",
        "g1_pval_adj_p75", "g1_pval_adj_p90",
        "n_systems_total", "n_systems_valid_coords", "n_systems_spatial_analysis",
        "stage1_pseudo_r2",
    ]
    rows = []
    for key in order:
        if key not in metrics:
            continue
        val = metrics[key]
        if key.startswith("n_systems"):
            val = int(val)
        rows.append({"metric": key, "value": val})
    return pd.DataFrame(rows)


def _check_stage1_refs(metrics: dict[str, float]) -> None:
    tol = 0.01
    for key, ref in STAGE1_REF.items():
        got = metrics[key]
        if abs(got - ref) > tol:
            print(f"  [WARN] {key}: got {got}, paper ref {ref} (tol={tol})")
        else:
            print(f"  [OK]   {key}: {got} (ref {ref})")


def main() -> None:
    print(SEP)
    print("S11 — STATISTICAL DETAILS FOR PAPER TABLE")
    print(SEP)

    required = [RAMPS_PQ, ALIGNED_PQ, EVENTPAIR_PQ, GATE1_PQ, COORDS_PQ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\nERRO: arquivos ausentes:")
        for p in missing:
            print(f"  - {p}")
        sys.exit(1)

    ramps = pd.read_parquet(RAMPS_PQ)
    ramps = ramps[ramps["split"] == "train"].copy()
    ramps["abs_mag"] = ramps["delta_k"].abs()

    aligned = pd.read_parquet(ALIGNED_PQ)
    event_summary = pd.read_parquet(EVENTPAIR_PQ)
    gate1 = pd.read_parquet(GATE1_PQ)
    coords = pd.read_parquet(COORDS_PQ)

    print("\n[1/3] Stage 1 logistic regression statistics...")
    stage1 = fit_stage1_statistics(aligned, ramps, event_summary)
    _check_stage1_refs(stage1)

    print("\n[2/3] Gate G1 adjusted p-value percentiles (significant pairs, d ≤ 5 km)...")
    g1 = extract_g1_pval_percentiles(gate1)
    n_sig = int(gate1[(gate1["dist_ij_m"] <= MAX_DIST_KM * 1000) & gate1["significant"]].shape[0])
    print(f"  Significant close pairs: {n_sig:,}")

    print("\n[3/3] System counts...")
    systems = count_systems(coords)
    for k, v in systems.items():
        print(f"  {k}: {v}")

    metrics: dict[str, float | int] = {}
    metrics.update(stage1)
    metrics.update(g1)
    metrics.update(systems)

    out_df = to_long_csv(metrics)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n  Salvo: {OUT_CSV.relative_to(cfg.ROOT)}  ({len(out_df)} linhas)")

    log_result(
        script="S11_statistical_details.py",
        gate="",
        phase="S11",
        params={"max_dist_km": MAX_DIST_KM, "g1_percentiles": G1_PERCENTILES},
        results={
            "n_metrics": len(out_df),
            "n_significant_g1_pairs": n_sig,
            **{k: metrics[k] for k in STAGE1_REF},
        },
        decision="S11 COMPLETE",
        action="Use s11_statistical_details.csv for supplementary statistical tables.",
        interpretation=(
            "Extracted Stage 1 GLM coefficient inference (estimate, SE, z, p, 95% CI), "
            "Gate G1 BH-adjusted p-value percentiles among significant close pairs, and "
            "system counts into a single long-format CSV for paper ingestion."
        ),
        paper_ref="Appendix / supplementary statistical details table",
    )


if __name__ == "__main__":
    main()
