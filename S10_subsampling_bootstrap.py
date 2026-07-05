"""
S10_subsampling_bootstrap.py — Network density sensitivity (station subsampling)
=================================================================================
Tests whether Gate G1 (significant close-pair fraction) and Stage 2 decay length
L̂ are stable when the number of retained stations is reduced by random subsampling.

For k ∈ {50, 80, 100, 120, 150} and b = 1..N_REPS:
  - Draw k stations without replacement (174 with valid coordinates)
  - Recompute G1 on training daily-block maxima (u = 0.95, block perm + BH)
  - Refit L̂ on matched events (β fixed at production value 0.399)

Outputs:
  results/gates/s10_subsampling_bootstrap.parquet
  results/gates/s10_subsampling_decision.md
  results/figures/s10_subsampling_stability.pdf
  paper/figures/s10_subsampling_stability.pdf  (copy for LaTeX)

Execute:
    python S10_subsampling_bootstrap.py
    python S10_subsampling_bootstrap.py --reps 20   # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.logger import log_result
from src.paper_figures import save_publication_figure
from A3_synthetic_tests import to_uniform
from C0_gate0 import haversine_matrix
from C1_gate1 import (
    build_daily_matrix,
    estimate_block_length,
    make_block_index_arrays,
)
from F5_heffernan_tawn_pilot import build_ecdf, laplace_transform
from F5_two_stage import alpha_exp

RAMPS_PQ = cfg.DIRS["interim"] / "ramps_split.parquet"
COORDS_PQ = cfg.DIRS["interim"] / "coords.parquet"
ALIGNED_PQ = cfg.DIRS["processed"] / "aligned_pairs.parquet"
EVENTPAIR_PQ = cfg.DIRS["gates"] / "event_pairing_summary.parquet"

OUT_PQ = cfg.DIRS["gates"] / "s10_subsampling_bootstrap.parquet"
OUT_MD = cfg.DIRS["gates"] / "s10_subsampling_decision.md"
OUT_FIG = cfg.DIRS["figures"] / "s10_subsampling_stability.pdf"
OUT_FIG_PAPER = cfg.ROOT / "paper" / "figures" / "s10_subsampling_stability.pdf"

K_VALUES = [50, 80, 100, 120, 150]
N_REPS_DEFAULT = 100
U_DECISION = 0.95
MAX_DIST_KM = cfg.G1["max_pair_dist_km"]
N_PERM = 199
FDR_ALPHA = cfg.G1["fdr_alpha"]
GO_THRESHOLD = cfg.G1["go_threshold"]
MIN_CLOSE_PAIRS = 30
MIN_MATCHED = 500
BETA_FIXED = 0.399
L_PROD = 3.72
L_CI = (3.41, 4.02)
FRAC_PROD = 0.521
SEED = cfg.SEED
SEP = "─" * 60


def ht_negloglik_fixed_beta(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dist_km: np.ndarray,
    beta: float,
) -> float:
    alpha0, log_L, mu, log_sigma = params
    alpha_d = alpha_exp(dist_km, alpha0, log_L)
    sigma = np.exp(log_sigma)
    scale = np.power(x, beta)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        return 1e10
    resid = (y - alpha_d * x - scale * mu) / (scale * sigma)
    ll = -np.log(scale * sigma) - 0.5 * resid ** 2
    if not np.all(np.isfinite(ll)):
        return 1e10
    return float(-np.sum(ll))


def fit_L_fixed_beta(
    x: np.ndarray,
    y: np.ndarray,
    dist_km: np.ndarray,
    beta: float = BETA_FIXED,
) -> tuple[float, float, bool]:
    decay0 = np.log(L_PROD)
    resid0 = y - 0.564 * x
    x0 = np.array([
        0.564,
        decay0,
        float(np.mean(resid0)),
        float(np.log(max(float(np.std(resid0)), 1e-3))),
    ])
    res = minimize(
        ht_negloglik_fixed_beta,
        x0,
        args=(x, y, dist_km, beta),
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 3000, "maxfev": 3000},
    )
    alpha0_hat, log_L_hat = res.x[0], res.x[1]
    return alpha0_hat, float(np.exp(log_L_hat)), bool(res.success)


def bh_significant(p_values: np.ndarray, alpha: float) -> np.ndarray:
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p_values)
    ranked = p_values[order]
    bh_thresh = (np.arange(1, m + 1) / m) * alpha
    below = ranked <= bh_thresh
    if below.any():
        p_crit = ranked[np.max(np.where(below)[0])]
    else:
        p_crit = 0.0
    return p_values <= p_crit


def run_g1_subsample(
    M: np.ndarray,
    stations: list[str],
    lat: np.ndarray,
    lon: np.ndarray,
    block_len: int,
    perm_idx: np.ndarray,
) -> dict:
    n_days, n_stations = M.shape
    if n_stations < 2:
        return {"status": "insufficient_stations"}

    U = np.empty_like(M)
    for j in range(n_stations):
        U[:, j] = to_uniform(M[:, j])

    dist_mat = haversine_matrix(lat, lon)
    i_idx, j_idx = np.triu_indices(n_stations, k=1)
    dist_ij_m = dist_mat[i_idx, j_idx]
    close_mask = dist_ij_m < (MAX_DIST_KM * 1000)
    i_close = i_idx[close_mask]
    j_close = j_idx[close_mask]
    n_close = int(close_mask.sum())
    if n_close < MIN_CLOSE_PAIRS:
        return {"status": "insufficient_close_pairs", "n_pairs_close": n_close}

    exceed_dec = U > U_DECISION
    chi_obs = np.empty(n_close)
    p_values = np.empty(n_close)

    for k in range(n_close):
        ei = exceed_dec[:, i_close[k]]
        ej = exceed_dec[:, j_close[k]]
        n_joint = int(np.sum(ei & ej))
        chi_obs[k] = n_joint / (n_days * (1 - U_DECISION))
        ej_perm = ej[perm_idx]
        perm_counts = (ei[None, :] & ej_perm).sum(axis=1)
        p_values[k] = (1 + np.sum(perm_counts >= n_joint)) / (N_PERM + 1)

    significant = bh_significant(p_values, FDR_ALPHA)
    n_sig = int(significant.sum())
    frac_sig = n_sig / n_close
    if n_sig > 0:
        chi_med = float(np.median(chi_obs[significant]))
    else:
        chi_med = float("nan")

    return {
        "status": "ok",
        "n_pairs_close": n_close,
        "n_pairs_significant": n_sig,
        "frac_significant": frac_sig,
        "chi_hat_median": chi_med,
    }


def prepare_stage2_rows(
    aligned: pd.DataFrame,
    station_set: set[str],
    ecdf_by_station: dict,
    dist_lookup: dict,
) -> pd.DataFrame:
    sub = aligned[
        aligned["station_ext"].isin(station_set)
        & aligned["station_partner"].isin(station_set)
        & aligned["matched"]
    ].copy()
    if sub.empty:
        return sub

    x_lap = np.full(len(sub), np.nan)
    y_lap = np.full(len(sub), np.nan)
    for sid, idx in sub.groupby("station_ext").groups.items():
        pos = sub.index.get_indexer(idx)
        if sid in ecdf_by_station:
            x_lap[pos] = laplace_transform(
                ecdf_by_station[sid](sub.loc[idx, "mag_ext"].to_numpy())
            )
    for sid, idx in sub.groupby("station_partner").groups.items():
        pos = sub.index.get_indexer(idx)
        if sid in ecdf_by_station:
            y_lap[pos] = laplace_transform(
                ecdf_by_station[sid](sub.loc[idx, "mag_partner"].to_numpy())
            )

    dist_km = np.array([
        dist_lookup.get((a, b), np.nan)
        for a, b in zip(sub["station_ext"], sub["station_partner"])
    ])
    sub = sub.assign(x_lap=x_lap, y_lap=y_lap, dist_km=dist_km)
    return sub[
        np.isfinite(sub["x_lap"])
        & np.isfinite(sub["y_lap"])
        & np.isfinite(sub["dist_km"])
    ]


def run_stage2_subsample(stage_rows: pd.DataFrame) -> dict:
    n_matched = len(stage_rows)
    if n_matched < MIN_MATCHED:
        return {"status": "insufficient_matched", "n_matched_events": n_matched}

    x = stage_rows["x_lap"].to_numpy()
    y = stage_rows["y_lap"].to_numpy()
    d = stage_rows["dist_km"].to_numpy()
    alpha0, L_hat, ok = fit_L_fixed_beta(x, y, d, beta=BETA_FIXED)
    if not ok or not np.isfinite(L_hat):
        return {"status": "fit_failed", "n_matched_events": n_matched}

    return {
        "status": "ok",
        "n_matched_events": n_matched,
        "L_hat": L_hat,
        "alpha0_hat": alpha0,
    }


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    rows = []
    for k in K_VALUES:
        g = ok[ok["k"] == k]
        rows.append({
            "k": k,
            "n_ok": len(g),
            "n_failed": len(df[(df["k"] == k) & (df["status"] != "ok")]),
            "frac_sig_median": g["frac_significant"].median(),
            "frac_sig_q25": g["frac_significant"].quantile(0.25),
            "frac_sig_q75": g["frac_significant"].quantile(0.75),
            "L_median": g["L_hat"].median(),
            "L_q25": g["L_hat"].quantile(0.25),
            "L_q75": g["L_hat"].quantile(0.75),
        })
    return pd.DataFrame(rows)


def assess_decision(summary: pd.DataFrame) -> tuple[str, list[str]]:
    notes: list[str] = []
    stable = True

    for _, row in summary.iterrows():
        k = int(row["k"])
        if row["n_ok"] == 0:
            notes.append(f"k={k}: no successful replicates.")
            stable = False
            continue
        frac_med = row["frac_sig_median"]
        L_med = row["L_median"]
        if frac_med < GO_THRESHOLD:
            notes.append(
                f"k={k}: median G1 fraction {frac_med:.1%} below gate ({GO_THRESHOLD:.0%})."
            )
            if k >= 80:
                stable = False
        if k >= 80 and (L_med < L_CI[0] or L_med > L_CI[1]):
            notes.append(
                f"k={k}: median L̂={L_med:.2f} km outside production CI [{L_CI[0]}, {L_CI[1]}]."
            )
            stable = False

    for k in [80, 100, 120, 150]:
        sub = summary[summary["k"] == k]
        if len(sub) and sub.iloc[0]["n_ok"] > 0:
            continue
        notes.append(f"k={k}: insufficient successful replicates for stability check.")
        stable = False

    decision = "estável" if stable else "instável"
    return decision, notes


def plot_stability(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    ok = df[df["status"] == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for k in K_VALUES:
        g = ok[ok["k"] == k]["frac_significant"]
        if g.empty:
            continue
        med = g.median()
        q25, q75 = g.quantile(0.25), g.quantile(0.75)
        ax.errorbar(k, med, yerr=[[med - q25], [q75 - med]], fmt="o", color="#2980b9",
                     capsize=4, markersize=6)
    ax.axhline(FRAC_PROD, color="#c0392b", ls="--", lw=1, label=f"Production ({FRAC_PROD:.1%})")
    ax.axhline(GO_THRESHOLD, color="#888888", ls="--", lw=1, label=f"Gate G1 ({GO_THRESHOLD:.0%})")
    ax.set_xlabel("Number of stations ($k$)")
    ax.set_ylabel("Fraction of significant close pairs")
    ax.set_title("(a) Gate G1 fraction vs. $k$")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_ylim(0, max(0.65, ok["frac_significant"].max() * 1.05))

    ax = axes[1]
    for k in K_VALUES:
        g = ok[ok["k"] == k]["L_hat"]
        if g.empty:
            continue
        med = g.median()
        q25, q75 = g.quantile(0.25), g.quantile(0.75)
        ax.errorbar(k, med, yerr=[[med - q25], [q75 - med]], fmt="o", color="#27ae60",
                     capsize=4, markersize=6)
    ax.axhline(L_PROD, color="#c0392b", ls="--", lw=1, label=f"Production ({L_PROD:.2f} km)")
    ax.axhspan(L_CI[0], L_CI[1], color="#c0392b", alpha=0.12, label="Production 95\\% CI")
    ax.set_xlabel("Number of stations ($k$)")
    ax.set_ylabel(r"Decay length $\hat{L}$ (km)")
    ax.set_title(r"(b) $\hat{L}$ vs. $k$")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    save_publication_figure(fig, OUT_FIG)
    OUT_FIG_PAPER.parent.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, OUT_FIG_PAPER)
    plt.close(fig)


def write_decision_md(df: pd.DataFrame, summary: pd.DataFrame, decision: str, notes: list[str]) -> None:
    lines = [
        "# S10 — Network density sensitivity (station subsampling bootstrap)",
        "",
        f"**Date:** {date.today().isoformat()}",
        "",
        "## Design",
        f"- Station pool: 174 systems with valid coordinates (training set 2014–2016 only)",
        f"- Subsample sizes k ∈ {K_VALUES}; {df['replicate'].nunique()} replicates per k",
        f"- G1: daily-block $\\hat{{\\chi}}(u={U_DECISION})$, block permutation (N={N_PERM}), BH α={FDR_ALPHA}",
        f"- Stage 2: exponential $\\hat{{L}}$ with β fixed at {BETA_FIXED}; min {MIN_MATCHED} matched events",
        "",
        "## Summary (medians and IQR over successful replicates)",
        "",
        "| k | OK reps | Failed | frac_sig median [IQR] | L̂ median [IQR] (km) |",
        "|---|---:|---:|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['k'])} | {int(row['n_ok'])} | {int(row['n_failed'])} | "
            f"{row['frac_sig_median']:.1%} [{row['frac_sig_q25']:.1%}, {row['frac_sig_q75']:.1%}] | "
            f"{row['L_median']:.2f} [{row['L_q25']:.2f}, {row['L_q75']:.2f}] |"
        )

    lines.extend([
        "",
        "## Failure counts by status",
        "",
    ])
    fail = df[df["status"] != "ok"].groupby(["k", "status"]).size().reset_index(name="n")
    if fail.empty:
        lines.append("No failed replicates.")
    else:
        for _, r in fail.iterrows():
            lines.append(f"- k={int(r['k'])}: {r['status']} × {int(r['n'])}")

    lines.extend([
        "",
        f"## Decision: **{decision.upper()}**",
        "",
        "Criterion: stable if G1 fraction stays above 30% and median L̂ remains within "
        f"the production 95% CI [{L_CI[0]}, {L_CI[1]}] km for k ≥ 80.",
        "",
    ])
    if notes:
        lines.append("### Notes")
        for n in notes:
            lines.append(f"- {n}")

    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=N_REPS_DEFAULT,
                        help=f"Replicates per k (default {N_REPS_DEFAULT})")
    args = parser.parse_args()
    n_reps = args.reps

    print(SEP)
    print("S10 — NETWORK DENSITY SENSITIVITY (STATION SUBSAMPLING)")
    print(SEP)

    for p in (RAMPS_PQ, COORDS_PQ, ALIGNED_PQ, EVENTPAIR_PQ):
        if not p.exists():
            print(f"ERRO: {p} não encontrado.")
            sys.exit(1)

    t0 = time.time()
    ramps_all = pd.read_parquet(RAMPS_PQ)
    ramps_train = ramps_all[ramps_all["split"] == "train"].copy()
    ramps_train["abs_mag"] = ramps_train["delta_k"].abs()

    coords = pd.read_parquet(COORDS_PQ).dropna(subset=["lat_centroid", "lon_centroid"])
    station_pool = sorted(coords["station_id"].astype(str).tolist())
    coords = coords.set_index("station_id")

    aligned = pd.read_parquet(ALIGNED_PQ)
    event_summary = pd.read_parquet(EVENTPAIR_PQ)

    ecdf_by_station = {}
    for sid, g in ramps_train.groupby("station_id"):
        ecdf_by_station[str(sid)] = build_ecdf(np.sort(g["abs_mag"].to_numpy()))

    dist_lookup: dict[tuple[str, str], float] = {}
    for _, row in event_summary.iterrows():
        d_km = row["dist_ij_m"] / 1000.0
        si, sj = str(row["station_i"]), str(row["station_j"])
        dist_lookup[(si, sj)] = d_km
        dist_lookup[(sj, si)] = d_km

    rng_master = np.random.default_rng(SEED)
    records: list[dict] = []

    print(f"\n  Station pool: n={len(station_pool)}")
    print(f"  Replicates per k: {n_reps}")
    print(f"  k values: {K_VALUES}\n")

    for k in K_VALUES:
        print(f"[k={k}]")
        for b in range(1, n_reps + 1):
            draw = rng_master.choice(station_pool, size=k, replace=False)
            station_set = set(draw.tolist())

            ramps_sub = ramps_train[ramps_train["station_id"].astype(str).isin(station_set)]
            M, _T, _days, stations_raw = build_daily_matrix(ramps_sub)
            stations = sorted(str(s) for s in stations_raw if str(s) in station_set and str(s) in coords.index)
            if len(stations) < 2:
                records.append({
                    "k": k, "replicate": b, "n_stations": len(stations),
                    "n_pairs_close": 0, "n_pairs_significant": 0,
                    "frac_significant": np.nan, "chi_hat_median": np.nan,
                    "L_hat": np.nan, "alpha0_hat": np.nan,
                    "n_matched_events": 0, "status": "insufficient_stations",
                })
                continue

            idx_map = {str(s): i for i, s in enumerate(stations_raw)}
            cols = [idx_map[s] for s in stations]
            M = M[:, cols]

            lat = coords.loc[stations, "lat_centroid"].to_numpy(dtype=float)
            lon = coords.loc[stations, "lon_centroid"].to_numpy(dtype=float)

            block_len = estimate_block_length(M)
            perm_rng = np.random.default_rng(SEED + k * 10_000 + b)
            perm_idx = make_block_index_arrays(M.shape[0], block_len, N_PERM, perm_rng)

            g1 = run_g1_subsample(M, stations, lat, lon, block_len, perm_idx)
            stage_rows = prepare_stage2_rows(aligned, station_set, ecdf_by_station, dist_lookup)
            s2 = run_stage2_subsample(stage_rows)

            if g1.get("status") != "ok":
                status = g1["status"]
            elif s2.get("status") != "ok":
                status = s2["status"]
            else:
                status = "ok"

            rec = {
                "k": k,
                "replicate": b,
                "n_stations": len(stations),
                "n_pairs_close": g1.get("n_pairs_close", 0),
                "n_pairs_significant": g1.get("n_pairs_significant", 0),
                "frac_significant": g1.get("frac_significant", np.nan),
                "chi_hat_median": g1.get("chi_hat_median", np.nan),
                "L_hat": s2.get("L_hat", np.nan),
                "alpha0_hat": s2.get("alpha0_hat", np.nan),
                "n_matched_events": s2.get("n_matched_events", len(stage_rows)),
                "status": status,
            }
            records.append(rec)

            if b % max(1, n_reps // 5) == 0 or b == n_reps:
                ok_so_far = sum(1 for r in records if r["k"] == k and r["status"] == "ok")
                print(f"  replicate {b}/{n_reps} — OK so far: {ok_so_far}")

    df = pd.DataFrame(records)
    cfg.DIRS["gates"].mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PQ, index=False)
    print(f"\n  Saved: {OUT_PQ.relative_to(cfg.ROOT)}")

    summary = build_summary_table(df)
    decision, notes = assess_decision(summary)
    write_decision_md(df, summary, decision, notes)
    print(f"  Saved: {OUT_MD.relative_to(cfg.ROOT)}")

    plot_stability(df, summary)
    print(f"  Saved: {OUT_FIG.relative_to(cfg.ROOT)}")
    print(f"  Saved: {OUT_FIG_PAPER.relative_to(cfg.ROOT)}")

    elapsed = time.time() - t0
    print(f"\n  Decision: {decision.upper()}")
    print(f"  Elapsed: {elapsed / 60:.1f} min")

    log_result(
        script="S10_subsampling_bootstrap.py",
        gate="S10",
        params={
            "k_values": K_VALUES,
            "n_reps": n_reps,
            "u": U_DECISION,
            "n_perm": N_PERM,
            "beta_fixed": BETA_FIXED,
            "min_close_pairs": MIN_CLOSE_PAIRS,
            "min_matched": MIN_MATCHED,
        },
        results={
            "decision": decision,
            "summary": summary.to_dict(orient="records"),
            "n_total_rows": len(df),
            "n_ok": int((df["status"] == "ok").sum()),
        },
        decision=f"S10 {decision.upper()}",
        action="Integrate Appendix A.6 and Discussion §5.4 per s10_subsampling_decision.md",
        interpretation=(
            f"Subsampling bootstrap ({n_reps} reps × {len(K_VALUES)} k values): "
            f"decision={decision}; see s10_subsampling_decision.md."
        ),
    )


if __name__ == "__main__":
    main()
