# scripts/20_adversarial_monte_carlo_degraded.py
"""
Step 20 (corrected): Adversarial / degraded Monte Carlo on the Step 18 "best variant" daily series.

Fixes vs previous:
1) Quarter blocks are now NON-OVERLAPPING calendar quarters (Period('Q')),
   not rolling 3-month windows (which caused overlapping and inflated target length).
2) total_days_target is now ALWAYS the original len(daily) horizon (apples-to-apples).

Still includes:
- per-bet drag: daily_pnl_adj = daily_pnl - drag_per_bet * daily_n_bets
- adversarial reweighting: bottom quantile blocks get higher weight
- month or quarter block bootstrap with replacement
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path("reports/bfsp/uk/greyhound/win/walkforward")
STEP18_DIR = ROOT / "step18_grid" / "tables"
IN_DAILY = STEP18_DIR / "best_variant_daily_step18.parquet"

OUT_DIR = ROOT / "step20_adversarial_mc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SIMS = 5000

# Bankrolls to test "ruin" probabilities: equity <= -bankroll at any time
BANKROLLS = [200, 300, 500, 750, 1000, 1500, 2000]

# Drag settings (units per bet)
DRAG_GRID = [0.00, 0.005, 0.01, 0.02]

# Reweighting settings
BAD_QUANTILE = 0.25        # bottom 25% blocks are "bad"
BAD_WEIGHT_MULT = 3.0      # bad blocks get 3x sampling probability
GOOD_WEIGHT_MULT = 1.0     # others get 1x

# Block settings
BLOCK_MODES = ["month", "quarter"]  # month = calendar months, quarter = calendar quarters (non-overlapping)

# Save a few sample paths for inspection
N_SAMPLE_PATHS = 30


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"event_date", "pnl", "n_bets"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Input parquet missing columns: {sorted(missing)}")

    out = df.copy()
    out["event_date"] = pd.to_datetime(out["event_date"])
    out = out.sort_values("event_date").reset_index(drop=True)
    return out


def max_drawdown_from_pnl(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(dd.min())


def worst_rolling_window(pnl: np.ndarray, window: int) -> float:
    if len(pnl) == 0:
        return 0.0
    if len(pnl) < window:
        return float(np.sum(pnl))
    s = pd.Series(pnl).rolling(window).sum()
    return float(s.min())


def build_blocks(daily: pd.DataFrame, block_mode: str) -> list[dict]:
    """
    Returns list of blocks:
      block = {"id": str, "dates": np.ndarray, "pnl": np.ndarray, "n_bets": np.ndarray, "block_pnl": float}
    """
    d = daily.copy()

    if block_mode == "month":
        d["block_id"] = d["event_date"].dt.to_period("M").astype(str)

    elif block_mode == "quarter":
        # NON-overlapping calendar quarters
        d["block_id"] = d["event_date"].dt.to_period("Q").astype(str)

    else:
        raise ValueError(f"Unknown block_mode: {block_mode}")

    blocks = []
    for bid, g in d.groupby("block_id", sort=True):
        blocks.append({
            "id": bid,
            "dates": g["event_date"].to_numpy(),
            "pnl": g["pnl_adj"].to_numpy(dtype=float),
            "n_bets": g["n_bets"].to_numpy(dtype=int),
            "block_pnl": float(g["pnl_adj"].sum()),
        })
    return blocks


def build_sampling_weights(
    blocks: list[dict],
    bad_quantile: float,
    bad_mult: float,
    good_mult: float
) -> np.ndarray:
    """
    Reweight blocks based on block_pnl: bottom quantile -> bad_mult, else -> good_mult.
    """
    pnl_vals = np.array([b["block_pnl"] for b in blocks], dtype=float)
    if len(pnl_vals) == 0:
        return np.array([], dtype=float)

    thresh = np.quantile(pnl_vals, bad_quantile)
    w = np.where(pnl_vals <= thresh, bad_mult, good_mult).astype(float)
    w = w / w.sum()
    return w


def simulate_paths(
    blocks: list[dict],
    weights: np.ndarray,
    total_days_target: int,
    n_sims: int,
    seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample blocks with replacement to match ORIGINAL horizon total_days_target (len(daily)).
    Returns:
      sims_df: per-sim summary rows
      paths_sample: concatenated sample equity paths (subset) for inspection
    """
    rng = np.random.default_rng(seed)

    if total_days_target <= 0:
        raise RuntimeError("total_days_target must be > 0")
    if len(blocks) == 0:
        raise RuntimeError("No blocks. Check input data / filtering.")
    if len(weights) != len(blocks) or not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("Weights invalid. Check weighting logic.")

    sims_rows = []
    sample_paths = []

    keep_sims = set(
        rng.choice(np.arange(n_sims), size=min(N_SAMPLE_PATHS, n_sims), replace=False).tolist()
    )

    for s in range(n_sims):
        pnl_parts = []
        date_parts = []
        n_days = 0

        while n_days < total_days_target:
            idx = int(rng.choice(len(blocks), p=weights))
            b = blocks[idx]
            pnl_parts.append(b["pnl"])
            date_parts.append(b["dates"])
            n_days += len(b["pnl"])

        pnl = np.concatenate(pnl_parts)[:total_days_target]
        dates = np.concatenate(date_parts)[:total_days_target]

        equity = np.cumsum(pnl)
        terminal = float(equity[-1]) if len(equity) else 0.0
        max_dd = max_drawdown_from_pnl(pnl)
        worst_20d = worst_rolling_window(pnl, 20)

        ruin = {f"ruin_{b}": int(np.any(equity <= -float(b))) for b in BANKROLLS}

        sims_rows.append({
            "sim": s,
            "terminal_pnl": terminal,
            "max_dd": max_dd,
            "worst_20d": worst_20d,
            **ruin,
        })

        if s in keep_sims:
            sample_paths.append(pd.DataFrame({
                "event_date": pd.to_datetime(dates),
                "pnl": pnl,
                "equity": equity,
                "sim": s
            }))

    sims_df = pd.DataFrame(sims_rows)
    paths_sample = (
        pd.concat(sample_paths, ignore_index=True)
        if sample_paths else
        pd.DataFrame(columns=["event_date", "pnl", "equity", "sim"])
    )

    return sims_df, paths_sample


def summarize_sims(sims: pd.DataFrame) -> dict:
    def pct(x, q):
        return float(np.quantile(x, q))

    terminal = sims["terminal_pnl"].to_numpy(dtype=float)
    maxdd = sims["max_dd"].to_numpy(dtype=float)
    w20 = sims["worst_20d"].to_numpy(dtype=float)

    summary = {
        "terminal_pnl": {
            "mean": float(np.mean(terminal)),
            "p05": pct(terminal, 0.05),
            "p50": pct(terminal, 0.50),
            "p95": pct(terminal, 0.95),
        },
        "max_dd": {
            "mean": float(np.mean(maxdd)),
            "p05": pct(maxdd, 0.05),
            "p50": pct(maxdd, 0.50),
            "p95": pct(maxdd, 0.95),
        },
        "worst_20d": {
            "mean": float(np.mean(w20)),
            "p05": pct(w20, 0.05),
            "p50": pct(w20, 0.50),
            "p95": pct(w20, 0.95),
        },
        "ruin_probs": {},
        "n_sims": int(len(sims)),
    }

    for b in BANKROLLS:
        col = f"ruin_{b}"
        summary["ruin_probs"][str(b)] = float(sims[col].mean()) if col in sims.columns else np.nan

    return summary


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("Loading best daily series from Step 18...")
    daily = load_daily(IN_DAILY)

    total_days_target = int(len(daily))
    total_months = int(daily["event_date"].dt.to_period("M").nunique())
    print(f"Days: {total_days_target:,} | Months: {total_months} | Total PnL: {daily['pnl'].sum():.2f}")

    results_rows = []
    summaries = {}

    for drag in DRAG_GRID:
        d = daily.copy()
        d["pnl_adj"] = d["pnl"].astype(float) - float(drag) * d["n_bets"].astype(float)

        for block_mode in BLOCK_MODES:
            blocks = build_blocks(d, block_mode=block_mode)
            weights = build_sampling_weights(
                blocks,
                bad_quantile=BAD_QUANTILE,
                bad_mult=BAD_WEIGHT_MULT,
                good_mult=GOOD_WEIGHT_MULT
            )

            tag = f"drag={drag:.3f}|block={block_mode}|badq={BAD_QUANTILE}|badx={BAD_WEIGHT_MULT:g}"
            print("\n" + "=" * 70)
            print(f"Running Step 20 adversarial MC: {tag}")
            print(f"Blocks: {len(blocks):,} | Target days: {total_days_target:,} (fixed)")

            sims_df, paths_sample = simulate_paths(
                blocks=blocks,
                weights=weights,
                total_days_target=total_days_target,
                n_sims=N_SIMS,
                seed=SEED
            )

            safe_tag = tag.replace("=", "_").replace("|", "__").replace(":", "_")
            sims_path = OUT_DIR / f"mc_sims_step20__{safe_tag}.csv"
            paths_path = OUT_DIR / f"mc_paths_sample_step20__{safe_tag}.parquet"

            sims_df.to_csv(sims_path, index=False)
            paths_sample.to_parquet(paths_path, index=False)

            summ = summarize_sims(sims_df)
            summaries[tag] = {
                "tag": tag,
                "drag_per_bet": drag,
                "block_mode": block_mode,
                "bad_quantile": BAD_QUANTILE,
                "bad_weight_mult": BAD_WEIGHT_MULT,
                "good_weight_mult": GOOD_WEIGHT_MULT,
                "summary": summ,
                "files": {
                    "sims_csv": str(sims_path),
                    "paths_sample_parquet": str(paths_path),
                }
            }

            t = summ["terminal_pnl"]
            dd = summ["max_dd"]
            w20 = summ["worst_20d"]
            print(f"Terminal PnL: mean={t['mean']:.2f} | p05={t['p05']:.2f} | p50={t['p50']:.2f} | p95={t['p95']:.2f}")
            print(f"Max DD     : mean={dd['mean']:.2f} | p05={dd['p05']:.2f} | p50={dd['p50']:.2f} | p95={dd['p95']:.2f}")
            print(f"Worst 20D  : mean={w20['mean']:.2f} | p05={w20['p05']:.2f} | p50={w20['p50']:.2f} | p95={w20['p95']:.2f}")

            ruin = summ["ruin_probs"]
            print("Ruin probs:")
            for b in BANKROLLS:
                print(f"  bankroll={b:>5}: p_ruin={ruin[str(b)]:.4f}")

            results_rows.append({
                "variant": tag,
                "drag_per_bet": drag,
                "block_mode": block_mode,
                "terminal_mean": t["mean"],
                "terminal_p05": t["p05"],
                "terminal_p50": t["p50"],
                "terminal_p95": t["p95"],
                "maxdd_mean": dd["mean"],
                "maxdd_p05": dd["p05"],
                "maxdd_p50": dd["p50"],
                "maxdd_p95": dd["p95"],
                "worst20_mean": w20["mean"],
                "worst20_p05": w20["p05"],
                "worst20_p50": w20["p50"],
                "worst20_p95": w20["p95"],
                **{f"p_ruin_{b}": ruin[str(b)] for b in BANKROLLS},
                "sims_csv": str(sims_path),
                "paths_sample_parquet": str(paths_path),
            })

    leaderboard = pd.DataFrame(results_rows)

    # Simple robustness sort: prefer high p05 and low ruin_500
    if "p_ruin_500" in leaderboard.columns:
        leaderboard["robust_score"] = leaderboard["terminal_p05"] - 1000.0 * leaderboard["p_ruin_500"]
    else:
        leaderboard["robust_score"] = leaderboard["terminal_p05"]

    leaderboard = leaderboard.sort_values(["robust_score", "terminal_p05"], ascending=False)

    lb_path = OUT_DIR / "mc_step20_leaderboard.csv"
    leaderboard.to_csv(lb_path, index=False)

    json_path = OUT_DIR / "_SUMMARY_step20_adversarial_mc.json"
    json_path.write_text(json.dumps(summaries, indent=2))

    print("\n=== STEP 20 COMPLETE ===")
    print(f"Saved leaderboard: {lb_path}")
    print(f"Saved summary JSON: {json_path}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()