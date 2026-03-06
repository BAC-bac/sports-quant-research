# scripts/19_monte_carlo_bankroll_model.py
"""
Step 19: Monte Carlo + bankroll modelling for best Step 18 variant.

Fix vs previous:
- Month-block bootstrap now samples months until reaching TARGET_DAYS,
  then trims to exactly TARGET_DAYS so arrays align.
- Sim paths write uses sim as a repeated array.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

# -------------------------
# CONFIG
# -------------------------
STEP18_DIR = Path("reports/bfsp/uk/greyhound/win/walkforward/step18_grid/tables")
BEST_DAILY_PATH = STEP18_DIR / "best_variant_daily_step18.parquet"
BEST_META_PATH = STEP18_DIR / "best_variant_meta_step18.json"

OUT_DIR = Path("reports/bfsp/uk/greyhound/win/walkforward/step19_monte_carlo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SIMS = 5000

# Bankrolls to evaluate "ruin" probability
# (ruin = equity <= -bankroll at any point, i.e. you hit 0)
BANKROLL_LEVELS = [200, 300, 500, 750, 1000, 1500, 2000]

# Optional: leverage / stake scaling
STAKE_MULT = 1.0

# Whether to save a sample of sim equity paths (can be big)
SAVE_SIM_PATHS = True
SAVE_PATHS_N = 250  # save only first N paths


# -------------------------
# HELPERS
# -------------------------
def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())

def worst_window(pnl: np.ndarray, window: int) -> float:
    if pnl.size == 0:
        return 0.0
    if pnl.size < window:
        return float(pnl.sum())
    s = pd.Series(pnl).rolling(window).sum()
    return float(s.min())

def time_to_recover_days(dates: np.ndarray, equity: np.ndarray) -> dict:
    """
    Time-to-recover (TTR) in days:
    For each drawdown episode, measure days from DD start to recovery.
    """
    if equity.size == 0:
        return {"median": np.nan, "p90": np.nan, "max": np.nan}

    peak = np.maximum.accumulate(equity)
    underwater = equity < peak

    ttrs = []
    i = 0
    n = equity.size
    while i < n:
        if not underwater[i]:
            i += 1
            continue
        dd_start = i
        target = peak[i]
        j = i
        while j < n and equity[j] < target:
            j += 1
        if j < n:
            delta_days = (pd.Timestamp(dates[j]) - pd.Timestamp(dates[dd_start])).days
            ttrs.append(delta_days)
        i = j + 1

    if not ttrs:
        return {"median": np.nan, "p90": np.nan, "max": np.nan}

    arr = np.asarray(ttrs, dtype=float)
    return {
        "median": float(np.nanmedian(arr)),
        "p90": float(np.nanpercentile(arr, 90)),
        "max": float(np.nanmax(arr)),
    }

def ruin_anytime(equity: np.ndarray, bankroll: float) -> bool:
    """
    equity is cumulative pnl (starting from 0).
    Ruin occurs if equity <= -bankroll at any point.
    """
    return bool((equity <= -bankroll).any())


# -------------------------
# LOAD
# -------------------------
def load_best_daily() -> pd.DataFrame:
    if not BEST_DAILY_PATH.exists():
        raise FileNotFoundError(f"Missing: {BEST_DAILY_PATH}")

    df = pd.read_parquet(BEST_DAILY_PATH)

    if "event_date" not in df.columns:
        # try index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "event_date"})
        else:
            raise ValueError("Could not find event_date column in best_variant_daily_step18.parquet")

    df["event_date"] = pd.to_datetime(df["event_date"])
    if "pnl" not in df.columns:
        raise ValueError("Expected 'pnl' column in best_variant_daily_step18.parquet")

    df = df.sort_values("event_date").reset_index(drop=True)
    df["pnl"] = df["pnl"].astype(float) * float(STAKE_MULT)
    df["equity"] = df["pnl"].cumsum()

    return df

def build_month_blocks(daily: pd.DataFrame) -> list[np.ndarray]:
    """
    Returns list of month pnl arrays, in chronological order.
    """
    d = daily.copy()
    d["month"] = d["event_date"].dt.to_period("M").astype(str)
    blocks = [g["pnl"].to_numpy(dtype=float) for _, g in d.groupby("month", sort=True)]
    return blocks


# -------------------------
# MONTE CARLO (Month Block Bootstrap)
# -------------------------
def simulate_path_month_blocks(
    blocks: list[np.ndarray],
    target_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample month blocks WITH replacement until total length >= target_days,
    then trim to exactly target_days.
    """
    out = []
    total = 0
    n_blocks = len(blocks)
    if n_blocks == 0:
        return np.array([], dtype=float)

    while total < target_days:
        i = int(rng.integers(0, n_blocks))
        b = blocks[i]
        out.append(b)
        total += b.size

    pnl = np.concatenate(out)
    if pnl.size > target_days:
        pnl = pnl[:target_days]
    return pnl


def run_month_block_bootstrap(daily: pd.DataFrame, n_sims: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    rng = np.random.default_rng(seed)

    blocks = build_month_blocks(daily)
    n_months = len(blocks)
    if n_months < 12:
        raise ValueError(f"Not enough months for block bootstrap (found {n_months}).")

    target_days = int(len(daily))
    dates = daily["event_date"].to_numpy()  # fixed calendar length

    sim_rows = []
    paths = []

    for s in range(n_sims):
        pnl = simulate_path_month_blocks(blocks, target_days=target_days, rng=rng)
        equity = np.cumsum(pnl)

        total = float(pnl.sum())
        dd = max_drawdown(equity)
        worst_20d = worst_window(pnl, 20)
        worst_5d = worst_window(pnl, 5)
        worst_1d = float(pnl.min()) if pnl.size else 0.0

        ttr = time_to_recover_days(dates, equity)

        ruin_cols = {f"ruin_p_{br}": float(ruin_anytime(equity, br)) for br in BANKROLL_LEVELS}

        sim_rows.append({
            "sim": s,
            "total_pnl": total,
            "max_dd": dd,
            "worst_1d": worst_1d,
            "worst_5d": worst_5d,
            "worst_20d": worst_20d,
            "ttr_p90_days": ttr["p90"],
            "ttr_max_days": ttr["max"],
            **ruin_cols,
        })

        if SAVE_SIM_PATHS and s < SAVE_PATHS_N:
            paths.append(pd.DataFrame({
                "event_date": dates,
                "pnl": pnl,
                "equity": equity,
                "sim": np.full(target_days, s, dtype=int),
            }))

    sims = pd.DataFrame(sim_rows)
    paths_df = pd.concat(paths, ignore_index=True) if (SAVE_SIM_PATHS and paths) else None
    return sims, paths_df


def summarize_sims(sims: pd.DataFrame) -> dict:
    def pct(series: pd.Series, q: float) -> float:
        return float(np.nanpercentile(series.to_numpy(dtype=float), q))

    out = {
        "n_sims": int(len(sims)),
        "stake_mult": float(STAKE_MULT),
        "terminal_pnl": {
            "mean": float(sims["total_pnl"].mean()),
            "p05": pct(sims["total_pnl"], 5),
            "p50": pct(sims["total_pnl"], 50),
            "p95": pct(sims["total_pnl"], 95),
        },
        "max_dd": {
            "mean": float(sims["max_dd"].mean()),
            "p05": pct(sims["max_dd"], 5),
            "p50": pct(sims["max_dd"], 50),
            "p95": pct(sims["max_dd"], 95),
        },
        "worst_20d": {
            "mean": float(sims["worst_20d"].mean()),
            "p05": pct(sims["worst_20d"], 5),
            "p50": pct(sims["worst_20d"], 50),
            "p95": pct(sims["worst_20d"], 95),
        },
    }

    for br in BANKROLL_LEVELS:
        out[f"ruin_prob_bankroll_{br}"] = float(sims[f"ruin_p_{br}"].mean())

    return out


# -------------------------
# MAIN
# -------------------------
def main():
    print("Loading best daily series from Step 18...")
    daily = load_best_daily()

    months = int(daily["event_date"].dt.to_period("M").nunique())
    print(f"Days: {len(daily):,} | Months: {months} | Total PnL: {daily['pnl'].sum():.2f}")

    print("\nRunning Monte Carlo month-block bootstrap...")
    sims, paths = run_month_block_bootstrap(daily, n_sims=N_SIMS, seed=SEED)

    sims_path = OUT_DIR / "mc_sims_step19.csv"
    sims.to_csv(sims_path, index=False)

    if paths is not None:
        paths_path = OUT_DIR / "mc_equity_paths_sample_step19.parquet"
        paths.to_parquet(paths_path, index=False)
        print(f"Saved sample equity paths: {paths_path}")

    summary = summarize_sims(sims)
    summary_path = OUT_DIR / "_SUMMARY_step19_monte_carlo.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nSaved sims: {sims_path}")
    print(f"Saved summary: {summary_path}")

    print("\n=== STEP 19 MONTE CARLO SUMMARY ===")
    tp = summary["terminal_pnl"]
    md = summary["max_dd"]
    w20 = summary["worst_20d"]

    print(f"Terminal PnL mean={tp['mean']:.2f} | p05={tp['p05']:.2f} | p50={tp['p50']:.2f} | p95={tp['p95']:.2f}")
    print(f"Max DD      mean={md['mean']:.2f} | p05={md['p05']:.2f} | p50={md['p50']:.2f} | p95={md['p95']:.2f}")
    print(f"Worst 20D   mean={w20['mean']:.2f} | p05={w20['p05']:.2f} | p50={w20['p50']:.2f} | p95={w20['p95']:.2f}")

    print("\nRuin probabilities (equity <= -bankroll at any time):")
    for br in BANKROLL_LEVELS:
        print(f"  bankroll={br:>5}: p_ruin={summary[f'ruin_prob_bankroll_{br}']:.4f}")


if __name__ == "__main__":
    main()