from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


# Where Step 10 saved the basket daily series
BASKET_DIR = Path("reports/bfsp/uk/greyhound/win/track_baskets")
OUT_DIR = BASKET_DIR / "diagnostics"


@dataclass
class Diag:
    file: str
    total_pnl: float
    n_days: int
    n_bets: int
    max_dd: float
    max_dd_pct: float
    ttr_median_days: float
    ttr_p90_days: float
    ttr_max_days: float
    streak_day_max: int
    streak_day_p90: float
    worst_1d: float
    worst_5d: float
    worst_20d: float


def _rolling_sum_min(x: pd.Series, window: int) -> float:
    if len(x) < window:
        return float(x.sum()) if len(x) > 0 else 0.0
    return float(x.rolling(window).sum().min())


def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity - peak


def _time_to_recovery_days(equity: pd.Series) -> np.ndarray:
    """
    For each drawdown episode, compute how many days it took to recover to a new high.
    Returns array of recovery lengths (in days).
    """
    eq = equity.to_numpy()
    peak = np.maximum.accumulate(eq)

    # drawdown boolean
    dd = eq < peak

    # Identify drawdown start indices (dd turns False->True)
    starts = np.where((~dd[:-1]) & (dd[1:]))[0] + 1
    if len(starts) == 0:
        return np.array([], dtype=int)

    # Identify recovery indices (dd turns True->False)
    ends = np.where((dd[:-1]) & (~dd[1:]))[0] + 1

    # If we end in drawdown, last episode has no recovery
    # We'll ignore unrecovered (common in long tails) by default
    rec_lengths = []
    for s in starts:
        e_candidates = ends[ends > s]
        if len(e_candidates) == 0:
            continue
        e = int(e_candidates[0])
        rec_lengths.append(e - s)

    return np.array(rec_lengths, dtype=int)


def _losing_streaks_daily(pnl: pd.Series) -> np.ndarray:
    """
    Losing streaks on DAILY pnl: consecutive days where pnl < 0.
    Returns streak lengths.
    """
    neg = (pnl < 0).to_numpy()
    if len(neg) == 0:
        return np.array([], dtype=int)

    streaks = []
    cur = 0
    for v in neg:
        if v:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
                cur = 0
    if cur > 0:
        streaks.append(cur)

    return np.array(streaks, dtype=int)


def diagnose_daily(daily: pd.DataFrame, file_label: str) -> Diag:
    # Expected columns: event_date, pnl, n_bets, equity
    if "equity" not in daily.columns:
        daily = daily.copy()
        daily["equity"] = daily["pnl"].cumsum()

    pnl = daily["pnl"].astype(float)
    equity = daily["equity"].astype(float)

    total_pnl = float(pnl.sum())
    n_days = int(len(daily))
    n_bets = int(daily["n_bets"].sum()) if "n_bets" in daily.columns else 0

    dd = _drawdown(equity)
    max_dd = float(dd.min()) if len(dd) else 0.0

    # dd% relative to running peak (avoid div-by-zero)
    peak = equity.cummax()
    denom = peak.replace(0, np.nan)
    dd_pct = (dd / denom) * 100.0
    max_dd_pct = float(dd_pct.min()) if dd_pct.notna().any() else 0.0

    # Time-to-recovery stats
    ttr = _time_to_recovery_days(equity)
    if len(ttr) == 0:
        ttr_median = 0.0
        ttr_p90 = 0.0
        ttr_max = 0.0
    else:
        ttr_median = float(np.median(ttr))
        ttr_p90 = float(np.percentile(ttr, 90))
        ttr_max = float(np.max(ttr))

    # Losing streak distribution (daily)
    streaks = _losing_streaks_daily(pnl)
    if len(streaks) == 0:
        streak_max = 0
        streak_p90 = 0.0
    else:
        streak_max = int(np.max(streaks))
        streak_p90 = float(np.percentile(streaks, 90))

    # Tail windows
    worst_1d = float(pnl.min()) if len(pnl) else 0.0
    worst_5d = _rolling_sum_min(pnl, 5)
    worst_20d = _rolling_sum_min(pnl, 20)

    return Diag(
        file=file_label,
        total_pnl=total_pnl,
        n_days=n_days,
        n_bets=n_bets,
        max_dd=max_dd,
        max_dd_pct=max_dd_pct,
        ttr_median_days=ttr_median,
        ttr_p90_days=ttr_p90,
        ttr_max_days=ttr_max,
        streak_day_max=streak_max,
        streak_day_p90=streak_p90,
        worst_1d=worst_1d,
        worst_5d=worst_5d,
        worst_20d=worst_20d,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Look for daily parquet outputs from Step 10
    daily_files = sorted(BASKET_DIR.glob("*_daily.parquet"))
    if not daily_files:
        raise RuntimeError(
            f"No *_daily.parquet files found in {BASKET_DIR}.\n"
            "Your Step 10 script may only be saving the SUMMARY csv.\n"
            "Fix Step 10 to write daily parquet per basket, then rerun."
        )

    rows = []
    json_out = {}

    for f in daily_files:
        daily = pd.read_parquet(f)
        d = diagnose_daily(daily, str(f))

        rows.append(d.__dict__)
        json_out[str(f)] = d.__dict__

        print(f"Diagnosed: {f.name}")

    df = pd.DataFrame(rows).sort_values("total_pnl", ascending=False)

    out_csv = OUT_DIR / "_SUMMARY_track_baskets_diagnostics.csv"
    df.to_csv(out_csv, index=False)

    out_json = OUT_DIR / "_SUMMARY_track_baskets_diagnostics.json"
    out_json.write_text(json.dumps(json_out, indent=2))

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")

    # Console-friendly view
    cols = [
        "file", "total_pnl", "n_bets", "n_days",
        "max_dd", "max_dd_pct",
        "ttr_median_days", "ttr_p90_days", "ttr_max_days",
        "streak_day_max", "streak_day_p90",
        "worst_1d", "worst_5d", "worst_20d"
    ]
    print("\n=== TOP RESULTS (sorted by total_pnl) ===")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()