from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd


# -----------------------------
# Paths (Greyhound WIN)
# -----------------------------
BIN_SWEEPS_DIR = Path("reports/bfsp/uk/greyhound/win/bin_sweeps")
OUT_DIR = Path("reports/bfsp/uk/greyhound/win/diagnostics")


# -----------------------------
# Helpers / Metrics
# -----------------------------
def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling sum with same length; first (window-1) are nan."""
    if window <= 1:
        return x.astype(float)
    out = np.full_like(x, np.nan, dtype=float)
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out[window - 1 :] = c[window:] - c[:-window]
    return out


def max_drawdown(equity: np.ndarray) -> tuple[float, float]:
    """
    Returns:
      max_dd (negative number),
      max_dd_pct (negative number; dd / peak_equity_abs)
    Uses equity curve that starts at 0.
    """
    eq = equity.astype(float)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = float(np.min(dd)) if len(dd) else 0.0

    # Percent dd relative to peak-to-date *absolute* capital proxy
    # If peak is 0 early on, avoid divide-by-zero by using max(1, peak_abs)
    peak_abs = np.maximum(1.0, np.abs(peak))
    dd_pct = dd / peak_abs
    max_dd_pct = float(np.min(dd_pct)) if len(dd_pct) else 0.0
    return max_dd, max_dd_pct


def time_to_recovery_days(equity: np.ndarray) -> np.ndarray:
    """
    For each drawdown episode: how many days to regain the prior peak.
    Returns array of recovery lengths in days.
    If never recovered by end, uses remaining length (so it shows fragility).
    """
    eq = equity.astype(float)
    n = len(eq)
    if n == 0:
        return np.array([], dtype=float)

    peak = np.maximum.accumulate(eq)
    rec_times = []

    i = 0
    while i < n:
        # Find start of drawdown: equity below peak
        if eq[i] >= peak[i]:
            i += 1
            continue

        # We're in drawdown; the peak level to recover to:
        target = peak[i]
        start = i

        # Advance until recovered (>= target) or end
        i += 1
        while i < n and eq[i] < target:
            i += 1

        if i < n:
            # recovered at i
            rec_times.append(i - start)
        else:
            # not recovered by end
            rec_times.append(n - start)

    return np.array(rec_times, dtype=float)


def losing_streaks_by_day(pnl: np.ndarray) -> np.ndarray:
    """
    Returns lengths of consecutive losing-day streaks (pnl < 0).
    """
    streaks = []
    run = 0
    for v in pnl:
        if v < 0:
            run += 1
        else:
            if run > 0:
                streaks.append(run)
                run = 0
    if run > 0:
        streaks.append(run)
    return np.array(streaks, dtype=float)


def tail_windows(pnl: np.ndarray) -> dict:
    """
    Tail behaviour based on rolling window sums.
    """
    pnl = pnl.astype(float)
    out = {
        "worst_1d": float(np.min(pnl)) if len(pnl) else 0.0,
        "worst_5d": float(np.nanmin(_rolling_sum(pnl, 5))) if len(pnl) >= 5 else float(np.sum(pnl)),
        "worst_20d": float(np.nanmin(_rolling_sum(pnl, 20))) if len(pnl) >= 20 else float(np.sum(pnl)),
    }
    return out


def diagnose_daily_df(daily: pd.DataFrame) -> dict:
    """
    daily expects:
      event_date (sortable), pnl, equity, n_bets
    """
    daily = daily.sort_values("event_date").reset_index(drop=True)

    pnl = daily["pnl"].to_numpy(dtype=float)
    equity = daily["equity"].to_numpy(dtype=float)

    total_pnl = float(np.sum(pnl))
    n_days = int(len(daily))
    n_bets = int(daily["n_bets"].sum()) if "n_bets" in daily.columns else int(n_days)

    max_dd, max_dd_pct = max_drawdown(equity)

    # Time-to-recovery stats
    ttr = time_to_recovery_days(equity)
    ttr_median = float(np.median(ttr)) if len(ttr) else 0.0
    ttr_p90 = float(np.quantile(ttr, 0.9)) if len(ttr) else 0.0
    ttr_max = float(np.max(ttr)) if len(ttr) else 0.0

    # Losing streaks (day-level)
    streaks = losing_streaks_by_day(pnl)
    streak_max = float(np.max(streaks)) if len(streaks) else 0.0
    streak_p90 = float(np.quantile(streaks, 0.9)) if len(streaks) else 0.0

    tails = tail_windows(pnl)

    return {
        "total_pnl": total_pnl,
        "n_bets": n_bets,
        "n_days": n_days,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "ttr_median_days": ttr_median,
        "ttr_p90_days": ttr_p90,
        "ttr_max_days": ttr_max,
        "streak_day_max": streak_max,
        "streak_day_p90": streak_p90,
        **tails,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(BIN_SWEEPS_DIR.glob("*_daily.parquet"))
    if not files:
        raise RuntimeError(f"No daily parquet files found in {BIN_SWEEPS_DIR}")

    rows = []
    for f in files:
        daily = pd.read_parquet(f)
        diag = diagnose_daily_df(daily)
        row = {"file": str(f), **diag}
        rows.append(row)
        print(f"Diagnosed {f.name}")

    df = pd.DataFrame(rows).sort_values("total_pnl", ascending=False)

    out_csv = OUT_DIR / "_SUMMARY_diagnostics.csv"
    df.to_csv(out_csv, index=False)

    out_json = OUT_DIR / "_SUMMARY_diagnostics.json"
    out_json.write_text(json.dumps(rows, indent=2))

    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()