# scripts/12_yearly_stability_track_baskets.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# Where Step 10 saved baskets:
BASKET_ROOT = Path("reports/bfsp/uk/greyhound/win/track_baskets")

# Output folder for this script:
OUT_ROOT = BASKET_ROOT / "stability"


def _max_drawdown(equity: np.ndarray) -> float:
    """Max drawdown in £ (negative number)."""
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def _time_to_recovery_days(equity: np.ndarray) -> np.ndarray:
    """
    For each time index i where equity is below the running peak, compute how many
    steps forward until it recovers back to >= that peak. If never recovers, use NaN.
    Returns array of recovery times (in number of rows / days).
    """
    n = len(equity)
    if n == 0:
        return np.array([], dtype=float)

    peak = np.maximum.accumulate(equity)
    ttr = np.full(n, np.nan, dtype=float)

    # For each i where below peak, search forward for recovery
    # This is O(n^2) worst-case, but per-year is small (~365 rows) so it's fine.
    for i in range(n):
        if equity[i] >= peak[i]:
            continue
        target = peak[i]
        j = i + 1
        while j < n and equity[j] < target:
            j += 1
        if j < n:
            ttr[i] = float(j - i)
        else:
            ttr[i] = np.nan
    return ttr


def _worst_window_sum(pnl: np.ndarray, window: int) -> float:
    """Worst (most negative) rolling sum over a window size."""
    if len(pnl) == 0:
        return 0.0
    if len(pnl) < window:
        return float(np.sum(pnl))
    s = pd.Series(pnl).rolling(window=window).sum()
    return float(s.min())


def diagnose_year(daily: pd.DataFrame) -> dict:
    """
    daily expected columns:
      event_date, pnl, equity
    """
    pnl = daily["pnl"].to_numpy(dtype=float)
    equity = daily["equity"].to_numpy(dtype=float)

    max_dd = _max_drawdown(equity)
    ttr = _time_to_recovery_days(equity)
    ttr_clean = ttr[~np.isnan(ttr)]

    out = {
        "pnl": float(pnl.sum()),
        "max_dd": float(max_dd),
        "worst_1d": float(np.min(pnl)) if len(pnl) else 0.0,
        "worst_5d": _worst_window_sum(pnl, 5),
        "worst_20d": _worst_window_sum(pnl, 20),
        "ttr_median": float(np.median(ttr_clean)) if len(ttr_clean) else np.nan,
        "ttr_p90": float(np.quantile(ttr_clean, 0.90)) if len(ttr_clean) else np.nan,
        "ttr_max": float(np.max(ttr_clean)) if len(ttr_clean) else np.nan,
        "n_days": int(len(daily)),
    }
    return out


def load_basket_files() -> list[Path]:
    files = sorted(BASKET_ROOT.glob("basket_*_daily.parquet"))
    return files


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    basket_files = load_basket_files()
    if not basket_files:
        raise RuntimeError(f"No basket daily parquet files found in: {BASKET_ROOT}")

    combined_rows = []

    for f in basket_files:
        daily = pd.read_parquet(f)

        # basic sanitation
        if "event_date" not in daily.columns or "pnl" not in daily.columns:
            print(f"[SKIP] {f} missing required columns")
            continue

        daily = daily.copy()
        daily["event_date"] = pd.to_datetime(daily["event_date"])
        daily = daily.sort_values("event_date")

        # ensure equity exists
        if "equity" not in daily.columns:
            daily["equity"] = daily["pnl"].cumsum()

        daily["year"] = daily["event_date"].dt.year.astype(int)

        rows = []
        for y, g in daily.groupby("year", sort=True):
            g = g.copy()
            # IMPORTANT: reset equity within the year, so yearly DD/TTR is "within-year"
            g["equity"] = g["pnl"].cumsum()

            diag = diagnose_year(g)
            rows.append({
                "basket_file": str(f),
                "basket": f.stem.replace("_daily", ""),
                "year": int(y),
                **diag,
            })

        yearly = pd.DataFrame(rows).sort_values("year")
        out_csv = OUT_ROOT / f"{f.stem}_YEARLY.csv"
        yearly.to_csv(out_csv, index=False)

        # scoreboard
        pos_years = int((yearly["pnl"] > 0).sum())
        total_years = int(len(yearly))
        print(f"\n{f.stem}: years={total_years} positive_years={pos_years} ({(pos_years/total_years*100):.1f}%)")
        print(yearly[["year", "pnl", "max_dd", "ttr_p90", "ttr_max", "worst_1d"]].to_string(index=False))

        combined_rows.append(yearly)

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined_path = OUT_ROOT / "_COMBINED_YEARLY.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nSaved combined: {combined_path}")

        # quick “best by stability” view (optional)
        # score: pnl / abs(max_dd) aggregated across years (rough proxy)
        tmp = combined.copy()
        tmp["profit_dd_ratio_year"] = tmp["pnl"] / (np.abs(tmp["max_dd"]) + 1e-9)
        agg = (tmp.groupby("basket", as_index=False)
                  .agg(
                      years=("year", "nunique"),
                      total_pnl=("pnl", "sum"),
                      avg_year_pnl=("pnl", "mean"),
                      avg_year_dd=("max_dd", "mean"),
                      avg_profit_dd_ratio=("profit_dd_ratio_year", "mean"),
                      positive_years=("pnl", lambda s: int((s > 0).sum())),
                  )
                  .sort_values(["avg_profit_dd_ratio", "total_pnl"], ascending=False))

        agg_path = OUT_ROOT / "_SUMMARY_YEARLY_STABILITY.csv"
        agg.to_csv(agg_path, index=False)
        print(f"Saved summary: {agg_path}")
        print("\n=== YEARLY STABILITY SUMMARY (top 10 by avg_profit_dd_ratio) ===")
        print(agg.head(10).to_string(index=False))


if __name__ == "__main__":
    main()