from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# Inputs (change bins here)
# -----------------------------
BIN_FILES = [
    "reports/bfsp/uk/greyhound/win/bin_sweeps/bsp_10_00_20_00_daily.parquet",
    "reports/bfsp/uk/greyhound/win/bin_sweeps/bsp_2_50_4_00_daily.parquet",
]

OUT_DIR = Path("reports/bfsp/uk/greyhound/win/stability")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Metrics
# -----------------------------
def max_drawdown(equity: np.ndarray) -> float:
    eq = equity.astype(float)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(np.min(dd)) if len(dd) else 0.0


def time_to_recovery_days(equity: np.ndarray) -> np.ndarray:
    """
    Recovery times within the segment.
    If never recovers inside the segment, counts to segment end.
    """
    eq = equity.astype(float)
    n = len(eq)
    if n == 0:
        return np.array([], dtype=float)

    peak = np.maximum.accumulate(eq)
    rec = []
    i = 0
    while i < n:
        if eq[i] >= peak[i]:
            i += 1
            continue
        target = peak[i]
        start = i
        i += 1
        while i < n and eq[i] < target:
            i += 1
        rec.append(i - start if i < n else n - start)
    return np.array(rec, dtype=float)


def yearly_report(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Expects event_date, pnl, equity.
    Produces per-year pnl, max_dd, ttr stats.
    """
    d = daily.copy()
    d["event_date"] = pd.to_datetime(d["event_date"])
    d["year"] = d["event_date"].dt.year.astype(int)

    rows = []
    for y, g in d.groupby("year", sort=True):
        g = g.sort_values("event_date").reset_index(drop=True)

        pnl = g["pnl"].to_numpy(dtype=float)
        # IMPORTANT: rebuild equity from within-year pnl so DD is within-year
        eq = np.cumsum(pnl)

        ttr = time_to_recovery_days(eq)

        rows.append({
            "year": int(y),
            "days": int(len(g)),
            "pnl": float(pnl.sum()),
            "max_dd": float(max_drawdown(eq)),
            "ttr_median": float(np.median(ttr)) if len(ttr) else 0.0,
            "ttr_p90": float(np.quantile(ttr, 0.9)) if len(ttr) else 0.0,
            "ttr_max": float(np.max(ttr)) if len(ttr) else 0.0,
            "worst_1d": float(np.min(pnl)) if len(pnl) else 0.0,
        })

    out = pd.DataFrame(rows).sort_values("year")
    out["cum_pnl"] = out["pnl"].cumsum()
    out["pos_year"] = (out["pnl"] > 0).astype(int)
    return out


def main():
    all_outputs = []

    for f in BIN_FILES:
        path = Path(f)
        if not path.exists():
            raise FileNotFoundError(f"Missing daily bin file: {path}")

        daily = pd.read_parquet(path)

        # Guard: ensure required columns exist
        need = {"event_date", "pnl", "equity"}
        missing = need - set(daily.columns)
        if missing:
            raise ValueError(f"{path} missing columns {missing}. Found: {list(daily.columns)}")

        rep = yearly_report(daily)
        rep.insert(0, "bin_file", str(path))
        all_outputs.append(rep)

        # Save per-bin
        out_csv = OUT_DIR / f"{path.stem}_YEARLY.csv"
        rep.to_csv(out_csv, index=False)
        print(f"Saved yearly report: {out_csv}")

        # Print quick summary
        n_years = len(rep)
        pos = int(rep["pos_year"].sum())
        print(f"\n{path.stem}: years={n_years} positive_years={pos} ({pos/n_years:.1%})")
        print(rep[["year","pnl","max_dd","ttr_p90","ttr_max","worst_1d"]].to_string(index=False))

    # Save combined
    combo = pd.concat(all_outputs, ignore_index=True)
    combo_csv = OUT_DIR / "_COMBINED_YEARLY.csv"
    combo.to_csv(combo_csv, index=False)
    print(f"\nSaved combined: {combo_csv}")


if __name__ == "__main__":
    main()