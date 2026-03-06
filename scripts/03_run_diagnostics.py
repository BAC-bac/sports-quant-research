from pathlib import Path
import json
import pandas as pd

from src.diagnostics.convexity import (
    drawdown_series, max_drawdown, drawdown_events,
    losing_streaks_daily, tails_daily, time_to_recovery_stats
)

IN_ROOT = Path("reports/bfsp/uk/place/bin_sweeps")
OUT_ROOT = Path("reports/bfsp/uk/place/diagnostics")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def diagnose_one(path: Path) -> dict:
    daily = pd.read_parquet(path)
    if daily.empty:
        return {"file": str(path), "empty": True}

    dd_df = drawdown_series(daily["equity"])
    mdd = max_drawdown(dd_df)

    dd_events = drawdown_events(daily)
    ttr = time_to_recovery_stats(dd_events)

    streaks_day = losing_streaks_daily(daily)
    streak_stats = {
        "count": int(streaks_day.count()),
        "max": int(streaks_day.max()) if not streaks_day.empty else 0,
        "p90": float(streaks_day.quantile(0.90)) if not streaks_day.empty else 0.0,
        "median": float(streaks_day.median()) if not streaks_day.empty else 0.0,
    }

    tails = tails_daily(daily, windows=(1, 5, 20))

    summary = {
        "file": str(path),
        "start": str(daily["event_date"].min()),
        "end": str(daily["event_date"].max()),
        "total_pnl": float(daily["pnl"].sum()),
        "n_days": int(len(daily)),
        "n_bets": int(daily["n_bets"].sum()),
        "max_drawdown": mdd,
        "time_to_recovery": ttr,
        "losing_streaks_day": streak_stats,
        "tails": tails,
    }

    # write detailed tables too
    name = path.stem.replace("_daily", "")
    dd_events.to_csv(OUT_ROOT / f"{name}_drawdown_events.csv", index=False)
    streaks_day.to_csv(OUT_ROOT / f"{name}_losing_streaks_day.csv", index=False)
    return summary


def main():
    files = sorted(IN_ROOT.glob("*_daily.parquet"))
    all_summaries = []

    for f in files:
        s = diagnose_one(f)
        all_summaries.append(s)
        print(f"Diagnosed {f.name}")

    out_json = OUT_ROOT / "_SUMMARY_diagnostics.json"
    out_csv = OUT_ROOT / "_SUMMARY_diagnostics.csv"

    out_json.write_text(json.dumps(all_summaries, indent=2))

    # Flatten for CSV convenience
    flat_rows = []
    for s in all_summaries:
        if s.get("empty"):
            flat_rows.append({"file": s["file"], "empty": True})
            continue
        flat_rows.append({
            "file": s["file"],
            "start": s["start"],
            "end": s["end"],
            "total_pnl": s["total_pnl"],
            "n_days": s["n_days"],
            "n_bets": s["n_bets"],
            "max_dd": s["max_drawdown"]["max_dd"],
            "max_dd_pct": s["max_drawdown"]["max_dd_pct"],
            "ttr_median_days": s["time_to_recovery"].get("median_days", 0),
            "ttr_p90_days": s["time_to_recovery"].get("p90_days", 0),
            "ttr_max_days": s["time_to_recovery"].get("max_days", 0),
            "streak_day_max": s["losing_streaks_day"]["max"],
            "streak_day_p90": s["losing_streaks_day"]["p90"],
            "worst_1d": s["tails"]["worst_1d"],
            "worst_5d": s["tails"]["worst_5d"],
            "worst_20d": s["tails"]["worst_20d"],
        })

    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()