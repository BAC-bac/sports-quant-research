from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


MASTER_ROOT = Path("data/master/bfsp/uk/greyhound/win")
OUT_DIR = Path("reports/bfsp/uk/greyhound/win/track_splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Focus pocket (our candidate)
BSP_LO = 10.0
BSP_HI = 20.0

# Regime split
PRE_END_YEAR = 2021  # pre = <= 2021, post = >= 2022


def max_drawdown(equity: np.ndarray) -> float:
    eq = equity.astype(float)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(np.min(dd)) if len(dd) else 0.0


def rolling_min_sum(x: np.ndarray, window: int) -> float:
    if len(x) < window:
        return float(np.sum(x)) if len(x) else 0.0
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    rs = c[window:] - c[:-window]
    return float(np.min(rs)) if len(rs) else 0.0


def per_track_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must contain: event_dt, menu_hint, bsp, pnl_1u
    menu_hint looks like "GB / Track  31st Mar" (horses) or similar for greyhounds.
    For greyhounds, it's usually "GB / <Track>" or a variant.
    We'll extract a track key robustly.
    """
    d = df.copy()

    # event_dt -> year, date
    d["event_dt"] = pd.to_datetime(d["event_dt"])
    d["year"] = d["event_dt"].dt.year.astype(int)
    d["event_date"] = d["event_dt"].dt.date

    # Track extraction:
    # menu_hint examples can vary; safest is:
    # - take text after "GB /" if present
    # - strip trailing date-like fragments
    mh = d["menu_hint"].astype(str)

    # Split on " / " and take last chunk
    track = mh.str.split("/", n=1).str[-1].str.strip()

    # Remove double spaces and trim
    track = track.str.replace(r"\s+", " ", regex=True).str.strip()

    # Some hints may include date/time tokens; remove common patterns
    # e.g. "Nottingham 31st Mar" or "Nottingham 2023-09-01"
    track = track.str.replace(r"\b\d{1,2}(st|nd|rd|th)\b", "", regex=True)
    track = track.str.replace(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b", "", regex=True, flags=0)
    track = track.str.replace(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", "", regex=True)
    track = track.str.replace(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", "", regex=True)

    track = track.str.replace(r"\s+", " ", regex=True).str.strip()

    d["track"] = track

    # Helper to aggregate daily pnl per track for DD/tails
    def agg_track(segment: pd.DataFrame) -> pd.Series:
        daily = (
            segment.groupby("event_date", as_index=False)
            .agg(pnl=("pnl_1u", "sum"))
            .sort_values("event_date")
        )
        pnl = daily["pnl"].to_numpy(dtype=float)
        eq = np.cumsum(pnl)
        return pd.Series(
            {
                "days": int(len(daily)),
                "bets": int(len(segment)),
                "pnl": float(pnl.sum()),
                "max_dd": float(max_drawdown(eq)),
                "worst_20d": float(rolling_min_sum(pnl, 20)),
            }
        )

    # Split pre/post
    pre = d[d["year"] <= PRE_END_YEAR]
    post = d[d["year"] >= (PRE_END_YEAR + 1)]

    pre_g = pre.groupby("track", sort=False).apply(agg_track).reset_index()
    pre_g = pre_g.rename(columns={
        "days": "pre_days", "bets": "pre_bets", "pnl": "pre_pnl",
        "max_dd": "pre_max_dd", "worst_20d": "pre_worst20"
    })

    post_g = post.groupby("track", sort=False).apply(agg_track).reset_index()
    post_g = post_g.rename(columns={
        "days": "post_days", "bets": "post_bets", "pnl": "post_pnl",
        "max_dd": "post_max_dd", "worst_20d": "post_worst20"
    })

    # Full
    full_g = d.groupby("track", sort=False).apply(agg_track).reset_index()
    full_g = full_g.rename(columns={
        "days": "all_days", "bets": "all_bets", "pnl": "all_pnl",
        "max_dd": "all_max_dd", "worst_20d": "all_worst20"
    })

    out = full_g.merge(pre_g, on="track", how="left").merge(post_g, on="track", how="left")

    # Stability / survival flags
    out["survived_post"] = (out["post_pnl"].fillna(0.0) > 0).astype(int)
    out["pre_positive"] = (out["pre_pnl"].fillna(0.0) > 0).astype(int)
    out["post_vs_pre"] = out["post_pnl"].fillna(0.0) - out["pre_pnl"].fillna(0.0)

    return out


def main():
    # Load only what we need (column prune)
    cols = ["event_dt", "menu_hint", "bsp", "won", "pnl_1u"]
    df = pd.read_parquet(MASTER_ROOT, columns=cols)

    # Filter valid bets and pocket
    df = df[df["bsp"].notna() & (df["bsp"] > 1.0)]
    df = df[(df["bsp"] >= BSP_LO) & (df["bsp"] < BSP_HI)].copy()

    print(f"Pocket BSP {BSP_LO}-{BSP_HI}: rows={len(df):,}")

    rep = per_track_metrics(df)

    # Basic quality filters so we don't get fooled by tiny samples
    rep["post_bets"] = rep["post_bets"].fillna(0).astype(int)
    rep["pre_bets"] = rep["pre_bets"].fillna(0).astype(int)
    rep["all_bets"] = rep["all_bets"].fillna(0).astype(int)

    # You can tune this threshold later; start conservative
    rep = rep[rep["all_bets"] >= 500].copy()

    # Save full report
    out_csv = OUT_DIR / f"track_regime_split_bsp_{int(BSP_LO)}_{int(BSP_HI)}.csv"
    rep.sort_values(["post_pnl", "all_pnl"], ascending=False).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Print top survivors
    top = rep.sort_values(["post_pnl", "all_pnl"], ascending=False).head(30)
    show_cols = [
        "track",
        "all_bets", "all_pnl", "all_max_dd", "all_worst20",
        "pre_bets", "pre_pnl",
        "post_bets", "post_pnl", "post_max_dd", "post_worst20",
        "survived_post"
    ]
    print("\nTop tracks by POST pnl (min 500 all_bets):")
    print(top[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()