# scripts/13_walkforward_track_baskets.py

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------

MASTER_ROOT = Path("data/master/bfsp/uk/greyhound/win")

OUT_ROOT = Path("reports/bfsp/uk/greyhound/win/walkforward")
OUT_DAILY = OUT_ROOT / "daily"
OUT_TABLES = OUT_ROOT / "tables"

BSP_LO, BSP_HI = 10.0, 20.0

# Walk-forward split
TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END   = pd.Timestamp("2021-12-31 23:59:59")
TEST_START  = pd.Timestamp("2022-01-01")

STAKE = 1.0

# Minimum bets in TRAIN for a track to be eligible for ranking
MIN_TRAIN_BETS = 500

# Baskets
TOP_KS = [3, 5, 10, 15]

# ----------------------------
# TRACK PARSING / CANON
# ----------------------------

# common short labels -> canonical
CANON_MAP = {
    # core UK tracks you saw
    "Hove": "Hove",
    "Harl": "Harlow",
    "Harlow": "Harlow",
    "Henl": "Henlow",
    "Henlow": "Henlow",
    "Monm": "Monmore",
    "Monmore": "Monmore",
    "Romfd": "Romford",
    "Romford": "Romford",
    "Newc": "Newcastle",
    "Newcastle": "Newcastle",
    "Sheff": "Sheffield",
    "Sheffield": "Sheffield",
    "Sund": "Sunderland",
    "Sunderland": "Sunderland",
    "Swin": "Swindon",
    "Swindon": "Swindon",
    "PBarr": "Perry Barr",
    "Perry Barr": "Perry Barr",
    "Nott": "Nottingham",
    "Nottingham": "Nottingham",
    "Crayfd": "Crayford",
    "Crayford": "Crayford",
    "CPark": "Central Park",
    "Central Park": "Central Park",
    "Donc": "Doncaster",
    "Doncaster": "Doncaster",
    "Kinsley": "Kinsley",
    "Kinsl": "Kinsley",
    "Towc": "Towcester",
    "Towcester": "Towcester",
    "Yarm": "Yarmouth",
    "Yarmouth": "Yarmouth",
    "Oxford": "Oxford",
    "Pelaw Grange": "Pelaw Grange",
    "Pgran": "Pelaw Grange",
    "Mullingar": "Mullingar",
    "Suffolk Downs": "Suffolk Downs",
    "Bvue": "Belle Vue",
    "Belle Vue": "Belle Vue",
    "Poole": "Poole",
    "Valley": "Valley",
    "Pboro": "Peterborough",
    "Peterborough": "Peterborough",
    "Star Pelaw": "Star Pelaw",
}

# labels that clearly look non-UK/IRE in your earlier outputs; we exclude these
# (this is intentionally conservative: better to exclude than pollute)
NON_UK_KEYWORDS = [
    "(AUS)", "(NZL)", "AUS", "NZL",
    "Wentworth", "Sandown", "Gosford", "Temora", "Geelong", "Dubbo",
    "Richmond", "Goulburn", "Casino", "Angle Park", "The Gardens",
    "Townsville", "Cranbourne", "Hobart", "Gawler", "Q1", "Ascot Park",
    "Manawatu", "Sale", "Taree", "Muswellbrook",
]

# menu_hint examples you showed:
# "Morning Cards / CPark 29th Jan"
# "Evening Cards / Donc 29th Jan"
# "Televised Cards / Romfd 29th Jan"
_MENU_TRACK_RE = re.compile(r"/\s*([A-Za-z]+)\b")

def extract_track_raw(menu_hint: str) -> str | None:
    if not isinstance(menu_hint, str) or not menu_hint:
        return None

    # quick reject obvious non-UK markers
    mh_upper = menu_hint.upper()
    for kw in NON_UK_KEYWORDS:
        if kw.upper() in mh_upper:
            return None

    m = _MENU_TRACK_RE.search(menu_hint)
    if not m:
        return None
    return m.group(1)

def canonical_track(track_raw: str | None) -> str | None:
    if track_raw is None:
        return None
    return CANON_MAP.get(track_raw, track_raw)

# ----------------------------
# CORE PNL + DAILY SERIES
# ----------------------------

def pnl_back_win(won: pd.Series, bsp: pd.Series, stake: float = 1.0) -> pd.Series:
    won_i = won.astype("int8").to_numpy()
    bsp_f = bsp.astype("float64").to_numpy()
    pnl = np.where(won_i == 1, stake * (bsp_f - 1.0), -stake)
    return pd.Series(pnl, index=won.index, name="pnl")

def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("event_date", as_index=False)
          .agg(
              pnl=("pnl", "sum"),
              n_bets=("pnl", "size"),
              avg_bsp=("bsp", "mean"),
          )
          .sort_values("event_date")
          .reset_index(drop=True)
    )
    daily["equity"] = daily["pnl"].cumsum()
    return daily

# ----------------------------
# DIAGNOSTICS (same spirit as your 03/11 scripts)
# ----------------------------

def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min())

def worst_window_sum(pnl: pd.Series, window: int) -> float:
    if pnl.empty:
        return 0.0
    if len(pnl) < window:
        return float(pnl.sum())
    return float(pnl.rolling(window).sum().min())

def time_to_recover_days(equity: pd.Series) -> tuple[float, float, float]:
    """
    Returns (median, p90, max) time-to-recover in days, computed on drawdown episodes.
    """
    if equity.empty:
        return (np.nan, np.nan, np.nan)

    peak = equity.cummax()
    in_dd = equity < peak

    # identify contiguous drawdown episodes
    ttrs = []
    i = 0
    n = len(equity)
    while i < n:
        if not in_dd.iloc[i]:
            i += 1
            continue
        start = i
        # peak level at start of DD episode
        peak_level = peak.iloc[i]
        i += 1
        while i < n and equity.iloc[i] < peak_level:
            i += 1
        end = i  # first index at/above peak_level or n
        ttrs.append(end - start)

    if not ttrs:
        return (0.0, 0.0, 0.0)

    arr = np.array(ttrs, dtype=float)
    return (float(np.median(arr)), float(np.quantile(arr, 0.90)), float(arr.max()))

def streak_stats_daily(pnl: pd.Series) -> tuple[int, float]:
    """
    Losing streak length in DAYS (consecutive pnl<0).
    Returns (max_streak, p90_streak).
    """
    if pnl.empty:
        return (0, 0.0)

    losses = (pnl < 0).to_numpy()
    streaks = []
    cur = 0
    for x in losses:
        if x:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)

    if not streaks:
        return (0, 0.0)

    arr = np.array(streaks, dtype=float)
    return (int(arr.max()), float(np.quantile(arr, 0.90)))

def diagnose_daily(daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {
            "total_pnl": 0.0,
            "n_bets": 0,
            "n_days": 0,
            "max_dd": 0.0,
            "max_dd_pct": 0.0,
            "ttr_median_days": np.nan,
            "ttr_p90_days": np.nan,
            "ttr_max_days": np.nan,
            "streak_day_max": 0,
            "streak_day_p90": np.nan,
            "worst_1d": 0.0,
            "worst_5d": 0.0,
            "worst_20d": 0.0,
        }

    total_pnl = float(daily["pnl"].sum())
    n_bets = int(daily["n_bets"].sum())
    n_days = int(len(daily))

    equity = daily["equity"]
    dd = max_drawdown(equity)

    # pct drawdown relative to ending equity magnitude (simple, not a bankroll model)
    denom = max(1e-9, float(np.abs(equity.iloc[-1])) + 1e-9)
    dd_pct = 100.0 * dd / denom

    ttr_med, ttr_p90, ttr_max = time_to_recover_days(equity)
    streak_max, streak_p90 = streak_stats_daily(daily["pnl"])

    return {
        "total_pnl": total_pnl,
        "n_bets": n_bets,
        "n_days": n_days,
        "max_dd": float(dd),
        "max_dd_pct": float(dd_pct),
        "ttr_median_days": float(ttr_med),
        "ttr_p90_days": float(ttr_p90),
        "ttr_max_days": float(ttr_max),
        "streak_day_max": int(streak_max),
        "streak_day_p90": float(streak_p90),
        "worst_1d": float(daily["pnl"].min()),
        "worst_5d": worst_window_sum(daily["pnl"], 5),
        "worst_20d": worst_window_sum(daily["pnl"], 20),
    }

def yearly_breakdown(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["year", "pnl", "max_dd", "ttr_p90", "ttr_max", "worst_1d"])

    d = daily.copy()
    d["year"] = pd.to_datetime(d["event_date"]).dt.year

    rows = []
    for y, g in d.groupby("year"):
        eq = g["pnl"].cumsum()
        ttr_med, ttr_p90, ttr_max = time_to_recover_days(eq)
        rows.append({
            "year": int(y),
            "pnl": float(g["pnl"].sum()),
            "max_dd": float(max_drawdown(eq)),
            "ttr_p90": float(ttr_p90),
            "ttr_max": float(ttr_max),
            "worst_1d": float(g["pnl"].min()),
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

# ----------------------------
# MAIN WALKFORWARD
# ----------------------------

def load_master() -> pd.DataFrame:
    cols = ["event_dt", "event_date", "menu_hint", "bsp", "won"]
    df = pd.read_parquet(MASTER_ROOT, columns=cols)

    # clean / filter bsp
    df = df[df["bsp"].notna()]
    df = df[(df["bsp"] >= BSP_LO) & (df["bsp"] < BSP_HI)].copy()

    # ensure datetime
    df["event_dt"] = pd.to_datetime(df["event_dt"], errors="coerce")
    df = df[df["event_dt"].notna()].copy()

    # track parsing
    df["track_raw"] = df["menu_hint"].map(extract_track_raw)
    df["track"] = df["track_raw"].map(canonical_track)

    # UK-only approximation: drop missing track or anything we filtered as non-UK
    df = df[df["track"].notna()].copy()

    # pnl
    df["pnl"] = pnl_back_win(df["won"], df["bsp"], STAKE)

    return df

def rank_tracks_train(df: pd.DataFrame) -> pd.DataFrame:
    train = df[(df["event_dt"] >= TRAIN_START) & (df["event_dt"] <= TRAIN_END)].copy()

    # per-track pnl
    per = (
        train.groupby("track", as_index=False)
             .agg(
                 train_bets=("pnl", "size"),
                 train_pnl=("pnl", "sum"),
             )
    )

    # compute per-track max_dd using daily series per track (more honest than bet-level)
    # do it efficiently by iterating tracks (74-ish tracks in UK pocket)
    dd_rows = []
    for t, g in train.groupby("track"):
        daily = (
            g.groupby("event_date", as_index=False)
             .agg(pnl=("pnl", "sum"))
             .sort_values("event_date")
        )
        eq = daily["pnl"].cumsum()
        dd_rows.append({"track": t, "train_max_dd": max_drawdown(eq)})

    dd_df = pd.DataFrame(dd_rows)
    per = per.merge(dd_df, on="track", how="left")

    # eligibility + ranking metric
    per = per[per["train_bets"] >= MIN_TRAIN_BETS].copy()
    per["train_profit_dd_ratio"] = per["train_pnl"] / (per["train_max_dd"].abs() + 1e-9)

    per = per.sort_values(["train_pnl", "train_profit_dd_ratio"], ascending=False).reset_index(drop=True)
    return per

def evaluate_basket_test(df: pd.DataFrame, tracks: list[str], basket_name: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    test = df[df["event_dt"] >= TEST_START].copy()
    test = test[test["track"].isin(tracks)].copy()

    daily = to_daily(test)
    diag = diagnose_daily(daily)
    yr = yearly_breakdown(daily)

    diag_row = {"basket": basket_name, "tracks": "|".join(tracks), **diag}
    return daily, diag_row, yr

def main():
    OUT_DAILY.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    print("Loading master (UK greyhound win, pocket 10–20)...")
    df = load_master()
    print(f"Pocket rows: {len(df):,}")
    print(f"Date range : {df['event_dt'].min()} -> {df['event_dt'].max()}")
    print(f"Unique tracks in pocket: {df['track'].nunique()}")

    print("\nRanking tracks using TRAIN only...")
    ranks = rank_tracks_train(df)
    out_ranks = OUT_TABLES / "train_track_ranking.csv"
    ranks.to_csv(out_ranks, index=False)
    print(f"Saved TRAIN ranking: {out_ranks}")
    print(ranks.head(15).to_string(index=False))

    summary_rows = []
    all_yearlies = []

    # ALL_UK basket (for reference, still walk-forward evaluated on TEST)
    all_tracks = sorted(df["track"].dropna().unique().tolist())
    daily, diag_row, yr = evaluate_basket_test(df, all_tracks, "WF_ALL_UK_10_20")
    p_daily = OUT_DAILY / "WF_ALL_UK_10_20_daily.parquet"
    daily.to_parquet(p_daily, index=False)
    diag_row["daily_file"] = str(p_daily)
    summary_rows.append(diag_row)
    yr["basket"] = "WF_ALL_UK_10_20"
    all_yearlies.append(yr)

    # TOP-K from TRAIN ranking
    for k in TOP_KS:
        top_tracks = ranks["track"].head(k).tolist()
        name = f"WF_TOP{k}_TRAIN_PNL"

        daily, diag_row, yr = evaluate_basket_test(df, top_tracks, name)
        p_daily = OUT_DAILY / f"{name}_daily.parquet"
        daily.to_parquet(p_daily, index=False)
        diag_row["daily_file"] = str(p_daily)
        summary_rows.append(diag_row)

        yr["basket"] = name
        all_yearlies.append(yr)

        print(f"\n{name}")
        print(f"Tracks: {top_tracks}")
        print(f"TEST total pnl: {diag_row['total_pnl']:.2f} | max_dd: {diag_row['max_dd']:.2f} | ttr_p90: {diag_row['ttr_p90_days']:.1f} days | worst_20d: {diag_row['worst_20d']:.2f}")

    summary = pd.DataFrame(summary_rows)
    out_sum = OUT_TABLES / "walkforward_summary.csv"
    summary.to_csv(out_sum, index=False)
    print(f"\nSaved walk-forward summary: {out_sum}")

    ydf = pd.concat(all_yearlies, ignore_index=True)
    out_y = OUT_TABLES / "walkforward_yearly.csv"
    ydf.to_csv(out_y, index=False)
    print(f"Saved walk-forward yearly: {out_y}")

    # friendly sort view
    view_cols = [
        "basket","total_pnl","n_bets","n_days","max_dd","ttr_p90_days","ttr_max_days",
        "worst_1d","worst_5d","worst_20d","daily_file"
    ]
    print("\n=== WALK-FORWARD TEST RESULTS (TEST period only) ===")
    print(summary[view_cols].sort_values("total_pnl", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()