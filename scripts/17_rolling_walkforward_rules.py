# scripts/17_rolling_walkforward_rules.py
"""
Step 17: Rolling walk-forward (monthly) with dynamic rule selection.

What it does (NO lookahead):
- Uses MASTER bet-level data (UK greyhound WIN), pocket BSP 10–20.
- For each TEST month:
    TRAIN = all prior months (expanding window)
    1) Rank tracks on TRAIN by total pnl (1u stake).
    2) Select TOP_N tracks (default 3).
    3) Find WORST BSP band on TRAIN among [10-12, 12-15, 15-20] by total pnl.
    TEST month strategy:
      - bet only selected tracks
      - EXCLUDE the worst band learned from TRAIN
- Writes:
    - monthly decisions table
    - test-month daily parquet files per month
    - full combined daily equity curve parquet + csv
    - diagnostics summary

Run:
    python scripts/17_rolling_walkforward_rules.py
"""

from __future__ import annotations

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd


# -------------------------
# CONFIG
# -------------------------
MASTER_ROOT = Path("data/master/bfsp/uk/greyhound/win")

OUT_BASE = Path("reports/bfsp/uk/greyhound/win/walkforward/rolling_step17")
OUT_DAILY_DIR = OUT_BASE / "daily_by_month"
OUT_TABLES_DIR = OUT_BASE / "tables"

# Pocket definition
BSP_LO = 10.0
BSP_HI = 20.0

# Bands used to pick "worst band" on TRAIN (must partition 10–20)
BANDS = [
    (10.0, 12.0, "10-12"),
    (12.0, 15.0, "12-15"),
    (15.0, 20.0, "15-20"),
]

TOP_N = 3
SEED = 42

# Minimum TRAIN activity for a track to be eligible (avoid tiny-sample noise)
MIN_TRAIN_BETS_PER_TRACK = 500

# Rolling schedule:
# We'll run strategy from START_MONTH to END_MONTH (inclusive), monthly.
# Use whatever range you want. Defaults align with your WF test window style.
START_MONTH = "2022-01"
END_MONTH = "2023-02"


# -------------------------
# TRACK CANONICALISATION (consistent with your Step 9/10/14 work)
# -------------------------
TRACK_ALIAS = {
    # abbreviations / short codes
    "cpark": "Central Park",
    "cp": "Central Park",
    "pbarr": "Perry Barr",
    "perry": "Perry Barr",          # critical (Step 13 had 'Perry')
    "p barr": "Perry Barr",
    "romfd": "Romford",
    "romf": "Romford",
    "monm": "Monmore",
    "henl": "Henlow",
    "harl": "Harlow",
    "sheff": "Sheffield",
    "sund": "Sunderland",
    "newc": "Newcastle",
    "nott": "Nottingham",
    "swin": "Swindon",
    "donc": "Doncaster",
    "towc": "Towcester",
    "kinsl": "Kinsley",
    "yarm": "Yarmouth",
    "pgran": "Pelaw Grange",
    "pboro": "Peterborough",
    "bvue": "Bvue",
    # common full names (normalize casing)
    "central park": "Central Park",
    "perry barr": "Perry Barr",
    "romford": "Romford",
    "monmore": "Monmore",
    "henlow": "Henlow",
    "harlow": "Harlow",
    "sheffield": "Sheffield",
    "sunderland": "Sunderland",
    "newcastle": "Newcastle",
    "nottingham": "Nottingham",
    "swindon": "Swindon",
    "doncaster": "Doncaster",
    "towcester": "Towcester",
    "kinsley": "Kinsley",
    "yarmouth": "Yarmouth",
    "pelaw grange": "Pelaw Grange",
    "peterborough": "Peterborough",
    "hove": "Hove",
    "crayford": "Crayford",
    "oxford": "Oxford",
    "poole": "Poole",
    "valley": "Valley",
    "mullingar": "Mullingar",
    "suffolk downs": "Suffolk Downs",
    "shelbourne park": "Shelbourne Park",
    "star pelaw": "Star Pelaw",
}

# Keep explicit (prevents AUS/NZ/etc.)
UK_CANON_TRACKS = {
    "Monmore", "Romford", "Hove", "Crayford", "Newcastle", "Central Park", "Harlow",
    "Sheffield", "Sunderland", "Swindon", "Perry Barr", "Nottingham", "Henlow",
    "Towcester", "Doncaster", "Kinsley", "Yarmouth", "Oxford", "Poole", "Valley",
    "Pelaw Grange", "Star Pelaw", "Peterborough", "Bvue",
    # include if you want UK+IRE in your “UK-only” bucket
    "Mullingar", "Shelbourne Park", "Suffolk Downs",
}

MENU_SPLIT_RE = re.compile(r"/\s*(.+)$")  # RHS of "X / RHS"
DATE_SUFFIX_RE = re.compile(
    r"\s+\d{1,2}(st|nd|rd|th)\s+[A-Za-z]{3,}\s*$",
    flags=re.IGNORECASE,
)

def canonical_track_from_menu_hint(menu_hint: str) -> str:
    """
    Examples:
      "Morning Cards / CPark 29th Jan" -> "Central Park"
      "Televised Cards / Romfd 29th Jan" -> "Romford"
    """
    if not isinstance(menu_hint, str) or not menu_hint.strip():
        return "UNKNOWN"

    s = menu_hint.strip()
    m = MENU_SPLIT_RE.search(s)
    rhs = m.group(1).strip() if m else s

    rhs = DATE_SUFFIX_RE.sub("", rhs).strip()

    key = re.sub(r"\s+", " ", rhs.strip().lower())

    # direct map
    if key in TRACK_ALIAS:
        return TRACK_ALIAS[key]

    # try first token (Donc, Nott, etc.)
    first = key.split(" ", 1)[0]
    if first in TRACK_ALIAS:
        return TRACK_ALIAS[first]

    # fallback: title-case raw
    return rhs.title()


# -------------------------
# PNL + DIAGNOSTICS
# -------------------------
def compute_pnl_1u(won: np.ndarray, bsp: np.ndarray) -> np.ndarray:
    # 1 unit stake
    return np.where(won == 1, (bsp - 1.0), -1.0)

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
    return float(pd.Series(pnl).rolling(window).sum().min())

def ttr_p90_days(daily: pd.DataFrame) -> float:
    """Time-to-recover p90 in days (simple drawdown episode scan)."""
    if daily.empty:
        return float("nan")
    eq = daily["equity"].to_numpy(dtype=float)
    dates = pd.to_datetime(daily["event_date"]).to_numpy()

    peak = np.maximum.accumulate(eq)
    underwater = eq < peak

    rec = []
    i = 0
    n = len(eq)
    while i < n:
        if not underwater[i]:
            i += 1
            continue
        start = i
        target = peak[i]
        j = i
        while j < n and eq[j] < target:
            j += 1
        if j < n:
            rec.append((pd.Timestamp(dates[j]) - pd.Timestamp(dates[start])).days)
        i = j + 1

    if not rec:
        return float("nan")
    return float(np.percentile(np.asarray(rec, dtype=float), 90))

def diagnose_daily(daily: pd.DataFrame) -> dict:
    pnl = daily["pnl"].to_numpy(dtype=float) if not daily.empty else np.array([])
    eq = daily["equity"].to_numpy(dtype=float) if not daily.empty else np.array([])
    return {
        "total_pnl": float(daily["pnl"].sum()) if not daily.empty else 0.0,
        "n_days": int(len(daily)),
        "n_bets": int(daily["n_bets"].sum()) if not daily.empty else 0,
        "max_dd": max_drawdown(eq),
        "ttr_p90_days": ttr_p90_days(daily),
        "worst_1d": float(pnl.min()) if pnl.size else 0.0,
        "worst_5d": worst_window(pnl, 5),
        "worst_20d": worst_window(pnl, 20),
    }


# -------------------------
# DATA LOADING + PREP
# -------------------------
def load_master_filtered() -> pd.DataFrame:
    """
    Loads MASTER and filters:
    - bsp in [10,20)
    - bsp > 1 (basic validity)
    - canonical track from menu_hint
    - UK canonical track set
    Also creates:
    - event_dt (datetime)
    - event_date (date)
    - month (YYYY-MM)
    """
    cols = ["event_dt", "event_date", "menu_hint", "bsp", "won"]
    df = pd.read_parquet(MASTER_ROOT, columns=cols)

    # event_dt
    df["event_dt"] = pd.to_datetime(df["event_dt"], errors="coerce")
    df = df[df["event_dt"].notna()].copy()

    # event_date safety
    if "event_date" not in df.columns or df["event_date"].isna().all():
        df["event_date"] = df["event_dt"].dt.date
    else:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date
        df.loc[pd.isna(df["event_date"]), "event_date"] = df.loc[pd.isna(df["event_date"]), "event_dt"].dt.date

    # validity + pocket
    df = df[df["bsp"].notna()].copy()
    df["bsp"] = df["bsp"].astype(float)
    df = df[(df["bsp"] > 1.0) & (df["bsp"] >= BSP_LO) & (df["bsp"] < BSP_HI)].copy()

    # track canonical
    df["track"] = df["menu_hint"].map(canonical_track_from_menu_hint)
    df = df[df["track"].isin(UK_CANON_TRACKS)].copy()

    # types
    df["won"] = df["won"].astype("int8")
    df["month"] = df["event_dt"].dt.to_period("M").astype(str)  # YYYY-MM

    return df


# -------------------------
# TRAIN RULE SELECTION
# -------------------------
def pick_top_tracks_and_worst_band(train: pd.DataFrame) -> tuple[list[str], str]:
    """
    Returns (top_tracks, worst_band_label) based on TRAIN only.
    """
    if train.empty:
        return ([], "NONE")

    # compute pnl
    pnl = compute_pnl_1u(train["won"].to_numpy(), train["bsp"].to_numpy())
    train = train.copy()
    train["pnl"] = pnl

    # track ranking (filter tiny tracks)
    by_track = (
        train.groupby("track", as_index=False)
        .agg(train_bets=("pnl", "size"), train_pnl=("pnl", "sum"))
    )
    by_track = by_track[by_track["train_bets"] >= MIN_TRAIN_BETS_PER_TRACK].copy()
    by_track = by_track.sort_values("train_pnl", ascending=False)

    top_tracks = by_track["track"].head(TOP_N).tolist()

    # worst band by TRAIN pnl (within selected tracks only — more realistic)
    tsel = train[train["track"].isin(top_tracks)].copy()
    if tsel.empty:
        return (top_tracks, "NONE")

    band_pnls = []
    for lo, hi, label in BANDS:
        d = tsel[(tsel["bsp"] >= lo) & (tsel["bsp"] < hi)]
        band_pnls.append((label, float(d["pnl"].sum()), int(len(d))))

    # worst = minimum pnl (most negative)
    band_pnls.sort(key=lambda x: x[1])
    worst_band = band_pnls[0][0]
    return (top_tracks, worst_band)


def apply_rule_to_month(test_month_df: pd.DataFrame, tracks: list[str], worst_band: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply strategy to TEST month:
    - filter to tracks
    - exclude worst_band (if not NONE)
    Returns (bet_level, daily)
    """
    df = test_month_df.copy()
    if tracks:
        df = df[df["track"].isin(tracks)].copy()

    if worst_band and worst_band != "NONE":
        # exclude band
        for lo, hi, label in BANDS:
            if label == worst_band:
                df = df[~((df["bsp"] >= lo) & (df["bsp"] < hi))].copy()
                break

    if df.empty:
        daily = pd.DataFrame(columns=["event_date", "pnl", "n_bets", "equity"])
        return df, daily

    df["pnl"] = compute_pnl_1u(df["won"].to_numpy(), df["bsp"].to_numpy())

    daily = (
        df.groupby("event_date", as_index=False)
        .agg(pnl=("pnl", "sum"), n_bets=("pnl", "size"))
        .sort_values("event_date")
    )
    daily["equity"] = daily["pnl"].cumsum()

    return df, daily


# -------------------------
# MAIN ROLLING WF
# -------------------------
def main():
    OUT_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading master (UK greyhound win, BSP 10–20)...")
    df = load_master_filtered()
    print(f"Pocket rows: {len(df):,}")
    if df.empty:
        raise RuntimeError("No data after filtering. Check MASTER path / columns / track parsing.")

    months = pd.period_range(START_MONTH, END_MONTH, freq="M").astype(str).tolist()
    all_months_available = sorted(df["month"].unique().tolist())
    print(f"Available months in data: {all_months_available[0]} -> {all_months_available[-1]}")
    print(f"Rolling months requested  : {months[0]} -> {months[-1]}")

    decisions = []
    combined_daily_parts = []

    for m in months:
        # TRAIN = all months < m
        train = df[df["month"] < m].copy()
        test = df[df["month"] == m].copy()

        if test.empty:
            print(f"[{m}] No test data. Skipping.")
            continue
        if train.empty:
            print(f"[{m}] No train history available (first month). Skipping.")
            continue

        top_tracks, worst_band = pick_top_tracks_and_worst_band(train)

        # apply rule to TEST month
        bet_level, daily = apply_rule_to_month(test, top_tracks, worst_band)
        diag = diagnose_daily(daily)

        decisions.append({
            "month": m,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "top_n": TOP_N,
            "min_train_bets_per_track": MIN_TRAIN_BETS_PER_TRACK,
            "top_tracks": ",".join(top_tracks),
            "worst_band_train": worst_band,
            "test_bets_after_rule": int(len(bet_level)),
            "test_total_pnl": diag["total_pnl"],
            "test_max_dd": diag["max_dd"],
            "test_ttr_p90_days": diag["ttr_p90_days"],
            "test_worst_20d": diag["worst_20d"],
        })

        # save month daily
        daily_path = OUT_DAILY_DIR / f"rolling_WF_TOP{TOP_N}_EXCL_{worst_band}_{m}_daily.parquet"
        daily.to_parquet(daily_path, index=False)

        # keep for combined equity
        if not daily.empty:
            daily2 = daily.copy()
            daily2["month"] = m
            daily2["rule_top_tracks"] = ",".join(top_tracks)
            daily2["rule_worst_band"] = worst_band
            combined_daily_parts.append(daily2)

        print(
            f"[{m}] tracks={top_tracks} | worst_band={worst_band} | "
            f"bets={len(bet_level):,} | pnl={diag['total_pnl']:.2f} | dd={diag['max_dd']:.2f}"
        )

    # Save decisions table
    decisions_df = pd.DataFrame(decisions)
    decisions_csv = OUT_TABLES_DIR / "rolling_decisions_step17.csv"
    decisions_df.to_csv(decisions_csv, index=False)

    # Combine daily + overall diagnostics
    if combined_daily_parts:
        comb = pd.concat(combined_daily_parts, ignore_index=True).sort_values(["event_date", "month"])
        # build continuous equity across months in chronological order
        comb = comb.sort_values("event_date").copy()
        comb["equity_global"] = comb["pnl"].cumsum()

        comb_parquet = OUT_TABLES_DIR / "rolling_daily_step17.parquet"
        comb_csv = OUT_TABLES_DIR / "rolling_daily_step17.csv"
        comb.to_parquet(comb_parquet, index=False)
        comb.to_csv(comb_csv, index=False)

        # diagnostics on combined
        comb_daily = comb.groupby("event_date", as_index=False).agg(pnl=("pnl", "sum"), n_bets=("n_bets", "sum"))
        comb_daily = comb_daily.sort_values("event_date")
        comb_daily["equity"] = comb_daily["pnl"].cumsum()
        diag = diagnose_daily(comb_daily)

        summary = {
            "start_month": START_MONTH,
            "end_month": END_MONTH,
            "top_n": TOP_N,
            "min_train_bets_per_track": MIN_TRAIN_BETS_PER_TRACK,
            "bands": [b[2] for b in BANDS],
            "combined": diag,
            "n_test_months": int(decisions_df.shape[0]),
        }
        summary_path = OUT_TABLES_DIR / "_SUMMARY_step17_rolling.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        print("\n=== STEP 17 ROLLING SUMMARY (combined across test months) ===")
        print(
            f"months={summary['n_test_months']} | "
            f"pnl={diag['total_pnl']:.2f} | dd={diag['max_dd']:.2f} | "
            f"ttr_p90={diag['ttr_p90_days']:.1f} | worst_20d={diag['worst_20d']:.2f}"
        )
        print(f"\nSaved decisions: {decisions_csv}")
        print(f"Saved combined daily: {comb_parquet}")
        print(f"Saved summary: {summary_path}")
    else:
        print("\nNo combined daily output (all months empty after rule).")
        print(f"Saved decisions: {decisions_csv}")


if __name__ == "__main__":
    main()