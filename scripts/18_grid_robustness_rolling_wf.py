# scripts/18_grid_robustness_rolling_wf.py
"""
Step 18: Robustness grid on the Step 17 rolling walk-forward rule.

What this does
--------------
For each TEST month in a chosen range (default 2022-01 -> 2023-02):

1) Build TRAIN set according to a TRAIN window mode:
   - expanding: all months < test_month
   - rolling_12m: last 12 months ending at month-1
   - rolling_24m: last 24 months ending at month-1

2) Rank tracks by TRAIN pnl (1 unit stake, win pnl = bsp-1, loss pnl = -1),
   with a minimum bet threshold per track.

3) Choose TOP_N tracks (e.g. 2/3/5).

4) Compute TRAIN pnl by BSP sub-band:
   - 10-12, 12-15, 15-20
   Then apply a band rule:
   - none
   - exclude_worst_always
   - exclude_worst_if_negative (only exclude if worst band TRAIN pnl < 0)

5) Apply the chosen tracks + band rule to the TEST month and record:
   - pnl, max_dd (within-month daily equity), bets

Finally, we aggregate across all test months and produce a leaderboard:
- total pnl, max_dd, ttr_p90, worst_20d, total bets, months covered

Outputs
-------
reports/bfsp/uk/greyhound/win/walkforward/step18_grid/
  tables/grid_leaderboard_step18.csv
  tables/grid_monthly_results_step18.csv
  tables/best_variant_daily_step18.parquet
  tables/best_variant_meta_step18.json
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

# -------------------------
# CONFIG
# -------------------------
MASTER_ROOT = Path("data/master/bfsp/uk/greyhound/win")

OUT_ROOT = Path("reports/bfsp/uk/greyhound/win/walkforward/step18_grid")
OUT_TABLES = OUT_ROOT / "tables"

BSP_LO = 10.0
BSP_HI = 20.0

# Default test range (inclusive months)
TEST_MONTH_START = "2022-01"
TEST_MONTH_END = "2026-02"

SEED = 42

# Grid choices
TOP_N_LIST = [2, 3, 5]
TRAIN_MODE_LIST = ["expanding", "rolling_12m", "rolling_24m"]
BAND_RULE_LIST = ["none", "exclude_worst_always", "exclude_worst_if_negative"]
MIN_TRAIN_BETS_LIST = [200, 500, 1000]

# Bands used in Step 15/16/17
BANDS = [
    ("10-12", 10.0, 12.0),
    ("12-15", 12.0, 15.0),
    ("15-20", 15.0, 20.0),
]

# -------------------------
# TRACK CANONICALISATION (consistent with Step 14/17 family)
# -------------------------
TRACK_ALIAS = {
    "cpark": "Central Park",
    "cp": "Central Park",
    "central park": "Central Park",

    "pbarr": "Perry Barr",
    "perry": "Perry Barr",  # critical
    "p barr": "Perry Barr",
    "perry barr": "Perry Barr",

    "romfd": "Romford",
    "romf": "Romford",
    "romford": "Romford",

    "monm": "Monmore",
    "monmore": "Monmore",

    "henl": "Henlow",
    "henlow": "Henlow",

    "harl": "Harlow",
    "harlow": "Harlow",

    "sheff": "Sheffield",
    "sheffield": "Sheffield",

    "sund": "Sunderland",
    "sunderland": "Sunderland",

    "newc": "Newcastle",
    "newcastle": "Newcastle",

    "nott": "Nottingham",
    "nottingham": "Nottingham",

    "swin": "Swindon",
    "swindon": "Swindon",

    "donc": "Doncaster",
    "doncaster": "Doncaster",

    "towc": "Towcester",
    "towcester": "Towcester",

    "kinsl": "Kinsley",
    "kinsley": "Kinsley",

    "yarm": "Yarmouth",
    "yarmouth": "Yarmouth",

    "pgran": "Pelaw Grange",
    "pelaw grange": "Pelaw Grange",

    "pboro": "Peterborough",
    "peterborough": "Peterborough",

    "bvue": "Bvue",
    "bvue.": "Bvue",

    "hove": "Hove",
    "crayford": "Crayford",
    "oxford": "Oxford",
    "poole": "Poole",
    "valley": "Valley",
    "mullingar": "Mullingar",
    "suffolk downs": "Suffolk Downs",
    "suffolk": "Suffolk Downs",
    "shelbourne park": "Shelbourne Park",
    "star pelaw": "Star Pelaw",
}

UK_CANON_TRACKS = {
    "Monmore", "Romford", "Hove", "Crayford", "Newcastle", "Central Park", "Harlow",
    "Sheffield", "Sunderland", "Swindon", "Perry Barr", "Nottingham", "Henlow",
    "Towcester", "Doncaster", "Kinsley", "Yarmouth", "Oxford", "Poole", "Valley",
    "Pelaw Grange", "Star Pelaw", "Peterborough", "Bvue",
    # UK+IRE as per your earlier canon outputs
    "Mullingar", "Shelbourne Park", "Suffolk Downs",
}

MENU_SPLIT_RE = re.compile(r"/\s*(.+)$")  # take RHS of "X / RHS"
DATE_SUFFIX_RE = re.compile(r"\s+\d{1,2}(st|nd|rd|th)\s+[A-Za-z]{3,}\s*$", flags=re.IGNORECASE)

def canonical_track_from_menu_hint(menu_hint: str) -> str:
    if not isinstance(menu_hint, str) or not menu_hint.strip():
        return "UNKNOWN"
    s = menu_hint.strip()
    m = MENU_SPLIT_RE.search(s)
    rhs = m.group(1).strip() if m else s
    rhs = DATE_SUFFIX_RE.sub("", rhs).strip()
    key = re.sub(r"\s+", " ", rhs.lower())
    # full rhs match first, then first token
    track = TRACK_ALIAS.get(key)
    if track is None:
        first = key.split(" ", 1)[0]
        track = TRACK_ALIAS.get(first)
    if track is None:
        track = rhs.title()
    return track

# -------------------------
# METRICS
# -------------------------
def compute_pnl_1u(df: pd.DataFrame) -> np.ndarray:
    won = df["won"].astype("int8").to_numpy()
    bsp = df["bsp"].astype("float64").to_numpy()
    return np.where(won == 1, (bsp - 1.0), -1.0)

def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = (
        df.groupby("event_date", as_index=False)
        .agg(pnl=("pnl", "sum"), n_bets=("pnl", "size"))
        .sort_values("event_date")
    )
    d["equity"] = d["pnl"].cumsum()
    return d

def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())

def worst_window(daily_pnl: np.ndarray, window: int) -> float:
    if len(daily_pnl) == 0:
        return 0.0
    if len(daily_pnl) < window:
        return float(np.sum(daily_pnl))
    s = pd.Series(daily_pnl).rolling(window).sum()
    return float(s.min())

def time_to_recover_days(daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {"p90": np.nan, "max": np.nan}
    eq = daily["equity"].to_numpy(dtype=float)
    dates = pd.to_datetime(daily["event_date"]).to_numpy()
    peak = np.maximum.accumulate(eq)
    underwater = eq < peak

    rec = []
    i, n = 0, len(eq)
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
        return {"p90": np.nan, "max": np.nan}

    arr = np.array(rec, dtype=float)
    return {"p90": float(np.nanpercentile(arr, 90)), "max": float(np.nanmax(arr))}

def diagnose_combined_daily(daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {"total_pnl": 0.0, "max_dd": 0.0, "ttr_p90": np.nan, "ttr_max": np.nan, "worst_20d": 0.0}
    total = float(daily["pnl"].sum())
    dd = max_drawdown(daily["equity"].to_numpy(dtype=float))
    ttr = time_to_recover_days(daily)
    worst20 = worst_window(daily["pnl"].to_numpy(dtype=float), 20)
    return {"total_pnl": total, "max_dd": dd, "ttr_p90": ttr["p90"], "ttr_max": ttr["max"], "worst_20d": worst20}

# -------------------------
# DATA LOADING
# -------------------------
def load_master_pocket() -> pd.DataFrame:
    cols = ["event_dt", "event_date", "menu_hint", "bsp", "won"]
    df = pd.read_parquet(MASTER_ROOT, columns=cols)

    df["event_dt"] = pd.to_datetime(df["event_dt"])
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date

    df = df[df["bsp"].notna() & (df["bsp"] > 1.0)].copy()
    df = df[(df["bsp"] >= BSP_LO) & (df["bsp"] < BSP_HI)].copy()

    df["track"] = df["menu_hint"].apply(canonical_track_from_menu_hint)
    df = df[df["track"].isin(UK_CANON_TRACKS)].copy()

    df["month"] = df["event_dt"].dt.to_period("M").astype(str)
    return df

# -------------------------
# TRAIN WINDOW HELPERS
# -------------------------
def month_range_inclusive(m0: str, m1: str) -> list[str]:
    p0 = pd.Period(m0, freq="M")
    p1 = pd.Period(m1, freq="M")
    return [str(p) for p in pd.period_range(p0, p1, freq="M")]

def train_months_for_test(test_month: str, mode: str) -> list[str]:
    t = pd.Period(test_month, freq="M")
    if mode == "expanding":
        # all months strictly before test_month
        return None  # sentinel: handled using df[df.month < test_month]
    if mode == "rolling_12m":
        start = t - 12
        end = t - 1
    elif mode == "rolling_24m":
        start = t - 24
        end = t - 1
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return [str(p) for p in pd.period_range(start, end, freq="M")]

# -------------------------
# RULE BUILDERS
# -------------------------
def pick_top_tracks(train_df: pd.DataFrame, top_n: int, min_bets: int) -> list[str]:
    if train_df.empty:
        return []
    g = train_df.groupby("track").agg(train_bets=("pnl", "size"), train_pnl=("pnl", "sum"))
    g = g[g["train_bets"] >= min_bets].copy()
    if g.empty:
        return []
    g = g.sort_values("train_pnl", ascending=False)
    return g.head(top_n).index.tolist()

def worst_band_on_train(train_df: pd.DataFrame) -> tuple[str, float]:
    """Returns (band_name, band_pnl) for the worst pnl band on TRAIN."""
    if train_df.empty:
        return ("", 0.0)
    rows = []
    for name, lo, hi in BANDS:
        d = train_df[(train_df["bsp"] >= lo) & (train_df["bsp"] < hi)]
        rows.append((name, float(d["pnl"].sum())))
    # "worst" = minimum pnl
    rows.sort(key=lambda x: x[1])
    return rows[0]

def apply_band_filter(df: pd.DataFrame, band_name: str) -> pd.DataFrame:
    if not band_name:
        return df
    for name, lo, hi in BANDS:
        if name == band_name:
            return df[~((df["bsp"] >= lo) & (df["bsp"] < hi))].copy()
    return df

# -------------------------
# MAIN GRID RUN
# -------------------------
def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Loading master (UK greyhound win, BSP 10–20)...")
    df = load_master_pocket()
    df["pnl"] = compute_pnl_1u(df)

    print(f"Pocket rows: {len(df):,}")
    if df.empty:
        raise RuntimeError("No data after pocket + UK filtering.")

    months_all = sorted(df["month"].unique().tolist())
    print(f"Available months in data: {months_all[0]} -> {months_all[-1]}")

    test_months = month_range_inclusive(TEST_MONTH_START, TEST_MONTH_END)
    print(f"Rolling months requested  : {test_months[0]} -> {test_months[-1]}")

    monthly_rows = []
    leaderboard_rows = []

    # For each variant we will also build the combined daily series across months
    # Store temporarily in memory; best variant gets written to parquet.
    variant_daily_map: dict[str, pd.DataFrame] = {}
    variant_meta_map: dict[str, dict] = {}

    for train_mode in TRAIN_MODE_LIST:
        for top_n in TOP_N_LIST:
            for band_rule in BAND_RULE_LIST:
                for min_bets in MIN_TRAIN_BETS_LIST:
                    variant = f"mode={train_mode}|topN={top_n}|band={band_rule}|minbets={min_bets}"

                    combined_daily_list = []
                    ok_months = 0
                    total_bets = 0

                    # Track decisions for this variant
                    decisions = []

                    for tm in test_months:
                        test_df = df[df["month"] == tm].copy()
                        if test_df.empty:
                            continue

                        # Build TRAIN
                        if train_mode == "expanding":
                            train_df = df[df["month"] < tm].copy()
                        else:
                            tmonths = train_months_for_test(tm, train_mode)
                            train_df = df[df["month"].isin(tmonths)].copy()

                        if train_df.empty:
                            continue

                        # Pick top tracks
                        tracks = pick_top_tracks(train_df, top_n=top_n, min_bets=min_bets)
                        if not tracks:
                            continue

                        train_sel = train_df[train_df["track"].isin(tracks)].copy()
                        worst_band_name, worst_band_pnl = worst_band_on_train(train_sel)

                        # Decide band to exclude
                        band_excluded = None
                        if band_rule == "none":
                            band_excluded = None
                        elif band_rule == "exclude_worst_always":
                            band_excluded = worst_band_name
                        elif band_rule == "exclude_worst_if_negative":
                            band_excluded = worst_band_name if worst_band_pnl < 0 else None
                        else:
                            raise ValueError(band_rule)

                        # Apply to TEST
                        test_sel = test_df[test_df["track"].isin(tracks)].copy()
                        if band_excluded:
                            test_sel = apply_band_filter(test_sel, band_excluded)

                        if test_sel.empty:
                            continue

                        total_bets += len(test_sel)

                        daily = to_daily(test_sel)
                        # Rebase equity to keep combined series consistent across months
                        # (We'll stitch and cumsum later anyway)
                        combined_daily_list.append(daily.assign(month=tm, variant=variant))
                        ok_months += 1

                        dd_month = max_drawdown(daily["equity"].to_numpy(dtype=float))
                        decisions.append({
                            "variant": variant,
                            "test_month": tm,
                            "train_mode": train_mode,
                            "top_n": top_n,
                            "band_rule": band_rule,
                            "min_train_bets": min_bets,
                            "tracks": ",".join(tracks),
                            "worst_band_train": worst_band_name,
                            "worst_band_train_pnl": worst_band_pnl,
                            "band_excluded": band_excluded or "",
                            "test_bets": int(len(test_sel)),
                            "test_pnl": float(test_sel["pnl"].sum()),
                            "test_max_dd": float(dd_month),
                        })

                    if ok_months < 3:
                        # too sparse; skip writing to leaderboard
                        continue

                    # Combine daily across months in chronological order
                    cd = pd.concat(combined_daily_list, ignore_index=True)
                    cd["event_date"] = pd.to_datetime(cd["event_date"])
                    cd = cd.sort_values("event_date").copy()
                    cd["pnl"] = cd["pnl"].astype(float)
                    cd["equity"] = cd["pnl"].cumsum()

                    diag = diagnose_combined_daily(cd)

                    leaderboard_rows.append({
                        "variant": variant,
                        "train_mode": train_mode,
                        "top_n": top_n,
                        "band_rule": band_rule,
                        "min_train_bets": min_bets,
                        "months": ok_months,
                        "total_bets": int(total_bets),
                        "total_pnl": diag["total_pnl"],
                        "max_dd": diag["max_dd"],
                        "profit_dd_ratio": (diag["total_pnl"] / abs(diag["max_dd"])) if diag["max_dd"] != 0 else np.nan,
                        "ttr_p90_days": diag["ttr_p90"],
                        "ttr_max_days": diag["ttr_max"],
                        "worst_20d": diag["worst_20d"],
                    })

                    monthly_rows.extend(decisions)

                    variant_daily_map[variant] = cd
                    variant_meta_map[variant] = {
                        "variant": variant,
                        "train_mode": train_mode,
                        "top_n": top_n,
                        "band_rule": band_rule,
                        "min_train_bets": min_bets,
                        "test_months": test_months,
                        "pocket": {"bsp_lo": BSP_LO, "bsp_hi": BSP_HI},
                        "seed": SEED,
                        "bands": BANDS,
                    }

    if not leaderboard_rows:
        raise RuntimeError("No variants produced results. Consider lowering MIN_TRAIN_BETS_LIST or widening month range.")

    df_lead = pd.DataFrame(leaderboard_rows)

    # Sort priority: total pnl desc, max_dd (less negative better => higher), worst_20d higher, then ratio
    df_lead = df_lead.sort_values(
        by=["total_pnl", "max_dd", "worst_20d", "profit_dd_ratio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    df_monthly = pd.DataFrame(monthly_rows)

    lead_path = OUT_TABLES / "grid_leaderboard_step18.csv"
    monthly_path = OUT_TABLES / "grid_monthly_results_step18.csv"

    df_lead.to_csv(lead_path, index=False)
    df_monthly.to_csv(monthly_path, index=False)

    # Save best variant daily + meta
    best_variant = str(df_lead.loc[0, "variant"])
    best_daily = variant_daily_map[best_variant]
    best_daily_path = OUT_TABLES / "best_variant_daily_step18.parquet"
    best_daily.to_parquet(best_daily_path, index=False)

    best_meta = variant_meta_map[best_variant]
    best_meta["best_leaderboard_row"] = df_lead.loc[0].to_dict()
    best_meta_path = OUT_TABLES / "best_variant_meta_step18.json"
    best_meta_path.write_text(json.dumps(best_meta, indent=2))

    print("\n=== STEP 18 GRID COMPLETE ===")
    print(f"Saved leaderboard : {lead_path}")
    print(f"Saved monthly     : {monthly_path}")
    print(f"Best variant      : {best_variant}")
    print(f"Saved best daily  : {best_daily_path}")
    print(f"Saved best meta   : {best_meta_path}")

    print("\nTop 10 variants:")
    print(df_lead.head(10).to_string(index=False))


if __name__ == "__main__":
    main()