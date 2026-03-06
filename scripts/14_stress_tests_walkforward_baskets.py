# scripts/14_stress_tests_walkforward_baskets.py
"""
Step 14 (corrected): Stress tests on WALK-FORWARD baskets (TEST period only),
using the SAME canonical track parsing + UK filter logic as Step 9/10/13.

Key fixes:
- Build WF_TOP3/WF_TOP5/WF_TOP10 track lists directly from Step 13 TRAIN ranking (train_track_ranking.csv).
- Remove df_test bug; WF_ALL_UK_10_20 is defined from the filtered TEST dataframe.
- Separate "basket track aliases" (e.g. Perry -> Perry Barr) from track parsing/canonical mapping.
- Avoid duplicate constants/imports and accidental re-definitions.
"""

from __future__ import annotations

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd

# -------------------------
# PATHS / CONFIG
# -------------------------
MASTER_ROOT = Path("data/master/bfsp/uk/greyhound/win")

WALKFWD_DIR = Path("reports/bfsp/uk/greyhound/win/walkforward")
TABLES_DIR = WALKFWD_DIR / "tables"
OUT_DIR = WALKFWD_DIR / "stress_tests"
TRAIN_RANKING_PATH = TABLES_DIR / "train_track_ranking.csv"

# Must match your walk-forward split
TEST_START = pd.Timestamp("2022-01-01")
TEST_END = pd.Timestamp("2023-02-20 23:59:59")

# Pocket definition
BSP_LO = 10.0
BSP_HI = 20.0

# Stress test parameters
SEED = 42
N_PERM = 2000
N_BOOT = 2000
REMOVE_TOP_N_DAYS = 10

BASKET_NAMES = [
    "WF_TOP3_TRAIN_PNL",
    "WF_TOP5_TRAIN_PNL",
    "WF_TOP10_TRAIN_PNL",
    "WF_ALL_UK_10_20",
]

# -------------------------
# Basket-name aliases (so Step 13 labels match Step 14 filtering)
# -------------------------
BASKET_TRACK_ALIAS = {
    "perry": "Perry Barr",
}

def norm_basket_track_name(t: str) -> str:
    key = re.sub(r"\s+", " ", str(t).strip().lower())
    return BASKET_TRACK_ALIAS.get(key, str(t).strip())

def build_basket_tracks_from_train_ranking() -> dict[str, list[str]]:
    if not TRAIN_RANKING_PATH.exists():
        raise FileNotFoundError(f"Missing TRAIN ranking file: {TRAIN_RANKING_PATH}")

    r = pd.read_csv(TRAIN_RANKING_PATH)

    # Normalize (important: Step 13 may include 'Perry')
    r["track"] = r["track"].apply(norm_basket_track_name)

    # Rank by train_pnl (same as Step 13 baskets)
    r = r.sort_values("train_pnl", ascending=False)

    return {
        "WF_TOP3_TRAIN_PNL": r["track"].head(3).tolist(),
        "WF_TOP5_TRAIN_PNL": r["track"].head(5).tolist(),
        "WF_TOP10_TRAIN_PNL": r["track"].head(10).tolist(),
    }

# -------------------------
# TRACK CANONICALISATION
# -------------------------
TRACK_CANON_MAP = {
    # common abbreviations from menu_hint
    "cpark": "Central Park",
    "pbarr": "Perry Barr",
    "perry": "Perry Barr",          # important fix
    "p barr": "Perry Barr",
    "romfd": "Romford",
    "romford": "Romford",
    "romf": "Romford",
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
    # already full names that appear
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
    "Mullingar", "Shelbourne Park", "Suffolk Downs",
}

MENU_SPLIT_RE = re.compile(r"/\s*(.+)$")  # take RHS of "X / RHS"
DATE_SUFFIX_RE = re.compile(
    r"\s+\d{1,2}(st|nd|rd|th)\s+[A-Za-z]{3,}\s*$",
    flags=re.IGNORECASE,
)

def canonical_track_from_menu_hint(menu_hint: str) -> tuple[str, str, str]:
    """
    Returns (track_raw, track_key, track_canon)

    Example:
      "Morning Cards / CPark 29th Jan" -> ("CPark", "cpark", "Central Park")
    """
    if not isinstance(menu_hint, str) or not menu_hint.strip():
        return ("", "", "UNKNOWN")

    s = menu_hint.strip()

    m = MENU_SPLIT_RE.search(s)
    rhs = m.group(1).strip() if m else s

    rhs = DATE_SUFFIX_RE.sub("", rhs).strip()

    track_raw = rhs
    key = re.sub(r"\s+", " ", track_raw.strip().lower())

    track = TRACK_CANON_MAP.get(key)
    if track is None:
        first = key.split(" ", 1)[0]
        track = TRACK_CANON_MAP.get(first)

    if track is None:
        track = track_raw.strip().title()

    return (track_raw, key, track)

def add_track_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["menu_hint"].apply(canonical_track_from_menu_hint)
    out["track_raw"] = parsed.apply(lambda x: x[0])
    out["track_key"] = parsed.apply(lambda x: x[1])
    out["track"] = parsed.apply(lambda x: x[2])
    return out

# -------------------------
# CORE METRICS / DIAGNOSTICS
# -------------------------
def compute_pnl_1u(df: pd.DataFrame, stake: float = 1.0) -> np.ndarray:
    won = df["won"].astype("int8").to_numpy()
    bsp = df["bsp"].astype("float64").to_numpy()
    return np.where(won == 1, stake * (bsp - 1.0), -stake)

def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("event_date", as_index=False)
        .agg(pnl=("pnl", "sum"), n_bets=("pnl", "size"), avg_bsp=("bsp", "mean"))
        .sort_values("event_date")
    )
    daily["equity"] = daily["pnl"].cumsum()
    return daily

def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())

def time_to_recover_days(daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {"median": np.nan, "p90": np.nan, "max": np.nan}

    eq = daily["equity"].to_numpy()
    dates = pd.to_datetime(daily["event_date"]).to_numpy()

    peak = np.maximum.accumulate(eq)
    underwater = eq < peak

    recover_times = []
    i, n = 0, len(eq)
    while i < n:
        if not underwater[i]:
            i += 1
            continue
        dd_start = i
        target = peak[i]
        j = i
        while j < n and eq[j] < target:
            j += 1
        if j < n:
            delta = (pd.Timestamp(dates[j]) - pd.Timestamp(dates[dd_start])).days
            recover_times.append(delta)
        i = j + 1

    if not recover_times:
        return {"median": np.nan, "p90": np.nan, "max": np.nan}

    arr = np.array(recover_times, dtype=float)
    return {
        "median": float(np.nanmedian(arr)),
        "p90": float(np.nanpercentile(arr, 90)),
        "max": float(np.nanmax(arr)),
    }

def worst_window(daily_pnl: np.ndarray, window: int) -> float:
    if len(daily_pnl) < window:
        return float(daily_pnl.sum()) if len(daily_pnl) else 0.0
    s = pd.Series(daily_pnl).rolling(window).sum()
    return float(s.min())

def streaks_negative_days(daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {"max": 0, "p90": 0}
    neg = (daily["pnl"].to_numpy() < 0).astype(int)
    streaks, cur = [], 0
    for x in neg:
        if x == 1:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    if not streaks:
        return {"max": 0, "p90": 0}
    arr = np.array(streaks)
    return {"max": int(arr.max()), "p90": float(np.percentile(arr, 90))}

def edge_metrics_bet_level(df: pd.DataFrame) -> dict:
    dfv = df[df["bsp"].notna() & (df["bsp"] > 1.0)].copy()
    if dfv.empty:
        return {"win_rate": np.nan, "implied": np.nan, "edge": np.nan, "ev_per_bet": np.nan}
    win_rate = float(dfv["won"].mean())
    implied = float((1.0 / dfv["bsp"]).mean())
    edge = win_rate - implied
    ev = float(np.mean(dfv["pnl"].to_numpy(dtype=float)))
    return {"win_rate": win_rate, "implied": implied, "edge": float(edge), "ev_per_bet": ev}

def diagnose_daily(daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {
            "total_pnl": 0.0, "n_bets": 0, "n_days": 0, "max_dd": 0.0,
            "ttr_median_days": np.nan, "ttr_p90_days": np.nan, "ttr_max_days": np.nan,
            "streak_day_max": 0, "streak_day_p90": 0,
            "worst_1d": 0.0, "worst_5d": 0.0, "worst_20d": 0.0,
        }

    total = float(daily["pnl"].sum())
    n_bets = int(daily["n_bets"].sum())
    n_days = int(len(daily))
    dd = max_drawdown(daily["equity"].to_numpy(dtype=float))
    ttr = time_to_recover_days(daily)
    streak = streaks_negative_days(daily)
    pnl = daily["pnl"].to_numpy(dtype=float)

    return {
        "total_pnl": total,
        "n_bets": n_bets,
        "n_days": n_days,
        "max_dd": dd,
        "ttr_median_days": ttr["median"],
        "ttr_p90_days": ttr["p90"],
        "ttr_max_days": ttr["max"],
        "streak_day_max": streak["max"],
        "streak_day_p90": streak["p90"],
        "worst_1d": float(pnl.min()),
        "worst_5d": worst_window(pnl, 5),
        "worst_20d": worst_window(pnl, 20),
    }

# -------------------------
# STRESS TESTS
# -------------------------
def perm_test_month_proxy_and_dd(daily: pd.DataFrame, rng: np.random.Generator) -> dict:
    """
    Permute daily pnl order.
    - Total pnl is invariant, so for "total robustness" we use a proxy: best-month pnl.
    - For drawdown robustness we compute how often permuted maxDD is worse-or-equal to actual.
    """
    pnl = daily["pnl"].to_numpy(dtype=float)

    d = daily.copy()
    d["month"] = pd.to_datetime(d["event_date"]).dt.to_period("M").astype(str)

    actual_eq = daily["equity"].to_numpy(dtype=float)
    actual_dd = max_drawdown(actual_eq)
    best_month_actual = float(d.groupby("month")["pnl"].sum().max())

    count_best_month_ge = 0
    count_dd_worse_or_eq = 0

    for _ in range(N_PERM):
        perm = rng.permutation(pnl)
        perm_eq = np.cumsum(perm)
        perm_dd = max_drawdown(perm_eq)

        # "worse-or-equal" means more negative (<=) since DD is negative
        if perm_dd <= actual_dd:
            count_dd_worse_or_eq += 1

        tmp = d.copy()
        tmp["pnl_perm"] = perm
        best_month_perm = float(tmp.groupby("month")["pnl_perm"].sum().max())
        if best_month_perm >= best_month_actual:
            count_best_month_ge += 1

    return {
        "p_best_month_ge_actual": count_best_month_ge / N_PERM,
        "p_dd_worse_or_equal_actual": count_dd_worse_or_eq / N_PERM,
        "actual_best_month": best_month_actual,
    }

def bootstrap_ci_total_and_dd(daily: pd.DataFrame, rng: np.random.Generator) -> dict:
    pnl = daily["pnl"].to_numpy(dtype=float)
    n = len(pnl)
    totals = np.empty(N_BOOT, dtype=float)
    dds = np.empty(N_BOOT, dtype=float)

    for i in range(N_BOOT):
        sample = pnl[rng.integers(0, n, size=n)]
        totals[i] = float(sample.sum())
        dds[i] = max_drawdown(np.cumsum(sample))

    return {
        "total_ci95": (float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5))),
        "dd_ci95": (float(np.percentile(dds, 2.5)), float(np.percentile(dds, 97.5))),
    }

def remove_top_days(daily: pd.DataFrame, n_remove: int) -> pd.DataFrame:
    if daily.empty:
        return daily
    d = daily.copy()
    idx = d["pnl"].nlargest(n_remove).index
    d.loc[idx, "pnl"] = 0.0
    d["equity"] = d["pnl"].cumsum()
    return d

def remove_best_month(daily: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if daily.empty:
        return daily, ""
    d = daily.copy()
    d["month"] = pd.to_datetime(d["event_date"]).dt.to_period("M").astype(str)
    month_pnl = d.groupby("month")["pnl"].sum()
    best_month = str(month_pnl.idxmax())
    d.loc[d["month"] == best_month, "pnl"] = 0.0
    d = d.drop(columns=["month"])
    d["equity"] = d["pnl"].cumsum()
    return d, best_month

# -------------------------
# DATA LOADING
# -------------------------
def load_test_bet_level() -> pd.DataFrame:
    cols = ["event_dt", "event_date", "menu_hint", "bsp", "won"]
    df = pd.read_parquet(MASTER_ROOT, columns=cols)

    df = df[df["bsp"].notna() & (df["bsp"] > 1.0)].copy()
    df = df[(df["bsp"] >= BSP_LO) & (df["bsp"] < BSP_HI)].copy()

    df["event_dt"] = pd.to_datetime(df["event_dt"])
    df = df[(df["event_dt"] >= TEST_START) & (df["event_dt"] <= TEST_END)].copy()

    df = add_track_columns(df)
    df = df[df["track"].isin(UK_CANON_TRACKS)].copy()

    return df

# -------------------------
# MAIN
# -------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Loading TEST pocket (UK greyhound win, BSP 10–20) bet-level data...")
    df = load_test_bet_level()

    print(f"Loaded bets: {len(df):,}")
    if df.empty:
        raise RuntimeError("No rows after TEST+POCKET+UK filtering. Check track parsing / UK set.")

    print(f"Date range: {df['event_dt'].min()} -> {df['event_dt'].max()}")
    print(f"Unique tracks: {df['track'].nunique()}")

    # Build baskets from TRAIN ranking (Step 13)
    basket_tracks = build_basket_tracks_from_train_ranking()

    # Define ALL basket from current filtered df (no df_test bug)
    basket_tracks["WF_ALL_UK_10_20"] = sorted(df["track"].unique().tolist())

    # Save track lists used
    track_list_path = OUT_DIR / "_BASKET_TRACKS.txt"
    with track_list_path.open("w") as f:
        for b in BASKET_NAMES:
            f.write(f"{b}:\n")
            for t in basket_tracks.get(b, []):
                f.write(f"  - {t}\n")
            f.write("\n")

    rows_out: list[dict] = []
    json_out: dict = {}

    for basket in BASKET_NAMES:
        print("\n" + "=" * 60)
        print(f"Running stress tests: {basket}")

        tracks = basket_tracks.get(basket, [])
        if not tracks:
            print(f"WARNING: No track list built for {basket}. Skipping.")
            continue

        dff = df[df["track"].isin(tracks)].copy()

        dff["pnl"] = compute_pnl_1u(dff, stake=1.0)
        daily = to_daily(dff)

        base_diag = diagnose_daily(daily)
        print(
            f"BASE: total_pnl={base_diag['total_pnl']:.2f} | "
            f"max_dd={base_diag['max_dd']:.2f} | "
            f"ttr_p90={base_diag['ttr_p90_days']:.1f} | "
            f"worst_20d={base_diag['worst_20d']:.2f}"
        )

        perm = perm_test_month_proxy_and_dd(daily, rng)
        boot = bootstrap_ci_total_and_dd(daily, rng)
        print(
            f"PERM p(best_month>=actual)={perm['p_best_month_ge_actual']:.3f} | "
            f"p(dd_worse_or_equal_actual)={perm['p_dd_worse_or_equal_actual']:.3f}"
        )
        print(f"BOOT total_ci95={boot['total_ci95']} | dd_ci95={boot['dd_ci95']}")

        daily_rm = remove_top_days(daily, REMOVE_TOP_N_DAYS)
        rm_diag = diagnose_daily(daily_rm)
        print(
            f"REMOVE TOP {REMOVE_TOP_N_DAYS} DAYS: "
            f"total_pnl={rm_diag['total_pnl']:.2f} | max_dd={rm_diag['max_dd']:.2f}"
        )

        daily_m, best_m = remove_best_month(daily)
        m_diag = diagnose_daily(daily_m)
        print(f"REMOVE BEST MONTH ({best_m}): total_pnl={m_diag['total_pnl']:.2f} | max_dd={m_diag['max_dd']:.2f}")

        edge = edge_metrics_bet_level(dff)
        print(
            f"EDGE: win_rate={edge['win_rate']:.4f} | implied={edge['implied']:.4f} | "
            f"edge={edge['edge']:.4f} | EV/bet={edge['ev_per_bet']:.4f}"
        )

        out = {
            "basket": basket,
            "tracks": tracks,
            "base": base_diag,
            "perm": perm,
            "boot": boot,
            "remove_top_days": rm_diag,
            "remove_best_month": {"best_month": best_m, **m_diag},
            "edge": edge,
            "meta": {
                "test_start": str(TEST_START),
                "test_end": str(TEST_END),
                "bsp_lo": BSP_LO,
                "bsp_hi": BSP_HI,
                "seed": SEED,
                "n_perm": N_PERM,
                "n_boot": N_BOOT,
                "remove_top_n_days": REMOVE_TOP_N_DAYS,
            },
        }
        json_out[basket] = out

        rows_out.append({
            "basket": basket,
            "n_tracks": len(tracks),
            "total_pnl": base_diag["total_pnl"],
            "max_dd": base_diag["max_dd"],
            "ttr_p90_days": base_diag["ttr_p90_days"],
            "worst_20d": base_diag["worst_20d"],
            "perm_p_best_month_ge_actual": perm["p_best_month_ge_actual"],
            "perm_p_dd_worse_or_equal": perm["p_dd_worse_or_equal_actual"],
            "boot_total_ci95_lo": boot["total_ci95"][0],
            "boot_total_ci95_hi": boot["total_ci95"][1],
            "boot_dd_ci95_lo": boot["dd_ci95"][0],
            "boot_dd_ci95_hi": boot["dd_ci95"][1],
            "rm_top_days_total_pnl": rm_diag["total_pnl"],
            "rm_top_days_max_dd": rm_diag["max_dd"],
            "rm_best_month": best_m,
            "rm_best_month_total_pnl": m_diag["total_pnl"],
            "rm_best_month_max_dd": m_diag["max_dd"],
            "edge_win_rate": edge["win_rate"],
            "edge_implied": edge["implied"],
            "edge": edge["edge"],
            "ev_per_bet": edge["ev_per_bet"],
        })

    json_path = OUT_DIR / "_SUMMARY_step14_stress_tests.json"
    csv_path = OUT_DIR / "_SUMMARY_step14_stress_tests.csv"

    json_path.write_text(json.dumps(json_out, indent=2))
    pd.DataFrame(rows_out).sort_values("total_pnl", ascending=False).to_csv(csv_path, index=False)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved track list: {track_list_path}")

if __name__ == "__main__":
    main()