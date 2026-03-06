# scripts/15_top3_forensics_decomposition.py
"""
Step 15: Forensic decomposition of WALK-FORWARD winners (TEST period only)

Focus:
- WF_TOP3_TRAIN_PNL (primary)
- WF_TOP5_TRAIN_PNL (comparison)
- WF_TOP10_TRAIN_PNL (optional)
- WF_ALL_UK_10_20 (baseline)

Outputs per basket:
1) Track contribution table (pnl, bets, winrate, max_dd, profit/dd ratio)
2) BSP sub-band decomposition: 10–12, 12–15, 15–20
3) Monthly pnl table in TEST
4) Concentration tests:
   - Share of pnl from top X% bets (by pnl)
   - Remove top X% winning bets, recompute pnl + max_dd
   - Remove top N best days, recompute pnl + max_dd

Assumptions:
- Master parquet exists at: data/master/bfsp/uk/greyhound/win
- Step 14 created basket track list at:
  reports/bfsp/uk/greyhound/win/walkforward/stress_tests/_BASKET_TRACKS.txt
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

WALKFWD_DIR = Path("reports/bfsp/uk/greyhound/win/walkforward")
STEP14_TRACKLIST = WALKFWD_DIR / "stress_tests" / "_BASKET_TRACKS.txt"

OUT_DIR = WALKFWD_DIR / "forensics_step15"

TEST_START = pd.Timestamp("2022-01-01")
TEST_END   = pd.Timestamp("2023-02-20 23:59:59")

BSP_LO = 10.0
BSP_HI = 20.0

BASKETS_TO_ANALYZE = [
    "WF_TOP3_TRAIN_PNL",
    "WF_TOP5_TRAIN_PNL",
    "WF_TOP10_TRAIN_PNL",
    "WF_ALL_UK_10_20",
]

# sub-bands inside 10–20
BSP_SUBBANDS = [
    (10.0, 12.0),
    (12.0, 15.0),
    (15.0, 20.0),
]

# concentration stress
REMOVE_TOP_WINNER_PCTS = [0.5, 1.0, 2.0, 5.0]   # % of winning bets
REMOVE_TOP_BET_PCTS    = [0.5, 1.0, 2.0, 5.0]   # % of all bets by pnl
REMOVE_TOP_N_DAYS_LIST = [5, 10, 20]


# -------------------------
# TRACK CANONICALISATION (same as Step 14 "spirit")
# -------------------------
TRACK_ALIAS = {
    "cpark": "Central Park",
    "pbarr": "Perry Barr",
    "perry": "Perry Barr",          # critical: Step 13 used "Perry"
    "p barr": "Perry Barr",
    "romfd": "Romford",
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

# Explicit UK(+IRE-ish) canonical list used in your earlier pipeline outputs
UK_CANON_TRACKS = {
    "Monmore", "Romford", "Hove", "Crayford", "Newcastle", "Central Park", "Harlow",
    "Sheffield", "Sunderland", "Swindon", "Perry Barr", "Nottingham", "Henlow",
    "Towcester", "Doncaster", "Kinsley", "Yarmouth", "Oxford", "Poole", "Valley",
    "Pelaw Grange", "Star Pelaw", "Peterborough", "Bvue",
    "Mullingar", "Shelbourne Park", "Suffolk Downs",
}

MENU_SPLIT_RE = re.compile(r"/\s*(.+)$")
DATE_SUFFIX_RE = re.compile(r"\s+\d{1,2}(st|nd|rd|th)\s+[A-Za-z]{3,}\s*$", flags=re.IGNORECASE)


def canonical_track_from_menu_hint(menu_hint: str) -> tuple[str, str, str]:
    if not isinstance(menu_hint, str) or not menu_hint.strip():
        return ("", "", "UNKNOWN")

    s = menu_hint.strip()
    m = MENU_SPLIT_RE.search(s)
    rhs = m.group(1).strip() if m else s

    rhs = DATE_SUFFIX_RE.sub("", rhs).strip()

    track_raw = rhs
    key = re.sub(r"\s+", " ", track_raw.strip().lower())

    track = TRACK_ALIAS.get(key)
    if track is None:
        first = key.split(" ", 1)[0]
        track = TRACK_ALIAS.get(first)

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
# PNL + DIAGNOSTICS
# -------------------------
def compute_pnl_1u(won: np.ndarray, bsp: np.ndarray, stake: float = 1.0) -> np.ndarray:
    return np.where(won == 1, stake * (bsp - 1.0), -stake)


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("event_date", as_index=False)
        .agg(pnl=("pnl", "sum"), n_bets=("pnl", "size"))
        .sort_values("event_date")
    )
    daily["equity"] = daily["pnl"].cumsum()
    return daily


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def profit_dd_ratio(total_pnl: float, max_dd_value: float) -> float:
    if max_dd_value == 0:
        return np.nan
    return float(total_pnl / abs(max_dd_value))


def basic_summary_from_bets(df_bets: pd.DataFrame) -> dict:
    if df_bets.empty:
        return {"total_pnl": 0.0, "max_dd": 0.0, "n_bets": 0, "win_rate": np.nan, "ratio": np.nan}

    daily = to_daily(df_bets)
    total = float(daily["pnl"].sum())
    dd = max_drawdown(daily["equity"].to_numpy(dtype=float))
    wr = float(df_bets["won"].mean()) if len(df_bets) else np.nan
    return {
        "total_pnl": total,
        "max_dd": dd,
        "n_bets": int(len(df_bets)),
        "win_rate": wr,
        "ratio": profit_dd_ratio(total, dd),
    }


# -------------------------
# LOAD BASKET TRACKS
# -------------------------
def load_basket_tracks() -> dict[str, list[str]]:
    tracks: dict[str, list[str]] = {}
    if not STEP14_TRACKLIST.exists():
        raise FileNotFoundError(f"Missing Step 14 track list: {STEP14_TRACKLIST}")

    cur = None
    for line in STEP14_TRACKLIST.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith(":"):
            name = line[:-1].strip()
            cur = name
            tracks[cur] = []
            continue
        if cur is not None and line.startswith("-"):
            t = line.lstrip("-").strip()
            # normalize alias like "Perry" -> "Perry Barr"
            key = re.sub(r"\s+", " ", t.strip().lower())
            t2 = TRACK_ALIAS.get(key, t.strip())
            tracks[cur].append(t2)

    return tracks


# -------------------------
# FORENSIC MODULES
# -------------------------
def track_contribution(df_bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trk, g in df_bets.groupby("track"):
        daily = to_daily(g)
        total = float(daily["pnl"].sum())
        dd = max_drawdown(daily["equity"].to_numpy(dtype=float))
        rows.append({
            "track": trk,
            "n_bets": int(len(g)),
            "win_rate": float(g["won"].mean()) if len(g) else np.nan,
            "total_pnl": total,
            "max_dd": dd,
            "profit_dd_ratio": profit_dd_ratio(total, dd),
        })
    out = pd.DataFrame(rows).sort_values(["total_pnl", "profit_dd_ratio"], ascending=False)
    return out


def bsp_subband_decomposition(df_bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi in BSP_SUBBANDS:
        d = df_bets[(df_bets["bsp"] >= lo) & (df_bets["bsp"] < hi)].copy()
        s = basic_summary_from_bets(d)
        rows.append({
            "band": f"{lo:.0f}-{hi:.0f}",
            "bsp_lo": lo,
            "bsp_hi": hi,
            **s
        })
    return pd.DataFrame(rows).sort_values("bsp_lo")


def monthly_pnl(df_bets: pd.DataFrame) -> pd.DataFrame:
    d = df_bets.copy()
    d["month"] = pd.to_datetime(d["event_dt"]).dt.to_period("M").astype(str)
    out = (
        d.groupby("month", as_index=False)
        .agg(total_pnl=("pnl", "sum"), n_bets=("pnl", "size"), win_rate=("won", "mean"))
        .sort_values("month")
    )
    return out


def concentration_tests(df_bets: pd.DataFrame) -> pd.DataFrame:
    """
    Concentration/fragility checks:
    - pnl share from top X% bets (by pnl)
    - remove top X% winning bets and recompute
    - remove top X% bets (all, by pnl) and recompute
    - remove top N best days and recompute
    """
    base = basic_summary_from_bets(df_bets)

    rows = []

    pnl_all = df_bets["pnl"].to_numpy(dtype=float)
    total_pnl = float(pnl_all.sum()) if len(pnl_all) else 0.0

    # Share from top X% bets (by pnl)
    if len(df_bets):
        df_sorted = df_bets.sort_values("pnl", ascending=False)
        for pct in [0.5, 1.0, 2.0, 5.0, 10.0]:
            k = max(1, int(round(len(df_sorted) * (pct / 100.0))))
            share = float(df_sorted["pnl"].head(k).sum() / total_pnl) if total_pnl != 0 else np.nan
            rows.append({
                "test": "share_top_pct_bets_by_pnl",
                "param": pct,
                "metric": "pnl_share",
                "value": share,
                **{f"base_{k}": v for k, v in base.items()},
            })

    # Remove top X% winning bets (biggest winners)
    winners = df_bets[df_bets["pnl"] > 0].copy().sort_values("pnl", ascending=False)
    for pct in REMOVE_TOP_WINNER_PCTS:
        if winners.empty:
            continue
        k = max(1, int(round(len(winners) * (pct / 100.0))))
        drop_idx = winners.head(k).index
        d = df_bets.drop(index=drop_idx).copy()
        s = basic_summary_from_bets(d)
        rows.append({
            "test": "remove_top_pct_winning_bets",
            "param": pct,
            "metric": "after_total_pnl",
            "value": s["total_pnl"],
            "after_max_dd": s["max_dd"],
            "after_ratio": s["ratio"],
            **{f"base_{k}": v for k, v in base.items()},
        })

    # Remove top X% bets overall (by pnl)
    df_sorted = df_bets.sort_values("pnl", ascending=False)
    for pct in REMOVE_TOP_BET_PCTS:
        if df_sorted.empty:
            continue
        k = max(1, int(round(len(df_sorted) * (pct / 100.0))))
        drop_idx = df_sorted.head(k).index
        d = df_bets.drop(index=drop_idx).copy()
        s = basic_summary_from_bets(d)
        rows.append({
            "test": "remove_top_pct_bets_by_pnl",
            "param": pct,
            "metric": "after_total_pnl",
            "value": s["total_pnl"],
            "after_max_dd": s["max_dd"],
            "after_ratio": s["ratio"],
            **{f"base_{k}": v for k, v in base.items()},
        })

    # Remove top N best days
    daily = to_daily(df_bets)
    for n in REMOVE_TOP_N_DAYS_LIST:
        d = daily.copy()
        idx = d["pnl"].nlargest(n).index
        d.loc[idx, "pnl"] = 0.0
        d["equity"] = d["pnl"].cumsum()
        total = float(d["pnl"].sum())
        dd = max_drawdown(d["equity"].to_numpy(dtype=float))
        rows.append({
            "test": "remove_top_n_best_days",
            "param": n,
            "metric": "after_total_pnl",
            "value": total,
            "after_max_dd": dd,
            "after_ratio": profit_dd_ratio(total, dd),
            **{f"base_{k}": v for k, v in base.items()},
        })

    return pd.DataFrame(rows)


# -------------------------
# MAIN
# -------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading TEST bets from master (UK greyhound win, BSP 10–20)...")
    cols = ["event_dt", "event_date", "menu_hint", "bsp", "won"]
    df = pd.read_parquet(MASTER_ROOT, columns=cols)

    # Basic cleaning
    df = df[df["bsp"].notna() & (df["bsp"] > 1.0)].copy()
    df["event_dt"] = pd.to_datetime(df["event_dt"])
    df = df[(df["event_dt"] >= TEST_START) & (df["event_dt"] <= TEST_END)].copy()
    df = df[(df["bsp"] >= BSP_LO) & (df["bsp"] < BSP_HI)].copy()

    # Canonical tracks + UK filter
    df = add_track_columns(df)
    df = df[df["track"].isin(UK_CANON_TRACKS)].copy()

    if df.empty:
        raise RuntimeError("No rows after TEST + pocket + UK filtering. Check config or master data.")

    # Bet pnl
    df["won"] = df["won"].astype("int8")
    df["bsp"] = df["bsp"].astype("float64")
    df["pnl"] = compute_pnl_1u(df["won"].to_numpy(), df["bsp"].to_numpy(), stake=1.0)

    print(f"Loaded bets: {len(df):,}")
    print(f"Date range : {df['event_dt'].min()} -> {df['event_dt'].max()}")
    print(f"Tracks     : {df['track'].nunique()}")

    basket_tracks = load_basket_tracks()

    # Always define ALL_UK basket based on what we loaded
    basket_tracks["WF_ALL_UK_10_20"] = sorted(df["track"].unique().tolist())

    meta = {
        "test_start": str(TEST_START),
        "test_end": str(TEST_END),
        "bsp_lo": BSP_LO,
        "bsp_hi": BSP_HI,
        "n_bets_loaded": int(len(df)),
        "n_tracks_loaded": int(df["track"].nunique()),
        "baskets": {},
    }

    for basket in BASKETS_TO_ANALYZE:
        print("\n" + "=" * 70)
        print(f"FORENSICS: {basket}")

        tracks = basket_tracks.get(basket, [])
        if basket != "WF_ALL_UK_10_20" and not tracks:
            print(f"WARNING: Missing track list for {basket}. Skipping.")
            continue

        d = df.copy()
        if basket != "WF_ALL_UK_10_20":
            d = d[d["track"].isin(tracks)].copy()

        base = basic_summary_from_bets(d)
        print(f"Base: pnl={base['total_pnl']:.2f} | dd={base['max_dd']:.2f} | ratio={base['ratio']:.3f} | bets={base['n_bets']:,}")

        # 1) Track contribution
        tc = track_contribution(d)
        tc_path = OUT_DIR / f"{basket}_track_contribution.csv"
        tc.to_csv(tc_path, index=False)

        # 2) BSP subbands
        sb = bsp_subband_decomposition(d)
        sb_path = OUT_DIR / f"{basket}_bsp_subbands.csv"
        sb.to_csv(sb_path, index=False)

        # 3) Monthly pnl
        mp = monthly_pnl(d)
        mp_path = OUT_DIR / f"{basket}_monthly_pnl.csv"
        mp.to_csv(mp_path, index=False)

        # 4) Concentration tests
        ct = concentration_tests(d)
        ct_path = OUT_DIR / f"{basket}_concentration_tests.csv"
        ct.to_csv(ct_path, index=False)

        meta["baskets"][basket] = {
            "tracks": tracks,
            "base": base,
            "paths": {
                "track_contribution": str(tc_path),
                "bsp_subbands": str(sb_path),
                "monthly_pnl": str(mp_path),
                "concentration_tests": str(ct_path),
            }
        }

        # quick console highlights (top 10 by pnl)
        if not tc.empty:
            print("Top tracks by pnl (TEST):")
            print(tc.head(10)[["track", "n_bets", "total_pnl", "max_dd", "profit_dd_ratio"]].to_string(index=False))

        if not sb.empty:
            print("BSP sub-bands (TEST):")
            print(sb[["band", "n_bets", "total_pnl", "max_dd", "ratio"]].to_string(index=False))

    # Save meta
    meta_path = OUT_DIR / "_META_step15_forensics.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print("\nSaved Step 15 outputs to:", OUT_DIR)
    print("Meta:", meta_path)


if __name__ == "__main__":
    main()