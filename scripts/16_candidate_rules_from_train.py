# scripts/16_candidate_rules_from_train.py
"""
Step 16: Candidate rule construction from TRAIN only,
evaluation on TEST only.

Candidates:
A) TOP2 tracks (by TRAIN pnl)
B) TOP3 tracks but exclude worst BSP sub-band (chosen using TRAIN)

Outputs:
- Train diagnostics
- Test diagnostics for each candidate
- CSV summary table
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

OUT_DIR = Path("reports/bfsp/uk/greyhound/win/walkforward/step16_candidates")

TRAIN_START = pd.Timestamp("2017-01-01")
TRAIN_END   = pd.Timestamp("2021-12-31 23:59:59")

TEST_START  = pd.Timestamp("2022-01-01")
TEST_END    = pd.Timestamp("2023-02-20 23:59:59")

BSP_LO = 10.0
BSP_HI = 20.0

BSP_SUBBANDS = [
    (10.0, 12.0),
    (12.0, 15.0),
    (15.0, 20.0),
]

# -------------------------
# TRACK CANONICALISATION
# -------------------------
TRACK_ALIAS = {
    "cpark": "Central Park",
    "pbarr": "Perry Barr",
    "perry": "Perry Barr",
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

UK_CANON_TRACKS = {
    "Monmore", "Romford", "Hove", "Crayford", "Newcastle", "Central Park", "Harlow",
    "Sheffield", "Sunderland", "Swindon", "Perry Barr", "Nottingham", "Henlow",
    "Towcester", "Doncaster", "Kinsley", "Yarmouth", "Oxford", "Poole", "Valley",
    "Pelaw Grange", "Star Pelaw", "Peterborough", "Bvue",
    "Mullingar", "Shelbourne Park", "Suffolk Downs",
}

MENU_SPLIT_RE = re.compile(r"/\s*(.+)$")
DATE_SUFFIX_RE = re.compile(r"\s+\d{1,2}(st|nd|rd|th)\s+[A-Za-z]{3,}\s*$", flags=re.IGNORECASE)


def canonical_track(menu_hint: str) -> str:
    if not isinstance(menu_hint, str):
        return "UNKNOWN"
    m = MENU_SPLIT_RE.search(menu_hint.strip())
    rhs = m.group(1).strip() if m else menu_hint.strip()
    rhs = DATE_SUFFIX_RE.sub("", rhs).strip()
    key = re.sub(r"\s+", " ", rhs.lower())
    trk = TRACK_ALIAS.get(key)
    if trk is None:
        trk = TRACK_ALIAS.get(key.split(" ", 1)[0])
    if trk is None:
        trk = rhs.title()
    return trk


# -------------------------
# PNL + METRICS
# -------------------------
def compute_pnl(won, bsp):
    return np.where(won == 1, bsp - 1.0, -1.0)


def to_daily(df):
    d = df.groupby("event_date", as_index=False)["pnl"].sum().sort_values("event_date")
    d["equity"] = d["pnl"].cumsum()
    return d


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def summary(df):
    if df.empty:
        return {"total_pnl": 0.0, "max_dd": 0.0, "ratio": np.nan, "bets": 0}
    daily = to_daily(df)
    total = float(daily["pnl"].sum())
    dd = max_dd(daily["equity"].to_numpy())
    ratio = total / abs(dd) if dd != 0 else np.nan
    return {"total_pnl": total, "max_dd": dd, "ratio": ratio, "bets": len(df)}


# -------------------------
# LOAD DATA
# -------------------------
def load_all():
    cols = ["event_dt", "event_date", "menu_hint", "bsp", "won"]
    df = pd.read_parquet(MASTER_ROOT, columns=cols)
    df = df[df["bsp"].notna() & (df["bsp"] > 1.0)]
    df["event_dt"] = pd.to_datetime(df["event_dt"])
    df = df[(df["bsp"] >= BSP_LO) & (df["bsp"] < BSP_HI)]
    df["track"] = df["menu_hint"].apply(canonical_track)
    df = df[df["track"].isin(UK_CANON_TRACKS)]
    df["pnl"] = compute_pnl(df["won"].astype(int), df["bsp"].astype(float))
    return df


# -------------------------
# MAIN
# -------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all()

    df_train = df[(df["event_dt"] >= TRAIN_START) & (df["event_dt"] <= TRAIN_END)].copy()
    df_test  = df[(df["event_dt"] >= TEST_START)  & (df["event_dt"] <= TEST_END)].copy()

    print(f"Train bets: {len(df_train):,}")
    print(f"Test bets : {len(df_test):,}")

    # -------------------------
    # 1) Rank tracks on TRAIN
    # -------------------------
    track_rank = (
        df_train.groupby("track")["pnl"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    top3 = track_rank["track"].head(3).tolist()
    top2 = track_rank["track"].head(2).tolist()

    print("Top 3 tracks (TRAIN):", top3)
    print("Top 2 tracks (TRAIN):", top2)

    # -------------------------
    # 2) Worst BSP band on TRAIN
    # -------------------------
    band_rows = []
    for lo, hi in BSP_SUBBANDS:
        d = df_train[(df_train["bsp"] >= lo) & (df_train["bsp"] < hi)]
        band_rows.append({
            "band": f"{lo:.0f}-{hi:.0f}",
            "pnl": d["pnl"].sum(),
            "bets": len(d)
        })

    band_df = pd.DataFrame(band_rows).sort_values("pnl")
    worst_band = band_df.iloc[0]["band"]
    print("Worst band on TRAIN:", worst_band)

    # -------------------------
    # Evaluate on TEST
    # -------------------------
    results = []

    # A) TOP2 only
    dA = df_test[df_test["track"].isin(top2)]
    sA = summary(dA)
    results.append({
        "candidate": "TOP2_tracks",
        "tracks": ",".join(top2),
        "band_filter": "None",
        **sA
    })

    # B) TOP3 but exclude worst band
    lo_bad, hi_bad = map(float, worst_band.split("-"))
    dB = df_test[df_test["track"].isin(top3)]
    dB = dB[~((dB["bsp"] >= lo_bad) & (dB["bsp"] < hi_bad))]
    sB = summary(dB)
    results.append({
        "candidate": "TOP3_exclude_worst_band",
        "tracks": ",".join(top3),
        "band_filter": f"exclude_{worst_band}",
        **sB
    })

    out_df = pd.DataFrame(results).sort_values("ratio", ascending=False)
    out_path = OUT_DIR / "step16_candidate_comparison.csv"
    out_df.to_csv(out_path, index=False)

    print("\nTEST comparison:")
    print(out_df.to_string(index=False))

    print("\nSaved to:", out_path)


if __name__ == "__main__":
    main()