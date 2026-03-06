from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


MASTER_ROOT = Path("data/master/bfsp/uk/greyhound/win")
TRACK_REPORT = Path("reports/bfsp/uk/greyhound/win/track_splits/uk_only_track_regime_split_bsp_10_20_CANON.csv")
OUT_DIR = Path("reports/bfsp/uk/greyhound/win/track_baskets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BSP_LO = 10.0
BSP_HI = 20.0
PRE_END_YEAR = 2021

# Basket sizes to test
TOP_NS = [3, 5, 10, 15]


# ---------- helpers ----------

def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(np.min(dd))


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("event_date", as_index=False)
        .agg(
            pnl=("pnl_1u", "sum"),
            n_bets=("pnl_1u", "size"),
            avg_bsp=("bsp", "mean"),
        )
        .sort_values("event_date")
    )
    daily["equity"] = daily["pnl"].cumsum()
    return daily


def summarize_daily(daily: pd.DataFrame, name: str) -> dict:
    pnl = float(daily["pnl"].sum()) if len(daily) else 0.0
    eq = daily["equity"].to_numpy(dtype=float) if len(daily) else np.array([], dtype=float)
    mdd = max_drawdown(eq)
    n_days = int(len(daily))
    n_bets = int(daily["n_bets"].sum()) if len(daily) else 0
    return {
        "basket": name,
        "total_pnl": pnl,
        "n_days": n_days,
        "n_bets": n_bets,
        "max_dd": mdd,
        "profit_dd_ratio": (pnl / abs(mdd)) if mdd != 0 else np.nan,
    }


def load_master_uk_10_20() -> pd.DataFrame:
    cols = ["event_dt", "event_date", "menu_hint", "bsp", "pnl_1u"]
    df = pd.read_parquet(MASTER_ROOT, columns=cols)

    df = df[df["bsp"].notna() & (df["bsp"] > 1.0)]
    df = df[(df["bsp"] >= BSP_LO) & (df["bsp"] < BSP_HI)].copy()

    # Minimal UK filter (exclude explicit tags)
    df["menu_hint"] = df["menu_hint"].astype(str)
    df = df[
        ~df["menu_hint"].str.contains(r"\(AUS\)|\(NZL\)|\(IRE\)|\(USA\)", na=False, regex=True)
    ].copy()

    df["event_dt"] = pd.to_datetime(df["event_dt"])
    df["year"] = df["event_dt"].dt.year.astype(int)

    return df


def attach_track(df: pd.DataFrame) -> pd.DataFrame:
    # Reuse the same extraction/canonicalisation logic used in script 09.
    # We duplicate it here to keep the basket builder self-contained.

    s = df["menu_hint"].astype(str)
    track = s.str.split("/", n=1).str[-1].str.strip()
    track = track.str.replace(r"\s+", " ", regex=True).str.strip()
    track = track.str.replace(r"(?i)\s+\d{1,2}(st|nd|rd|th)\s+[a-z]{3,}$", "", regex=True).str.strip()
    track = track.str.replace(r"\s+", " ", regex=True).str.strip()

    tl = track.str.lower()

    mapping = {
        "romfd": "romford",
        "monm": "monmore",
        "newc": "newcastle",
        "sheff": "sheffield",
        "crayfd": "crayford",
        "cpark": "central park",
        "pbarr": "perry barr",
        "nott": "nottingham",
        "sund": "sunderland",
        "swin": "swindon",
        "harl": "harlow",
        "henl": "henlow",
        "monmore green": "monmore",
        "central pk": "central park",
        "perrybarr": "perry barr",
    }
    tl = tl.replace(mapping)

    df["track"] = tl.str.title()
    return df


# ---------- main ----------

def main():
    rep = pd.read_csv(TRACK_REPORT)

    # Candidate lists
    rep_sorted_ratio = rep.sort_values("post_profit_dd_ratio", ascending=False).copy()
    rep_sorted_pnl = rep.sort_values("post_pnl", ascending=False).copy()

    # Only consider tracks with enough bets overall, and enough post bets to be meaningful
    rep_sorted_ratio = rep_sorted_ratio[(rep_sorted_ratio["all_bets"] >= 500) & (rep_sorted_ratio["post_bets"] >= 500)]
    rep_sorted_pnl = rep_sorted_pnl[(rep_sorted_pnl["all_bets"] >= 500) & (rep_sorted_pnl["post_bets"] >= 500)]

    df = load_master_uk_10_20()
    df = attach_track(df)

    print(f"Loaded UK 10–20 rows: {len(df):,}")
    print(f"Unique tracks: {df['track'].nunique():,}")

    # Benchmark: all UK 10–20
    baskets = []

    daily_all = to_daily(df)
    out_all = OUT_DIR / "basket_ALL_UK_10_20_daily.parquet"
    daily_all.to_parquet(out_all, index=False)
    baskets.append(summarize_daily(daily_all, "ALL_UK_10_20"))

    # Build Top-N baskets by two ranking methods
    for N in TOP_NS:
        top_ratio = rep_sorted_ratio["track"].head(N).tolist()
        top_pnl = rep_sorted_pnl["track"].head(N).tolist()

        d1 = df[df["track"].isin(top_ratio)].copy()
        daily1 = to_daily(d1)
        out1 = OUT_DIR / f"basket_TOP{N}_BY_POST_RATIO_daily.parquet"
        daily1.to_parquet(out1, index=False)
        baskets.append(summarize_daily(daily1, f"TOP{N}_BY_POST_RATIO"))

        d2 = df[df["track"].isin(top_pnl)].copy()
        daily2 = to_daily(d2)
        out2 = OUT_DIR / f"basket_TOP{N}_BY_POST_PNL_daily.parquet"
        daily2.to_parquet(out2, index=False)
        baskets.append(summarize_daily(daily2, f"TOP{N}_BY_POST_PNL"))

    summary = pd.DataFrame(baskets).sort_values("profit_dd_ratio", ascending=False)
    out_sum = OUT_DIR / "_SUMMARY_track_baskets.csv"
    summary.to_csv(out_sum, index=False)

    print(f"\nSaved basket summary: {out_sum}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()