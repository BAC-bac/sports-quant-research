from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


MASTER_ROOT = Path("data/master/bfsp/uk/greyhound/win")
OUT_DIR = Path("reports/bfsp/uk/greyhound/win/track_splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BSP_LO = 10.0
BSP_HI = 20.0
PRE_END_YEAR = 2021
MIN_BETS_PER_TRACK = 500


# --- Core metrics helpers -----------------------------------------------------

def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(np.min(dd))


def rolling_min_sum(x: np.ndarray, window: int) -> float:
    if len(x) == 0:
        return 0.0
    if len(x) < window:
        return float(np.sum(x))
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    rs = c[window:] - c[:-window]
    return float(np.min(rs)) if len(rs) else 0.0


# --- Track parsing + canonicalisation ----------------------------------------

def extract_track(menu_hint: pd.Series) -> pd.Series:
    """
    Extract track name from greyhound MENU_HINT.

    Observed format includes meeting date like:
      'Hove 25th May'
      'Romford 27th May'
    We strip the trailing ' <DD><st/nd/rd/th> <Month>' portion.

    This regex is case-insensitive and robust to Title Case.
    """
    s = menu_hint.astype(str)

    # Take substring after first '/'
    track = s.str.split("/", n=1).str[-1].str.strip()
    track = track.str.replace(r"\s+", " ", regex=True).str.strip()

    # Remove trailing ' 25th May' style suffix (case-insensitive)
    # Examples matched: "25th May", "2nd Aug", "1st Jun", "23rd Feb"
    track = track.str.replace(
        r"(?i)\s+\d{1,2}(st|nd|rd|th)\s+[a-z]{3,}$",
        "",
        regex=True
    ).str.strip()

    # Collapse whitespace again after stripping
    track = track.str.replace(r"\s+", " ", regex=True).str.strip()

    return track


def canonicalise_track(track: pd.Series) -> pd.Series:
    """
    Canonicalise common abbreviations and variants.
    """
    t = track.astype(str).str.strip()
    t = t.str.replace(r"\s+", " ", regex=True)

    tl = t.str.lower()

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

        # extras
        "monmore green": "monmore",
        "central pk": "central park",
        "perrybarr": "perry barr",
    }

    tl = tl.replace(mapping)

    return tl.str.title()


def is_uk_track(track: pd.Series) -> pd.Series:
    """
    Minimal UK filter: exclude explicit country-tagged tracks.
    """
    t = track.astype(str)
    return (
        ~t.str.contains(r"\(AUS\)", na=False) &
        ~t.str.contains(r"\(NZL\)", na=False) &
        ~t.str.contains(r"\(IRE\)", na=False) &
        ~t.str.contains(r"\(USA\)", na=False)
    )


# --- Main --------------------------------------------------------------------

def main():
    cols = ["event_dt", "menu_hint", "bsp", "pnl_1u"]
    df = pd.read_parquet(MASTER_ROOT, columns=cols)

    df = df[df["bsp"].notna() & (df["bsp"] > 1.0)]
    df = df[(df["bsp"] >= BSP_LO) & (df["bsp"] < BSP_HI)].copy()

    df["event_dt"] = pd.to_datetime(df["event_dt"])
    df["year"] = df["event_dt"].dt.year.astype(int)
    df["event_date"] = df["event_dt"].dt.date

    # Track extraction
    df["track_raw"] = extract_track(df["menu_hint"])
    df = df[is_uk_track(df["track_raw"])].copy()

    df["track"] = canonicalise_track(df["track_raw"])

    print(f"UK-only rows in {BSP_LO}-{BSP_HI} pocket: {len(df):,}")
    print(f"Unique track_raw labels       : {df['track_raw'].nunique():,}")
    print(f"Unique canonical track labels : {df['track'].nunique():,}")

    # Quick sanity: show a few raw/canonical pairs
    sample = df[["menu_hint", "track_raw", "track"]].drop_duplicates().head(12)
    print("\nSample track parsing (menu_hint -> track_raw -> track):")
    print(sample.to_string(index=False))

    top_counts = df["track"].value_counts().head(30)
    print("\nTop 30 CANONICAL tracks by bet count:")
    print(top_counts.to_string())

    def agg(segment: pd.DataFrame) -> pd.Series:
        daily = (
            segment.groupby("event_date", as_index=False)
            .agg(pnl=("pnl_1u", "sum"))
            .sort_values("event_date")
        )
        pnl = daily["pnl"].to_numpy(dtype=float)
        eq = np.cumsum(pnl)
        return pd.Series({
            "bets": int(len(segment)),
            "pnl": float(pnl.sum()),
            "max_dd": float(max_drawdown(eq)),
            "worst_20d": float(rolling_min_sum(pnl, 20)),
        })

    pre = df[df["year"] <= PRE_END_YEAR]
    post = df[df["year"] >= (PRE_END_YEAR + 1)]

    pre_g = pre.groupby("track").apply(agg).reset_index().rename(columns={
        "bets": "pre_bets", "pnl": "pre_pnl", "max_dd": "pre_max_dd", "worst_20d": "pre_worst20"
    })
    post_g = post.groupby("track").apply(agg).reset_index().rename(columns={
        "bets": "post_bets", "pnl": "post_pnl", "max_dd": "post_max_dd", "worst_20d": "post_worst20"
    })
    full_g = df.groupby("track").apply(agg).reset_index().rename(columns={
        "bets": "all_bets", "pnl": "all_pnl", "max_dd": "all_max_dd", "worst_20d": "all_worst20"
    })

    rep = full_g.merge(pre_g, on="track", how="left").merge(post_g, on="track", how="left")
    rep["survived_post"] = (rep["post_pnl"].fillna(0.0) > 0).astype(int)

    rep["post_profit_dd_ratio"] = rep["post_pnl"] / rep["post_max_dd"].abs().replace(0, np.nan)
    rep["all_profit_dd_ratio"] = rep["all_pnl"] / rep["all_max_dd"].abs().replace(0, np.nan)

    out_full = OUT_DIR / "uk_only_track_regime_split_bsp_10_20_CANON.csv"
    rep.sort_values(["post_pnl", "all_pnl"], ascending=False).to_csv(out_full, index=False)
    print(f"\nSaved full canonical report: {out_full}")

    rep_f = rep[rep["all_bets"] >= MIN_BETS_PER_TRACK].copy()
    rep_f = rep_f.sort_values(["post_profit_dd_ratio", "post_pnl"], ascending=False)

    print(f"\nTop CANONICAL UK tracks by convex-health (min {MIN_BETS_PER_TRACK} bets):")
    cols_show = [
        "track",
        "all_bets", "all_pnl", "all_max_dd", "all_profit_dd_ratio",
        "post_bets", "post_pnl", "post_max_dd", "post_profit_dd_ratio",
        "survived_post",
    ]
    print(rep_f[cols_show].head(25).to_string(index=False))


if __name__ == "__main__":
    main()