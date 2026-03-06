import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.DataFrame:
    peak = equity.cummax()
    dd = equity - peak
    dd_pct = np.where(peak != 0, dd / peak, 0.0)
    return pd.DataFrame({"equity": equity, "peak": peak, "dd": dd, "dd_pct": dd_pct})


def max_drawdown(dd_df: pd.DataFrame) -> dict:
    # Most negative dd
    trough_idx = dd_df["dd"].idxmin()
    trough_dd = float(dd_df.loc[trough_idx, "dd"])
    trough_dd_pct = float(dd_df.loc[trough_idx, "dd_pct"])

    peak_idx = dd_df.loc[:trough_idx, "equity"].idxmax()
    peak_equity = float(dd_df.loc[peak_idx, "equity"])
    trough_equity = float(dd_df.loc[trough_idx, "equity"])

    return {
        "peak_idx": int(peak_idx),
        "trough_idx": int(trough_idx),
        "peak_equity": peak_equity,
        "trough_equity": trough_equity,
        "max_dd": trough_dd,
        "max_dd_pct": trough_dd_pct,
    }


def drawdown_events(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Create a table of drawdown episodes:
    start (peak date), trough date, recovery date, dd amount, dd pct, duration.
    """
    eq = daily["equity"].astype(float).reset_index(drop=True)
    dd_df = drawdown_series(eq)

    in_dd = dd_df["dd"] < 0
    events = []

    i = 0
    while i < len(dd_df):
        if not in_dd.iloc[i]:
            i += 1
            continue

        # drawdown started at first negative after a peak
        start = i - 1 if i > 0 else i
        # move until drawdown ends (dd returns to 0)
        j = i
        trough = i
        while j < len(dd_df) and in_dd.iloc[j]:
            if dd_df["dd"].iloc[j] < dd_df["dd"].iloc[trough]:
                trough = j
            j += 1
        recovery = j if j < len(dd_df) else None

        peak_eq = float(dd_df["peak"].iloc[i])
        dd_amt = float(dd_df["dd"].iloc[trough])
        dd_pct = float(dd_df["dd_pct"].iloc[trough])

        events.append({
            "start_idx": start,
            "trough_idx": trough,
            "recovery_idx": recovery,
            "peak_equity": peak_eq,
            "dd_amount": dd_amt,
            "dd_pct": dd_pct,
            "duration_days": (recovery - start) if recovery is not None else (len(dd_df) - start),
        })

        i = j if j is not None else len(dd_df)

    return pd.DataFrame(events)


def losing_streaks_from_sequence(is_loss: np.ndarray) -> pd.Series:
    streaks = []
    cur = 0
    for v in is_loss:
        if v:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
                cur = 0
    if cur > 0:
        streaks.append(cur)
    return pd.Series(streaks, dtype="int64")


def losing_streaks_daily(daily: pd.DataFrame) -> pd.Series:
    # loss day = pnl < 0
    is_loss = (daily["pnl"].astype(float).values < 0)
    return losing_streaks_from_sequence(is_loss)


def tails_daily(daily: pd.DataFrame, windows=(1, 5, 20)) -> dict:
    out = {}
    pnl = daily["pnl"].astype(float).reset_index(drop=True)

    for w in windows:
        roll = pnl.rolling(w).sum() if w > 1 else pnl
        out[f"worst_{w}d"] = float(roll.min())
        out[f"p1_{w}d"] = float(np.nanpercentile(roll, 1))
        out[f"p5_{w}d"] = float(np.nanpercentile(roll, 5))
    return out


def time_to_recovery_stats(dd_events: pd.DataFrame) -> dict:
    if dd_events.empty:
        return {"n_events": 0}

    durations = dd_events["duration_days"].astype(int)
    return {
        "n_events": int(len(durations)),
        "median_days": float(durations.median()),
        "p90_days": float(np.percentile(durations, 90)),
        "max_days": float(durations.max()),
    }