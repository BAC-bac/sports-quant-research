from pathlib import Path
import pandas as pd
import numpy as np

MASTER_ROOT = Path("data/master/bfsp/uk/place")

def load_bin_data():
    df = pd.read_parquet(MASTER_ROOT, columns=["event_date","bsp","placed"])
    df = df[(df["bsp"] >= 3.0) & (df["bsp"] < 5.0)]
    df = df[df["bsp"].notna() & (df["bsp"] > 1.0)]
    df = df.sort_values("event_date")
    return df

def compute_bet_streaks(df):
    # loss if not placed
    is_loss = (df["placed"].values == 0)
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
    return pd.Series(streaks)

def main():
    df = load_bin_data()
    streaks = compute_bet_streaks(df)

    print("Win rate:", df["placed"].mean())
    print("Total bets:", len(df))
    print("Total losing streak episodes:", len(streaks))
    print("Max losing streak (bets):", streaks.max())
    print("Median streak:", streaks.median())
    print("P90 streak:", streaks.quantile(0.9))
    print("P99 streak:", streaks.quantile(0.99))

if __name__ == "__main__":
    main()