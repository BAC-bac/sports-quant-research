from pathlib import Path
import pandas as pd
import numpy as np

MASTER_ROOT = Path("data/master/bfsp/uk/place")

def main():
    df = pd.read_parquet(MASTER_ROOT, columns=["bsp","placed"])
    df = df[(df["bsp"] >= 3.0) & (df["bsp"] < 5.0)]
    df = df[df["bsp"].notna() & (df["bsp"] > 1.0)]

    win_rate = df["placed"].mean()
    implied = (1.0 / df["bsp"]).mean()

    pnl = np.where(df["placed"] == 1, df["bsp"] - 1.0, -1.0)

    ev = pnl.mean()
    std = pnl.std()
    sharpe = ev / std * np.sqrt(len(pnl))

    print("Win rate:", win_rate)
    print("Mean implied probability:", implied)
    print("Edge (win - implied):", win_rate - implied)
    print("EV per bet:", ev)
    print("Std per bet:", std)
    print("Sharpe (naive):", sharpe)

if __name__ == "__main__":
    main()