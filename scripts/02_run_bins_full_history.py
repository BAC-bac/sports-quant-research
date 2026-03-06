from pathlib import Path
import pandas as pd
import numpy as np
import yaml

MASTER_ROOT = Path("data/master/bfsp/uk/place")
OUT_ROOT = Path("reports/bfsp/uk/place/bin_sweeps")


def load_bins():
    cfg = yaml.safe_load(Path("config/bins_place_bsp.yaml").read_text())
    stake = float(cfg.get("stake", 1.0))
    bins = cfg["bins"]
    return stake, bins


def load_master(columns=None) -> pd.DataFrame:
    # Load partitioned parquet dataset (column prune)
    df = pd.read_parquet(MASTER_ROOT, columns=columns)
    return df


def compute_runner_pnl(df: pd.DataFrame, stake: float) -> pd.Series:
    # BACK place: profit = stake*(bsp-1) if placed else -stake
    # placed is 0/1
    placed = df["placed"].astype("int8")
    bsp = df["bsp"].astype("float64")
    pnl = np.where(placed == 1, stake * (bsp - 1.0), -stake)
    return pd.Series(pnl, index=df.index, name="pnl")


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    # aggregate to daily pnl
    daily = (
        df.groupby("event_date", as_index=False)
          .agg(
              pnl=("pnl", "sum"),
              n_bets=("pnl", "size"),
              avg_bsp=("bsp", "mean"),
          )
          .sort_values("event_date")
    )
    daily["equity"] = daily["pnl"].cumsum()
    return daily


def run_bin(df_base: pd.DataFrame, stake: float, bin_spec: dict) -> pd.DataFrame:
    lo = float(bin_spec["bsp_min"])
    hi = float(bin_spec["bsp_max"])

    df = df_base[(df_base["bsp"] >= lo) & (df_base["bsp"] < hi)].copy()
    if df.empty:
        return pd.DataFrame(columns=["event_date", "pnl", "n_bets", "avg_bsp", "equity"])

    df["pnl"] = compute_runner_pnl(df, stake)
    daily = to_daily(df)
    return daily


def main():
    stake, bins = load_bins()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load only what we need for performance
    cols = ["event_date", "bsp", "placed"]
    df = load_master(columns=cols)

    # filter to valid bets (engine rule: bsp must exist and be > 1.0)
    df = df[df["bsp"].notna() & (df["bsp"] > 1.0)].copy()

    # Ensure event_date is sortable / consistent
    # (it should already be python date objects from build)
    print(f"Loaded MASTER rows: {len(df):,}")

    summary_rows = []

    for b in bins:
        name = b["name"]
        daily = run_bin(df, stake, b)
        out_path = OUT_ROOT / f"{name}_daily.parquet"
        daily.to_parquet(out_path, index=False)

        total_pnl = float(daily["pnl"].sum()) if not daily.empty else 0.0
        max_equity = float(daily["equity"].max()) if not daily.empty else 0.0
        min_equity = float(daily["equity"].min()) if not daily.empty else 0.0
        n_bets = int(daily["n_bets"].sum()) if not daily.empty else 0
        n_days = int(len(daily))

        summary_rows.append({
            "bin": name,
            "bsp_min": b["bsp_min"],
            "bsp_max": b["bsp_max"],
            "stake": stake,
            "total_pnl": total_pnl,
            "n_bets": n_bets,
            "n_days": n_days,
            "equity_max": max_equity,
            "equity_min": min_equity,
            "daily_file": str(out_path),
        })

        print(f"Saved {name}: days={n_days} bets={n_bets} pnl={total_pnl:.2f}")

    summary = pd.DataFrame(summary_rows).sort_values("bsp_min")
    summary_path = OUT_ROOT / "_SUMMARY_bins.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()