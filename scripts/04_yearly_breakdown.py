from pathlib import Path
import pandas as pd

BIN_PATH = Path("reports/bfsp/uk/place/bin_sweeps/bsp_3_00_5_00_daily.parquet")

def main():
    daily = pd.read_parquet(BIN_PATH)

    # Add year from event_date
    daily["year"] = pd.to_datetime(daily["event_date"]).dt.year

    yearly = (
        daily.groupby("year", as_index=False)
             .agg(
                 pnl=("pnl","sum"),
                 days=("pnl","size")
             )
             .sort_values("year")
    )

    print(yearly.to_string(index=False))

    print("\nCumulative by year:")
    yearly["cum"] = yearly["pnl"].cumsum()
    print(yearly.to_string(index=False))

if __name__ == "__main__":
    main()