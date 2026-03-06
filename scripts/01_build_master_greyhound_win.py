from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------
# Greyhound WIN master builder
# ----------------------------
WIN_GLOB = "dwbfgreyhoundwin*.csv"

RAW_ROOT = Path("/media/ben/STORAGE/Greyhound Racing")
MASTER_ROOT = Path("data/master/bfsp/uk/greyhound/win")

DT_FORMAT = "%d-%m-%Y %H:%M"  # e.g. 31-03-2021 14:40

CANONICAL_COLS = [
    "event_id",
    "menu_hint",
    "event_name",
    "event_dt",
    "selection_id",
    "selection_name",
    "win_lose",
    "bsp",
    "ppwap",
    "morningwap",
    "ppmax",
    "ppmin",
    "ipmax",
    "ipmin",
    "morningtradedvol",
    "pptradedvol",
    "iptradedvol",
]


def discover_files() -> list[Path]:
    return sorted(RAW_ROOT.rglob(WIN_GLOB))


def normalize_file(path: Path) -> pd.DataFrame:
    """
    Read one daily CSV and normalize to canonical schema + derived columns.
    Hardened for real-world data:
      - python engine is more tolerant
      - skip malformed lines (extra commas etc.)
      - replace bad unicode
    """
    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )

    # Normalize headers
    df.columns = [c.strip().lower() for c in df.columns]

    # Schema guard
    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing} | found={list(df.columns)[:30]}"
        )

    df = df[CANONICAL_COLS].copy()

    # Parse event_dt
    df["event_dt"] = pd.to_datetime(df["event_dt"], format=DT_FORMAT, errors="coerce")
    if df["event_dt"].isna().any():
        bad = df[df["event_dt"].isna()].head(3)
        raise ValueError(f"Unparsable event_dt rows sample:\n{bad}")

    # Partitions + helpers
    df["event_date"] = df["event_dt"].dt.date
    df["year"] = df["event_dt"].dt.year.astype("int16")
    df["month"] = df["event_dt"].dt.month.astype("int8")

    # WIN outcome
    df["won"] = pd.to_numeric(df["win_lose"], errors="coerce").fillna(0).astype("int8")

    # Numeric coercions
    numeric_cols = [
        "bsp",
        "ppwap",
        "morningwap",
        "ppmax",
        "ppmin",
        "ipmax",
        "ipmin",
        "morningtradedvol",
        "pptradedvol",
        "iptradedvol",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived features
    df["implied_prob"] = 1.0 / df["bsp"]
    df["log_bsp"] = np.log(df["bsp"].where(df["bsp"] > 0))

    # 1-unit back bet PnL
    df["pnl_1u"] = np.where(df["won"] == 1, df["bsp"] - 1.0, -1.0)

    return df


def write_partitioned(df: pd.DataFrame) -> None:
    MASTER_ROOT.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)

    pq.write_to_dataset(
        table,
        root_path=str(MASTER_ROOT),
        partition_cols=["year", "month"],
    )


def main() -> None:
    files = discover_files()
    if not files:
        raise RuntimeError(f"No greyhound WIN files found under {RAW_ROOT} matching {WIN_GLOB}")

    print(f"Raw root   : {RAW_ROOT}")
    print(f"Master root: {MASTER_ROOT}")
    print(f"Found {len(files)} greyhound WIN files")

    bad_rows = []
    manifest_rows = []
    ok = 0

    for i, f in enumerate(files, 1):
        try:
            df = normalize_file(f)
            write_partitioned(df)
            ok += 1

            manifest_rows.append(
                {
                    "file": str(f),
                    "rows": int(len(df)),
                    "min_event_dt": df["event_dt"].min(),
                    "max_event_dt": df["event_dt"].max(),
                }
            )

        except Exception as e:
            bad_rows.append({"file": str(f), "error": repr(e)})
            print(f"[SKIP] {f} -> {type(e).__name__}: {e}")

        if i % 50 == 0:
            print(f"Processed {i}/{len(files)} (ok={ok}, bad={len(bad_rows)})")

    # Save reports
    MASTER_ROOT.mkdir(parents=True, exist_ok=True)

    if manifest_rows:
        manifest_path = MASTER_ROOT / "_manifest_build_master.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        print(f"Saved manifest: {manifest_path}")

    if bad_rows:
        bad_path = MASTER_ROOT / "_bad_files.csv"
        pd.DataFrame(bad_rows).to_csv(bad_path, index=False)
        print(f"Saved bad files report: {bad_path}")

    print(f"Done. ok={ok}, bad={len(bad_rows)}")


if __name__ == "__main__":
    main()