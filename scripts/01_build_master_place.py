import os
import re
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
import numpy as np

PLACE_GLOB = "dwbfpricesukplace*.csv"

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

DT_FORMAT = "%d-%m-%Y %H:%M"  # matches your file e.g. 23-02-2026 15:30


@dataclass(frozen=True)
class Paths:
    raw_root: Path
    master_root: Path


def load_paths() -> Paths:
    cfg = yaml.safe_load(Path("config/paths.yaml").read_text())
    raw_root = Path(cfg["raw"]["bfsp_horse_racing_root"])
    master_root = Path(cfg["master"]["root"]) / cfg["master"]["dataset"]
    return Paths(raw_root=raw_root, master_root=master_root)


def discover_place_files(raw_root: Path) -> list[Path]:
    # Your structure is year/Results/<Month Year>/dwbfpricesukplaceDDMMYYYY.csv
    # We'll just scan all years for robustness.
    return sorted(raw_root.rglob(PLACE_GLOB))


def normalize_one_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ✅ Make headers case-insensitive: lowercase + strip
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        # Helpful debugging: show first few columns we DID find
        raise ValueError(f"{path} missing columns: {missing} | found={list(df.columns)[:30]}")

    df = df[CANONICAL_COLS].copy()

    # Parse datetime and derive partitions
    df["event_dt"] = pd.to_datetime(df["event_dt"], format=DT_FORMAT, errors="coerce")
    if df["event_dt"].isna().any():
        bad = df[df["event_dt"].isna()].head(3)
        raise ValueError(f"{path} has unparsable event_dt rows, sample:\n{bad}")

    df["event_date"] = df["event_dt"].dt.date
    df["year"] = df["event_dt"].dt.year.astype("int16")
    df["month"] = df["event_dt"].dt.month.astype("int8")

    # Canonical outcome naming
    df["placed"] = pd.to_numeric(df["win_lose"], errors="coerce").fillna(0).astype("int8")

    # Canonical numeric types
    numeric_cols = [
        "bsp", "ppwap", "morningwap", "ppmax", "ppmin", "ipmax", "ipmin",
        "morningtradedvol", "pptradedvol", "iptradedvol"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived features
    df["implied_prob"] = 1.0 / df["bsp"]
    df["log_bsp"] = np.log(df["bsp"].where(df["bsp"] > 0))

    # Ensure ids are strings
    df["event_id"] = df["event_id"].astype(str)
    df["selection_id"] = df["selection_id"].astype(str)

    return df


def append_partitioned_parquet(df: pd.DataFrame, master_root: Path) -> None:
    """
    Writes to partitioned dataset:
      master_root/year=YYYY/month=MM/part-*.parquet
    """
    master_root.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)

    pq.write_to_dataset(
        table,
        root_path=str(master_root),
        partition_cols=["year", "month"],
        existing_data_behavior="overwrite_or_ignore",
    )


def build_master():
    paths = load_paths()

    files = discover_place_files(paths.raw_root)
    if not files:
        raise RuntimeError(f"No files found under {paths.raw_root}")

    print(f"Found {len(files)} place files")

    # Manifest for traceability
    manifest_rows = []

    for i, f in enumerate(files, 1):
        df = normalize_one_file(f)
        append_partitioned_parquet(df, paths.master_root)

        manifest_rows.append({
            "file": str(f),
            "rows": len(df),
            "min_event_dt": df["event_dt"].min(),
            "max_event_dt": df["event_dt"].max(),
        })

        if i % 50 == 0:
            print(f"Processed {i}/{len(files)}")

    manifest = pd.DataFrame(manifest_rows)
    out_manifest = paths.master_root / "_manifest_build_master.csv"
    manifest.to_csv(out_manifest, index=False)
    print(f"Saved manifest: {out_manifest}")


if __name__ == "__main__":
    build_master()