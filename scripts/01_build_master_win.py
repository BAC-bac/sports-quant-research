import re
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# ----------------------------
# WIN master builder settings
# ----------------------------
WIN_GLOB = "dwbfpricesukwin*.csv"

# Canonical schema we expect from the downloader
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

DT_FORMAT = "%d-%m-%Y %H:%M"  # e.g. 31-03-2021 14:40


@dataclass(frozen=True)
class Paths:
    raw_root: Path
    master_root: Path


def load_paths() -> Paths:
    """
    Reads config/paths.yaml.

    Expected minimal shape (example):
      raw:
        bfsp_horse_racing_root: "/media/ben/STORAGE/Horse Racing"
      master:
        root: "data/master"
        datasets:
          place: "bfsp/uk/place"
          win:   "bfsp/uk/win"

    If your paths.yaml is older and only has:
      master:
        root: "data/master"
        dataset: "bfsp/uk/place"

    then update it to the datasets form above (recommended).
    """
    cfg = yaml.safe_load(Path("config/paths.yaml").read_text())

    raw_root = Path(cfg["raw"]["bfsp_horse_racing_root"])

    master_cfg = cfg.get("master", {})
    master_root_base = Path(master_cfg.get("root", "data/master"))

    # Prefer master.datasets.win if present
    datasets = master_cfg.get("datasets", {})
    if isinstance(datasets, dict) and "win" in datasets:
        master_root = master_root_base / datasets["win"]
    else:
        # Fallback: if only "dataset" exists, we force win by replacing trailing 'place' with 'win'
        # (This prevents accidental writing into the place dataset.)
        ds = master_cfg.get("dataset", "bfsp/uk/win")
        ds = re.sub(r"/place$", "/win", ds)
        master_root = master_root_base / ds

    return Paths(raw_root=raw_root, master_root=master_root)


def discover_win_files(raw_root: Path) -> list[Path]:
    # Your structure: /media/.../YYYY/Results/<Month YYYY>/dwbfpricesukwinDDMMYYYY.csv
    # We scan everything robustly.
    return sorted(raw_root.rglob(WIN_GLOB))


def normalize_one_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize headers: lowercase + strip
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} missing columns: {missing} | found={list(df.columns)[:30]}"
        )

    df = df[CANONICAL_COLS].copy()

    # Parse event_dt
    df["event_dt"] = pd.to_datetime(df["event_dt"], format=DT_FORMAT, errors="coerce")
    if df["event_dt"].isna().any():
        bad = df[df["event_dt"].isna()].head(3)
        raise ValueError(f"{path} has unparsable event_dt rows, sample:\n{bad}")

    # Partitions + helpers
    df["event_date"] = df["event_dt"].dt.date
    df["year"] = df["event_dt"].dt.year.astype("int16")
    df["month"] = df["event_dt"].dt.month.astype("int8")

    # WIN outcome naming
    # win_lose should be 1 for win, 0 for loss (sometimes strings/float -> coerce)
    df["won"] = pd.to_numeric(df["win_lose"], errors="coerce").fillna(0).astype("int8")

    # Canonical numeric columns
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

    # Derived features (useful later for modelling/diagnostics)
    df["implied_prob"] = 1.0 / df["bsp"]
    df["log_bsp"] = np.log(df["bsp"].where(df["bsp"] > 0))

    # Canonical 1-unit bet PnL for WIN markets
    # If won: profit = (bsp - 1); else: -1
    df["pnl_1u"] = np.where(df["won"] == 1, df["bsp"] - 1.0, -1.0)

    # Ensure ids are strings for joins/grouping stability
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
        # Note: safest approach is to build into a fresh folder. If rerunning, delete win master first.
        existing_data_behavior="overwrite_or_ignore",
    )


def build_master() -> None:
    paths = load_paths()

    files = discover_win_files(paths.raw_root)
    if not files:
        raise RuntimeError(f"No WIN files found under {paths.raw_root} matching {WIN_GLOB}")

    print(f"Raw root   : {paths.raw_root}")
    print(f"Master root: {paths.master_root}")
    print(f"Found {len(files)} WIN files")

    manifest_rows = []

    for i, f in enumerate(files, 1):
        df = normalize_one_file(f)
        append_partitioned_parquet(df, paths.master_root)

        manifest_rows.append(
            {
                "file": str(f),
                "rows": len(df),
                "min_event_dt": df["event_dt"].min(),
                "max_event_dt": df["event_dt"].max(),
            }
        )

        if i % 50 == 0:
            print(f"Processed {i}/{len(files)}")

    manifest = pd.DataFrame(manifest_rows)
    out_manifest = paths.master_root / "_manifest_build_master.csv"
    manifest.to_csv(out_manifest, index=False)
    print(f"Saved manifest: {out_manifest}")


if __name__ == "__main__":
    build_master()