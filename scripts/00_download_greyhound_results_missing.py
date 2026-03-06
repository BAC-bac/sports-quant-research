# scripts/00_download_greyhound_results_missing.py
"""
Download missing UK Greyhound Racing results files into:
  /media/ben/STORAGE/Greyhound Racing/YYYY/results/<Month YYYY>/

This is a "wrapper" script that:
- detects latest available date already stored
- builds the missing date range up to yesterday (Europe/London)
- creates year/month folders
- calls YOUR existing downloader for each day

You only need to implement download_one_day() to match your existing horse script method.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import date, datetime, timedelta
import re
import subprocess
import sys
import time

# -----------------------------
# CONFIG
# -----------------------------
STORAGE_ROOT = Path("/media/ben/STORAGE/Greyhound Racing")

# Folder naming convention you described:
# /media/ben/STORAGE/Greyhound Racing/2025/results/August 2025/
RESULTS_SUBDIR = "results"

# Safety: don't spam. Add a small pause between days if needed.
SLEEP_SECONDS_BETWEEN_DAYS = 0.3

# If your downloader can do retries, you can keep this low.
MAX_RETRIES_PER_DAY = 2


# -----------------------------
# Helpers: month folder naming
# -----------------------------
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

def month_folder_name(d: date) -> str:
    return f"{MONTH_NAMES[d.month]} {d.year}"

def year_results_dir(d: date) -> Path:
    return STORAGE_ROOT / str(d.year) / RESULTS_SUBDIR / month_folder_name(d)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Detect latest stored date
# -----------------------------
_DATE_RE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")  # matches 2025-08-16 or 20250816

def find_latest_date_in_storage(root: Path) -> date | None:
    """
    Best-effort scan: looks for date-like patterns in filenames under:
      root/YYYY/results/<Month YYYY>/*
    Returns latest date found, or None if none detected.
    """
    latest: date | None = None

    if not root.exists():
        return None

    # Only scan results folders (fast + relevant)
    for year_dir in sorted(root.glob("[0-9]" * 4)):
        res_dir = year_dir / RESULTS_SUBDIR
        if not res_dir.exists():
            continue

        for month_dir in res_dir.iterdir():
            if not month_dir.is_dir():
                continue

            for fp in month_dir.glob("*"):
                if not fp.is_file():
                    continue

                m = _DATE_RE.search(fp.name)
                if not m:
                    continue
                y, mo, da = map(int, m.groups())
                try:
                    d = date(y, mo, da)
                except ValueError:
                    continue

                if (latest is None) or (d > latest):
                    latest = d

    return latest


# -----------------------------
# Core: you implement this ONE function
# -----------------------------
def download_one_day(target_day: date, out_dir: Path) -> None:
    """
    Implement the actual download call here.

    You have 2 good options:

    OPTION A (recommended): call your existing horse-style Python downloader script:
        subprocess.run([sys.executable, "path/to/horse_downloader.py",
                        "--market", "ukgreyhoundwin",
                        "--date", target_day.isoformat(),
                        "--out", str(out_dir)], check=True)

    OPTION B: call your existing shell script / CLI:
        subprocess.run(["/path/to/bulk_download_one_day.sh",
                        "ukgreyhoundwin", target_day.isoformat(), str(out_dir)], check=True)

    For now, this function raises so you don't accidentally run a no-op.
    """
    raise NotImplementedError(
        "Edit download_one_day() to call your existing downloader (see docstring)."
    )


# -----------------------------
# Runner
# -----------------------------
def daterange(start: date, end: date):
    """Inclusive date range."""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

def main():
    # Yesterday (UK time) – we’ll treat "today" as not complete yet.
    # If you *do* want to include today, change to date.today()
    end_day = date.today() - timedelta(days=1)

    latest = find_latest_date_in_storage(STORAGE_ROOT)
    if latest is None:
        print("No existing dates detected in storage.")
        print("Set a manual start date inside the script or add CLI parsing if you want.")
        return

    start_day = latest + timedelta(days=1)

    if start_day > end_day:
        print(f"Storage is already up to date (latest={latest.isoformat()}, yesterday={end_day.isoformat()}).")
        return

    print("=== GREYHOUND RESULTS DOWNLOAD (missing range) ===")
    print(f"Storage root : {STORAGE_ROOT}")
    print(f"Latest found : {latest.isoformat()}")
    print(f"Download from: {start_day.isoformat()} -> {end_day.isoformat()}")

    failures = []

    for d in daterange(start_day, end_day):
        out_dir = year_results_dir(d)
        ensure_dir(out_dir)

        ok = False
        for attempt in range(1, MAX_RETRIES_PER_DAY + 1):
            try:
                print(f"[{d.isoformat()}] attempt {attempt} -> {out_dir}")
                download_one_day(d, out_dir)
                ok = True
                break
            except NotImplementedError as e:
                # Make this loud and stop immediately
                print(str(e))
                return
            except subprocess.CalledProcessError as e:
                print(f"  ERROR: downloader returned non-zero exit code: {e}")
                time.sleep(0.8)
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                time.sleep(0.8)

        if not ok:
            failures.append(d.isoformat())

        time.sleep(SLEEP_SECONDS_BETWEEN_DAYS)

    print("\n=== COMPLETE ===")
    if failures:
        print(f"Failed days ({len(failures)}):")
        for x in failures:
            print(" -", x)
    else:
        print("All days downloaded successfully ✅")

if __name__ == "__main__":
    main()