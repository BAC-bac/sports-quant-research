#!/usr/bin/env python3
"""
Download BFSP daily CSV files for a given month for UK Greyhound WIN/PLACE.

Expected filenames:
- dwbfgreyhoundwinDDMMYYYY.csv
- dwbfgreyhoundplaceDDMMYYYY.csv

Saves into:
  /media/ben/STORAGE/Greyhound Racing/{YYYY}/results/{MonthName} {YYYY}/

Features:
- skip if already exists
- retries
- dry-run mode
- simple success/fail summary
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import time
from pathlib import Path
from typing import Iterable, List, Tuple
import urllib.request
import urllib.error


DEFAULT_BASE_URL = "https://promo.betfair.com/betfairsp/prices"
DEFAULT_PREFIXES = ["dwbfgreyhoundwin", "dwbfgreyhoundplace"]


def month_date_range(year: int, month: int) -> List[dt.date]:
    last_day = calendar.monthrange(year, month)[1]
    return [dt.date(year, month, d) for d in range(1, last_day + 1)]


def ddmmyyyy(d: dt.date) -> str:
    return d.strftime("%d%m%Y")


def month_folder_name(year: int, month: int) -> str:
    return f"{calendar.month_name[month]} {year}"


def build_url(base_url: str, prefix: str, d: dt.date) -> str:
    # e.g. https://promo.betfair.com/betfairsp/prices/dwbfgreyhoundwin16082025.csv
    return f"{base_url.rstrip('/')}/{prefix}{ddmmyyyy(d)}.csv"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_file(url: str, out_path: Path, timeout: int = 45) -> Tuple[bool, str]:
    """
    Returns (ok, message).
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) BFSPDownloader/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        out_path.write_bytes(data)
        # sanity: if the response is suspiciously small, you might want to flag it
        if out_path.stat().st_size < 1000:
            return False, f"Downloaded but file too small ({out_path.stat().st_size} bytes)"
        return True, "OK"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"URL error: {e.reason}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True, help="1-12")
    ap.add_argument("--storage-root", type=str, default="/media/ben/STORAGE/Greyhound Racing")
    ap.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    ap.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        help="Override/add prefix (can be used multiple times). If omitted, uses WIN+PLACE defaults.",
    )
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.35, help="seconds between requests")
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    year = args.year
    month = args.month
    if month < 1 or month > 12:
        raise SystemExit("month must be 1..12")

    prefixes = args.prefixes if args.prefixes else DEFAULT_PREFIXES

    storage_root = Path(args.storage_root)
    out_dir = storage_root / str(year) / "results" / month_folder_name(year, month)
    ensure_dir(out_dir)

    dates = month_date_range(year, month)

    print(f"BASE URL : {args.base_url}")
    print(f"YEAR/MON : {year}-{month:02d}")
    print(f"OUT DIR  : {out_dir}")
    print(f"PREFIXES : {prefixes}")
    print(f"DAYS     : {len(dates)}")
    print(f"DRY RUN  : {args.dry_run}")
    print("")

    ok_count = 0
    skip_count = 0
    fail_count = 0
    failures: List[str] = []

    for d in dates:
        for prefix in prefixes:
            filename = f"{prefix}{ddmmyyyy(d)}.csv"
            out_path = out_dir / filename
            url = build_url(args.base_url, prefix, d)

            if out_path.exists() and out_path.stat().st_size > 1000:
                skip_count += 1
                continue

            if args.dry_run:
                print(f"[DRY] {url} -> {out_path}")
                continue

            success = False
            last_msg = ""
            for attempt in range(1, args.retries + 2):  # retries + first try
                ok, msg = download_file(url, out_path, timeout=args.timeout)
                last_msg = msg
                if ok:
                    success = True
                    break
                # cleanup partial file
                try:
                    if out_path.exists():
                        out_path.unlink()
                except Exception:
                    pass
                time.sleep(args.sleep + 0.2)

            if success:
                ok_count += 1
            else:
                fail_count += 1
                failures.append(f"{filename} <- {url} ({last_msg})")

            time.sleep(args.sleep)

    print("\n=== MONTH DOWNLOAD SUMMARY ===")
    print(f"OK   : {ok_count}")
    print(f"SKIP : {skip_count}")
    print(f"FAIL : {fail_count}")

    if failures:
        print("\nFailures (first 25):")
        for x in failures[:25]:
            print(" -", x)


if __name__ == "__main__":
    main()
