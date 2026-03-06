#!/usr/bin/env python3
"""
Run month downloads for a whole year (or a month range) for UK Greyhound WIN/PLACE,
saving into your /media/ben/STORAGE/Greyhound Racing structure.

Calls bfsp_download_month_greyhound.py as a subprocess for simplicity & repeatability.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--start-month", type=int, default=1)
    ap.add_argument("--end-month", type=int, default=12)
    ap.add_argument("--storage-root", type=str, default="/media/ben/STORAGE/Greyhound Racing")
    ap.add_argument("--base-url", type=str, default="https://promo.betfair.com/betfairsp/prices")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--script", type=str, default="bfsp_download_month_greyhound.py")
    args = ap.parse_args()

    if not (1 <= args.start_month <= 12 and 1 <= args.end_month <= 12 and args.start_month <= args.end_month):
        raise SystemExit("Invalid month range.")

    for m in range(args.start_month, args.end_month + 1):
        cmd = [
            sys.executable,
            args.script,
            "--year", str(args.year),
            "--month", str(m),
            "--storage-root", args.storage_root,
            "--base-url", args.base_url,
            "--retries", str(args.retries),
            "--sleep", str(args.sleep),
            "--timeout", str(args.timeout),
        ]
        if args.dry_run:
            cmd.append("--dry-run")

        print("\n" + "=" * 70)
        print("Running:", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"ERROR: month {m:02d} failed with exit code {rc}. Stopping.")
            sys.exit(rc)

    print("\nYear download complete.")


if __name__ == "__main__":
    main()
