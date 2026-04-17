"""
archive_logs.py — move rows older than N days from live CSVs into *_archive.csv.

The dashboard parses the full CSV on every page load; once each log exceeds a
few thousand rows the embedded JS arrays get heavy. This script keeps the live
log lean while preserving history in a sibling archive file.

Usage:
  py archive_logs.py               # default: archive rows older than 90 days
  py archive_logs.py --days 60     # custom window
  py archive_logs.py --dry-run     # report what would move, don't write
"""
import argparse
import csv
import datetime as dt
import os

FILES = ["coinbase_trade_log.csv", "alpaca_trade_log.csv"]


def parse_ts(s: str):
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def archive(path: str, cutoff: dt.datetime, dry_run: bool) -> tuple[int, int]:
    if not os.path.isfile(path):
        return 0, 0
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return 0, 0
    header, data = rows[0], rows[1:]

    keep, archive_rows = [], []
    for r in data:
        ts = parse_ts(r[0]) if r else None
        (archive_rows if ts and ts < cutoff else keep).append(r)

    if not archive_rows:
        return 0, len(keep)

    if not dry_run:
        arch_path = path.replace(".csv", "_archive.csv")
        arch_exists = os.path.isfile(arch_path)
        with open(arch_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not arch_exists:
                w.writerow(header)
            w.writerows(archive_rows)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(keep)
    return len(archive_rows), len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cutoff = dt.datetime.now() - dt.timedelta(days=args.days)
    print(f"Archiving rows older than {cutoff:%Y-%m-%d} "
          f"({'dry-run' if args.dry_run else 'writing'})\n")
    for path in FILES:
        moved, kept = archive(path, cutoff, args.dry_run)
        print(f"  {path}: {moved} archived, {kept} kept")


if __name__ == "__main__":
    main()
