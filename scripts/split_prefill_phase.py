#!/usr/bin/env python3
"""Select a same-day phase from an already parsed Prefill CSV.

Parse the complete log first so request progress and arrival matching are preserved.
The output keeps every original column and forward ID; --since is inclusive and --until
is exclusive. See prefill/SKILL.md for the full workflow.
"""
import argparse
import csv
from datetime import time
from pathlib import Path


def clock_time(value):
    try:
        parsed = time.fromisoformat(value)
        if parsed.tzinfo is not None:
            raise ValueError("timezone is not supported")
        return parsed
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected HH:MM:SS[.ffffff]") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--since", required=True, type=clock_time)
    parser.add_argument("--until", type=clock_time)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    if args.until is not None and args.until <= args.since:
        parser.error("--until must follow --since within the same day")

    with args.input.open(newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if not fields or not {"time", "forward_idx"}.issubset(fields):
            parser.error("input must contain time and forward_idx columns")
        rows = list(reader)
    selected = [row for row in rows
                if clock_time(row["time"]) >= args.since
                and (args.until is None or clock_time(row["time"]) < args.until)]
    if not selected:
        parser.error("no rows in the selected phase; check the phase marker")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as target:
        writer = csv.DictWriter(target, fields)
        writer.writeheader()
        writer.writerows(selected)
    count = len({row["forward_idx"] for row in selected})
    print(f"{args.output}: {len(selected)}/{len(rows)} rows, {count} forwards")


if __name__ == "__main__":
    main()
