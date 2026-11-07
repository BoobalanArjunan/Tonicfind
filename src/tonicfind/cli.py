from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import __version__
from .check import check_file
from .labels import LabelError, from_filenames, read_csv
from .report import text, write_csv


def _job(args):
    it, min_clarity, support_gap = args
    return check_file(*it, min_clarity=min_clarity, support_gap=support_gap)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tonicfind", description="Check that the key labels on your music match the audio.")
    ap.add_argument("--version", action="version", version=f"tonicfind {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check labels from a CSV, or from a folder of files named like song_key-Am.wav")
    c.add_argument("source", type=Path, help="labels.csv (columns: file, key) or a folder")
    c.add_argument("--audio-dir", type=Path, help="folder the CSV's file names are relative to (default: the CSV's folder)")
    c.add_argument("--out", type=Path, help="write every verdict to this CSV")
    c.add_argument("--show-pass", action="store_true", help="also list files that passed")
    c.add_argument("--jobs", type=int, default=1, help="files analysed in parallel (default 1)")
    c.add_argument("--min-clarity", type=float, default=0.55, help="minimum correlation clarity threshold (default: 0.55)")
    c.add_argument("--support-gap", type=float, default=0.05, help="margin to consider label supported by audio (default: 0.05)")
    a = ap.parse_args(argv)

    try:
        if a.source.is_dir():
            items, problems = from_filenames(a.source)
        elif a.source.is_file():
            items, problems = read_csv(a.source, a.audio_dir)
        else:
            print(f"error: {a.source} not found", file=sys.stderr)
            return 2
    except LabelError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not items:
        print("error: nothing to check" + (f" ({len(problems)} label problems, e.g. {problems[0]})" if problems else ""), file=sys.stderr)
        return 2
    if a.jobs > 1:
        job_args = [(it, a.min_clarity, a.support_gap) for it in items]
        with ProcessPoolExecutor(a.jobs) as ex:
            vs = list(ex.map(_job, job_args, chunksize=1))
    else:
        vs = [check_file(*it, min_clarity=a.min_clarity, support_gap=a.support_gap) for it in items]
    print(text(vs, problems, a.show_pass))
    if a.out:
        write_csv(a.out, vs)
    return 1 if any(v.level == "FAIL" for v in vs) else 0


if __name__ == "__main__":
    sys.exit(main())
