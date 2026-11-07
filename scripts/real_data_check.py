"""Real-data check for tonicfind. Needs real songs WITH trusted key labels (for example the GiantSteps Key dataset, or
GTZAN with its key annotations), as a folder of audio plus a labels CSV (file,key).

    python scripts/real_data_check.py --labels labels.csv --audio-dir songs/ --out real_check

It answers:
  1. FALSE ALARMS: with the TRUE labels, how many songs does tonicfind call WRONG (FAIL)? Should be a few percent at most.
  2. HOW MANY WRONG LABELS IT CATCHES: it swaps each label for a different key that is NOT a relative / parallel / fifth
     neighbour of the true key, and counts FAIL.
  3. Plain accuracy of the key estimate, scored the standard MIREX way: exact 1.0, a fifth away 0.5, relative 0.3,
     parallel 0.2, anything else 0. Also the share of exact matches.
Your files are never changed. Writes <out>/results.md and <out>/verdicts.csv.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from tonicfind.check import check_file, relation
from tonicfind.estimate import all_keys, key_name
from tonicfind.labels import read_csv
from tonicfind.report import write_csv

MIREX = {"fifth": 0.5, "relative": 0.3, "parallel": 0.2}


def mirex_score(true, est) -> float:
    if true == est:
        return 1.0
    r = relation(true, est)
    if r is None:
        return 0.0
    return next(v for k, v in MIREX.items() if k in r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path)
    ap.add_argument("--limit", type=int, default=300, help="songs to use (default 300)")
    ap.add_argument("--out", type=Path, default=Path("real_check"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-clarity", type=float, default=0.55, help="minimum clarity threshold (default: 0.55)")
    ap.add_argument("--support-gap", type=float, default=0.05, help="support gap threshold (default: 0.05)")
    a = ap.parse_args()
    items, problems = read_csv(a.labels, a.audio_dir)
    if len(items) < 20:
        sys.exit(f"only {len(items)} usable rows ({len(problems)} problems); need at least 20")
    rng = random.Random(a.seed)
    rng.shuffle(items)
    items = items[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)

    keys = all_keys()
    true_v, wrong_v, scores = [], [], []
    for i, (p, key, text) in enumerate(items, 1):
        v = check_file(p, key, text, min_clarity=a.min_clarity, support_gap=a.support_gap)
        true_v.append(v)
        if v.estimate:
            far = [k for k in keys if k != key and relation(key, k) is None]
            wrong_v.append(check_file(p, rng.choice(far), min_clarity=a.min_clarity, support_gap=a.support_gap))
        if v.status != "ERROR":
            est = next((k for k in keys if key_name(k) == v.estimate), None)
            scores.append(mirex_score(key, est) if est else 0.0)
        if i % 25 == 0:
            print(f"  {i}/{len(items)}", flush=True)
    write_csv(a.out / "verdicts.csv", true_v)

    n = len(true_v)
    ct = Counter(v.level for v in true_v)
    cw = Counter(v.level for v in wrong_v)
    exact = sum(1 for s in scores if s == 1.0)
    lines = [
        "# tonicfind real-data check", "",
        f"Songs used: {n} (of {len(items)} sampled; {len(problems)} label rows could not be used)", "",
        "## 1. False alarms (true labels)", "",
        f"PASS {ct['PASS']}, WARN {ct['WARN']}, REVIEW {ct['REVIEW']}, FAIL {ct['FAIL']}, ERROR {ct['ERROR']}",
        f"FAIL rate on true labels: {100 * ct['FAIL'] / max(n, 1):.1f}%", "",
        "## 2. Planted wrong labels (an unrelated key swapped in)", "",
        f"Tested: {len(wrong_v)}  FAIL {cw['FAIL']}  REVIEW {cw['REVIEW']}  WARN {cw['WARN']}  PASS {cw['PASS']}",
        f"Caught as FAIL: {100 * cw['FAIL'] / max(len(wrong_v), 1):.1f}%   "
        f"Missed (PASS, wrongly accepted): {100 * cw['PASS'] / max(len(wrong_v), 1):.1f}%", "",
        "## 3. Key estimate accuracy", "",
        f"Exact match: {100 * exact / max(len(scores), 1):.1f}%   MIREX weighted score: {sum(scores) / max(len(scores), 1):.3f}",
        "",
    ]
    (a.out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
