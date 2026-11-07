from __future__ import annotations

import csv
from collections import Counter
from typing import List

from .check import Verdict


def summary_line(vs: List[Verdict]) -> str:
    c = Counter(v.level for v in vs)
    return (f"{len(vs)} files: {c['PASS']} pass, {c['WARN']} warn (relative/parallel/fifth key), "
            f"{c['REVIEW']} need a look, {c['FAIL']} fail, {c['ERROR']} unreadable")


def text(vs: List[Verdict], problems: List[str], show_pass: bool = False) -> str:
    L = []
    for v in vs:
        if v.level == "PASS" and not show_pass:
            continue
        L.append(f"{v.level:6} {v.file}  label {v.label}  measured {v.estimate or '-'}  {v.reason}")
    if problems:
        L += ["", f"{len(problems)} label problems (not checked):"] + [f"  {p}" for p in problems[:50]]
        if len(problems) > 50:
            L.append(f"  ... and {len(problems) - 50} more")
    L += ["", summary_line(vs)]
    return "\n".join(L)


def write_csv(path, vs: List[Verdict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label_key", "measured_key", "verdict", "reason", "clarity", "tuning_cents"])
        for v in vs:
            w.writerow([v.file, v.label, v.estimate or "", v.level, v.reason, round(v.clarity, 3), round(v.tuning_cents, 1)])
