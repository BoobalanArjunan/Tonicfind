"""Where the labelled key comes from: a CSV (columns for file and key) or the file name ("song_key-Am.wav").

Understood key spellings:
  note names   C, C major, Cmaj, C#m, Db minor, F♯ min, Bbm   (a bare note name means major; "m"/"min"/"minor" = minor)
  Camelot      8B (C major), 8A (A minor)                       (letter A = minor, B = major)
  Open Key     1d (C major), 1m (A minor)                       (d = major, m = minor)
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .estimate import KeyId

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}
FILE_COLS = ("file", "filename", "path", "name", "audio", "track")
KEY_COLS = ("key", "label", "label_key", "tonality")
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE = re.compile(r"^([A-Ga-g])\s*([#♯b♭]?)\s*(.*)$")
_CAMELOT = re.compile(r"^(1[0-2]|[1-9])\s*([ABab])$")
_OPEN = re.compile(r"^(1[0-2]|[1-9])\s*([dDmM])$")
_NAME_RE = re.compile(r"key[-_ ]?([A-Ga-g][#♯b♭]?(?:maj|major|min|minor|m)?|(?:1[0-2]|[1-9])[ABab]|(?:1[0-2]|[1-9])[dm])(?![A-Za-z0-9#])")


class LabelError(Exception):
    pass


def parse_key(text: str) -> Optional[KeyId]:
    """Returns (tonic 0-11, 'major'|'minor') or None if the text is not a key."""
    s = (text or "").strip()
    if not s:
        return None
    m = _CAMELOT.match(s)
    if m:
        n, major = int(m.group(1)), m.group(2).upper() == "B"
        t = ((n - 8) * 7) % 12
        return (t, "major") if major else ((t + 9) % 12, "minor")
    m = _OPEN.match(s)
    if m:
        n, major = int(m.group(1)), m.group(2) == "d" or m.group(2) == "D"
        t = ((n - 1) * 7) % 12
        return (t, "major") if major else ((t + 9) % 12, "minor")
    m = _NOTE.match(s)
    if not m:
        return None
    letter, acc, rest = m.group(1).upper(), m.group(2), m.group(3).strip()
    t = _PC[letter] + (1 if acc in ("#", "♯") else -1 if acc in ("b", "♭") else 0)
    low = rest.lower().replace(".", "").replace("-", "").strip()
    if rest == "m" or low in ("min", "minor", "mnr", "minr"):
        mode = "minor"
    elif rest == "" or rest == "M" or low in ("maj", "major", "mj"):
        mode = "major"
    else:
        return None
    return (t % 12, mode)


def key_from_name(name: str) -> Optional[KeyId]:
    m = _NAME_RE.search(name)
    return parse_key(m.group(1)) if m else None


def read_csv(path: Path, audio_dir: Optional[Path] = None) -> Tuple[List[Tuple[Path, KeyId, str]], List[str]]:
    """Returns ([(audio path, key, label text as written)], [problems]). Unreadable labels are reported, not guessed."""
    base = audio_dir or path.parent
    items, problems = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in (rd.fieldnames or [])}
        fc = next((cols[c] for c in FILE_COLS if c in cols), None)
        kc = next((cols[c] for c in KEY_COLS if c in cols), None)
        if not fc or not kc:
            raise LabelError(f"{path.name}: need a file column ({'/'.join(FILE_COLS)}) and a key column ({'/'.join(KEY_COLS)}); found {list(cols)}")
        for i, row in enumerate(rd, start=2):
            name, raw = (row.get(fc) or "").strip(), (row.get(kc) or "").strip()
            k = parse_key(raw)
            if k is None:
                problems.append(f"row {i}: '{name}' has no usable key ('{raw}')")
                continue
            p = Path(name) if Path(name).is_absolute() else base / name
            if not p.exists():
                problems.append(f"row {i}: file not found: {name}")
                continue
            items.append((p, k, raw))
    return items, problems


def from_filenames(folder: Path) -> Tuple[List[Tuple[Path, KeyId, str]], List[str]]:
    items, problems = [], []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXT:
            continue
        m = _NAME_RE.search(p.stem)
        k = parse_key(m.group(1)) if m else None
        if k is None:
            problems.append(f"no key in file name (expected e.g. song_key-Am.wav): {p.name}")
        else:
            items.append((p, k, m.group(1)))
    return items, problems
