"""Chroma -> most likely key, by comparing the chroma with the 24 Krumhansl-Kessler key profiles (12 major, 12 minor)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .features import FS, HOP, chroma_frames

MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SEGMENTS = 3
MIN_SEGMENT_SECONDS = 10.0

KeyId = Tuple[int, str]          # (tonic 0-11 with C=0, "major" | "minor")


def key_name(k: KeyId) -> str:
    return f"{NAMES[k[0]]} {k[1]}"


def all_keys() -> List[KeyId]:
    return [(t, "major") for t in range(12)] + [(t, "minor") for t in range(12)]


def _corr(v: np.ndarray, p: np.ndarray) -> float:
    v = v - v.mean()
    p = p - p.mean()
    d = np.sqrt((v * v).sum() * (p * p).sum())
    return float((v * p).sum() / d) if d > 1e-12 else 0.0


def key_scores(vec: np.ndarray) -> np.ndarray:
    """24 correlations: index 0-11 = major with tonic C..B, 12-23 = minor with tonic C..B."""
    return np.array([_corr(vec, np.roll(MAJOR, t)) for t in range(12)] + [_corr(vec, np.roll(MINOR, t)) for t in range(12)])


def score_of(scores: np.ndarray, k: KeyId) -> float:
    return float(scores[k[0] + (0 if k[1] == "major" else 12)])


def best_of(scores: np.ndarray) -> KeyId:
    i = int(np.argmax(scores))
    return (i, "major") if i < 12 else (i - 12, "minor")


@dataclass
class Key:
    key: Optional[KeyId]                    # None = too short / silent / no pitched content
    score: float = 0.0                      # correlation of the best key (higher = clearer key)
    second: float = 0.0                     # correlation of the runner-up
    scores: Optional[np.ndarray] = None     # all 24
    segments: List[KeyId] = field(default_factory=list)   # key of each third of the file (only for long enough files)
    tuning_cents: float = 0.0

    def score_at(self, k: KeyId) -> float:
        return score_of(self.scores, k) if self.scores is not None else 0.0


def estimate(x: np.ndarray) -> Key:
    C, tuning = chroma_frames(x)
    if C is None:
        return Key(None, tuning_cents=100.0 * tuning)
    sc = key_scores(C.mean(axis=0))
    order = np.sort(sc)[::-1]
    segs: List[KeyId] = []
    sec_per_frame = HOP / FS
    if len(C) * sec_per_frame >= SEGMENTS * MIN_SEGMENT_SECONDS:
        for part in np.array_split(C, SEGMENTS):
            segs.append(best_of(key_scores(part.mean(axis=0))))
    return Key(best_of(sc), float(order[0]), float(order[1]), sc, segs, 100.0 * tuning)
