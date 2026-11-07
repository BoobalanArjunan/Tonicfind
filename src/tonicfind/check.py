"""Compare a labelled key with the audio and give a verdict."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .estimate import Key, KeyId, estimate, key_name
from .features import load_mono

MIN_CLARITY = 0.55         # best-key correlation below this: the audio has no clear key, so the label cannot be judged
SUPPORT_GAP = 0.05         # the label counts as "also supported by the audio" if its score is within this of the best

OK, NEIGHBOUR, WRONG, UNSURE, ERROR = "OK", "NEIGHBOUR", "WRONG", "UNSURE", "ERROR"


@dataclass
class Verdict:
    file: str
    label: str
    status: str
    estimate: Optional[str]
    reason: str
    clarity: float = 0.0
    tuning_cents: float = 0.0

    @property
    def level(self) -> str:
        return {OK: "PASS", NEIGHBOUR: "WARN", UNSURE: "REVIEW", WRONG: "FAIL", ERROR: "ERROR"}[self.status]


def relation(label: KeyId, est: KeyId) -> Optional[str]:
    """How two different keys are related, if they are one of the three pairs people confuse most; else None."""
    (lt, lm), (et, em) = label, est
    if lm != em:
        major_t, minor_t = (lt, et) if lm == "major" else (et, lt)
        if (major_t - minor_t) % 12 == 3:
            return "relative key (shares the same notes)"
        if lt == et:
            return "same tonic, other mode (parallel key)"
    elif (et - lt) % 12 in (5, 7):
        return "a fifth away (shares all but one note)"
    return None


def judge(label: KeyId, k: Key, file: str = "", label_text: str = "",
          min_clarity: float = MIN_CLARITY, support_gap: float = SUPPORT_GAP) -> Verdict:
    lt = label_text or key_name(label)
    if k.key is None:
        return Verdict(file, lt, UNSURE, None, "too short, silent or without pitched sound: cannot measure a key", 0.0, k.tuning_cents)
    est = key_name(k.key)
    base = dict(clarity=k.score, tuning_cents=k.tuning_cents)
    if k.segments and any(s != k.key and relation(k.key, s) is None for s in k.segments):
        return Verdict(file, lt, UNSURE, est, "key changes inside the file: " + ", ".join(key_name(s) for s in k.segments), **base)
    if k.score < min_clarity:
        return Verdict(file, lt, UNSURE, est, f"no clear key (clarity {k.score:.2f}); label cannot be judged", **base)
    if k.key == label:
        return Verdict(file, lt, OK, est, "label matches the audio", **base)
    rel = relation(label, k.key)
    if rel:
        return Verdict(file, lt, NEIGHBOUR, est, f"audio says {est}: {rel}; often hard to tell apart", **base)
    if k.score_at(label) >= k.score - support_gap:
        return Verdict(file, lt, UNSURE, est, f"audio supports both the label and {est}", **base)
    return Verdict(file, lt, WRONG, est, f"audio says {est}, label says {key_name(label)}", **base)


def check_file(path, label: KeyId | str, label_text: str = "",
               min_clarity: float = MIN_CLARITY, support_gap: float = SUPPORT_GAP) -> Verdict:
    p = Path(path)
    if isinstance(label, str):
        from .labels import parse_key
        parsed = parse_key(label)
        if parsed is None:
            return Verdict(p.name, label, ERROR, None, f"unrecognized key format: '{label}'")
        raw_text = label_text or label
        key_id = parsed
    else:
        key_id = label
        raw_text = label_text or key_name(key_id)

    try:
        k = estimate(load_mono(p))
    except Exception as e:  # unreadable audio
        return Verdict(p.name, raw_text, ERROR, None, f"could not read audio: {e}")
    return judge(key_id, k, p.name, raw_text, min_clarity=min_clarity, support_gap=support_gap)


def verify_file(path, label: str | KeyId,
                min_clarity: float = MIN_CLARITY, support_gap: float = SUPPORT_GAP) -> Verdict:
    """Check that an audio file's key matches the given label.

    Example:
        >>> verdict = tonicfind.verify_file("song.wav", "Am")
        >>> print(verdict.level, verdict.reason)
        PASS label matches the audio
    """
    return check_file(path, label, min_clarity=min_clarity, support_gap=support_gap)


def estimate_file(path) -> Key:
    """Measure the musical key of an audio file.

    Example:
        >>> key_info = tonicfind.estimate_file("song.wav")
        >>> print(tonicfind.key_name(key_info.key), key_info.score)
    """
    return estimate(load_mono(Path(path)))
