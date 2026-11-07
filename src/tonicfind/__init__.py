"""tonicfind: does the key label on a music file match the audio?"""
from ._version import __version__
from .check import MIN_CLARITY, SUPPORT_GAP, Verdict, check_file, estimate_file, judge, verify_file
from .estimate import Key, all_keys, estimate, key_name
from .labels import parse_key

__all__ = [
    "__version__",
    "verify_file",
    "estimate_file",
    "check_file",
    "judge",
    "Verdict",
    "Key",
    "estimate",
    "key_name",
    "all_keys",
    "parse_key",
    "MIN_CLARITY",
    "SUPPORT_GAP",
]
