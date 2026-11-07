import sys
from functools import lru_cache
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from synth import SR, make_song  # noqa: E402


@lru_cache(maxsize=None)
def song(tonic, mode, seed=0, **kw):
    return make_song(tonic, mode, seed=seed, **kw)


def write(path, x, sr=SR):
    sf.write(str(path), x, sr)
    return path
