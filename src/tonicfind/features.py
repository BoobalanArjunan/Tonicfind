"""Audio -> chroma (how much of each of the 12 pitch classes is sounding, per frame) plus the file's tuning offset.

Light on purpose: numpy, scipy, soundfile only. Only clear spectral peaks are used (not the whole spectrum), the peaks'
pitch is refined between FFT bins, and the file's own tuning (a recording a bit sharp or flat of A=440) is removed first.
"""
from __future__ import annotations

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

FS = 22050
N_FFT = 8192                    # 2.7 Hz per bin: low notes need this to be told apart
HOP = 4096                      # one frame every 0.19 s
FMIN, FMAX = 100.0, 2500.0
PEAK_FLOOR = 0.02               # a peak must be at least 2% of the loudest peak in its frame
SILENT_FRAME = 1e-3             # frames with under 0.1% of the loudest frame's energy are ignored
PITCH_TOL = 0.35                # a peak further than this (in semitones) from a note is not counted as a note
MAX_SECONDS = 180.0             # long files: only the middle 3 minutes are used
MIN_SECONDS = 3.0


def load_mono(path, max_seconds: float = MAX_SECONDS) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != FS:
        g = gcd(int(sr), FS)
        x = resample_poly(x, FS // g, int(sr) // g).astype(np.float32)
    n = int(max_seconds * FS)
    if len(x) > n:
        s = (len(x) - n) // 2
        x = x[s:s + n]
    return x


def chroma_frames(x: np.ndarray):
    """Returns (chroma [frames x 12, each row sums to 1], tuning in semitones in [-0.5, 0.5]) or (None, 0.0) if the
    audio is shorter than 3 s, silent, or has no pitched peaks."""
    if len(x) < MIN_SECONDS * FS or not np.any(np.abs(x) > 1e-6):
        return None, 0.0
    _, _, Z = stft(x, fs=FS, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None, padded=False)
    M = np.abs(Z)
    energy = (M ** 2).sum(axis=0)
    keep = energy > SILENT_FRAME * energy.max()
    if not keep.any():
        return None, 0.0
    M = M[:, keep]
    bw = FS / N_FFT
    k0, k1 = int(FMIN / bw), int(FMAX / bw)
    S = M[k0 - 1:k1 + 2]                                   # one extra bin each side for peak picking
    mid, left, right = S[1:-1], S[:-2], S[2:]
    fmax = mid.max(axis=0, keepdims=True) + 1e-20
    mask = (mid > left) & (mid >= right) & (mid > PEAK_FLOOR * fmax)
    k, f = np.nonzero(mask)
    if len(k) == 0:
        return None, 0.0
    a, b, c = np.log(left[k, f] + 1e-20), np.log(mid[k, f] + 1e-20), np.log(right[k, f] + 1e-20)
    denom = a - 2 * b + c
    d = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / denom, 0.0)
    d = np.clip(d, -0.5, 0.5)
    freq = (k0 + k + d) * bw
    midi = 69.0 + 12.0 * np.log2(freq / 440.0)
    w = np.sqrt(mid[k, f] / fmax[0, f])

    frac = midi - np.round(midi)
    ang = 2 * np.pi * frac
    tuning = float(np.arctan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum()) / (2 * np.pi))

    cm = midi - tuning
    near = np.round(cm)
    ok = np.abs(cm - near) <= PITCH_TOL
    pc = (near[ok].astype(int)) % 12
    C = np.zeros((M.shape[1], 12))
    np.add.at(C, (f[ok], pc), w[ok])
    rs = C.sum(axis=1)
    good = rs > 0
    if not good.any():
        return None, tuning
    return C[good] / rs[good, None], tuning
