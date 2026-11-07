"""Synthetic songs with a KNOWN key: chord progression + bass + melody on the scale + hats + noise, optionally detuned."""
import numpy as np

SR = 22050
MAJOR_PROG = [(0, 4, 7), (5, 9, 0), (7, 11, 2), (0, 4, 7), (9, 0, 4), (5, 9, 0), (7, 11, 2), (0, 4, 7)]
MINOR_PROG = [(0, 3, 7), (5, 8, 0), (7, 11, 2), (0, 3, 7), (8, 0, 3), (5, 8, 0), (7, 11, 2), (0, 3, 7)]
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]


def _tone(midi, dur, detune_cents=0.0, harmonics=6):
    f = 440.0 * 2 ** ((midi - 69 + detune_cents / 100.0) / 12.0)
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, harmonics + 1) if f * h < SR / 2.2)
    env = np.minimum(1.0, t / 0.02) * np.exp(-t * 0.8)
    return s * env


def make_song(tonic, mode, seconds=40.0, seed=0, detune_cents=0.0, noise=0.02, chord_seconds=2.0, drums=True,
              melody=True, progression=None):
    rng = np.random.RandomState(seed)
    x = np.zeros(int(seconds * SR) + 3 * SR)
    prog = progression or (MAJOR_PROG if mode == "major" else MINOR_PROG)
    scale = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    t, i = 0.0, 0
    while t < seconds:
        chord = prog[i % len(prog)]
        for d in chord:
            tone = _tone(60 + (tonic + d) % 12, chord_seconds + 0.2, detune_cents)
            s = int(t * SR)
            x[s:s + len(tone)] += 0.5 * tone
        bass = _tone(36 + (tonic + chord[0]) % 12, chord_seconds + 0.2, detune_cents, harmonics=4)
        s = int(t * SR)
        x[s:s + len(bass)] += 0.9 * bass
        if melody:
            nt = chord_seconds / 4
            for j in range(4):
                deg = scale[rng.choice(len(scale), p=[.22, .1, .18, .1, .22, .1, .08])]
                m = 72 + (tonic + deg) % 12
                tone = _tone(m, nt + 0.1, detune_cents, harmonics=4)
                s = int((t + j * nt) * SR)
                x[s:s + len(tone)] += 0.4 * tone
        t += chord_seconds
        i += 1
    if drums:
        for tt in np.arange(0, seconds, 0.25):
            n = int(0.03 * SR)
            s = int(tt * SR)
            x[s:s + n] += 0.15 * rng.randn(n) * np.exp(-np.arange(n) / SR * 150)
    x += noise * rng.randn(len(x))
    x = x[: int(seconds * SR)]
    return (0.8 * x / np.abs(x).max()).astype(np.float32)
