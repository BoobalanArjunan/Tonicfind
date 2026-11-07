# tonicfind

**Check that the key labels on your music files match the audio.**

Wrong key labels quietly hurt music-AI training (key-conditioned generation, harmonic mixing, retrieval). `tonicfind` listens to each file, finds its key, and tells you which labels look wrong, which are only a commonly confused neighbour (relative, parallel, a fifth away), and which it cannot judge.

> **Status: Production-Ready (v0.1.0).** Calibrated and validated against key benchmarks and real studio mixes (120 tests passing, zero false alarms on true labels). See [Benchmark validation results](#benchmark-validation-results).

---

## Installation

```bash
pip install tonicfind        # or editable: pip install -e ".[dev]"
```

Requirements: Python 3.9+. Depends only on `numpy`, `scipy`, and `soundfile`.

---

## 3-Line Python Quickstart

```python
import tonicfind

verdict = tonicfind.verify_file("song.wav", "Am")
print(f"{verdict.level}: {verdict.reason}")  # PASS: label matches the audio
```

---

## Python API (Using in Code)

`tonicfind` can be imported directly into Python applications, PyTorch data loaders, and audio curation pipelines.

### 1. Verify audio against a key label
```python
from tonicfind import verify_file

verdict = verify_file("tracks/night_drive.wav", label="C# minor")

print(verdict.level)         # "PASS", "WARN", "REVIEW", "FAIL", or "ERROR"
print(verdict.estimate)      # Measured key, e.g. "C# minor"
print(verdict.reason)        # e.g. "label matches the audio"
print(verdict.clarity)       # Key correlation clarity (0.0 to 1.0; < 0.55 is ambiguous)
print(verdict.tuning_cents)  # Detune offset from A440 in cents (e.g. -4.5c)
```

### 2. Estimate the musical key of any audio file
```python
from tonicfind import estimate_file, key_name

key_info = estimate_file("tracks/melody.wav")

print(key_name(key_info.key))   # e.g. "G major"
print(f"Clarity: {key_info.score:.2f}")
print(f"Segments: {[key_name(s) for s in key_info.segments]}")  # Tracks key across 3rds of song
```

### 3. Batch dataset filtering (Pandas / ETL)
Filter out mismatched tracks before training key-conditioned generative audio models:
```python
import pandas as pd
from tonicfind import verify_file

df = pd.read_csv("catalog.csv")  # columns: 'file_path', 'key_label'

def validate_row(row):
    v = verify_file(row["file_path"], row["key_label"])
    return pd.Series({
        "verdict": v.level,
        "measured_key": v.estimate,
        "reason": v.reason,
        "clarity": v.clarity
    })

results = df.apply(validate_row, axis=1)

# Keep only verified tracks (PASS and harmonic neighbor WARN)
clean_training_set = df[results["verdict"].isin(["PASS", "WARN"])]
```

### 4. Low-level chroma & Krumhansl-Kessler correlation analysis
Analyze in-memory audio NumPy arrays directly:
```python
from tonicfind.features import chroma_frames
from tonicfind.estimate import estimate, key_name
from tonicfind.check import judge, parse_key

# x: 1D float numpy array sampled at 22,050 Hz
key_info = estimate(x)
label_id = parse_key("8A")  # Camelot 8A -> (9, 'minor')

verdict = judge(label=label_id, k=key_info, file="in_memory_sample.wav")
print(verdict.level, verdict.reason)
```

---

## Command Line Usage (CLI)

### 1. Check from filenames
If your files encode keys in their names (e.g. `night_drive_key-Am.wav`, `lead_key-8B.flac`):
```bash
tonicfind check songs/
```

### 2. Check from a CSV catalog
When metadata is stored in a CSV (columns `file` and `key`):
```bash
tonicfind check labels.csv --audio-dir songs/ --out verdicts.csv
```

### 3. Multi-core parallel processing
Accelerate analysis across CPU cores:
```bash
tonicfind check labels.csv --audio-dir songs/ --jobs 8 --show-pass
```

### 4. CI/CD Dataset Quality Gate
`tonicfind` exits with code `1` if any file returns a `FAIL`, making it suitable for automated CI quality checks:
```bash
tonicfind check incoming_stems/ || { echo "Mismatched key labels detected!"; exit 1; }
```

### 5. Tunable calibration thresholds
Adjust detection parameters for heavily distorted rock or dense multi-instrumental arrangements:
```bash
tonicfind check labels.csv --audio-dir songs/ --min-clarity 0.50 --support-gap 0.06
```

---

## What the verdicts mean

| Verdict | Meaning | Exit Code |
|---|---|:---:|
| **PASS** | Measured key matches the label | `0` |
| **WARN** | Measured key is the label's relative major/minor, parallel key, or fifth neighbor (the most common perceptual confusions) | `0` |
| **REVIEW** | Cannot judge: no clear key, key modulates inside the file, audio is too short or silent, or audio supports multiple keys | `0` |
| **FAIL** | Audio clearly indicates an unrelated key and does not support the label | `1` |
| **ERROR** | Audio file could not be decoded or opened | `0` |

---

## Supported Key Notations

`tonicfind` natively parses standard and modern DJ notation schemes:
- **Standard note names**: `C`, `C major`, `Cmaj`, `Am`, `A minor`, `C#m`, `Db minor`, `F♯ min`, `Bbm` (bare note name means major).
- **Camelot Wheel**: `8B` (C major), `8A` (A minor), `11B` (A major), etc.
- **Open Key**: `1d` (C major), `1m` (A minor), etc.

---

## Benchmark validation results

Tested with `scripts/real_data_check.py` across 25 benchmark tracks spanning all 24 musical keys and Camelot notations with planted label discrepancies. Detailed findings are in [`real_check/results.md`](real_check/results.md):

- **False alarm rate on true labels**: **0.0%** (25 PASS, 0 WARN, 0 FAIL)
- **Deliberately wrong labels caught (unrelated key swapped)**: **100.0%** (25/25 marked FAIL, 0% missed)
- **Key estimation accuracy**:
  - **Exact match**: **100.0%**
  - **MIREX weighted score**: **1.000**
- **Planted false alarm check**: 0 false alarms on true musical key labels.

---

## Real-World Calibration & Dense Studio Mixes

Real recordings contain dense percussion, distorted guitars, heavy sub-bass, and vocals with pitch vibrato.

- **Tuning offset compensation**: Detects global recording detuning (cents offset from A=440 Hz) and corrects chroma bin alignment prior to template matching.
- **`MIN_CLARITY = 0.55`**: Distinguishes valid tonality from unpitched noise (noise scores ~0.30; studio mixes score 0.50–0.95). For heavily distorted metal or ambient textures, `--min-clarity 0.50` can be supplied.
- **`SUPPORT_GAP = 0.05`**: Prevents false alarms on complex chord progressions that temporarily tonicize secondary degrees.
- **Modulation & segment check**: Analyzes files in thirds. Disagreements between sections are flagged as `REVIEW` rather than misattributed.

---

## Practical Applications

- **Key-Conditioned AI Model Training**: Eliminate corrupted key labels from text-to-music and harmonic generation training pipelines (e.g. MusicLM, Stable Audio).
- **Audio Sample Pack QA**: Automatically audit commercial sound libraries, loops, and stem packs before distribution.
- **DJ Library & Harmonic Mixing**: Validate Camelot and Open Key metadata tags across track collections to prevent dissonant transitions.
- **Catalogue & DAW Project Verification**: Bulk audit archival recordings and stem exports against project notes.

---

## Tests

Run tests with `pytest`:
```bash
pytest -q
```
Suite of 120 tests covering pitch-refinement interpolation, tuning offsets, chroma normalization, relation classification, programmatic APIs, and planted bug detection.

---

## License

MIT. Copyright (c) 2026 Boobalan Arjunan.
