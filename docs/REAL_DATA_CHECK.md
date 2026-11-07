# Real-data check (do this before trusting tonicfind)

tonicfind has only been tested on **synthetic songs with a known key**. Real music is harder: dense mixes, drums,
vocals, modulations, ambiguous major/minor. Nothing about real-music accuracy is claimed until this check is run.

## What you need
Real songs with **trusted** key labels. Check each dataset's licence before using or sharing it. Options:
- GiantSteps Key (electronic music, Beatport snippets, labels from the dataset)
- GTZAN with its key annotations
- Your own files with labels you trust

Make a CSV with two columns, `file,key` (keys like `Am`, `C# minor`, `8A`).

## Run it
```bash
pip install -e ".[dev]"
python scripts/real_data_check.py --labels labels.csv --audio-dir songs/ --out real_check
```
Send me `real_check/results.md`. Three numbers matter:
1. **FAIL rate on true labels**: how often a correct label is wrongly rejected (false alarms).
2. **Caught as FAIL / missed**: how many planted wrong labels are caught, and how many get a PASS.
3. **Exact match and MIREX score**: how good the key estimate is by itself.

## How I will use the numbers
The two judgement defaults, `MIN_CLARITY = 0.55` and `SUPPORT_GAP = 0.05` in `src/tonicfind/check.py`, were set on synthetic
songs and noise. They will be adjusted from your results, and the README will quote the measured numbers.
