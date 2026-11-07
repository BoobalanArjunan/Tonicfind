"""Generate benchmark songs for TonicFind across all 24 musical keys."""
import csv
import sys
from pathlib import Path
import soundfile as sf

# Add tests directory to import synth helper
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from synth import make_song, SR

# 24 standard keys + 1 Camelot code
KEY_SPECS = [
    ("song_01_C_major.wav", 0, "major", "C major"),
    ("song_02_C_minor.wav", 0, "minor", "C minor"),
    ("song_03_CSharp_major.wav", 1, "major", "C# major"),
    ("song_04_CSharp_minor.wav", 1, "minor", "C# minor"),
    ("song_05_D_major.wav", 2, "major", "D major"),
    ("song_06_D_minor.wav", 2, "minor", "D minor"),
    ("song_07_Eb_major.wav", 3, "major", "Eb major"),
    ("song_08_Eb_minor.wav", 3, "minor", "Eb minor"),
    ("song_09_E_major.wav", 4, "major", "E major"),
    ("song_10_E_minor.wav", 4, "minor", "E minor"),
    ("song_11_F_major.wav", 5, "major", "F major"),
    ("song_12_F_minor.wav", 5, "minor", "F minor"),
    ("song_13_FSharp_major.wav", 6, "major", "F# major"),
    ("song_14_FSharp_minor.wav", 6, "minor", "F# minor"),
    ("song_15_G_major.wav", 7, "major", "G major"),
    ("song_16_G_minor.wav", 7, "minor", "G minor"),
    ("song_17_Ab_major.wav", 8, "major", "Ab major"),
    ("song_18_Ab_minor.wav", 8, "minor", "Ab minor"),
    ("song_19_A_major.wav", 9, "major", "A major"),
    ("song_20_A_minor.wav", 9, "minor", "A minor"),
    ("song_21_Bb_major.wav", 10, "major", "Bb major"),
    ("song_22_Bb_minor.wav", 10, "minor", "Bb minor"),
    ("song_23_B_major.wav", 11, "major", "B major"),
    ("song_24_B_minor.wav", 11, "minor", "B minor"),
    ("song_25_Am_Camelot_8A.wav", 9, "minor", "8A"),
]

def main():
    root = Path(__file__).parent.parent
    audio_dir = root / "benchmark_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = root / "benchmark_labels.csv"
    
    print(f"Generating {len(KEY_SPECS)} benchmark songs...")
    rows = [("file", "key")]
    
    for i, (filename, tonic, mode, label) in enumerate(KEY_SPECS, 1):
        filepath = audio_dir / filename
        if not filepath.exists():
            audio = make_song(tonic, mode, seconds=30.0, seed=i * 7)
            sf.write(str(filepath), audio, SR)
            print(f"  [{i}/{len(KEY_SPECS)}] Created {filename} ({label})")
        else:
            print(f"  [{i}/{len(KEY_SPECS)}] Already exists: {filename}")
        rows.append((filename, label))
        
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Wrote labels to {csv_path}")

if __name__ == "__main__":
    main()

