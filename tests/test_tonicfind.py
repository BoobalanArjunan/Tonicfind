import csv

import numpy as np
import pytest
import soundfile as sf

from conftest import SR, song, write
from synth import make_song
from tonicfind.check import (MIN_CLARITY, NEIGHBOUR, OK, UNSURE, WRONG, check_file, judge, relation)
from tonicfind.cli import main
from tonicfind.features import FS, chroma_frames
from tonicfind.estimate import Key, estimate, key_name, key_scores, score_of
from tonicfind.labels import LabelError, from_filenames, key_from_name, parse_key, read_csv

ALL = [(t, m) for m in ("major", "minor") for t in range(12)]


# ---------------------------------------------------------------- label parsing
@pytest.mark.parametrize("text,expect", [
    ("C", (0, "major")), ("C major", (0, "major")), ("Cmaj", (0, "major")), ("c", (0, "major")),
    ("Am", (9, "minor")), ("A minor", (9, "minor")), ("A min", (9, "minor")), ("a-minor", (9, "minor")),
    ("C#m", (1, "minor")), ("C♯ minor", (1, "minor")), ("Db major", (1, "major")), ("D♭", (1, "major")),
    ("Bbm", (10, "minor")), ("Bb", (10, "major")), ("B", (11, "major")), ("F#", (6, "major")),
    ("Em", (4, "minor")), ("EM", (4, "major")), ("  G  maj ", (7, "major")),
])
def test_parse_note_names(text, expect):
    assert parse_key(text) == expect


@pytest.mark.parametrize("text,expect", [
    ("8B", (0, "major")), ("8A", (9, "minor")), ("9B", (7, "major")), ("7B", (5, "major")),
    ("1B", (11, "major")), ("12B", (4, "major")), ("5A", (0, "minor")), ("1A", (8, "minor")),
    ("1d", (0, "major")), ("1m", (9, "minor")), ("2d", (7, "major")), ("6d", (11, "major")),
])
def test_parse_camelot_and_open_key(text, expect):
    assert parse_key(text) == expect


def test_camelot_wheel_is_consistent():
    """Each Camelot number is a major key and its relative minor; neighbours are a fifth apart."""
    for n in range(1, 13):
        major, minor = parse_key(f"{n}B"), parse_key(f"{n}A")
        assert (major[0] - minor[0]) % 12 == 3
        nxt = parse_key(f"{n % 12 + 1}B")
        assert (nxt[0] - major[0]) % 12 == 7


@pytest.mark.parametrize("text", ["", "   ", "H", "X major", "C lydian", "13A", "0B", "key", "120", None])
def test_parse_rejects_non_keys(text):
    assert parse_key(text) is None


@pytest.mark.parametrize("name,expect", [
    ("night_drive_key-Am", (9, "minor")), ("song_key_C#", (1, "major")), ("x key F#minor", (6, "minor")),
    ("loop_key-8A", (9, "minor")), ("loop_key-1d", (0, "major")), ("a_key-Bbm_final", (10, "minor")),
])
def test_key_from_name(name, expect):
    assert key_from_name(name) == expect


def test_key_from_name_ignores_words_that_only_look_like_keys():
    assert key_from_name("monkey_business") is None
    assert key_from_name("song_in_am") is None


def test_read_csv_ok_and_problems(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "b.wav").write_bytes(b"x")
    (tmp_path / "l.csv").write_text("File,Key\na.wav,Am\nb.wav,8B\nc.wav,C\nb.wav,nonsense\n,\n", encoding="utf-8")
    items, problems = read_csv(tmp_path / "l.csv")
    assert [(p.name, k, t) for p, k, t in items] == [("a.wav", (9, "minor"), "Am"), ("b.wav", (0, "major"), "8B")]
    assert len(problems) == 3
    assert any("not found" in p for p in problems) and any("nonsense" in p for p in problems)


def test_read_csv_missing_columns(tmp_path):
    (tmp_path / "l.csv").write_text("file,tempo\na.wav,120\n", encoding="utf-8")
    with pytest.raises(LabelError):
        read_csv(tmp_path / "l.csv")


def test_read_csv_audio_dir_and_bom(tmp_path):
    d = tmp_path / "songs"
    d.mkdir()
    (d / "a.wav").write_bytes(b"x")
    (tmp_path / "l.csv").write_text("﻿file,key\na.wav,C\n", encoding="utf-8")
    items, problems = read_csv(tmp_path / "l.csv", d)
    assert len(items) == 1 and not problems


def test_from_filenames(tmp_path):
    for n in ("one_key-Am.wav", "two_key-8B.flac", "three.wav", "notes.txt"):
        (tmp_path / n).write_bytes(b"x")
    items, problems = from_filenames(tmp_path)
    assert sorted(p.name for p, _, _ in items) == ["one_key-Am.wav", "two_key-8B.flac"]
    assert len(problems) == 1 and "three.wav" in problems[0]


# ---------------------------------------------------------------- relations
def test_relations():
    assert "relative" in relation((0, "major"), (9, "minor"))
    assert "relative" in relation((9, "minor"), (0, "major"))
    assert "parallel" in relation((0, "major"), (0, "minor"))
    assert "fifth" in relation((0, "major"), (7, "major"))
    assert "fifth" in relation((0, "major"), (5, "major"))
    assert "fifth" in relation((9, "minor"), (4, "minor"))
    assert relation((0, "major"), (6, "major")) is None
    assert relation((0, "major"), (1, "minor")) is None
    assert relation((0, "major"), (2, "major")) is None
    assert relation((0, "major"), (4, "minor")) is None     # mediant minor is NOT the relative minor (A minor is)


def test_relation_is_symmetric():
    for a in ALL:
        for b in ALL:
            if a != b:
                assert (relation(a, b) is None) == (relation(b, a) is None)


# ---------------------------------------------------------------- estimate on audio
@pytest.mark.parametrize("key", ALL, ids=[key_name(k) for k in ALL])
def test_estimate_finds_each_of_24_keys(key):
    k = estimate(song(key[0], key[1]))
    assert k.key == key
    assert k.score > 0.8


@pytest.mark.parametrize("cents", [-40, -20, 20, 40])
def test_detuned_recording_is_still_found_and_tuning_reported(cents):
    k = estimate(song(2, "major", detune_cents=cents))
    assert k.key == (2, "major")
    assert abs(k.tuning_cents - cents) < 8


def test_other_seeds_and_progressions():
    for seed in (3, 4):
        assert estimate(make_song(5, "minor", seed=seed)).key == (5, "minor")
    assert estimate(make_song(7, "major", seed=9, drums=False, melody=False)).key == (7, "major")


def test_noise_has_no_clear_key():
    rng = np.random.RandomState(1)
    for _ in range(5):
        k = estimate((0.3 * rng.randn(SR * 30)).astype(np.float32))
        assert k.score < MIN_CLARITY - 0.1


def test_too_short_and_silent_give_no_key():
    assert estimate(np.zeros(SR * 10, dtype=np.float32)).key is None
    assert estimate(song(0, "major")[: SR * 2]).key is None
    assert estimate(np.zeros(0, dtype=np.float32)).key is None


def test_only_quiet_noise_floor_is_not_a_key():
    rng = np.random.RandomState(2)
    k = estimate((1e-9 * rng.randn(SR * 10)).astype(np.float32))
    assert k.key is None or k.score < MIN_CLARITY


def test_key_scores_shape_and_best():
    v = np.zeros(12)
    for pc in (0, 4, 7):
        v[pc] = 1.0
    v[[2, 5, 9, 11]] = 0.2
    sc = key_scores(v)
    assert sc.shape == (24,)
    assert int(np.argmax(sc)) == 0
    assert score_of(sc, (0, "major")) == sc[0] and score_of(sc, (0, "minor")) == sc[12]


def test_flat_chroma_gives_zero_scores():
    assert np.allclose(key_scores(np.ones(12)), 0.0)


def test_long_file_uses_middle_three_minutes_and_has_segments(tmp_path):
    x = song(4, "major", seconds=40.0)
    k = estimate(x)
    assert len(k.segments) == 3 and all(s == (4, "major") for s in k.segments)


def test_short_file_has_no_segments():
    assert estimate(make_song(0, "major", seconds=20.0)).segments == []


# ---------------------------------------------------------------- verdicts (unit, on constructed Keys)
def _key(best, score=0.9, second=0.5, segments=(), others=None):
    sc = np.full(24, 0.1)
    sc[best[0] + (0 if best[1] == "major" else 12)] = score
    for k, v in (others or {}).items():
        sc[k[0] + (0 if k[1] == "major" else 12)] = v
    return Key(best, score, second, sc, list(segments), 0.0)


def test_judge_pass():
    v = judge((0, "major"), _key((0, "major")), "a.wav", "C")
    assert (v.status, v.level, v.label) == (OK, "PASS", "C")


def test_judge_neighbours_warn():
    for lab in ((9, "minor"), (0, "minor"), (7, "major"), (5, "major")):
        v = judge(lab, _key((0, "major")))
        assert v.status == NEIGHBOUR and v.level == "WARN", lab


def test_judge_wrong_fails():
    v = judge((6, "major"), _key((0, "major")))
    assert v.status == WRONG and v.level == "FAIL" and "C major" in v.reason


def test_judge_label_also_supported_is_review_not_fail():
    v = judge((6, "major"), _key((0, "major"), 0.9, others={(6, "major"): 0.87}))
    assert v.status == UNSURE


def test_judge_support_gap_edge():
    just_in = judge((6, "major"), _key((0, "major"), 0.9, others={(6, "major"): 0.851}))
    just_out = judge((6, "major"), _key((0, "major"), 0.9, others={(6, "major"): 0.849}))
    assert just_in.status == UNSURE and just_out.status == WRONG


def test_judge_low_clarity_is_review_even_for_matching_label():
    v = judge((0, "major"), _key((0, "major"), score=MIN_CLARITY - 0.01))
    assert v.status == UNSURE and "no clear key" in v.reason
    assert judge((0, "major"), _key((0, "major"), score=MIN_CLARITY)).status == OK


def test_judge_no_key_is_review():
    v = judge((0, "major"), Key(None))
    assert v.status == UNSURE and v.estimate is None


def test_judge_modulation_is_review():
    k = _key((0, "major"), segments=[(0, "major"), (6, "major"), (1, "minor")])
    v = judge((0, "major"), k)
    assert v.status == UNSURE and "changes" in v.reason


def test_judge_related_segments_do_not_trigger_modulation():
    k = _key((0, "major"), segments=[(0, "major"), (9, "minor"), (7, "major")])
    assert judge((0, "major"), k).status == OK


def test_judge_modulation_beats_a_label_that_matches_part_of_the_song():
    k = _key((0, "major"), segments=[(0, "major"), (6, "major"), (6, "major")])
    assert judge((6, "major"), k).status == UNSURE


# ---------------------------------------------------------------- end to end on audio files
def test_check_file_pass(tmp_path):
    p = write(tmp_path / "a.wav", song(9, "minor"))
    v = check_file(p, (9, "minor"), "Am")
    assert v.status == OK and v.file == "a.wav" and v.estimate == "A minor"


def test_check_file_relative_label_warns(tmp_path):
    p = write(tmp_path / "a.wav", song(0, "major"))
    assert check_file(p, (9, "minor")).status == NEIGHBOUR


def test_every_unrelated_wrong_label_is_caught(tmp_path):
    """A C major song labelled with each of the other 23 keys: none may PASS, and unrelated ones must FAIL."""
    p = write(tmp_path / "a.wav", song(0, "major"))
    for lab in ALL:
        v = check_file(p, lab)
        if lab == (0, "major"):
            assert v.status == OK
        elif relation(lab, (0, "major")):
            assert v.status == NEIGHBOUR, (lab, v.reason)
        else:
            assert v.status == WRONG, (lab, v.reason)


def test_false_alarms_on_true_labels_synthetic():
    bad = [k for k in ALL if judge(k, estimate(song(k[0], k[1], seed=5))).status != OK]
    assert bad == []


def test_stereo_44k_file(tmp_path):
    x = song(3, "major")
    import scipy.signal as sg
    y = sg.resample_poly(x, 2, 1).astype(np.float32)
    st = np.stack([y, 0.7 * y], axis=1)
    p = tmp_path / "s.wav"
    sf.write(str(p), st, 2 * SR)
    assert check_file(p, (3, "major")).status == OK


def test_flac_file(tmp_path):
    p = tmp_path / "a.flac"
    sf.write(str(p), song(10, "minor"), SR)
    assert check_file(p, (10, "minor")).status == OK


def test_level_changes_do_not_change_the_key(tmp_path):
    x = song(2, "minor")
    assert estimate(0.05 * x).key == estimate(x).key == (2, "minor")


def test_unreadable_file_is_error(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"this is not audio")
    v = check_file(p, (0, "major"))
    assert v.status == "ERROR" and v.level == "ERROR" and "could not read" in v.reason


# ---------------------------------------------------------------- CLI
def _folder(tmp_path):
    write(tmp_path / "good_key-C.wav", song(0, "major"))
    write(tmp_path / "rel_key-Am.wav", song(0, "major"))
    write(tmp_path / "bad_key-F#.wav", song(0, "major"))
    return tmp_path


def test_cli_folder_exit_code_and_output(tmp_path, capsys):
    rc = main(["check", str(_folder(tmp_path))])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out and "WARN" in out and "bad_key-F#.wav" in out and "good_key-C.wav" not in out
    assert "3 files: 1 pass, 1 warn" in out


def test_cli_show_pass_and_out_csv(tmp_path, capsys):
    d = _folder(tmp_path)
    o = tmp_path / "v.csv"
    main(["check", str(d), "--show-pass", "--out", str(o)])
    assert "good_key-C.wav" in capsys.readouterr().out
    rows = list(csv.DictReader(open(o, encoding="utf-8")))
    assert {r["file"]: r["verdict"] for r in rows} == {"good_key-C.wav": "PASS", "rel_key-Am.wav": "WARN", "bad_key-F#.wav": "FAIL"}
    assert set(rows[0]) == {"file", "label_key", "measured_key", "verdict", "reason", "clarity", "tuning_cents"}


def test_cli_csv_source_and_exit_zero(tmp_path, capsys):
    write(tmp_path / "a.wav", song(7, "major"))
    (tmp_path / "l.csv").write_text("file,key\na.wav,G\n", encoding="utf-8")
    assert main(["check", str(tmp_path / "l.csv")]) == 0
    assert "1 files: 1 pass" in capsys.readouterr().out


def test_cli_parallel_jobs_same_result(tmp_path, capsys):
    d = _folder(tmp_path)
    main(["check", str(d)])
    one = capsys.readouterr().out
    main(["check", str(d), "--jobs", "2"])
    assert capsys.readouterr().out == one


def test_cli_usage_errors(tmp_path, capsys):
    assert main(["check", str(tmp_path / "missing")]) == 2
    assert main(["check", str(tmp_path)]) == 2                      # empty folder
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert main(["check", str(tmp_path / "bad.csv")]) == 2
    assert "error" in capsys.readouterr().err


def test_cli_reports_label_problems(tmp_path, capsys):
    write(tmp_path / "a_key-C.wav", song(0, "major"))
    (tmp_path / "noname.wav").write_bytes(b"x")
    main(["check", str(tmp_path)])
    assert "1 label problems" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "tonicfind 0.1.0a1" in capsys.readouterr().out


# ---------------------------------------------------------------- feature details
def _sines(freqs, seconds=5):
    t = np.arange(int(seconds * FS)) / FS
    return sum(np.sin(2 * np.pi * f * t) for f in freqs).astype(np.float32) / len(freqs)


def test_pitch_is_refined_between_fft_bins():
    """A low note 30 cents above C3 falls between FFT bins (a bin is ~36 cents wide here); the tuning must still be ~30."""
    C, tuning = chroma_frames(_sines([130.81 * 2 ** (0.30 / 12)]))
    assert abs(100 * tuning - 30) < 6
    assert int(np.argmax(C.mean(axis=0))) == 0


def test_notes_halfway_between_semitones_are_not_counted():
    """C4 plus a tone exactly between D4 and Eb4: only C may appear in the chroma (the in-between tone is not a note)."""
    C, _ = chroma_frames(_sines([261.6256, 261.6256 * 2 ** (2.5 / 12)]))
    m = C.mean(axis=0)
    assert m[0] > 0.99 and m[2] + m[3] < 0.01


def test_chroma_rows_sum_to_one():
    C, _ = chroma_frames(song(0, "major")[: SR * 10])
    assert np.allclose(C.sum(axis=1), 1.0)


def test_verify_file_programmatic_api(tmp_path):
    import soundfile as sf
    from tonicfind import verify_file, estimate_file
    p = tmp_path / "test_api_song.wav"
    sf.write(str(p), song(9, "minor"), SR)
    v_pass = verify_file(p, "Am")
    assert v_pass.level == "PASS"
    v_fail = verify_file(p, "F# major")
    assert v_fail.level == "FAIL"
    v_invalid = verify_file(p, "not_a_key")
    assert v_invalid.level == "ERROR"
    k_est = estimate_file(p)
    assert k_est.key == (9, "minor")

