#!/usr/bin/env python3
"""
Preprocessing for abc2chord: build a CSV of 8th-note rows with melody features
and forward-filled chord labels, with 12-key augmentation for Random Forest.

Usage (from project root):
  python scripts/preprocess_training_data.py path/to/tunes.abc [--out training_data.csv]
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import music21
except ImportError:
    music21 = None

# Project root for optional path fix
ROOT = Path(__file__).resolve().parent.parent
NO_CHORD = "N.C."


def split_abc_file(content: str) -> list[str]:
    """Split multi-tune ABC content into one string per tune (X:1, X:2, ...)."""
    # Split on line that starts with X: (tune index)
    chunks = re.split(r"\n(?=X:\d+\s*\n)", content.strip())
    tunes = []
    for c in chunks:
        c = c.strip()
        if not c or not c.startswith("X:"):
            continue
        tunes.append(c)
    return tunes


def get_scores_from_abc_path(abc_path: str):
    """Yield one music21 Score per tune from a file (single or multi-tune)."""
    path = Path(abc_path)
    if not path.exists():
        raise FileNotFoundError(abc_path)
    content = path.read_text(encoding="utf-8", errors="replace")
    tunes = split_abc_file(content)
    if not tunes:
        # Single tune: parse whole file
        yield music21.converter.parse(str(path))
        return
    for tune_str in tunes:
        try:
            score = music21.converter.parse(tune_str, format="abc")
            # May be Opus if tune_str somehow had multiple X:
            if hasattr(score, "scores") and score.scores:
                for s in score.scores:
                    yield s
            else:
                yield score
        except Exception:
            continue


def get_melody_notes(score):
    """First voice/part as flat stream of notes and rests with offsets."""
    if hasattr(score, "parts") and len(score.parts) > 0:
        stream = score.parts[0].flatten()
    else:
        stream = score.flatten()
    return list(stream.notesAndRests)


def get_chord_timeline(score):
    """List of (offset_quarter, chord_figure), sorted by offset."""
    flat = score.flatten()
    chords = flat.getElementsByClass(music21.harmony.ChordSymbol)
    if not chords:
        chords = score.chordify().flatten().getElementsByClass(music21.harmony.ChordSymbol)
    out = [(float(c.offset), c.figure) for c in chords]
    out.sort(key=lambda x: x[0])
    return out


def get_tonic(score):
    """Tonic of the key as a string (e.g. 'C', 'F#')."""
    try:
        ks = score.flat.getElementsByClass(music21.key.KeySignature)
        if ks:
            k = ks[0]
            if hasattr(k, "tonic") and k.tonic is not None:
                return k.tonic.name
        key = score.flat.getElementsByClass(music21.key.Key)
        if key:
            return key[0].tonic.name
    except Exception:
        pass
    return "C"


def get_measure_quarters(score):
    """Quarter notes per measure (e.g. 4 for 4/4)."""
    try:
        ts = score.flat.getElementsByClass(music21.meter.TimeSignature)
        if ts:
            return float(ts[0].numerator) / float(ts[0].denominator) * 4  # 4/4 -> 4
    except Exception:
        pass
    return 4.0


def active_chord_at(chord_timeline, offset_quarter, default=NO_CHORD):
    """Forward fill: chord active at offset_quarter (last chord at or before this time)."""
    best = default
    for o, fig in chord_timeline:
        if o <= offset_quarter:
            best = fig
        else:
            break
    return best


def note_at_offset(notes, offset_quarter):
    """
    Return the note or rest whose [start, start+duration) contains offset_quarter.
    Returns (pitch_midi, duration_quarter); pitch_midi is 0 for rest.
    """
    for n in notes:
        start = float(n.offset)
        dur = float(n.duration.quarterLength)
        if start <= offset_quarter < start + dur:
            if n.isNote:
                return (n.pitch.ps, dur)
            return (0, dur)
    return (0, 0.0)


def extract_rows_for_tune(score, transpose_semitones=0):
    """
    Build one row per 8th-note beat for this score (optionally transposed).
    Returns list of dicts: pitch_midi, duration, beat_pos, key_sig, prev_note, active_chord.
    """
    if transpose_semitones != 0:
        score = score.transpose(transpose_semitones)

    notes = get_melody_notes(score)
    chord_timeline = get_chord_timeline(score)
    tonic = get_tonic(score)
    measure_quarters = get_measure_quarters(score)
    measure_8ths = int(measure_quarters * 2)  # 8 for 4/4

    if not notes:
        return []

    # Piece length in quarters (from last note end)
    end = max(float(n.offset) + float(n.duration.quarterLength) for n in notes)
    num_8ths = max(1, int(end * 2))

    rows = []
    prev_pitch = 0  # MIDI of note on previous 8th-note beat (0 if rest)
    for e in range(num_8ths):
        offset_quarter = e / 2.0
        pitch_midi, duration = note_at_offset(notes, offset_quarter)
        # beat_pos: 1.0, 1.5, 2.0, ... within measure
        beat_pos = 1.0 + (e % measure_8ths) / 2.0
        active = active_chord_at(chord_timeline, offset_quarter)
        rows.append({
            "pitch_midi": int(pitch_midi),
            "duration": round(duration, 4),
            "beat_pos": round(beat_pos, 2),
            "key_sig": tonic,
            "prev_note": int(prev_pitch),
            "active_chord": active,
        })
        prev_pitch = int(pitch_midi)  # next row's prev_note = this beat's pitch (0 if rest)

    return rows


def run_preprocess(abc_path: str, output_path: str = "training_data.csv", augment_12_keys: bool = True):
    """
    Load ABC file(s), extract 8th-note rows with forward-filled chords,
    optionally augment with 12 transpositions, write CSV.
    """
    if music21 is None:
        raise RuntimeError("music21 is required. Install with: pip install music21")

    all_rows = []
    tune_count = 0
    for score in get_scores_from_abc_path(abc_path):
        tune_count += 1
        if augment_12_keys:
            for semitones in range(12):
                rows = extract_rows_for_tune(score, transpose_semitones=semitones)
                all_rows.extend(rows)
        else:
            rows = extract_rows_for_tune(score, transpose_semitones=0)
            all_rows.extend(rows)

    if not all_rows:
        print("No rows extracted. Check that the ABC file has melody and parses correctly.", file=sys.stderr)
        return 0

    df = pd.DataFrame(all_rows)
    # Column order for RF: pitch_midi, duration, beat_pos, key_sig, prev_note, active_chord
    df = df[["pitch_midi", "duration", "beat_pos", "key_sig", "prev_note", "active_chord"]]
    df.to_csv(output_path, index=False)
    print(f"Processed {tune_count} tune(s), {len(df)} rows -> {output_path}")
    return len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ABC tunes to CSV with 8th-note grid and forward-filled chords."
    )
    parser.add_argument(
        "abc_file",
        type=str,
        help="Path to ABC file (single or multi-tune, e.g. 135 tunes).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="training_data.csv",
        help="Output CSV path (default: training_data.csv).",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable 12-key transposition augmentation.",
    )
    args = parser.parse_args()

    try:
        run_preprocess(
            args.abc_file,
            output_path=args.out,
            augment_12_keys=not args.no_augment,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
