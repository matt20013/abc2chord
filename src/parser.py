import re
import music21
import music21.key
import music21.meter
import sys
import argparse

# ── Chord simplification ──────────────────────────────────────────────────────

# Mapping trace: collect (raw, simplified) pairs until we hit the limit.
_TRACE_EVENTS: list[tuple[str, str]] = []
_TRACE_LIMIT = 100
_TRACE_PRINTED = False

# Scale-degree validation: printed once for the first tune that has a chord.
_SD_DEBUG_DONE = False
_SD_INTERVAL_NAMES = [
    "Tonic (1)", "b2", "Maj 2nd", "b3", "Maj 3rd",
    "P4",        "b5", "P5",      "b6", "Maj 6th",
    "b7",        "Maj 7th",
]


# Enharmonic map: normalise all sharp roots to their flat equivalents so the
# model learns one chord class per pitch regardless of notation choice.
# Applied after title-casing, before extension collapsing.
_ENHARMONIC_MAP = {
    "A#": "Bb", "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab",
}


def simplify_chord_label(chord_str: str) -> str:
    """
    Normalise a raw chord label to a compact, consistent vocabulary.

    Rules (applied in order):
      0. Strip parentheses            (F) → F
      1. Strip slash-bass             G/B → G, D/F# → D
      2a. Fix ABC dash-flat root      B-m → Bbm, E-7 → Eb7  (dash = flat in compound)
      2b. Fix ABC dash-minor (EOL)    B-  → Bm              (dash = minor at end)
      3. Title-case root letter       bm  → Bm, dm → Dm
      4. Enharmonic normalisation     A#→Bb, C#→Db, D#→Eb, F#→Gb, G#→Ab
      5. Collapse extensions:
           m / m7 / m9               → plain minor  (Cm7→Cm)
           maj7 / add9 / sus4        → plain major  (Dmaj7→D)
           7 / 9 / 11 / 13           → dominant 7   (A9→A7)
           dim / dim7 / o / o7       → dim          (Bdim7→Bdim)
           aug / +                   → aug          (C#+→Cbaug via step 4, or Dbaug)
    Preserved distinctions: Major, Minor (m), Dominant 7 (7), dim, aug.
    """
    if not chord_str or chord_str == "N.C.":
        return chord_str

    c = chord_str.strip()

    # 0. Strip parentheses — "(F)" and "F" are the same optional/alternative harmony
    c = re.sub(r"^\((.+)\)$", r"\1", c)
    if not c:
        return chord_str

    # 1. Strip slash-bass (take everything before the first '/')
    c = c.split("/")[0]

    # 2a. Fix ABC dash-flat in compound chords: letter + '-' + more chars
    #     e.g.  B-m → Bbm   E-7 → Eb7   B-dim → Bbdim
    c = re.sub(r"^([A-Ga-g])-(\S)", lambda m: m.group(1) + "b" + m.group(2), c)

    # 2b. Fix ABC dash-minor at end-of-string: B- → Bm  (also catches Bb- → Bbm)
    c = re.sub(r"^([A-Ga-g][b]?)-$", lambda m: m.group(1) + "m", c)

    # 3. Title-case root letter
    if c:
        c = c[0].upper() + c[1:]

    # 4. Enharmonic normalisation — map sharp roots to flat equivalents
    #    Check two-char root first (e.g. "A#") before one-char
    if len(c) >= 2 and c[:2] in _ENHARMONIC_MAP:
        c = _ENHARMONIC_MAP[c[:2]] + c[2:]

    # Parse root (note letter + optional accidental b/#)
    root_match = re.match(r"^([A-G][b#]?)", c)
    if not root_match:
        return c  # unrecognised – pass through unchanged
    root = root_match.group(1)
    suffix = c[len(root):]

    # Empty suffix → plain major
    if suffix == "":
        return root

    # Minor extensions: m, m7, m9, m11, m13, etc.
    if re.match(r"^m\d*$", suffix):
        return root + "m"

    # Major extensions: maj7, maj9, maj11, add9, add11, sus2, sus4, sus, 2, 4
    if re.match(r"^(maj\d*|add\d+|sus\d*|2|4)$", suffix, re.IGNORECASE):
        return root

    # Diminished: dim, dim7, o, o7
    if re.match(r"^(dim7?|o7?)$", suffix, re.IGNORECASE):
        return root + "dim"

    # Augmented: aug, +
    if re.match(r"^(aug|\+)$", suffix, re.IGNORECASE):
        return root + "aug"

    # Dominant 7th and higher extensions: 7, 9, 11, 13 (optionally followed by more digits)
    if re.match(r"^(7|9|11|13)\d*$", suffix):
        return root + "7"

    # Fallback: return root + whatever suffix is left (preserves unusual labels)
    return root + suffix


def print_chord_mapping_trace() -> None:
    """
    Print the first TRACE_LIMIT raw → simplified chord mapping events.
    Safe to call multiple times; only prints once.
    """
    global _TRACE_PRINTED
    if _TRACE_PRINTED:
        return
    _TRACE_PRINTED = True
    print("\n── Chord Mapping Trace (first 100 events) ─────────────────────────")
    for raw, simplified in _TRACE_EVENTS:
        arrow = "  →  " if raw != simplified else "  =  "
        print(f"    {raw:<22}{arrow}{simplified}")
    print("────────────────────────────────────────────────────────────────────\n")


def _get_key_info(score) -> tuple[int, str]:
    """Return (tonic_pitch_class 0-11, human-readable key label) for the score."""
    flat = score.flatten()
    key_objs = list(flat.getElementsByClass(music21.key.Key))
    if key_objs:
        k = key_objs[0]
        return k.tonic.pitchClass, f"{k.tonic.name} {k.mode}"
    key_sigs = list(flat.getElementsByClass(music21.key.KeySignature))
    if key_sigs:
        try:
            k = key_sigs[0].asKey()
            return k.tonic.pitchClass, f"{k.tonic.name} {k.mode}"
        except Exception:
            pass
    return 2, "D major (default)"  # most common in Irish / folk


def _get_meter_numerator(score) -> int:
    """Return the time-signature numerator (e.g. 4 for 4/4, 6 for 6/8), default 4."""
    ts_objs = list(score.flatten().getElementsByClass(music21.meter.TimeSignature))
    if ts_objs:
        return int(ts_objs[0].numerator)
    return 4


# Largest meter numerator we expect (12/8); used for normalization.
_METER_NUM_MAX = 12.0


def _print_scale_degree_debug(notes: list, key_label: str) -> None:
    """Print a one-time validation table: Original Pitch → Key → Scale Degree."""
    global _SD_DEBUG_DONE
    if _SD_DEBUG_DONE:
        return
    # Only trigger on a tune that actually has chord annotations.
    if not any(r["target_chord"] != "N.C." for r in notes):
        return
    _SD_DEBUG_DONE = True
    print("\n── Scale Degree Validation (first annotated tune) ──────────────────")
    print(f"   Key detected : {key_label}")
    print(f"   {'Note':<7} {'MIDI':>5}  {'formula':^18}  {'Degree':>6}  Interval")
    print(f"   {'─'*7}  {'─'*5}  {'─'*18}  {'─'*6}  {'─'*14}")
    tonic_pc = None
    for row in notes[:12]:
        midi    = int(row["pitch"])
        sd      = int(row["scale_degree"])
        if tonic_pc is None:
            tonic_pc = (midi - sd) % 12  # back-calculate for display
        pitch_obj  = music21.pitch.Pitch(midi)
        note_name  = pitch_obj.nameWithOctave
        formula    = f"({midi} - {(midi - sd) % 12 + (midi // 12) * 12 - (midi % 12 - (midi % 12))}) % 12"
        formula    = f"({midi} - tonic) % 12"
        interval   = _SD_INTERVAL_NAMES[sd]
        print(f"   {note_name:<7}  {midi:>5}  {formula:^18}  {sd:>6}  {interval}")
    print("────────────────────────────────────────────────────────────────────\n")


def _extract_features_from_score(score):
    """Extract (Note_Context, Chord_Label) from a single music21 Score.

    Feature vector per note:
        pitch        – MIDI semitone (normalised in tune_to_arrays)
        duration     – quarter-length duration
        beat         – beat position within the bar
        measure      – measure number
        is_rest      – always 0 for notes (rests are skipped)
        scale_degree – chromatic interval above tonic (0-11), captures modal context
        meter_norm   – time-sig numerator / 12, encodes genre rhythmic feel
    """
    if hasattr(score, "parts") and len(score.parts) > 0:
        melody = score.parts[0].flatten()
    else:
        melody = score.flatten()

    chords = score.flatten().getElementsByClass(music21.harmony.ChordSymbol)
    if not chords:
        chords = score.chordify().flatten().getElementsByClass(music21.harmony.ChordSymbol)
    chords = sorted(chords, key=lambda x: x.offset)

    tonic_pc, key_label = _get_key_info(score)
    meter_num  = _get_meter_numerator(score)
    meter_norm = min(meter_num, _METER_NUM_MAX) / _METER_NUM_MAX

    dataset = []
    for n in melody.notesAndRests:
        active_chord = "N.C."
        current_best_chord = None
        for c in chords:
            if c.offset <= n.offset:
                current_best_chord = c
            else:
                break
        if current_best_chord:
            raw_chord = current_best_chord.figure
            active_chord = simplify_chord_label(raw_chord)
            # Accumulate mapping trace (raw → simplified) until limit reached
            if len(_TRACE_EVENTS) < _TRACE_LIMIT:
                _TRACE_EVENTS.append((raw_chord, active_chord))

        if n.isNote:
            scale_degree = (n.pitch.pitchClass - tonic_pc) % 12  # 0-11
            dataset.append({
                "pitch":        n.pitch.ps,
                "duration":     n.duration.quarterLength,
                "beat":         n.beat,
                "measure":      n.measureNumber,
                "is_rest":      0,
                "scale_degree": scale_degree,
                "meter_norm":   meter_norm,
                "target_chord": active_chord,
            })
    _print_scale_degree_debug(dataset, key_label)
    return dataset


def _split_abc_file(content):
    """Split multi-tune ABC content into one string per tune (X:1, X: 2, etc.)."""
    # Match X: followed by optional space and digits (start of new tune)
    chunks = re.split(r"\n(?=X:\s*\d)", content.strip(), flags=re.IGNORECASE)
    tunes = []
    for c in chunks:
        c = c.strip()
        if not c or not re.match(r"X:\s*\d", c, re.IGNORECASE):
            continue
        tunes.append(c)
    return tunes


def extract_features_from_abc(abc_file_path):
    """
    Parses an ABC file and creates a dataset of (Note_Context, Chord_Label).
    For multi-tune files (X:1, X:2, ...), returns the first tune only (backward compatible).
    """
    with open(abc_file_path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    tunes = _split_abc_file(content)
    if not tunes:
        score = music21.converter.parse(abc_file_path)
        if hasattr(score, "scores") and score.scores:
            return _extract_features_from_score(score.scores[0])
        return _extract_features_from_score(score)
    score = music21.converter.parse(tunes[0], format="abc")
    if hasattr(score, "scores") and score.scores:
        return _extract_features_from_score(score.scores[0])
    return _extract_features_from_score(score)


def _iter_scores_from_abc(abc_file_path):
    """Yield raw music21 Score objects from an ABC file (one per tune)."""
    with open(abc_file_path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    tunes = _split_abc_file(content)
    if not tunes:
        score = music21.converter.parse(abc_file_path)
        if hasattr(score, "scores") and score.scores:
            yield from score.scores
        else:
            yield score
        return
    for tune_str in tunes:
        try:
            score = music21.converter.parse(tune_str, format="abc")
            if hasattr(score, "scores") and score.scores:
                yield from score.scores
            else:
                yield score
        except Exception:
            continue


def extract_all_tunes_from_abc(abc_file_path):
    """
    Parse an ABC file and yield one dataset per tune (handles multi-tune files).
    """
    for score in _iter_scores_from_abc(abc_file_path):
        yield _extract_features_from_score(score)

def main():
    parser = argparse.ArgumentParser(description='Extract features from ABC file.')
    parser.add_argument('file', type=str, help='Path to the ABC file')
    args = parser.parse_args()

    try:
        data = extract_features_from_abc(args.file)
        print(f"Extracted {len(data)} notes.")
        for item in data[:5]:
            print(item)
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
