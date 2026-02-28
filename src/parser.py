import re
import music21
import music21.key
import music21.meter
import music21.bar
import sys
import argparse

from .constants import _NOTE_TO_PC, _PC_TO_NOTE, _DEGREE_NAMES, _DEGREE_TO_OFFSET


def absolute_to_degree(chord_root_pc: int, key_tonic_pc: int,
                       chord_quality: str = "",
                       is_major_key: bool = True) -> str:
    """
    "Musician's Logic" core: rotate the chromatic scale by the key tonic and
    map the chord quality to a standard Roman Numeral string.

    Args:
        chord_root_pc : MIDI pitch class of the chord root (0-11).
        key_tonic_pc  : MIDI pitch class of the key tonic  (0-11).
        chord_quality : suffix string that carries the quality, e.g. "m", "7",
                        "dim", "aug", "".  Pass the raw simplified suffix.
        is_major_key  : reserved for future modal handling; currently unused.

    Returns:
        Roman numeral string, e.g. "I", "ii", "V7", "bVII", "viidim".

    Examples (key of G, tonic_pc=7):
        absolute_to_degree(7, 7, "")    →  "I"      G major
        absolute_to_degree(0, 7, "")    →  "IV"     C major
        absolute_to_degree(2, 7, "7")   →  "V7"     D7
        absolute_to_degree(5, 7, "")    →  "bVII"   F major  (common in jigs)
        absolute_to_degree(9, 7, "m")   →  "ii"     Am
    """
    # Step 1: interval relative to tonic (0-11)
    interval = (chord_root_pc - key_tonic_pc) % 12

    # Step 2: map interval → base Roman numeral
    degree_map = {
        0: "I",   1: "bII",  2: "II",   3: "bIII",
        4: "III", 5: "IV",   6: "bV",   7: "V",
        8: "bVI", 9: "VI",  10: "bVII", 11: "VII",
    }
    root_degree = degree_map[interval]

    # Step 3: lower to minor case when quality contains 'm' but not 'dim'
    q = chord_quality.strip()
    if "m" in q.lower() and "dim" not in q.lower():
        root_degree = root_degree.lower()

    # Step 4: append quality suffixes
    if "7" in q:
        root_degree += "7"
    elif "dim" in q.lower():
        root_degree += "dim"
    elif "aug" in q.lower() or "+" in q:
        root_degree += "aug"

    return root_degree


def chord_to_degree(chord_str: str, tonic_pc: int) -> str:
    """Convert an absolute chord label string to a Roman-numeral degree.

    Parses the root and quality from chord_str, then delegates to
    absolute_to_degree() for the interval arithmetic.

    Examples (key of D, tonic_pc=2):
        "A7"  →  "V7"      "Bm"  →  "vi"      "G"  →  "IV"
    """
    if not chord_str or chord_str == "N.C.":
        return chord_str

    root_match = re.match(r"^([A-G][b#]?)", chord_str)
    if not root_match:
        return chord_str

    root    = root_match.group(1)
    suffix  = chord_str[len(root):]
    root_pc = _NOTE_TO_PC.get(root)
    if root_pc is None:
        return chord_str

    return absolute_to_degree(root_pc, tonic_pc, suffix)


def map_chord_to_degree(chord_label: str, key_context) -> str:
    """
    Public API: convert an absolute chord label to its Roman-numeral degree.

    key_context accepts multiple types so callers don't need to know internals:
        int            → treated directly as tonic pitch class (0-11)
        str            → root note name, e.g. "D", "Gb"  (major assumed)
        music21.key.Key / KeySignature → tonic read from .tonic.pitchClass

    Examples in Key of G (tonic_pc=7):
        map_chord_to_degree("G",  "G")   →  "I"
        map_chord_to_degree("C",  7)     →  "IV"
        map_chord_to_degree("D7", key_G) →  "V7"   (where key_G is a music21 Key)
        map_chord_to_degree("Am", "G")   →  "ii"
    """
    if isinstance(key_context, int):
        tonic_pc = key_context
    elif isinstance(key_context, str):
        root = key_context.strip()
        root = root[0].upper() + root[1:] if root else root
        tonic_pc = _NOTE_TO_PC.get(root[:2] if len(root) > 1 and root[1] in "b#" else root[:1], 0)
    else:
        try:
            tonic_pc = key_context.tonic.pitchClass
        except AttributeError:
            tonic_pc = 0
    return chord_to_degree(chord_label, tonic_pc)


# ── Aggressive "Super-Class" collapse ────────────────────────────────────────
# Maps 33 raw degree strings → 7 functional super-classes.
# Designed for folk/Celtic music where harmonic function > exact voicing.
#
# Super-classes:
#   I    — tonic major home base
#   i    — minor / sub-tonic family  (ii, iii, vi, bii merged in)
#   V    — dominant "pull"           (V7, v, II7, bv, vii merged in)
#   IV   — subdominant departure     (IV7, iv merged in)
#   bVII — modal flat-seven          (folk / Celtic sound)
#   bIII — modal flat-three          (romantic / modern folk shifts)
#   N.C. — no chord (unchanged)
#
# Degrees absent from this map are left unchanged; they will be very rare
# (<15 tokens each) and the vocabulary builder will assign them <UNK>.
_AGGRESSIVE_MAP: dict[str, str] = {
    # Tonic major
    "I":      "I",   "I7":    "I",   "II":   "I",   "III":  "I",
    # Minor / sub-tonic family
    "i":      "i",   "ii":    "i",   "iii":  "i",   "vi":   "i",
    "bii":    "i",   "bvi":   "i",   "VI7":  "i",   "VI":   "i",
    "VIdim":  "i",   "bIIdim":"i",   "bII":  "i",
    # Dominant function
    "V":      "V",   "V7":    "V",   "v":    "V",   "bv":   "V",
    "II7":    "V",   "III7":  "V",   "vii":  "V",   "Vaug": "V",
    # Subdominant
    "IV":     "IV",  "IV7":   "IV",  "iv":   "IV",
    # Modal flat-seven
    "bVII":   "bVII", "bVII7": "bVII", "VII7": "bVII", "VIIaug": "bVII",
    # Modal flat-three
    "bIII":   "bIII", "bVI":  "bIII", "bVI7": "bIII",
    # Pass-through
    "N.C.":   "N.C.",
}


def apply_super_class(degree: str) -> str:
    """Collapse a Roman-numeral degree string to its functional super-class.

    Uses _AGGRESSIVE_MAP.  Degrees not in the map are returned unchanged
    (they will be very rare and fall through to <UNK> at inference time).

    Examples:
        apply_super_class("V7")   →  "V"
        apply_super_class("ii")   →  "i"
        apply_super_class("bVII") →  "bVII"
        apply_super_class("bvi")  →  "i"
    """
    return _AGGRESSIVE_MAP.get(degree, degree)


# Shorter alias for internal pipeline use.
map_to_degree = chord_to_degree


def degree_to_chord(degree_str: str, tonic_pc: int) -> str:
    """Convert a Roman-numeral degree string back to an absolute chord label.

    Examples (key of D, tonic_pc=2):
        "V7"  →  "A7"      "vi"  →  "Bm"      "IV"  →  "G"

    Used by inference scripts to display human-readable absolute chord names
    when the model was trained in degree mode.
    """
    if not degree_str or degree_str == "N.C.":
        return degree_str

    # Peel quality suffix (greedy: check longer suffixes first)
    quality = ""
    base = degree_str
    for sfx in ("dim", "aug", "7"):
        if base.endswith(sfx):
            quality = sfx
            base = base[: -len(sfx)]
            break

    offset = _DEGREE_TO_OFFSET.get(base)
    if offset is None:
        return degree_str   # unrecognised degree — pass through

    chord_pc   = (tonic_pc + offset) % 12
    root       = _PC_TO_NOTE[chord_pc]
    # Lowercase stem → minor quality.  Skip the 'b' flat prefix when checking.
    stem_char  = base[1] if base.startswith("b") and len(base) > 1 else base[0]
    is_minor   = stem_char.islower()

    if quality == "dim":
        return root + "dim"
    if quality == "aug":
        return root + "aug"
    if quality == "7":
        return root + ("m7" if is_minor else "7")
    return root + ("m" if is_minor else "")


# ── Chord simplification ──────────────────────────────────────────────────────

# Mapping trace: collect (raw, simplified) pairs until we hit the limit.
_TRACE_EVENTS: list[tuple[str, str]] = []
_TRACE_LIMIT = 100
_TRACE_PRINTED = False

# Relaxed mode flag — set once before training via set_relaxed_mode().
# True  (default) : collapse m7→m, maj7→root  (smaller vocabulary, more samples/class)
# False (strict)  : keep m7 and maj7 as distinct classes
_RELAXED_MODE: bool = True


def set_relaxed_mode(relaxed: bool) -> None:
    """Configure the global chord simplification mode.

    Call this before parsing any training data.

    relaxed=True  (default) — Collapse all minor extensions to m, all major
        extensions to the root.  Targets ~24–30 classes.  Recommended for
        small datasets where samples-per-class matters most.

    relaxed=False (strict)  — Keep m7 and maj7 as distinct classes; still
        collapse m9/m11 → m7 and maj9/maj11 → maj7.  Adds ~24 extra classes
        but preserves the dominant-seventh vs tonic-seventh distinction.
    """
    global _RELAXED_MODE
    _RELAXED_MODE = relaxed


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


def simplify_chord_label(chord_str: str, relaxed: bool | None = None) -> str:
    """
    Normalise a raw chord label to a compact, consistent vocabulary.

    Steps 0-4 are always applied (ABC artefact removal, enharmonic normalisation).
    Step 5 (extension collapsing) depends on the relaxed mode.

    relaxed=True  — "Relaxed" (default) — collapse to minimal vocabulary:
        m, m7, m9 …         → m          (Em7   → Em)
        maj7, add9, 6, sus4 → root only  (Dmaj7 → D)
        7, 9, 11, 13        → 7          (A9    → A7)
        dim / dim7          → dim
        aug / aug7 / +      → aug
        Preserved: Major, m, 7, dim, aug  (~24–30 classes)

    relaxed=False — "Strict" — keeps m7 and maj7 as distinct classes:
        m           → m          (plain minor unchanged)
        m7          → m7         (kept — distinct from m)
        m9, m11 …   → m7         (collapsed to m7)
        maj7        → maj7       (kept — distinct from major)
        maj9, maj11 → maj7       (collapsed to maj7)
        add9, sus4  → root only  (non-maj7 major extensions still collapse)
        Everything else unchanged vs relaxed mode.

    If relaxed is None, the module-level _RELAXED_MODE is used.
    """
    if relaxed is None:
        relaxed = _RELAXED_MODE

    if not chord_str or chord_str == "N.C.":
        return chord_str

    c = chord_str.strip()

    # 0. Strip parentheses — "(F)" and "F" are the same optional/alternative harmony
    c = re.sub(r"^\((.+)\)$", r"\1", c)
    if not c:
        return chord_str

    # 1. Strip slash-bass (take everything before the first '/')
    c = c.split("/")[0]

    # 1b. Strip ABC octave markers (commas and apostrophes) from anywhere in the
    #     chord label — e.g. "B," → "B", "A,m" → "Am", "G,7" → "G7".
    c = re.sub(r"[,']", '', c)

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

    # Empty suffix → plain major (both modes)
    if suffix == "":
        return root

    # ── Step 5: extension collapsing ─────────────────────────────────────────

    # Minor family
    if re.match(r"^m\d*$", suffix):
        if relaxed:
            return root + "m"          # Em7 → Em, Em9 → Em
        else:
            if suffix == "m":
                return root + "m"      # plain minor unchanged
            if suffix == "m7":
                return root + "m7"     # kept as distinct class
            return root + "m7"         # m9, m11, m13 → m7

    # Major extensions (maj7, add9, 6, sus4, sus2, sus, 2, 4)
    if re.match(r"^(maj\d*|add\d+|6|sus\d*|2|4)$", suffix, re.IGNORECASE):
        if relaxed:
            return root                # Dmaj7 → D, Dadd9 → D
        else:
            if re.match(r"^maj7$", suffix, re.IGNORECASE):
                return root + "maj7"   # kept as distinct class
            if re.match(r"^maj\d+$", suffix, re.IGNORECASE):
                return root + "maj7"   # maj9, maj11 → maj7
            return root                # add9, sus4, 6, 2, 4 → plain major

    # Diminished: dim, dim7, o, o7 (both modes)
    if re.match(r"^(dim7?|o7?)$", suffix, re.IGNORECASE):
        return root + "dim"

    # Augmented: aug, aug7, + (both modes)
    if re.match(r"^(aug7?|\+)$", suffix, re.IGNORECASE):
        return root + "aug"

    # Dominant 7th and higher extensions: 7, 9, 11, 13 (both modes)
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
    mode_tag = "relaxed" if _RELAXED_MODE else "strict"
    print(f"\n── Chord Mapping Trace [{mode_tag} mode] (first 100 events) ────────")
    for raw, simplified in _TRACE_EVENTS:
        arrow = "  →  " if raw != simplified else "  =  "
        print(f"    {raw:<22}{arrow}{simplified}")
    print("────────────────────────────────────────────────────────────────────\n")


def _get_key_info(score) -> tuple[int, str]:
    """Return (tonic_pitch_class 0-11, human-readable key label) for the score.

    Uses only the FIRST key object so callers that don't need per-note tracking
    still get a simple single value (backward-compatible).
    """
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


def _get_adaptive_key_timeline(score) -> list[tuple[float, int, str]]:
    """
    Adaptive Chunking for Local Key Detection.

    Segments the score based on structural boundaries (repeats, double bars, explicit keys)
    and falls back to an 8-bar grid heuristic for long sections.
    Analyzes the local key for each segment independently.

    Returns:
        list of (offset, tonic_pc, label) sorted by offset.
    """
    # Work with the first part if available to iterate measures
    if hasattr(score, "parts") and score.parts:
        part = score.parts[0]
    else:
        part = score

    # Flatten once to easily grab notes by offset range later
    flat_score = score.flatten()

    # 1. Pickup Detection & Measure Iteration
    measures = list(part.getElementsByClass(music21.stream.Measure))
    if not measures:
        # Fallback for streams without measures (e.g. raw notes)
        return _get_key_timeline_legacy(score)

    # Check for pickup measure (Measure 0 or incomplete first measure)
    first_m = measures[0]

    # Estimate full measure duration
    ts = first_m.timeSignature
    if not ts:
        ts_iter = part.flatten().getElementsByClass(music21.meter.TimeSignature)
        if ts_iter:
            ts = ts_iter[0]

    full_duration = ts.barDuration.quarterLength if ts else 4.0

    # Check if first measure is pickup
    # (shorter than full bar OR explicitly numbered 0)
    is_pickup = (first_m.duration.quarterLength < full_duration) or (first_m.measureNumber == 0)

    # 2. Identify Structural Boundaries
    # Always include 0.0 and end of score
    boundaries = {0.0, float(part.duration.quarterLength)}

    for m in measures:
        m_start = float(m.offset)
        m_end = m_start + m.duration.quarterLength

        # Explicit Key Changes inside measure
        # Use recurse() or getElementsByClass on the measure
        for k in m.getElementsByClass([music21.key.Key, music21.key.KeySignature]):
            # Key offset is relative to measure
            boundaries.add(float(m_start + k.offset))

        # Repeats & Double Bars (usually at end of measure)
        if m.rightBarline:
            # Check style or type. In music21, .style is a Style object, .type is the string.
            b_type = getattr(m.rightBarline, 'type', None)
            if b_type in ['double', 'final', 'light-light', 'light-heavy', 'heavy-light', 'heavy-heavy']:
                 boundaries.add(float(m_end))
            # Repeat barlines
            if isinstance(m.rightBarline, music21.bar.Repeat):
                 boundaries.add(float(m_end))

    sorted_bounds = sorted(list(boundaries))

    # 3. Refine Segments (The 8-Bar Grid) & Collect Final Chunks
    final_chunks = []

    for i in range(len(sorted_bounds) - 1):
        start = sorted_bounds[i]
        end = sorted_bounds[i+1]

        if start >= end:
            continue

        # Find measures in this range
        # Use a small epsilon to handle float precision
        seg_measures = [m for m in measures if m.offset >= start - 0.001 and (m.offset + m.duration.quarterLength) <= end + 0.001]

        if not seg_measures:
            # No full measures (e.g. pickup only, or mid-measure split), just keep the chunk
            final_chunks.append((start, end))
            continue

        # Filter out pickup measure from the "counting" logic if it's the very first measure
        valid_measures = [m for m in seg_measures if not (is_pickup and m is first_m)]

        if len(valid_measures) <= 12:
             # Short enough segment, keep as one
             final_chunks.append((start, end))
        else:
             # Split into ~8 bar blocks
             current_block_start = start
             count = 0
             for m in valid_measures:
                 count += 1
                 if count >= 8:
                     # Split point at end of this measure
                     split_point = float(m.offset + m.duration.quarterLength)
                     final_chunks.append((current_block_start, split_point))
                     current_block_start = split_point
                     count = 0
             # Add remainder
             if current_block_start < end:
                 final_chunks.append((current_block_start, end))

    # 4. Analyze Key for each Chunk
    results = []

    # Pre-fetch all notes to avoid repeated iteration if possible,
    # but flat_score.notes is a list/iterator.
    all_notes = list(flat_score.notes)

    for (start, end) in final_chunks:
        # Extract notes in this time range
        chunk_notes = [n for n in all_notes if n.offset >= start - 0.001 and n.offset < end - 0.001]

        if not chunk_notes:
            # Inherit previous key or default
            if results:
                k_prev = results[-1]
                results.append((start, k_prev[1], k_prev[2]))
            else:
                # Default
                results.append((start, 2, "D major (default)"))
            continue

        # Analyze
        s_temp = music21.stream.Stream()
        for n in chunk_notes:
            s_temp.append(n)

        k = s_temp.analyze('key')
        results.append((start, k.tonic.pitchClass, f"{k.tonic.name} {k.mode}"))

    # 5. Merge adjacent same keys
    if not results:
        return _get_key_timeline_legacy(score)

    merged_timeline = []
    curr_start, curr_pc, curr_lbl = results[0]

    for i in range(1, len(results)):
        next_start, next_pc, next_lbl = results[i]
        if next_pc == curr_pc:
            # Same key, extend current chunk
            continue
        else:
            # New key
            merged_timeline.append((curr_start, curr_pc, curr_lbl))
            curr_start, curr_pc, curr_lbl = next_start, next_pc, next_lbl
    merged_timeline.append((curr_start, curr_pc, curr_lbl))

    return merged_timeline

def _get_key_timeline_legacy(score) -> list[tuple[float, int, str]]:
    """Return [(offset, tonic_pc, label), …] for every key change in the score.

    When patch_key_changes.py has injected an inline [K:X] at a section
    boundary, music21 places an additional Key object at the corresponding
    offset.  Sorting by offset lets _active_key_at() walk the timeline and
    return the correct local key for any given note offset.

    Falls back to the initial key / default when no Key objects exist.
    """
    flat     = score.flatten()
    key_objs = list(flat.getElementsByClass(music21.key.Key))
    if key_objs:
        return [
            (float(k.offset), k.tonic.pitchClass, f"{k.tonic.name} {k.mode}")
            for k in sorted(key_objs, key=lambda k: k.offset)
        ]
    # No Key objects: try KeySignature and fall back to default
    key_sigs = list(flat.getElementsByClass(music21.key.KeySignature))
    if key_sigs:
        try:
            k = key_sigs[0].asKey()
            return [(0.0, k.tonic.pitchClass, f"{k.tonic.name} {k.mode}")]
        except Exception:
            pass
    return [(0.0, 2, "D major (default)")]

def _get_key_timeline(score):
    return _get_adaptive_key_timeline(score)


def _active_key_at(
    timeline: list[tuple[float, int, str]],
    offset: float,
) -> tuple[int, str]:
    """Return (tonic_pc, label) for the most recent key change at or before offset.

    Walks the pre-sorted timeline linearly; stops as soon as a future key is
    encountered so the last valid entry is returned.
    """
    tpc, label = timeline[0][1], timeline[0][2]
    for t, p, lbl in timeline:
        if t <= offset:
            tpc, label = p, lbl
        else:
            break
    return tpc, label


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

    # Build the full key timeline for this score.  If patch_key_changes.py has
    # injected an inline [K:X] at a section boundary, music21 will have placed
    # an additional Key object at that offset, and notes in the B-part will
    # automatically receive the correct local tonic.
    key_timeline = _get_key_timeline(score)
    meter_num    = _get_meter_numerator(score)
    meter_norm   = min(meter_num, _METER_NUM_MAX) / _METER_NUM_MAX

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
            raw_chord    = current_best_chord.figure
            active_chord = simplify_chord_label(raw_chord, relaxed=_RELAXED_MODE)
            # Accumulate mapping trace (raw → simplified) until limit reached
            if len(_TRACE_EVENTS) < _TRACE_LIMIT:
                _TRACE_EVENTS.append((raw_chord, active_chord))

        if n.isNote:
            # Use the locally-active key at this note's offset so that
            # B-part notes after an injected [K:X] use the new tonic.
            tonic_pc, key_label = _active_key_at(key_timeline, float(n.offset))
            scale_degree = (n.pitch.pitchClass - tonic_pc) % 12  # 0-11
            dataset.append({
                "pitch":        n.pitch.ps,
                "duration":     n.duration.quarterLength,
                "beat":         n.beat,
                "measure":      n.measureNumber,
                "is_rest":      0,
                "scale_degree": scale_degree,
                "meter_norm":   meter_norm,
                "key_tonic_pc": tonic_pc,    # per-note local tonic for degree mode
                "key_label":    key_label,   # human-readable key for display
                "target_chord": active_chord,
            })
    _print_scale_degree_debug(dataset, key_timeline[0][2])
    return dataset


def _sanitise_abc_body(abc_str: str) -> str:
    """
    Strip non-pitch annotation characters that music21 misreads as notes.

    Removed from non-header lines only (header lines like K:, M:, T: are
    left untouched). Quoted chord symbols (e.g. "Caug") are also protected.

    Stripped:
      x\\d*  — invisible rests     (x2, x)
      y\\d*  — playback spacers    (y4, y)
      J      — slide ornament (non-standard)
      S      — segno marker (non-standard in some ABC dialects)
      H      — fermata / hold ornament
      L      — accent ornament
      u      — up-bow decoration
      v      — down-bow decoration
    """
    out = []
    for line in abc_str.splitlines():
        if re.match(r'^[A-Za-z]:', line):
            out.append(line)            # header line — leave unchanged
            continue
        # Remove invisible rests and spacers (with optional length digit)
        line = re.sub(r'[xy]\d*', '', line)
        # Remove decoration chars from unquoted segments only
        # (splits on "..." to protect chord symbols like "Caug")
        parts = re.split(r'(".*?")', line)
        clean_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 0:              # outside quotes
                part = re.sub(r'[JSHLuv]', '', part)
            clean_parts.append(part)
        out.append(''.join(clean_parts))
    return '\n'.join(out)


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
    score = music21.converter.parse(_sanitise_abc_body(tunes[0]), format="abc")
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
            score = music21.converter.parse(_sanitise_abc_body(tune_str), format="abc")
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
