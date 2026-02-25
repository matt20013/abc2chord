#!/usr/bin/env python3
"""
scripts/audit_data.py — data quality audit for the abc2chord training corpus.

For every tune in the training files:
  1. KEY AUDIT     Compare the declared K: header key against pitch-content
                   analysis (Krumhansl-Schmuckler).  Flags conflicts to
                   reports/key_conflicts.csv.
  2. CHORD OOV     Collect every simplified chord label and flag any that fall
                   outside the expected ~35-class vocabulary to
                   reports/chord_oov.csv.

Usage (from project root):
    python scripts/audit_data.py
    python scripts/audit_data.py --abc-paths path/to/file.abc ...
    python scripts/audit_data.py --quick 50   # first 50 tunes only
"""
import os
import sys
import re
import csv
import argparse
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import warnings
warnings.filterwarnings("ignore")

import music21
import music21.pitch
import music21.key
import music21.harmony

from src.parser import (
    _split_abc_file,
    _sanitise_abc_body,
    _get_key_info,
    simplify_chord_label,
    _NOTE_TO_PC,
    _PC_TO_NOTE,
)
from src.training_data import TRAINING_ABC_FILES, _resolve_abc_paths

REPORTS_DIR = os.path.join(ROOT, "reports")

# ── Expected clean vocabulary (post simplify_chord_label) ────────────────────
_ROOTS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_EXPECTED_CHORDS = (
    {r for r in _ROOTS}               # plain major
    | {r + "m"   for r in _ROOTS}     # minor
    | {r + "7"   for r in _ROOTS}     # dominant 7
    | {r + "dim" for r in _ROOTS}     # diminished
    | {r + "aug" for r in _ROOTS}     # augmented
    | {"N.C."}
)

# Semitone differences and their common musical interpretations.
_DELTA_LABELS = {
    0: "MATCH",
    1: "SEMITONE",  2: "TONE",
    3: "RELATIVE",  9: "RELATIVE",     # relative major / minor
    4: "MEDIANT",   8: "MEDIANT",
    5: "SUBDOMINANT/MODAL", 7: "MODAL",# 5th: e.g. D→G (Dmixolydian reads as G major)
    6: "TRITONE",
    10: "SUBTONE",  11: "LEADING_TONE",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_raw_header_key(abc_str: str) -> tuple[str | None, str]:
    """Return (normalized_root, full_raw_K_value) from an ABC tune string.

    Root is title-cased and sharp-roots are converted to flat equivalents so
    comparisons are consistent.  Returns (None, 'missing') if no K: field.
    """
    _sharp_to_flat = {"A#": "Bb", "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab"}
    for line in abc_str.splitlines():
        m = re.match(r"^K:\s*(.+)", line.strip(), re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            root_m = re.match(r"^([A-Ga-g][b#]?)", raw)
            if root_m:
                root = root_m.group(1)
                root = root[0].upper() + root[1:]
                root = _sharp_to_flat.get(root, root)
                return root, raw
            if raw.lower() in ("none", "hp", "hpipe"):
                return "C", raw
    return None, "missing"


def _get_tune_meta(abc_str: str) -> tuple[str, str]:
    """Return (tune_id, title) from ABC header fields X: and T:."""
    tune_id = "?"
    title   = "Unknown"
    for line in abc_str.splitlines():
        if re.match(r"^X:\s*\d", line):
            tune_id = line.split(":", 1)[1].strip()
        elif re.match(r"^T:", line) and title == "Unknown":
            title = line.split(":", 1)[1].strip()
    return tune_id, title


def _classify_delta(delta: int) -> str:
    return _DELTA_LABELS.get(delta % 12, "CONFLICT")


def _extract_raw_chords(score) -> list[str]:
    """Return all raw ChordSymbol figure strings from a parsed score."""
    chords = list(score.flatten().getElementsByClass(music21.harmony.ChordSymbol))
    if not chords:
        chords = list(
            score.chordify()
                 .flatten()
                 .getElementsByClass(music21.harmony.ChordSymbol)
        )
    return [c.figure for c in chords]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ABC training data for key conflicts and OOV chords.")
    parser.add_argument("--abc-paths", nargs="*", default=None,
                        help="ABC files to audit (default: full training corpus).")
    parser.add_argument("--quick", type=int, default=None, metavar="N",
                        help="Stop after N tunes per file (for a fast preview).")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Only report conflicts where analyze('key') confidence "
                             ">= this value (0–1).  Default 0 = report all.")
    args = parser.parse_args()

    abc_paths = _resolve_abc_paths(args.abc_paths)
    abc_paths = [p for p in abc_paths if os.path.isfile(p)]
    if not abc_paths:
        print("No ABC files found.  Pass --abc-paths or check TRAINING_ABC_FILES.",
              file=sys.stderr)
        return 1

    os.makedirs(REPORTS_DIR, exist_ok=True)
    conflicts_path = os.path.join(REPORTS_DIR, "key_conflicts.csv")
    oov_path       = os.path.join(REPORTS_DIR, "chord_oov.csv")

    conflict_rows: list[dict] = []
    oov_rows:      list[dict] = []

    # Global tallies for the summary block
    total_tunes    = 0
    total_chords   = 0
    oov_counter:   Counter = Counter()       # simplified_chord → count
    chord_counter: Counter = Counter()       # simplified_chord → count (all)

    for abc_path in abc_paths:
        fname = os.path.basename(abc_path)
        print(f"\n{'─'*60}")
        print(f"File: {fname}")

        with open(abc_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        tune_strings = _split_abc_file(content)
        if not tune_strings:
            tune_strings = [content]

        if args.quick:
            tune_strings = tune_strings[: args.quick]

        file_conflicts = 0
        file_oov       = 0

        for tune_str in tune_strings:
            total_tunes += 1
            tune_id, title = _get_tune_meta(tune_str)
            header_root, raw_key_str = _parse_raw_header_key(tune_str)

            # ── Parse score ──────────────────────────────────────────────────
            try:
                score = music21.converter.parse(
                    _sanitise_abc_body(tune_str), format="abc"
                )
                if hasattr(score, "scores") and score.scores:
                    score = score.scores[0]
            except Exception as exc:
                print(f"  [{tune_id}] parse error: {exc}")
                continue

            # ── Key audit ────────────────────────────────────────────────────
            stated_tpc, stated_label = _get_key_info(score)
            stated_root = _PC_TO_NOTE[stated_tpc]

            try:
                analyzed_key  = score.analyze("key")
                analyzed_tpc  = analyzed_key.tonic.pitchClass
                analyzed_root = _PC_TO_NOTE[analyzed_tpc]
                analyzed_mode = analyzed_key.mode
                analyzed_conf = round(float(analyzed_key.correlationCoefficient), 3)
            except Exception:
                analyzed_tpc  = stated_tpc
                analyzed_root = stated_root
                analyzed_mode = "unknown"
                analyzed_conf = 0.0

            delta     = (analyzed_tpc - stated_tpc) % 12
            relation  = _classify_delta(delta)
            is_match  = (delta == 0)

            if not is_match and analyzed_conf >= args.min_confidence:
                file_conflicts += 1
                conflict_rows.append({
                    "file":               fname,
                    "tune_id":            tune_id,
                    "title":              title,
                    "raw_key_header":     raw_key_str,
                    "stated_tonic":       stated_root,
                    "analyzed_tonic":     analyzed_root,
                    "analyzed_mode":      analyzed_mode,
                    "analyzed_confidence": analyzed_conf,
                    "delta_semitones":    delta,
                    "relation":           relation,
                })

            # ── Chord OOV check ──────────────────────────────────────────────
            raw_chords = _extract_raw_chords(score)
            tune_oov: Counter = Counter()

            for raw in raw_chords:
                simplified = simplify_chord_label(raw)
                chord_counter[simplified] += 1
                total_chords += 1
                if simplified not in _EXPECTED_CHORDS:
                    tune_oov[simplified] += 1
                    oov_counter[simplified] += 1

            for chord, count in tune_oov.items():
                file_oov += 1
                oov_rows.append({
                    "file":             fname,
                    "tune_id":          tune_id,
                    "title":            title,
                    "simplified_chord": chord,
                    "count_in_tune":    count,
                })

        print(f"  Tunes: {len(tune_strings):>4}  |  "
              f"Key conflicts: {file_conflicts:>3}  |  "
              f"OOV chord types: {file_oov:>3}")

    # ── Write reports ─────────────────────────────────────────────────────────
    conflict_fields = [
        "file", "tune_id", "title", "raw_key_header",
        "stated_tonic", "analyzed_tonic", "analyzed_mode",
        "analyzed_confidence", "delta_semitones", "relation",
    ]
    with open(conflicts_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=conflict_fields)
        w.writeheader()
        w.writerows(conflict_rows)

    oov_fields = ["file", "tune_id", "title", "simplified_chord", "count_in_tune"]
    with open(oov_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=oov_fields)
        w.writeheader()
        w.writerows(oov_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"AUDIT SUMMARY")
    print(f"{'═'*60}")
    print(f"  Tunes audited      : {total_tunes}")
    print(f"  Key conflicts      : {len(conflict_rows)}  "
          f"({len(conflict_rows)/max(total_tunes,1):.1%} of tunes)")

    if conflict_rows:
        from collections import Counter as C
        rel_counts = C(r["relation"] for r in conflict_rows)
        for rel, n in rel_counts.most_common():
            print(f"    {rel:<22}: {n}")

    print(f"  Chord tokens total : {total_chords}")
    print(f"  OOV chord types    : {len(oov_counter)}")
    if oov_counter:
        print("  Top OOV chords:")
        for chord, n in oov_counter.most_common(10):
            print(f"    {chord:<20} {n:>5}x")

    print(f"\n  Vocab coverage (top 20 simplified chords):")
    for chord, n in chord_counter.most_common(20):
        in_vocab = "✓" if chord in _EXPECTED_CHORDS else "✗ OOV"
        print(f"    {chord:<16} {n:>6}  {in_vocab}")

    print(f"\n  Reports written to:")
    print(f"    {conflicts_path}")
    print(f"    {oov_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
