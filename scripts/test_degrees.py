#!/usr/bin/env python3
"""
scripts/test_degrees.py — visual validation of Roman Numeral degree mapping.

Loads sample tunes from the training corpus (one per key where possible) and
prints a mapping table:

    [Original Key]  |  [Chord]  (pc=N)  →  interval  →  [Degree]

The key goal: confirm that the same V7→I perfect cadence maps identically
across every key (D7→G, A7→D, E7→A, etc. all become V7→I).

Usage:
    python scripts/test_degrees.py
    python scripts/test_degrees.py --n-tunes 10   # how many tunes to sample
    python scripts/test_degrees.py --key G         # show only tunes in this key
"""
import os
import sys
import argparse
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import warnings
warnings.filterwarnings("ignore")

from src.parser import (
    _iter_scores_from_abc,
    _extract_features_from_score,
    _get_key_info,
    absolute_to_degree,
    chord_to_degree,
    simplify_chord_label,
    _NOTE_TO_PC,
    _PC_TO_NOTE,
)
from src.training_data import TRAINING_ABC_FILES, _resolve_abc_paths, tune_has_chords

# ── ANSI colours ──────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELL  = "\033[93m"
RED   = "\033[91m"
DIM   = "\033[2m"
RESET = "\033[0m"

# Highlight these degrees as "musically important"
_CADENTIAL = {"V7", "V", "IV", "I", "i"}


def _get_title(score) -> str:
    try:
        md = score.metadata
        if md and md.title:
            return md.title
    except Exception:
        pass
    return "Unknown"


def _parse_chord_components(chord_str: str) -> tuple[int | None, str]:
    """Return (root_pc, quality_suffix) or (None, '') for unknown chords."""
    import re
    m = re.match(r"^([A-G][b#]?)", chord_str)
    if not m:
        return None, ""
    root = m.group(1)
    return _NOTE_TO_PC.get(root), chord_str[len(root):]


def print_tune_table(score, tonic_pc: int, key_label: str,
                     target_key: str | None = None) -> list[tuple[str, str]]:
    """
    Print the chord→degree mapping table for one tune.
    Returns list of (absolute_chord, degree) pairs for cadence analysis.
    """
    tune = _extract_features_from_score(score)
    if not tune_has_chords(tune):
        return []

    # Deduplicate while preserving encounter order
    seen: dict[str, str] = {}
    for row in tune:
        chord = row["target_chord"]
        if chord == "N.C." or chord in seen:
            continue
        degree = chord_to_degree(chord, tonic_pc)
        seen[chord] = degree

    if not seen:
        return []

    title = _get_title(score)
    tonic_name = _PC_TO_NOTE[tonic_pc]

    print(f"\n{BOLD}── {title}  [{CYAN}{key_label}{RESET}{BOLD}]{RESET}")
    print(f"   {'Chord':<8}  {'Root PC':>7}  {'Interval':>10}  {'Degree':<10}  formula")
    print(f"   {'─'*8}  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*20}")

    pairs = []
    for chord, degree in seen.items():
        root_pc, quality = _parse_chord_components(chord)
        if root_pc is None:
            continue
        interval = (root_pc - tonic_pc) % 12

        # Colour coding
        if degree in ("I", "i"):
            deg_col = f"{GREEN}{BOLD}{degree:<10}{RESET}"
        elif degree in _CADENTIAL:
            deg_col = f"{YELL}{degree:<10}{RESET}"
        else:
            deg_col = f"{degree:<10}"

        print(f"   {chord:<8}  {root_pc:>7}  {interval:>10}  {deg_col}  "
              f"({root_pc} - {tonic_pc}) % 12 = {interval}")
        pairs.append((chord, degree))

    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Roman Numeral degree mappings on real training tunes.")
    parser.add_argument("--abc-paths", nargs="*", default=None)
    parser.add_argument("--n-tunes",   type=int, default=8,
                        help="Number of tunes to sample (default: 8)")
    parser.add_argument("--key",       type=str, default=None,
                        help="Filter to tunes in this key only, e.g. --key G")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    abc_paths = _resolve_abc_paths(args.abc_paths)
    abc_paths = [p for p in abc_paths if os.path.isfile(p)]
    if not abc_paths:
        print("No ABC files found.", file=sys.stderr)
        return 1

    # ── Load scores and group by key ─────────────────────────────────────────
    print("Loading tunes…", end=" ", flush=True)
    by_key: defaultdict[int, list] = defaultdict(list)
    for path in abc_paths:
        try:
            for score in _iter_scores_from_abc(path):
                ds = _extract_features_from_score(score)
                if not tune_has_chords(ds):
                    continue
                tpc, label = _get_key_info(score)
                by_key[tpc].append((score, tpc, label))
        except Exception:
            continue

    total = sum(len(v) for v in by_key.values())
    print(f"{total} chord-annotated tunes found across {len(by_key)} keys.")

    # ── Select sample tunes ───────────────────────────────────────────────────
    rng = random.Random(args.seed)
    sample: list[tuple] = []

    if args.key:
        # Filter to requested key
        filter_tpc = _NOTE_TO_PC.get(args.key[:2] if len(args.key) > 1
                                     and args.key[1] in "b#" else args.key[:1])
        pool = by_key.get(filter_tpc, [])
        sample = rng.sample(pool, min(args.n_tunes, len(pool)))
    else:
        # Pick one tune per key (ordered by key frequency, most common first)
        sorted_keys = sorted(by_key, key=lambda k: -len(by_key[k]))
        for tpc in sorted_keys:
            if len(sample) >= args.n_tunes:
                break
            entry = rng.choice(by_key[tpc])
            sample.append(entry)

    if not sample:
        print("No matching tunes found.", file=sys.stderr)
        return 1

    # ── Print tables ──────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═'*68}{RESET}")
    print(f"{BOLD}  DEGREE MAPPING VALIDATION  "
          f"({len(sample)} tunes){RESET}")
    print(f"{BOLD}{'═'*68}{RESET}")

    all_pairs: list[tuple[str, str, str]] = []   # (absolute, degree, key_label)
    for score, tpc, key_label in sample:
        pairs = print_tune_table(score, tpc, key_label)
        for abs_chord, degree in pairs:
            all_pairs.append((abs_chord, degree, key_label))

    # ── Perfect Cadence Verification ─────────────────────────────────────────
    print(f"\n{BOLD}{'═'*68}{RESET}")
    print(f"{BOLD}  PERFECT CADENCE  VERIFICATION  (V7 → I){RESET}")
    print(f"{BOLD}{'═'*68}{RESET}")
    print(f"  {'Key':<18}  {'Absolute V7':<12}  {'Maps to':<8}  {'Absolute I':<12}  {'Maps to'}")
    print(f"  {'─'*18}  {'─'*12}  {'─'*8}  {'─'*12}  {'─'*8}")

    verified = 0
    for tpc in sorted(by_key.keys()):
        tonic_name = _PC_TO_NOTE[tpc]
        # Dominant 7 root is 7 semitones above tonic
        v7_pc    = (tpc + 7) % 12
        v7_root  = _PC_TO_NOTE[v7_pc]
        v7_chord = f"{v7_root}7"
        v7_degree = absolute_to_degree(v7_pc, tpc, "7")

        # Tonic chord
        i_chord  = tonic_name
        i_degree = absolute_to_degree(tpc, tpc, "")

        ok = (v7_degree == "V7" and i_degree == "I")
        mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        if ok:
            verified += 1
        print(f"  {tonic_name + ' major':<18}  {v7_chord:<12}  "
              f"{YELL}{v7_degree:<8}{RESET}  {i_chord:<12}  "
              f"{GREEN}{i_degree:<8}{RESET}  {mark}")

    print(f"\n  {GREEN}{BOLD}{verified}/12{RESET} keys: V7→I maps "
          f"{'universally ✓' if verified == 12 else 'with errors ✗'}")

    # ── Degree vocabulary discovered ─────────────────────────────────────────
    from collections import Counter
    degree_counts = Counter(deg for _, deg, _ in all_pairs if deg != "N.C.")
    print(f"\n{BOLD}{'═'*68}{RESET}")
    print(f"{BOLD}  DEGREE VOCABULARY  (from {len(all_pairs)} chord tokens){RESET}")
    print(f"{BOLD}{'═'*68}{RESET}")
    for deg, n in degree_counts.most_common():
        bar = "█" * min(40, int(40 * n / degree_counts.most_common(1)[0][1]))
        col = GREEN if deg in ("I", "i") else YELL if deg in _CADENTIAL else ""
        print(f"  {col}{deg:<10}{RESET}  {n:>5}  {bar}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
