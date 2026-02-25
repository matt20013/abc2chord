#!/usr/bin/env python3
"""
scripts/patch_key_changes.py — inject inline key changes at B-part boundaries.

Reads reports/key_conflicts.csv and, for tunes where:
  • relation is MODAL, RELATIVE, or SUBTONE  (configurable via --relations)
  • analyzed_confidence ≥ --min-confidence   (default 0.85)

…injects a single inline [K:X] token at the first end-of-repeat (:|) marker
in the tune body.  This tells music21 (and therefore src/parser.py) to use the
analysed key for all notes in the B-part, which standardises V→I degree labels
across local key changes.

Safety guarantees
-----------------
• Dry-run is the DEFAULT.  Pass --apply to actually write changes.
• A .bak copy of every modified file is written before the first change.
• Already-injected positions are detected and skipped (idempotent).
• Tunes with no :| repeat marker are skipped (through-composed tunes).
• Tunes whose K: header already contains an explicit mode tag (Dor, Mix, Phr,
  Lyd) are skipped — they already carry the correct key context.

Usage
-----
    python scripts/patch_key_changes.py                    # dry-run
    python scripts/patch_key_changes.py --apply            # write changes
    python scripts/patch_key_changes.py --apply --min-confidence 0.90
    python scripts/patch_key_changes.py --relations MODAL  # only MODAL rows
"""

import argparse
import csv
import os
import re
import shutil
import sys

# ---------------------------------------------------------------------------
# File-path resolution (mirrors TRAINING_ABC_FILES / DATA_ABC_FILES)
# ---------------------------------------------------------------------------
_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(_HERE)
_TUNES = os.path.join(_ROOT, "..", "tunes", "abcs")
_DATA  = os.path.join(_ROOT, "data")
_SEARCH_DIRS = [_TUNES, _DATA]

_ELIGIBLE_RELATIONS = {"MODAL", "RELATIVE", "SUBTONE"}

# Mode suffixes that, when present in the K: header, mean the key is already
# explicitly specified — no injection needed.
_EXPLICIT_MODE_RE = re.compile(r"(dor|mix|phr|lyd|loc|ion|aeo)", re.IGNORECASE)


def _find_abc_file(basename: str) -> str | None:
    for d in _SEARCH_DIRS:
        p = os.path.join(d, basename)
        if os.path.isfile(p):
            return p
    return None


# ---------------------------------------------------------------------------
# ABC tune splitting / header-body utilities
# ---------------------------------------------------------------------------

def _split_abc_file(content: str) -> list[str]:
    """Split multi-tune ABC content into individual tune strings."""
    chunks = re.split(r"\n(?=X:\s*\d)", content.strip(), flags=re.IGNORECASE)
    return [
        c.strip() for c in chunks
        if c.strip() and re.match(r"X:\s*\d", c.strip(), re.IGNORECASE)
    ]


def _get_tune_id(tune_str: str) -> int | None:
    m = re.match(r"X:\s*(\d+)", tune_str.strip())
    return int(m.group(1)) if m else None


def _get_raw_key_header(tune_str: str) -> str:
    """Return the raw value of the K: field (last occurrence)."""
    last = ""
    for line in tune_str.splitlines():
        if re.match(r"^K:", line, re.IGNORECASE):
            last = line[2:].strip()
    return last


def _split_at_last_k(tune_str: str) -> tuple[str, str]:
    """Return (header_through_last_K_line, body_after_K)."""
    lines = tune_str.splitlines()
    last_k = None
    for i, line in enumerate(lines):
        if re.match(r"^K:", line, re.IGNORECASE):
            last_k = i
    if last_k is None:
        return tune_str, ""
    return "\n".join(lines[: last_k + 1]), "\n".join(lines[last_k + 1 :])


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

def _abc_key_token(tonic: str, mode: str) -> str:
    """
    Build the ABC inline key token string (without the surrounding [K:]).
    Examples:
        ("G",  "major") → "Gmaj"
        ("E",  "minor") → "Em"
        ("Gb", "minor") → "Gbm"
    """
    t = tonic.strip().capitalize()
    if mode.lower().startswith("min"):
        return f"{t}m"
    return f"{t}maj"


def _find_b_part_pos(body: str) -> int:
    """
    Return the character position just AFTER the first :| end-of-repeat marker.
    Falls back to the position between the two colons of :: (double repeat bar).
    Returns -1 if no repeat marker is found (through-composed tune).
    """
    m = re.search(r":\|", body)
    if m:
        return m.end()
    m = re.search(r"::", body)
    if m:
        return m.start() + 1   # after the first ':'
    return -1


def _already_injected(body: str, pos: int) -> bool:
    """Return True if an [K:…] token immediately follows pos (idempotency check)."""
    return bool(re.match(r"\s*\[K:", body[pos : pos + 20]))


def _inject_key_change(
    tune_str: str, tonic: str, mode: str
) -> tuple[str | None, str]:
    """
    Attempt to inject [K:X] at the B-part boundary of tune_str.

    Returns:
        (patched_tune_str, message)   on success
        (None, reason)                if injection is impossible or unnecessary
    """
    raw_k = _get_raw_key_header(tune_str)
    if _EXPLICIT_MODE_RE.search(raw_k):
        return None, f"K: header already specifies mode ({raw_k!r}) — skipping"

    header, body = _split_at_last_k(tune_str)
    if not body.strip():
        return None, "no body after K: field"

    pos = _find_b_part_pos(body)
    if pos == -1:
        return None, "no :| repeat marker found (through-composed?)"

    if _already_injected(body, pos):
        return None, "inline [K:…] already present at this position (idempotent)"

    token = _abc_key_token(tonic, mode)
    new_body = body[:pos] + f"[K:{token}]" + body[pos:]
    patched  = header + "\n" + new_body
    return patched, f"injected [K:{token}] after char {pos}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="Write changes to disk.  Default is dry-run (no writes).",
    )
    ap.add_argument(
        "--out-dir", metavar="DIR", default=None,
        help=(
            "Write patched copies to DIR/<basename> instead of modifying files "
            "in-place.  Unmodified source files are also copied so DIR contains "
            "a complete, self-contained training corpus.  Originals are never "
            "touched.  Requires --apply."
        ),
    )
    ap.add_argument(
        "--min-confidence", type=float, default=0.85, metavar="F",
        help="Minimum analyzed_confidence threshold (default 0.85).",
    )
    ap.add_argument(
        "--csv",
        default=os.path.join(_ROOT, "reports", "key_conflicts.csv"),
        help="Path to key_conflicts.csv (default: reports/key_conflicts.csv).",
    )
    ap.add_argument(
        "--relations", nargs="+", default=list(_ELIGIBLE_RELATIONS),
        metavar="REL",
        help=f"Relations to process (default: {' '.join(_ELIGIBLE_RELATIONS)}).",
    )
    args = ap.parse_args()

    eligible  = set(args.relations)
    dry_run   = not args.apply
    out_dir   = args.out_dir

    if out_dir and not dry_run:
        os.makedirs(out_dir, exist_ok=True)

    if dry_run:
        print("── DRY-RUN ─────────────────────────────────────────────────────────")
        msg = (f"   Patched copies would go to: {out_dir}"
               if out_dir else "   No files will be modified.")
        print(msg)
        print("   Pass --apply to write changes.\n")

    if not os.path.isfile(args.csv):
        sys.exit(f"ERROR: {args.csv} not found.  Run scripts/audit_data.py first.")

    # ------------------------------------------------------------------
    # Parse CSV → group qualifying patches by file
    # ------------------------------------------------------------------
    # basename → {tune_id → (title, tonic, mode, confidence, relation)}
    patches: dict[str, dict[int, tuple]] = {}

    with open(args.csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            relation   = row.get("relation", "").strip()
            confidence = float(row.get("analyzed_confidence", 0))
            if relation not in eligible:
                continue
            if confidence < args.min_confidence:
                continue
            basename = row["file"].strip()
            tune_id  = int(row["tune_id"])
            patches.setdefault(basename, {})[tune_id] = (
                row.get("title", "").strip(),
                row["analyzed_tonic"].strip(),
                row["analyzed_mode"].strip(),
                confidence,
                relation,
            )

    if not patches:
        print("No qualifying rows found.  Try --min-confidence 0.80 or "
              "--relations RELATIVE MODAL SUBTONE")
        return

    total_eligible = sum(len(v) for v in patches.values())
    total_injected = 0
    total_skipped  = 0
    backed_up: set[str] = set()

    # ------------------------------------------------------------------
    # Process file by file
    # ------------------------------------------------------------------
    for basename, tune_patches in sorted(patches.items()):
        abc_path = _find_abc_file(basename)
        if abc_path is None:
            print(f"  SKIP  {basename}  — file not found in {_SEARCH_DIRS}")
            total_skipped += len(tune_patches)
            continue

        with open(abc_path, encoding="utf-8", errors="replace") as fh:
            original_content = fh.read()

        tune_map: dict[int, str] = {
            _get_tune_id(t): t
            for t in _split_abc_file(original_content)
            if _get_tune_id(t) is not None
        }

        modified_content = original_content
        file_changes = 0

        for tune_id, (title, tonic, mode, conf, relation) in sorted(tune_patches.items()):
            if tune_id not in tune_map:
                print(f"  MISS  {basename} X:{tune_id} \"{title}\"  "
                      f"— tune not found in file")
                total_skipped += 1
                continue

            original_tune = tune_map[tune_id]
            patched_tune, note = _inject_key_change(original_tune, tonic, mode)

            token = _abc_key_token(tonic, mode)
            action = "WOULD" if dry_run else "✓"

            if patched_tune is None:
                print(f"  SKIP  {basename} X:{tune_id} \"{title}\"  — {note}")
                total_skipped += 1
                continue

            print(f"  {action}  {basename} X:{tune_id} \"{title}\"  "
                  f"conf={conf:.3f} rel={relation}  → [K:{token}]")
            print(f"         ({note})")

            if not dry_run:
                if original_tune in modified_content:
                    modified_content = modified_content.replace(
                        original_tune, patched_tune, 1
                    )
                    file_changes += 1
                    total_injected += 1
                else:
                    print(f"         WARNING: could not locate exact tune text "
                          f"for replacement — skipped")
                    total_skipped += 1
            else:
                total_injected += 1   # count as "would inject"

        if file_changes > 0 and not dry_run:
            if out_dir:
                # Write patched copy to out_dir; original is untouched.
                dest = os.path.join(out_dir, basename)
                with open(dest, "w", encoding="utf-8") as fh:
                    fh.write(modified_content)
                print(f"  WROTE   {dest}  ({file_changes} tune(s) patched)\n")
            else:
                # In-place write with .bak backup.
                bak_path = abc_path + ".bak"
                if abc_path not in backed_up:
                    shutil.copy2(abc_path, bak_path)
                    backed_up.add(abc_path)
                    print(f"\n  BACKUP  {abc_path}")
                    print(f"       →  {bak_path}")
                with open(abc_path, "w", encoding="utf-8") as fh:
                    fh.write(modified_content)
                print(f"  WROTE   {abc_path}  ({file_changes} tune(s) modified)\n")
        elif file_changes > 0:
            print()

    # ------------------------------------------------------------------
    # Copy unmodified source files into out_dir so it is self-contained
    # ------------------------------------------------------------------
    if out_dir and not dry_run:
        patched_basenames = set(patches.keys())
        all_sources: list[str] = []
        for d in _SEARCH_DIRS:
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn.endswith(".abc"):
                        all_sources.append(os.path.join(d, fn))

        copied_unmodified: list[str] = []
        for src in all_sources:
            bn = os.path.basename(src)
            dest = os.path.join(out_dir, bn)
            if bn not in patched_basenames and not os.path.exists(dest):
                shutil.copy2(src, dest)
                copied_unmodified.append(bn)

        if copied_unmodified:
            print(f"  COPIED (unmodified) → {out_dir}:")
            for fn in sorted(copied_unmodified):
                print(f"    {fn}")
            print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    tag = " [DRY-RUN]" if dry_run else ""
    print(f"\n── Summary{tag} {'─' * (55 - len(tag))}")
    print(f"  Eligible rows : {total_eligible}")
    verb = "Would inject" if dry_run else "Injected"
    print(f"  {verb:<13} : {total_injected}")
    print(f"  Skipped       : {total_skipped}")
    if backed_up:
        print(f"  Backups       : {len(backed_up)} file(s)")
    if dry_run:
        print(f"\n  Re-run with --apply to write changes to disk.")

    # ------------------------------------------------------------------
    # Print suggested training command
    # ------------------------------------------------------------------
    if out_dir and not dry_run:
        abc_files = sorted(
            os.path.join(out_dir, f)
            for f in os.listdir(out_dir)
            if f.endswith(".abc")
        )
        paths_str = " \\\n    ".join(abc_files)
        print(f"\n── Suggested training command ─────────────────────────────────────")
        print(f"  python scripts/train_lstm.py \\")
        print(f"    --target-type degree \\")
        print(f"    --epochs 50 \\")
        print(f"    --abc-paths \\")
        print(f"    {paths_str}")


if __name__ == "__main__":
    main()
