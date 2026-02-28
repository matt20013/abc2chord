# ── Pitch-class lookup tables ─────────────────────────────────────────────────

_NOTE_TO_PC: dict[str, int] = {
    "C": 0,  "C#": 1,  "Db": 1,  "D": 2,  "D#": 3,  "Eb": 3,
    "E": 4,  "F": 5,   "F#": 6,  "Gb": 6, "G": 7,   "G#": 8,
    "Ab": 8, "A": 9,   "A#": 10, "Bb": 10, "B": 11,
}

# Flat-preferred spelling for reconstructing chord names from a pitch class.
_PC_TO_NOTE: list[str] = [
    "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"
]
