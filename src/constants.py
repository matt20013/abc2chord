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
# Chromatic degree above tonic → (MAJOR roman, minor roman)
_DEGREE_NAMES: dict[int, tuple[str, str]] = {
    0:  ("I",    "i"),
    1:  ("bII",  "bii"),
    2:  ("II",   "ii"),
    3:  ("bIII", "biii"),
    4:  ("III",  "iii"),
    5:  ("IV",   "iv"),
    6:  ("bV",   "bv"),
    7:  ("V",    "v"),
    8:  ("bVI",  "bvi"),
    9:  ("VI",   "vi"),
    10: ("bVII", "bvii"),
    11: ("VII",  "vii"),
}
# Reverse: roman stem → semitone offset above tonic
_DEGREE_TO_OFFSET: dict[str, int] = {
    "I": 0,   "bII": 1,  "II": 2,   "bIII": 3, "III": 4,
    "IV": 5,  "bV": 6,   "V": 7,    "bVI": 8,  "VI": 9,
    "bVII": 10, "VII": 11,
    "i": 0,   "bii": 1,  "ii": 2,   "biii": 3, "iii": 4,
    "iv": 5,  "bv": 6,   "v": 7,    "bvi": 8,  "vi": 9,
    "bvii": 10, "vii": 11,
}
