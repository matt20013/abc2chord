import numpy as np
from pychord import Chord

# Note-to-pitch-class mapping (copied from src/parser.py to avoid circular imports)
_NOTE_TO_PC = {
    "C": 0,  "C#": 1,  "Db": 1,  "D": 2,  "D#": 3,  "Eb": 3,
    "E": 4,  "F": 5,   "F#": 6,  "Gb": 6, "G": 7,   "G#": 8,
    "Ab": 8, "A": 9,   "A#": 10, "Bb": 10, "B": 11,
}

_PC_TO_NOTE = [
    "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"
]

def get_note_pc(note_name):
    """Map a note name (e.g., 'C', 'F#', 'Gb') to its pitch class (0-11)."""
    # pychord might return just note names, but if it returns octaves, strip them.
    # Assuming pychord components are just note names.
    return _NOTE_TO_PC.get(note_name)

def encode_chord_to_target(chord_str, key_tonic_pc):
    """
    Convert a chord string to a 12-element multi-hot vector relative to the key tonic.

    Args:
        chord_str (str): The chord label (e.g., "G7", "Am", "N.C.").
        key_tonic_pc (int): The pitch class of the local key tonic (0-11).

    Returns:
        np.array: A 12-element array of 0.0 and 1.0 (float32).
    """
    target = np.zeros(12, dtype=np.float32)

    if not chord_str or chord_str == "N.C.":
        return target

    try:
        # Handle simple cases manually if pychord fails or for speed?
        # But pychord is robust.
        # Note: pychord expects standard chord names.
        # Our dataset might have simplified chords like "G" or "Am".
        c = Chord(chord_str)
        components = c.components()
    except Exception:
        # Fallback or silent failure (return zeros)
        # print(f"Warning: Could not parse chord '{chord_str}'")
        return target

    for note in components:
        pc = get_note_pc(note)
        if pc is not None:
            rel_pc = (pc - key_tonic_pc) % 12
            target[rel_pc] = 1.0

    return target

def _generate_template_vector(indices):
    """Helper to create a 12-element vector from a list of active indices."""
    v = np.zeros(12, dtype=np.float32)
    v[indices] = 1.0
    return v

def get_chord_templates():
    """
    Generate a dictionary of relative chord templates.
    Returns:
        list of (root_offset, quality_suffix, vector)
    """
    # Base qualities (indices relative to root 0)
    qualities = {
        "":      [0, 4, 7],           # Major
        "m":     [0, 3, 7],           # Minor
        "7":     [0, 4, 7, 10],       # Dominant 7
        "m7":    [0, 3, 7, 10],       # Minor 7
        "maj7":  [0, 4, 7, 11],       # Major 7
        "dim":   [0, 3, 6],           # Diminished
        "dim7":  [0, 3, 6, 9],        # Diminished 7
        "aug":   [0, 4, 8],           # Augmented
        "sus4":  [0, 5, 7],           # Suspended 4
        "sus2":  [0, 2, 7],           # Suspended 2
    }

    templates = []

    for root_offset in range(12):
        for suffix, intervals in qualities.items():
            # Shift intervals by root_offset
            indices = [(i + root_offset) % 12 for i in intervals]
            vec = _generate_template_vector(indices)
            templates.append((root_offset, suffix, vec))

    return templates

# Pre-compute templates once
_CHORD_TEMPLATES = get_chord_templates()

def decode_target_to_chord(target_vector, key_tonic_pc):
    """
    Find the closest chord label for the given relative target vector.

    Args:
        target_vector (np.array): 12-element predicted vector (logits or probs).
                                  Should ideally be probabilities (0-1 range).
        key_tonic_pc (int): The pitch class of the local key tonic.

    Returns:
        str: The best matching absolute chord string (e.g., "G7").
    """
    # Ensure vector is numpy array
    target_vector = np.array(target_vector, dtype=np.float32)

    # Handle "N.C." case: if vector has very low energy
    if np.max(target_vector) < 0.1: # Threshold can be tuned
        return "N.C."

    best_score = -1.0
    best_chord = "N.C."

    # Using Cosine Similarity: (A . B) / (|A| |B|)
    # Pre-computed template norms could speed this up, but this is fast enough.
    target_norm = np.linalg.norm(target_vector)
    if target_norm < 1e-6:
        return "N.C."

    for root_offset, suffix, template_vec in _CHORD_TEMPLATES:
        template_norm = np.linalg.norm(template_vec) # Pre-computable
        if template_norm < 1e-6:
            continue

        score = np.dot(target_vector, template_vec) / (target_norm * template_norm)

        if score > best_score:
            best_score = score
            # Convert relative root back to absolute pitch
            abs_root_pc = (key_tonic_pc + root_offset) % 12
            root_note = _PC_TO_NOTE[abs_root_pc]
            best_chord = f"{root_note}{suffix}"

    return best_chord
