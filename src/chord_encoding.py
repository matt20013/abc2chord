import numpy as np
import re

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
    return _NOTE_TO_PC.get(note_name)

def _generate_template_vector(indices_or_weights):
    """Helper to create a 12-element vector from a list of active indices or (index, weight) tuples."""
    v = np.zeros(12, dtype=np.float32)
    for item in indices_or_weights:
        if isinstance(item, tuple):
            idx, weight = item
            v[idx] = weight
        else:
            v[item] = 1.0
    return v

def get_chord_templates(hierarchical=False):
    """
    Generate a dictionary of relative chord templates.
    Returns:
        list of (root_offset, quality_suffix, vector)
    """
    # Base qualities (indices relative to root 0)
    qualities_flat = {
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

    qualities_hierarchical = {
        "":      [(0, 1.0), (4, 0.9), (7, 0.6)],
        "m":     [(0, 1.0), (3, 0.9), (7, 0.6)],
        "7":     [(0, 1.0), (4, 0.9), (7, 0.5), (10, 0.8)],
        "m7":    [(0, 1.0), (3, 0.9), (7, 0.5), (10, 0.7)],
        "maj7":  [(0, 1.0), (4, 0.9), (7, 0.5), (11, 0.8)],
        "dim":   [(0, 1.0), (3, 0.9), (6, 0.8)],
        "dim7":  [(0, 1.0), (3, 0.9), (6, 0.8)], # Mapped to dim weights as placeholder/fallback
        "aug":   [(0, 1.0), (4, 0.9), (8, 0.8)],
        "sus4":  [(0, 1.0), (5, 0.9), (7, 0.6)],
        "sus2":  [(0, 1.0), (2, 0.9), (7, 0.6)],
    }

    # Choose base set
    qualities = qualities_hierarchical if hierarchical else qualities_flat

    templates = []

    for root_offset in range(12):
        for suffix, intervals in qualities.items():
            # Shift intervals by root_offset
            if hierarchical:
                shifted = [((idx + root_offset) % 12, weight) for idx, weight in intervals]
            else:
                shifted = [(idx + root_offset) % 12 for idx in intervals]

            vec = _generate_template_vector(shifted)
            templates.append((root_offset, suffix, vec))

    return templates

# Cache for templates
_TEMPLATES_CACHE = {}

def _get_cached_templates(hierarchical=False):
    if hierarchical not in _TEMPLATES_CACHE:
        _TEMPLATES_CACHE[hierarchical] = get_chord_templates(hierarchical)
    return _TEMPLATES_CACHE[hierarchical]

def encode_chord_to_target(chord_str, key_tonic_pc, hierarchical=False):
    """
    Convert a chord string to a 12-element multi-hot vector relative to the key tonic.

    Args:
        chord_str (str): The chord label (e.g., "G7", "Am", "N.C.").
        key_tonic_pc (int): The pitch class of the local key tonic (0-11).
        hierarchical (bool): Whether to use soft/weighted targets.

    Returns:
        np.array: A 12-element array of float32.
    """
    target = np.zeros(12, dtype=np.float32)

    if not chord_str or chord_str == "N.C.":
        return target

    # Parse root and suffix manually
    root_match = re.match(r"^([A-G][b#]?)", chord_str)
    if not root_match:
        return target

    root_note = root_match.group(1)
    suffix = chord_str[len(root_note):]

    root_pc = get_note_pc(root_note)
    if root_pc is None:
        return target

    # Calculate relative root
    rel_root = (root_pc - key_tonic_pc) % 12

    # Get templates
    templates = _get_cached_templates(hierarchical)

    # Find matching template: (root_offset, suffix, vec)
    # We loop to find the exact match for (rel_root, suffix)
    found_vec = None
    for r_off, sfx, vec in templates:
        if r_off == rel_root and sfx == suffix:
            found_vec = vec
            break

    if found_vec is not None:
        return found_vec

    # If suffix not found (e.g. unknown extension), return zero vector or fallback?
    # Current behavior: return zeros (implicit in target init)
    return target


def decode_target_to_chord(target_vector, key_tonic_pc, hierarchical=False):
    """
    Find the closest chord label for the given relative target vector.

    Args:
        target_vector (np.array): 12-element predicted vector (logits or probs).
        key_tonic_pc (int): The pitch class of the local key tonic.
        hierarchical (bool): Whether to use hierarchical templates for comparison.

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
    target_norm = np.linalg.norm(target_vector)
    if target_norm < 1e-6:
        return "N.C."

    templates = _get_cached_templates(hierarchical)

    for root_offset, suffix, template_vec in templates:
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
