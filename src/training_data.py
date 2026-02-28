"""
Load training data from ABC files.

Default training corpus (TRAINING_ABC_FILES):
  From ~/Documents/GitHub/tunes/abcs/:
    maplewood.abc       – 134 tunes with chords
    maplewood_other.abc –  30 tunes with chords
    NEFR.abc            – 147 tunes with chords
    pgh_sets_tunebook   –  76 tunes with chords
    andy_cutting_tunes  –  53 tunes with chords
                          ─────────────────────
  Total                   440 annotated tunes

Tunes without chords are silently skipped.
"""
import os
import random
from collections import Counter
import numpy as np
from .parser import extract_features_from_abc, extract_all_tunes_from_abc, _iter_scores_from_abc, _extract_features_from_score

# Project root (parent of src/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Local clone: abc2chord/../tunes/abcs/
TUNES_ABCS_DIR = os.path.join(_ROOT, "..", "tunes", "abcs")

TRAINING_ABC_URLS = [
    "https://github.com/matt20013/tunes/blob/master/abcs/maplewood.abc",
    "https://github.com/matt20013/tunes/blob/master/abcs/maplewood_other.abc",
]
TRAINING_ABC_FILES = [
    os.path.join(TUNES_ABCS_DIR, "maplewood.abc"),
    os.path.join(TUNES_ABCS_DIR, "maplewood_other.abc"),
    os.path.join(TUNES_ABCS_DIR, "NEFR.abc"),
    os.path.join(TUNES_ABCS_DIR, "pgh_sets_tunebook.abc"),
    os.path.join(TUNES_ABCS_DIR, "andy_cutting_tunes.abc"),
]
# Fallback to data/ when default paths don't exist (e.g. local full sets)
DATA_ABC_FILES = [
    os.path.join(_ROOT, "data", "maplewood.abc"),
    os.path.join(_ROOT, "data", "maplewood_other.abc"),
]


def _resolve_abc_paths(abc_paths):
    """
    Return the list of ABC paths to load.

    - If abc_paths is given explicitly (CLI --abc-paths), use those.
    - Otherwise, collect every file that actually exists from TRAINING_ABC_FILES,
      then supplement with any DATA_ABC_FILES entries not already covered.
      This means the tunes-repo files AND any data/ overrides are all used.
    """
    if abc_paths is not None:
        return [p for p in abc_paths if os.path.isfile(p)] or list(abc_paths)

    found = [p for p in TRAINING_ABC_FILES if os.path.isfile(p)]

    # Add data/ files whose basename isn't already covered by a tunes-repo file
    covered_names = {os.path.basename(p) for p in found}
    for p in DATA_ABC_FILES:
        if os.path.isfile(p) and os.path.basename(p) not in covered_names:
            found.append(p)

    return found if found else list(TRAINING_ABC_FILES)

# Chord label meaning "no chord"
NO_CHORD = "N.C."


def tune_has_chords(dataset):
    """Return True if the extracted tune has at least one chord (not N.C.)."""
    if not dataset:
        return False
    return any(row.get("target_chord") and row["target_chord"] != NO_CHORD for row in dataset)


def load_training_data(data_dir=None, abc_paths=None):
    """
    Load training (note, chord) pairs from ABC files. Only includes tunes that
    have at least one chord; tunes without chords are disregarded.

    Args:
        data_dir: Optional base directory for relative paths in abc_paths.
        abc_paths: List of paths to ABC files. Defaults to TRAINING_ABC_FILES.

    Returns:
        List of feature dicts from all included tunes (combined).
    """
    abc_paths = _resolve_abc_paths(abc_paths)
    if data_dir:
        abc_paths = [os.path.join(data_dir, p) if not os.path.isabs(p) else p for p in abc_paths]

    combined = []
    for path in abc_paths:
        if not os.path.isfile(path):
            continue
        try:
            for dataset in extract_all_tunes_from_abc(path):
                if tune_has_chords(dataset):
                    combined.extend(dataset)
        except Exception:
            continue
    return combined


def load_training_tunes(data_dir=None, abc_paths=None, augment_keys=False):
    """
    Load training data as one sequence per tune (for LSTM). Only includes tunes
    that have at least one chord.

    Args:
        augment_keys: If True, transpose every tune through all 12 keys (12× data).

    Returns:
        List of lists: each inner list is a tune's list of feature dicts.
    """
    abc_paths = _resolve_abc_paths(abc_paths)
    if data_dir:
        abc_paths = [os.path.join(data_dir, p) if not os.path.isabs(p) else p for p in abc_paths]

    tunes = []
    for path in abc_paths:
        if not os.path.isfile(path):
            continue
        try:
            for score in _iter_scores_from_abc(path):
                # Always extract and check the base score first
                base_dataset = _extract_features_from_score(score)
                if not tune_has_chords(base_dataset):
                    continue  # Skip unchorded tune entirely

                tunes.append(base_dataset)

                if augment_keys:
                    for semitones in range(1, 12):
                        s = score.transpose(semitones)
                        dataset = _extract_features_from_score(s)
                        if tune_has_chords(dataset):
                            tunes.append(dataset)
        except Exception:
            continue
    return tunes


def train_val_split(tunes, val_fraction=0.1, seed=42):
    """
    Split a list of tunes into (train_tunes, val_tunes) before augmentation.
    Validation tunes are kept at their original key (no augmentation applied).
    """
    rng = random.Random(seed)
    indices = list(range(len(tunes)))
    rng.shuffle(indices)
    n_val = max(1, int(len(tunes) * val_fraction))
    val_idx = set(indices[:n_val])
    train = [t for i, t in enumerate(tunes) if i not in val_idx]
    val = [t for i, t in enumerate(tunes) if i in val_idx]
    return train, val


def calculate_class_weights(tunes, vocab):
    """
    Compute log-smoothed class weights over all chords in tunes.

    Formula: w = log(1 + total / count)

    Inverse-frequency weighting (total / count) produces extreme ratios (e.g.
    500:1 for G vs. Abdim) that cause the model to hallucinate rare chords.
    The log compresses that ratio to roughly 7:1, preserving a gentle nudge
    toward under-represented chords without rewarding wild guesses.

    <PAD> gets weight 0. Returns a float32 tensor.
    """
    import torch
    all_chords = [row["target_chord"] for tune in tunes for row in tune]
    counts = Counter(all_chords)
    total = len(all_chords)
    n_classes = len(vocab.label_to_idx)
    weights = np.ones(n_classes, dtype=np.float32)
    for label, idx in vocab.label_to_idx.items():
        c = counts.get(label, 0)
        if c > 0:
            weights[idx] = np.log(1.0 + total / c)
    weights /= weights.mean()
    if vocab.PAD in vocab.label_to_idx:
        weights[vocab.label_to_idx[vocab.PAD]] = 0.0
    return torch.tensor(weights)
