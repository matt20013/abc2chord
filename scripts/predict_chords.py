#!/usr/bin/env python3
"""
scripts/predict_chords.py — Inference script for Multi-Hot LSTM model.

Loads a trained model, parses an ABC file, splits it into sections
(A-part, B-part), estimates the key for each section, and predicts chords.
"""

import argparse
import collections
import json
import os
import sys

import music21
import numpy as np
import torch

# Ensure src module is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from src.model import load_model_and_vocab, tune_to_arrays, predict_chords
from src.parser import _extract_features_from_score
from src.chord_encoding import decode_target_to_chord

# ---------------------------------------------------------------------------
# 1. Structure Analysis & Key Estimation
# ---------------------------------------------------------------------------

Section = collections.namedtuple("Section", ["start", "end", "key_obj", "index"])

def analyze_structure(score) -> list[Section]:
    """
    Split score into sections based on repeat boundaries and analyze key for each.
    Returns a list of Section objects.
    """
    # Use the first part if available (melody), otherwise flatten the whole score
    if hasattr(score, "parts") and score.parts:
        flat = score.parts[0].flatten()
    else:
        flat = score.flatten()

    duration = flat.duration.quarterLength

    # Find repeat boundaries
    boundaries = {0.0, float(duration)}
    for el in flat.getElementsByClass(music21.bar.Repeat):
        if el.direction == 'end':
            boundaries.add(float(el.offset))

    sorted_bounds = sorted(list(boundaries))

    sections = []
    for i in range(len(sorted_bounds) - 1):
        start = sorted_bounds[i]
        end = sorted_bounds[i+1]

        # Skip empty sections (if any)
        if start >= end:
            continue

        # Extract notes in this range for analysis
        notes = []
        for n in flat.notes:
            if n.offset >= start and n.offset < end:
                notes.append(n)

        if not notes:
            # Fallback if no notes: use previous key or default
            key = music21.key.Key('C')
        else:
            # Create a temp stream for analysis
            s = music21.stream.Stream()
            for n in notes:
                s.insert(n.offset - start, n)
            key = s.analyze('key')

        sections.append(Section(start, end, key, i))

    return sections

def apply_keys(score, sections: list[Section]):
    """
    Insert analyzed Key objects into the score at section starts.
    """
    if hasattr(score, "parts") and score.parts:
        part = score.parts[0]
    else:
        part = score

    for sec in sections:
        # Check if there's already a key at this exact offset (recurse into measures)
        to_remove = []
        for k in part.recurse().getElementsByClass(music21.key.Key):
            try:
                k_offset = k.getOffsetInHierarchy(part)
            except Exception:
                k_offset = k.offset

            if float(k_offset) == float(sec.start):
                to_remove.append(k)

        for k in to_remove:
            container = k.activeSite
            if container:
                container.remove(k)

        part.insert(sec.start, sec.key_obj)

# ---------------------------------------------------------------------------
# 2. Prediction & Post-processing
# ---------------------------------------------------------------------------

MeasureData = collections.namedtuple("MeasureData", ["number", "chord", "section_index", "key_label"])

def predict_for_score(score, model, vocab, device):
    """
    Extract features and run inference.
    Returns (features, predictions_absolute_chords).
    """
    features = _extract_features_from_score(score)
    if not features:
        return [], []

    # tune_to_arrays expects a list of feature dicts
    # vocab=None is fine
    X, _ = tune_to_arrays(features, vocab=None, normalize=True)

    # Run prediction -> logits (1, Seq, 12)
    logits = predict_chords(model, X, lengths=len(X), vocab=vocab, device=device)
    if logits.ndim == 3:
        logits = logits[0]

    predictions = []
    for i, row in enumerate(features):
        logit_vec = logits[i]
        prob_vec = 1.0 / (1.0 + np.exp(-logit_vec))

        key_tonic = row.get("key_tonic_pc", 0)
        chord_str = decode_target_to_chord(prob_vec, key_tonic)
        predictions.append(chord_str)

    return features, predictions

def post_process_predictions(features, predictions, sections):
    """
    Group predictions by measure and apply majority voting.
    """
    # Group by measure
    measures = collections.defaultdict(list)
    measure_keys = {} # measure_num -> key_tonic_pc (from first note)

    for feat, pred_chord in zip(features, predictions):
        m_num = feat["measure"]
        measures[m_num].append(pred_chord)
        if m_num not in measure_keys:
            measure_keys[m_num] = (feat["key_tonic_pc"], feat["key_label"])

    sorted_m_nums = sorted(measures.keys())
    results = []

    for m_num in sorted_m_nums:
        chord_list = measures[m_num]
        # Majority vote
        counts = collections.Counter(chord_list)
        winner_chord = counts.most_common(1)[0][0]

        tonic_pc, key_label = measure_keys[m_num]

        # winner_chord is already absolute string
        results.append(MeasureData(m_num, winner_chord, -1, key_label))

    return results

def assign_sections_to_measures(score, measure_data: list[MeasureData], sections: list[Section]):
    """
    Map each measure datum to a section index based on measure offsets in score.
    """
    m_offsets = {}
    if hasattr(score, "parts") and score.parts:
        part = score.parts[0]
    else:
        part = score

    for m in part.getElementsByClass(music21.stream.Measure):
        m_offsets[m.measureNumber] = float(m.offset)

    updated_data = []
    for md in measure_data:
        off = m_offsets.get(md.number, 0.0)
        # Find section
        sec_idx = 0
        for sec in sections:
            if off >= sec.start and off < sec.end:
                sec_idx = sec.index
                break
            if off >= sec.start and sec == sections[-1]:
                sec_idx = sec.index

        updated_data.append(md._replace(section_index=sec_idx))

    return updated_data

def format_output(measure_data: list[MeasureData], sections: list[Section]):
    """
    Print Lead Sheet style output.
    """
    current_section = -1
    section_labels = {}
    for i, s in enumerate(sections):
        label = chr(ord('A') + i) # A, B, C...
        key_str = f"{s.key_obj.tonic.name} {s.key_obj.mode}"
        section_labels[s.index] = f"[{label}-Part] (Key: {key_str})"

    line_buffer = []

    for md in measure_data:
        if md.section_index != current_section:
            if line_buffer:
                print("| " + " | ".join(line_buffer) + " |")
                line_buffer = []

            print(f"\n{section_labels.get(md.section_index, '[Unknown Part]')}:")
            current_section = md.section_index

        line_buffer.append(md.chord if md.chord else "N.C.")

    if line_buffer:
        print("| " + " | ".join(line_buffer) + " |")
    print()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Predict chords for an ABC tune.")
    parser.add_argument("abc_file", help="Path to the ABC file.")
    parser.add_argument("--checkpoint", default=os.path.join(_ROOT, "checkpoints", "degree"),
                        help="Path to checkpoint directory (default: checkpoints/degree)")
    args = parser.parse_args()

    # 1. Load Model
    if not os.path.isdir(args.checkpoint):
        sys.exit(f"Error: Checkpoint directory not found: {args.checkpoint}")

    print(f"Loading model from {args.checkpoint}...", file=sys.stderr)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, vocab = load_model_and_vocab(args.checkpoint, device=device)
    except Exception as e:
        sys.exit(f"Error loading model: {e}")

    # 2. Parse ABC
    print(f"Parsing {args.abc_file}...", file=sys.stderr)
    try:
        score = music21.converter.parse(args.abc_file)
    except Exception as e:
        sys.exit(f"Error parsing ABC file: {e}")

    # 3. Analyze Structure & Apply Keys
    print("Analyzing structure...", file=sys.stderr)
    sections = analyze_structure(score)
    apply_keys(score, sections)

    # 4 & 5. Extract Features & Predict
    print("Running inference...", file=sys.stderr)
    features, predictions = predict_for_score(score, model, vocab, device)

    if not features:
        sys.exit("Error: No notes found in score.")

    # 6. Post-process
    measure_data = post_process_predictions(features, predictions, sections)
    measure_data = assign_sections_to_measures(score, measure_data, sections)

    # 7. Output
    format_output(measure_data, sections)

if __name__ == "__main__":
    main()
