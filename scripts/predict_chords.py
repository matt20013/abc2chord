#!/usr/bin/env python3
"""
scripts/predict_chords.py — Inference script for Degree-based LSTM model.

Loads a trained model and vocabulary, parses an ABC file, splits it into sections
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
from src.parser import _extract_features_from_score, degree_to_chord

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
        # (Filtering notes is safer than slicing stream which can be slow/buggy)
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
    # We operate on the score's parts or the score itself?
    # _extract_features_from_score uses score.flatten().
    # Inserting into the original score parts ensures it propagates.

    # Find the main part (usually the first one with notes)
    if hasattr(score, "parts") and score.parts:
        part = score.parts[0]
    else:
        part = score

    for sec in sections:
        # Check if there's already a key at this exact offset (recurse into measures)
        to_remove = []
        # Use recurse() to find keys inside measures
        for k in part.recurse().getElementsByClass(music21.key.Key):
            # Calculate absolute offset relative to the part
            try:
                k_offset = k.getOffsetInHierarchy(part)
            except Exception:
                # Fallback if hierarchy is broken or ambiguous
                k_offset = k.offset

            if float(k_offset) == float(sec.start):
                to_remove.append(k)

        for k in to_remove:
            # Remove from its immediate container (e.g., Measure)
            # activeSite is usually the container being iterated or the primary one
            container = k.activeSite
            if container:
                container.remove(k)

        # Insert new key into the part directly (it will be floating, not in a measure)
        # This is fine for _extract_features_from_score which flattens anyway.
        part.insert(sec.start, sec.key_obj)

# ---------------------------------------------------------------------------
# 2. Prediction & Post-processing
# ---------------------------------------------------------------------------

MeasureData = collections.namedtuple("MeasureData", ["number", "chord", "section_index", "key_label"])

def predict_for_score(score, model, vocab, device):
    """
    Extract features and run inference.
    Returns (features, predictions).
    """
    features = _extract_features_from_score(score)
    if not features:
        return [], []

    # tune_to_arrays expects a list of feature dicts (which is what features is)
    # normalize=True matches training default
    X, _ = tune_to_arrays(features, vocab=None, normalize=True)

    # Run prediction
    # X is (seq_len, input_dim)
    # predict_chords expects X, and if vocab is provided, returns decoded strings
    predictions = predict_chords(model, X, lengths=len(X), vocab=vocab, device=device)

    # predict_chords returns a list of lists (batch_size=1) because we passed 2D array unsqueezed
    if predictions and isinstance(predictions[0], list):
        predictions = predictions[0]

    return features, predictions

def post_process_predictions(features, predictions, sections):
    """
    Group predictions by measure and apply majority voting.
    Map degrees back to chords using local key.
    """
    # Group by measure
    measures = collections.defaultdict(list)
    measure_keys = {} # measure_num -> key_tonic_pc (from first note)

    for feat, pred_degree in zip(features, predictions):
        m_num = feat["measure"]
        measures[m_num].append(pred_degree)
        if m_num not in measure_keys:
            measure_keys[m_num] = (feat["key_tonic_pc"], feat["key_label"])

    sorted_m_nums = sorted(measures.keys())
    results = []

    for m_num in sorted_m_nums:
        degrees = measures[m_num]
        # Majority vote
        counts = collections.Counter(degrees)
        winner_degree = counts.most_common(1)[0][0]

        tonic_pc, key_label = measure_keys[m_num]

        # Convert to absolute chord
        abs_chord = degree_to_chord(winner_degree, tonic_pc)

        # Identify section index
        # We need the offset of the measure to match against sections.
        # Ideally features would have offset, but they have 'beat', 'measure'.
        # We can infer section from key_label if sections have unique keys? No.
        # But we processed sections sequentially.
        # Let's assign section index based on where this measure falls.
        # Since we don't have measure offsets easily here (features don't store absolute offset),
        # we can rely on the key_label change or just sequential order if we assume measures are sorted.
        # Wait, features are sorted by time (as they come from _extract_features_from_score).
        # We can track section changes.

        # Better: use the original score structure?
        # Or, just use the fact that sections are split by time.
        # But we aggregated by measure.
        # Let's assume measures are monotonic.

        # We can find which section this measure belongs to by looking at the key_label
        # stored in features. But key labels might be same for different sections.
        # However, for the display purpose "A-Part", "B-Part", we need to know when section changes.

        # Let's try to map measure number to section.
        # We can't easily do it without offset.
        # But wait, `_extract_features_from_score` processes notes in order.
        # So the list of features is ordered.
        # The measure numbers increase.

        # Let's just iterate through the sorted measures and detect section boundaries
        # by checking if we crossed a section boundary?
        # We don't have offsets here.

        # Workaround: pass the score and build a map of measure_num -> section_index?
        # Or just use the sections list and measure objects.

        # Simpler: The features have 'key_tonic_pc'.
        # If the key changes, it's a strong hint.
        # But if key doesn't change (A: G, B: G), we lose the boundary info.

        # Solution: Add 'section_index' to features during extraction? No, can't modify parser easily.
        # Use the fact that we inserted keys at specific offsets.
        # If we iterate through measures in the score, we know their offsets.

        results.append(MeasureData(m_num, abs_chord, -1, key_label)) # section_index to be filled later

    return results

def assign_sections_to_measures(score, measure_data: list[MeasureData], sections: list[Section]):
    """
    Map each measure datum to a section index based on measure offsets in score.
    """
    # Create a map of measure_number -> offset
    # Note: measure numbers might not be unique if there are multiple parts,
    # but we assume single melody line.

    m_offsets = {}
    # Use parts[0]
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
            # Handle edge case where measure starts exactly at end (shouldn't happen for valid measures)
            # or last section
            if off >= sec.start and sec == sections[-1]:
                sec_idx = sec.index

        updated_data.append(md._replace(section_index=sec_idx))

    return updated_data

def format_output(measure_data: list[MeasureData], sections: list[Section]):
    """
    Print Lead Sheet style output.
    """
    current_section = -1

    # Pre-compute section keys for display
    section_keys = {s.index: s.key_obj for s in sections}

    # Generate section labels (A, B, C...)
    section_labels = {}
    for i, s in enumerate(sections):
        label = chr(ord('A') + i) # A, B, C...
        key_str = f"{s.key_obj.tonic.name} {s.key_obj.mode}"
        section_labels[s.index] = f"[{label}-Part] (Key: {key_str})"

    # Buffer for current line
    line_buffer = []

    for md in measure_data:
        if md.section_index != current_section:
            # New section starting
            if line_buffer:
                print("| " + " | ".join(line_buffer) + " |")
                line_buffer = []

            print(f"\n{section_labels.get(md.section_index, '[Unknown Part]')}:")
            current_section = md.section_index

        line_buffer.append(md.chord if md.chord else "N.C.")

    if line_buffer:
        print("| " + " | ".join(line_buffer) + " |")
    print() # Final newline

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
        # music21 converter.parse can handle ABC files
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
