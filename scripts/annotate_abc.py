#!/usr/bin/env python3
"""
scripts/annotate_abc.py — Annotate an ABC file with predicted chords.

Loads a trained model and vocabulary, parses an ABC file, predicts chords,
and writes them back into the ABC source code in place.
"""

import argparse
import collections
import os
import re
import sys
import json

import music21
import numpy as np
import torch
import torch.nn as nn

# Ensure src module is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# Explicitly import the parser and model utilities
from src.parser import _extract_features_from_score, degree_to_chord
from src.model import tune_to_arrays, predict_chords
# ---------------------------------------------------------------------------
# 1. Prediction Helpers
# ---------------------------------------------------------------------------

class LSTMChordModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=32, n_layers=2, dropout=0.5, 
                 one_hot_scale_degree=True): # <--- MATCHING THE TRAINED ARCHITECTURE
        super(LSTMChordModel, self).__init__()
        self.input_dim = 17 if one_hot_scale_degree else 6
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, n_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.fc(out)

def predict_for_score(score, model, vocab, device):
    """
    Extract features and run inference.
    Returns (features, predictions).
    """
    features = _extract_features_from_score(score)
    if not features:
        return [], []

    # 1. Prepare data
    X, _ = tune_to_arrays(features, normalize=True)
    X_tensor = torch.FloatTensor(X).unsqueeze(0).to(device)
    lengths = torch.tensor([len(X)])

    # 2. Run raw inference (don't use predict_chords here to avoid the .decode() error)
    model.eval()
    with torch.no_grad():
        output = model(X_tensor, lengths)
        # Get the index of the highest logit for each note
        pred_indices = output.argmax(dim=-1).squeeze(0).cpu().numpy()

    # 3. Manually decode using your dict (handling string vs int keys)
    # Your vocab dict is likely { "0": "<PAD>", "1": "I", ... }
    inv_vocab = {int(k): v for k, v in vocab.items()}
    predictions = [inv_vocab.get(int(idx), "<UNK>") for idx in pred_indices]

    return features, predictions
def get_measure_chords(features, predictions):
    """
    Group predictions by measure and apply majority voting.
    Map degrees back to chords using local key.

    Returns:
        dict: measure_index (0-based) -> predicted_chord_string
    """
    if not features or not predictions:
        return {}

    # Group by measure number
    measures = collections.defaultdict(list)
    measure_keys = {} # measure_num -> key_tonic_pc

    # We need to map measure number to an index (0, 1, 2...) corresponding to the sequence of measures
    # But features only have measure number.
    # The measure numbers are usually sequential integers, but can skip or repeat?
    # music21 usually ensures unique measure numbers unless repeat happens.
    # However, for annotation, we just need measure number -> chord.
    # Wait, if we use measure number, we rely on music21 measure numbering matching our text parsing.
    # This might be fragile if music21 numbers differ.

    # Let's stick to measure number for grouping first.
    for feat, pred_degree in zip(features, predictions):
        m_num = feat["measure"]
        measures[m_num].append(pred_degree)
        if m_num not in measure_keys:
            measure_keys[m_num] = feat["key_tonic_pc"]

    # Resolve chords
    measure_chords = {}
    for m_num, degrees in measures.items():
        counts = collections.Counter(degrees)
        winner_degree = counts.most_common(1)[0][0]
        tonic_pc = measure_keys[m_num]
        abs_chord = degree_to_chord(winner_degree, tonic_pc)
        if abs_chord == "N.C.":
            continue # Don't annotate N.C.
        measure_chords[m_num] = abs_chord

    return measure_chords

# ---------------------------------------------------------------------------
# 2. Text Processing & Annotation
# ---------------------------------------------------------------------------

def count_notes_in_segment(segment):
    """
    Heuristic to count notes in an ABC segment.
    Strips strings, comments, and fields.
    """
    # Remove comments
    s = re.sub(r'%.*', '', segment)
    # Remove strings (chords/annotations)
    s = re.sub(r'"[^"]*"', '', s)
    # Remove inline fields [K:...]
    s = re.sub(r'\[[A-Z]:.*?\]', '', s)
    # Count note-like patterns [A-G]
    # This is rough but should distinguish music from rests/empty space
    notes = re.findall(r"[A-Ga-g]", s)
    return len(notes)

def annotate_abc_content(content, score_measures, measure_chords):
    """
    Injects chords into ABC content.

    Args:
        content (str): The raw ABC file content.
        score_measures (list): List of music21 Measure objects (to align with text).
        measure_chords (dict): measure_number -> chord string.

    Returns:
        str: Annotated ABC content.
    """

    # Split content by bar lines, capturing delimiters
    # Bar lines: |, ||, |], |:, :|, ::
    parts = re.split(r'([:|]?\|[:|]?)', content)

    out_parts = []

    # Pointers
    m_idx = 0  # Index into score_measures

    # State
    in_header = True

    # We iterate through parts. Even indices are content, Odd are delimiters.
    # But re.split returns [content, delimiter, content, delimiter...]

    for i, part in enumerate(parts):
        # Even index: segment content
        if i % 2 == 0:
            segment = part

            # Check if we are still in header
            # Heuristic: if line starts with X: T: M: K: L: P:
            # and we haven't seen music yet.
            # But parts might contain newlines.
            # If segment contains K: line, body likely starts after.

            # Split segment into header and body
            lines = segment.splitlines(keepends=True)
            header_lines = []
            body_lines = []

            for line in lines:
                stripped = line.strip()
                if in_header:
                    header_lines.append(line)
                    # Check if this line ends the header (usually K:)
                    if stripped.startswith("K:"):
                        in_header = False
                else:
                    body_lines.append(line)

            header_part = "".join(header_lines)
            body_part = "".join(body_lines)

            # If we are in body, process body_part
            if not in_header and m_idx < len(score_measures):
                # If body_part is empty, we might be just finishing the header in this segment
                # Skip alignment if no content
                if not body_part.strip() and count_notes_in_segment(body_part) == 0:
                    # Append strictly header_part + body_part (which is just whitespace)
                    pass
                else:
                    m = score_measures[m_idx]

                    # Heuristic check
                    seg_notes = count_notes_in_segment(body_part)
                    # music21 measure note count (excluding rests)
                    m_notes = len([n for n in m.flatten().notes if not n.isRest])

                    match = True
                    if m_notes > 0 and seg_notes == 0:
                        # Score has notes, text has none -> Mismatch
                        match = False
                    elif m_notes == 0 and seg_notes > 0:
                        # Score has no notes, text has notes -> Mismatch
                        match = False

                    # If match, insert chord
                    if match:
                        chord = measure_chords.get(m.measureNumber)

                        if chord:
                            # Remove existing chords
                            cleaned = re.sub(r'"[A-G][b#]?[a-zA-Z0-9/]*"', '', body_part)

                            # Find insertion point: after leading whitespace/newlines
                            match_ws = re.match(r'^(\s*)', cleaned)
                            prefix = match_ws.group(1) if match_ws else ""
                            rest = cleaned[len(prefix):]

                            body_part = f'{prefix}"{chord}"{rest}'

                        # Advance measure pointer
                        m_idx += 1

            segment = header_part + body_part
            out_parts.append(segment)
        else:
            # Odd index: delimiter
            out_parts.append(part)

    return "".join(out_parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Annotate an ABC file with predicted chords.")
    parser.add_argument("abc_file", help="Path to the ABC file.")
    parser.add_argument("--checkpoint", default=os.path.join(_ROOT, "checkpoints", "degree"),
                        help="Path to checkpoint directory")
    parser.add_argument("--mock", action="store_true", help="Use mock model (for testing).")
    args = parser.parse_args()

    # 1. Load Model (Correctly indented inside main)
    if args.mock:
        model, vocab = None, None
        print("Using MOCK model.", file=sys.stderr)
    else:
        # Use our robust loading logic
        target_path = args.checkpoint
        checkpoint_dir = target_path if os.path.isdir(target_path) else os.path.dirname(target_path)
        
        print(f"Loading model from {target_path}...", file=sys.stderr)
        try:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            # We bypass the old src.model.load_model_and_vocab if it's outdated
            # and use our fixed logic directly
            checkpoint = torch.load(target_path if not os.path.isdir(target_path) else os.path.join(target_path, "lstm_chord.pt"), map_location=device)
            
            # Load Vocab
            vocab_path = os.path.join(checkpoint_dir, "chord_vocab.json")
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            
            config = checkpoint.get('config', {})
            model = LSTMChordModel(
                vocab_size=len(vocab),
                hidden_dim=config.get('hidden_dim', 32),
                n_layers=config.get('n_layers', 2),
                one_hot_scale_degree=config.get('one_hot_scale_degree', True)
            )
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            model.to(device)
            model.eval()
        except Exception as e:
            sys.exit(f"Error loading model: {e}")
    # 2. Parse ABC
    print(f"Parsing {args.abc_file}...", file=sys.stderr)
    try:
        score = music21.converter.parse(args.abc_file)
    except Exception as e:
        sys.exit(f"Error parsing ABC file: {e}")

    # 3. Predict
    print("Running inference...", file=sys.stderr)
    if args.mock:
        # Dummy predictions: Measure 1 -> Am, Measure 2 -> G7 ...
        # Need to know measures in score
        features = [] # Dummy
        predictions = [] # Dummy
        # We need measure_chords directly
        measure_chords = {}
        # Iterate measures to populate mock
        if hasattr(score, "parts") and score.parts:
            part = score.parts[0]
        else:
            part = score

        # Mock pattern: Am, Dm, G7, C
        mock_pattern = ["Am", "Dm", "G7", "C"]
        i = 0
        for m in part.getElementsByClass(music21.stream.Measure):
            measure_chords[m.measureNumber] = mock_pattern[i % len(mock_pattern)]
            i += 1
    else:
        features, predictions = predict_for_score(score, model, vocab, device)
        if not features:
            sys.exit("Error: No features extracted from score.")
        measure_chords = get_measure_chords(features, predictions)

    # 4. Annotate
    print("Annotating...", file=sys.stderr)
    with open(args.abc_file, "r") as f:
        content = f.read()

    # Get linear list of measures from score for alignment
    if hasattr(score, "parts") and score.parts:
        measures_list = list(score.parts[0].getElementsByClass(music21.stream.Measure))
    else:
        measures_list = list(score.getElementsByClass(music21.stream.Measure))

    annotated_content = annotate_abc_content(content, measures_list, measure_chords)

    # 5. Save
    base_name = os.path.splitext(args.abc_file)[0]
    out_file = f"{base_name}_predicted.abc"
    with open(out_file, "w") as f:
        f.write(annotated_content)

    print(f"Saved annotated file to {out_file}", file=sys.stderr)

if __name__ == "__main__":
    main()
