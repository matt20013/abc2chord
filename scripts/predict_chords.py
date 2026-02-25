#!/usr/bin/env python3
"""
scripts/predict_chords.py — Inference script for Multi-Hot LSTM model.

Loads a trained model, parses an ABC file, and predicts chords, outputting
a Lead Sheet style format. 
"""

import argparse
import collections
import os
import sys
import numpy as np
import torch
import music21
import json

# Ensure src module is importable for utilities
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

try:
    from src.model import load_model_and_vocab, tune_to_arrays, predict_chords
    from src.parser import _extract_features_from_score
    from src.chord_encoding import decode_target_to_chord
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure you are running from the project root.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Predict chords using the Multi-Hot LSTM model.")
    parser.add_argument("abc_file", help="Path to the ABC file")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint directory")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading model from {args.checkpoint}...", file=sys.stderr)
    target_path = args.checkpoint
    checkpoint_dir = target_path if os.path.isdir(target_path) else os.path.dirname(target_path)
    
    model, vocab = load_model_and_vocab(checkpoint_dir, device=device)
    
    # Load hierarchical setting from config
    hierarchical = False
    config_path = os.path.join(checkpoint_dir, "model_config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
            hierarchical = cfg.get("hierarchical_targets", False)

    # 2. Parse ABC
    print(f"Parsing {args.abc_file}...", file=sys.stderr)
    try:
        score = music21.converter.parse(args.abc_file)
    except Exception as e:
        sys.exit(f"Error parsing ABC file: {e}")
    
    # 3. Extract Features (adaptive chunking natively handles local keys)
    features = _extract_features_from_score(score)
    if not features:
        print("No notes found in score.", file=sys.stderr)
        return

    # 4. Predict
    X, _ = tune_to_arrays(features, vocab=None, normalize=True)
    logits = predict_chords(model, X, lengths=len(X), vocab=vocab, device=device)
    
    # Handle batch dimension
    if logits.ndim == 3:
        logits = logits[0]

    predictions = []
    for i, row in enumerate(features):
        logit_vec = logits[i]
        # Convert logits to probabilities using Sigmoid
        prob_vec = 1.0 / (1.0 + np.exp(-logit_vec))
        
        key_tonic = row.get("key_tonic_pc", 0)
        chord_str = decode_target_to_chord(prob_vec, key_tonic, hierarchical=hierarchical)
        predictions.append(chord_str)

    # 5. Majority Vote per Measure
    measures = collections.defaultdict(list)
    measure_keys = {}
    
    for feat, pred_chord in zip(features, predictions):
        m_num = feat["measure"]
        measures[m_num].append(pred_chord)
        if m_num not in measure_keys:
            measure_keys[m_num] = feat.get("key_label", "Unknown Key")

    # 6. Output Lead Sheet format
    print("\n" + "="*50)
    print(f" Predicted Chords for: {os.path.basename(args.abc_file)}")
    print("="*50)
    
    current_key = None
    line_buffer = []
    
    for m_num in sorted(measures.keys()):
        chord_list = measures[m_num]
        winner_chord = collections.Counter(chord_list).most_common(1)[0][0]
        key_label = measure_keys[m_num]
        
        if key_label != current_key:
            if line_buffer:
                print("| " + " | ".join(line_buffer) + " |")
                line_buffer = []
            print(f"\n[{key_label}]:")
            current_key = key_label
            
        line_buffer.append(f"{winner_chord:<4}")
        
        # Print 4 measures per line
        if len(line_buffer) == 4:
            print("| " + " | ".join(line_buffer) + " |")
            line_buffer = []

    if line_buffer:
        print("| " + " | ".join(line_buffer) + " |")
    print("\nDone.")

if __name__ == "__main__":
    main()