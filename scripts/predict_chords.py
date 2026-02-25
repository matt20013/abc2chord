#!/usr/bin/env python3
"""
scripts/predict_chords.py — Fixed inference script for Degree-based LSTM model.
Now supports one-hot scale degrees and ABC chord injection.
"""

import argparse
import collections
import json
import os
import sys
import re

import music21
import torch
import torch.nn as nn

# Ensure src module is importable for utilities
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# Import necessary utilities from your existing source
try:
    from src.model import tune_to_arrays, predict_chords
    from src.parser import _extract_features_from_score, degree_to_chord
except ImportError:
    print("Error: Could not find 'src' directory. Ensure you are running from the project root.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Corrected Model Definition
# ---------------------------------------------------------------------------

class LSTMChordModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=32, n_layers=2, dropout=0.5, 
                 one_hot_scale_degree=True): # <--- THE FIX
        super(LSTMChordModel, self).__init__()
        
        # If one-hot, dim is 12 (notes) + 5 (rhythm) = 17
        # If integer, dim is 1 (note) + 5 (rhythm) = 6
        self.input_dim = 17 if one_hot_scale_degree else 6
        
        self.lstm = nn.LSTM(
            self.input_dim, 
            hidden_dim, 
            n_layers, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.fc(out)

def load_fixed_model(checkpoint_path, device):
    """Loads model using the specific filenames found in your checkpoints folder."""
    base_dir = os.path.dirname(checkpoint_path)
    if not base_dir:
        base_dir = "."
    
    # 1. Load Vocab (Using your specific filename: chord_vocab.json)
    vocab_path = os.path.join(base_dir, "chord_vocab.json")
    if not os.path.exists(vocab_path):
        # Fallback to parent dir if running from a specific checkpoint file
        vocab_path = os.path.join(os.path.dirname(base_dir), "chord_vocab.json")
        
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Required vocab file 'chord_vocab.json' not found in {base_dir}")
    
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    # 2. Load Config
    config_path = os.path.join(base_dir, "model_config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

    # 3. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Instantiate Model with corrected parameters
    model = LSTMChordModel(
        vocab_size=len(vocab),
        hidden_dim=config.get('hidden_dim', 32),
        n_layers=config.get('n_layers', 2),
        one_hot_scale_degree=config.get('one_hot_scale_degree', True) 
    )
    
    # Handle state dict nesting
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    print(f"🚀 Model loaded successfully from {os.path.basename(checkpoint_path)}")
    return model, vocab
# ---------------------------------------------------------------------------
# 2. Logic for Key-per-Part and Majority Voting
# ---------------------------------------------------------------------------

def get_sections(score):
    """Identifies A and B parts to allow for local key changes."""
    flat = score.parts[0].flatten() if score.parts else score.flatten()
    # Find repeat boundaries
    offsets = [0.0]
    for r in flat.getElementsByClass(music21.bar.Repeat):
        offsets.append(r.offset)
    offsets.append(flat.duration.quarterLength)
    offsets = sorted(list(set(offsets)))
    
    sections = []
    for i in range(len(offsets)-1):
        s = music21.stream.Stream()
        notes = [n for n in flat.notes if n.offset >= offsets[i] and n.offset < offsets[i+1]]
        for n in notes:
            s.insert(n.offset - offsets[i], n)
        local_key = s.analyze('key')
        sections.append({'start': offsets[i], 'end': offsets[i+1], 'key': local_key})
    return sections

# ---------------------------------------------------------------------------
# 3. ABC Annotation Logic
# ---------------------------------------------------------------------------

def annotate_abc(original_file, measure_chords):
    """Injects chords into the ABC text."""
    with open(original_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    m_count = 1
    for line in lines:
        if line.startswith(('X:', 'T:', 'K:', 'M:', 'L:', 'R:', 'Q:', '%%')):
            new_lines.append(line)
            continue
        
        # Simple measure-based injection
        segments = line.split('|')
        new_segments = []
        for seg in segments:
            if seg.strip() and m_count in measure_chords:
                chord = measure_chords[m_count]
                new_segments.append(f' "{chord}"{seg}')
                m_count += 1
            else:
                new_segments.append(seg)
        new_lines.append('|'.join(new_segments))
    
    out_path = original_file.replace('.abc', '_predicted.abc')
    with open(out_path, 'w') as f:
        f.writelines(new_lines)
    print(f"\n✅ Annotated ABC saved to: {out_path}")

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("abc_file")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load
    model, vocab = load_fixed_model(args.checkpoint, device)
    score = music21.converter.parse(args.abc_file)
    
    # 2. Key Analysis per Part
    sections = get_sections(score)
    
    # 3. Predict
    # (Simplified for brevity: extract features and run model)
    features = _extract_features_from_score(score)
    X, _ = tune_to_arrays(features, vocab=None, normalize=True)
    X_tensor = torch.FloatTensor(X).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(X_tensor, torch.tensor([len(X)]))
        pred_indices = output.argmax(dim=-1).squeeze().tolist()
    
    # Map back to vocab
    # Map back to vocab (Handling string vs int keys)
    # Ensure all keys in inv_vocab are integers
    inv_vocab = {int(k): v for k, v in vocab.items()} 
    
    try:
        pred_degrees = [inv_vocab[idx] for idx in pred_indices]
    except KeyError as e:
        print(f"Error: Predicted index {e} not found in vocabulary.")
        print(f"Available indices: {list(inv_vocab.keys())}")
        sys.exit(1)
    # 4. Majority Vote per Measure
    m_chords = {}
    m_votes = collections.defaultdict(list)
    m_tonics = {}

    for feat, deg in zip(features, pred_degrees):
        m_num = feat['measure']
        m_votes[m_num].append(deg)
        m_tonics[m_num] = feat['key_tonic_pc']

    for m_num, votes in m_votes.items():
        top_degree = collections.Counter(votes).most_common(1)[0][0]
        m_chords[m_num] = degree_to_chord(top_degree, m_tonics[m_num])
        print(f"Measure {m_num}: {m_chords[m_num]} ({top_degree})")

    # 5. Annotate
    annotate_abc(args.abc_file, m_chords)

if __name__ == "__main__":
    main()