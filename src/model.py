"""
LSTM-based chord prediction: given a sequence of note features, predict the
chord label at each timestep.
"""
import os
import json
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.chord_encoding import encode_chord_to_target, decode_target_to_chord

# ── Feature layout ────────────────────────────────────────────────────────────
# Absolute MIDI pitch is intentionally excluded from the feature vector.
# Scale degree (relative to key tonic) already captures melodic function in a
# key-invariant way, making raw pitch redundant and a memorisation cue.
#
# Base features (4):  duration, beat, measure, is_rest
# Scale-degree block: one-hot → 12 floats  [sd_0 … sd_11]  INPUT_DIM = 17
#                     scalar  →  1 float   [sd_norm]        INPUT_DIM =  6
# + meter_norm (1)
#
# Use get_input_dim() everywhere instead of a literal.
_BASE_FEATURE_KEYS  = ["duration", "beat", "measure", "is_rest"]
_BASE_FEATURE_DIM   = len(_BASE_FEATURE_KEYS)   # 4  (meter_norm appended after SD)
_SD_KEYS_ONEHOT     = [f"sd_{i}" for i in range(12)]
_SD_KEYS_SCALAR     = ["scale_degree_norm"]


def get_input_dim(one_hot_scale_degree: bool = True) -> int:
    """Return the feature-vector length for the chosen scale-degree encoding."""
    sd_dim = 12 if one_hot_scale_degree else 1
    return _BASE_FEATURE_DIM + sd_dim + 1   # +1 for meter_norm


INPUT_DIM = get_input_dim(one_hot_scale_degree=True)   # = 17

# Kept for display / debug use outside this module (e.g. quick_test.py).
PITCH_MIN, PITCH_MAX = 21, 108


class ChordVocabulary:
    """Maps chord labels to indices. Build from training data or load from file."""

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):
        self.label_to_idx = {self.PAD: 0, self.UNK: 1}
        self.idx_to_label = {0: self.PAD, 1: self.UNK}

    def add_label(self, label):
        if label not in self.label_to_idx:
            idx = len(self.label_to_idx)
            self.label_to_idx[label] = idx
            self.idx_to_label[idx] = label

    def fit(self, tunes):
        """Build vocabulary from list of tunes (each tune = list of feature dicts)."""
        for tune in tunes:
            for row in tune:
                c = row.get("target_chord")
                if c is not None:
                    self.add_label(c)
        n_real = len(self) - 2  # exclude <PAD> and <UNK>
        print(f"[ChordVocabulary] {n_real} chord labels  "
              f"(+<PAD>+<UNK> = {len(self)} total classes)")
        return self

    def encode(self, label):
        return self.label_to_idx.get(label, self.label_to_idx[self.UNK])

    def decode(self, idx):
        return self.idx_to_label.get(idx, self.UNK)

    def __len__(self):
        return len(self.label_to_idx)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.idx_to_label, f, indent=0)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            idx_to_label = {int(k): v for k, v in json.load(f).items()}
        vocab = cls()
        vocab.idx_to_label = idx_to_label
        vocab.label_to_idx = {v: k for k, v in idx_to_label.items()}
        return vocab


_DURATION_MAX    = 8.0    # anything longer gets clipped to 1.0
_BEAT_MAX        = 8.0    # covers 7/8, 6/8, etc.
_MEASURE_MAX     = 200.0  # normalise against a large but finite value
_SCALE_DEG_MAX   = 11.0   # chromatic degrees 0-11
_METER_NUM_MAX   = 12.0   # largest expected time-sig numerator (12/8)


def tune_to_arrays(tune, vocab=None, normalize=True, one_hot_scale_degree=True):
    """Convert one tune (list of feature dicts) to (features, chord_indices).

    Absolute pitch is excluded; scale degree encodes melodic function in a
    key-invariant way, making raw MIDI pitch redundant.

    Feature vector layout:
        one_hot_scale_degree=True  (default, INPUT_DIM=17):
            [dur, beat, meas, is_rest, sd_0…sd_11, meter_norm]
        one_hot_scale_degree=False (scalar,  INPUT_DIM=6):
            [dur, beat, meas, is_rest, sd_norm,    meter_norm]

    Dicts without scale_degree / meter_norm fall back to sensible defaults
    so older data remains loadable.
    """
    features = []
    targets = []
    for row in tune:
        if normalize:
            dur  = min(float(row["duration"]), _DURATION_MAX) / _DURATION_MAX
            beat = max(0.0, min(1.0, (float(row["beat"]) - 1.0) / (_BEAT_MAX - 1.0)))
            meas = min(float(row["measure"]), _MEASURE_MAX) / _MEASURE_MAX
            mtr  = float(row.get("meter_norm", 4.0 / _METER_NUM_MAX))
        else:
            dur  = float(row["duration"])
            beat = float(row["beat"])
            meas = float(row["measure"])
            mtr  = float(row.get("meter_norm", 4.0 / _METER_NUM_MAX))

        sd_raw = int(row.get("scale_degree", 0)) % 12
        if one_hot_scale_degree:
            sd_vec = [0.0] * 12
            sd_vec[sd_raw] = 1.0
        else:
            sd_vec = [sd_raw / _SCALE_DEG_MAX]

        features.append([dur, beat, meas, float(row["is_rest"])] + sd_vec + [mtr])

        # New target encoding
        chord_str = row.get("target_chord")
        key_tonic = row.get("key_tonic_pc", 0) # Default to C if missing
        target_vec = encode_chord_to_target(chord_str, key_tonic)
        targets.append(target_vec)

    X = np.array(features, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    return X, y


class ChordSequenceDataset(Dataset):
    """Dataset of (padded) note sequences and chord labels for LSTM."""

    def __init__(self, tunes, vocab=None, max_len=None, normalize=True,
                 one_hot_scale_degree=True):
        self.vocab = vocab
        self.normalize = normalize
        self.one_hot_scale_degree = one_hot_scale_degree
        self.tunes = []
        self.lengths = []
        for tune in tunes:
            X, y = tune_to_arrays(tune, vocab=vocab, normalize=normalize,
                                  one_hot_scale_degree=one_hot_scale_degree)
            if len(X) == 0:
                continue
            self.tunes.append((X, y))
            self.lengths.append(len(X))
        self.max_len = max_len or max(self.lengths) if self.lengths else 0

    def __len__(self):
        return len(self.tunes)

    def __getitem__(self, i):
        X, y = self.tunes[i]
        seq_len = len(X)
        if seq_len < self.max_len:
            pad_len = self.max_len - seq_len
            # Pad features with 0
            X_pad = np.zeros((pad_len, X.shape[1]), dtype=np.float32)
            X = np.vstack([X, X_pad])

            # Pad y with -1.0 (mask value) for all 12 dims
            y_pad = np.full((pad_len, 12), -1.0, dtype=np.float32)
            y = np.vstack([y, y_pad])

        return (
            torch.from_numpy(X),
            torch.from_numpy(y),
            seq_len,
        )


def collate_padded(batch):
    """Collate so we get (X, y, lengths); X and y already padded in __getitem__."""
    X = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return X, y, lengths


if TORCH_AVAILABLE:

    class LSTMChordModel(nn.Module):
        """LSTM that takes a sequence of note features and predicts chord at each step."""

        def __init__(self, input_dim=INPUT_DIM, hidden_dim=32, num_layers=2, num_classes=12,
                     dropout=0.5, bidirectional=True):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.num_classes = num_classes
            self.bidirectional = bidirectional
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            # Explicit dropout applied to LSTM output before the classifier,
            # forcing the model to find multiple independent chord cues.
            self.output_dropout = nn.Dropout(dropout)
            fc_in = hidden_dim * 2 if bidirectional else hidden_dim
            self.fc = nn.Linear(fc_in, num_classes)

        def forward(self, x, lengths=None):
            # x: (batch, seq_len, input_dim)
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                packed_out, _ = self.lstm(packed)
                # total_length ensures output matches padded input (for loss alignment with y)
                out, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_out, batch_first=True, total_length=x.size(1)
                )
            else:
                out, _ = self.lstm(x)
            out = self.output_dropout(out)
            logits = self.fc(out)  # (batch, seq_len, num_classes)
            return logits


def train_epoch(model, loader, criterion, optimizer, device, pad_idx=0):
    model.train()
    total_loss = 0.0
    n = 0
    for X, y, lengths in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X, lengths=lengths)

        # logits (B, T, 12), y (B, T, 12) -> flatten
        logits_flat = logits.view(-1, 12)
        y_flat = y.view(-1, 12)

        # Mask: first element of y vector is -1.0 if padding
        mask = (y_flat[:, 0] != -1.0)

        if mask.sum() == 0:
            continue

        loss = criterion(logits_flat[mask], y_flat[mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * mask.sum().item()
        n += mask.sum().item()
    return total_loss / n if n else 0.0


def eval_epoch(model, loader, criterion, device, pad_idx=0):
    """Evaluate on a dataloader; returns (loss, accuracy) over non-pad positions."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X, lengths=lengths)

            logits_flat = logits.view(-1, 12)
            y_flat = y.view(-1, 12)

            mask = (y_flat[:, 0] != -1.0)
            if mask.sum() == 0:
                continue

            loss = criterion(logits_flat[mask], y_flat[mask])

            # Accuracy: Exact match of thresholded predictions
            probs = torch.sigmoid(logits_flat[mask])
            preds = (probs > 0.5).float()

            # Check equality across 12 dimensions
            matches = (preds == y_flat[mask]).all(dim=1)
            correct += matches.sum().item()

            total_loss += loss.item() * mask.sum().item()
            n += mask.sum().item()
    return (total_loss / n if n else 0.0), (correct / n if n else 0.0)


def predict_chords(model, X, lengths=None, vocab=None, device=None):
    """Run model on one or more sequences; return raw logits (numpy)."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for prediction.")
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        if isinstance(X, np.ndarray) and X.ndim == 2:
            X = torch.from_numpy(X).float().unsqueeze(0)
            if lengths is not None:
                lengths = torch.tensor([lengths], dtype=torch.long)
        X = X.to(device)
        logits = model(X, lengths=lengths)

    return logits.cpu().numpy()


def load_model_and_vocab(checkpoint_dir, device=None):
    """Load LSTMChordModel and ChordVocabulary from a checkpoint directory."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required.")
    vocab_path = os.path.join(checkpoint_dir, "chord_vocab.json")
    model_path = os.path.join(checkpoint_dir, "lstm_chord.pt")
    config_path = os.path.join(checkpoint_dir, "model_config.json")

    # Try to load vocab, return empty if missing
    vocab = ChordVocabulary.load(vocab_path)

    kwargs = {"num_classes": 12}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
            # Filter kwargs for LSTMChordModel
            valid_keys = {"input_dim", "hidden_dim", "num_layers", "num_classes", "dropout", "bidirectional"}
            for k, v in cfg.items():
                if k in valid_keys:
                    kwargs[k] = v

    model = LSTMChordModel(**kwargs)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device or "cpu"))
    if device is not None:
        model = model.to(device)
    return model, vocab


def predict_chords_from_tune(model, tune, vocab, device=None, normalize=True):
    """
    Predict chord labels for one tune (list of feature dicts from parser).
    Returns a list of chord label strings, one per note.
    """
    X, _ = tune_to_arrays(tune, vocab=None, normalize=normalize)
    if len(X) == 0:
        return []
    lengths = len(X)

    # logits: (1, Seq, 12)
    logits = predict_chords(model, X, lengths=lengths, vocab=vocab, device=device)
    logits = logits[0]

    labels = []
    for i, row in enumerate(tune):
        logit_vec = logits[i]
        prob_vec = 1.0 / (1.0 + np.exp(-logit_vec))

        key_tonic = row.get("key_tonic_pc", 0)
        chord_str = decode_target_to_chord(prob_vec, key_tonic)
        labels.append(chord_str)

    return labels
