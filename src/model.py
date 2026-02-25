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

# Feature keys consumed from each note dict (order defines the model's input vector).
# pitch        – MIDI semitone
# duration     – quarter-length duration
# beat         – beat position within bar
# measure      – measure number
# is_rest      – 0/1 flag (always 0 for notes; reserved for future rest-handling)
# scale_degree – chromatic interval above key tonic (0-11), captures modal context
# meter_norm   – time-sig numerator / 12, distinguishes jig / reel / polka / waltz
FEATURE_KEYS = [
    "pitch", "duration", "beat", "measure", "is_rest",
    "scale_degree", "meter_norm",
]
INPUT_DIM = len(FEATURE_KEYS)  # single source of truth; currently 7

# Pitch is MIDI 0–127; we normalize. Beat/measure can be large; we'll normalize per tune or clip.
PITCH_MIN, PITCH_MAX = 21, 108  # rough piano range


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


def tune_to_arrays(tune, vocab=None, normalize=True):
    """Convert one tune (list of feature dicts) to (features, chord_indices).

    Output feature order matches FEATURE_KEYS:
        [pitch, duration, beat, measure, is_rest, scale_degree, meter_norm]
    Older dicts (without scale_degree / meter_norm) fall back to sensible
    defaults so pre-existing checkpoints and tests remain loadable.
    """
    features = []
    chords = []
    for row in tune:
        p = row["pitch"]
        if normalize:
            p    = max(0.0, min(1.0, (float(p) - PITCH_MIN) / (PITCH_MAX - PITCH_MIN + 1e-8)))
            dur  = min(float(row["duration"]), _DURATION_MAX) / _DURATION_MAX
            beat = max(0.0, min(1.0, (float(row["beat"]) - 1.0) / (_BEAT_MAX - 1.0)))
            meas = min(float(row["measure"]), _MEASURE_MAX) / _MEASURE_MAX
            sdeg = row.get("scale_degree", 0) / _SCALE_DEG_MAX   # 0-11 → 0-1
            mtr  = float(row.get("meter_norm", 4.0 / _METER_NUM_MAX))  # default 4/4
        else:
            dur  = float(row["duration"])
            beat = float(row["beat"])
            meas = float(row["measure"])
            sdeg = float(row.get("scale_degree", 0))
            mtr  = float(row.get("meter_norm", 4.0 / _METER_NUM_MAX))
        features.append([
            p,
            dur,
            beat,
            meas,
            float(row["is_rest"]),
            sdeg,
            mtr,
        ])
        chords.append(row["target_chord"])

    X = np.array(features, dtype=np.float32)
    if vocab is not None:
        y = np.array([vocab.encode(c) for c in chords], dtype=np.int64)
        return X, y
    return X, chords


class ChordSequenceDataset(Dataset):
    """Dataset of (padded) note sequences and chord labels for LSTM."""

    def __init__(self, tunes, vocab, max_len=None, normalize=True):
        self.vocab = vocab
        self.normalize = normalize
        self.tunes = []
        self.lengths = []
        for tune in tunes:
            X, y = tune_to_arrays(tune, vocab=vocab, normalize=normalize)
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
            X = np.vstack([X, np.zeros((pad_len, X.shape[1]), dtype=np.float32)])
            y = np.concatenate([y, np.full(pad_len, self.vocab.label_to_idx[ChordVocabulary.PAD], dtype=np.int64)])
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

        def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, num_layers=2, num_classes=50,
                     dropout=0.2, bidirectional=False):
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
        # logits (B, T, C), y (B, T) -> flatten, ignore pad
        B, T, C = logits.shape
        logits_flat = logits.view(-1, C)
        y_flat = y.view(-1)
        mask = y_flat != pad_idx
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
            B, T, C = logits.shape
            logits_flat = logits.view(-1, C)
            y_flat = y.view(-1)
            mask = y_flat != pad_idx
            if mask.sum() == 0:
                continue
            loss = criterion(logits_flat[mask], y_flat[mask])
            preds = logits_flat[mask].argmax(dim=-1)
            correct += (preds == y_flat[mask]).sum().item()
            total_loss += loss.item() * mask.sum().item()
            n += mask.sum().item()
    return (total_loss / n if n else 0.0), (correct / n if n else 0.0)


def predict_chords(model, X, lengths=None, vocab=None, device=None):
    """Run model on one or more sequences; return chord indices or labels."""
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
        pred = logits.argmax(dim=-1)  # (batch, seq_len)
    pred = pred.cpu().numpy()
    if vocab is not None and pred.size > 0:
        if pred.ndim == 1:
            return [vocab.decode(int(p)) for p in pred]
        return [[vocab.decode(int(p)) for p in row] for row in pred]
    return pred


def load_model_and_vocab(checkpoint_dir, device=None):
    """Load LSTMChordModel and ChordVocabulary from a checkpoint directory."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required.")
    vocab_path = os.path.join(checkpoint_dir, "chord_vocab.json")
    model_path = os.path.join(checkpoint_dir, "lstm_chord.pt")
    config_path = os.path.join(checkpoint_dir, "model_config.json")
    vocab = ChordVocabulary.load(vocab_path)
    kwargs = {"num_classes": len(vocab)}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            kwargs.update(json.load(f))
    model = LSTMChordModel(**kwargs)
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
    labels = predict_chords(model, X, lengths=lengths, vocab=vocab, device=device)
    return labels[0] if isinstance(labels[0], list) else labels
