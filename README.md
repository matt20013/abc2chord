# abc2chord

## Setup (venv)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then use `python` and `pip` as usual. To train the LSTM:

```bash
python scripts/train_lstm.py --epochs 50 --out checkpoints
```

## Training data

Training data is read from a local clone of [matt20013/tunes](https://github.com/matt20013/tunes). Clone it next to this repo:

```bash
git clone https://github.com/matt20013/tunes.git ../tunes
```

The loader uses `../tunes/abcs/maplewood.abc` and `../tunes/abcs/maplewood_other.abc`. Tunes without chords are disregarded.

Alternatively, to copy the ABC files into `data/` and use those:

```bash
python scripts/download_training_data.py
```

Then call `load_training_data(abc_paths=["data/maplewood.abc", "data/maplewood_other.abc"])` (or set paths relative to a `data_dir`).

## LSTM chord prediction

An LSTM model predicts the chord label at each note from a sequence of note features (pitch, duration, beat, measure, rest).

**Train** (from project root, with `../tunes` clone and PyTorch installed):

```bash
pip install torch
python scripts/train_lstm.py --epochs 50 --out checkpoints
```

Options: `--hidden 128`, `--layers 2`, `--lr 1e-3`, `--batch-size 8`, `--abc-paths file1.abc file2.abc`.

**Use the model:**

```python
from src.model import load_model_and_vocab, predict_chords_from_tune
from src.parser import extract_features_from_abc

model, vocab = load_model_and_vocab("checkpoints", device="cpu")
tune = extract_features_from_abc("path/to/tune.abc")
predicted_chords = predict_chords_from_tune(model, tune, vocab)
# predicted_chords[i] is the predicted chord for tune[i]
```

## Preprocessing for Random Forest (8th-note grid CSV)

A separate pipeline builds a single CSV for a Random Forest classifier: one row per **8th-note beat**, with forward-filled chord labels and 12-key augmentation.

**Logic:**

- **Parsing:** music21 parses the ABC; melody from the first voice, chord symbols from the score.
- **Alignment (forward fill):** For every 8th-note beat, the “active chord” is the last chord seen at or before that time (sparse ABC chords are latched forward).
- **Features per row:** `pitch_midi` (0 if rest), `duration`, `beat_pos` (1.0, 1.5, 2.0, … in the measure), `key_sig` (tonic), `prev_note` (MIDI of the previous beat’s note), target `active_chord`.
- **Augmentation:** Each tune is transposed into all 12 keys before extraction (12× the number of tunes).

**Run** (from project root; requires `music21` and `pandas`):

```bash
python scripts/preprocess_training_data.py path/to/your_135_tunes.abc --out training_data.csv
```

Options: `--out FILE` (default `training_data.csv`), `--no-augment` (disable 12-key transposition). Output columns: `pitch_midi`, `duration`, `beat_pos`, `key_sig`, `prev_note`, `active_chord`.