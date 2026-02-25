# abc2chord

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**abc2chord** is a machine learning pipeline that automatically predicts standard folk accompaniment chords for traditional melodies written in ABC notation. Instead of using complex rule-based music theory algorithms, it uses a Bidirectional LSTM trained on a corpus of human-chorded folk tunes (Irish, Scottish, Old-Time, etc.).

## Project Overview

Traditional folk music accompaniment relies heavily on functional harmony and context. `abc2chord` leverages deep learning to capture these nuances, achieving ~66% accuracy in predicting appropriate chords for monophonic melodies.

The system is designed to:
- Parse ABC notation files.
- Handle local key changes and modulations.
- Predict harmonically viable chords using a trained LSTM model.
- Export results back into ABC format or for web-based playback.

## Core Innovations

The success of this model relies on three major data-prep breakthroughs rather than brute-force deep learning:

*   **Key-Invariant Input (Scale Degrees):** The model never sees absolute pitches (like C# or Bb). Instead, the parser uses `music21` to detect the local key and converts every note into a 12-dimensional one-hot "Scale Degree" array (the interval distance from the tonic). This means a $V \to I$ cadence looks mathematically identical to the LSTM whether it's in G major or Eb major.
*   **"Aggressive" Functional Super-Classes:** Instead of predicting 35+ exact chord qualities (which causes the model to hallucinate complex jazz chords), we map the target labels to 7 Roman Numeral "Super-Classes" (I, i, V, IV, bVII, bIII, N.C.). This forces the LSTM to learn the fundamental harmonic movement of folk music.
*   **Adaptive Chunking:** Folk tunes modulate frequently (usually at the B-part). The parser dynamically segments the melody using explicit markers (`|:`, `||`, `P:`, `[K:]`) or falls back to an 8-bar heuristic grid (ignoring pickup measures). It then calculates the local tonic for each chunk so the Scale Degree math is always anchored correctly.

## Architecture

### Model
*   **Input Dimension:** 17 features per time-step:
    *   12-dim one-hot scale degree
    *   Duration
    *   Beat position
    *   Measure number
    *   `is_rest`
    *   `meter_norm`
*   **Network:** 2-Layer Bidirectional LSTM (Hidden size = 32, Dropout = 0.5) designed to look at both the past and future context of a melody phrase.
*   **Output:** Softmax distribution over the 9 vocabulary classes (7 super-classes + PAD + UNK).

## Workflow & Key Modules

1.  **Preprocessing (`src/parser.py`):** Reads ABC files, performs adaptive chunking for local keys, and extracts the 17-dim feature vectors.
2.  **Training (`scripts/train_lstm.py` & `src/model.py`):** Trains the model using a smoothed class-weighting formula to prevent common chords from dominating. Tracks experiments via Git hash (`src/experiment_tracker.py`).
3.  **Inference (`scripts/annotate_abc.py`):** Runs an un-chorded ABC file through the model, applies "Majority Voting" per measure to smooth out jitter, translates the predicted Roman Numerals back to absolute chords (e.g., V -> D), and uses regex to cleanly inject them back into the ABC text.
4.  **Web Export (`scripts/export_onnx.py`):** Wraps the PyTorch model to calculate Softmax and returns the Top-3 predictions (Top-K) per note, then exports it to a ~170KB ONNX file for the browser.

## File Structure

```text
.
├── data/                       # Training data and samples
├── reports/                    # Generated reports (conflicts, OOV chords)
├── scripts/                    # Command-line utilities
│   ├── annotate_abc.py         # Main inference script
│   ├── train_lstm.py           # Model training script
│   ├── preprocess_training_data.py
│   └── ...
├── src/                        # Core library code
│   ├── model.py                # LSTM definition and inference logic
│   ├── parser.py               # ABC parsing and feature extraction
│   ├── augment.py              # Data augmentation logic
│   └── ...
├── tests/                      # Unit tests
├── requirements.txt
└── README.md
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/abc2chord.git
    cd abc2chord
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate   # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training
To train the LSTM model on your dataset (requires `music21` and PyTorch):

```bash
python scripts/train_lstm.py --epochs 50 --out checkpoints
```

Options:
*   `--hidden 128`, `--layers 2`, `--lr 1e-3` to tune hyperparameters.
*   `--abc-paths` to specify training files.

### 2. Inference (Annotation)
To automatically add chords to an existing ABC file:

```bash
python scripts/annotate_abc.py input_tune.abc --checkpoint checkpoints/degree
```

This will generate `input_tune_predicted.abc` with the predicted chords injected inline.

### 3. Web Export
To export the trained model to ONNX format for web usage:

```bash
python scripts/export_onnx.py --model checkpoints/latest.pth --out web_model.onnx
```

## Future Roadmap: Web Integration

The ultimate goal is to drop the `chord_model.onnx` file into a React/Next.js frontend (`abc-tunebook-editor-template`). Using `onnxruntime-web` and `abcjs`, the frontend will:
1.  Extract features natively in JavaScript.
2.  Run inference with zero server cost.
3.  Offer users a dropdown of "Top 3 Chord Suggestions" for every measure.
