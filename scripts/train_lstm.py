#!/usr/bin/env python3
"""
Train the LSTM chord-prediction model on ABC training data.
Run from project root: python scripts/train_lstm.py [options]
"""
import os
import sys
import argparse
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.experiment_tracker import ExperimentTracker
from src.training_data import (
    load_training_tunes,
    train_val_split,
    calculate_class_weights,
    TRAINING_ABC_FILES,
)
from src.model import (
    ChordVocabulary,
    ChordSequenceDataset,
    LSTMChordModel,
    train_epoch,
    eval_epoch,
    collate_padded,
    TORCH_AVAILABLE,
    INPUT_DIM,
)

try:
    import torch
except ImportError:
    torch = None


def main():
    parser = argparse.ArgumentParser(description="Train LSTM chord model")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--hidden",       type=int,   default=128,  help="LSTM hidden size")
    parser.add_argument("--layers",       type=int,   default=2,    help="LSTM layers")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--lr-patience",  type=int,   default=10)
    parser.add_argument("--lr-factor",    type=float, default=0.5)
    parser.add_argument("--min-lr",       type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.1,  help="Fraction of tunes held out for validation")
    parser.add_argument("--out",          type=str,   default="checkpoints")
    parser.add_argument("--abc-paths",    nargs="*",  default=None, help="ABC files (default: training set)")
    parser.add_argument("--augment-keys", action="store_true", help="Transpose every tune into all 12 keys")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
    args = parser.parse_args()

    if not TORCH_AVAILABLE or torch is None:
        print("PyTorch is required: pip install torch", file=sys.stderr)
        return 1

    # ── Experiment tracking ──────────────────────────────────────────────────
    tracker = ExperimentTracker(experiments_dir=os.path.join(ROOT, "experiments"))
    tracker.start_run(vars(args))

    # ── Data ────────────────────────────────────────────────────────────────
    abc_paths = args.abc_paths or TRAINING_ABC_FILES
    from src.parser import (
        _extract_features_from_score,
        _iter_scores_from_abc,
        print_chord_mapping_trace,
    )
    from src.training_data import tune_has_chords
    import random

    # Collect all raw scores across all files so we can do a clean split
    all_scores = []
    for path in abc_paths:
        if not os.path.isfile(path):
            continue
        try:
            for score in _iter_scores_from_abc(path):
                ds = _extract_features_from_score(score)
                if tune_has_chords(ds):
                    all_scores.append(score)
        except Exception:
            continue

    if not all_scores:
        print("No tunes found. Check ABC paths:", file=sys.stderr)
        return 1

    # Shuffle with fixed seed and split before any augmentation
    random.seed(42)
    random.shuffle(all_scores)
    n_val = max(1, int(len(all_scores) * args.val_fraction))
    val_scores  = all_scores[:n_val]
    train_scores = all_scores[n_val:]

    # Val: original key only
    val_tunes = [_extract_features_from_score(s) for s in val_scores]

    # Train: augment through all 12 keys if requested
    train_tunes = []
    semitone_range = range(12) if args.augment_keys else range(1)
    for score in train_scores:
        for semitones in semitone_range:
            s = score.transpose(semitones) if semitones != 0 else score
            ds = _extract_features_from_score(s)
            if tune_has_chords(ds):
                train_tunes.append(ds)

    # Emit the first-100-events mapping trace now that all scores have been parsed
    print_chord_mapping_trace()

    print(f"Split: {len(train_scores)} train tunes → {len(train_tunes)} sequences | "
          f"{len(val_scores)} val tunes → {len(val_tunes)} sequences")

    if not train_tunes:
        print("No training tunes after augmentation.", file=sys.stderr)
        return 1

    # ── Vocabulary & weights ────────────────────────────────────────────────
    vocab = ChordVocabulary().fit(train_tunes)
    num_classes = len(vocab)
    class_weights = calculate_class_weights(train_tunes, vocab)
    print(f"Chord vocabulary: {num_classes} classes")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Datasets & loaders ──────────────────────────────────────────────────
    train_dataset = ChordSequenceDataset(train_tunes, vocab, normalize=True)
    val_dataset   = ChordSequenceDataset(val_tunes,   vocab,
                                         max_len=train_dataset.max_len, normalize=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=collate_padded,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=min(args.batch_size, len(val_dataset)),
        shuffle=False,
        collate_fn=collate_padded,
    ) if val_tunes else None

    # ── Model ────────────────────────────────────────────────────────────────
    model = LSTMChordModel(
        input_dim=INPUT_DIM,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_classes=num_classes,
        dropout=0.2,
        bidirectional=args.bidirectional,
    ).to(device)
    bidir_tag = "bidirectional" if args.bidirectional else "unidirectional"
    print(f"Model: {args.layers}-layer {bidir_tag} LSTM, hidden={args.hidden}")

    pad_idx = vocab.label_to_idx[ChordVocabulary.PAD]
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        ignore_index=pad_idx,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_lr,
    )

    # ── Checkpoints ──────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    vocab.save(os.path.join(args.out, "chord_vocab.json"))
    model_config = {
        "input_dim": INPUT_DIM, "hidden_dim": args.hidden,
        "num_layers": args.layers, "num_classes": num_classes,
        "dropout": 0.2, "bidirectional": args.bidirectional,
    }
    with open(os.path.join(args.out, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved vocabulary and config to {args.out}/")

    # ── Training loop ────────────────────────────────────────────────────────
    best_metric = float("inf")  # val_loss if val exists, else train_loss
    best_epoch = 0
    best_path = os.path.join(args.out, tracker.versioned_model_name("best_model"))
    epoch_losses = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, pad_idx=pad_idx)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(train_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        lr_tag = f"  lr {current_lr:.2e}→{new_lr:.2e}" if new_lr != current_lr else f"  lr={current_lr:.2e}"

        val_loss, val_acc = None, None
        if val_loader:
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device, pad_idx=pad_idx)
            line = (f"Epoch {epoch}/{args.epochs}  "
                    f"train={train_loss:.4f}  val={val_loss:.4f}  "
                    f"val_acc={val_acc:.1%}{lr_tag}")
            monitor = val_loss
        else:
            line = f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}{lr_tag}"
            monitor = train_loss

        print(line)
        tracker.log_epoch(epoch, train_loss, val_loss=val_loss, val_acc=val_acc, lr=new_lr)

        if monitor < best_metric:
            best_metric = monitor
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    final_lr = optimizer.param_groups[0]["lr"]
    print(f"\nBest {'val' if val_loader else 'train'} loss {best_metric:.4f} "
          f"at epoch {best_epoch} -> {best_path}")
    final_path = os.path.join(args.out, "lstm_chord.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")

    tracker.finish_run(
        final_train_loss=train_loss,
        best_val_loss=best_metric if val_loader else None,
        final_lr=final_lr,
        best_model_path=best_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
