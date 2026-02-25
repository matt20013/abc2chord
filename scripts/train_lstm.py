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
    get_input_dim,
)

try:
    import torch
except ImportError:
    torch = None


def main():
    parser = argparse.ArgumentParser(description="Train LSTM chord model")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=8)
    parser.add_argument("--hidden",        type=int,   default=32,   help="LSTM hidden size per direction")
    parser.add_argument("--layers",        type=int,   default=2,    help="LSTM layers")
    parser.add_argument("--dropout",       type=float, default=0.5,  help="Dropout probability (LSTM + output layer)")
    parser.add_argument("--weight-decay",  type=float, default=1e-4, help="L2 weight decay for Adam")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--lr-patience",  type=int,   default=10)
    parser.add_argument("--lr-factor",    type=float, default=0.5)
    parser.add_argument("--min-lr",       type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2,  help="Fraction of tunes held out for validation")
    parser.add_argument("--out",          type=str,   default="checkpoints")
    parser.add_argument("--abc-paths",    nargs="*",  default=None, help="ABC files (default: training set)")
    parser.add_argument("--augment-keys", action="store_true",
                        help="Transpose every tune into all 12 keys. "
                             "Not recommended: scale-degree one-hot already provides "
                             "key-invariance; augmentation adds 12x compute with no new signal.")
    parser.add_argument("--bidirectional",        action=argparse.BooleanOptionalAction, default=True,
                        help="Bidirectional LSTM (default: on; use --no-bidirectional to disable)")
    parser.add_argument("--scale-degree-onehot",  action=argparse.BooleanOptionalAction, default=True,
                        help="One-hot encode scale degree (12-dim). Default: on. "
                             "Use --no-scale-degree-onehot for single normalised float.")
    parser.add_argument("--target-type", choices=["absolute", "degree", "aggressive"],
                        default="degree",
                        help="Chord label space for training. "
                             "'degree' (default) = Roman-numeral degrees (~33 classes). "
                             "'aggressive' = 7 functional super-classes "
                             "(I, i, V, IV, bVII, bIII, N.C.) — best for small datasets. "
                             "'absolute' = raw chord names — ~35 classes.")
    parser.add_argument("--target-mode",
                        choices=["absolute", "degree", "aggressive"], default=None,
                        help="Alias for --target-type.")
    parser.add_argument("--relaxed", action=argparse.BooleanOptionalAction, default=True,
                        help="Relaxed chord simplification (default: on). "
                             "Collapses m7→m and maj7→root for a smaller, denser vocabulary. "
                             "Use --no-relaxed to keep m7 and maj7 as distinct classes.")
    args = parser.parse_args()
    # --target-mode is an alias; it wins if both are specified.
    if args.target_mode is not None:
        args.target_type = args.target_mode

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
        chord_to_degree,
        apply_super_class,
        set_relaxed_mode,
    )
    from src.training_data import tune_has_chords

    # Apply relaxed mode globally before any parsing begins so every
    # simplify_chord_label call uses the right collapsing strategy.
    set_relaxed_mode(args.relaxed)
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

    # ── Label conversion ──────────────────────────────────────────────────────
    if args.target_type in ("degree", "aggressive"):
        for tune in train_tunes + val_tunes:
            for row in tune:
                tpc = row.get("key_tonic_pc", 2)
                deg = chord_to_degree(row["target_chord"], tpc)
                if args.target_type == "aggressive":
                    deg = apply_super_class(deg)
                row["target_chord"] = deg
        mode_label = ("Roman-numeral degrees" if args.target_type == "degree"
                      else "functional super-classes (I/i/V/IV/bVII/bIII)")
        print(f"[{args.target_type} mode] Converted target chords → {mode_label}.")

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

    sd_onehot = args.scale_degree_onehot
    input_dim = get_input_dim(sd_onehot)

    # ── Datasets & loaders ──────────────────────────────────────────────────
    train_dataset = ChordSequenceDataset(train_tunes, vocab, normalize=True,
                                         one_hot_scale_degree=sd_onehot)
    val_dataset   = ChordSequenceDataset(val_tunes, vocab,
                                         max_len=train_dataset.max_len, normalize=True,
                                         one_hot_scale_degree=sd_onehot)
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
        input_dim=input_dim,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_classes=num_classes,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    bidir_tag = "bidirectional" if args.bidirectional else "unidirectional"
    sd_tag = "onehot-12" if sd_onehot else "scalar"
    relax_tag = "relaxed" if args.relaxed else "strict"
    print(f"Model: {args.layers}-layer {bidir_tag} LSTM, hidden={args.hidden}, "
          f"dropout={args.dropout}, wd={args.weight_decay}, "
          f"scale_degree={sd_tag}, input_dim={input_dim}, "
          f"target={args.target_type}/{relax_tag} ({num_classes} classes)  "
          + ("← 7 super-classes" if args.target_type == "aggressive" else ""))

    pad_idx = vocab.label_to_idx[ChordVocabulary.PAD]
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        ignore_index=pad_idx,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_lr,
    )

    # ── Checkpoints ──────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    vocab.save(os.path.join(args.out, "chord_vocab.json"))
    model_config = {
        "input_dim": input_dim, "hidden_dim": args.hidden,
        "num_layers": args.layers, "num_classes": num_classes,
        "dropout": args.dropout, "bidirectional": args.bidirectional,
        "one_hot_scale_degree": sd_onehot,
        "target_type": args.target_type,
        "relaxed": args.relaxed,
    }
    with open(os.path.join(args.out, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved vocabulary and config to {args.out}/")

    # ── Training loop ────────────────────────────────────────────────────────
    best_metric = float("inf")  # val_loss if val exists, else train_loss
    best_epoch = 0
    best_path = os.path.join(args.out, tracker.versioned_model_name("best_model"))

    # Initialise so the interrupt handler always has valid values even if
    # Ctrl+C is pressed before the first epoch completes.
    train_loss = float("nan")
    val_loss, val_acc = None, None
    last_epoch = 0

    try:
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
            last_epoch = epoch

            if monitor < best_metric:
                best_metric = monitor
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)

    except KeyboardInterrupt:
        print("\n")
        interrupted_path = os.path.join(args.out, "interrupted_model.pt")
        torch.save(model.state_dict(), interrupted_path)
        final_lr = optimizer.param_groups[0]["lr"]
        tracker.finish_run(
            final_train_loss=train_loss if last_epoch > 0 else None,
            best_val_loss=best_metric if (val_loader and best_epoch > 0) else None,
            final_lr=final_lr,
            best_model_path=interrupted_path,
            status="interrupted",
        )
        print(f"Training interrupted. Progress saved.")
        print(f"  Completed epochs : {last_epoch}/{args.epochs}")
        print(f"  Weights saved to : {interrupted_path}")
        if best_epoch > 0:
            print(f"  Best model at    : {best_path}  (epoch {best_epoch})")
        return 0

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
