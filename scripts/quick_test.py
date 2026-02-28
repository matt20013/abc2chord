#!/usr/bin/env python3
"""
Inference Probe — run the best checkpoint on a single validation tune and
print a side-by-side truth table with musical diagnosis.

Usage (from project root):
    python scripts/quick_test.py --checkpoints checkpoints \
        --abc-paths data/maplewood.abc data/maplewood_other.abc \
        [--tune-index N]   # omit for random val tune
        [--val-fraction 0.2]
"""
import os
import sys
import argparse
import random
import math
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import warnings
warnings.filterwarnings("ignore")

import numpy as np

# ── ANSI colours ────────────────────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── Interval names for scale degree display ──────────────────────────────────
_SD_NAMES = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"]

# Harmonically acceptable substitutions in folk/Irish music.
# Any pair here is treated as a "near miss" rather than a hard error.
_ACCEPTABLE_SUBS = {
    frozenset({"G",  "Em"}),
    frozenset({"D",  "Bm"}),
    frozenset({"A",  "Gbm"}),
    frozenset({"C",  "Am"}),
    frozenset({"F",  "Dm"}),
    frozenset({"Bb", "Gm"}),
    frozenset({"Eb", "Cm"}),
    frozenset({"A7", "A"}),
    frozenset({"D7", "D"}),
    frozenset({"G7", "G"}),
    frozenset({"E7", "E"}),
    frozenset({"C7", "C"}),
}


def _is_acceptable(truth: str, pred: str) -> bool:
    return frozenset({truth, pred}) in _ACCEPTABLE_SUBS


def _find_best_checkpoint(checkpoints_dir: str) -> str:
    """Return path to the best model .pt file, falling back to lstm_chord.pt."""
    # Prefer versioned best_model_*.pt
    candidates = sorted(glob.glob(os.path.join(checkpoints_dir, "best_model*.pt")))
    if candidates:
        return candidates[-1]
    # Fall back to final model
    fallback = os.path.join(checkpoints_dir, "lstm_chord.pt")
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(
        f"No model checkpoint found in {checkpoints_dir}. "
        "Run train_lstm.py first."
    )


def main():
    parser = argparse.ArgumentParser(description="Quick inference probe on one val tune")
    parser.add_argument("--checkpoints",  default="checkpoints")
    parser.add_argument("--abc-paths",    nargs="*", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--tune-index",   type=int,   default=None,
                        help="Index into val set (0-based). Omit for random.")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="RNG seed for the train/val split (must match training).")
    parser.add_argument("--no-color",     action="store_true")
    args = parser.parse_args()

    if args.no_color:
        global RED, YELLOW, GREEN, CYAN, BOLD, RESET
        RED = YELLOW = GREEN = CYAN = BOLD = RESET = ""

    # ── Torch + model ────────────────────────────────────────────────────────
    try:
        import torch
    except ImportError:
        print("PyTorch is required.", file=sys.stderr)
        return 1

    from src.model import LSTMChordModel, tune_to_arrays, get_input_dim
    from src.parser import (
        _iter_scores_from_abc, _extract_features_from_score,
        chord_to_degree, degree_to_chord,
    )
    from src.training_data import tune_has_chords, TRAINING_ABC_FILES
    from src.chord_encoding import decode_target_to_chord

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_dir   = args.checkpoints
    model_path = _find_best_checkpoint(ckpt_dir)
    config_path = os.path.join(ckpt_dir, "model_config.json")

    import json

    # Load state dict first so we can infer the actual saved architecture.
    # This is more reliable than trusting model_config.json, which may have
    # been overwritten by a more recent (untrained) run.
    state_dict = torch.load(model_path, map_location="cpu")

    def _infer_cfg(sd):
        """Read architecture hyperparameters directly from weight shapes."""
        ih_l0   = sd["lstm.weight_ih_l0"]          # (4*H, input_dim)
        fc_w    = sd["fc.weight"]                   # (num_classes, fc_in)
        hidden  = ih_l0.shape[0] // 4
        inp     = ih_l0.shape[1]
        bidir   = "lstm.weight_ih_l0_reverse" in sd
        fc_in   = fc_w.shape[1]
        n_cls   = fc_w.shape[0]
        # Count layers
        layers  = sum(1 for k in sd if k.startswith("lstm.weight_ih_l") and
                      not k.endswith("_reverse"))
        return {
            "input_dim": inp, "hidden_dim": hidden, "num_layers": layers,
            "num_classes": n_cls, "bidirectional": bidir,
        }

    inferred = _infer_cfg(state_dict)

    # Start from config file defaults, override with inferred shape truth
    cfg = {"dropout": 0.3, "one_hot_scale_degree": True, "target_type": "absolute"}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg.update(json.load(f))
    cfg.update(inferred)   # weight shapes are ground truth

    # Derive one_hot_scale_degree from input_dim heuristic if not in config
    if "one_hot_scale_degree" not in cfg or cfg.get("input_dim") != \
            (get_input_dim(cfg.get("one_hot_scale_degree", True))):
        cfg["one_hot_scale_degree"] = (cfg["input_dim"] == get_input_dim(True))

    sd_onehot   = cfg["one_hot_scale_degree"]
    target_type = cfg.get("target_type", "absolute")
    degree_mode = (target_type == "degree")

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))

    model = LSTMChordModel(
        input_dim   = cfg["input_dim"],
        hidden_dim  = cfg["hidden_dim"],
        num_layers  = cfg["num_layers"],
        num_classes = cfg["num_classes"],
        dropout     = cfg.get("dropout", 0.3),
        bidirectional = cfg.get("bidirectional", True),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    mode_tag = f"{CYAN}degree mode{RESET}" if degree_mode else "absolute mode"
    print(f"{CYAN}Loaded{RESET} {model_path}  (input_dim={cfg['input_dim']}, "
          f"classes={cfg['num_classes']}, target={mode_tag}, device={device})")

    # ── Replicate val split (identical seed + logic as train_lstm.py) ────────
    abc_paths = args.abc_paths or TRAINING_ABC_FILES
    abc_paths = [p for p in abc_paths if os.path.isfile(p)]
    if not abc_paths:
        print("No ABC files found. Pass --abc-paths.", file=sys.stderr)
        return 1

    all_scores = []
    for path in abc_paths:
        try:
            for score in _iter_scores_from_abc(path):
                ds = _extract_features_from_score(score)
                if tune_has_chords(ds):
                    all_scores.append(score)
        except Exception:
            continue

    random.seed(args.seed)
    random.shuffle(all_scores)
    n_val = max(1, int(len(all_scores) * args.val_fraction))
    val_scores = all_scores[:n_val]

    if not val_scores:
        print("Validation set is empty.", file=sys.stderr)
        return 1

    # Pick a tune
    if args.tune_index is not None:
        idx = args.tune_index % len(val_scores)
    else:
        idx = random.randint(0, len(val_scores) - 1)

    score = val_scores[idx]
    tune  = _extract_features_from_score(score)

    # In degree mode: remap ground-truth labels to Roman numerals so they
    # match what the model was trained to predict.
    tune_tonic_pc = tune[0].get("key_tonic_pc", 2) if tune else 2
    tune_key_label = tune[0].get("key_label", "D major (default)") if tune else "D major"
    if degree_mode:
        for row in tune:
            tpc = row.get("key_tonic_pc", tune_tonic_pc)
            row["target_chord"] = chord_to_degree(row["target_chord"], tpc)

    # Try to get the tune title
    title = "Unknown"
    try:
        md = score.metadata
        if md and md.title:
            title = md.title
    except Exception:
        pass

    key_info = f"  {CYAN}Key: {tune_key_label}{RESET}" if degree_mode else ""
    print(f"\n{BOLD}Tune {idx+1}/{len(val_scores)}: {title}{RESET}{key_info}\n")

    # ── Inference ────────────────────────────────────────────────────────────
    X, _ = tune_to_arrays(tune, normalize=True,
                          one_hot_scale_degree=sd_onehot)
    if len(X) == 0:
        print("Tune has no notes.", file=sys.stderr)
        return 1

    X_t = torch.from_numpy(X).float().unsqueeze(0).to(device)  # (1, T, F)
    lengths = torch.tensor([len(X)], dtype=torch.long)

    with torch.no_grad():
        logits = model(X_t, lengths=lengths)          # (1, T, C)
        probs  = torch.softmax(logits, dim=-1)[0]     # (T, C)
        preds  = probs.argmax(dim=-1).cpu().numpy()   # (T,)
        conf   = probs.max(dim=-1).values.cpu().numpy()

    # ── Truth table ──────────────────────────────────────────────────────────
    # In degree mode the Ground Truth / Predicted columns show "V7 (A7)" —
    # the degree label followed by the absolute chord name in parentheses.
    col_w = [6, 10, 18, 18, 10]
    gt_header  = "Ground Truth" + ("  (abs)" if degree_mode else "")
    pred_header = "Predicted" + ("    (abs)" if degree_mode else "")
    header = (f"{'Bar/Bt':<{col_w[0]}}  {'Note (Deg)':<{col_w[1]}}  "
              f"{gt_header:<{col_w[2]}}  {pred_header:<{col_w[3]}}  "
              f"{'Conf %':>{col_w[4]}}")
    sep = "─" * len(header)

    print(sep)
    print(f"{BOLD}{header}{RESET}")
    print(sep)

    n_correct    = 0
    n_total      = 0
    n_acceptable = 0
    confident_errors = []   # (bar, beat, truth, pred, conf)
    pred_counter: dict[str, int] = {}
    truth_counter: dict[str, int] = {}
    error_pairs:  list[tuple[str, str]] = []

    import music21.pitch

    hierarchical = cfg.get("hierarchical_targets", False)

    prev_chord = None
    for i, row in enumerate(tune):
        truth = row["target_chord"]

        # Decode prediction using decode_target_to_chord
        prob_vec = probs[i].cpu().numpy()
        key_tonic = row.get("key_tonic_pc", tune_tonic_pc)
        pred_chord_str = decode_target_to_chord(prob_vec, key_tonic, hierarchical=hierarchical)

        if degree_mode:
            pred = chord_to_degree(pred_chord_str, key_tonic)
        else:
            pred = pred_chord_str

        c     = float(conf[i]) * 100.0
        note_midi = int(row["pitch"])
        sd        = int(row.get("scale_degree", 0))

        note_name = music21.pitch.Pitch(note_midi).nameWithOctave

        bar  = row.get("measure", "?")
        beat = row.get("beat", "?")
        beat_str = f"{beat:.0f}" if isinstance(beat, float) else str(beat)
        pos  = f"{bar}/{beat_str}"
        note_col = f"{note_name}({_SD_NAMES[sd]})"

        # In degree mode: build display strings with absolute name in parens
        # e.g. "V7 (A7)" — lets the musician read the table without a lookup table
        if degree_mode:
            row_tonic = row.get("key_tonic_pc", tune_tonic_pc)
            truth_abs = degree_to_chord(truth, row_tonic)
            pred_abs  = degree_to_chord(pred,  row_tonic)
            truth_disp = f"{truth}({truth_abs})"
            pred_disp  = f"{pred}({pred_abs})"
        else:
            truth_disp = truth
            pred_disp  = pred
            truth_abs  = truth
            pred_abs   = pred

        # Stats (compare degree labels in degree mode, absolute labels otherwise)
        truth_counter[truth] = truth_counter.get(truth, 0) + 1
        if truth not in ("N.C.",):
            n_total += 1
            pred_counter[pred] = pred_counter.get(pred, 0) + 1
            correct = (truth == pred)
            # Acceptable-substitution check: use absolute names in both modes
            acceptable = (not correct) and _is_acceptable(truth_abs, pred_abs)
            if correct:
                n_correct += 1
            elif acceptable:
                n_acceptable += 1
            else:
                error_pairs.append((truth_disp, pred_disp))
            if not correct and c >= 85.0:
                confident_errors.append((bar, beat_str, truth_disp, pred_disp, c))

        # Row formatting
        if truth == pred:
            pred_col = f"{GREEN}{pred_disp:<{col_w[3]}}{RESET}"
            row_prefix = "  "
        elif acceptable:
            pred_col = f"{YELLOW}{pred_disp:<{col_w[3]}}{RESET}"
            row_prefix = "~ "
        else:
            pred_col = f"{RED}{pred_disp:<{col_w[3]}}{RESET}"
            row_prefix = "✗ "

        conf_str = f"{c:>9.1f}%"
        if not (truth == pred) and c >= 85.0:
            conf_str = f"{RED}{BOLD}{conf_str}{RESET}"

        print(f"{row_prefix}{pos:<{col_w[0]}}  {note_col:<{col_w[1]}}  "
              f"{truth_disp:<{col_w[2]}}  {pred_col}  {conf_str}")

    print(sep)

    # ── Summary stats ────────────────────────────────────────────────────────
    acc          = n_correct / n_total if n_total else 0.0
    near_miss_pct = n_acceptable / n_total if n_total else 0.0
    hard_errors  = n_total - n_correct - n_acceptable

    print(f"\n{BOLD}── Summary ─────────────────────────────────────────────────{RESET}")
    print(f"  Notes evaluated : {n_total}")
    print(f"  {GREEN}Correct         : {n_correct}  ({acc:.1%}){RESET}")
    print(f"  {YELLOW}Near miss (~)   : {n_acceptable}  ({near_miss_pct:.1%})  "
          f"[harmonically acceptable substitution]{RESET}")
    print(f"  {RED}Hard errors     : {hard_errors}  "
          f"({hard_errors/n_total:.1%}){RESET}" if n_total else "")

    if confident_errors:
        print(f"\n{BOLD}{RED}── Confident Errors (≥85% confidence, wrong) ───────────────{RESET}")
        for bar, beat, truth, pred, c in confident_errors:
            print(f"  Bar {bar} beat {beat}:  truth={truth}  pred={RED}{pred}{RESET}  "
                  f"conf={RED}{c:.1f}%{RESET}")

    # ── Musical Diagnosis ────────────────────────────────────────────────────
    print(f"\n{BOLD}── Musical Diagnosis ───────────────────────────────────────{RESET}")

    # Most over-predicted chord
    if pred_counter:
        top_pred = max(pred_counter, key=pred_counter.get)
        top_truth = max(truth_counter, key=truth_counter.get) if truth_counter else "?"
        tonic_bias = pred_counter.get(top_truth, 0) / n_total if n_total else 0
        if tonic_bias > 0.55:
            print(f"  {RED}⚠  Tonic over-prediction: model guesses '{top_truth}' "
                  f"{tonic_bias:.0%} of the time.{RESET}")
        else:
            print(f"  Most predicted chord : {top_pred} "
                  f"({pred_counter[top_pred]} times)")

    # Most confused pairs
    if error_pairs:
        from collections import Counter
        pair_counts = Counter(error_pairs)
        top_pair, count = pair_counts.most_common(1)[0]
        truth_p, pred_p = top_pair
        print(f"  Most confused pair   : {truth_p} → predicted as {pred_p} "
              f"({count}x)")

        # IV→V / I→V confusion heuristic
        iv_v = sum(1 for t, p in error_pairs
                   if (t in ("G","C","F") and p in ("A","D","E","G","C"))
                   or (t in ("A","D","E") and p in ("G","C","F")))
        if iv_v > 2:
            print(f"  {YELLOW}Pattern: IV↔V confusion detected ({iv_v} cases) — "
                  f"model may be struggling with cadential motion.{RESET}")

    # Near-miss commentary
    if near_miss_pct > 0.1:
        print(f"  {YELLOW}Note: {near_miss_pct:.0%} of 'errors' are harmonically acceptable "
              f"substitutions (e.g. G↔Em, D↔Bm).{RESET}")
        print(f"  Effective musical accuracy ≈ "
              f"{BOLD}{(n_correct + n_acceptable)/n_total:.1%}{RESET}")

    if acc >= 0.70:
        print(f"  {GREEN}✓ Model is tracking harmony well on this tune.{RESET}")
    elif acc >= 0.50:
        print(f"  {YELLOW}Model is partially tracking harmony. "
              f"Check bar-level transitions.{RESET}")
    else:
        print(f"  {RED}Model is struggling — likely defaulting to the most common chord.{RESET}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
