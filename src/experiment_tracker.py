"""
Experiment tracking with Git integration for abc2chord.

Saves a structured JSON entry for every run to experiments/run_history.json,
tagged with the commit hash so results are always reproducible.
"""
import json
import os
import subprocess
import datetime

_HISTORY_FILE = "run_history.json"


def get_git_info():
    """
    Returns (commit_hash, short_hash, is_dirty).
    Falls back to ("unknown", "unknown", False) when Git is unavailable.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        short = commit[:7]
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit, short, bool(status)
    except Exception:
        return "unknown", "unknown", False


def _load_history(path):
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_history(path, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


class ExperimentTracker:
    """
    Usage:
        tracker = ExperimentTracker()
        tracker.start_run(vars(args))
        for epoch in ...:
            tracker.log_epoch(epoch, train_loss, val_loss, val_acc, lr)
        tracker.finish_run(final_train_loss, best_val_loss, final_lr, best_model_path)
    """

    def __init__(self, experiments_dir="experiments"):
        self.experiments_dir = experiments_dir
        self.history_path = os.path.join(experiments_dir, _HISTORY_FILE)
        self._run = None
        self._commit = "unknown"
        self._short = "unknown"
        self._dirty = False

    # ── Public API ──────────────────────────────────────────────────────────

    def start_run(self, args_dict):
        """
        Call once before training begins. Captures git state and prints a
        warning if the working tree has uncommitted changes.

        Returns the short commit hash for use in filenames.
        """
        self._commit, self._short, self._dirty = get_git_info()
        self._print_header()

        self._run = {
            "run_id": f"{self._short}_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}",
            "timestamp": datetime.datetime.now().isoformat(),
            "commit": self._commit,
            "short_commit": self._short,
            "dirty_worktree": self._dirty,
            "args": args_dict,
            "epoch_losses": [],
            "final_train_loss": None,
            "best_val_loss": None,
            "final_lr": None,
            "best_model_path": None,
            "status": "running",
        }
        return self._short

    def versioned_model_name(self, prefix="best_model"):
        """Return a filename like best_model_a1b2c3d.pt (or _dirty suffix)."""
        suffix = "_dirty" if self._dirty else ""
        return f"{prefix}_{self._short}{suffix}.pt"

    def log_epoch(self, epoch, train_loss, val_loss=None, val_acc=None, lr=None):
        """Append one epoch's metrics. Called inside the training loop."""
        if self._run is None:
            return
        entry = {"epoch": epoch, "train_loss": round(float(train_loss), 6)}
        if val_loss is not None:
            entry["val_loss"] = round(float(val_loss), 6)
        if val_acc is not None:
            entry["val_acc"] = round(float(val_acc), 4)
        if lr is not None:
            entry["lr"] = lr
        self._run["epoch_losses"].append(entry)

    def finish_run(self, final_train_loss, best_val_loss, final_lr, best_model_path):
        """Persist the completed run to run_history.json."""
        if self._run is None:
            return
        self._run["final_train_loss"] = round(float(final_train_loss), 6)
        self._run["best_val_loss"] = (
            round(float(best_val_loss), 6) if best_val_loss is not None else None
        )
        self._run["final_lr"] = final_lr
        self._run["best_model_path"] = best_model_path
        self._run["status"] = "completed"
        self._append_to_history()
        print(f"\n[tracker] Run logged → {self.history_path}  (id: {self._run['run_id']})")

    # ── Internals ────────────────────────────────────────────────────────────

    def _print_header(self):
        bar = "─" * 60
        print(bar)
        print(f"  abc2chord experiment tracker")
        print(f"  Commit : {self._commit}")
        if self._dirty:
            print("  ⚠  WARNING: DIRTY_WORKTREE — uncommitted changes detected.")
            print("     Results tagged as dirty and may not be fully reproducible.")
        print(bar)

    def _append_to_history(self):
        history = _load_history(self.history_path)
        history.append(self._run)
        _save_history(self.history_path, history)
