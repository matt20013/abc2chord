import unittest
import os
import shutil
import tempfile
import json
from unittest.mock import patch, MagicMock, mock_open
from src.experiment_tracker import ExperimentTracker, get_git_info

class TestExperimentTracker(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(experiments_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("subprocess.check_output")
    def test_get_git_info_clean(self, mock_subprocess):
        mock_subprocess.side_effect = [
            b"abcdef1234567890\n", # rev-parse HEAD
            b"" # status --porcelain (clean)
        ]
        commit, short, dirty = get_git_info()
        self.assertEqual(commit, "abcdef1234567890")
        self.assertEqual(short, "abcdef1")
        self.assertFalse(dirty)

    @patch("subprocess.check_output")
    def test_get_git_info_dirty(self, mock_subprocess):
        mock_subprocess.side_effect = [
            b"abcdef1234567890\n",
            b" M src/model.py\n"
        ]
        commit, short, dirty = get_git_info()
        self.assertTrue(dirty)

    @patch("subprocess.check_output")
    def test_get_git_info_error(self, mock_subprocess):
        mock_subprocess.side_effect = Exception("git error")
        commit, short, dirty = get_git_info()
        self.assertEqual(commit, "unknown")
        self.assertEqual(short, "unknown")
        self.assertFalse(dirty)

    @patch("src.experiment_tracker.get_git_info")
    def test_start_run(self, mock_git):
        mock_git.return_value = ("commit", "short", False)

        args = {"lr": 0.001}
        # prompt_if_dirty=False to avoid input() call
        short = self.tracker.start_run(args, prompt_if_dirty=False)

        self.assertEqual(short, "short")
        self.assertIsNotNone(self.tracker._run)
        self.assertEqual(self.tracker._run["status"], "running")
        self.assertEqual(self.tracker._run["args"], args)

    def test_log_epoch(self):
        self.tracker._run = {"epoch_losses": []}
        self.tracker.log_epoch(1, 0.5, 0.4, 0.9, 0.001)

        entry = self.tracker._run["epoch_losses"][0]
        self.assertEqual(entry["epoch"], 1)
        self.assertEqual(entry["train_loss"], 0.5)
        self.assertEqual(entry["val_loss"], 0.4)
        self.assertEqual(entry["val_acc"], 0.9)
        self.assertEqual(entry["lr"], 0.001)

    @patch("src.experiment_tracker.get_git_info")
    def test_finish_run(self, mock_git):
        mock_git.return_value = ("commit", "short", False)
        self.tracker.start_run({}, prompt_if_dirty=False)

        self.tracker.finish_run(0.1, 0.2, 0.001, "model.pt")

        self.assertEqual(self.tracker._run["status"], "completed")
        self.assertEqual(self.tracker._run["final_train_loss"], 0.1)

        # Check if saved to file
        history_path = os.path.join(self.test_dir, "run_history.json")
        self.assertTrue(os.path.exists(history_path))

        with open(history_path) as f:
            history = json.load(f)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["run_id"], self.tracker._run["run_id"])

    def test_versioned_model_name(self):
        self.tracker._short = "abc"
        self.tracker._dirty = False
        self.assertEqual(self.tracker.versioned_model_name(), "best_model_abc.pt")

        self.tracker._dirty = True
        self.assertEqual(self.tracker.versioned_model_name(), "best_model_abc_dirty.pt")

if __name__ == "__main__":
    unittest.main()
