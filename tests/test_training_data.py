import unittest
import os
import shutil
import tempfile
import torch
from unittest.mock import patch, MagicMock
from src.training_data import (
    load_training_data,
    load_training_tunes,
    train_val_split,
    calculate_class_weights,
    _resolve_abc_paths,
    tune_has_chords,
    NO_CHORD
)
from src.model import ChordVocabulary

class TestTrainingData(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.abc_file = os.path.join(self.test_dir, "test.abc")
        with open(self.abc_file, "w") as f:
            f.write("""X:1
T:Test
L:1/4
K:C
"C" C D E F |
""")
        self.vocab = ChordVocabulary()
        self.vocab.add_label("C")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_tune_has_chords(self):
        dataset_with_chords = [{"target_chord": "C"}, {"target_chord": "N.C."}]
        dataset_no_chords = [{"target_chord": "N.C."}, {"target_chord": "N.C."}]
        dataset_empty = []

        self.assertTrue(tune_has_chords(dataset_with_chords))
        self.assertFalse(tune_has_chords(dataset_no_chords))
        self.assertFalse(tune_has_chords(dataset_empty))

    def test_resolve_abc_paths(self):
        # Test explicit paths
        paths = [self.abc_file]
        resolved = _resolve_abc_paths(paths)
        self.assertEqual(resolved, paths)

        # Test default paths (should return existing files from TRAINING_ABC_FILES)
        # We can mock os.path.isfile to simulate default files existing
        with patch("os.path.isfile") as mock_isfile:
            mock_isfile.return_value = True
            resolved = _resolve_abc_paths(None)
            self.assertTrue(len(resolved) > 0)

    def test_load_training_data(self):
        data = load_training_data(abc_paths=[self.abc_file])
        self.assertTrue(len(data) > 0)
        self.assertEqual(data[0]["target_chord"], "C")

    def test_load_training_tunes(self):
        tunes = load_training_tunes(abc_paths=[self.abc_file], augment_keys=False)
        self.assertEqual(len(tunes), 1)
        self.assertEqual(tunes[0][0]["target_chord"], "C")

        # Test with augmentation (mocking score.transpose to speed up?)
        # Actually with just 1 tune and short content, real augmentation is fast enough.
        tunes_aug = load_training_tunes(abc_paths=[self.abc_file], augment_keys=True)
        self.assertEqual(len(tunes_aug), 12)

    def test_train_val_split(self):
        tunes = list(range(100))
        train, val = train_val_split(tunes, val_fraction=0.1, seed=42)
        self.assertEqual(len(train) + len(val), 100)
        self.assertTrue(len(val) >= 1)
        self.assertTrue(len(train) > len(val))

        # Verify deterministic split
        train2, val2 = train_val_split(tunes, val_fraction=0.1, seed=42)
        self.assertEqual(train, train2)
        self.assertEqual(val, val2)

    def test_calculate_class_weights(self):
        tunes = [[{"target_chord": "C"}, {"target_chord": "C"}]]
        # Vocab has PAD, UNK, C
        self.vocab.add_label("C")

        weights = calculate_class_weights(tunes, self.vocab)
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(len(weights), len(self.vocab))

        # PAD weight should be 0
        pad_idx = self.vocab.label_to_idx[ChordVocabulary.PAD]
        self.assertEqual(weights[pad_idx], 0.0)

if __name__ == "__main__":
    unittest.main()
