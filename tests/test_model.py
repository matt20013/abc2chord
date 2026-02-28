import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import os
import torch
import tempfile
import json
from src.model import (
    ChordVocabulary,
    tune_to_arrays,
    ChordSequenceDataset,
    collate_padded,
    LSTMChordModel,
    train_epoch,
    eval_epoch,
    predict_chords,
    load_model_and_vocab,
    predict_chords_from_tune,
    INPUT_DIM,
    get_input_dim
)

class TestChordVocabulary(unittest.TestCase):
    def setUp(self):
        self.vocab = ChordVocabulary()

    def test_initial_state(self):
        self.assertEqual(len(self.vocab), 2)
        self.assertEqual(self.vocab.label_to_idx[ChordVocabulary.PAD], 0)
        self.assertEqual(self.vocab.label_to_idx[ChordVocabulary.UNK], 1)

    def test_add_label(self):
        self.vocab.add_label("C")
        self.assertEqual(len(self.vocab), 3)
        self.assertEqual(self.vocab.label_to_idx["C"], 2)
        self.assertEqual(self.vocab.idx_to_label[2], "C")

    def test_fit(self):
        tunes = [[{"target_chord": "C"}, {"target_chord": "G"}], [{"target_chord": "Am"}]]
        self.vocab.fit(tunes)
        self.assertIn("C", self.vocab.label_to_idx)
        self.assertIn("G", self.vocab.label_to_idx)
        self.assertIn("Am", self.vocab.label_to_idx)

    def test_encode_decode(self):
        self.vocab.add_label("C")
        idx = self.vocab.encode("C")
        self.assertEqual(idx, 2)
        label = self.vocab.decode(2)
        self.assertEqual(label, "C")

        # Test unknown
        idx_unk = self.vocab.encode("Db")
        self.assertEqual(idx_unk, 1)
        label_unk = self.vocab.decode(999)
        self.assertEqual(label_unk, ChordVocabulary.UNK)

    def test_save_load(self):
        self.vocab.add_label("C")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vocab.json")
            self.vocab.save(path)

            loaded_vocab = ChordVocabulary.load(path)
            self.assertEqual(len(loaded_vocab), 3)
            self.assertEqual(loaded_vocab.encode("C"), 2)

class TestTuneToArrays(unittest.TestCase):
    def setUp(self):
        self.vocab = ChordVocabulary()
        self.vocab.add_label("C")
        self.vocab.add_label("G")

    def test_tune_to_arrays_normalized_one_hot(self):
        tune = [
            {"duration": 1.0, "beat": 1.0, "measure": 1, "is_rest": 0, "scale_degree": 0, "target_chord": "C", "meter_norm": 0.5},
            {"duration": 0.5, "beat": 2.0, "measure": 1, "is_rest": 0, "scale_degree": 7, "target_chord": "G", "meter_norm": 0.5}
        ]
        X, y = tune_to_arrays(tune, vocab=self.vocab, normalize=True, one_hot_scale_degree=True)

        self.assertEqual(X.shape, (2, 17)) # 4 base + 12 one-hot + 1 meter
        self.assertEqual(y.shape, (2, 12)) # 12-dim Multi-Hot

        # Check scale degree one-hot
        # First note: scale_degree 0 -> index 4 (duration, beat, measure, is_rest take 0-3)
        self.assertEqual(X[0, 4], 1.0)
        # Second note: scale_degree 7 -> index 4+7 = 11
        self.assertEqual(X[1, 11], 1.0)

    def test_tune_to_arrays_unnormalized_scalar(self):
        tune = [
            {"duration": 4.0, "beat": 1.0, "measure": 1, "is_rest": 0, "scale_degree": 0, "target_chord": "C", "meter_norm": 0.5}
        ]
        X, y = tune_to_arrays(tune, vocab=None, normalize=False, one_hot_scale_degree=False)

        self.assertEqual(X.shape, (1, 6)) # 4 base + 1 scalar + 1 meter
        self.assertEqual(X[0, 0], 4.0) # duration
        self.assertEqual(X[0, 4], 0.0) # scale degree scalar (0/11)
        self.assertEqual(y.shape, (1, 12)) # 12-dim Multi-Hot

class TestChordSequenceDataset(unittest.TestCase):
    def setUp(self):
        self.vocab = ChordVocabulary()
        self.vocab.add_label("C")
        self.tune = [
            {"duration": 1.0, "beat": 1.0, "measure": 1, "is_rest": 0, "scale_degree": 0, "target_chord": "C"}
        ]

    def test_dataset_padding(self):
        tunes = [self.tune, self.tune] # 2 tunes
        dataset = ChordSequenceDataset(tunes, self.vocab, max_len=5)

        self.assertEqual(len(dataset), 2)
        X, y, length = dataset[0]

        self.assertEqual(X.shape[0], 5) # padded length
        self.assertEqual(length, 1)     # original length

        # Check padding (after index 0)
        self.assertTrue(torch.all(X[1:] == 0))
        # y should be padded with -1.0 vectors
        self.assertTrue(torch.all(y[1:] == -1.0))

class TestLSTMChordModel(unittest.TestCase):
    def setUp(self):
        self.model = LSTMChordModel(input_dim=17, hidden_dim=8, num_layers=1, num_classes=12)

    def test_forward_pass(self):
        batch_size = 2
        seq_len = 10
        input_dim = 17
        X = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([10, 5])

        logits = self.model(X, lengths=lengths)

        self.assertEqual(logits.shape, (batch_size, seq_len, 12))

class TestTrainingFunctions(unittest.TestCase):
    def setUp(self):
        self.model = LSTMChordModel(input_dim=17, hidden_dim=8, num_layers=1, num_classes=12)
        # Using BCEWithLogitsLoss for Multi-Hot Target
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.device = torch.device("cpu")

        # Mock data
        X = torch.randn(2, 5, 17)
        # y needs to be (Batch, Seq, 12) float
        y = torch.randn(2, 5, 12)
        lengths = torch.tensor([5, 5])
        self.loader = [(X, y, lengths)]

    def test_train_epoch(self):
        loss = train_epoch(self.model, self.loader, self.criterion, self.optimizer, self.device)
        self.assertIsInstance(loss, float)

    def test_eval_epoch(self):
        loss, acc = eval_epoch(self.model, self.loader, self.criterion, self.device)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)

class TestInference(unittest.TestCase):
    def setUp(self):
        self.vocab = ChordVocabulary()
        self.vocab.add_label("C")
        self.model = LSTMChordModel(input_dim=17, hidden_dim=8, num_classes=12)

    def test_predict_chords(self):
        X = torch.randn(1, 5, 17)
        lengths = torch.tensor([5])

        logits = predict_chords(self.model, X, lengths=lengths, vocab=self.vocab)
        self.assertEqual(len(logits), 1)
        self.assertEqual(len(logits[0]), 5)
        # predict_chords now returns logits, so it should be a numpy array of shape (1, 5, 12)
        self.assertEqual(logits.shape, (1, 5, 12))

    def test_predict_chords_default_lengths(self):
        """Verify predict_chords handles default lengths=None with torch tensor."""
        X = torch.randn(1, 5, 17)
        # We wrap forward to check if it was called with lengths=None
        with patch.object(self.model, 'forward', wraps=self.model.forward) as mock_forward:
            logits = predict_chords(self.model, X, lengths=None)
            mock_forward.assert_called_once()
            _, kwargs = mock_forward.call_args
            self.assertIsNone(kwargs.get('lengths'))
            self.assertEqual(logits.shape, (1, 5, 12))

    def test_predict_chords_numpy_default_lengths(self):
        """Verify predict_chords handles default lengths=None with numpy array."""
        X_np = np.random.randn(5, 17).astype(np.float32)
        with patch.object(self.model, 'forward', wraps=self.model.forward) as mock_forward:
            logits = predict_chords(self.model, X_np, lengths=None)
            mock_forward.assert_called_once()
            _, kwargs = mock_forward.call_args
            self.assertIsNone(kwargs.get('lengths'))
            self.assertEqual(logits.shape, (1, 5, 12))

    def test_predict_chords_from_tune(self):
        tune = [
            {"duration": 1.0, "beat": 1.0, "measure": 1, "is_rest": 0, "scale_degree": 0, "target_chord": "C", "key_tonic_pc": 0}
        ]
        preds = predict_chords_from_tune(self.model, tune, self.vocab)
        self.assertEqual(len(preds), 1)
        self.assertIsInstance(preds[0], str)

class TestLoadModel(unittest.TestCase):
    def test_load_model_and_vocab(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy vocab
            vocab = ChordVocabulary()
            vocab.add_label("C")
            vocab.save(os.path.join(tmpdir, "chord_vocab.json"))

            # Create dummy model with correct num_classes
            model = LSTMChordModel(num_classes=12)
            torch.save(model.state_dict(), os.path.join(tmpdir, "lstm_chord.pt"))

            # Create dummy config
            with open(os.path.join(tmpdir, "model_config.json"), "w") as f:
                json.dump({"hidden_dim": 32}, f)

            loaded_model, loaded_vocab = load_model_and_vocab(tmpdir)
            self.assertIsInstance(loaded_model, LSTMChordModel)
            self.assertIsInstance(loaded_vocab, ChordVocabulary)
            self.assertEqual(loaded_model.hidden_dim, 32)
            self.assertEqual(loaded_model.num_classes, 12)

class TestHelpers(unittest.TestCase):
    def test_get_input_dim(self):
        self.assertEqual(get_input_dim(True), 17)
        self.assertEqual(get_input_dim(False), 6)

    def test_collate_padded(self):
        batch = [
            (torch.randn(5, 10), torch.randn(5, 12), 5),
            (torch.randn(5, 10), torch.randn(5, 12), 5)
        ]
        X, y, lengths = collate_padded(batch)
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[2], 12)
        self.assertEqual(lengths.shape[0], 2)

if __name__ == "__main__":
    unittest.main()
