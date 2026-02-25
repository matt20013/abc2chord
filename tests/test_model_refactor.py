import unittest
import numpy as np
import torch
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import tune_to_arrays, ChordSequenceDataset, LSTMChordModel, train_epoch, eval_epoch

class TestModelRefactor(unittest.TestCase):
    def setUp(self):
        # Create a dummy tune
        self.tune = [
            {
                "duration": 1.0, "beat": 1.0, "measure": 1.0, "is_rest": 0.0,
                "scale_degree": 0, "meter_norm": 0.5,
                "target_chord": "C", "key_tonic_pc": 0
            },
            {
                "duration": 1.0, "beat": 2.0, "measure": 1.0, "is_rest": 0.0,
                "scale_degree": 7, "meter_norm": 0.5,
                "target_chord": "G", "key_tonic_pc": 0
            },
            # N.C. case
            {
                "duration": 1.0, "beat": 3.0, "measure": 1.0, "is_rest": 0.0,
                "scale_degree": 4, "meter_norm": 0.5,
                "target_chord": "N.C.", "key_tonic_pc": 0
            }
        ]

    def test_tune_to_arrays(self):
        X, y = tune_to_arrays(self.tune)
        self.assertEqual(X.shape, (3, 17)) # 17 features
        self.assertEqual(y.shape, (3, 12)) # 12 targets

        # Check C Major target (0, 4, 7)
        c_vec = y[0]
        self.assertEqual(c_vec[0], 1.0)
        self.assertEqual(c_vec[4], 1.0)
        self.assertEqual(c_vec[7], 1.0)

        # Check G Major target (Dominant in C -> 7, 11, 2)
        g_vec = y[1]
        self.assertEqual(g_vec[7], 1.0)
        self.assertEqual(g_vec[11], 1.0)
        self.assertEqual(g_vec[2], 1.0)

        # Check N.C.
        nc_vec = y[2]
        self.assertEqual(nc_vec.sum(), 0.0)

    def test_dataset_padding(self):
        dataset = ChordSequenceDataset([self.tune], max_len=5)
        X, y, length = dataset[0]

        self.assertEqual(X.shape, (5, 17))
        self.assertEqual(y.shape, (5, 12))
        self.assertEqual(length, 3)

        # Check padding in y (index 3 and 4 should be -1.0)
        self.assertTrue((y[3] == -1.0).all())
        self.assertTrue((y[4] == -1.0).all())

        # Check data valid
        self.assertTrue((y[0] != -1.0).any()) # Should have some 1s or 0s

    def test_model_forward(self):
        model = LSTMChordModel(input_dim=17, hidden_dim=8, num_classes=12)
        dataset = ChordSequenceDataset([self.tune], max_len=5)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        for X, y, lengths in loader:
            logits = model(X, lengths=lengths)
            self.assertEqual(logits.shape, (1, 5, 12))

    def test_train_step(self):
        model = LSTMChordModel(input_dim=17, hidden_dim=8, num_classes=12)
        dataset = ChordSequenceDataset([self.tune], max_len=5)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())

        loss = train_epoch(model, loader, criterion, optimizer, device="cpu")
        self.assertGreater(loss, 0.0)

if __name__ == "__main__":
    unittest.main()
