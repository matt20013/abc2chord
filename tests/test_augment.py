import unittest
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock
from src.augment import transpose_tune
import music21

class TestAugment(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.abc_file = os.path.join(self.test_dir, "test.abc")
        with open(self.abc_file, "w") as f:
            f.write("X:1\nT:Test\nL:1/4\nK:C\nC D E F|")

        self.output_dir = os.path.join(self.test_dir, "augmented")
        os.makedirs(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("music21.converter.parse")
    def test_transpose_tune_mocked(self, mock_parse):
        # Mock score and its transpose/write methods
        mock_score = MagicMock()
        mock_transposed = MagicMock()
        mock_score.transpose.return_value = mock_transposed

        mock_parse.return_value = mock_score

        transpose_tune(self.abc_file, self.output_dir)

        # Verify calls
        mock_parse.assert_called_with(self.abc_file)
        # Should be called 12 times (range -6 to 6)
        self.assertEqual(mock_score.transpose.call_count, 12)
        self.assertEqual(mock_transposed.write.call_count, 12)

    def test_transpose_tune_integration(self):
        # Actual run with a simple file
        transpose_tune(self.abc_file, self.output_dir)

        files = os.listdir(self.output_dir)
        self.assertEqual(len(files), 12)
        for i in range(-6, 6):
            expected = f"test_transpose_{i}.musicxml"
            self.assertIn(expected, files)

if __name__ == "__main__":
    unittest.main()
