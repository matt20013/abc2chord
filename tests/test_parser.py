import unittest
import os
import shutil
import sys
# Ensure src is in path
sys.path.append(os.getcwd())

from src.parser import extract_features_from_abc
from src.augment import transpose_tune

class TestParser(unittest.TestCase):
    def setUp(self):
        self.sample_abc = "data/sample.abc"
        self.output_dir = "data/augmented_test"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def test_extract_features(self):
        data = extract_features_from_abc(self.sample_abc)
        self.assertTrue(len(data) > 0)
        # Check first note
        first_note = data[0]
        # In data/sample.abc: "C" C ...
        # C4 is 60
        self.assertEqual(first_note['pitch'], 60.0)
        self.assertEqual(first_note['target_chord'], 'C')

    def test_augmentation(self):
        transpose_tune(self.sample_abc, self.output_dir)
        files = [f for f in os.listdir(self.output_dir) if f.endswith('.musicxml')]
        self.assertTrue(len(files) >= 12)

        # Test parsing one of the augmented files
        # sample_transpose_2.musicxml -> +2 semitones
        test_file = os.path.join(self.output_dir, "sample_transpose_2.musicxml")

        # Ensure file exists before parsing
        self.assertTrue(os.path.exists(test_file))

        data = extract_features_from_abc(test_file)
        self.assertTrue(len(data) > 0)
        # Expect transposition by 2 semitones (C -> D)
        first_note = data[0]
        self.assertEqual(first_note['pitch'], 62.0)
        self.assertEqual(first_note['target_chord'], 'D')

    def tearDown(self):
        # clean up output dir
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

if __name__ == '__main__':
    unittest.main()
