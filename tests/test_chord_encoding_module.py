import unittest
import numpy as np
from src.chord_encoding import encode_chord_to_target, decode_target_to_chord

class TestChordEncoding(unittest.TestCase):
    def test_encode_simple(self):
        # C Major in C (Tonic)
        # C (0), E (4), G (7) -> Relative: 0, 4, 7
        target = encode_chord_to_target("C", 0)
        expected = np.zeros(12, dtype=np.float32)
        expected[[0, 4, 7]] = 1.0
        np.testing.assert_array_equal(target, expected)

    def test_encode_dominant(self):
        # G Major in C (Dominant)
        # G (7), B (11), D (2) -> Relative: 7, 11, 2
        target = encode_chord_to_target("G", 0)
        expected = np.zeros(12, dtype=np.float32)
        expected[[7, 11, 2]] = 1.0
        np.testing.assert_array_equal(target, expected)

    def test_encode_minor_seventh(self):
        # Am7 in C (Submediant)
        # A (9), C (0), E (4), G (7) -> Relative: 9, 0, 4, 7
        target = encode_chord_to_target("Am7", 0)
        expected = np.zeros(12, dtype=np.float32)
        expected[[9, 0, 4, 7]] = 1.0
        np.testing.assert_array_equal(target, expected)

    def test_encode_nc(self):
        target = encode_chord_to_target("N.C.", 0)
        expected = np.zeros(12, dtype=np.float32)
        np.testing.assert_array_equal(target, expected)

    def test_decode_exact(self):
        # Create a perfect G7 vector relative to C (Root=G=7)
        # G7 = G, B, D, F -> 7, 11, 2, 5
        vec = np.zeros(12, dtype=np.float32)
        vec[[7, 11, 2, 5]] = 1.0

        chord = decode_target_to_chord(vec, 0) # Key C
        self.assertEqual(chord, "G7")

    def test_decode_shifted_key(self):
        # Create a perfect D7 vector relative to G (Root=D=2 relative to C, but in G key?)
        # Wait, decode_target_to_chord takes a relative vector.
        # If Key is G (7), and we want D7 (Dominant), the root is D (2).
        # Relative root = (2 - 7) % 12 = 7.
        # So we expect a vector with root at offset 7.
        # D7 notes: D (2), F# (6), A (9), C (0).
        # Relative to G (7):
        # D -> (2-7)%12 = 7
        # F# -> (6-7)%12 = 11
        # A -> (9-7)%12 = 2
        # C -> (0-7)%12 = 5
        # Indices: 7, 11, 2, 5. Same relative structure as G7 in C.

        vec = np.zeros(12, dtype=np.float32)
        vec[[7, 11, 2, 5]] = 1.0

        chord = decode_target_to_chord(vec, 7) # Key G
        self.assertEqual(chord, "D7")

    def test_decode_noisy(self):
        # G7 with some noise
        vec = np.array([0.1, 0.0, 0.9, 0.0, 0.0, 0.8, 0.0, 0.9, 0.1, 0.0, 0.1, 0.9], dtype=np.float32)
        # Indices close to 1: 2 (D), 5 (F), 7 (G), 11 (B).
        # Should be G7 (in C).
        chord = decode_target_to_chord(vec, 0)
        self.assertEqual(chord, "G7")

if __name__ == "__main__":
    unittest.main()
