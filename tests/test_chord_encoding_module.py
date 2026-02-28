import unittest
import numpy as np
from src.chord_encoding import encode_chord_to_target, decode_target_to_chord, get_chord_templates

class TestChordEncoding(unittest.TestCase):
    def test_get_chord_templates_structure(self):
        # Test default flat templates
        templates = get_chord_templates(hierarchical=False)
        self.assertEqual(len(templates), 120)  # 12 roots * 10 qualities

        for template in templates:
            self.assertEqual(len(template), 3)
            root_offset, suffix, vec = template
            self.assertIsInstance(root_offset, int)
            self.assertIsInstance(suffix, str)
            self.assertIsInstance(vec, np.ndarray)
            self.assertEqual(vec.shape, (12,))
            self.assertEqual(vec.dtype, np.float32)

    def test_get_chord_templates_flat_values(self):
        templates = get_chord_templates(hierarchical=False)

        # Test Major chord at root 0 (e.g. C in key of C)
        # Expected: 1.0 at indices 0, 4, 7
        t_maj_0 = next(t for t in templates if t[0] == 0 and t[1] == "")
        expected_maj_0 = np.zeros(12, dtype=np.float32)
        expected_maj_0[[0, 4, 7]] = 1.0
        np.testing.assert_array_equal(t_maj_0[2], expected_maj_0)

        # Test Minor chord at root 2 (e.g. Dm in key of C)
        # Expected: 1.0 at indices (0+2)%12=2, (3+2)%12=5, (7+2)%12=9
        t_m_2 = next(t for t in templates if t[0] == 2 and t[1] == "m")
        expected_m_2 = np.zeros(12, dtype=np.float32)
        expected_m_2[[2, 5, 9]] = 1.0
        np.testing.assert_array_equal(t_m_2[2], expected_m_2)

    def test_get_chord_templates_hierarchical_values(self):
        templates = get_chord_templates(hierarchical=True)

        # Test Major chord at root 0
        # Expected: 1.0 at 0, 0.9 at 4, 0.6 at 7
        t_maj_0 = next(t for t in templates if t[0] == 0 and t[1] == "")
        expected_maj_0 = np.zeros(12, dtype=np.float32)
        expected_maj_0[[0, 4, 7]] = [1.0, 0.9, 0.6]
        np.testing.assert_array_almost_equal(t_maj_0[2], expected_maj_0)

        # Test Dominant 7 chord at root 7 (e.g. G7 in key of C)
        # Expected for 7: [(0, 1.0), (4, 0.9), (7, 0.5), (10, 0.8)]
        # Shifted by 7:
        # (0+7)%12 = 7   -> 1.0
        # (4+7)%12 = 11  -> 0.9
        # (7+7)%12 = 2   -> 0.5
        # (10+7)%12 = 5  -> 0.8
        t_dom7_7 = next(t for t in templates if t[0] == 7 and t[1] == "7")
        expected_dom7_7 = np.zeros(12, dtype=np.float32)
        expected_dom7_7[[7, 11, 2, 5]] = [1.0, 0.9, 0.5, 0.8]
        np.testing.assert_array_almost_equal(t_dom7_7[2], expected_dom7_7)

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
