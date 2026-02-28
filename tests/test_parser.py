import unittest
import os
import shutil
import tempfile
from src.parser import (
    extract_features_from_abc,
    extract_all_tunes_from_abc,
    absolute_to_degree,
    map_chord_to_degree,
    degree_to_chord,
    simplify_chord_label,
    _get_key_timeline,
    _active_key_at,
    _sanitise_abc_body
)
import music21

class TestParser(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_abc = os.path.join(self.test_dir, "test.abc")
        with open(self.sample_abc, "w") as f:
            f.write("""X:1
T:Test Tune
M:4/4
L:1/4
K:C
"C" C C G G | "F" A A "C" G2 | "F" F F "C" E E | "G" D D "C" C2 |
""")
        self.output_dir = os.path.join(self.test_dir, "augmented")
        os.makedirs(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_extract_features(self):
        data = extract_features_from_abc(self.sample_abc)
        self.assertTrue(len(data) > 0)
        # Check first note
        first_note = data[0]
        # In test.abc: "C" C ...
        # C4 is 60
        self.assertEqual(first_note['pitch'], 60.0)
        self.assertEqual(first_note['target_chord'], 'C')
        self.assertEqual(first_note['scale_degree'], 0) # C in C major
        self.assertEqual(first_note['key_tonic_pc'], 0) # C

    def test_absolute_to_degree(self):
        # Key of C (0)
        self.assertEqual(absolute_to_degree(0, 0, ""), "I")
        self.assertEqual(absolute_to_degree(7, 0, ""), "V")
        self.assertEqual(absolute_to_degree(7, 0, "7"), "V7")
        self.assertEqual(absolute_to_degree(9, 0, "m"), "vi")
        self.assertEqual(absolute_to_degree(5, 0, ""), "IV")

        # Key of G (7)
        self.assertEqual(absolute_to_degree(7, 7, ""), "I")
        self.assertEqual(absolute_to_degree(0, 7, ""), "IV")
        self.assertEqual(absolute_to_degree(2, 7, "7"), "V7")

    def test_map_chord_to_degree(self):
        # Using pitch class integer
        self.assertEqual(map_chord_to_degree("G", 7), "I")
        self.assertEqual(map_chord_to_degree("C", 7), "IV")

        # Using root string
        self.assertEqual(map_chord_to_degree("G", "G"), "I")
        self.assertEqual(map_chord_to_degree("Am", "G"), "ii")

        # Using music21 key object
        k = music21.key.Key("G")
        self.assertEqual(map_chord_to_degree("D7", k), "V7")

    def test_degree_to_chord(self):
        # Edge Cases
        self.assertEqual(degree_to_chord("", 0), "")
        self.assertEqual(degree_to_chord(None, 0), None)
        self.assertEqual(degree_to_chord("N.C.", 0), "N.C.")
        self.assertEqual(degree_to_chord("XYZ", 0), "XYZ")

        # Basic Chords (Key of C)
        self.assertEqual(degree_to_chord("I", 0), "C")
        self.assertEqual(degree_to_chord("ii", 0), "Dm")
        self.assertEqual(degree_to_chord("IV", 0), "F")
        self.assertEqual(degree_to_chord("V", 0), "G")
        self.assertEqual(degree_to_chord("vi", 0), "Am")

        # Qualities (Key of C)
        self.assertEqual(degree_to_chord("V7", 0), "G7")
        self.assertEqual(degree_to_chord("ii7", 0), "Dm7")
        self.assertEqual(degree_to_chord("viidim", 0), "Bdim")
        self.assertEqual(degree_to_chord("IIIaug", 0), "Eaug")

        # Flat Roots (Key of C)
        self.assertEqual(degree_to_chord("bVII", 0), "Bb")
        self.assertEqual(degree_to_chord("bIII", 0), "Eb")
        self.assertEqual(degree_to_chord("bvi", 0), "Abm")

        # Different Tonic (Key of D, pc=2)
        self.assertEqual(degree_to_chord("I", 2), "D")
        self.assertEqual(degree_to_chord("V7", 2), "A7")
        self.assertEqual(degree_to_chord("vi", 2), "Bm")
        self.assertEqual(degree_to_chord("IV", 2), "G")

    def test_simplify_chord_label(self):
        # Relaxed mode
        self.assertEqual(simplify_chord_label("Am7", relaxed=True), "Am")
        self.assertEqual(simplify_chord_label("Cmaj7", relaxed=True), "C")
        self.assertEqual(simplify_chord_label("G7", relaxed=True), "G7")
        self.assertEqual(simplify_chord_label("Dadd9", relaxed=True), "D")
        self.assertEqual(simplify_chord_label("Edim7", relaxed=True), "Edim")

        # Strict mode
        self.assertEqual(simplify_chord_label("Am7", relaxed=False), "Am7")
        self.assertEqual(simplify_chord_label("Cmaj7", relaxed=False), "Cmaj7")
        self.assertEqual(simplify_chord_label("G7", relaxed=False), "G7")

        # Normalization
        self.assertEqual(simplify_chord_label("C#", relaxed=True), "Db")
        self.assertEqual(simplify_chord_label("A#m", relaxed=True), "Bbm")
        self.assertEqual(simplify_chord_label("(C)", relaxed=True), "C")
        self.assertEqual(simplify_chord_label("C/E", relaxed=True), "C")

    def test_key_timeline_and_active_key(self):
        # Create a score with key change
        s = music21.stream.Score()
        p = music21.stream.Part()
        k1 = music21.key.Key("C")
        # Use insert to respect offsets
        p.insert(0.0, k1)
        k2 = music21.key.Key("G")
        p.insert(4.0, k2)
        s.insert(0.0, p)

        timeline = _get_key_timeline(s)
        self.assertEqual(len(timeline), 2)
        self.assertEqual(timeline[0][1], 0) # C
        self.assertEqual(timeline[1][1], 7) # G

        self.assertEqual(_active_key_at(timeline, 0.0), (0, 'C major'))
        self.assertEqual(_active_key_at(timeline, 3.0), (0, 'C major'))
        self.assertEqual(_active_key_at(timeline, 4.0), (7, 'G major'))
        self.assertEqual(_active_key_at(timeline, 10.0), (7, 'G major'))

    def test_sanitise_abc_body(self):
        abc_content = """X:1
K:C
C D E F | x4 | y4 | "Am"A B c d |
"""
        sanitized = _sanitise_abc_body(abc_content)
        self.assertNotIn("x4", sanitized)
        self.assertNotIn("y4", sanitized)
        self.assertIn("A B c d", sanitized)

    def test_extract_all_tunes_from_abc(self):
        multi_abc = os.path.join(self.test_dir, "multi.abc")
        with open(multi_abc, "w") as f:
            f.write("""X:1
T:Tune 1
L:1/4
K:C
C C C C |

X:2
T:Tune 2
L:1/4
K:G
G G G G |
""")
        tunes = list(extract_all_tunes_from_abc(multi_abc))
        self.assertEqual(len(tunes), 2)
        self.assertEqual(tunes[0][0]['key_tonic_pc'], 0)
        self.assertEqual(tunes[1][0]['key_tonic_pc'], 7)

if __name__ == '__main__':
    unittest.main()
