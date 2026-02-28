import unittest
import os
import shutil
import tempfile
from src.parser import (
    extract_features_from_abc,
    extract_all_tunes_from_abc,
    absolute_to_degree,
    map_chord_to_degree,
    simplify_chord_label,
    apply_super_class,
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

    def test_apply_super_class(self):
        # Tonic major
        self.assertEqual(apply_super_class("I"), "I")
        self.assertEqual(apply_super_class("I7"), "I")
        self.assertEqual(apply_super_class("II"), "I")

        # Minor / sub-tonic family
        self.assertEqual(apply_super_class("i"), "i")
        self.assertEqual(apply_super_class("ii"), "i")
        self.assertEqual(apply_super_class("vi"), "i")
        self.assertEqual(apply_super_class("bvi"), "i")
        self.assertEqual(apply_super_class("VIdim"), "i")

        # Dominant function
        self.assertEqual(apply_super_class("V"), "V")
        self.assertEqual(apply_super_class("V7"), "V")
        self.assertEqual(apply_super_class("v"), "V")
        self.assertEqual(apply_super_class("bv"), "V")
        self.assertEqual(apply_super_class("II7"), "V")
        self.assertEqual(apply_super_class("vii"), "V")

        # Subdominant
        self.assertEqual(apply_super_class("IV"), "IV")
        self.assertEqual(apply_super_class("iv"), "IV")

        # Modal flat-seven
        self.assertEqual(apply_super_class("bVII"), "bVII")

        # Modal flat-three
        self.assertEqual(apply_super_class("bIII"), "bIII")
        self.assertEqual(apply_super_class("bVI"), "bIII")

        # Pass-through / N.C.
        self.assertEqual(apply_super_class("N.C."), "N.C.")

        # Unmapped / Fallback (returned unchanged)
        self.assertEqual(apply_super_class("random_string"), "random_string")
        self.assertEqual(apply_super_class("VIIdim7"), "VIIdim7")
        self.assertEqual(apply_super_class("IIIaug"), "IIIaug")
        self.assertEqual(apply_super_class("bVIIdim"), "bVIIdim")

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
