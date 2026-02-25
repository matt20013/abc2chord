import unittest
import music21
import sys
import os

# Ensure src module is importable
sys.path.insert(0, os.getcwd())

from src.parser import _extract_features_from_score, _get_adaptive_key_timeline

class TestAdaptiveKey(unittest.TestCase):
    def test_modulation_detection(self):
        # Create a stream with a modulation
        s = music21.stream.Score()
        p = music21.stream.Part()

        # Section 1: C Major (C D E F G A B C)
        # 4 measures of C Major
        m1 = music21.stream.Measure()
        m1.append(music21.note.Note("C4", type='quarter'))
        m1.append(music21.note.Note("D4", type='quarter'))
        m1.append(music21.note.Note("E4", type='quarter'))
        m1.append(music21.note.Note("F4", type='quarter'))
        p.append(m1)

        m2 = music21.stream.Measure()
        m2.append(music21.note.Note("G4", type='quarter'))
        m2.append(music21.note.Note("A4", type='quarter'))
        m2.append(music21.note.Note("B4", type='quarter'))
        m2.append(music21.note.Note("C5", type='quarter'))
        p.append(m2)

        # Double barline to signal section change
        m2.rightBarline = music21.bar.Barline(style='double')

        # Section 2: G Major (G A B C D E F# G)
        # 4 measures of G Major
        m3 = music21.stream.Measure()
        m3.append(music21.note.Note("G4", type='quarter'))
        m3.append(music21.note.Note("A4", type='quarter'))
        m3.append(music21.note.Note("B4", type='quarter'))
        m3.append(music21.note.Note("C5", type='quarter'))
        p.append(m3)

        m4 = music21.stream.Measure()
        m4.append(music21.note.Note("D5", type='quarter'))
        m4.append(music21.note.Note("E5", type='quarter'))
        m4.append(music21.note.Note("F#5", type='quarter'))
        m4.append(music21.note.Note("G5", type='quarter'))
        p.append(m4)

        s.append(p)

        # Run extraction
        features = _extract_features_from_score(s)

        # Check tonics
        tonics = [f['key_tonic_pc'] for f in features]
        unique_tonics = set(tonics)

        self.assertTrue(len(unique_tonics) > 1, "Should detect at least two keys")

        # Ideally check values
        first_half = tonics[:8]
        second_half = tonics[8:]

        # Check that first half has one dominant key and second half has another
        # Or at least they are different
        self.assertNotEqual(set(first_half), set(second_half))

    def test_adaptive_chunking_heuristic(self):
        # Test 8-bar fallback
        # 16 bars of C Major, no internal barlines except regular ones
        s = music21.stream.Score()
        p = music21.stream.Part()

        for i in range(16):
            m = music21.stream.Measure()
            # C Major chord to be unambiguous
            m.append(music21.note.Note("C4"))
            m.append(music21.note.Note("E4"))
            m.append(music21.note.Note("G4"))
            p.append(m)

        s.append(p)

        # This should result in one chunk (merged) or two chunks of same key
        timeline = _get_adaptive_key_timeline(s)
        # Expect 1 entry if merged, or multiple entries with same key
        # My implementation merges adjacent same keys.
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0][1], 0) # C Major (tonic 0)

    def test_pickup_handling(self):
        # 1 measure of pickup (incomplete)
        s = music21.stream.Score()
        p = music21.stream.Part()

        # Pickup: 1 quarter note in 4/4
        m0 = music21.stream.Measure()
        m0.timeSignature = music21.meter.TimeSignature('4/4')
        m0.append(music21.note.Note("C4", type='quarter'))
        p.append(m0)

        # Rest of score
        m1 = music21.stream.Measure()
        m1.append(music21.note.Note("C4", type='whole'))
        p.append(m1)

        s.append(p)

        # Check that pickup is detected?
        # Assuming implementation handles it silently.
        # But features should have correct key.
        features = _extract_features_from_score(s)
        self.assertTrue(len(features) > 0)

if __name__ == '__main__':
    unittest.main()
