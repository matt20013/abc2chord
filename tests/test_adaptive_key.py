import unittest
import music21
import sys
import os

# Ensure src module is importable
sys.path.insert(0, os.getcwd())

from src.parser import _get_adaptive_key_timeline, _extract_features_from_score

class TestAdaptiveKey(unittest.TestCase):
    def test_modulation_detection_abc(self):
        # ABC string with a modulation:
        # First 4 bars: C Major
        # Double barline ||
        # Next 4 bars: G Major (DISTINCTIVE: using F#)
        abc_data = """X:1
T:Modulation Test
M:4/4
L:1/4
K:C
C E G c | C E G c | C E G c | C E G c ||
G B d ^f | G B d ^f | G B d ^f | G B d ^f |]
"""
        s = music21.converter.parse(abc_data, format='abc')

        timeline = _get_adaptive_key_timeline(s)

        self.assertTrue(len(timeline) >= 2, f"Expected at least 2 key segments, got {len(timeline)}")

        # Check first segment
        t1_offset, t1_pc, t1_label = timeline[0]
        self.assertEqual(t1_offset, 0.0)
        self.assertEqual(t1_pc, 0) # C Major

        # Check second segment (should be at offset 16.0)
        t2_offset, t2_pc, t2_label = timeline[1]
        self.assertAlmostEqual(t2_offset, 16.0, delta=0.1)
        self.assertEqual(t2_pc, 7) # G Major

    def test_adaptive_chunking_heuristic_abc(self):
        # Test the fallback grid logic.
        # 16 bars total.
        # music21 might parse the first measure as 0 (Pickup) if not careful.
        # So we accept split at 32.0 (if M1) or 36.0 (if M0).

        abc_data = """X:2
T:Grid Fallback Test
M:4/4
L:1/4
K:C
"""
        # 8 bars of C Major
        c_bars = "C E G c | " * 8
        # 8 bars of F# Major (distinctive)
        fs_bars = "^F ^A ^c ^f | " * 8
        abc_data += c_bars + fs_bars

        s = music21.converter.parse(abc_data, format='abc')

        timeline = _get_adaptive_key_timeline(s)

        self.assertTrue(len(timeline) >= 2, f"Grid fallback failed to split segments. Got {len(timeline)}")

        # First segment: C Major (0)
        self.assertEqual(timeline[0][1], 0)

        # Second segment: F# Major (6)
        split_offset = timeline[1][0]
        self.assertTrue(abs(split_offset - 32.0) < 0.1 or abs(split_offset - 36.0) < 0.1,
                        f"Split offset {split_offset} not expected (32.0 or 36.0)")
        self.assertEqual(timeline[1][1], 6)

    def test_pickup_handling_abc(self):
        # Pickup measure (1 beat) in 4/4
        # ABC: "C | C4 | ..."
        # This is definitively a pickup (shorter than 4/4).

        abc_data = """X:3
T:Pickup Test
M:4/4
L:1/4
K:C
C | C E G c | C E G c | C E G c | C E G c |
"""
        s = music21.converter.parse(abc_data, format='abc')

        timeline = _get_adaptive_key_timeline(s)

        # Check that we get a valid key analysis starting at 0.0
        self.assertTrue(len(timeline) >= 1)
        self.assertEqual(timeline[0][1], 0) # C Major

    def test_explicit_key_change_abc(self):
        # Explicit key change mid-stream.
        # Note: If music21 ABC parser fails to parse [K:D] inline, we manually insert it
        # to ensure we test the *timeline logic* given correct input structure.
        abc_data = """X:4
T:Explicit Key Change
M:4/4
L:1/4
K:C
C E G c |
K:D
D F# A d |
"""
        s = music21.converter.parse(abc_data, format='abc')

        # Check if Key('D') exists at offset 4.0
        # If not, insert it manually to simulate patch_key_changes.py / correct parsing
        part = s.parts[0] if s.parts else s
        flat = part.flatten()
        has_key_d = False
        for k in flat.getElementsByClass(music21.key.Key):
            if k.offset >= 3.9 and k.tonic.name == 'D':
                has_key_d = True
                break

        if not has_key_d:
            # Manually insert key change at start of Measure 1 (offset 4.0)
            # Find Measure 1
            measures = part.getElementsByClass(music21.stream.Measure)
            if len(measures) > 1:
                m1 = measures[1] # Measure 1
                k = music21.key.Key('D')
                m1.insert(0, k)

        timeline = _get_adaptive_key_timeline(s)

        self.assertTrue(len(timeline) >= 2, f"Timeline merged or failed: {timeline}")
        self.assertEqual(timeline[0][1], 0) # C Major
        self.assertEqual(timeline[1][1], 2) # D Major
        self.assertAlmostEqual(timeline[1][0], 4.0, delta=0.1) # Bar 2

if __name__ == '__main__':
    unittest.main()
