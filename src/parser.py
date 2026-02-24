import music21
import sys
import argparse

def extract_features_from_abc(abc_file_path):
    """
    Parses an ABC file and creates a dataset of (Note_Context, Chord_Label).
    """
    # Load the tune
    score = music21.converter.parse(abc_file_path)

    # We assume Part 1 is melody, and Chords are embedded as 'ChordSymbol' or 'TextExpression'
    # Check if there are parts, otherwise use the score itself (if it's a single part)
    if hasattr(score, 'parts') and len(score.parts) > 0:
        melody = score.parts[0].flatten()
    else:
        melody = score.flatten()

    # Try to get chords directly from the score first
    chords = score.flatten().getElementsByClass(music21.harmony.ChordSymbol)
    if not chords:
         # Fallback to chordify if no explicit ChordSymbols found (though chordify is usually for creating chords from notes)
         # In ABC, chords usually appear as ChordSymbol objects if parsed correctly.
         chords = score.chordify().flatten().getElementsByClass(music21.harmony.ChordSymbol)

    # Sort chords by offset
    chords = sorted(chords, key=lambda x: x.offset)

    dataset = []

    for n in melody.notesAndRests:
        # Find the chord active at this note's offset
        active_chord = "N.C." # No Chord

        # Since ABC chords often have 0 duration, we assume the chord lasts until the next one.
        # Find the latest chord that started at or before this note.
        current_best_chord = None
        for c in chords:
            if c.offset <= n.offset:
                current_best_chord = c
            else:
                break

        if current_best_chord:
            active_chord = current_best_chord.figure

        # Build the feature row
        if n.isNote:
            feature = {
                'pitch': n.pitch.ps,      # Pitch as a float (MIDI number)
                'duration': n.duration.quarterLength,
                'beat': n.beat,
                'measure': n.measureNumber,
                'is_rest': 0,
                'target_chord': active_chord
            }
            dataset.append(feature)

    return dataset

def main():
    parser = argparse.ArgumentParser(description='Extract features from ABC file.')
    parser.add_argument('file', type=str, help='Path to the ABC file')
    args = parser.parse_args()

    try:
        data = extract_features_from_abc(args.file)
        print(f"Extracted {len(data)} notes.")
        for item in data[:5]:
            print(item)
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
