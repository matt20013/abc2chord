import time
import cProfile
from pstats import Stats
import music21

from src.parser import _extract_features_from_score

def create_synthetic_score(num_notes, num_chords):
    score = music21.stream.Score()
    part = music21.stream.Part()

    # Add a bunch of chords
    for i in range(num_chords):
        c = music21.harmony.ChordSymbol('C')
        c.offset = i * 4.0
        part.append(c)

    # Add a bunch of notes
    for i in range(num_notes):
        n = music21.note.Note('C4')
        n.offset = i * 0.5
        n.duration.quarterLength = 0.5
        part.append(n)

    score.append(part)
    return score

def main():
    print("Creating synthetic score...")
    score = create_synthetic_score(1000, 100)

    print("Running benchmark...")

    start_time = time.time()

    profiler = cProfile.Profile()
    profiler.enable()

    features = _extract_features_from_score(score)

    profiler.disable()
    end_time = time.time()

    print(f"Extraction took {end_time - start_time:.4f} seconds")
    print(f"Extracted {len(features)} features")

    stats = Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(10)

if __name__ == '__main__':
    main()
