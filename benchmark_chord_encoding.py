import time
import timeit
from src.chord_encoding import encode_chord_to_target

def run_benchmark():
    # Warmup
    encode_chord_to_target("Cmaj7", 0, hierarchical=False)

    setup = """
from src.chord_encoding import encode_chord_to_target
    """

    stmt = """
encode_chord_to_target("G7", 0, hierarchical=False)
encode_chord_to_target("Am", 0, hierarchical=False)
encode_chord_to_target("F#dim7", 2, hierarchical=False)
encode_chord_to_target("Bbmaj7", 5, hierarchical=False)
    """

    times = timeit.repeat(stmt, setup, number=10000, repeat=5)
    print(f"Baseline (min of 5 runs, 10000 loops each): {min(times):.5f} seconds")

if __name__ == '__main__':
    run_benchmark()
