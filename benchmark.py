import time
import numpy as np
from src.chord_encoding import decode_target_to_chord

def run_benchmark():
    # Setup
    key_tonic_pc = 0
    hierarchical = False
    np.random.seed(42)
    # Generate 100,000 random target vectors
    target_vectors = np.random.rand(100000, 12).astype(np.float32)

    # Pre-warm cache
    decode_target_to_chord(target_vectors[0], key_tonic_pc, hierarchical)

    # Benchmark
    start_time = time.perf_counter()
    for vec in target_vectors:
        decode_target_to_chord(vec, key_tonic_pc, hierarchical)
    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"Benchmark duration: {duration:.4f} seconds")

if __name__ == '__main__':
    run_benchmark()
