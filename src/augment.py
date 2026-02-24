import music21
import os
import argparse

def transpose_tune(abc_file_path, output_dir):
    """
    Transposes an ABC tune into all 12 keys and saves them as MusicXML.
    """
    filename = os.path.basename(abc_file_path)
    base_name, _ = os.path.splitext(filename)
    # Output format will be MusicXML because music21 cannot write robust ABC files.
    ext = ".musicxml"

    try:
        score = music21.converter.parse(abc_file_path)
    except Exception as e:
        print(f"Error parsing {abc_file_path}: {e}")
        return

    # Transpose from -6 to +5 semitones
    # This covers a full octave range roughly centered around the original key
    for i in range(-6, 6):
        # We generate 12 versions from -6 to +5 semitones.

        try:
            # Transpose by i semitones
            transposed_score = score.transpose(i)

            # Construct new filename
            new_filename = f"{base_name}_transpose_{i}{ext}"
            output_path = os.path.join(output_dir, new_filename)

            # Write to file
            transposed_score.write('musicxml', fp=output_path)
            print(f"Saved {output_path}")
        except Exception as e:
             print(f"Error transposing/saving interval {i}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Transpose ABC file into all 12 keys.')
    parser.add_argument('file', type=str, help='Path to the ABC file')
    parser.add_argument('output_dir', type=str, help='Directory to save transposed files')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    transpose_tune(args.file, args.output_dir)

if __name__ == "__main__":
    main()
