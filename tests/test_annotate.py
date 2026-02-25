import unittest
import os
import sys
import shutil
import tempfile
import torch

# Ensure scripts and src are in path
sys.path.insert(0, os.getcwd())

# We import from scripts.annotate_abc
# But it's a script, not a module.
# We can use importlib or sys.path hacks.
# Or just exec the file? No.
# Since I added shebang and structure, it can be imported if I add scripts to path.
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))

try:
    import annotate_abc
except ImportError:
    # If import fails, try relative import if run from root
    from scripts import annotate_abc

class TestAnnotate(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.abc_file = os.path.join(self.test_dir, "test.abc")

        # Create a dummy ABC file
        with open(self.abc_file, "w") as f:
            f.write("""X:1
T:Test Tune
M:4/4
L:1/4
K:C
CDEF | GABc | cBAG | FEDC |
""")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_annotate_mock(self):
        # Run annotate_abc.main with --mock
        # We need to simulate sys.argv
        original_argv = sys.argv
        sys.argv = ["annotate_abc.py", self.abc_file, "--mock"]

        try:
            annotate_abc.main()
        except SystemExit as e:
            # Main might exit on success or failure?
            # Ideally main shouldn't exit if successful.
            # But argparse might exit if help requested or error.
            if e.code != 0:
                self.fail(f"annotate_abc exited with code {e.code}")
        finally:
            sys.argv = original_argv

        # Check output file
        out_file = os.path.join(self.test_dir, "test_predicted.abc")
        self.assertTrue(os.path.exists(out_file))

        with open(out_file, "r") as f:
            content = f.read()

        print("Annotated content:\n", content)

        # Verify chords are inserted
        # Mock pattern: Am, Dm, G7, C
        # Measure 0 (CDEF): "Am"
        # Measure 1 (GABc): "Dm"
        # Measure 2 (cBAG): "G7"
        # Measure 3 (FEDC): "C"

        # Check for presence of chords in correct order
        # We expect | "Am" CDEF | "Dm" GABc | "G7" cBAG | "C" FEDC |
        # But allow for spacing variations

        # First measure might not have preceding bar line if it follows K:
        self.assertRegex(content, r'(?:\||^|\n)\s*"Am"\s*CDEF')
        self.assertRegex(content, r'\|\s*"Dm"\s*GABc')
        self.assertRegex(content, r'\|\s*"G7"\s*cBAG')
        self.assertRegex(content, r'\|\s*"C"\s*FEDC')

    def test_annotate_with_existing_chords(self):
        # Create ABC with existing chords
        with open(self.abc_file, "w") as f:
            f.write("""X:1
T:Test Tune Existing
M:4/4
K:C
"G" CDEF | "D7" GABc |
""")

        original_argv = sys.argv
        sys.argv = ["annotate_abc.py", self.abc_file, "--mock"]

        try:
            annotate_abc.main()
        except SystemExit as e:
            if e.code != 0:
                self.fail(f"annotate_abc exited with code {e.code}")
        finally:
            sys.argv = original_argv

        out_file = os.path.join(self.test_dir, "test_predicted.abc")
        with open(out_file, "r") as f:
            content = f.read()

        print("Annotated content (replace):\n", content)

        # Expect "Am" instead of "G", "Dm" instead of "D7"
        self.assertRegex(content, r'(?:\||^|\n)\s*"Am"\s*CDEF')
        self.assertRegex(content, r'\|\s*"Dm"\s*GABc')
        self.assertNotRegex(content, r'"G"')
        self.assertNotRegex(content, r'"D7"')

    def test_annotate_skip_empty_segments(self):
        # Create ABC with Z (rest measure)
        # Assuming music21 skips Z, annotate_abc should skip Z too
        with open(self.abc_file, "w") as f:
            f.write("""X:1
T:Test Tune Skip
M:4/4
K:C
CDEF | Z | GABc |
""")

        original_argv = sys.argv
        sys.argv = ["annotate_abc.py", self.abc_file, "--mock"]

        try:
            annotate_abc.main()
        except SystemExit as e:
            if e.code != 0:
                self.fail(f"annotate_abc exited with code {e.code}")
        finally:
            sys.argv = original_argv

        out_file = os.path.join(self.test_dir, "test_predicted.abc")
        with open(out_file, "r") as f:
            content = f.read()

        print("Annotated content (skip):\n", content)

        # Expect: | "Am" CDEF | Z | "Dm" GABc |
        # Note: If music21 skips Z, the measures become 0:CDEF, 1:GABc.
        # Mock pattern: 0->Am, 1->Dm.
        # Text alignment:
        # CDEF (match m0) -> Insert Am
        # Z (mismatch m1) -> Skip
        # GABc (match m1) -> Insert Dm

        self.assertRegex(content, r'(?:\||^|\n)\s*"Am"\s*CDEF')
        self.assertRegex(content, r'\|\s*Z\s*\|') # Z should be untouched (no chord)
        self.assertRegex(content, r'\|\s*"Dm"\s*GABc')

if __name__ == '__main__':
    unittest.main()
