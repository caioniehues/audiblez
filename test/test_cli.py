"""CLI tests.

`audiblez --help` and the no-arg path do not import the heavy TTS stack (core is
imported lazily inside cli_main after argument parsing), so these run anywhere the
package is importable — no models, ffmpeg, or network required. End-to-end epub
-> m4b conversion is covered by the CI smoke job and test_main.py.
"""
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args):
    return subprocess.run(
        [sys.executable, '-m', 'audiblez.cli', *args],
        cwd=REPO_ROOT, capture_output=True, text=True)


class CliHelpTest(unittest.TestCase):
    def test_help_lists_voices_and_usage(self):
        proc = run_cli('--help')
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0)
        self.assertIn('usage:', out)
        self.assertIn('af_sky', out)

    def test_no_args_exits_nonzero_with_usage(self):
        proc = run_cli()
        out = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('usage:', out)


if __name__ == '__main__':
    unittest.main()
