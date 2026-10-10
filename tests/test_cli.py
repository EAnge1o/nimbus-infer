import os
import subprocess
import sys
import unittest

from nimbus_infer.cli import configure_console_output

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestConsoleOutput(unittest.TestCase):
    def test_configure_is_safe_and_idempotent(self):
        configure_console_output()
        configure_console_output()  # must not raise

    def test_printing_model_output_survives_non_utf8_console(self):
        """Regression: the demo used to crash on Windows' cp1252 stdout.

        Generated text is arbitrary decoded bytes, so on Windows the demo died
        with UnicodeEncodeError. ``configure_console_output`` must make printing
        it non-fatal.
        """
        env = dict(os.environ, PYTHONIOENCODING="cp1252")
        code = (
            "from nimbus_infer.cli import configure_console_output\n"
            "configure_console_output()\n"
            "print(chr(0x4F60) * 3 + chr(0xFFFD))\n"  # unencodable in cp1252
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()