"""transitphot — local FITS-to-light-curve pipeline for transit photometry."""

import sys
from pathlib import Path


def cli_command() -> list[str]:
    """The command that runs transitphot's CLI in a child process.

    From source that is ``python -m transitphot.cli``. In the Windows build
    (PyInstaller) there is no separate interpreter: ``sys.executable`` is the
    GUI exe itself, so the GUI calls the console exe built alongside it.
    """
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).with_name("transitphot-cli.exe")
        return [str(exe)]
    return [sys.executable, "-m", "transitphot.cli"]
