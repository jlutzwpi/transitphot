# PyInstaller spec for the Windows build. Run packaging/build.ps1, not this
# directly. Produces dist/transitphot/ with two executables sharing one copy
# of the libraries:
#   transitphot.exe      the GUI (no console window)
#   transitphot-cli.exe  the command line; the GUI runs each step through it
from PyInstaller.utils.hooks import (collect_data_files, collect_submodules,
                                     copy_metadata)

# photutils and ccdproc compiled modules import each other in ways the
# import scanner cannot see, so take every submodule.
hidden = collect_submodules("transitphot")
for pkg in ("photutils", "ccdproc"):
    hidden += collect_submodules(pkg, filter=lambda n: ".tests" not in n)
datas = []
for pkg in ("astropy", "astropy_iers_data", "astroquery", "photutils", "ccdproc"):
    datas += collect_data_files(pkg)
# Several of these look themselves up with importlib.metadata at import time.
for pkg in ("astropy", "astroquery", "photutils", "ccdproc", "numpy", "scipy",
            "matplotlib", "requests", "pyvo", "keyring", "transitphot"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass
try:                                   # optional: limb-darkened fit via batman
    hidden += collect_submodules("batman")
except Exception:
    pass

# The serve front end, test suites and dev tools are not shipped.
excludes = ["fastapi", "uvicorn", "segno", "pytest", "IPython", "jupyter",
            "notebook", "sphinx", "PyQt5", "PyQt6", "PySide2", "PySide6"]

a = Analysis(
    ["entry_gui.py", "entry_cli.py"],
    pathex=[".."],
    hiddenimports=hidden,
    datas=datas,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

# Child processes get PYTHONUTF8/PYTHONUNBUFFERED from the GUI, but a frozen
# interpreter ignores PYTHON* variables, so bake the equivalents in:
# UTF-8 mode for file I/O, unbuffered so the log streams line by line.
opts = [("X utf8", None, "OPTION"), ("u", None, "OPTION")]

gui_scripts = [s for s in a.scripts if s[0] != "entry_cli"]
cli_scripts = [s for s in a.scripts if s[0] != "entry_gui"]

gui = EXE(pyz, gui_scripts, opts, exclude_binaries=True,
          name="transitphot", console=False, upx=False)
cli = EXE(pyz, cli_scripts, opts, exclude_binaries=True,
          name="transitphot-cli", console=True, upx=False)

COLLECT(gui, cli, a.binaries, a.datas, name="transitphot", upx=False)
