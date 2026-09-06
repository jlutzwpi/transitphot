"""
Plate solving and WCS inspection.

ASIAIR (and most capture software) plate solves for *pointing* — to slew and
center the target — but the solution usually isn't written into the saved
light frames. Those frames carry the mount's reported coordinates
(OBJCTRA/OBJCTDEC), not a true WCS.

transitphot locates the target and comparison stars by coordinate, so it
needs a real WCS: CRVAL/CRPIX and a CD or PC matrix. This module wraps ASTAP
to write one into each frame, and provides a check so you find out before a
long run rather than during it.

Order of operations:
    capture -> transitphot calibrate -> transitphot solve -> transitphot run

Solving after calibration means the solved headers travel with the frames you
actually measure.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

WCS_KEYS = ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2")
ROT_KEYS = ("CD1_1", "CD1_2", "CD2_1", "CD2_2", "PC1_1", "CDELT1")


@dataclass
class WcsReport:
    path: Path
    has_wcs: bool
    ra_deg: float | None = None
    dec_deg: float | None = None
    note: str = ""


def inspect(path: Path) -> WcsReport:
    """Does this frame carry a usable WCS?"""
    from astropy.io import fits

    try:
        h = fits.getheader(path)
    except Exception as exc:                            # noqa: BLE001
        return WcsReport(path, False, note=f"unreadable: {exc}")

    has_core = all(k in h for k in WCS_KEYS)
    has_rot = any(k in h for k in ROT_KEYS)
    if has_core and has_rot:
        return WcsReport(path, True, float(h["CRVAL1"]), float(h["CRVAL2"]))

    hint = ""
    if h.get("OBJCTRA") or h.get("RA"):
        hint = ("has mount coordinates (OBJCTRA/RA) but no WCS — "
                "pointing info only, needs solving")
    elif not has_core:
        hint = "no WCS keywords"
    else:
        hint = "WCS incomplete (no CD/PC matrix)"
    return WcsReport(path, False, note=hint)


def inspect_dir(directory: Path) -> list[WcsReport]:
    return [inspect(p) for p in sorted(Path(directory).glob("*.fit*"))]


def find_astap() -> str | None:
    """Locate the ASTAP executable on PATH or in common install locations."""
    for name in ("astap", "astap_cli", "astap.exe", "astap_cli.exe"):
        found = shutil.which(name)
        if found:
            return found
    candidates = [
        Path(r"C:\Program Files\astap\astap.exe"),
        Path(r"C:\Program Files (x86)\astap\astap.exe"),
        Path("/usr/bin/astap"), Path("/usr/local/bin/astap"),
        Path("/opt/astap/astap"),
        Path("/Applications/ASTAP.app/Contents/MacOS/astap"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def solve_file(path: Path, astap: str, ra_deg: float | None = None,
               dec_deg: float | None = None, radius_deg: float = 10.0,
               fov_deg: float | None = None, timeout: int = 120) -> bool:
    """
    Solve one frame and write the WCS back into its header (-update).

    Passing an approximate RA/Dec (from your TransitPlanner plan) and a
    search radius makes solving dramatically faster: ASTAP starts near the
    right place instead of searching the whole sky.
    """
    cmd = [astap, "-f", str(path), "-update"]
    if ra_deg is not None and dec_deg is not None:
        # ASTAP takes RA in hours for -ra
        cmd += ["-ra", f"{ra_deg / 15.0:.6f}", "-spd", f"{dec_deg + 90.0:.6f}",
                "-r", str(radius_deg)]
    if fov_deg:
        cmd += ["-fov", f"{fov_deg:.4f}"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[solve] {path.name}: timed out after {timeout}s")
        return False
    if res.returncode != 0:
        msg = (res.stdout or res.stderr or "").strip().splitlines()
        print(f"[solve] {path.name}: failed ({msg[-1] if msg else 'no output'})")
        return False
    return inspect(path).has_wcs


def solve_dir(directory: Path, ra_deg: float | None = None,
              dec_deg: float | None = None, radius_deg: float = 10.0,
              fov_deg: float | None = None, skip_solved: bool = True
              ) -> tuple[int, int]:
    """
    Solve every frame in a directory. Returns (solved, failed).

    Frames that already carry a WCS are skipped by default, so re-running
    after a partial failure only works on what's left.
    """
    astap = find_astap()
    if not astap:
        raise RuntimeError(
            "ASTAP not found. Install it from https://www.hnsky.org/astap.htm "
            "(plus a star database such as D50), and make sure the executable "
            "is on your PATH."
        )
    paths = sorted(Path(directory).glob("*.fit*"))
    if not paths:
        raise FileNotFoundError(f"No FITS files in {directory}")

    solved = failed = 0
    for i, p in enumerate(paths, 1):
        if skip_solved and inspect(p).has_wcs:
            solved += 1
            continue
        ok = solve_file(p, astap, ra_deg, dec_deg, radius_deg, fov_deg)
        solved += ok
        failed += not ok
        if i % 25 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)} frames — {solved} solved, {failed} failed")
    return solved, failed
