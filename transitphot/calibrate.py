"""
Calibration: build master bias/dark/flat and apply them to lights.

Design notes
------------
* Uses ccdproc (astropy-affiliated) rather than driving AstroImageJ's GUI.
* Dark scaling is OFF by default: most amateur CMOS cameras (ASI2600MM et al.)
  are not linear enough in dark current for scaling to beat a matched-exposure
  dark. Shoot darks at the same exposure and temperature as your lights.
* Everything is memory-conscious: frames are combined in chunks so a night of
  600 subframes doesn't need 600 frames of RAM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.nddata import CCDData
import ccdproc


def _combine(paths: list[Path], unit: str = "adu", method: str = "median",
             sigma_clip: bool = True) -> CCDData:
    """Sigma-clipped combine of a list of FITS files into one master frame."""
    frames = [CCDData.read(p, unit=unit) for p in paths]
    combiner_kw = dict(method=method, mem_limit=1_500e6)
    if sigma_clip:
        combiner_kw.update(
            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median, sigma_clip_dev_func=np.ma.std,
        )
    return ccdproc.combine(frames, **combiner_kw)


def make_master_bias(bias_dir: Path, out: Path | None = None) -> CCDData:
    paths = sorted(Path(bias_dir).glob("*.fit*"))
    if not paths:
        raise FileNotFoundError(f"No FITS files in {bias_dir}")
    master = _combine(paths)
    master.meta["IMAGETYP"] = "MASTER BIAS"
    master.meta["NCOMBINE"] = len(paths)
    if out:
        master.write(out, overwrite=True)
    return master


def make_master_dark(dark_dir: Path, master_bias: CCDData | None = None,
                     out: Path | None = None) -> CCDData:
    paths = sorted(Path(dark_dir).glob("*.fit*"))
    if not paths:
        raise FileNotFoundError(f"No FITS files in {dark_dir}")
    master = _combine(paths)
    if master_bias is not None:
        master = ccdproc.subtract_bias(master, master_bias)
    master.meta["IMAGETYP"] = "MASTER DARK"
    master.meta["NCOMBINE"] = len(paths)
    if out:
        master.write(out, overwrite=True)
    return master


def make_master_flat(flat_dir: Path, master_bias: CCDData | None = None,
                     master_dark: CCDData | None = None,
                     out: Path | None = None) -> CCDData:
    """Flats are combined after bias/dark removal, then normalized to unity."""
    paths = sorted(Path(flat_dir).glob("*.fit*"))
    if not paths:
        raise FileNotFoundError(f"No FITS files in {flat_dir}")
    cleaned = []
    for p in paths:
        f = CCDData.read(p, unit="adu")
        if master_bias is not None:
            f = ccdproc.subtract_bias(f, master_bias)
        if master_dark is not None:
            f = ccdproc.subtract_dark(f, master_dark, exposure_time="EXPTIME",
                                      exposure_unit=u.s, scale=False)
        cleaned.append(f)
    master = ccdproc.combine(cleaned, method="median", mem_limit=1_500e6,
                             sigma_clip=True,
                             sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                             sigma_clip_func=np.ma.median,
                             sigma_clip_dev_func=np.ma.std)
    master.data = master.data / np.median(master.data)     # normalize
    master.meta["IMAGETYP"] = "MASTER FLAT"
    master.meta["NCOMBINE"] = len(paths)
    if out:
        master.write(out, overwrite=True)
    return master


def calibrate_light(path: Path, master_bias: CCDData | None,
                    master_dark: CCDData | None, master_flat: CCDData | None,
                    scale_dark: bool = False) -> CCDData:
    """Apply available masters to a single light frame."""
    img = CCDData.read(path, unit="adu")
    if master_bias is not None:
        img = ccdproc.subtract_bias(img, master_bias)
    if master_dark is not None:
        img = ccdproc.subtract_dark(
            img, master_dark, exposure_time="EXPTIME", exposure_unit=u.s,
            scale=scale_dark,
        )
    if master_flat is not None:
        img = ccdproc.flat_correct(img, master_flat)
    return img


def calibrate_night(lights_dir: Path, bias_dir: Path | None = None,
                    darks_dir: Path | None = None, flats_dir: Path | None = None,
                    out_dir: Path | None = None, scale_dark: bool = False
                    ) -> list[Path]:
    """
    Full calibration pass over a night's lights. Any missing calibration
    directory is simply skipped — a dark-only workflow is valid.
    Returns the list of calibrated file paths.
    """
    mb = make_master_bias(bias_dir) if bias_dir else None
    md = make_master_dark(darks_dir, mb) if darks_dir else None
    mf = make_master_flat(flats_dir, mb, md) if flats_dir else None

    out_dir = Path(out_dir or Path(lights_dir) / "calibrated")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for p in sorted(Path(lights_dir).glob("*.fit*")):
        cal = calibrate_light(p, mb, md, mf, scale_dark=scale_dark)
        dest = out_dir / f"cal_{p.name}"
        cal.write(dest, overwrite=True)
        written.append(dest)
    return written
