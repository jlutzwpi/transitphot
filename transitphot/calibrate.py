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

from .photometry import fits_files

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
    paths = fits_files(bias_dir)
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
    paths = fits_files(dark_dir)
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


def _binning_of(ccd) -> str:
    """Binning as N.I.N.A. records it, e.g. '2x2', or '?' if absent."""
    h = ccd.meta
    for kx, ky in (("XBINNING", "YBINNING"), ("BINX", "BINY"),
                   ("CCDXBIN", "CCDYBIN")):
        if h.get(kx) and h.get(ky):
            return f"{int(h[kx])}x{int(h[ky])}"
    return "?"


def _check_shapes(name: str, frame, reference, ref_name: str,
                  rebin: bool = False):
    """
    Confirm a calibration frame matches the frames it will be applied to.

    A shape mismatch is nearly always a binning mismatch — flats taken at
    1x1 against lights at 2x2, say — and without this check it surfaces deep
    inside ccdproc as an unhelpful broadcast error after several minutes of
    combining.

    With rebin=True an integer-factor mismatch is fixed by summing blocks of
    the finer frame, which is what hardware binning does. That is a fair
    approximation for a flat (the response is linear), and a poor idea for
    bias or dark frames, whose noise does not combine the same way.
    """
    if frame.shape == reference.shape:
        return frame
    fy, fx = frame.shape
    ry, rx = reference.shape
    detail = (f"{name} is {fx}x{fy} ({_binning_of(frame)}) but {ref_name} is "
              f"{rx}x{ry} ({_binning_of(reference)})")
    if not rebin or fy % ry or fx % rx:
        raise SystemExit(
            f"Calibration frames don't match: {detail}.\n"
            f"Flats, darks and bias must be taken at the same binning as the "
            f"lights.\nEither re-shoot them to match, drop --flats and "
            f"calibrate with bias and darks only,\nor pass --rebin-flats to "
            f"bin finer flats down in software.")
    ky, kx = fy // ry, fx // rx
    import numpy as _np
    print(f"  rebinning {name} by {kx}x{ky} to match {ref_name} ({detail})")
    data = frame.data.reshape(ry, ky, rx, kx).sum(axis=(1, 3))
    out = frame.copy()
    out.data = data
    if out.uncertainty is not None:
        out.uncertainty = None          # no longer valid after summing
    return out



def make_master_flat(flat_dir: Path, master_bias: CCDData | None = None,
                     master_dark: CCDData | None = None,
                     out: Path | None = None,
                     rebin: bool = False) -> CCDData:
    """Flats are combined after bias/dark removal, then normalized to unity."""
    paths = fits_files(flat_dir)
    if not paths:
        raise FileNotFoundError(f"No FITS files in {flat_dir}")
    cleaned = []
    scaled_note = [False]
    for p in paths:
        f = CCDData.read(p, unit="adu")
        if master_bias is not None:
            f = _check_shapes("the flats", f, master_bias, "the bias", rebin)
            f = ccdproc.subtract_bias(f, master_bias)
        if master_dark is not None:
            f = _check_shapes("the flats", f, master_dark, "the darks", rebin)
            # Sky flats are short; the darks are usually shot to match the
            # lights. Subtracting a 60 s dark from a 2 s flat removes far
            # more than is there and drives pixels to zero or below, which
            # then divides into the lights as infinity. Scale by exposure
            # when they differ, and say so.
            t_flat = float(f.meta.get("EXPTIME") or f.meta.get("EXPOSURE") or 0)
            t_dark = float(master_dark.meta.get("EXPTIME")
                           or master_dark.meta.get("EXPOSURE") or 0)
            if t_flat and t_dark and abs(t_flat - t_dark) > 0.1 * t_dark:
                if not scaled_note[0]:
                    print(f"  flats are {t_flat:g}s and the darks {t_dark:g}s "
                          f"— scaling the dark by exposure for the flats")
                    scaled_note[0] = True
                f = ccdproc.subtract_dark(f, master_dark,
                                          exposure_time="EXPTIME",
                                          exposure_unit=u.s, scale=True)
            else:
                f = ccdproc.subtract_dark(f, master_dark,
                                          exposure_time="EXPTIME",
                                          exposure_unit=u.s, scale=False)
        cleaned.append(f)
    master = ccdproc.combine(cleaned, method="median", mem_limit=1_500e6,
                             sigma_clip=True,
                             sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                             sigma_clip_func=np.ma.median,
                             sigma_clip_dev_func=np.ma.std)
    norm = float(np.median(master.data))
    if not np.isfinite(norm) or norm <= 0:
        raise SystemExit(
            "The master flat has a median of zero or less, so it cannot be "
            "used.\nThis usually means the darks subtracted from the flats "
            "were far longer exposures.\nRe-shoot flat darks to match the "
            "flats, or calibrate the flats with bias only.")
    master.data = master.data / norm                       # normalize

    # A pixel at or below zero divides into infinity. Leave those pixels
    # uncorrected (a factor of 1) rather than poisoning every light frame,
    # and report how many, since a large count means the flats are wrong.
    bad = ~np.isfinite(master.data) | (master.data <= 0.05)
    n_bad = int(bad.sum())
    if n_bad:
        frac = n_bad / master.data.size
        master.data[bad] = 1.0
        msg = (f"  {n_bad} flat pixel(s) ({frac:.2%}) were zero, negative or "
               f"not finite; left uncorrected")
        if frac > 0.01:
            msg += ("\n  That is a lot — check the flats are properly "
                    "exposed and that their darks match.")
        print(msg)
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


def _report_flat(master) -> None:
    """
    Describe the master flat's shape, and flag it when it looks wrong.

    A flat should be brightest near the optical axis and fall off toward the
    corners. If the center is DIMMER than the corners, dividing by it makes
    the science frames darker in the middle — the "donut" look — and the
    flat is describing a different optical configuration than the lights
    were taken through.
    """
    d = np.asarray(master.data, dtype=float)
    ny, nx = d.shape
    cy, cx = ny // 2, nx // 2
    h = max(min(ny, nx) // 10, 8)
    center = float(np.nanmedian(d[cy - h:cy + h, cx - h:cx + h]))
    corners = float(np.nanmedian(np.concatenate([
        d[:2 * h, :2 * h].ravel(), d[:2 * h, -2 * h:].ravel(),
        d[-2 * h:, :2 * h].ravel(), d[-2 * h:, -2 * h:].ravel()])))
    if not (np.isfinite(center) and np.isfinite(corners) and corners > 0):
        return
    ratio = center / corners
    print(f"  master flat: center/corner {ratio:.3f} "
          f"(vignetting {100 * (1 - corners / center):+.1f}% at the corners)")
    if ratio < 0.98:
        print("  WARNING: the flat is DIMMER at the center than the corners. "
              "Dividing by it\n  will darken the middle of every light frame. "
              "Check the flats were taken\n  through the same optics, "
              "reducer and filter as the lights.")



def calibrate_night(lights_dir: Path, bias_dir: Path | None = None,
                    darks_dir: Path | None = None, flats_dir: Path | None = None,
                    out_dir: Path | None = None, scale_dark: bool = False,
                    rebin_flats: bool = False) -> list[Path]:
    """
    Full calibration pass over a night's lights. Any missing calibration
    directory is simply skipped — a dark-only workflow is valid.
    Returns the list of calibrated file paths.
    """
    out_dir = Path(out_dir or Path(lights_dir) / "calibrated")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write the masters, but in their own subfolder. When calibration goes
    # wrong the master is the thing you need to look at — and putting them
    # beside the calibrated lights means the next run globs them in as
    # science frames.
    masters = out_dir / "masters"
    masters.mkdir(parents=True, exist_ok=True)
    mb = (make_master_bias(bias_dir, out=masters / "master_bias.fits")
          if bias_dir else None)
    md = (make_master_dark(darks_dir, mb, out=masters / "master_dark.fits")
          if darks_dir else None)
    mf = (make_master_flat(flats_dir, mb, md, rebin=rebin_flats,
                           out=masters / "master_flat.fits")
          if flats_dir else None)
    if mb is not None or md is not None or mf is not None:
        print(f"  masters written to {masters}")
    if mf is not None:
        _report_flat(mf)

    written = []
    for p in fits_files(lights_dir):
        cal = calibrate_light(p, mb, md, mf, scale_dark=scale_dark)
        dest = out_dir / f"cal_{p.name}"
        cal.write(dest, overwrite=True)
        written.append(dest)
    return written
