"""
Export a finished measurement for submission.

Formats:
  * ExoClock / AAVSO-style light-curve text file (time, flux, error)
  * a summary JSON with the fitted mid-time and O-C, suitable for posting
    back to TransitPlanner

IMPORTANT: submission formats change. Before your first submission, check the
current requirements on the ExoClock and AAVSO pages — this writes a sensible,
well-labelled file, but the receiving organization's spec is authoritative.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


AAVSO_FILTERS = {
    "CV", "CBB", "U", "B", "V", "R", "I",
    "SZ", "SU", "SG", "SR", "SI", "TG", "TB", "TR", "O",
}

# How an imaging filter maps to an AAVSO ShortName. The distinction that
# matters: R/B/V are the PHOTOMETRIC standards (Cousins, Johnson) with
# defined bandpasses. A broadband LRGB imaging filter is not one of those,
# so it reports as the tri-colour equivalent — claiming Cousins R implies a
# photometric calibration an LRGB set does not have. Luminance is a UV/IR
# blocking filter, not "no filter", so it is O with a description.
FILTER_MAP = {
    "L": ("O", "Luminance (UV/IR blocking, ~400-700 nm)"),
    "LUM": ("O", "Luminance (UV/IR blocking, ~400-700 nm)"),
    "LUMINANCE": ("O", "Luminance (UV/IR blocking, ~400-700 nm)"),
    "CLEAR": ("CV", ""),
    "NONE": ("CV", ""),
    "R": ("TR", "Broadband LRGB red, not Cousins R"),
    "RED": ("TR", "Broadband LRGB red, not Cousins R"),
    "G": ("TG", "Broadband LRGB green"),
    "GREEN": ("TG", "Broadband LRGB green"),
    "B": ("TB", "Broadband LRGB blue, not Johnson B"),
    "BLUE": ("TB", "Broadband LRGB blue, not Johnson B"),
    "RC": ("R", ""),          # an actual Cousins R
    "V": ("V", ""),
    "I": ("I", ""),
    "IC": ("I", ""),
    "CBB": ("CBB", ""),
    "HA": ("O", "H-alpha narrowband"),
    "S": ("O", "SII narrowband"),
    "O3": ("O", "OIII narrowband"),
}


def aavso_filter(name: str | None) -> tuple[str, str]:
    """
    Map a filter name to an (AAVSO ShortName, description) pair.

    An unrecognised filter returns ("O", the original name), which is the
    honest answer: O means "something else, described in Notes".
    """
    if not name:
        return "CV", ""
    key = name.strip().upper()
    if key in FILTER_MAP:
        return FILTER_MAP[key]
    if key in AAVSO_FILTERS:
        return key, ""
    return "O", f"Filter reported as '{name}'"



def write_lightcurve(path: Path, bjd, flux, flux_err, meta: dict) -> Path:
    """Three-column light curve with a commented metadata header."""
    path = Path(path)
    with path.open("w") as f:
        f.write("# Transit light curve\n")
        f.write(f"# generated: {datetime.now(timezone.utc).isoformat()}\n")
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        f.write("# BJD_TDB  rel_flux  rel_flux_err\n")
        for t, fl, e in zip(bjd, flux, flux_err):
            f.write(f"{t:.6f} {fl:.6f} {e:.6f}\n")
    return path


def write_summary(path: Path, fit, meta: dict, predicted_mid_bjd=None) -> Path:
    from .fitting import o_minus_c_minutes

    out = {
        "observed_mid_bjd_tdb": round(fit.mid_bjd, 6),
        "mid_uncertainty_minutes": round(fit.mid_err_minutes, 2),
        "depth_ppm": round(fit.depth_ppm, 1),
        "depth_err_ppm": round(fit.depth_err * 1e6, 1),
        "duration_hours": round(fit.duration_days * 24, 3),
        "residual_rms_ppm": round(fit.rms_ppm, 1),
        "n_points": fit.n_points,
        **meta,
    }
    if predicted_mid_bjd:
        out["predicted_mid_bjd_tdb"] = round(predicted_mid_bjd, 6)
        out["o_minus_c_minutes"] = round(
            o_minus_c_minutes(fit.mid_bjd, predicted_mid_bjd), 2)
    Path(path).write_text(json.dumps(out, indent=2))
    return Path(path)


def write_aavso(path, bjd_tdb, flux, flux_err, *, obscode: str,
                star_name: str, exoplanet_name: str,
                exposure_s: float, filter_code: str,
                binning: str = "1x1",
                software: str = "transitphot",
                ra: str | None = None, dec: str | None = None,
                priors: str = "", results: str = "", notes: str = "",
                airmass=None, measurement_type: str = "Rnflux"):
    """
    Write an AAVSO Exoplanet Database report file.

    Format reference: AAVSO Exoplanet Report File Format v0.62. One
    observation per file: a parameter block of #KEY= lines, then one
    measurement per line.

    Notes on choices made here:
    * DATE_TYPE is BJD_TDB, which is what the pipeline produces and the most
      precise option the format accepts.
    * MEASUREMENT_TYPE defaults to Rnflux — the curve is normalised relative
      flux, not a differential magnitude.
    * A CMOS camera is reported as OBSTYPE=CCD per the spec, with the actual
      sensor named in NOTES.
    * AIRMASS is written as a detrend column when supplied, so the reviewer
      can see and re-fit the trend rather than taking our detrending on
      faith.
    """
    path = Path(path)
    detrend = "Airmass" if airmass is not None else ""

    head = [
        "#TYPE=EXOPLANET",
        f"#OBSCODE={obscode}",
        f"#SOFTWARE={software}",
        "#DELIM=,",
        "#DATE_TYPE=BJD_TDB",
        "#OBSTYPE=CCD",
        f"#STAR_NAME={star_name}",
    ]
    if ra and dec:
        head += [f"#RA={ra}", f"#DEC={dec}", "#EPOCH=J2000"]
    head += [
        f"#EXOPLANET_NAME={exoplanet_name}",
        f"#BINNING={binning}",
        f"#EXPOSURE_TIME={exposure_s:g}",
        f"#FILTER={filter_code}",
        f"#DETREND_PARAMETERS={detrend}",
        f"#MEASUREMENT_TYPE={measurement_type}",
    ]
    if priors:
        head.append(f"#PRIORS={priors[:250]}")
    if results:
        head.append(f"#RESULTS={results[:250]}")
    if notes:
        head.append(f"#NOTES={notes}")

    lines = list(head)
    # Commented column header, as AAVSO's own sample files carry it.
    lines.append("# DATE DIFF MERR DETREND_1 DETREND_2 DETREND_3 DETREND_4")
    for i, (t, f, e) in enumerate(zip(bjd_tdb, flux, flux_err)):
        err = f"{e:.6f}" if np.isfinite(e) else "na"
        # All four detrend columns are always written, unused ones as n/a
        # placeholders — the spec calls for placeholders and the reference
        # files emit the full set.
        cols = [f"{float(airmass[i]):.6f}"] if airmass is not None else []
        cols += ["n/a"] * (4 - len(cols))
        lines.append(f"{t:.8f},{f:.6f},{err}," + ",".join(cols))

    path.write_text("\n".join(lines) + "\n")
    return path
