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
