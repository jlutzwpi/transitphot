"""
Alignment, aperture photometry, and differential light-curve extraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.detection import DAOStarFinder


def estimate_fwhm(data: np.ndarray, fwhm_guess: float = 4.0) -> float:
    """
    Rough FWHM from detected sources; used to size apertures.

    Falls back to the guess when detection fails — which happens on frames
    taken through thick cloud, where there simply aren't enough sources above
    threshold. Using a sane default keeps those frames in the series (they
    will show up as low flux, which is correct) instead of aborting the run.
    """
    import warnings
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # NoDetectionsWarning is expected
        try:
            finder = DAOStarFinder(fwhm=fwhm_guess, threshold=5.0 * std)
            srcs = finder(data - median)
        except Exception:                        # noqa: BLE001
            srcs = None
    if srcs is None or len(srcs) == 0:
        return fwhm_guess
    # DAOStarFinder's sharpness relates to profile width; use a robust proxy
    return float(np.clip(fwhm_guess * (1.0 / np.median(srcs["sharpness"]) / 3.0),
                         2.0, 12.0))


def iter_frames(paths: list[Path]):
    """
    Yield (path, data, header) for each frame, unmodified.

    Deliberately NO image registration: every frame is plate-solved, so
    aperture positions are computed from that frame's own WCS. This is
    better than aligning — resampling pixels to a common grid interpolates
    flux between neighbours, which is exactly the quantity we are trying to
    measure. Photometry belongs on original pixels.
    """
    for p in paths:
        try:
            yield p, fits.getdata(p).astype(float), fits.getheader(p)
        except Exception as exc:                      # noqa: BLE001
            print(f"[read] skipped {p.name}: {exc}")


def align_to_reference(paths: list[Path], reference: Path | None = None):
    """
    Optional fallback for UNSOLVED frames: triangle-matching registration
    via astroalign. Requires the optional `astroalign` extra
    (`pip install "transitphot[align]"`), which needs a C compiler on
    Windows. Prefer plate solving and iter_frames().
    """
    try:
        import astroalign as aa
    except ImportError as exc:                        # noqa: BLE001
        raise SystemExit(
            "astroalign is not installed. It is only needed for unsolved "
            "frames — plate solve instead:\n"
            "  transitphot solve --lights <dir> --ra <deg> --dec <deg>"
        ) from exc

    ref_path = reference or paths[0]
    ref = fits.getdata(ref_path).astype(float)
    for p in paths:
        data = fits.getdata(p).astype(float)
        hdr = fits.getheader(p)
        if p == ref_path:
            yield p, data, hdr
            continue
        try:
            registered, _ = aa.register(data, ref)
            yield p, registered, hdr
        except Exception as exc:                      # noqa: BLE001
            print(f"[align] skipped {p.name}: {exc}")


def measure(data: np.ndarray, positions: list[tuple[float, float]],
            r_ap: float, r_in: float, r_out: float) -> np.ndarray:
    """
    Aperture photometry with local background annulus.
    Returns background-subtracted flux per position.
    """
    aps = CircularAperture(positions, r=r_ap)
    anns = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    phot = aperture_photometry(data, [aps, anns])

    ann_masks = anns.to_mask(method="center")
    bkg_median = []
    for m in ann_masks:
        vals = m.multiply(data)
        vals = vals[m.data > 0]
        _, med, _ = sigma_clipped_stats(vals[np.isfinite(vals)], sigma=3.0)
        bkg_median.append(med)
    bkg_total = np.array(bkg_median) * aps.area
    return np.array(phot["aperture_sum_0"]) - bkg_total


def sky_to_pixel(header: dict, ra_deg: float, dec_deg: float):
    """Convert catalog coordinates to pixels using the frame's WCS."""
    w = WCS(header)
    x, y = w.all_world2pix(ra_deg, dec_deg, 0)
    return float(x), float(y)


def frame_time(header: dict) -> float:
    """Mid-exposure time as BJD-ish JD (UTC). Real BJD_TDB conversion is
    applied later, once the site and target are known."""
    t = header.get("DATE-OBS")
    exp = float(header.get("EXPTIME", 0))
    return Time(t, format="isot", scale="utc").jd + (exp / 2) / 86400.0


def differential_curve(target_flux: np.ndarray, comp_flux: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensemble differential photometry: divide the target by the summed
    comparison flux, then normalize to the out-of-transit median.

    comp_flux: (n_comps, n_frames)
    Returns (normalized flux, per-point uncertainty estimate).
    """
    ensemble = np.nansum(comp_flux, axis=0)
    rel = target_flux / ensemble
    norm = rel / np.nanmedian(rel)

    # Poisson-limited uncertainty propagated through the ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        err = norm * np.sqrt(
            np.where(target_flux > 0, 1.0 / target_flux, np.nan)
            + np.where(ensemble > 0, 1.0 / ensemble, np.nan)
        )
    return norm, err


def filter_session_frames(paths, headers=None, max_gap_hours: float = 6.0):
    """
    Keep only frames belonging to the main observing session.

    Folders accumulate strays — a test frame from another night, a file
    copied in by mistake. Those carry timestamps days away from the session
    and would stretch the light curve's time axis into uselessness. This
    finds the largest cluster of frames in time and returns just those.
    """
    times = []
    for p in paths:
        try:
            times.append(frame_time(fits.getheader(p)))
        except Exception:                        # noqa: BLE001
            times.append(np.nan)
    times = np.array(times, dtype=float)
    ok = np.isfinite(times)
    if ok.sum() < 2:
        return list(paths), []

    order = np.argsort(np.where(ok, times, np.inf))
    sorted_t = times[order]
    # split wherever consecutive frames are further apart than max_gap
    gaps = np.diff(sorted_t) > (max_gap_hours / 24.0)
    group_id = np.concatenate([[0], np.cumsum(gaps)])
    counts = np.bincount(group_id[: ok.sum()])
    main = int(np.argmax(counts))

    keep_idx = set(order[: ok.sum()][group_id[: ok.sum()] == main])
    kept = [paths[i] for i in range(len(paths)) if i in keep_idx]
    dropped = [paths[i] for i in range(len(paths)) if i not in keep_idx]
    return kept, dropped
