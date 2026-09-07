"""
Alignment, aperture photometry, and differential light-curve extraction.
"""

from __future__ import annotations

from pathlib import Path

import math

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.detection import DAOStarFinder


def measure_fwhm(data: np.ndarray, xy: tuple[float, float],
                 box: float = 15.0) -> float:
    """
    FWHM from the area of the star's half-maximum region.

    Method: subtract the local background, find the peak, count pixels above
    half the peak, and convert that area to an equivalent diameter
    (FWHM = 2*sqrt(area/pi)).

    Why not second moments: a moment calculation over a cutout is dominated
    by whatever fills the cutout. With background noise present, the moment
    of a 40x40 box returns sigma ~ 11 px regardless of the star, which is how
    an earlier version reported FWHM 17 px for stars actually 4 px wide — and
    sized apertures at 34 px, swallowing neighbors and sky. Half-max area
    only counts pixels the star genuinely lights up, so noise contributes
    almost nothing.
    """
    x, y = xy
    h, w = data.shape
    x0, x1 = int(max(x - box, 0)), int(min(x + box, w))
    y0, y1 = int(max(y - box, 0)), int(min(y + box, h))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return float("nan")

    cut = data[y0:y1, x0:x1].astype(float)
    _, med, std = sigma_clipped_stats(cut, sigma=3.0)
    sub = cut - med
    peak = float(np.nanmax(sub))
    if not np.isfinite(peak) or std <= 0 or peak < 5 * std:
        return float("nan")                      # nothing bright enough here

    # Count only pixels near the centre, so a neighbouring star in the corner
    # of the cutout can't inflate the area.
    py, px = np.unravel_index(int(np.nanargmax(sub)), sub.shape)
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    near = ((xx - px) ** 2 + (yy - py) ** 2) < (box * 0.8) ** 2
    area = int(np.sum((sub > 0.5 * peak) & near))
    if area < 2:
        return float("nan")
    fwhm = 2.0 * math.sqrt(area / math.pi)
    return float(fwhm) if 1.0 < fwhm < 20.0 else float("nan")


def session_fwhm(paths, positions_fn, sample: int = 15,
                 default: float = 4.0) -> float:
    """
    One FWHM for the whole session, from a sample of frames.

    The aperture radius MUST be constant across the series. Aperture
    photometry measures a fixed fraction of a star's light; if the aperture
    changes size between frames, that fraction changes, and the resulting
    flux variation is indistinguishable from a real signal. Seeing does vary
    through a night, but a fixed aperture sized for the worst of it is far
    better than one that tracks it.
    """
    vals = []
    step = max(len(paths) // sample, 1)
    for p in paths[::step][:sample]:
        try:
            data = fits.getdata(p).astype(float)
            hdr = fits.getheader(p)
            xy = positions_fn(hdr)
            if xy is None:
                continue
            f = measure_fwhm(data, xy)
            if np.isfinite(f):
                vals.append(f)
        except Exception:                            # noqa: BLE001
            continue
    if not vals:
        return default
    # 80th percentile: size for the poorer-seeing frames so no frame has its
    # star spilling outside the aperture.
    return float(np.percentile(vals, 80))


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


def refine_position(data: np.ndarray, xy: tuple[float, float], box: float = 12.0,
                    min_snr: float = 3.0):
    """
    Refine a catalog-projected position onto the actual star, and verify a
    star is really there.

    Returns (x, y, ok). ok is False when no source is detectable at the
    expected place — which happens when a frame carries a stale WCS (common
    around a meridian flip), when cloud swallowed the field, or when the
    target drifted off the sensor. Those frames must be dropped: measuring
    empty sky produces a flux ratio that explodes and destroys the fit.
    """
    from photutils.centroids import centroid_com

    x, y = xy
    h, w = data.shape
    x0, x1 = int(max(x - box, 0)), int(min(x + box, w))
    y0, y1 = int(max(y - box, 0)), int(min(y + box, h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return x, y, False                      # off the sensor

    cut = data[y0:y1, x0:x1].astype(float)
    _, med, std = sigma_clipped_stats(cut, sigma=3.0)
    peak = np.nanmax(cut) - med
    if not np.isfinite(peak) or std <= 0 or peak < min_snr * std:
        return x, y, False                      # nothing detectable here

    sub = cut - med
    sub[sub < 0] = 0
    try:
        cx, cy = centroid_com(sub)
    except Exception:                           # noqa: BLE001
        return x, y, False
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return x, y, False
    nx, ny = x0 + cx, y0 + cy
    # a centroid that ran to the edge of the box is not a real detection
    if abs(nx - x) > box * 0.8 or abs(ny - y) > box * 0.8:
        return x, y, False
    return float(nx), float(ny), True


def measure(data: np.ndarray, positions: list[tuple[float, float]],
            r_ap, r_in: float, r_out: float) -> np.ndarray:
    """
    Aperture photometry with a local background annulus.

    r_ap may be a single radius or a sequence of radii. Measuring several
    radii in one pass costs almost nothing (the background annulus is shared)
    and lets the pipeline choose the aperture that actually minimises scatter
    rather than guessing a multiple of FWHM.

    Returns background-subtracted flux, shaped (n_radii, n_positions) when
    given a sequence, or (n_positions,) for a single radius.
    """
    radii = np.atleast_1d(np.asarray(r_ap, dtype=float))
    anns = CircularAnnulus(positions, r_in=r_in, r_out=r_out)

    ann_masks = anns.to_mask(method="center")
    bkg_median = []
    for m in ann_masks:
        vals = m.multiply(data)
        vals = vals[m.data > 0]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            bkg_median.append(np.nan)
            continue
        _, med, _ = sigma_clipped_stats(vals, sigma=3.0)
        bkg_median.append(med)
    bkg_median = np.array(bkg_median)

    out = []
    for r in radii:
        aps = CircularAperture(positions, r=float(r))
        phot = aperture_photometry(data, aps)
        out.append(np.array(phot["aperture_sum"]) - bkg_median * aps.area)
    out = np.array(out)
    return out[0] if np.ndim(r_ap) == 0 else out


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


def clean_curve(times, flux, err=None, sigma: float = 5.0):
    """
    Drop frames whose relative flux is a wild outlier.

    Necessary because a single frame with a corrupted measurement (empty
    aperture, satellite trail, cloud) produces a ratio orders of magnitude
    from unity, and least-squares fitting will happily wreck the whole model
    chasing it. Uses MAD, so a contiguous block of bad frames can't inflate
    the threshold and hide itself.
    """
    times = np.asarray(times, float)
    flux = np.asarray(flux, float)
    good = np.isfinite(times) & np.isfinite(flux) & (flux > 0)
    if good.sum() < 5:
        return good

    med = np.median(flux[good])
    mad = 1.4826 * np.median(np.abs(flux[good] - med))
    if mad <= 0:
        return good
    good &= np.abs(flux - med) < sigma * mad
    return good


def normalize_out_of_transit(times, flux, mid=None, duration_days=None):
    """
    Normalize to the OUT-OF-TRANSIT baseline when the transit window is
    known, rather than to the median of everything.

    Matters more than it sounds: if most of your frames fall inside the
    transit — easy to do when the session is short — the global median sits
    partway down the dip, the baseline reads high, and the measured depth
    comes out wrong.
    """
    times = np.asarray(times, float)
    flux = np.asarray(flux, float)
    if mid is None or duration_days is None:
        ref = np.nanmedian(flux)
    else:
        oot = np.abs(times - mid) > (duration_days / 2.0)
        ref = np.nanmedian(flux[oot]) if oot.sum() >= 5 else np.nanmedian(flux)
    return flux / ref if np.isfinite(ref) and ref != 0 else flux
