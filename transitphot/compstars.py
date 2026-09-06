"""
Comparison-star selection.

This is the step amateurs most often get wrong, and the one AstroImageJ
leaves entirely manual. Bad comparison stars are the difference between a
clean 3 mmag light curve and an unusable one.

Selection criteria, in the order they matter:

1. **Brightness** — within ~1.5 mag of the target. Much brighter risks
   saturation; much fainter adds its own shot noise to every measurement.
2. **Color** — similar BP-RP to the target. Differential atmospheric
   extinction is color-dependent, so a red comparison against a blue target
   introduces a slow trend across the night that mimics (or hides) a transit.
3. **Isolation** — no neighbour within the photometric aperture; blends
   corrupt the flux.
4. **Non-variability** — Gaia's variability flag plus a stability check
   against the actual frames.
5. **Proximity** — closer to the target means more similar airmass and
   optical path, but this matters far less than 1-3, so it's the tiebreak.

Ensemble photometry (summing several comparisons) beats a single star:
the combined reference has lower shot noise, and one misbehaving star is
diluted rather than fatal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table


@dataclass
class CompCandidate:
    ra_deg: float
    dec_deg: float
    mag: float
    color: float | None
    sep_arcmin: float
    score: float
    reasons: list[str]


def query_field(ra_deg: float, dec_deg: float, radius_arcmin: float = 20.0,
                mag_limit: float = 16.0) -> Table:
    """Fetch Gaia sources in the field. Requires network access."""
    from astroquery.gaia import Gaia

    q = f"""
    SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp,
           phot_variable_flag, ruwe
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_arcmin / 60.0})) = 1
      AND phot_g_mean_mag < {mag_limit}
    """
    return Gaia.launch_job_async(q).get_results()


def color_tolerance_for(filter_band: str | None) -> float:
    """
    How closely a comparison star's color must match the target.

    Differential atmospheric extinction is wavelength-dependent, so a color
    mismatch produces a slow trend across the night as airmass changes. A
    narrower, redder passband reduces (but does not remove) the effect:
    two stars of different spectral type still have different effective
    wavelengths within the same filter.
    """
    if not filter_band:
        return 0.5
    b = filter_band.strip().upper().rstrip("C")
    if b in {"L", "LUM", "CLEAR", "C", "NONE"}:
        return 0.4          # widest band, strongest color dependence
    if b in {"B", "G", "V"}:
        return 0.5
    if b in {"R"}:
        return 0.8          # narrower and redder: more forgiving
    if b in {"I", "IR", "SLOAN I", "IZ"}:
        return 1.0
    return 0.5


def select(target_ra: float, target_dec: float, target_mag: float,
           sources: Table, n: int = 5,
           mag_tolerance: float = 1.5,
           color_tolerance: float | None = None,
           min_separation_arcsec: float = 15.0,
           target_color: float | None = None,
           filter_band: str | None = None) -> list[CompCandidate]:
    """
    Rank field stars as comparison candidates. Returns the best `n`.

    The score is deliberately interpretable — each candidate carries the
    reasons it ranked where it did, so a user can sanity-check the choice
    instead of trusting a black box.
    """
    if color_tolerance is None:
        color_tolerance = color_tolerance_for(filter_band)

    tgt = SkyCoord(target_ra * u.deg, target_dec * u.deg)
    coords = SkyCoord(sources["ra"], sources["dec"], unit="deg")
    seps = tgt.separation(coords).arcmin

    # Pairwise separations, for isolation testing
    idx, sep2d, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)
    nearest_arcsec = sep2d.arcsec

    out: list[CompCandidate] = []
    for i, row in enumerate(sources):
        mag = float(row["phot_g_mean_mag"])
        color = float(row["bp_rp"]) if row["bp_rp"] is not np.ma.masked else None
        sep = float(seps[i])

        if sep < 0.05:                       # this is the target itself
            continue
        dmag = abs(mag - target_mag)
        if dmag > mag_tolerance:
            continue
        if nearest_arcsec[i] < min_separation_arcsec:
            continue                          # blended with a neighbour
        if str(row["phot_variable_flag"]).upper().startswith("VARIABLE"):
            continue

        reasons = []
        score = 100.0

        score -= (dmag / mag_tolerance) * 30
        reasons.append(f"Δmag {dmag:+.2f}")

        if target_color is not None and color is not None:
            dcol = abs(color - target_color)
            score -= min(dcol / color_tolerance, 2.0) * 25
            reasons.append(f"ΔBP-RP {dcol:.2f}")
            if dcol > color_tolerance:
                reasons.append("color mismatch — extinction trend risk")

        score -= min(sep / 20.0, 1.0) * 10
        reasons.append(f"{sep:.1f}' away")

        if float(row["ruwe"] or 0) > 1.4:
            score -= 15
            reasons.append("high RUWE — possible unresolved binary")

        out.append(CompCandidate(float(row["ra"]), float(row["dec"]), mag,
                                 color, sep, round(score, 1), reasons))

    out.sort(key=lambda c: -c.score)
    return out[:n]


def check_stability(fluxes: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """
    Post-hoc check against the actual data: a comparison star whose
    normalized flux scatters more than `threshold` (fractional) across the
    session is misbehaving — clouds, drift onto a bad column, or genuine
    variability. Returns a boolean mask of stars to keep.

    fluxes: (n_stars, n_frames) array of measured fluxes.
    """
    norm = fluxes / np.nanmedian(fluxes, axis=1, keepdims=True)
    scatter = np.nanstd(norm, axis=1)
    return scatter < threshold
