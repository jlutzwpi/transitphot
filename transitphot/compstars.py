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
3. **Isolation** — no neighbor within the photometric aperture; blends
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


def query_vsx(ra_deg: float, dec_deg: float, radius_arcmin: float = 20.0):
    """
    Known variable stars in the field, from AAVSO's VSX via VizieR.

    Gaia's phot_variable_flag is sparse: it reads VARIABLE only for sources
    Gaia's own variability pipeline classified, and NOT_AVAILABLE — "not
    assessed", not "constant" — for almost everything else. VSX aggregates
    ASAS-SN, ZTF, Kepler, TESS and decades of other surveys, so it catches
    far more of the variables that would otherwise pass as comparisons.

    Returns (SkyCoord, names, types), or None if VizieR can't be reached —
    in which case the caller carries on with Gaia's flag alone.
    """
    try:
        from astroquery.vizier import Vizier
        v = Vizier(columns=["Name", "Type", "RAJ2000", "DEJ2000"],
                   row_limit=-1, timeout=60)
        res = v.query_region(SkyCoord(ra_deg * u.deg, dec_deg * u.deg),
                             radius=radius_arcmin * u.arcmin,
                             catalog="B/vsx/vsx")
    except Exception:                                    # noqa: BLE001
        return None
    if not res or len(res[0]) == 0:
        return SkyCoord([], [], unit="deg"), [], []
    t = res[0]
    try:
        coords = SkyCoord(t["RAJ2000"], t["DEJ2000"], unit=(u.deg, u.deg))
    except Exception:                                    # noqa: BLE001
        coords = SkyCoord(t["RAJ2000"], t["DEJ2000"],
                          unit=(u.hourangle, u.deg))
    names = [str(x) for x in t["Name"]]
    types = [str(x) for x in t["Type"]] if "Type" in t.colnames else [""] * len(t)
    return coords, names, types


def target_color_from(sources: Table, ra_deg: float, dec_deg: float,
                      max_sep_arcsec: float = 3.0) -> float | None:
    """
    The target's own BP-RP color, taken from the same Gaia query.

    Color matching needs the target's color to compare against. Without it
    the check silently does nothing, which is exactly what happened: the
    selection accepted a target_color but was never given one.
    """
    if len(sources) == 0:
        return None
    tgt = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    coords = SkyCoord(sources["ra"], sources["dec"], unit="deg")
    sep = tgt.separation(coords).arcsec
    i = int(np.argmin(sep))
    if sep[i] > max_sep_arcsec:
        return None
    c = sources["bp_rp"][i]
    if c is np.ma.masked or c is None:
        return None
    try:
        return float(c)
    except (TypeError, ValueError):
        return None


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
           filter_band: str | None = None,
           variables=None,
           rejected: list | None = None,
           vsx_match_arcsec: float = 5.0) -> list[CompCandidate]:
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

    # Known variables from VSX: index each Gaia source to its nearest entry.
    vsx_hit = [None] * len(sources)
    if variables is not None and len(variables[0]) > 0:
        vcoords, vnames, vtypes = variables
        vi, vsep, _ = coords.match_to_catalog_sky(vcoords)
        for i in range(len(sources)):
            if vsep[i].arcsec <= vsx_match_arcsec:
                vsx_hit[i] = (vnames[vi[i]], vtypes[vi[i]])

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
            continue                          # blended with a neighbor
        if str(row["phot_variable_flag"]).upper().startswith("VARIABLE"):
            if rejected is not None:
                rejected.append((mag, sep, "flagged variable by Gaia"))
            continue
        if vsx_hit[i] is not None:
            if rejected is not None:
                name, vtype = vsx_hit[i]
                rejected.append((mag, sep, f"known variable {name}"
                                 + (f" ({vtype})" if vtype else "")
                                 + " in VSX"))
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
    Post-hoc check against the actual data, measured DIFFERENTIALLY.

    Critical subtlety: clouds, transparency changes and airmass dim every
    star in the field together. Differential photometry divides that out, so
    common-mode variation is exactly what we must NOT penalize. An earlier
    version of this function tested each star's raw flux scatter and rejected
    every comparison on a partly cloudy night — discarding good data for the
    one reason that doesn't matter.

    So: normalize each star against the ensemble of the *others*, then test
    the residual scatter. That isolates variation intrinsic to the star (real
    variability, drift onto a bad column, a satellite trail) from variation
    shared by the whole field.

    fluxes: (n_stars, n_frames)
    Returns a boolean mask of stars to keep.
    """
    fluxes = np.asarray(fluxes, dtype=float)
    n_stars = fluxes.shape[0]
    if n_stars == 1:
        return np.array([True])

    scatters = np.full(n_stars, np.nan)
    for i in range(n_stars):
        others = np.delete(fluxes, i, axis=0)
        ensemble = np.nansum(others, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = fluxes[i] / ensemble
        med = np.nanmedian(ratio)
        if not np.isfinite(med) or med == 0:
            continue
        norm = ratio / med
        # robust scatter: MAD isn't thrown off by a few bad frames
        scatters[i] = 1.4826 * np.nanmedian(np.abs(norm - 1.0))

    finite = np.isfinite(scatters)
    keep = finite & (scatters < threshold)
    if keep.sum() == 0 and finite.any():
        # Everything looked marginal (a genuinely rough night). Keep the best
        # half rather than failing outright — a noisy curve beats no curve,
        # and the reported scatter tells the user what they got.
        order = np.argsort(np.where(finite, scatters, np.inf))
        keep[order[:max(n_stars // 2, 1)]] = True
    return keep


def stability_report(fluxes: np.ndarray) -> np.ndarray:
    """Differential scatter per star, in ppm — for reporting to the user."""
    fluxes = np.asarray(fluxes, dtype=float)
    out = np.full(fluxes.shape[0], np.nan)
    for i in range(fluxes.shape[0]):
        others = np.delete(fluxes, i, axis=0)
        if others.size == 0:
            continue
        ensemble = np.nansum(others, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = fluxes[i] / ensemble
        med = np.nanmedian(ratio)
        if np.isfinite(med) and med != 0:
            norm = ratio / med
            out[i] = 1.4826 * np.nanmedian(np.abs(norm - 1.0)) * 1e6
    return out
