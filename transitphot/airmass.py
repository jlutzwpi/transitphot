"""
Airmass detrending.

Every one of these light curves has shown a U-shaped baseline: the target
sits high mid-session and lower at both ends, so atmospheric extinction dims
it symmetrically around culmination. A polynomial in time can absorb some of
that, but the trend is not a polynomial in time — it is a function of
airmass, which depends on where the target actually was.

Fitting an explicit airmass term is what AstroImageJ does with its "detrend
vectors", and it matters for more than tidiness: when the baseline is not
modeled properly, the transit parameters absorb it. The trapezoid tilts its
continuum and reads the depth shallow; the limb-darkened model holds its
continuum flat and reads it deep. On TrES-3 b those two behaviors produced
depths of 14,524 and 23,246 ppm from the same frames, and mid-times five
minutes apart.

The physical model is Beer-Lambert: observed flux is attenuated by
exp(-k * X), where X is airmass and k the extinction coefficient in the
observing band. Differential photometry removes most of this — the
comparison stars are dimmed too — but only exactly so if target and
comparisons have identical colors. They never do, so a residual
color-dependent term survives, and that is what we fit.
"""

from __future__ import annotations

import numpy as np


def airmass_series(jd_utc, ra_deg: float, dec_deg: float,
                   lat_deg: float, lon_deg: float,
                   elevation_m: float = 0.0) -> np.ndarray:
    """Airmass at each timestamp, via the Kasten-Young approximation."""
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg,
                        height=elevation_m * u.m)
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    t = Time(np.asarray(jd_utc, dtype=float), format="jd", scale="utc")
    alt = target.transform_to(AltAz(obstime=t, location=loc)).alt.deg

    z = 90.0 - np.asarray(alt, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        X = 1.0 / (np.cos(np.radians(z))
                   + 0.50572 * (96.07995 - z) ** -1.6364)
    return np.where(np.isfinite(X) & (alt > 0), X, 40.0)


def detrend_factor(airmass: np.ndarray, k: float) -> np.ndarray:
    """
    Multiplicative extinction term, normalized to the median airmass.

    Normalizing means k is the only free parameter and the term equals 1 at
    the middle of the session, so it cannot trade against the overall
    baseline level.
    """
    X = np.asarray(airmass, dtype=float)
    return np.exp(-k * (X - np.nanmedian(X)))


def describe(airmass: np.ndarray) -> str:
    X = np.asarray(airmass, dtype=float)
    return (f"airmass {np.nanmin(X):.2f}-{np.nanmax(X):.2f} "
            f"(median {np.nanmedian(X):.2f})")


def transit_window_risk(epoch_bjd: float, period_days: float,
                        duration_hours: float, target_mag: float,
                        ra_deg: float, dec_deg: float,
                        lat_deg: float, lon_deg: float,
                        elevation_m: float = 0.0,
                        baseline_hours: float = 1.0,
                        now_jd: float | None = None):
    """
    Predict airmass across the next transit's observing window and flag it
    as risky before a single frame is taken.

    A bright target tolerates a session running out to airmass 2 (30 deg
    elevation) just fine - it has SNR to spare. A faint one does not: on
    Kepler-731 b (G=15.3), the transit itself occurred at a perfectly
    reasonable airmass 1.23, but the post-egress baseline needed to anchor
    the fit ran the session out to airmass 2.02, and that tail (not the
    transit) is what dragged the whole night below AAVSO-submission
    quality. Compare a clean night on this same setup, TrES-2 b (G=11.3),
    which never exceeded airmass 1.23 all night. The failure mode is
    silent until the data is already bad, which is exactly why this needs
    to be caught before committing a night to it rather than after.

    These thresholds aren't a hard physical cliff - they're this project's
    accumulated evidence of where "comps are faint too" and "scintillation
    noise is a bigger fraction of a weak signal" start to bite:
      target_mag <= 13.0        -> flag only above airmass 2.2
      13.0 < target_mag <= 14.5 -> flag above airmass 2.0
      target_mag > 14.5         -> flag above airmass 1.5

    Returns (predicted_mid_jd, max_airmass, limit, risky: bool).
    """
    import time as _time

    if now_jd is None:
        now_jd = _time.time() / 86400.0 + 2440587.5

    n = round((now_jd - epoch_bjd) / period_days)
    mid = epoch_bjd + n * period_days
    if mid < now_jd:                      # always predict the NEXT transit
        n += 1
        mid = epoch_bjd + n * period_days

    half_window = duration_hours / 24.0 / 2.0 + baseline_hours / 24.0
    grid = np.linspace(mid - half_window, mid + half_window, 60)
    X = airmass_series(grid, ra_deg, dec_deg, lat_deg, lon_deg, elevation_m)
    max_airmass = float(np.nanmax(X))

    if target_mag <= 13.0:
        limit = 2.2
    elif target_mag <= 14.5:
        limit = 2.0
    else:
        limit = 1.5

    return mid, max_airmass, limit, max_airmass > limit


# Filters that pass most or all of the visible band, and so admit the Moon's
# light with little rejection. Named explicitly rather than inferred (e.g.
# "wide-ish"), because guessing wrong in either direction is worse than
# asking the filter to be spelled one of these ways.
BROADBAND_FILTERS = frozenset({"L", "LUM", "LUMINANCE", "CLEAR", "CLR", "C"})


def moon_glow_risk(mid_jd: float, ra_deg: float, dec_deg: float,
                   lat_deg: float, lon_deg: float, elevation_m: float,
                   target_mag: float, filter_band: str | None):
    """
    Flag a bright Moon through a broadband filter on a faint target.

    On Kepler-731 b (G=15.3, Luminance filter), a 72%-illuminated Moon at
    33 deg altitude pushed sky background to 7-15x the star's own signal
    in the photometric aperture - on an airmass-good frame, early in the
    night, well before guiding or elevation had anything to do with it.
    That made the whole night sky-noise-limited by the time a single
    frame was taken, regardless of equipment quality.

    A narrow filter (R, V, g, r, i...) rejects most of the Moon's light
    while a bright target barely needs the SNR margin broadband buys it;
    a faint target needs exactly the opposite trade. Separation from the
    Moon matters far less than this: scattered moonlight floods the whole
    sky, not just the line of sight to the Moon itself, so this checks
    illumination and altitude, not angular distance to the target.

    Returns (illum_fraction, moon_alt_deg, risky: bool, reason: str|None).
    """
    from astropy.coordinates import AltAz, EarthLocation, get_body
    from astropy.time import Time
    from astropy import units as u

    loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg,
                        height=elevation_m * u.m)
    t = Time(mid_jd, format="jd", scale="utc")
    moon = get_body("moon", t, location=loc)
    sun = get_body("sun", t, location=loc)
    elong = moon.separation(sun).deg
    illum = 0.5 * (1.0 - np.cos(np.radians(elong)))
    moon_alt = float(moon.transform_to(AltAz(obstime=t, location=loc)).alt.deg)

    is_broadband = (filter_band or "").strip().upper() in BROADBAND_FILTERS
    risky = (target_mag > 14.5 and moon_alt > 0.0 and illum > 0.4
            and is_broadband)

    reason = None
    if risky:
        reason = (
            f"Moon {illum * 100:.0f}% illuminated at {moon_alt:.0f} deg "
            f"altitude, through a broadband ({filter_band}) filter on a "
            f"mag {target_mag:.1f} target - sky background is likely to "
            f"dominate the star's own signal all night. A narrower filter "
            f"(R or V) would help far more here than better guiding, no "
            f"clouds, or higher elevation.")

    return illum, moon_alt, risky, reason
