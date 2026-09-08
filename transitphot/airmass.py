"""
Airmass detrending.

Every one of these light curves has shown a U-shaped baseline: the target
sits high mid-session and lower at both ends, so atmospheric extinction dims
it symmetrically around culmination. A polynomial in time can absorb some of
that, but the trend is not a polynomial in time — it is a function of
airmass, which depends on where the target actually was.

Fitting an explicit airmass term is what AstroImageJ does with its "detrend
vectors", and it matters for more than tidiness: when the baseline is not
modelled properly, the transit parameters absorb it. The trapezoid tilts its
continuum and reads the depth shallow; the limb-darkened model holds its
continuum flat and reads it deep. On TrES-3 b those two behaviours produced
depths of 14,524 and 23,246 ppm from the same frames, and mid-times five
minutes apart.

The physical model is Beer-Lambert: observed flux is attenuated by
exp(-k * X), where X is airmass and k the extinction coefficient in the
observing band. Differential photometry removes most of this — the
comparison stars are dimmed too — but only exactly so if target and
comparisons have identical colours. They never do, so a residual
colour-dependent term survives, and that is what we fit.
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
    Multiplicative extinction term, normalised to the median airmass.

    Normalising means k is the only free parameter and the term equals 1 at
    the middle of the session, so it cannot trade against the overall
    baseline level.
    """
    X = np.asarray(airmass, dtype=float)
    return np.exp(-k * (X - np.nanmedian(X)))


def describe(airmass: np.ndarray) -> str:
    X = np.asarray(airmass, dtype=float)
    return (f"airmass {np.nanmin(X):.2f}-{np.nanmax(X):.2f} "
            f"(median {np.nanmedian(X):.2f})")
