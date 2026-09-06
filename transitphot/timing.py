"""
Time conversion for photometry.

Why this matters: a frame's DATE-OBS is UTC at your telescope. The transit
happens at the barycenter of the solar system, and Earth's orbital position
shifts light arrival time by up to ±8.3 minutes over a year. Submitting
JD(UTC) mid-times instead of BJD_TDB introduces exactly that error — larger
than the timing precision the measurement is capable of, and larger than the
ephemeris drift the observation is meant to correct.

So: always convert before fitting or submitting.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time


def jd_utc_to_bjd_tdb(jd_utc: np.ndarray, ra_deg: float, dec_deg: float,
                      lat_deg: float, lon_deg: float,
                      elevation_m: float = 0.0) -> np.ndarray:
    """
    Convert an array of JD(UTC) mid-exposure times to BJD_TDB.

    Applies both corrections astropy provides:
      * the Rømer delay (light travel time to the solar-system barycenter),
      * the UTC -> TDB timescale conversion (relativistic clock differences).
    """
    site = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg,
                         height=elevation_m * u.m)
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    t = Time(np.asarray(jd_utc, dtype=float), format="jd", scale="utc",
             location=site)
    ltt = t.light_travel_time(target, kind="barycentric")
    return (t.tdb + ltt).jd


def bjd_to_phase(bjd: np.ndarray, epoch_bjd: float, period_days: float
                 ) -> np.ndarray:
    """Orbital phase in [-0.5, 0.5), with 0 at mid-transit."""
    ph = ((np.asarray(bjd) - epoch_bjd) / period_days) % 1.0
    return np.where(ph > 0.5, ph - 1.0, ph)
