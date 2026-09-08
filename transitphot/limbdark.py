"""
Limb-darkened transit fitting.

The trapezoid model in fitting.py is robust and fast, but a real transit is
not trapezoidal: the stellar disc is brighter at its centre than its limb, so
the light curve has curved shoulders and a rounded floor. Two consequences
measured on real data:

* **Depth comes out shallow.** A trapezoid's flat bottom sits above the true
  centre depth of a limb-darkened profile. Across four targets this ran
  10-15% low against catalogue values, while AstroImageJ's limb-darkened fit
  recovered them.
* **Ingress shape is wrong**, which can pull the mid-time when the baseline
  is short or asymmetric.

This module fits the Mandel-Agol model via `batman`, which is the same
physics AstroImageJ uses. It is optional: `pip install "transitphot[ld]"`.
When batman is absent the CLI falls back to the trapezoid and says so.

Parameters fitted: mid-time, planet-to-star radius ratio, scaled semi-major
axis, inclination, plus a quadratic baseline. Limb-darkening coefficients
are held fixed — they are poorly constrained by a single ground-based
transit, and floating them mostly buys degeneracy with depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class LimbDarkFit:
    mid_bjd: float
    mid_err_days: float
    rp_rs: float                 # planet/star radius ratio
    rp_rs_err: float
    a_rs: float                  # semi-major axis in stellar radii
    inclination_deg: float
    depth_ppm: float             # (Rp/Rs)^2 — the catalogue convention
    depth_err_ppm: float
    central_depth_ppm: float     # the observed dip at mid-transit
    duration_hours: float
    rms_ppm: float
    n_points: int
    u1: float
    u2: float

    @property
    def mid_err_minutes(self) -> float:
        return self.mid_err_days * 24 * 60


def available() -> bool:
    try:
        import batman  # noqa: F401
        return True
    except ImportError:
        return False


def _model_factory(t, period, u1, u2):
    """
    Build a model evaluator. Uses batman when installed — it is the reference
    implementation and what AstroImageJ uses — and otherwise falls back to the
    numpy integrator below, so limb-darkened fitting works with no extra
    dependency.
    """
    if not available():
        return _numpy_model_factory(period, u1, u2)
    import batman

    params = batman.TransitParams()
    params.per = period
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = "quadratic"
    params.u = [u1, u2]

    def model(t_, mid, rp, a, inc, base, slope, curve):
        params.t0 = mid
        params.rp = rp
        params.a = a
        params.inc = inc
        m = batman.TransitModel(params, np.asarray(t_, dtype=float))
        flux = m.light_curve(params)
        dt = np.asarray(t_, dtype=float) - mid
        return flux * (base + slope * dt + curve * dt ** 2)

    return model


def transit_duration_hours(period, rp_rs, a_rs, inc_deg) -> float:
    """Total (first-to-fourth contact) duration from the fitted geometry."""
    b = a_rs * np.cos(np.radians(inc_deg))
    arg = ((1 + rp_rs) ** 2 - b ** 2) / (a_rs ** 2 - b ** 2)
    if arg <= 0:
        return float("nan")
    return float(period / np.pi * np.arcsin(np.sqrt(arg)) * 24.0)


def fit(bjd, flux, flux_err=None, *, period: float,
        expected_mid: float, expected_depth: float | None = None,
        expected_duration_hours: float | None = None,
        u1: float = 0.35, u2: float = 0.25) -> LimbDarkFit:
    """
    Fit the Mandel-Agol model. `period` is required — the model needs it to
    convert between orbital phase and time, and it is known far better from
    the archive than a single transit could determine.
    """
    bjd = np.asarray(bjd, dtype=float)
    flux = np.asarray(flux, dtype=float)
    good = np.isfinite(bjd) & np.isfinite(flux)
    bjd, flux = bjd[good], flux[good]
    sigma = (np.asarray(flux_err, dtype=float)[good]
             if flux_err is not None else None)
    if sigma is not None and not np.all(np.isfinite(sigma) & (sigma > 0)):
        sigma = None

    rp0 = np.sqrt(expected_depth) if expected_depth else 0.12
    # a/Rs from the expected duration, if given; otherwise a typical hot-Jupiter
    if expected_duration_hours:
        a0 = period * 24.0 / (np.pi * expected_duration_hours)
        a0 = float(np.clip(a0, 2.0, 60.0))
    else:
        a0 = 8.0

    model = _model_factory(bjd, period, u1, u2)

    # Bounds: the quadratic baseline is deliberately tight. Over a few-hour
    # window a loose curvature term can imitate a broad shallow transit, and
    # the fit will happily shrink the planet to nothing to let it. Rp/Rs is
    # also held near the catalogue value for the same reason — a single
    # ground-based transit does not redetermine planet size.
    lo = [bjd.min(), rp0 * 0.5, 1.5, 60.0, 0.95, -2.0, -3.0]
    hi = [bjd.max(), rp0 * 1.6, 60.0, 90.0, 1.05, 2.0, 3.0]

    # Multi-start over mid-time: the chi-square surface has shallow local
    # minima, and a single descent lands in whichever one it started nearest.
    span = bjd.max() - bjd.min()
    starts = [expected_mid] + list(np.linspace(bjd.min() + 0.15 * span,
                                               bjd.max() - 0.15 * span, 7))
    best = best_cov = None
    best_chi2 = np.inf
    for m in starts:
        p0 = [float(np.clip(m, lo[0], hi[0])), rp0, a0, 88.0, 1.0, 0.0, 0.0]
        p0 = [float(np.clip(v, a, b)) for v, a, b in zip(p0, lo, hi)]
        try:
            popt, pcov = curve_fit(model, bjd, flux, p0=p0, bounds=(lo, hi),
                                   sigma=sigma,
                                   absolute_sigma=sigma is not None,
                                   maxfev=40000)
        except Exception:                               # noqa: BLE001
            continue
        resid = flux - model(bjd, *popt)
        chi2 = float(np.sum((resid / (sigma if sigma is not None else 1.0)) ** 2))
        if chi2 < best_chi2:
            best, best_cov, best_chi2 = popt, pcov, chi2
    if best is None:
        raise RuntimeError("Limb-darkened fit did not converge.")
    popt, pcov = best, best_cov

    perr = np.sqrt(np.diag(pcov))
    resid = flux - model(bjd, *popt)

    mid, rp, a, inc = popt[0], popt[1], popt[2], popt[3]
    depth = rp ** 2
    depth_err = 2 * rp * perr[1]
    # Two different numbers get called "depth". (Rp/Rs)^2 is the geometric
    # ratio catalogues list; the observed dip at mid-transit is deeper,
    # because the planet covers the bright centre of a limb-darkened disc.
    # AstroImageJ quotes the latter, so report both to avoid comparing
    # incompatible quantities.
    central = 1.0 - float(_occulted_flux([0.0], rp, u1, u2)[0])
    return LimbDarkFit(
        mid_bjd=float(mid), mid_err_days=float(perr[0]),
        rp_rs=float(rp), rp_rs_err=float(perr[1]),
        a_rs=float(a), inclination_deg=float(inc),
        depth_ppm=float(depth * 1e6), depth_err_ppm=float(depth_err * 1e6),
        central_depth_ppm=float(central * 1e6),
        duration_hours=transit_duration_hours(period, rp, a, inc),
        rms_ppm=float(np.std(resid) * 1e6), n_points=len(bjd),
        u1=u1, u2=u2,
    )


def model_curve(t, fit_result: LimbDarkFit, period: float,
                base: float = 1.0, slope: float = 0.0, curve: float = 0.0):
    """Evaluate the fitted model on an arbitrary time grid, for plotting."""
    model = _model_factory(t, period, fit_result.u1, fit_result.u2)
    return model(t, fit_result.mid_bjd, fit_result.rp_rs, fit_result.a_rs,
                 fit_result.inclination_deg, base, slope, curve)


# ----------------------------------------------------------------------
# Fallback model — no batman required
# ----------------------------------------------------------------------
def _occulted_flux(z: np.ndarray, p: float, u1: float, u2: float,
                   n_ann: int = 400) -> np.ndarray:
    """
    Relative flux of a quadratically limb-darkened star occulted by an opaque
    disc of radius p (in stellar radii) at projected separation z.

    Computed by integrating over annuli of the stellar disc: for each annulus
    of radius r, work out what fraction of its circumference lies behind the
    planet, weight by the limb-darkening intensity, and sum. Accurate to a
    few ppm at 400 annuli, which is far below any ground-based noise floor,
    and needs nothing beyond numpy.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    r = (np.arange(n_ann) + 0.5) / n_ann          # annulus centres, 0..1
    dr = 1.0 / n_ann
    mu = np.sqrt(np.clip(1.0 - r ** 2, 0.0, 1.0))
    intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    total = np.sum(intensity * 2.0 * np.pi * r * dr)

    R = r[None, :]                                 # (1, n_ann)
    Z = np.abs(z)[:, None]                         # (n_time, 1)

    # fraction of each annulus hidden behind the planet disc
    frac = np.zeros_like(R * Z)
    fully = (Z + p) <= R                           # planet inside the annulus?
    # no overlap when the planet disc is entirely outside or inside
    no_overlap = (R >= Z + p) | (R <= Z - p)
    inside = (R <= p - Z)                          # annulus wholly covered
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = (R ** 2 + Z ** 2 - p ** 2) / (2.0 * R * Z)
    cosang = np.clip(cosang, -1.0, 1.0)
    partial = ~no_overlap & ~inside & (Z > 0)
    frac = np.where(partial, np.arccos(cosang) / np.pi, frac)
    frac = np.where(inside, 1.0, frac)
    # planet exactly centred: annuli inside p are fully covered
    centred = (Z == 0) & (R <= p)
    frac = np.where(centred, 1.0, frac)

    blocked = np.sum(frac * intensity * 2.0 * np.pi * R * dr, axis=1)
    return 1.0 - blocked / total


def _numpy_model_factory(period, u1, u2):
    """Same signature as the batman factory, implemented in numpy."""
    def model(t_, mid, rp, a, inc, base, slope, curve):
        t_ = np.asarray(t_, dtype=float)
        phase = 2.0 * np.pi * (t_ - mid) / period
        # projected separation in stellar radii (circular orbit)
        z = a * np.sqrt(np.sin(phase) ** 2
                        + (np.cos(np.radians(inc)) * np.cos(phase)) ** 2)
        # behind the star on the far side: no transit
        z = np.where(np.cos(phase) < 0, 99.0, z)
        flux = _occulted_flux(z, rp, u1, u2)
        dt = t_ - mid
        return flux * (base + slope * dt + curve * dt ** 2)
    return model
