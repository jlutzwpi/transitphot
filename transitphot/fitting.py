"""
Transit model fitting — extract the observed mid-transit time.

Model: a trapezoid (flat baseline, linear ingress, flat bottom, linear
egress) rather than a full Mandel-Agol limb-darkened model. The trade-off:

* A trapezoid has 5 free parameters and fits robustly on noisy amateur data.
* Limb darkening rounds the bottom of a real transit, so a trapezoid slightly
  underestimates depth — but mid-time, the quantity ExoClock and AAVSO
  actually need, is symmetric and comes out essentially unbiased.
* If you want depth-accurate modeling, feed the CSV to `batman` or
  `exoplanet` afterwards. This module is about timing.

An optional linear airmass/time trend is fitted simultaneously, because
residual differential extinction is the single most common systematic in
amateur light curves and absorbing it into the model beats ignoring it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class TransitFit:
    mid_bjd: float
    mid_err_days: float
    depth: float
    depth_err: float
    duration_days: float
    ingress_days: float
    baseline_slope: float
    rms_ppm: float
    n_points: int

    @property
    def mid_err_minutes(self) -> float:
        return self.mid_err_days * 24 * 60

    @property
    def depth_ppm(self) -> float:
        return self.depth * 1e6


def trapezoid(t, mid, depth, duration, ingress, base, slope):
    """
    Trapezoidal transit plus a linear baseline trend.

    duration = first-to-fourth contact (total)
    ingress  = duration of the ingress ramp (= egress ramp)
    """
    dt = np.asarray(t, dtype=float) - mid
    half_total = duration / 2.0
    half_flat = max(half_total - ingress, 1e-9)

    f = np.ones_like(dt)
    # fully in transit
    f = np.where(np.abs(dt) <= half_flat, 1.0 - depth, f)
    # ingress / egress ramps
    ramp = (np.abs(dt) > half_flat) & (np.abs(dt) < half_total)
    frac = (half_total - np.abs(dt)) / ingress
    f = np.where(ramp, 1.0 - depth * np.clip(frac, 0, 1), f)
    return f * (base + slope * dt)


def fit(bjd: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None = None,
        expected_mid: float | None = None,
        expected_duration_hours: float | None = None,
        expected_depth: float | None = None,
        fix_duration: bool = False) -> TransitFit:
    """
    Fit the trapezoid model. Priors from TransitPlanner's prediction make the
    fit far more stable on marginal data — pass them when you have them.

    Implementation notes: BJD values are ~2.46e6, so fitting a mid-time
    directly is badly conditioned — we solve in days-from-origin and shift
    back at the end. The mid-time also has many shallow local minima on noisy
    data, so we multi-start across the window and keep the lowest chi-square
    rather than trusting a single descent.
    """
    bjd = np.asarray(bjd, dtype=float)
    flux = np.asarray(flux, dtype=float)
    good = np.isfinite(bjd) & np.isfinite(flux)
    bjd, flux = bjd[good], flux[good]
    sigma = (np.asarray(flux_err, dtype=float)[good]
             if flux_err is not None else None)
    if sigma is not None and not np.all(np.isfinite(sigma) & (sigma > 0)):
        sigma = None

    origin = float(np.floor(bjd.min()))
    x = bjd - origin                                    # ~O(1), well conditioned

    mid0 = ((expected_mid - origin) if expected_mid is not None
            else float(np.median(x)))
    dur0 = (expected_duration_hours or 2.5) / 24.0
    dep0 = expected_depth if expected_depth is not None else max(
        1.0 - float(np.percentile(flux, 5)), 1e-4)

    # Mid-time must lie INSIDE the observed window: a fit that puts
    # mid-transit beyond the data has not measured anything.
    lo = [x.min(), 1e-5, dur0 * 0.3, 1e-4, 0.9, -5.0]
    hi = [x.max(), 0.5,  dur0 * 3.0, dur0,  1.1,  5.0]
    if fix_duration and expected_duration_hours:
        # Duration is well known from the archive and trades against depth
        # and mid-time in a noisy fit. Pinning it removes a degeneracy that
        # otherwise lets the model wander onto a trend instead of the transit.
        lo[2], hi[2] = dur0 * 0.999, dur0 * 1.001

    # Multi-start: the prior, plus a scan across the observed window.
    starts = [mid0] + list(np.linspace(x.min() + dur0 / 2,
                                       x.max() - dur0 / 2, 9))
    best, best_chi2 = None, np.inf
    for m in starts:
        m = float(np.clip(m, lo[0] + 1e-6, hi[0] - 1e-6))
        p0 = [m, dep0, dur0, dur0 * 0.15, 1.0, 0.0]
        p0 = [float(np.clip(v, l, h)) for v, l, h in zip(p0, lo, hi)]
        try:
            # soft_l1 loss: quadratic near zero, linear in the tails, so a
            # few surviving outliers bend the fit instead of dictating it.
            popt, pcov = curve_fit(trapezoid, x, flux, p0=p0, bounds=(lo, hi),
                                   sigma=sigma, absolute_sigma=sigma is not None,
                                   loss="soft_l1", f_scale=0.01, maxfev=20000)
        except Exception:                               # noqa: BLE001
            continue
        resid = flux - trapezoid(x, *popt)
        chi2 = float(np.sum((resid / (sigma if sigma is not None else 1.0)) ** 2))
        if chi2 < best_chi2:
            best, best_chi2, best_cov, best_resid = popt, chi2, pcov, resid

    if best is None:
        raise RuntimeError("Transit fit did not converge — check the light curve.")

    perr = np.sqrt(np.diag(best_cov))
    # NOTE: with sigma=None, curve_fit already scales the covariance by
    # chi2/dof (absolute_sigma=False), so no further rescaling here — doing
    # it twice collapses the reported uncertainty to ~0.
    #
    # With a robust loss the covariance estimate is unreliable and can come
    # back non-finite or absurdly large. A mid-time error bar is the number
    # people submit, so when it looks broken we measure it directly by
    # bootstrapping the residuals instead of reporting garbage.
    span_days = float(x.max() - x.min())
    if not np.isfinite(perr[0]) or perr[0] > span_days:
        perr = _bootstrap_errors(x, flux, best, lo, hi, sigma)

    return TransitFit(
        mid_bjd=float(best[0]) + origin, mid_err_days=float(perr[0]),
        depth=float(best[1]), depth_err=float(perr[1]),
        duration_days=float(best[2]), ingress_days=float(best[3]),
        baseline_slope=float(best[5]),
        rms_ppm=float(np.std(best_resid) * 1e6), n_points=len(x),
    )


def o_minus_c_minutes(observed_mid_bjd: float, predicted_mid_bjd: float
                      ) -> float:
    """Observed minus Calculated, in minutes — the number that updates an
    ephemeris. Positive means the transit ran late."""
    return (observed_mid_bjd - predicted_mid_bjd) * 24 * 60


def _bootstrap_errors(x, flux, popt, lo, hi, sigma, n_boot: int = 24):
    """
    Residual bootstrap: refit repeatedly on resampled residuals and take the
    scatter of the recovered parameters. Slower than reading the covariance
    matrix, but it reports what the data actually constrain.
    """
    rng = np.random.default_rng(12345)
    model = trapezoid(x, *popt)
    resid = flux - model
    draws = []
    for _ in range(n_boot):
        sample = model + rng.choice(resid, size=len(resid), replace=True)
        try:
            p, _ = curve_fit(trapezoid, x, sample, p0=popt, bounds=(lo, hi),
                             sigma=sigma, loss="soft_l1", f_scale=0.01,
                             maxfev=20000)
            draws.append(p)
        except Exception:                               # noqa: BLE001
            continue
    if len(draws) < 5:
        return np.full(len(popt), np.nan)
    return np.std(np.array(draws), axis=0)
