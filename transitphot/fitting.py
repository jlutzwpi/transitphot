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
    baseline_curve: float
    k_extinction: float
    baseline_model: str
    rms_ppm: float
    n_points: int

    @property
    def mid_err_minutes(self) -> float:
        return self.mid_err_days * 24 * 60

    @property
    def depth_ppm(self) -> float:
        return self.depth * 1e6


def trapezoid(t, mid, depth, duration, ingress, base, slope, curve=0.0,
              k_ext=0.0, airmass=None):
    """
    Trapezoidal transit multiplied by a quadratic baseline.

    duration = first-to-fourth contact (total)
    ingress  = duration of the ingress ramp (= egress ramp)
    base, slope, curve = baseline continuum, in days from mid

    The quadratic term matters: airmass changes non-linearly through a
    session, so a straight-line baseline leaves curvature that the transit
    parameters absorb — typically by stretching the ingress ramp into a
    V-shape, which inflates the uncertainty on both depth and mid-time.
    """
    dt = np.asarray(t, dtype=float) - mid
    half_total = duration / 2.0
    half_flat = max(half_total - ingress, 1e-9)

    f = np.ones_like(dt)
    f = np.where(np.abs(dt) <= half_flat, 1.0 - depth, f)
    ramp = (np.abs(dt) > half_flat) & (np.abs(dt) < half_total)
    frac = (half_total - np.abs(dt)) / ingress
    f = np.where(ramp, 1.0 - depth * np.clip(frac, 0, 1), f)
    cont = base + slope * dt + curve * dt ** 2
    if airmass is not None and k_ext:
        X = np.asarray(airmass, dtype=float)
        cont = cont * np.exp(-k_ext * (X - np.nanmedian(X)))
    return f * cont


def fit(bjd: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None = None,
        expected_mid: float | None = None,
        expected_duration_hours: float | None = None,
        expected_depth: float | None = None,
        fix_duration: bool = False,
        airmass: np.ndarray | None = None) -> TransitFit:
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
    # Ingress is bounded to 4-30% of the total duration. Geometrically it is
    # about (Rp/R*) x T14 for a non-grazing transit — a few percent to ~20%.
    # Leaving it free to reach 100% lets the model become a V and swallow
    # baseline curvature instead of measuring a transit.
    lo = [x.min(), 1e-5, dur0 * 0.3, dur0 * 0.04, 0.9, -5.0, -50.0]
    hi = [x.max(), 0.5,  dur0 * 3.0, dur0 * 0.30, 1.1,  5.0,  50.0]

    # With an airmass series available, fit a differential extinction
    # coefficient instead of leaning on the time polynomial. Typical residual
    # k for well colour-matched comparisons is a few hundredths of a
    # magnitude per airmass; the bounds are generous but not unbounded.
    # Airmass and the quadratic time term are degenerate over a short
    # session — airmass IS very nearly quadratic in time around culmination —
    # so fitting both lets them trade against each other and neither means
    # anything. Use the airmass term when we have it, since it is the actual
    # physics, and pin the quadratic to zero.
    # Airmass and the quadratic time term describe similar shapes over a
    # short session, so fitting both is degenerate. But which one is right is
    # a property of the night, not a rule: if the baseline trend really is
    # extinction, the airmass term explains it with one parameter; if it is
    # something else (focus drift, flat-field structure as the field moves),
    # only the polynomial can follow it. Fit both and keep whichever the data
    # prefers — decided below by BIC, which penalises the extra parameter.
    air_full = (np.asarray(airmass, dtype=float)[good]
                if airmass is not None else None)
    air = None
    use_air = False
    if fix_duration and expected_duration_hours:
        # Duration is well known from the archive and trades against depth
        # and mid-time in a noisy fit. Pinning it removes a degeneracy that
        # otherwise lets the model wander onto a trend instead of the transit.
        lo[2], hi[2] = dur0 * 0.999, dur0 * 1.001

    # Multi-start: the prior, plus a scan across the observed window.
    starts = [mid0] + list(np.linspace(x.min() + dur0 / 2,
                                       x.max() - dur0 / 2, 9))
    def _search(use_airmass: bool):
        nonlocal air, use_air
        use_air = use_airmass
        air = air_full if use_airmass else None
        _lo, _hi = list(lo), list(hi)
        if use_airmass:
            _lo[6], _hi[6] = -1e-9, 1e-9      # curvature off, airmass instead
            _lo, _hi = _lo + [-0.5], _hi + [0.5]
        _best, _cov, _chi2 = None, None, np.inf
        for m in starts:
            mm = float(np.clip(m, _lo[0] + 1e-6, _hi[0] - 1e-6))
            p0 = [mm, dep0, dur0, dur0 * 0.12, 1.0, 0.0, 0.0]
            if use_airmass:
                p0 = p0 + [0.0]
            p0 = [float(np.clip(v, a, b)) for v, a, b in zip(p0, _lo, _hi)]
            fn = ((lambda tt, *pp: trapezoid(tt, *pp, airmass=air))
                  if use_airmass else trapezoid)
            try:
                popt, pcov = curve_fit(fn, x, flux, p0=p0, bounds=(_lo, _hi),
                                       sigma=sigma,
                                       absolute_sigma=sigma is not None,
                                       loss="soft_l1", f_scale=0.01,
                                       maxfev=20000)
            except Exception:                           # noqa: BLE001
                continue
            resid = flux - fn(x, *popt)
            c2 = float(np.sum((resid / (sigma if sigma is not None else 1.0)) ** 2))
            if c2 < _chi2:
                _best, _cov, _chi2 = popt, pcov, c2
        return _best, _cov, _chi2, _lo, _hi

    n = len(x)
    cand = []
    b_poly = _search(False)
    if b_poly[0] is not None:
        rss = np.sum((flux - trapezoid(x, *b_poly[0])) ** 2)
        bic = n * np.log(rss / n) + len(b_poly[0]) * np.log(n)
        cand.append(("polynomial", b_poly, bic))
    if air_full is not None:
        b_air = _search(True)
        if b_air[0] is not None:
            air = air_full
            rss = np.sum((flux - trapezoid(x, *b_air[0], airmass=air_full)) ** 2)
            bic = n * np.log(rss / n) + len(b_air[0]) * np.log(n)
            cand.append(("airmass", b_air, bic))
    if not cand:
        raise RuntimeError("Transit fit did not converge — check the light curve.")
    label, (best, best_cov, best_chi2, lo, hi), _ = min(cand, key=lambda c: c[2])
    use_air = label == "airmass"
    air = air_full if use_air else None
    baseline_model = label

    # Uncertainties: the robust (soft_l1) loss finds parameter values that
    # outliers can't dictate, but its covariance is not a meaningful error
    # estimate. So re-fit with ordinary least squares, starting from the
    # robust solution and clipping the points it flagged as outliers, and
    # take the errors from that. This separates the two jobs: robustness for
    # the values, standard statistics for the error bars.
    model_fn = ((lambda tt, *pp: trapezoid(tt, *pp, airmass=air))
                if use_air else trapezoid)
    resid_r = flux - model_fn(x, *best)
    best_resid = resid_r
    s = 1.4826 * np.median(np.abs(resid_r - np.median(resid_r)))
    inl = np.abs(resid_r) < 4 * s if s > 0 else np.ones(len(x), bool)
    try:
        ls_fn = ((lambda tt, *pp: trapezoid(tt, *pp, airmass=air[inl]))
                 if use_air else trapezoid)
        _, cov_ls = curve_fit(
            ls_fn, x[inl], flux[inl], p0=best, bounds=(lo, hi),
            sigma=(sigma[inl] if sigma is not None else None),
            absolute_sigma=sigma is not None, maxfev=20000)
        perr = np.sqrt(np.diag(cov_ls))
    except Exception:                                   # noqa: BLE001
        perr = np.sqrt(np.diag(best_cov))

    span_days = float(x.max() - x.min())
    if not np.all(np.isfinite(perr)) or perr[0] > span_days:
        perr = _bootstrap_errors(x, flux, best, lo, hi, sigma)

    return TransitFit(
        mid_bjd=float(best[0]) + origin, mid_err_days=float(perr[0]),
        depth=float(best[1]), depth_err=float(perr[1]),
        duration_days=float(best[2]), ingress_days=float(best[3]),
        baseline_slope=float(best[5]), baseline_curve=float(best[6]),
        k_extinction=float(best[7]) if use_air else 0.0,
        baseline_model=baseline_model,
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
