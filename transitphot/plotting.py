"""Light-curve plot: data, binned points, fitted model, residuals."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_lightcurve(bjd, flux, flux_err=None, fit_result=None,
                    title: str = "", out: Path = Path("lightcurve.png"),
                    bin_minutes: float = 5.0,
                    predicted_mid: float | None = None,
                    duration_days: float | None = None,
                    meridian_bjd: float | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .fitting import trapezoid

    bjd = np.asarray(bjd, float)
    flux = np.asarray(flux, float)
    t0 = np.floor(bjd.min())
    x = (bjd - t0) * 24.0                       # hours since start of day

    has_fit = fit_result is not None
    fig, axes = plt.subplots(
        2 if has_fit else 1, 1, figsize=(9, 6.5 if has_fit else 4.5),
        sharex=True, gridspec_kw={"height_ratios": [3, 1]} if has_fit else None,
    )
    ax = axes[0] if has_fit else axes

    ax.plot(x, flux, ".", ms=3, color="#9aa4b8", alpha=0.6, label="frames")

    # time-binned points — what the eye should actually judge
    nbins = max(int((x.max() - x.min()) * 60 / bin_minutes), 1)
    idx = np.digitize(x, np.linspace(x.min(), x.max(), nbins + 1)[1:-1])
    bx = np.array([x[idx == i].mean() for i in range(nbins) if (idx == i).any()])
    by = np.array([flux[idx == i].mean() for i in range(nbins) if (idx == i).any()])
    be = np.array([flux[idx == i].std() / max(np.sqrt((idx == i).sum()), 1)
                   for i in range(nbins) if (idx == i).any()])
    ax.errorbar(bx, by, yerr=be, fmt="o", ms=5, color="#1b2a4a",
                capsize=2, label=f"{bin_minutes:.0f}-min bins")

    # Predicted contacts and the meridian flip, as labelled verticals.
    def vline(ax, when, color, label, style="--"):
        if when is None:
            return
        xv = (when - t0) * 24.0
        if not (x.min() - 0.2 <= xv <= x.max() + 0.2):
            return
        ax.axvline(xv, ls=style, lw=1.1, color=color, alpha=0.75)
        ax.annotate(label, xy=(xv, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -12), textcoords="offset points",
                    rotation=90, va="top", ha="left",
                    fontsize=8.5, color=color)

    if predicted_mid is not None and duration_days:
        vline(ax, predicted_mid - duration_days / 2, "#2a6f97", "predicted ingress")
        vline(ax, predicted_mid + duration_days / 2, "#2a6f97", "predicted egress")
    vline(ax, meridian_bjd, "#b07d2b", "meridian flip", style=":")

    if has_fit:
        fine = np.linspace(bjd.min(), bjd.max(), 800)
        model = trapezoid(fine, fit_result.mid_bjd, fit_result.depth,
                          fit_result.duration_days, fit_result.ingress_days,
                          1.0, fit_result.baseline_slope,
                          fit_result.baseline_curve)
        ax.plot((fine - t0) * 24.0, model, "-", lw=1.8, color="#c1121f",
                label="trapezoid fit")
        ax.axvline((fit_result.mid_bjd - t0) * 24.0, ls="--", lw=1,
                   color="#c1121f", alpha=0.5)

    ax.set_ylabel("normalized flux")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(title or "Transit light curve")
    ax.grid(alpha=0.15)

    if has_fit:
        resid = flux - trapezoid(bjd, fit_result.mid_bjd, fit_result.depth,
                                 fit_result.duration_days,
                                 fit_result.ingress_days, 1.0,
                                 fit_result.baseline_slope,
                                 fit_result.baseline_curve)
        axes[1].plot(x, resid * 1e6, ".", ms=3, color="#9aa4b8", alpha=0.6)
        axes[1].axhline(0, color="#c1121f", lw=1)
        axes[1].set_ylabel("resid (ppm)")
        axes[1].grid(alpha=0.15)

    (axes[1] if has_fit else ax).set_xlabel(f"hours (BJD_TDB - {t0:.0f})")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    return out
