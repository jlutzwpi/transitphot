"""Light-curve plot: data, binned points, fitted model, residuals."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _tc_text(name, fit, predicted_mid) -> str | None:
    """'Tc (name) <BJD> ± err min, O−C ±x.xx min', so the plot alone answers
    "when was mid-transit?" without the log or summary."""
    tc = getattr(fit, "mid_bjd", None)
    if tc is None or not np.isfinite(tc):
        return None
    s = f"{name} Tc {tc:.5f}"
    err = getattr(fit, "mid_err_minutes", None)
    if err is not None and np.isfinite(err):
        s += f" ± {err:.1f} min"
    if predicted_mid is not None:
        s += f", O−C {(tc - predicted_mid) * 1440:+.2f} min"
    return s


def plot_lightcurve(bjd, flux, flux_err=None, fit_result=None,
                    title: str = "", out: Path = Path("lightcurve.png"),
                    bin_minutes: float = 5.0,
                    predicted_mid: float | None = None,
                    duration_days: float | None = None,
                    meridian_bjd: float | None = None,
                    ld_fit=None, period: float | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .fitting import trapezoid

    bjd = np.asarray(bjd, float)
    flux = np.asarray(flux, float)
    t0 = np.floor(bjd.min())
    x = (bjd - t0) * 24.0                       # hours since start of day

    has_fit = fit_result is not None
    show_resid = has_fit or ld_fit is not None
    fig, axes = plt.subplots(
        2 if show_resid else 1, 1, figsize=(9, 6.5 if show_resid else 4.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]} if show_resid else None,
    )
    ax = axes[0] if show_resid else axes

    # Remove the fitted meridian-flip offset from the DATA before anything is
    # drawn. The step is real — the field lands on different pixels after
    # the flip — but it is an instrumental offset, not something the planet
    # did. This used to happen after the points and bins were already
    # plotted, so the correction never reached the figure and the post-flip
    # half sat visibly below a model that had no step in it.
    corrected_note = None
    flux_raw = flux              # the LD model carries its own step
    if has_fit:
        step = getattr(fit_result, "flip_step", 0.0)
        fb = getattr(fit_result, "flip_bjd", None)
        if fb and step:
            flux = np.where(bjd > fb, flux / (1.0 + step), flux)
            corrected_note = (f"flip offset of {step*1e6:+.0f} ppm removed "
                              f"from the data")

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

    fine = np.linspace(bjd.min(), bjd.max(), 800)
    if has_fit:
        # The flip offset was already divided out of the data above.
        def _model(tt):
            # No step here: the data above has already been divided by it, so
            # the model must be continuous or the correction is applied twice.
            # The baseline is the fitted level, not 1.0 — the curve is
            # normalized to its own median, which is not where the fit put
            # the continuum.
            return trapezoid(tt, fit_result.mid_bjd, fit_result.depth,
                             fit_result.duration_days,
                             fit_result.ingress_days,
                             getattr(fit_result, "baseline_flux", 1.0),
                             fit_result.baseline_slope,
                             fit_result.baseline_curve)

        model = _model(fine)
        ax.plot((fine - t0) * 24.0, model, "-", lw=1.8, color="#c1121f",
                label="trapezoid fit")
        ax.axvline((fit_result.mid_bjd - t0) * 24.0, ls="--", lw=1,
                   color="#c1121f", alpha=0.5)

    # Overlay the limb-darkened model when one was fitted. Drawing both is
    # the point of running them together: the shapes differ most through
    # ingress and egress, which is exactly where the mid-time comes from.
    if ld_fit is not None and period:
        from .limbdark import model_curve
        _ld_step = getattr(ld_fit, "flip_step", 0.0)
        _ld_fb = getattr(ld_fit, "flip_bjd", None)
        ld_model = model_curve(fine, ld_fit, period,
                               base=getattr(ld_fit, "baseline_flux", 1.0))
        if _ld_fb and _ld_step:
            ld_model = np.where(np.asarray(fine) > _ld_fb,
                                ld_model / (1.0 + _ld_step), ld_model)
        ax.plot((fine - t0) * 24.0, ld_model, "-", lw=1.8, color="#2a6f97",
                label="limb-darkened fit")
        ax.axvline((ld_fit.mid_bjd - t0) * 24.0, ls="--", lw=1,
                   color="#2a6f97", alpha=0.5)

    ax.set_ylabel("normalized flux")
    ax.legend(frameon=False, fontsize=9)
    # Mid-transit times get their own line under the stats: they are what
    # timing observers ask for first, and the legend has no room for them.
    tc_lines = [(t, c, x, ha) for t, c, x, ha in (
        (_tc_text("trapezoid", fit_result, predicted_mid) if has_fit else None,
         "#c1121f", 0.0, "left"),
        (_tc_text("limb-darkened", ld_fit, predicted_mid)
         if ld_fit is not None else None, "#2a6f97", 1.0, "right"),
    ) if t]
    ax.set_title(title or "Transit light curve", pad=34 if tc_lines else 20)

    # A quick-glance summary under the title: whether there's a detection
    # at all (SNR), how deep (depth), and how big the planet is relative
    # to the star (Rp/Rs)^2 - the geometric quantity, not the trapezoid's
    # flux-ratio depth, so it only applies when the limb-darkened fit ran.
    stats = []
    if has_fit and fit_result.depth_err:
        snr = fit_result.depth / fit_result.depth_err
        stats.append(f"SNR {snr:.1f}σ")
        stats.append(f"depth {fit_result.depth_ppm:,.0f}±"
                     f"{fit_result.depth_err * 1e6:,.0f} ppm")
    if ld_fit is not None:
        # (Rp/Rs)^2 is exactly ld_fit.depth_ppm/1e6 (see limbdark.py's own
        # "catalog convention" depth), so its uncertainty is depth_err_ppm
        # scaled the same way - reusing it keeps this consistent with the
        # depth +/- already printed to the console for the same fit.
        stats.append(f"(Rp/Rs)² {ld_fit.rp_rs ** 2:.4f}±"
                     f"{ld_fit.depth_err_ppm / 1e6:.4f}")
    if corrected_note:
        stats.append(corrected_note)
    if stats:
        ax.text(0.5, 1.0 + (0.035 if tc_lines else 0.0), "   •   ".join(stats),
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color="#555", style="italic")
    # Colored like the fit curves: trapezoid at left, limb-darkened at right.
    for text, color, xpos, ha in tc_lines:
        ax.text(xpos, 1.0, text, transform=ax.transAxes, ha=ha, va="bottom",
                fontsize=8.5, color=color)

    ax.grid(alpha=0.15)

    if has_fit:
        resid = flux - _model(bjd)
        axes[1].plot(x, resid * 1e6, ".", ms=3, color="#c1121f", alpha=0.55,
                     label="trapezoid")
        if ld_fit is not None and period:
            from .limbdark import model_curve as _mc
            # model_curve includes the LD fit's own flip step, so compare it
            # with the uncorrected data, not the trapezoid-corrected flux.
            ld_resid = flux_raw - _mc(bjd, ld_fit, period,
                                  base=getattr(ld_fit, "baseline_flux", 1.0))
            axes[1].plot(x, ld_resid * 1e6, ".", ms=3, color="#2a6f97",
                         alpha=0.55, label="limb-darkened")
            axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
        axes[1].axhline(0, color="#333", lw=0.8)
        axes[1].set_ylabel("resid (ppm)")
        axes[1].grid(alpha=0.15)

    (axes[1] if show_resid else ax).set_xlabel(f"hours (BJD_TDB - {t0:.0f})")
    fig.tight_layout()
    from .annotate import _save_figure
    _save_figure(fig, Path(out))
    return out
