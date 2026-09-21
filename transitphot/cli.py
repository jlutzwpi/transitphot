"""
transitphot — command line entry point.

    transitphot calibrate --lights ./lights --darks ./darks --flats ./flats
    transitphot run --lights ./lights/calibrated --ra 343.041 --dec 35.447 \
                    --target-mag 10.2 --out curve.csv
    transitphot plan 1467 --token <api-token>     # pull a plan from the app
"""

from __future__ import annotations

import argparse
import sys
import csv
from pathlib import Path

import numpy as np


def cmd_calibrate(args):
    from .calibrate import calibrate_night
    written = calibrate_night(
        Path(args.lights),
        Path(args.bias) if args.bias else None,
        Path(args.darks) if args.darks else None,
        Path(args.flats) if args.flats else None,
        Path(args.out) if args.out else None,
    )
    print(f"Calibrated {len(written)} frames -> {written[0].parent if written else '-'}")


def cmd_run(args):
    if not getattr(args, "lights", None):
        raise SystemExit("--lights is required")
    import warnings
    from astropy.wcs import FITSFixedWarning
    # ASIAIR headers lack MJD-OBS; astropy derives it from DATE-OBS and says
    # so for every frame. Correct behavior, useless as 243 lines of output.
    warnings.simplefilter("ignore", FITSFixedWarning)
    from . import photometry as ph
    from . import compstars as cs
    from astropy.io import fits

    paths = sorted(Path(args.lights).glob("*.fit*"))
    if not paths:
        raise SystemExit(f"No FITS files in {args.lights}")
    # All products land in one folder per target, so a night's results stay
    # together instead of scattering across the working directory.
    stem = Path(args.out).stem
    folder = (Path(args.outdir) if args.outdir else Path(
        args.target_name or f"target_{args.ra:.4f}{args.dec:+.4f}")).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    args.out = str(folder / Path(args.out).name)
    if args.plot:
        args.plot = str(folder / Path(args.plot).name)
    print(f"Results -> {folder.resolve()}")

    paths, strays = ph.filter_session_frames(paths)
    if strays:
        print(f"Ignoring {len(strays)} frame(s) outside the main session "
              f"(e.g. {strays[0].name})")
    print(f"{len(paths)} frames")

    hdr0 = fits.getheader(paths[0])
    # Comparison stars: query the field, rank, then verify against the data
    print("Querying field for comparison stars…")
    radius = args.radius
    if radius is None:
        from astropy.io import fits as _fits
        radius = ph.field_radius_arcmin(_fits.getheader(paths[0]))
        print(f"Comparison search radius {radius:.1f}' (from the frame's WCS)")
    field = cs.query_field(args.ra, args.dec, radius_arcmin=radius)
    comps = cs.select(args.ra, args.dec, args.target_mag, field,
                      n=args.n_comps, filter_band=args.filter_band)
    if not comps:
        raise SystemExit("No suitable comparison stars found — widen --radius "
                         "or relax the magnitude tolerance.")
    print("Comparison stars selected:")
    for c in comps:
        print(f"  G={c.mag:.2f}  score {c.score}  ({'; '.join(c.reasons)})")

    # One aperture for the whole session — see photometry.session_fwhm.
    def _target_xy(hdr):
        try:
            return ph.sky_to_pixel(hdr, args.ra, args.dec)
        except Exception:                            # noqa: BLE001
            return None

    exposure_s = args.exposure if args.exposure else ph.session_exposure(paths)
    if exposure_s:
        print(f"Exposure {exposure_s:g}s per frame")
    else:
        print("WARNING: could not determine the exposure time from the FITS "
              "headers or filenames. Pass --exposure if you need it in the "
              "AAVSO report.")

    fwhm = args.fwhm if args.fwhm else ph.session_fwhm(paths, _target_xy,
                                                      report=True)
    if not args.fwhm and not (1.5 <= fwhm <= 12.0):
        print(f"WARNING: measured session FWHM {fwhm:.1f} px is outside the "
              f"usual 2-8 px range. Check the finder chart; override with "
              f"--fwhm <px> if it looks wrong.")

    # Try a ladder of apertures and let the data pick. For a faint,
    # background-limited target the best aperture is near 1x FWHM — every
    # extra pixel adds sky noise without adding much star. For a bright one
    # a wider aperture wins. Guessing a single multiple gets this wrong.
    if args.aperture_scale:
        scales = [args.aperture_scale]
    else:
        top = args.max_aperture_scale
        scales = [round(0.7 * (top / 0.7) ** (i / 6), 2) for i in range(7)]
    RADII = [sc * fwhm for sc in scales]
    R_IN, R_OUT = 3.5 * fwhm, 6.0 * fwhm
    print(f"Session FWHM {fwhm:.2f} px | trying apertures "
          + ", ".join(f"{r:.1f}" for r in RADII)
          + f" px | sky annulus {R_IN:.1f}-{R_OUT:.1f} px")

    # Comparison stars can be selected inside the search radius but still
    # fall off the sensor: the radius is a half-diagonal, while a star near
    # the short axis has far less room. Check once on the reference frame and
    # drop those, rather than failing every frame because one comparison is
    # permanently absent.
    from astropy.io import fits as _fits
    try:
        _hdr = _fits.getheader(paths[0])
        _data = _fits.getdata(paths[0])
        _h, _w = _data.shape
        keep_comps, off = [], []
        for c in comps:
            try:
                cx, cy = ph.sky_to_pixel(_hdr, c.ra_deg, c.dec_deg)
            except Exception:                            # noqa: BLE001
                off.append(c); continue
            margin = 60
            if margin <= cx <= _w - margin and margin <= cy <= _h - margin:
                keep_comps.append(c)
            else:
                off.append(c)
        if off:
            print(f"Dropping {len(off)} comparison star(s) that fall outside "
                  f"the frame ({', '.join(f'G={c.mag:.2f} at {c.sep_arcmin:.1f}' + chr(39) for c in off)})")
            comps = keep_comps
        if len(comps) < 2:
            raise SystemExit(
                "Fewer than two comparison stars fall inside the frame. "
                "Reduce --radius, or check that the field is as expected.")
    except SystemExit:
        raise
    except Exception as exc:                             # noqa: BLE001
        print(f"  (could not pre-check comparison positions: {exc})")

    times, tflux, cflux = [], [], []
    ref_snapshot = None          # (data, header, positions, aperture radii)
    n_rejected = n_done = 0
    for p, data, hdr in ph.iter_frames(paths):
        try:
            tx, ty = ph.sky_to_pixel(hdr, args.ra, args.dec)
            positions = [(tx, ty)] + [ph.sky_to_pixel(hdr, c.ra_deg, c.dec_deg)
                                      for c in comps]
        except Exception:
            raise SystemExit(
                "Frames have no usable WCS. ASIAIR solves for pointing but "
                "usually doesn't write a WCS into the subs.\n"
                "  transitphot check --lights <dir>     # confirm\n"
                f"  transitphot solve --lights <dir> --ra {args.ra} "
                f"--dec {args.dec}")

        # Refine each catalog position onto the actual star. A frame where
        # the target or a comparison isn't detectable has a bad WCS (stale
        # solutions around a meridian flip are the usual cause) or was lost
        # to cloud — measuring it would inject a wild flux ratio.
        refined, oks = [], []
        for xy in positions:
            rx, ry, ok = ph.refine_position(data, xy)
            refined.append((rx, ry))
            oks.append(ok)
        # The target must be there; a comparison that is momentarily lost
        # (cloud, a cosmic ray, drifting near an edge) costs that star for
        # this frame, not the whole frame.
        if not oks[0] or sum(oks[1:]) < 2:
            n_rejected += 1
            continue
        positions = refined

        if ref_snapshot is None:
            ref_snapshot = (data, hdr, positions, (RADII[len(RADII)//2], R_IN, R_OUT))
        flux = ph.measure(data, positions, r_ap=RADII, r_in=R_IN, r_out=R_OUT)
        times.append(ph.frame_time(hdr))
        tflux.append(flux[:, 0])           # (n_radii,)
        cflux.append(flux[:, 1:])          # (n_radii, n_comps)
        n_done += 1
        if n_done % 25 == 0:
            print(f"  measured {n_done}/{len(paths)} frames")

    if not times:
        raise SystemExit(
            f"No frames could be measured — the target was not detectable at "
            f"RA {args.ra}, Dec {args.dec} in any of the {len(paths)} frames.\n"
            f"  {n_rejected} frame(s) rejected at the centroid check.\n"
            f"The usual cause is coordinates that don't match the data: check "
            f"that --ra/--dec belong to the target in this folder, not a "
            f"previous run's target. Open the finder chart from an earlier "
            f"successful run to compare, or plate-solve one frame and inspect "
            f"where it points."
        )

    tflux = np.array(tflux)                          # (n_frames, n_radii)
    cflux = np.array(cflux)                          # (n_frames, n_radii, n_comps)

    # Pick the aperture that minimises median comparison-star scatter. The
    # comparisons are (presumed) constant stars, so whichever aperture makes
    # them most stable is the one measuring flux best — and choosing on the
    # comparisons rather than the target avoids biasing the transit itself.
    if len(RADII) > 1:
        print("Aperture scan (median comparison-star scatter):")
        best_i, best_med = 0, np.inf
        for i, r in enumerate(RADII):
            sc = cs.stability_report(cflux[:, i, :].T)
            med = float(np.nanmedian(sc))
            if med < best_med:
                best_i, best_med = i, med
            print(f"  {r:5.1f} px  {med:8.0f} ppm")
        # A minimum at the edge of the ladder is not a minimum — scatter may
        # still be falling beyond it. Say so, because the chosen aperture is
        # then a limit of the search rather than a property of the data.
        if best_i == len(RADII) - 1:
            print(f"  NOTE: the best aperture is the largest tried. Scatter "
                  f"may still be improving — re-run with "
                  f"--aperture-scale {RADII[-1] / fwhm * 1.3:.1f} or larger, "
                  f"or --max-aperture-scale to widen the scan.")
        elif best_i == 0:
            print("  NOTE: the best aperture is the smallest tried; the "
                  "optimum may be narrower still.")
        print(f"  -> using {RADII[best_i]:.1f} px")
    else:
        best_i = 0
    tflux = tflux[:, best_i]
    cflux = cflux[:, best_i, :].T                    # (n_comps, n_frames)

    scatter_ppm = cs.stability_report(cflux)
    keep = cs.check_stability(cflux)
    print("Comparison star stability (differential scatter):")
    for c, sc, k in zip(comps, scatter_ppm, keep):
        mark = "kept   " if k else "dropped"
        print(f"  {mark} G={c.mag:.2f}  {sc:7.0f} ppm")
    if keep.sum() == 0:
        raise SystemExit(
            "No usable comparison stars. This is rare — clouds alone should "
            "not cause it, since transparency is common-mode and divides "
            "out. Check the finder chart for aperture placement.")
    if n_rejected:
        print(f"Rejected {n_rejected} frame(s): target or a comparison star "
              f"was not detectable at its expected position "
              f"(stale WCS, cloud, or off-sensor)")
    # Weight by 1/scatter^2 rather than hard-dropping: a noisy comparison
    # still carries real transparency information, and shrinking the ensemble
    # raises its own shot noise. Only catastrophically bad stars (>3x the
    # best star's scatter) are vetoed outright.
    weights = ph.weights_from_scatter(scatter_ppm)
    finite = np.isfinite(scatter_ppm)
    if finite.any():
        best = float(np.nanmin(scatter_ppm[finite]))
        veto = finite & (scatter_ppm > 3 * best)
        weights[veto] = 0.0
        if veto.any():
            print(f"Vetoed {int(veto.sum())} comparison star(s) with scatter "
                  f">3x the best")
    tot = weights.sum()
    if tot > 0:
        print("Ensemble weights: " + ", ".join(
            f"G={c.mag:.2f} {100*w/tot:.0f}%" for c, w in zip(comps, weights)))
    # Drop cloud-hit frames before building the curve: they carry a fraction
    # of the photons and dominate the noise budget.
    tmask, transp = ph.transparency_mask(cflux, min_fraction=args.min_transparency)
    if (~tmask).any():
        print(f"Dropping {int((~tmask).sum())} frame(s) below "
              f"{args.min_transparency:.0%} transparency (cloud)")
        tflux, cflux = tflux[tmask], cflux[:, tmask]
        times = list(np.asarray(times, float)[tmask])

    norm, err = ph.differential_curve(tflux, cflux, weights=weights)

    times = np.array(times, dtype=float)
    if args.trim_start or args.trim_end:
        t0, t1 = times.min(), times.max()
        keep_t = np.ones(len(times), bool)
        if args.trim_start:
            keep_t &= times >= t0 + args.trim_start / 1440.0
        if args.trim_end:
            keep_t &= times <= t1 - args.trim_end / 1440.0
        print(f"Trimmed {int((~keep_t).sum())} point(s) from the session ends")
        times, norm, err = times[keep_t], norm[keep_t], err[keep_t]

    good = ph.clean_curve(times, norm)
    if good.sum() < len(good):
        print(f"Clipped {len(good) - good.sum()} outlier point(s) from the curve")
    times, norm, err = times[good], norm[good], err[good]

    # Normalize against out-of-transit points when the window is known
    if args.predicted_mid and args.duration_hours:
        norm = ph.normalize_out_of_transit(
            times, norm, args.predicted_mid, args.duration_hours / 24.0)

    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jd_utc", "rel_flux", "rel_flux_err"])
        for t, fl, e in zip(times, norm, err):
            w.writerow([f"{t:.6f}", f"{fl:.6f}", f"{e:.6f}"])
    rms = np.nanstd(norm) * 1e6
    print(f"Wrote {out} — {len(times)} points, scatter {rms:.0f} ppm")
    print("Compare that scatter against the precision TransitPlanner predicted; "
          "a large gap means the model needs calibrating for your setup.")

    # --- validation artifacts: see exactly what was measured ---
    if ref_snapshot is not None:
      # Everything below is diagnostic. The measurement is already written;
      # losing it because a PNG failed to save would be absurd, so failures
      # here warn and move on.
      try:
          from .annotate import (make_finder_chart, write_region_file,
                                 save_reference_frame)
          rdata, rhdr, rpos, (r_ap, r_in, r_out) = ref_snapshot
          labels = [f"C{i+1} G={c.mag:.1f}" for i, c in enumerate(comps)]
          base = Path(args.out).with_suffix("")
          chart = make_finder_chart(rdata, rpos[0], rpos[1:], r_ap, r_in, r_out,
                                    out=base.with_name(base.name + "_finder.png"),
                                    comp_labels=labels,
                                    title=f"Aperture placement — RA {args.ra} Dec {args.dec}")
          reg = write_region_file(base.with_name(base.name + "_apertures.reg"),
                                  rpos[0], rpos[1:], r_ap, r_in, r_out,
                                  comp_labels=[f"C{i+1}" for i in range(len(comps))])
          ref = save_reference_frame(rdata, rhdr,
                                     base.with_name(base.name + "_reference.fits"))
          print(f"Validation: {chart.name} (green=target, red=comparisons)")
          print(f"            {reg.name} + {ref.name} — load the .reg over the "
                f"FITS in DS9 or AstroImageJ to check placement at full resolution")
      except Exception as exc:                          # noqa: BLE001
        print(f"WARNING: could not write validation artifacts "
              f"({type(exc).__name__}: {exc}). "
              f"The light curve, fit and AAVSO report are unaffected.")

    # Predicted mid-time from the archive ephemeris, for whichever transit
    # this session actually covers.
    if args.epoch_bjd and args.period_days and not args.predicted_mid:
        centre = float(np.median(times))
        n = round((centre - args.epoch_bjd) / args.period_days)
        args.predicted_mid = args.epoch_bjd + n * args.period_days
        args.pred_system = "bjd_tdb"
        print(f"Predicted mid-transit from ephemeris (epoch + {n} x period): "
              f"{args.predicted_mid:.5f} BJD_TDB")

    # --- BJD_TDB ---
    if args.lat is not None and args.lon is not None:
        from .timing import jd_utc_to_bjd_tdb
        bjd = jd_utc_to_bjd_tdb(times, args.ra, args.dec,
                                args.lat, args.lon, args.elevation)
        shift = (bjd - times).mean() * 24 * 60
        print(f"Converted to BJD_TDB (barycentric shift {shift:+.2f} min)")
    else:
        bjd = times
        print("NOTE: times are JD(UTC). Pass --lat/--lon for BJD_TDB — "
              "required for ExoClock/AAVSO submission.")

    # --- fit ---
    if args.fit:
        from .fitting import fit as fit_transit, o_minus_c_minutes

        # Airmass detrending: every session so far has shown a U-shaped
        # baseline from differential extinction. Fitting it explicitly stops
        # the transit parameters from absorbing it.
        air = None
        if args.detrend_airmass and args.lat is not None and args.lon is not None:
            from .airmass import airmass_series, describe
            air = airmass_series(times, args.ra, args.dec,
                                 args.lat, args.lon, args.elevation)
            print(f"  detrending against {describe(air)}")
        elif args.detrend_airmass:
            print("  airmass detrending needs --lat/--lon; skipped")

        res = fit_transit(
            bjd, norm, err,
            fix_duration=args.fix_duration,
            airmass=air,
            expected_mid=args.predicted_mid,
            expected_duration_hours=args.duration_hours,
            expected_depth=(args.depth_ppm / 1e6) if args.depth_ppm else None,
        )
        print("")
        print(f"  mid-transit  {res.mid_bjd:.5f} BJD_TDB "
              f"(±{res.mid_err_minutes:.1f} min)")
        print(f"  depth        {res.depth_ppm:.0f} ± {res.depth_err*1e6:.0f} ppm")
        print(f"  duration     {res.duration_days*24:.2f} h")
        print(f"  residual RMS {res.rms_ppm:.0f} ppm")
        if air is not None:
            if res.baseline_model == "airmass":
                print(f"  baseline     airmass model, k = "
                      f"{res.k_extinction:+.4f} mag/airmass "
                      f"(residual colour mismatch with the comparisons)")
            else:
                print("  baseline     polynomial (the airmass model did not "
                      "improve the fit on this night)")
        ld_result = None
        if args.model in ("ld", "both") and args.period_days:
            from . import limbdark as _ld
            try:
                lf = _ld.fit(bjd, norm, err, period=args.period_days,
                             expected_mid=(args.predicted_mid
                                           or float(np.median(bjd))),
                             expected_depth=(args.depth_ppm / 1e6
                                             if args.depth_ppm else None),
                             expected_duration_hours=args.duration_hours)
                ld_result = lf
                impl = "batman" if _ld.available() else "built-in integrator"
                print(f"\n  limb-darkened fit ({impl}):")
                print(f"    mid-transit  {lf.mid_bjd:.5f} BJD_TDB "
                      f"(±{lf.mid_err_minutes:.1f} min)")
                print(f"    depth        {lf.depth_ppm:.0f} ± "
                      f"{lf.depth_err_ppm:.0f} ppm  [(Rp/Rs)^2, catalog "
                      f"convention]")
                print(f"    central dip  {lf.central_depth_ppm:.0f} ppm  "
                      f"[observed at mid-transit; compare with AstroImageJ]")
                print(f"    Rp/Rs {lf.rp_rs:.4f}  a/Rs {lf.a_rs:.2f}  "
                      f"i {lf.inclination_deg:.2f} deg")
                print(f"    duration     {lf.duration_hours:.2f} h")
                print(f"    residual RMS {lf.rms_ppm:.0f} ppm")
                if args.predicted_mid:
                    print(f"    O-C          "
                          f"{(lf.mid_bjd - args.predicted_mid) * 1440:+.2f} min")
                if args.model == "both":
                    # Which model to believe is a property of the data, not a
                    # rule. Say what the fits actually show and let the
                    # observer judge.
                    notes = []
                    if lf.mid_err_minutes > 3 * max(res.mid_err_minutes, 0.1):
                        notes.append("the limb-darkened mid-time is much less "
                                     "certain — its geometry is probably "
                                     "under-constrained by this series")
                    if args.depth_ppm:
                        exp_rp = (args.depth_ppm / 1e6) ** 0.5
                        if abs(lf.rp_rs - exp_rp) > 0.25 * exp_rp:
                            notes.append(f"fitted Rp/Rs {lf.rp_rs:.3f} is far "
                                         f"from the catalog {exp_rp:.3f} — "
                                         f"treat the limb-darkened result with "
                                         f"caution")
                    better = ("limb-darkened" if lf.rms_ppm < res.rms_ppm
                              else "trapezoid")
                    notes.append(f"lower residual RMS: {better} "
                                 f"({min(lf.rms_ppm, res.rms_ppm):.0f} vs "
                                 f"{max(lf.rms_ppm, res.rms_ppm):.0f} ppm)")
                    for n in notes:
                        print(f"    note: {n}")
            except Exception as exc:                    # noqa: BLE001
                print(f"  limb-darkened fit failed ({exc}); "
                      f"trapezoid values stand")
        elif args.model in ("ld", "both"):
            print("  limb-darkened fit needs --period; skipped")

        if args.predicted_mid:
            pred = args.predicted_mid
            if args.pred_system == "jd_utc":
                if args.lat is None or args.lon is None:
                    raise SystemExit(
                        "--predicted-mid-system jd_utc needs --lat/--lon to "
                        "convert the prediction to BJD_TDB.")
                from .timing import jd_utc_to_bjd_tdb
                pred = float(jd_utc_to_bjd_tdb(
                    np.array([pred]), args.ra, args.dec,
                    args.lat, args.lon, args.elevation)[0])
                print(f"  prediction converted JD(UTC) -> BJD_TDB "
                      f"({(pred - args.predicted_mid)*24*60:+.2f} min)")
            oc = o_minus_c_minutes(res.mid_bjd, pred)
            print(f"  O-C          {oc:+.2f} min "
                  f"({'late' if oc > 0 else 'early'}) [both BJD_TDB]")

        try:
            from .export import write_lightcurve, write_summary, write_aavso
            meta = {"target_ra_deg": args.ra, "target_dec_deg": args.dec,
                    "n_comparison_stars": int(keep.sum()),
                    "time_system": "BJD_TDB" if args.lat is not None else "JD_UTC"}
            base = Path(args.out).with_suffix("")
            write_lightcurve(base.with_suffix(".txt"), bjd, norm, err, meta)
            write_summary(base.with_name(base.name + "_summary.json"), res, meta,
                          args.predicted_mid)

            if args.aavso_obscode:
                planet = args.target_name or "unknown"
                star = args.star_name or planet.rstrip().removesuffix(" b")
                from .export import aavso_filter
                # An explicit --aavso-filter wins; otherwise map the imaging
                # filter to a ShortName and carry its description into NOTES,
                # which the format requires whenever the code is "O".
                if args.aavso_filter:
                    filt, filt_desc = args.aavso_filter.upper(), ""
                else:
                    filt, filt_desc = aavso_filter(args.filter_band)
                    if filt_desc:
                        print(f"  filter {args.filter_band} -> AAVSO '{filt}' "
                              f"({filt_desc})")
                priors = (f"period {args.period_days} d; "
                          f"duration {args.duration_hours} h fixed; "
                          f"depth prior {args.depth_ppm} ppm; "
                          f"quadratic limb darkening u1=0.35 u2=0.25")
                results = (f"Tc={res.mid_bjd:.5f} BJD_TDB "
                           f"+/-{res.mid_err_minutes:.1f} min; "
                           f"depth={res.depth_ppm:.0f}+/-{res.depth_err*1e6:.0f} "
                           f"ppm; duration={res.duration_days*24:.2f} h; "
                           f"residual RMS={res.rms_ppm:.0f} ppm")
                notes = ((f"Filter: {filt_desc}. " if filt_desc else "")
                         + f"{len(bjd)} points. Aperture {RADII[best_i]:.1f} px, "
                         f"sky annulus {R_IN:.1f}-{R_OUT:.1f} px, "
                         f"{int(keep.sum())} comparison stars. "
                         f"Reduced with transitphot (CMOS sensor reported as "
                         f"CCD per format).")
                ap = write_aavso(
                    base.with_name(base.name + "_aavso.txt"),
                    bjd, norm, err,
                    obscode=args.aavso_obscode,
                    star_name=star, exoplanet_name=planet,
                    exposure_s=exposure_s,
                    filter_code=filt, binning=args.binning,
                    ra=f"{args.ra:.6f}", dec=f"{args.dec:+.6f}",
                    priors=priors, results=results, notes=notes,
                    airmass=air)
                print(f"  wrote {ap.name} (AAVSO Exoplanet Database report)")
            print(f"  wrote {base.with_suffix('.txt').name} and "
                  f"{base.name}_summary.json")

            if args.plot:
                from .plotting import plot_lightcurve
                from .timing import meridian_crossing
                mer = None
                if args.lat is not None and args.lon is not None:
                    mer_jd = meridian_crossing(times, args.ra, args.lat, args.lon)
                    if mer_jd is not None:
                        # convert to the same axis the curve is plotted on
                        from .timing import jd_utc_to_bjd_tdb
                        mer = float(jd_utc_to_bjd_tdb(
                            np.array([mer_jd]), args.ra, args.dec,
                            args.lat, args.lon, args.elevation)[0])
                        print(f"  meridian crossing at {mer:.5f} BJD_TDB")
                plot_lightcurve(bjd, norm, err,
                                res if args.model != "ld" else None,
                                title=f"RA {args.ra} Dec {args.dec}",
                                out=Path(args.plot),
                                predicted_mid=(pred if args.predicted_mid else None),
                                duration_days=(args.duration_hours / 24.0
                                               if args.duration_hours else None),
                                meridian_bjd=mer,
                                ld_fit=ld_result, period=args.period_days)
                print(f"  wrote {args.plot}")
        except Exception as exc:                        # noqa: BLE001
            print(f"WARNING: could not write outputs or plot ({exc}).")
            print("The fitted values printed above are still valid.")

def cmd_sync(args):
    from .sync import copy_new, wait_until_clock, wait_until_idle, _fits_files
    src, dst = Path(args.source), Path(args.dest)
    print(f"Source: {src}")
    print(f"Destination: {dst}")
    if not src.exists():
        raise SystemExit(
            f"Source not reachable: {src}\n"
            f"If this is a network share, check the processing PC can open it "
            f"in Explorer — a share needing credentials will hang rather than "
            f"fail cleanly.")
    found = _fits_files(src)
    print(f"Found {len(found)} FITS file(s) under the source")
    if not found:
        raise SystemExit("Nothing to copy. Check the folder contains .fit/"
                         ".fits/.fts files (subfolders are searched too).")

    if args.start:
        wait_until_clock(args.start)
    if args.after_idle:
        wait_until_idle(src, args.after_idle, poll_s=args.poll)

    copied, skipped = copy_new(src, dst, throttle_s=args.throttle,
                               settle_s=args.settle, dry_run=args.dry_run)
    print(f"Done: {copied} copied, {skipped} skipped -> {dst}")
    if copied and not args.dry_run:
        print("Next: transitphot calibrate --lights <dir> ...")


def cmd_check(args):
    from .solve import inspect_dir
    reports = inspect_dir(Path(args.lights))
    if not reports:
        raise SystemExit(f"No FITS files in {args.lights}")
    ok = [r for r in reports if r.has_wcs]
    bad = [r for r in reports if not r.has_wcs]
    print(f"{len(reports)} frames: {len(ok)} with WCS, {len(bad)} without")
    if ok:
        r = ok[0]
        print(f"  example solved frame points at "
              f"RA {r.ra_deg:.4f}  Dec {r.dec_deg:+.4f}")
    if bad:
        notes = {}
        for r in bad:
            notes[r.note] = notes.get(r.note, 0) + 1
        for note, n in notes.items():
            print(f"  {n} frames: {note}")
        print("\nRun `transitphot solve --lights ... --ra ... --dec ...` "
              "before `transitphot run`.")
    else:
        print("All frames ready for `transitphot run`.")


def cmd_solve(args):
    from .solve import solve_dir
    solved, failed = solve_dir(
        Path(args.lights), args.ra, args.dec, args.radius_deg, args.fov)
    print(f"Done: {solved} solved, {failed} failed")
    if failed:
        print("Failed frames are usually clouded, trailed, or too few stars. "
              "Re-run to retry only the unsolved ones.")


def cmd_serve(args):
    try:
        from .server import serve
    except ImportError as exc:                          # noqa: BLE001
        raise SystemExit(
            f"The web server needs FastAPI and uvicorn ({exc}).\n"
            f'  pip install "transitphot[serve]"') from exc
    serve(host=args.host, port=args.port, use_token=not args.no_token,
          new_token=args.new_token)


def cmd_night(args):
    from .night import run_night
    run_night(args, cmd_sync=cmd_sync, cmd_calibrate=cmd_calibrate,
              cmd_check=cmd_check, cmd_solve=cmd_solve, cmd_run=cmd_run)


def main():
    # Windows defaults stdout to cp1252, which cannot encode the characters
    # used in the reports (Delta, multiplication sign, ellipsis). That is
    # survivable in a console but fatal when output is piped — as it is from
    # the GUI. Force UTF-8 and degrade gracefully if a character still can't
    # be represented, rather than crashing a 20-minute run over a label.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                               # noqa: BLE001
            pass

    p = argparse.ArgumentParser(prog="transitphot",
                                description="Local transit photometry pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="build masters and calibrate lights")
    c.add_argument("--lights", required=True)
    c.add_argument("--bias")
    c.add_argument("--darks")
    c.add_argument("--flats")
    c.add_argument("--out")
    c.set_defaults(func=cmd_calibrate)

    r = sub.add_parser("run", help="align, measure, and extract a light curve")
    r.add_argument("--lights", help="folder of calibrated, plate-solved frames")
    r.add_argument("--ra", type=float, required=True, help="target RA in degrees")
    r.add_argument("--dec", type=float, required=True, help="target Dec in degrees")
    r.add_argument("--target-mag", type=float, required=True)
    r.add_argument("--radius", type=float, default=None,
                   help="comparison search radius in arcmin (default: the "
                        "frame's half-diagonal, read from its WCS)")
    r.add_argument("--n-comps", type=int, default=5)
    r.add_argument("--filter", dest="filter_band",
                   help="filter used (R, L, V...) — relaxes the color match "
                        "criterion for narrower bands")
    r.add_argument("--out", default="lightcurve.csv",
                   help="output filename; written inside the results folder")
    r.add_argument("--target-name", dest="target_name",
                   help="target name — used as the results folder name")
    r.add_argument("--outdir", help="results folder (default: ./<target-name>)")
    r.add_argument("--lat", type=float, help="site latitude (enables BJD_TDB)")
    r.add_argument("--lon", type=float, help="site longitude, east positive")
    r.add_argument("--elevation", type=float, default=0.0)
    r.add_argument("--fit", action="store_true", help="fit the transit model")
    r.add_argument("--predicted-mid", type=float,
                   help="predicted mid-transit time, for O-C")
    r.add_argument("--epoch-bjd", type=float, dest="epoch_bjd",
                   help="reference transit epoch (BJD_TDB) from the archive; "
                        "with --period, the predicted mid-time for THIS night "
                        "is computed for you")
    r.add_argument("--period", type=float, dest="period_days",
                   help="orbital period in days, used with --epoch-bjd")
    r.add_argument("--predicted-mid-system", dest="pred_system",
                   choices=["bjd_tdb", "jd_utc"], default="bjd_tdb",
                   help="time system of --predicted-mid. AstroImageJ reports "
                        "geocentric JD(UTC); the NASA archive reports BJD_TDB. "
                        "Mixing them biases O-C by the barycentric correction, "
                        "up to 8 minutes.")
    r.add_argument("--duration-hours", type=float)
    r.add_argument("--depth-ppm", type=float)
    r.add_argument("--plot", help="write a PNG light curve to this path")
    r.add_argument("--aavso-obscode", dest="aavso_obscode",
                   help="your AAVSO observer code; supplying it writes an "
                        "AAVSO Exoplanet Database report file")
    r.add_argument("--aavso-filter", dest="aavso_filter",
                   help="AAVSO filter ShortName (e.g. R, V, CV for clear). "
                        "Defaults to --filter if it is a valid designation.")
    r.add_argument("--exposure", type=float,
                   help="exposure time in seconds; read from the FITS headers "
                        "when omitted")
    r.add_argument("--binning", default="1x1",
                   help="camera binning as AAVSO expects it (1x1, 2x2...)")
    r.add_argument("--star-name", dest="star_name",
                   help="host star name for the AAVSO report; defaults to the "
                        "target name with any trailing planet letter removed")
    r.add_argument("--fwhm", type=float,
                   help="override the measured session FWHM, in pixels")
    r.add_argument("--min-transparency", type=float, default=0.6,
                   dest="min_transparency",
                   help="drop frames whose summed comparison flux falls below "
                        "this fraction of the session median (cloud)")
    r.add_argument("--trim-start", type=float, default=0.0, dest="trim_start",
                   help="discard this many minutes from the start of the run")
    r.add_argument("--trim-end", type=float, default=0.0, dest="trim_end",
                   help="discard this many minutes from the end of the run")
    r.add_argument("--model", choices=["trapezoid", "ld", "both"],
                   default="trapezoid",
                   help="transit model: 'trapezoid' (robust, depth reads ~10%% "
                        "low), 'ld' (limb-darkened, correct depth), or 'both' "
                        "to compare them on the same data")
    r.add_argument("--no-detrend-airmass", action="store_false",
                   dest="detrend_airmass",
                   help="skip fitting a differential-extinction term against "
                        "airmass (on by default when --lat/--lon are given)")
    r.add_argument("--fix-duration", action="store_true", dest="fix_duration",
                   help="hold transit duration at --duration-hours; removes a "
                        "degeneracy that destabilizes fits on noisy data")
    r.add_argument("--max-aperture-scale", type=float, default=2.6,
                   dest="max_aperture_scale",
                   help="largest aperture to try, as a multiple of FWHM "
                        "(default 2.6; raise it if the scan reports the best "
                        "aperture is the largest tried)")
    r.add_argument("--aperture-scale", type=float, dest="aperture_scale",
                   help="fix the aperture at this multiple of FWHM instead "
                        "of scanning for the best")
    r.set_defaults(func=cmd_run, detrend_airmass=True)

    y = sub.add_parser("sync", help="copy frames from the capture device, "
                                    "deferred until the session is over")
    y.add_argument("--source", required=True,
                   help=r"capture device folder, e.g. \\ASIAIR\sdcard\Autorun")
    y.add_argument("--dest", required=True, help="local destination folder")
    y.add_argument("--start", help="wait until this local time first (HH:MM)")
    y.add_argument("--after-idle", type=float, dest="after_idle",
                   help="wait until no file has changed for this many minutes; "
                        "detects the end of a plan without needing to know it")
    y.add_argument("--poll", type=float, default=30.0,
                   help="seconds between idle checks (default 30)")
    y.add_argument("--throttle", type=float, default=0.5,
                   help="seconds between file copies (default 0.5)")
    y.add_argument("--settle", type=float, default=5.0,
                   help="seconds a file must be size-stable before copying")
    y.add_argument("--dry-run", action="store_true", dest="dry_run")
    y.set_defaults(func=cmd_sync)

    # "night" takes every argument "run" does, plus the sync and calibration
    # folders, so one command covers copy -> calibrate -> solve -> measure.
    ngt = sub.add_parser(
        "night", parents=[r], conflict_handler="resolve",
        help="the whole chain: wait, copy, calibrate, solve, measure")
    ngt.add_argument("--source", help="capture device folder to copy from; "
                                      "omit if the frames are already local")
    ngt.add_argument("--lights-root", required=True, dest="lights_root",
                     help="local folder holding (or to receive) the lights")
    ngt.add_argument("--bias")
    ngt.add_argument("--darks")
    ngt.add_argument("--flats")
    ngt.add_argument("--start", help="wait until this local time (HH:MM)")
    ngt.add_argument("--after-idle", type=float, dest="after_idle",
                     default=15.0,
                     help="start once the source has been unchanged this many "
                          "minutes (default 15)")
    ngt.add_argument("--poll", type=float, default=30.0)
    ngt.add_argument("--fov", type=float, help="field height in degrees, "
                                               "speeds plate solving")
    ngt.add_argument("--no-solve", action="store_true", dest="no_solve",
                     help="skip plate solving; unsolved frames are ignored")
    ngt.add_argument("--recalibrate", action="store_true",
                     help="recalibrate even if calibrated frames exist")
    ngt.set_defaults(func=cmd_night, detrend_airmass=True)

    sv = sub.add_parser("serve",
                        help="run a phone-friendly web page for this computer")
    sv.add_argument("--host", default="0.0.0.0",
                    help="interface to bind (default all, so a phone can reach it)")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--no-token", action="store_true", dest="no_token",
                    help="skip the URL token; only on a trusted network")
    sv.add_argument("--new-token", action="store_true", dest="new_token",
                    help="replace the saved token; earlier addresses stop working")
    sv.set_defaults(func=cmd_serve)

    k = sub.add_parser("check", help="report which frames have a usable WCS")
    k.add_argument("--lights", required=True)
    k.set_defaults(func=cmd_check)

    v = sub.add_parser("solve", help="plate solve frames with ASTAP (writes WCS)")
    v.add_argument("--lights", required=True)
    v.add_argument("--ra", type=float, help="approximate field RA in degrees")
    v.add_argument("--dec", type=float, help="approximate field Dec in degrees")
    v.add_argument("--radius-deg", type=float, default=10.0,
                   dest="radius_deg", help="search radius around --ra/--dec")
    v.add_argument("--fov", type=float, help="field height in degrees, speeds solving")
    v.set_defaults(func=cmd_solve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
