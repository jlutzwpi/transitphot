"""
transitphot — command line entry point.

    transitphot calibrate --lights ./lights --darks ./darks --flats ./flats
    transitphot run --lights ./lights/calibrated --ra 343.041 --dec 35.447 \
                    --target-mag 10.2 --out curve.csv
    transitphot plan 1467 --token <api-token>     # pull a plan from the app
"""

from __future__ import annotations

import argparse
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
    folder = Path(args.outdir) if args.outdir else Path(
        args.target_name or f"target_{args.ra:.4f}{args.dec:+.4f}")
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
    field = cs.query_field(args.ra, args.dec, radius_arcmin=args.radius)
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

    fwhm = args.fwhm if args.fwhm else ph.session_fwhm(paths, _target_xy)
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
        scales = [0.7, 0.9, 1.1, 1.3, 1.6, 2.0]
    RADII = [sc * fwhm for sc in scales]
    R_IN, R_OUT = 3.5 * fwhm, 6.0 * fwhm
    print(f"Session FWHM {fwhm:.2f} px | trying apertures "
          + ", ".join(f"{r:.1f}" for r in RADII)
          + f" px | sky annulus {R_IN:.1f}-{R_OUT:.1f} px")

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
        refined, all_ok = [], True
        for xy in positions:
            rx, ry, ok = ph.refine_position(data, xy)
            refined.append((rx, ry))
            all_ok &= ok
        if not all_ok:
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
            mark = ""
            if med < best_med:
                best_i, best_med, mark = i, med, ""
            print(f"  {r:5.1f} px  {med:8.0f} ppm")
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
        res = fit_transit(
            bjd, norm, err,
            fix_duration=args.fix_duration,
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

        from .export import write_lightcurve, write_summary
        meta = {"target_ra_deg": args.ra, "target_dec_deg": args.dec,
                "n_comparison_stars": int(keep.sum()),
                "time_system": "BJD_TDB" if args.lat is not None else "JD_UTC"}
        base = Path(args.out).with_suffix("")
        write_lightcurve(base.with_suffix(".txt"), bjd, norm, err, meta)
        write_summary(base.with_name(base.name + "_summary.json"), res, meta,
                      args.predicted_mid)
        print(f"  wrote {base.with_suffix('.txt').name} and "
              f"{base.name}_summary.json")

        if args.plot:
            from .plotting import plot_lightcurve
            plot_lightcurve(bjd, norm, err, res, title=f"RA {args.ra} Dec {args.dec}",
                            out=Path(args.plot))
            print(f"  wrote {args.plot}")


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


def main():
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
    r.add_argument("--lights", required=True)
    r.add_argument("--ra", type=float, required=True, help="target RA in degrees")
    r.add_argument("--dec", type=float, required=True, help="target Dec in degrees")
    r.add_argument("--target-mag", type=float, required=True)
    r.add_argument("--radius", type=float, default=20.0, help="field radius arcmin")
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
    r.add_argument("--fix-duration", action="store_true", dest="fix_duration",
                   help="hold transit duration at --duration-hours; removes a "
                        "degeneracy that destabilizes fits on noisy data")
    r.add_argument("--aperture-scale", type=float, dest="aperture_scale",
                   help="fix the aperture at this multiple of FWHM instead "
                        "of scanning for the best")
    r.set_defaults(func=cmd_run)

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
