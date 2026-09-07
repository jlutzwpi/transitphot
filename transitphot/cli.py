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

    times, tflux, cflux = [], [], []
    ref_snapshot = None          # (data, header, positions, aperture radii)
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
        fwhm = ph.estimate_fwhm(data)
        r_ap, r_in, r_out = 1.5 * fwhm, 3 * fwhm, 5 * fwhm
        if ref_snapshot is None:
            ref_snapshot = (data, hdr, positions, (r_ap, r_in, r_out))
        flux = ph.measure(data, positions, r_ap=r_ap, r_in=r_in, r_out=r_out)
        times.append(ph.frame_time(hdr))
        tflux.append(flux[0])
        cflux.append(flux[1:])

    tflux = np.array(tflux)
    cflux = np.array(cflux).T                       # (n_comps, n_frames)

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
    norm, err = ph.differential_curve(tflux, cflux[keep])

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

    times = np.array(times)
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
            oc = o_minus_c_minutes(res.mid_bjd, args.predicted_mid)
            print(f"  O-C          {oc:+.2f} min "
                  f"({'late' if oc > 0 else 'early'})")

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
    r.add_argument("--out", default="lightcurve.csv")
    r.add_argument("--lat", type=float, help="site latitude (enables BJD_TDB)")
    r.add_argument("--lon", type=float, help="site longitude, east positive")
    r.add_argument("--elevation", type=float, default=0.0)
    r.add_argument("--fit", action="store_true", help="fit the transit model")
    r.add_argument("--predicted-mid", type=float,
                   help="predicted mid-transit BJD_TDB, for O-C")
    r.add_argument("--duration-hours", type=float)
    r.add_argument("--depth-ppm", type=float)
    r.add_argument("--plot", help="write a PNG light curve to this path")
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
