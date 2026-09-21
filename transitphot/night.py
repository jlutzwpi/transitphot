"""
The whole night in one command.

Between finishing a capture run and having a measurement there are four
steps, each of which currently needs someone at the processing computer.
This chains them: wait for the capture device to go quiet, copy the frames,
calibrate, plate solve whatever needs it, and run the photometry.

Designed to be started once — from a phone, before bed — and left alone. So
it keeps going where it sensibly can: if solving fails for a handful of
frames, the run proceeds with the rest. It stops only where continuing would
produce a wrong answer rather than a noisy one.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _say(msg: str):
    print(msg, flush=True)


def _step(n: int, total: int, title: str):
    _say(f"\n[{n}/{total}] {title}")
    _say("-" * (len(title) + 8))


def run_night(args, *, cmd_sync, cmd_calibrate, cmd_check, cmd_solve, cmd_run):
    """
    Execute the chain. Takes the existing command functions so there is one
    implementation of each step, not two.
    """
    started = time.time()
    steps = 4 if args.source else 3
    n = 0

    # ---- 1. sync from the capture device -------------------------------
    if args.source:
        n += 1
        _step(n, steps, "Copy frames from the capture device")
        sync_args = argparse.Namespace(
            source=args.source, dest=args.lights_root,
            start=args.start, after_idle=args.after_idle,
            poll=args.poll, throttle=0.5, settle=5.0, dry_run=False,
        )
        cmd_sync(sync_args)

    lights = Path(args.lights_root)
    if not any(lights.rglob("*.fit*")):
        raise SystemExit(f"No FITS files under {lights} — nothing to process.")

    # ---- 2. calibrate ---------------------------------------------------
    n += 1
    _step(n, steps, "Calibrate")
    cal_dir = lights / "calibrated"
    if cal_dir.exists() and any(cal_dir.glob("*.fit*")) and not args.recalibrate:
        _say(f"Calibrated frames already present in {cal_dir}; keeping them "
             f"(pass --recalibrate to redo).")
    else:
        cal_args = argparse.Namespace(
            lights=str(lights), bias=args.bias, darks=args.darks,
            flats=args.flats, out=None,
        )
        cmd_calibrate(cal_args)

    work = cal_dir if any(cal_dir.glob("*.fit*")) else lights

    # ---- 3. plate solve what needs it -----------------------------------
    n += 1
    _step(n, steps, "Check WCS and plate solve")
    from .solve import inspect_dir
    reports = inspect_dir(work)
    missing = [r for r in reports if not r.has_wcs]
    _say(f"{len(reports)} frames, {len(reports) - len(missing)} already solved")
    if missing:
        if args.no_solve:
            _say(f"{len(missing)} frame(s) unsolved; --no-solve set, they will "
                 f"be skipped during photometry.")
        else:
            solve_args = argparse.Namespace(
                lights=str(work), ra=args.ra, dec=args.dec,
                radius_deg=10.0, fov=args.fov,
            )
            try:
                cmd_solve(solve_args)
            except SystemExit as exc:
                # A missing solver should not throw away the night: frames
                # that already have a WCS are still measurable.
                _say(f"Plate solving unavailable ({exc}). Continuing with the "
                     f"{len(reports) - len(missing)} frames that are solved.")

    # ---- 4. photometry ---------------------------------------------------
    n += 1
    _step(n, steps, "Photometry and fit")
    run_args = argparse.Namespace(**vars(args))
    run_args.lights = str(work)
    cmd_run(run_args)

    mins = (time.time() - started) / 60
    _say(f"\nFinished in {mins:.1f} min.")
