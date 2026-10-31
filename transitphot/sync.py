"""
Deferred file sync — copy a night's frames after the session, not during it.

Why deferred: an ASIAIR (or any small capture computer) runs its camera,
USB stack and file server on modest hardware. Sustained network reads of
50 MB frames while it is simultaneously writing new ones competes for the
same resources, and the symptoms when something gives — peripherals
dropping off the USB bus, the card going undetected — cost you the whole
session. Photometry is processed afterward regardless, so there is nothing
to gain from copying in real time.

Two ways to defer:

* **--start HH:MM** — begin at a wall-clock time you know the plan ends.
* **--after-idle N** — watch the source and begin once no file has been
  created or modified for N minutes. This detects "the plan finished"
  without needing to know anything about the capture software, and handles
  a plan that runs long or ends early.

Copies are conservative: a file must be size-stable before it is copied,
existing files are skipped, and each copy is verified by size afterwards.
"""

from __future__ import annotations

import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

from .photometry import FITS_SUFFIXES  # noqa: E402


def _fits_files(root: Path) -> list[Path]:
    out = []
    for suf in FITS_SUFFIXES:
        out.extend(root.rglob(f"*{suf}"))
        out.extend(root.rglob(f"*{suf.upper()}"))
    return sorted(set(out))


def _latest_mtime(files: list[Path]) -> float:
    best = 0.0
    for p in files:
        try:
            best = max(best, p.stat().st_mtime)
        except OSError:
            continue
    return best


def wait_until_clock(hhmm: str) -> None:
    """Sleep until the next occurrence of HH:MM local time."""
    hh, mm = (int(x) for x in hhmm.split(":"))
    now = datetime.now()
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    wait = (target - now).total_seconds()
    print(f"Waiting until {target:%Y-%m-%d %H:%M} "
          f"({wait / 60:.0f} min) before starting the copy…")
    while True:
        left = (target - datetime.now()).total_seconds()
        if left <= 0:
            return
        time.sleep(min(30, left))


def wait_until_idle(source: Path, minutes: float, poll_s: float = 30.0) -> None:
    """
    Block until nothing in `source` has changed for `minutes`.

    Polls gently — the point of this whole module is to stay off the capture
    device's back while it is working.
    """
    print(f"Waiting for {source} to be idle for {minutes:.0f} min "
          f"(checking every {poll_s:.0f}s)…")
    need = minutes * 60.0
    while True:
        files = _fits_files(source)
        last = _latest_mtime(files)
        idle = time.time() - last if last else need
        if idle >= need:
            print(f"Source idle for {idle / 60:.1f} min — starting copy "
                  f"({len(files)} files present).")
            return
        print(f"  {len(files)} files, last change {idle / 60:.1f} min ago")
        time.sleep(poll_s)


def _is_stable(path: Path, settle_s: float = 5.0) -> bool:
    """A file still being written is not safe to copy."""
    try:
        a = path.stat().st_size
        time.sleep(settle_s)
        b = path.stat().st_size
        return a == b and b > 0
    except OSError:
        return False


def copy_new(source: Path, dest: Path, throttle_s: float = 0.5,
             settle_s: float = 5.0, dry_run: bool = False) -> tuple[int, int]:
    """
    Copy FITS files that aren't already at the destination, preserving the
    directory layout below `source`.

    Returns (copied, skipped). A short pause between files keeps the copy
    from monopolising the source device even after the session has ended.

    Reports throughput as it goes, split into transfer time and throttle
    time: "slow" has two very different causes here (the source/network
    path's own speed, versus this function's own deliberate per-file
    pause) and a user watching the console has no way to tell them apart
    without the split - throttle_s=0.5 between a night's worth of ~35MB
    frames is a small fraction of total time, but on a slow link with many
    small files it would not be, and only the numbers can say which.
    """
    source, dest = Path(source), Path(dest)
    files = _fits_files(source)
    if not files:
        print(f"No FITS files found under {source}")
        return 0, 0

    copied = skipped = 0
    bytes_copied = 0
    transfer_s = throttled_s = 0.0
    for i, src in enumerate(files, 1):
        rel = src.relative_to(source)
        out = dest / rel
        try:
            if out.exists() and out.stat().st_size == src.stat().st_size:
                skipped += 1
                continue
        except OSError:
            pass

        if not _is_stable(src, settle_s):
            print(f"  skipping {rel} — still being written")
            skipped += 1
            continue

        if dry_run:
            print(f"  would copy {rel}")
            copied += 1
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        try:
            size = src.stat().st_size
            t0 = time.monotonic()
            shutil.copy2(src, tmp)
            dt = time.monotonic() - t0
            if tmp.stat().st_size != size:
                raise OSError("size mismatch after copy")
            tmp.replace(out)
            copied += 1
            bytes_copied += size
            transfer_s += dt
        except Exception as exc:                        # noqa: BLE001
            print(f"  FAILED {rel}: {exc}")
            tmp.unlink(missing_ok=True)
            skipped += 1

        if copied % 10 == 0 and copied:
            rate = bytes_copied / transfer_s / 1e6 if transfer_s else 0.0
            print(f"  copied {copied}/{len(files)} — "
                  f"{bytes_copied / 1e6:,.0f} MB at {rate:.1f} MB/s "
                  f"transferring ({throttled_s:.0f}s spent in the "
                  f"courtesy pause between files, not counted in that rate)")
        t0 = time.monotonic()
        time.sleep(throttle_s)
        throttled_s += time.monotonic() - t0

    if copied and not dry_run:
        rate = bytes_copied / transfer_s / 1e6 if transfer_s else 0.0
        total_s = transfer_s + throttled_s
        print(f"  done: {bytes_copied / 1e6:,.0f} MB in {total_s:.0f}s "
              f"({rate:.1f} MB/s while actually transferring, "
              f"{throttled_s:.0f}s / {100 * throttled_s / total_s:.0f}% "
              f"of that was the courtesy pause between files, not copy "
              f"time - pass --throttle 0 to skip it if the source isn't "
              f"actively imaging anymore)")

    return copied, skipped
