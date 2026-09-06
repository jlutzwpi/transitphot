# transitphot

Local FITS-to-light-curve pipeline for exoplanet transit photometry.
Companion to [TransitPlanner](https://transits.justinlutz.com).

Your data stays on your machine — only the finished light curve is small
enough to share, and only if you choose to.

## Install

    pip install -e .

## Use

    # 1. Calibrate a night
    transitphot calibrate --lights ./lights --darks ./darks --flats ./flats

    # 2. Check whether your frames carry a WCS (ASIAIR subs usually do NOT)
    transitphot check --lights ./lights/calibrated

    # 3. Plate solve if needed — writes the WCS into each header
    transitphot solve --lights ./lights/calibrated --ra 343.0415 --dec 35.4471

    # 4. Extract a differential light curve
    transitphot run --lights ./lights/calibrated \
        --ra 343.0415 --dec 35.4471 --target-mag 12.4 \
        --out wasp10b.csv

Frames must be plate-solved so the pipeline can find your target and
comparison stars by coordinate. ASIAIR and similar controllers solve for
*pointing* but generally don't write a WCS into the saved subs — run
`transitphot check` to find out, and `transitphot solve` (which wraps
[ASTAP](https://www.hnsky.org/astap.htm)) to fix it. Install ASTAP and a
star database (D50 is sufficient for typical SCT fields) first.

## What it does that AstroImageJ makes you do by hand

* Picks comparison stars automatically — matched in brightness *and color*,
  screened for blends, variability, and high-RUWE binaries
* Drops comparison stars that misbehave during the session
* Uses ensemble photometry rather than a single reference star
* Sizes apertures from the measured FWHM of each frame

## Validation outputs

Every run writes three artifacts so you can check the pipeline instead of
trusting it:

* `*_finder.png` — the reference frame with a **green** circle on the target
  and **red** circles on each comparison star, apertures and sky annuli
  drawn, comparisons labeled with magnitude.
* `*_apertures.reg` — DS9 region file. Load it over the FITS in DS9
  (Region > Load Regions) or AstroImageJ (File > Import > Region file) to
  inspect placement at full resolution.
* `*_reference.fits` — the aligned frame the regions refer to, so the
  overlay and pixels always match.

Check these before submitting anything. Things they catch: an aperture
centered on a neighbor rather than the target, a sky annulus sitting on a
bright star, a comparison that's actually a close double.

## Status

Early. Calibration, alignment, comparison selection and differential
photometry are implemented; BJD_TDB conversion, transit model fitting and
ExoClock-format export are next.
