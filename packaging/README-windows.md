transitphot (TransitPlanner Processor) for Windows
==================================================

transitphot turns a night of FITS frames into an exoplanet transit light
curve: it calibrates the frames, plate solves them, picks comparison stars
from Gaia, measures them, fits the transit, and writes a report ready for
the AAVSO Exoplanet Database. Your frames never leave your computer.

You do not need to install Python.


GETTING STARTED
---------------

1. Unzip the whole folder somewhere permanent, for example
   C:\Program Files\transitphot or your Documents folder. Keep everything
   together: transitphot.exe needs the _internal folder beside it.

2. Install ASTAP, the free plate solver, from https://www.hnsky.org/astap.htm,
   together with one of its star databases (D50 suits most fields). Install
   it to the default location (C:\Program Files\astap). You can skip this
   only if your frames are already plate solved.

3. Double-click transitphot.exe.

   The first time, Windows may say "Windows protected your PC". The program
   is not code-signed; click "More info", then "Run anyway". Some antivirus
   programs also flag new unsigned programs; allow it if yours does.

4. In the window:
   - choose your lights folder (and bias, darks and flats if you have them)
   - enter the target, or press "Look up target in NASA archive"
   - press 1. Calibrate, then 2. Check WCS
   - if the check finds frames without a WCS, press 2b. Plate solve
   - press 3. Run photometry

   Progress appears in the window. The results land in a folder named for
   the target: the light curve plot, the data, a finder chart showing the
   comparison stars, and the AAVSO report (if you entered an AAVSO observer
   code). Your settings are remembered for next time.

An internet connection is needed while it runs: comparison stars come from
the Gaia catalog and the time conversion to BJD_TDB uses current
Earth-rotation data.


CHECKING THE RESULT
-------------------

- Open the finder chart: the target should be circled in green and the
  comparison stars in red, all on real stars.
- Look at the light curve plot: the dip should sit between the predicted
  ingress and egress lines.
- The results folder also contains lc_aij.radec and lc_aij_settings.txt for
  repeating the measurement in AstroImageJ, if you want a second opinion.


COMMAND LINE
------------

transitphot-cli.exe is the same program for the command line or scripts:

    transitphot-cli.exe --help
    transitphot-cli.exe run --help


QUESTIONS
---------

Source and issues: https://github.com/jlutzwpi/transitphot
