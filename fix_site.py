import glob, math, shutil, os
from astropy.io import fits

SRC = r"D:\xfer\Qatar-3b\lights\pipelineout"
DST = r"D:\xfer\Qatar-3b\lights\fixed"      # work on copies

os.makedirs(DST, exist_ok=True)
paths = [p for p in sorted(glob.glob(SRC + r"\*.fit*"))
         if "master" not in os.path.basename(p).lower()]
print(f"{len(paths)} frames")

for p in paths:
    out = os.path.join(DST, os.path.basename(p))
    shutil.copy2(p, out)
    with fits.open(out, mode="update") as hdul:
        h = hdul[0].header
        # the correct site is already here, under different keywords
        if "LAT-OBS" in h and "LONG-OBS" in h:
            h["SITELAT"] = float(h["LAT-OBS"])
            h["SITELONG"] = float(h["LONG-OBS"])
        # AIRMASS reads 1.00006 at 37 degrees altitude; recompute it
        if "OBJCTALT" in h:
            alt = float(h["OBJCTALT"])
            if alt > 0:
                z = 90.0 - alt
                h["AIRMASS"] = 1.0 / (math.cos(math.radians(z))
                                      + 0.50572 * (96.07995 - z) ** -1.6364)
        h.add_history("SITELAT/SITELONG set from LAT-OBS/LONG-OBS; "
                      "AIRMASS recomputed from OBJCTALT")
print(f"written to {DST}")