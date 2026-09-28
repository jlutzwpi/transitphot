import glob, sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                aperture_photometry)

RA, DEC = 359.15201, 36.21299
R_AP, R_IN, R_OUT = 7.0, 22.0, 33.0
folder = r"D:\xfer\Qatar-3b\lights\calibrated"

paths = sorted(glob.glob(folder + r"\*.fit*"))
print(f"{len(paths)} frames; measuring the first")
p = paths[0]
data = fits.getdata(p).astype(float)
hdr = fits.getheader(p)
x, y = WCS(hdr).all_world2pix(RA, DEC, 0)
print(f"{p.split(chr(92))[-1]}   target at pixel ({x:.1f}, {y:.1f})")

ap = CircularAperture((x, y), R_AP)
an = CircularAnnulus((x, y), R_IN, R_OUT)
vals = an.to_mask(method="center").multiply(data)
vals = vals[vals != 0]
_, sky, _ = sigma_clipped_stats(vals)
total = aperture_photometry(data, ap)["aperture_sum"][0]
print(f"sky per pixel {sky:.1f}")
print(f"target counts (sky-subtracted) {total - sky*ap.area:.0f}")