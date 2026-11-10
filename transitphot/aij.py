"""
Hand a transitphot reduction to AstroImageJ.

AIJ has no supported way to run Multi-Aperture or Multi-Plot unattended, so
this does not drive it. Instead it writes the two things that let one pass
through the AIJ GUI reproduce transitphot's measurement:

* ``<name>_aij.radec`` — the target and the comparison ensemble, in AIJ's own
  RA/Dec aperture-list format (Image window > File > "Import apertures from
  RA/Dec list...", or drag the file onto the image). The format is copied
  from AIJ's writer (astroj/IJU.java, "Export apertures to RA/Dec list"):
  seven ``#`` header lines, then ``RA (sexagesimal hours), Dec (sexagesimal
  degrees), ref star 0/1, centroid 0/1, magnitude`` per aperture, target
  first. AIJ's reader (IJU.openRaDecApertures) converts RA x15 and places each
  aperture through the open image's WCS, so the plate-solved frame must be
  open first.

* ``<name>_aij_settings.txt`` — every value the Multi-Aperture dialog and the
  Multi-Plot / transit-fit panels need, labelled with AIJ's own field names,
  plus transitphot's results to compare against. AIJ keeps those settings in
  its preferences file; writing that blind could break an install, so these
  are for typing in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

G_CGS = 6.674e-8


def _sex(value: float, places: int, signed: bool) -> str:
    """Sexagesimal like AIJ's decToSex: hh:mm:ss.sss / +dd:mm:ss.ss."""
    sign = "-" if value < 0 else ("+" if signed else "")
    v = abs(value)
    d = int(v)
    m = int((v - d) * 60)
    s = (v - d - m / 60) * 3600
    s = round(s, places)
    if s >= 60:                       # carry rounding, e.g. 59.9996 -> 60.000
        s -= 60
        m += 1
    if m >= 60:
        m -= 60
        d += 1
    width = 3 + places if places else 2
    return f"{sign}{d:02d}:{m:02d}:{s:0{width}.{places}f}"


def write_radec(path: Path, ra_deg: float, dec_deg: float,
                target_mag: float | None, comps) -> Path:
    """comps: objects with ra_deg, dec_deg, mag (the ensemble actually used)."""
    lines = [
        "#RA in decimal or sexagesimal HOURS",
        "#Dec in decimal or sexagesimal DEGREES",
        "#Ref Star=0,1,missing (0=target star, 1=ref star, missing->first ap=target, others=ref)",
        "#Centroid=0,1,missing (0=do not centroid, 1=centroid, missing=centroid)",
        "#Apparent Magnitude or missing (value = apparent magnitude, or value > 99 or missing = no mag info)",
        "#Add one comma separated line per aperture in the following format:",
        "#RA, Dec, Ref Star, Centroid, Magnitude",
    ]

    def row(ra, dec, ref, mag):
        m = f"{mag:.3f}" if mag is not None and np.isfinite(mag) else "99.999"
        return f"{_sex(ra / 15.0, 3, False)}, {_sex(dec, 2, True)}, {ref}, 1, {m}"

    lines.append(row(ra_deg, dec_deg, 0, target_mag))
    for c in comps:
        lines.append(row(c.ra_deg, c.dec_deg, 1, c.mag))
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _hm(bjd: float, offset: float) -> str:
    return f"{bjd:.5f}  (enter {bjd - offset:.5f} if the X offset is {offset:.0f})"


def write_settings(path: Path, *, target: str, lights: str, radec_name: str,
                   ra_deg: float, dec_deg: float, lat: float | None,
                   lon: float | None, elevation: float | None,
                   r_ap: float, r_in: float, r_out: float, comps, weights,
                   n_points: int, period: float | None,
                   duration_h: float | None, depth_ppm: float | None,
                   predicted_mid: float | None, flip_bjd: float | None,
                   bjd_span: tuple[float, float], airmass_used: bool,
                   trap=None, ld=None, u1: float = 0.35,
                   u2: float = 0.25) -> Path:
    off = float(np.floor(bjd_span[0]))
    L: list[str] = []
    add = L.append

    add(f"AstroImageJ settings for {target} — written by transitphot")
    add("=" * 72)
    add("Values are labelled with AIJ's own field names. Times are BJD_TDB.")
    add("Multi-Plot usually subtracts a whole-day X offset from the time axis;")
    add(f"if it shows the offset as {off:.0f}, enter the shorter values given.")
    add("")

    add("1. OPEN THE IMAGES")
    add(f"   File > Import > Image Sequence, folder: {lights}")
    add("   (tick 'Use virtual stack'). The frames are already calibrated and")
    add("   plate-solved by transitphot/ASTAP — skip the Data Processor.")
    add("")

    add("2. LOAD THE APERTURES")
    add(f"   Image window > File > Import apertures from RA/Dec list... > {radec_name}")
    add("   (or drag the file onto the image). The image must be open first:")
    add("   AIJ places each aperture through that frame's WCS.")
    add("   T1 = target; C2... = transitphot's comparison ensemble:")
    tot = float(np.sum(weights)) if len(weights) else 0.0
    for i, (c, w) in enumerate(zip(comps, weights)):
        pct = f"{100 * w / tot:.0f}%" if tot > 0 else "-"
        add(f"     C{i + 2}  G={c.mag:.2f}  RA {c.ra_deg:.5f}  Dec {c.dec_deg:+.5f}"
            f"   transitphot weight {pct}")
    add("   AIJ sums the comparisons unweighted, so expect small differences")
    add("   from transitphot's weighted ensemble.")
    add("")

    add("3. MULTI-APERTURE (Analyze > Multi-Aperture)")
    add(f"   Radius of object aperture ............ {r_ap:.1f}")
    add(f"   Inner radius of background annulus ... {r_in:.1f}")
    add(f"   Outer radius of background annulus ... {r_out:.1f}")
    add("   Use variable aperture ................ OFF (transitphot uses one")
    add("                                          fixed radius all night)")
    add("   Use RA/Dec to locate aperture positions  ON  (survives the flip)")
    add("   Centroid apertures ................... ON")
    add("   When it offers 'Place N previously used apertures', take it —")
    add("   those are the imported ones. Then run it through every frame.")
    add("")

    add("4. ADD BJD_TDB TO THE TABLE")
    add("   Multi-Plot only gets BJD_TDB from a header keyword these frames")
    add("   don't have. In Multi-Plot: Table > 'Add new astronomical data")
    add("   columns to table...', tick BJD_TDB (and AIRMASS if the table has no")
    add("   AIRMASS column), and in the MP Coordinate Converter enter:")
    add(f"     Target RA  {_sex(ra_deg / 15.0, 3, False)}  ({ra_deg / 15.0:.6f} h)")
    add(f"     Target Dec {_sex(dec_deg, 2, True)}  ({dec_deg:+.6f} deg)")
    if lat is not None and lon is not None:
        add(f"     Observatory lat {lat:+.4f}  lon {lon:+.4f} (east +)"
            + (f"  elevation {elevation:.0f} m" if elevation is not None else ""))
    add("   then press 'Update Table'.")
    add("")

    add("5. MULTI-PLOT MAIN PANEL")
    add("   Default X-data ....... BJD_TDB")
    add("   Y-data ............... rel_flux_T1 (normalized)")
    det = ["AIRMASS"] if airmass_used else []
    if flip_bjd is not None:
        det.append("Meridian_Flip")
    add("   Detrend Vars ......... " + (", ".join(det) if det else "none"))
    if predicted_mid is not None and duration_h:
        ing = predicted_mid - duration_h / 48.0
        egr = predicted_mid + duration_h / 48.0
        add(f"   V. Marker 1 (ingress)  {_hm(ing, off)}")
        add(f"   V. Marker 2 (egress)   {_hm(egr, off)}")
        add("   Fit and Normalize Region Selection:")
        add(f"     Left  .............. {_hm(ing, off)}")
        add(f"     Right .............. {_hm(egr, off)}")
        add("     (out-of-transit data either side of these normalizes the curve)")
    if flip_bjd is not None:
        add("   Meridian Flip > Flip Time")
        add(f"                          {_hm(flip_bjd, off)}")
        add("                          (from the frames' PIERSIDE headers)")
    else:
        add("   Meridian Flip ........ none — the frames show no flip;")
        add("                          leave Meridian_Flip out of the detrend list")
    add("")

    add("6. TRANSIT FIT PANEL (the fit window for rel_flux_T1)")
    add("   Enable Transit Fit ... ON")
    if period:
        add(f"   Orbital Parameters > Period (days) ... {period}   Cir ON")
    rho = None
    if ld is not None and period:
        p_s = period * 86400.0
        rho = 3 * np.pi * ld.a_rs ** 3 / (G_CGS * p_s ** 2)
    add("   Host Star Parameters (enter one):")
    if rho is not None:
        add(f"     rho* (cgs) ......... {rho:.3f}   (from transitphot's fitted a/R*)")
    else:
        add("     use the catalog Teff or R*/M* for this host")
    add("   Transit Parameters (prior centers; tick Lock where noted):")
    add("     Baseline Flux ...... 1.0")
    rp2 = (ld.rp_rs ** 2 if ld is not None
           else (depth_ppm / 1e6 if depth_ppm else None))
    if rp2 is not None:
        add(f"     (Rp/R*)^2 .......... {rp2:.5f}"
            + ("   (transitphot limb-darkened fit)" if ld is not None
               else "   (catalog depth)"))
    if ld is not None:
        add(f"     a/R* ............... {ld.a_rs:.2f}")
        add(f"     Inclination (deg) .. {ld.inclination_deg:.2f}")
    if predicted_mid is not None:
        add(f"     Tc ................. {_hm(predicted_mid, off)}")
        add("                          (ephemeris prediction — leave unlocked;")
        add("                          this is the value being measured)")
    add(f"     Linear LD u1 ....... {u1}   Lock ON")
    add(f"     Quad LD u2 ......... {u2}   Lock ON  (what transitphot assumes)")
    add("")

    add("7. TRANSITPHOT'S RESULT, TO COMPARE")
    add(f"   {n_points} points, aperture {r_ap:.1f} px, annulus {r_in:.1f}-{r_out:.1f} px,"
        f" {len(comps)} comparisons")
    if trap is not None:
        add(f"   trapezoid       Tc {trap.mid_bjd:.5f} +/- {trap.mid_err_minutes:.1f} min"
            f"   depth {trap.depth_ppm:.0f} ppm   RMS {trap.rms_ppm:.0f} ppm")
        if getattr(trap, "flip_step", 0.0):
            add(f"                   flip step {trap.flip_step * 1e6:+.0f} ppm")
    if ld is not None:
        add(f"   limb-darkened   Tc {ld.mid_bjd:.5f} +/- {ld.mid_err_minutes:.1f} min"
            f"   (Rp/R*)^2 {ld.rp_rs ** 2:.5f}   RMS {ld.rms_ppm:.0f} ppm")
        add(f"                   central dip {ld.central_depth_ppm:.0f} ppm"
            f"  (compare with AIJ's Depth)")
    if predicted_mid is not None:
        for name, r in (("trapezoid", trap), ("limb-darkened", ld)):
            if r is not None:
                add(f"   O-C ({name}) {(r.mid_bjd - predicted_mid) * 1440:+.2f} min")
    add("")
    add("8. A SECOND OPINION ON THE COMPARISONS")
    add("   Re-run Multi-Aperture with AIJ's own automatic comparison selection,")
    add("   same apertures. The spread between AIJ-with-these-stars and")
    add("   AIJ-with-its-own-stars is a direct measure of how much the")
    add("   comparison choice moves Tc on this night.")

    path = Path(path)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
