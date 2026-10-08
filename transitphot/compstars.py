"""
Comparison-star selection.

This is the step amateurs most often get wrong, and the one AstroImageJ
leaves entirely manual. Bad comparison stars are the difference between a
clean 3 mmag light curve and an unusable one.

Selection criteria, in the order they matter:

1. **Brightness** — within ~1.5 mag of the target. Much brighter risks
   saturation; much fainter adds its own shot noise to every measurement.
2. **Color** — similar BP-RP to the target. Differential atmospheric
   extinction is color-dependent, so a red comparison against a blue target
   introduces a slow trend across the night that mimics (or hides) a transit.
3. **Isolation** — no neighbor within the photometric aperture; blends
   corrupt the flux.
4. **Non-variability** — Gaia's variability flag plus a stability check
   against the actual frames.
5. **Proximity** — closer to the target means more similar airmass and
   optical path, but this matters far less than 1-3, so it's the tiebreak.

Ensemble photometry (summing several comparisons) beats a single star:
the combined reference has lower shot noise, and one misbehaving star is
diluted rather than fatal.
"""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table


@dataclass
class CompCandidate:
    ra_deg: float
    dec_deg: float
    mag: float
    color: float | None
    sep_arcmin: float
    score: float
    reasons: list[str]


CACHE_DIR = Path.home() / ".transitphot" / "gaia"


def _cache_path(ra_deg, dec_deg, radius_arcmin, mag_limit) -> Path:
    # n/s for the declination sign, so the filename reads the way a human
    # expects; "p" stands in for the decimal point only.
    sign = "n" if dec_deg >= 0 else "s"
    key = (f"ra{ra_deg:.4f}_dec{sign}{abs(dec_deg):.4f}"
           f"_r{radius_arcmin:.1f}_g{mag_limit:.1f}")
    return CACHE_DIR / (key.replace(".", "p") + ".ecsv")


def _with_timeout(fn, seconds: float):
    """
    Run fn() and give up after `seconds`.

    A retry loop is useless against a call that never returns, and
    astroquery's async TAP job has no timeout of its own: when ESA's archive
    accepted the job and then stopped answering, this blocked for hours in
    the middle of an unattended run.

    The worker is a daemon thread, so an abandoned query cannot keep the
    process alive. Returns (value, None) or (None, error).
    """
    import threading

    box = {}

    def work():
        try:
            box["value"] = fn()
        except BaseException as exc:                     # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        return None, TimeoutError(f"no answer after {seconds:.0f}s")
    return box.get("value"), box.get("error")



def _query_vizier_gaia(ra_deg, dec_deg, radius_arcmin, mag_limit):
    """
    The same Gaia DR3 catalog, served by CDS instead of ESA.

    Different organization, different infrastructure: when ESA's archive is
    down — as it was, with a 500, in the middle of an unattended night — CDS
    is usually still answering. VizieR names the columns differently, so map
    them onto what the rest of this module expects.

    Returns None rather than raising, so the caller can fall through.
    """
    try:
        from astroquery.vizier import Vizier
        v = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS", "Gmag", "BP-RP",
                            "RUWE", "VarFlag"],
                   column_filters={"Gmag": f"<{mag_limit}"},
                   row_limit=-1, timeout=120)
        res = v.query_region(SkyCoord(ra_deg * u.deg, dec_deg * u.deg),
                             radius=radius_arcmin * u.arcmin,
                             catalog="I/355/gaiadr3")
    except Exception:                                    # noqa: BLE001
        return None
    if not res or len(res[0]) == 0:
        return None

    t = res[0]
    out = Table()
    try:
        out["source_id"] = t["Source"]
        out["ra"] = np.asarray(t["RA_ICRS"], dtype=float)
        out["dec"] = np.asarray(t["DE_ICRS"], dtype=float)
        out["phot_g_mean_mag"] = np.asarray(t["Gmag"], dtype=float)
        out["bp_rp"] = (np.asarray(t["BP-RP"], dtype=float)
                        if "BP-RP" in t.colnames
                        else np.full(len(t), np.nan))
        out["ruwe"] = (np.asarray(t["RUWE"], dtype=float)
                       if "RUWE" in t.colnames else np.zeros(len(t)))
        # VizieR reports variability as a flag column when it reports it at
        # all; anything we cannot read becomes NOT_AVAILABLE, which is how
        # Gaia itself marks "not assessed".
        if "VarFlag" in t.colnames:
            out["phot_variable_flag"] = [str(x) for x in t["VarFlag"]]
        else:
            out["phot_variable_flag"] = ["NOT_AVAILABLE"] * len(t)
    except Exception:                                    # noqa: BLE001
        return None
    return out



def query_field(ra_deg: float, dec_deg: float, radius_arcmin: float = 20.0,
                mag_limit: float = 16.0, attempts: int = 2,
                use_cache: bool = True, timeout_s: float = 90.0) -> Table:
    """
    Fetch Gaia sources in the field.

    Cached on disk and retried, because this is the one step that depends on
    somebody else's server staying up. The Gaia archive is explicitly warning
    of instability ahead of DR4, and a 500 here otherwise throws away a whole
    night's unattended processing at the final step — the frames are fine,
    but the run dies and needs a human.

    The field for a given target does not change, so a cached answer is as
    good as a fresh one and removes the dependency entirely on a re-run.
    """
    import time as _time

    cache = _cache_path(ra_deg, dec_deg, radius_arcmin, mag_limit)
    if use_cache and cache.exists():
        try:
            t = Table.read(cache, format="ascii.ecsv")
            print(f"Comparison catalog from cache ({len(t)} sources; "
                  f"delete {cache} to refresh)")
            return t
        except Exception:                                # noqa: BLE001
            pass

    from astroquery.gaia import Gaia

    q = f"""
    SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp,
           phot_variable_flag, ruwe
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_arcmin / 60.0})) = 1
      AND phot_g_mean_mag < {mag_limit}
    """

    last = None
    for i in range(attempts):
        table, last = _with_timeout(
            lambda: Gaia.launch_job_async(q).get_results(), timeout_s)
        if table is not None and last is None:
            break
        if i < attempts - 1:
            wait = 15 * (i + 1)
            print(f"  Gaia query failed ({last}); retrying in {wait}s "
                  f"({attempts - i - 1} left)")
            _time.sleep(wait)
        table = None
    else:
        print(f"  Gaia archive unavailable ({last}); trying VizieR for the "
              f"same catalog")
        table, verr = _with_timeout(
            lambda: _query_vizier_gaia(ra_deg, dec_deg, radius_arcmin,
                                       mag_limit), timeout_s)
        if verr is not None:
            print(f"  VizieR also failed ({verr})")
            table = None
        if table is None:
            raise SystemExit(
                f"Neither the Gaia archive nor VizieR answered ({last}).\n"
                f"The frames are fine — re-run this step when one of them is "
                f"back.")
        print(f"  VizieR returned {len(table)} sources")

    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            table.write(cache, format="ascii.ecsv", overwrite=True)
        except Exception:                                # noqa: BLE001
            pass
    return table


def query_vsx(ra_deg: float, dec_deg: float, radius_arcmin: float = 20.0):
    """
    Known variable stars in the field, from AAVSO's VSX via VizieR.

    Gaia's phot_variable_flag is sparse: it reads VARIABLE only for sources
    Gaia's own variability pipeline classified, and NOT_AVAILABLE — "not
    assessed", not "constant" — for almost everything else. VSX aggregates
    ASAS-SN, ZTF, Kepler, TESS and decades of other surveys, so it catches
    far more of the variables that would otherwise pass as comparisons.

    Returns (SkyCoord, names, types), or None if VizieR can't be reached —
    in which case the caller carries on with Gaia's flag alone.
    """
    try:
        from astroquery.vizier import Vizier
        v = Vizier(columns=["Name", "Type", "RAJ2000", "DEJ2000"],
                   row_limit=-1, timeout=60)
        res = v.query_region(SkyCoord(ra_deg * u.deg, dec_deg * u.deg),
                             radius=radius_arcmin * u.arcmin,
                             catalog="B/vsx/vsx")
    except Exception:                                    # noqa: BLE001
        return None
    if not res or len(res[0]) == 0:
        return SkyCoord([], [], unit="deg"), [], []
    t = res[0]
    try:
        coords = SkyCoord(t["RAJ2000"], t["DEJ2000"], unit=(u.deg, u.deg))
    except Exception:                                    # noqa: BLE001
        coords = SkyCoord(t["RAJ2000"], t["DEJ2000"],
                          unit=(u.hourangle, u.deg))
    names = [str(x) for x in t["Name"]]
    types = [str(x) for x in t["Type"]] if "Type" in t.colnames else [""] * len(t)
    return coords, names, types


def target_color_from(sources: Table, ra_deg: float, dec_deg: float,
                      max_sep_arcsec: float = 3.0) -> float | None:
    """
    The target's own BP-RP color, taken from the same Gaia query.

    Color matching needs the target's color to compare against. Without it
    the check silently does nothing, which is exactly what happened: the
    selection accepted a target_color but was never given one.
    """
    if len(sources) == 0:
        return None
    tgt = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    coords = SkyCoord(sources["ra"], sources["dec"], unit="deg")
    sep = tgt.separation(coords).arcsec
    i = int(np.argmin(sep))
    if sep[i] > max_sep_arcsec:
        return None
    c = sources["bp_rp"][i]
    if c is np.ma.masked or c is None:
        return None
    try:
        return float(c)
    except (TypeError, ValueError):
        return None


def color_tolerance_for(filter_band: str | None) -> float:
    """
    How closely a comparison star's color must match the target.

    Differential atmospheric extinction is wavelength-dependent, so a color
    mismatch produces a slow trend across the night as airmass changes. A
    narrower, redder passband reduces (but does not remove) the effect:
    two stars of different spectral type still have different effective
    wavelengths within the same filter.
    """
    if not filter_band:
        return 0.5
    b = filter_band.strip().upper().rstrip("C")
    if b in {"L", "LUM", "CLEAR", "C", "NONE"}:
        return 0.4          # widest band, strongest color dependence
    if b in {"B", "G", "V"}:
        return 0.5
    if b in {"R"}:
        return 0.8          # narrower and redder: more forgiving
    if b in {"I", "IR", "SLOAN I", "IZ"}:
        return 1.0
    return 0.5


def select(target_ra: float, target_dec: float, target_mag: float,
           sources: Table, n: int = 5,
           mag_tolerance: float = 1.5,
           color_tolerance: float | None = None,
           min_separation_arcsec: float = 15.0,
           target_color: float | None = None,
           filter_band: str | None = None,
           variables=None,
           rejected: list | None = None,
           vsx_match_arcsec: float = 5.0) -> list[CompCandidate]:
    """
    Rank field stars as comparison candidates. Returns the best `n`.

    The score is deliberately interpretable — each candidate carries the
    reasons it ranked where it did, so a user can sanity-check the choice
    instead of trusting a black box.
    """
    if color_tolerance is None:
        color_tolerance = color_tolerance_for(filter_band)

    tgt = SkyCoord(target_ra * u.deg, target_dec * u.deg)
    coords = SkyCoord(sources["ra"], sources["dec"], unit="deg")
    seps = tgt.separation(coords).arcmin

    # Known variables from VSX: index each Gaia source to its nearest entry.
    vsx_hit = [None] * len(sources)
    if variables is not None and len(variables[0]) > 0:
        vcoords, vnames, vtypes = variables
        vi, vsep, _ = coords.match_to_catalog_sky(vcoords)
        for i in range(len(sources)):
            if vsep[i].arcsec <= vsx_match_arcsec:
                vsx_hit[i] = (vnames[vi[i]], vtypes[vi[i]])

    # Pairwise separations, for isolation testing
    idx, sep2d, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)
    nearest_arcsec = sep2d.arcsec

    out: list[CompCandidate] = []
    for i, row in enumerate(sources):
        mag = float(row["phot_g_mean_mag"])
        color = float(row["bp_rp"]) if row["bp_rp"] is not np.ma.masked else None
        sep = float(seps[i])

        if sep < 0.05:                       # this is the target itself
            continue
        dmag = abs(mag - target_mag)
        if dmag > mag_tolerance:
            continue
        if nearest_arcsec[i] < min_separation_arcsec:
            continue                          # blended with a neighbor
        if str(row["phot_variable_flag"]).upper().startswith("VARIABLE"):
            if rejected is not None:
                rejected.append((mag, sep, "flagged variable by Gaia"))
            continue
        if vsx_hit[i] is not None:
            if rejected is not None:
                name, vtype = vsx_hit[i]
                rejected.append((mag, sep, f"known variable {name}"
                                 + (f" ({vtype})" if vtype else "")
                                 + " in VSX"))
            continue

        reasons = []
        score = 100.0

        score -= (dmag / mag_tolerance) * 30
        reasons.append(f"Δmag {dmag:+.2f}")

        if target_color is not None and color is not None:
            dcol = abs(color - target_color)
            score -= min(dcol / color_tolerance, 2.0) * 25
            reasons.append(f"ΔBP-RP {dcol:.2f}")
            if dcol > color_tolerance:
                reasons.append("color mismatch — extinction trend risk")

        score -= min(sep / 20.0, 1.0) * 10
        reasons.append(f"{sep:.1f}' away")

        if float(row["ruwe"] or 0) > 1.4:
            score -= 15
            reasons.append("high RUWE — possible unresolved binary")

        out.append(CompCandidate(float(row["ra"]), float(row["dec"]), mag,
                                 color, sep, round(score, 1), reasons))

    out.sort(key=lambda c: -c.score)
    return out[:n]


def check_stability(fluxes: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """
    Post-hoc check against the actual data, measured DIFFERENTIALLY.

    Critical subtlety: clouds, transparency changes and airmass dim every
    star in the field together. Differential photometry divides that out, so
    common-mode variation is exactly what we must NOT penalize. An earlier
    version of this function tested each star's raw flux scatter and rejected
    every comparison on a partly cloudy night — discarding good data for the
    one reason that doesn't matter.

    So: normalize each star against the ensemble of the *others*, then test
    the residual scatter. That isolates variation intrinsic to the star (real
    variability, drift onto a bad column, a satellite trail) from variation
    shared by the whole field.

    fluxes: (n_stars, n_frames)
    Returns a boolean mask of stars to keep.
    """
    fluxes = np.asarray(fluxes, dtype=float)
    n_stars = fluxes.shape[0]
    if n_stars == 1:
        return np.array([True])

    scatters = np.full(n_stars, np.nan)
    for i in range(n_stars):
        others = np.delete(fluxes, i, axis=0)
        ensemble = np.nansum(others, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = fluxes[i] / ensemble
        med = np.nanmedian(ratio)
        if not np.isfinite(med) or med == 0:
            continue
        norm = ratio / med
        # robust scatter: MAD isn't thrown off by a few bad frames
        scatters[i] = 1.4826 * np.nanmedian(np.abs(norm - 1.0))

    finite = np.isfinite(scatters)
    keep = finite & (scatters < threshold)
    if keep.sum() == 0 and finite.any():
        # Everything looked marginal (a genuinely rough night). Keep the best
        # half rather than failing outright — a noisy curve beats no curve,
        # and the reported scatter tells the user what they got.
        order = np.argsort(np.where(finite, scatters, np.inf))
        keep[order[:max(n_stars // 2, 1)]] = True
    return keep


def stability_report(fluxes: np.ndarray) -> np.ndarray:
    """Differential scatter per star, in ppm — for reporting to the user."""
    fluxes = np.asarray(fluxes, dtype=float)
    out = np.full(fluxes.shape[0], np.nan)
    for i in range(fluxes.shape[0]):
        others = np.delete(fluxes, i, axis=0)
        if others.size == 0:
            continue
        ensemble = np.nansum(others, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = fluxes[i] / ensemble
        med = np.nanmedian(ratio)
        if np.isfinite(med) and med != 0:
            norm = ratio / med
            out[i] = 1.4826 * np.nanmedian(np.abs(norm - 1.0)) * 1e6
    return out


def choose_ensemble(target_flux, comp_flux, out_of_transit,
                    max_n: int = 8, min_n: int = 5,
                    say=print) -> list:
    """
    Pick the SET of comparison stars that yields the cleanest light curve.

    Choosing the stars that look steadiest individually is not the same as
    choosing the combination that measures best. Comparison stars share
    errors — colour-dependent extinction, flat-field structure, a gradient
    across the frame — so a star that is mediocre alone can cancel another's
    systematics, and two excellent stars can share the same one.

    So judge each candidate set by what we actually care about: the scatter
    of the target's differential light curve outside transit. Greedy forward
    selection, adding whichever star most improves that, stopping when none
    does. With fifty candidates this costs a few hundred evaluations where
    trying every subset would need millions.

    Returns the chosen indices into comp_flux.
    """
    target_flux = np.asarray(target_flux, dtype=float)
    comp_flux = np.asarray(comp_flux, dtype=float)
    oot = np.asarray(out_of_transit, dtype=bool)
    n_comp = comp_flux.shape[0]
    min_n = min(min_n, n_comp)
    max_n = max(max_n, min_n)
    if n_comp <= min_n or oot.sum() < 10:
        return list(range(n_comp))

    def score(idx):
        if not idx:
            return np.inf
        ens = np.nansum(comp_flux[list(idx), :], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = target_flux / ens
        r = rel[oot]
        r = r[np.isfinite(r)]
        if r.size < 10:
            return np.inf
        m = np.median(r)
        if not np.isfinite(m) or m <= 0:
            return np.inf
        return 1.4826 * np.median(np.abs(r - m)) / m * 1e6

    chosen: list = []
    best = np.inf
    while len(chosen) < min(max_n, n_comp):
        gains = [(score(chosen + [j]), j)
                 for j in range(n_comp) if j not in chosen]
        gains.sort()
        if not gains or not np.isfinite(gains[0][0]):
            break
        s, j = gains[0]

        # Below the floor, keep adding regardless. The scatter of ~60
        # out-of-transit points is itself uncertain by roughly 1/sqrt(2N),
        # about 9%, so a search over many subsets WILL find combinations that
        # fit that noise. A small ensemble also concentrates the weight: a
        # three-star set where one star carries two thirds of it has nothing
        # to balance that star if it misbehaves.
        if len(chosen) < min_n:
            chosen.append(j)
            best = s
            continue

        # Above the floor, only take an extra star if it genuinely helps.
        if s < best * (1.0 - 0.02):
            chosen.append(j)
            best = s
        else:
            break

    say(f"Ensemble search: {len(chosen)} of {n_comp} candidates give the "
        f"cleanest curve ({best:.0f} ppm out of transit)")
    return sorted(chosen)
