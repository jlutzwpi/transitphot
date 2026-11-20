"""
Transit ephemerides from several sources, compared before you observe.

No single catalog can be trusted to be current. On TOI-3757 b (Nov 2026)
the NASA archive's best-looking entry was the 2019 discovery ephemeris,
claiming +/-4.3 min for the night but actually 16 min off; ExoClock did not
list the planet at all; Exoplanet Watch's fit, which folds in amateur
timings, predicted the transit within 1.6 min. The only way to see that
coming is to ask every source and look at how far apart they are.

Sources:
  * NASA Exoplanet Archive (TAP): every published ephemeris in `ps`, plus
    the merged `pscomppars` row for coordinates, magnitude, depth, duration
  * ExoClock: a timing-focused, community-refined ephemeris
  * Exoplanet Watch (NASA/JPL + AAVSO): fitted ephemerides including the
    observations submitted to the AAVSO exoplanet database
  * ExoFOP TOI table: TESS ephemerides, for TOI targets only

Each download is cached under ~/.transitphot/ephemeris, so a 503 from one
service falls back to the last good copy instead of failing the lookup.
All of these are linear ephemerides: none can follow a decaying orbit such
as WASP-12 b's, and the disagreement check is what flags that too.

Stdlib only, because the GUI is.
"""

from __future__ import annotations

import csv
import html
import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CACHE_DIR = Path.home() / ".transitphot" / "ephemeris"
HEADERS = {"User-Agent": "transitphot"}

NASA_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
EXOCLOCK_URL = "https://www.exoclock.space/database/planets_json"
EPW_URL = ("https://epw.ipac.caltech.edu/results/datasets/"
           "exoplanet_watch_results.json")
EXOFOP_URL = ("https://exofop.ipac.caltech.edu/tess/download_toi.php"
              "?sort=toi&output=csv")

# Sources disagreeing by more than this many minutes (and by more than
# their combined uncertainty allows) trigger a warning.
DISAGREE_MIN = 3.0
# A candidate whose own prediction is looser than this says little about
# where the transit is, so it is listed but kept out of the spread.
INFORMATIVE_MIN = 15.0


class LookupFailed(Exception):
    """No source could be reached and nothing usable was cached."""


def now_jd() -> float:
    return time.time() / 86400.0 + 2440587.5


def _key(name: str) -> str:
    """'WASP-10 b', 'WASP-10b' and 'wasp-10 B' all compare equal."""
    return re.sub(r"\s+", "", str(name)).lower()


def normalize_name(name: str) -> str:
    """
    The archive's spelling: a space before the planet letter and catalog
    prefixes in their usual case — 'toi-3629b' -> 'TOI-3629 b'. Exact-match
    lookups fail on 'TOI-3629b', and a prefix match on it finds nothing.
    """
    n = re.sub(r"\s+", " ", str(name).strip())
    n = re.sub(r"(\d)\s*([a-hA-H])$", lambda m: f"{m[1]} {m[2].lower()}", n)
    m = re.match(r"([A-Za-z]+)(-.*)", n)
    if m and m[1].upper() in {"TOI", "WASP", "HAT", "KELT", "XO", "TRES",
                              "GPX", "QATAR", "KOI", "HATS", "NGTS", "COROT",
                              "TIC", "LHS", "GJ", "HD"}:
        prefix = {"TRES": "TrES", "QATAR": "Qatar", "COROT": "CoRoT"}.get(
            m[1].upper(), m[1].upper())
        n = prefix + m[2]
    return n


def sexagesimal_to_deg(text, hours: bool) -> float | None:
    """'HH:MM:SS.ss' or '+DD:MM:SS.ss' -> decimal degrees (ExoClock's format;
    RA in hours)."""
    if not text:
        return None
    try:
        text = str(text).strip()
        sign = -1.0 if text.startswith("-") else 1.0
        h, m, s = (float(x) for x in text.lstrip("+-").split(":"))
        deg = sign * (h + m / 60.0 + s / 3600.0)
        return deg * 15.0 if hours else deg
    except (TypeError, ValueError):
        return None


def _float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


@dataclass
class Ephemeris:
    source: str
    t0: float                    # BJD_TDB
    period: float                # days
    t0_err: float = 0.0          # days; 0 = not published
    period_err: float = 0.0
    note: str = ""               # e.g. "offline copy from 2 days ago"

    def errors(self) -> tuple[float, float]:
        """Published uncertainties, or ones inferred from the decimals given
        when a source publishes none."""
        dE, dP = abs(self.t0_err or 0.0), abs(self.period_err or 0.0)
        if dE == 0 and dP == 0:
            dP = 10 ** -max(len(repr(self.period).split(".")[-1]), 1)
            dE = 10 ** -max(len(repr(self.t0).split(".")[-1]), 1)
        return dE, dP

    def predict(self, jd: float) -> tuple[float, int, float]:
        """(mid-time, orbits since epoch, uncertainty in days) for the
        transit nearest jd. Uncertainty grows as sqrt(dT0^2 + (n dP)^2)."""
        n = round((jd - self.t0) / self.period)
        dE, dP = self.errors()
        return self.t0 + n * self.period, n, math.sqrt(dE ** 2 + (n * dP) ** 2)


@dataclass
class Comparison:
    best: Ephemeris
    tc: float                    # the chosen ephemeris' next mid-transit
    unc_min: float
    spread_min: float            # max minus min over informative sources
    lines: list[str] = field(default_factory=list)
    disagree: bool = False


# ----------------------------------------------------------------------
# Downloads with an on-disk fallback
# ----------------------------------------------------------------------
_SSL = None


def _ssl_context():
    """
    The system trust store plus certifi's, when certifi is installed.
    Python on Windows can build a chain through a stale certificate in the
    Windows store and report "certificate has expired" for a valid site:
    Exoplanet Watch's Let's Encrypt chain fails that way while curl and
    browsers accept it. certifi's current roots let OpenSSL find the good
    path. certifi isn't a dependency here (the GUI is stdlib-only), but
    astroquery/requests bring it along in practice.
    """
    global _SSL
    if _SSL is None:
        import ssl
        _SSL = ssl.create_default_context()
        try:
            import certifi
            _SSL.load_verify_locations(certifi.where())
        except Exception:                               # noqa: BLE001
            pass
    return _SSL


def _curl(url: str, timeout: float) -> bytes | None:
    """Fetch with the system curl, or None if there is none or it fails."""
    import shutil
    import subprocess
    import sys
    exe = shutil.which("curl")
    if not exe:
        return None
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        p = subprocess.run([exe, "-sSfL", "-A", HEADERS["User-Agent"],
                            "--max-time", str(int(timeout)), url],
                           capture_output=True, timeout=timeout + 10,
                           creationflags=flags)
    except Exception:                                   # noqa: BLE001
        return None
    return p.stdout if p.returncode == 0 and p.stdout else None


def _fetch(url: str, timeout: float) -> bytes:
    """
    urllib first; on a refusal (403/429/503), the system curl. The NASA
    archive's front end refuses Python's HTTP client for long stretches —
    the 403s and 503s observers see — while curl from the same machine, same
    minute, gets a 200. Windows 10+ ships curl.exe.
    """
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=_ssl_context()) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429, 503):
            data = _curl(url, timeout)
            if data is not None:
                return data
        raise


def _cached(fname: str, url: str, max_age_days: float, timeout: float,
            validate) -> tuple[bytes, str]:
    """
    The file's bytes and a note ('' when fresh). A fresh cache is used as
    is; otherwise download, validate (a 200 can still be a challenge page)
    and replace. If the download fails, fall back to any older copy.
    """
    path = CACHE_DIR / fname
    age = ((time.time() - path.stat().st_mtime) / 86400.0
           if path.exists() else None)
    if age is not None and age < max_age_days:
        return path.read_bytes(), ""
    try:
        data = _fetch(url, timeout)
        validate(data)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return data, ""
    except Exception as exc:                            # noqa: BLE001
        if age is not None:
            return path.read_bytes(), (f"offline copy from {age:.0f} day(s) "
                                       f"ago; download failed ({exc})")
        raise


def _json_ok(data: bytes):
    json.loads(data)


def _csv_ok(data: bytes):
    head = data[:2000].split(b"\n", 1)[0]
    if b"TOI" not in head or b"Period (days)" not in head:
        raise ValueError("not the TOI table")


# ----------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------
def _nasa_query(adql: str, timeout: float = 90):
    url = (f"{NASA_TAP}?query={urllib.parse.quote(adql)}&format=json")
    return json.loads(_fetch(url, timeout))


def nasa(name: str, log=print) -> tuple[dict | None, list[Ephemeris]]:
    """
    The merged pscomppars row (coordinates etc.) and every published
    ephemeris in `ps`. Per-planet results are cached, so an outage falls
    back to the last answer for that planet.
    """
    cache = CACHE_DIR / "nasa" / f"{_key(name)}.json"
    wanted = ("pl_name, ra, dec, sy_vmag, sy_gaiamag, pl_orbper, "
              "pl_tranmid, pl_trandur, pl_trandep")
    safe = name.replace("'", "''")
    queries = [
        f"SELECT {wanted} FROM pscomppars WHERE pl_name = '{safe}'",
        f"SELECT {wanted} FROM pscomppars "
        f"WHERE UPPER(pl_name) = UPPER('{safe}')",
        f"SELECT {wanted} FROM pscomppars "
        f"WHERE UPPER(pl_name) LIKE UPPER('{safe}%')",
    ]
    comp, ps, err = None, None, None
    try:
        for adql in queries:
            # The archive sometimes answers with a 403 browser challenge or
            # a 503; one retry after a pause usually gets through.
            for delay in (0, 5):
                if delay:
                    time.sleep(delay)
                try:
                    rows = _nasa_query(adql)
                    err = None
                    break
                except Exception as exc:                # noqa: BLE001
                    err, rows = exc, None
            if err is not None:
                raise err
            if rows:
                comp = dict(rows[0])
                break
        if comp is not None:
            pl = comp["pl_name"].replace("'", "''")
            # No "AND x IS NOT NULL" here: the archive's firewall answers
            # that pattern with a 403 (it reads as SQL injection), which
            # was the "forbidden" error observers kept hitting. Rows
            # without an ephemeris are dropped below instead.
            ps = _nasa_query(
                "SELECT pl_tranmid, pl_tranmiderr1, pl_orbper, "
                "pl_orbpererr1, pl_refname FROM ps "
                f"WHERE pl_name = '{pl}'")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"comp": comp, "ps": ps}))
        else:
            log(f"  not in the NASA archive under '{name}'")
        note = ""
    except Exception as exc:                            # noqa: BLE001
        if not cache.exists():
            log(f"  NASA archive unavailable ({exc})")
            return None, []
        age = (time.time() - cache.stat().st_mtime) / 86400.0
        saved = json.loads(cache.read_text())
        comp, ps = saved["comp"], saved["ps"]
        note = f"offline copy from {age:.0f} day(s) ago; archive failed ({exc})"
        log(f"  NASA archive unavailable; using its {note.split(';')[0]}")

    out = []
    for r in ps or []:
        t0, p = _float(r.get("pl_tranmid")), _float(r.get("pl_orbper"))
        if not t0 or not p or p <= 0:
            continue
        ref = html.unescape(
            re.sub(r"<[^>]+>", "", r.get("pl_refname") or "")).strip()
        out.append(Ephemeris(f"NASA: {ref or 'archive'}", t0, p,
                             _float(r.get("pl_tranmiderr1")) or 0.0,
                             _float(r.get("pl_orbpererr1")) or 0.0, note))
    if not out and comp and comp.get("pl_tranmid") and comp.get("pl_orbper"):
        out.append(Ephemeris("NASA: archive default",
                             float(comp["pl_tranmid"]),
                             float(comp["pl_orbper"]), note=note))
    return comp, out


def exoclock(name: str, log=print) -> tuple[dict | None, Ephemeris | None]:
    """ExoClock's current ephemeris, plus coordinates and the like in the
    pscomppars shape so it can stand in when NASA is down."""
    try:
        data, note = _cached("exoclock.json", EXOCLOCK_URL, 1, 60, _json_ok)
    except Exception as exc:                            # noqa: BLE001
        log(f"  ExoClock unavailable ({exc})")
        return None, None
    e = next((v for k, v in json.loads(data).items() if _key(k) == _key(name)),
             None)
    if e is None:
        return None, None
    info = {"pl_name": name,
            "ra": sexagesimal_to_deg(e.get("ra_j2000"), hours=True),
            "dec": sexagesimal_to_deg(e.get("dec_j2000"), hours=False),
            "sy_vmag": e.get("v_mag"), "sy_gaiamag": e.get("gaia_g_mag"),
            "pl_trandur": e.get("duration_hours")}
    if e.get("rp_over_rs"):
        info["pl_trandep"] = float(e["rp_over_rs"]) ** 2 * 100  # percent
    t0 = _float(e.get("t0_bjd_tdb") or e.get("ephem_mid_time"))
    p = _float(e.get("period_days") or e.get("ephem_period"))
    eph = None
    if t0 and p:
        eph = Ephemeris("ExoClock", t0, p,
                        abs(_float(e.get("t0_unc") or e.get("ephem_mid_time_e1")) or 0.0),
                        abs(_float(e.get("period_unc") or e.get("ephem_period_e1")) or 0.0),
                        note)
    return info, eph


def exoplanet_watch(name: str, log=print) -> Ephemeris | None:
    """Exoplanet Watch's fitted ephemeris. The file is large (~28 MB) and
    changes rarely, so it is refreshed weekly."""
    try:
        data, note = _cached("exoplanet_watch.json", EPW_URL, 7, 180, _json_ok)
    except Exception as exc:                            # noqa: BLE001
        log(f"  Exoplanet Watch unavailable ({exc})")
        return None
    t = next((x for x in json.loads(data).get("targets", [])
              if _key(x.get("name", "")) == _key(name)), None)
    e = (t or {}).get("ephemeris") or {}
    t0, p = _float(e.get("Tc")), _float(e.get("Period"))
    if not t0 or not p:
        return None
    return Ephemeris("Exoplanet Watch", t0, p, _float(e.get("Tc_err")) or 0.0,
                     _float(e.get("Period_err")) or 0.0, note)


def exofop_toi(name: str, ref_period: float | None,
               log=print) -> Ephemeris | None:
    """
    The TESS ephemeris for a 'TOI-NNNN x' planet. The planet letter does not
    map reliably onto the .01/.02 suffix, so with several candidates on one
    star, pick the one whose period matches the other sources.
    """
    m = re.match(r"\s*TOI[-\s]?(\d+)", name, re.I)
    if not m:
        return None
    try:
        data, note = _cached("exofop_toi.csv", EXOFOP_URL, 1, 120, _csv_ok)
    except Exception as exc:                            # noqa: BLE001
        log(f"  ExoFOP unavailable ({exc})")
        return None
    rows = [r for r in csv.DictReader(io.StringIO(data.decode("utf-8", "replace")))
            if r.get("TOI", "").split(".")[0] == m.group(1)]
    if len(rows) > 1:
        if not ref_period:
            return None
        rows = [r for r in rows
                if abs((_float(r.get("Period (days)")) or 0) - ref_period)
                < 0.01 * ref_period]
    if len(rows) != 1:
        return None
    r = rows[0]
    t0, p = _float(r.get("Epoch (BJD)")), _float(r.get("Period (days)"))
    if not t0 or not p:
        return None
    return Ephemeris(f"ExoFOP TOI {r['TOI']}", t0, p,
                     _float(r.get("Epoch (BJD) err")) or 0.0,
                     _float(r.get("Period (days) err")) or 0.0, note)


def gather(name: str, log=print) -> tuple[dict | None, list[Ephemeris]]:
    """
    Planet parameters (pscomppars shape: pl_name, ra, dec, sy_gaiamag,
    sy_vmag, pl_trandur, pl_trandep) and every ephemeris found.
    """
    name = normalize_name(name)
    log("  asking the NASA Exoplanet Archive")
    comp, cands = nasa(name, log)
    canonical = (comp or {}).get("pl_name") or name
    log("  asking ExoClock, Exoplanet Watch" +
        (", ExoFOP" if re.match(r"\s*TOI", canonical, re.I) else ""))
    ec_info, ec = exoclock(canonical, log)
    if ec is None and canonical != name:
        ec_info, ec = exoclock(name, log)
    if ec:
        cands.append(ec)
    epw = exoplanet_watch(canonical, log)
    if epw:
        cands.append(epw)
    ref_p = next((c.period for c in cands), None)
    fop = exofop_toi(canonical, ref_p, log)
    if fop:
        # The archive keeps its own copy of the ExoFOP row; the direct one
        # is at least as fresh, so drop the copy when they match.
        cands = [c for c in cands
                 if not (c.source == "NASA: ExoFOP"
                         and abs(c.t0 - fop.t0) < 1e-5
                         and abs(c.period - fop.period) < 1e-8)]
        cands.append(fop)
    info = comp or ec_info
    if info is not None and comp is None:
        log("  planet parameters from ExoClock")
    return info, cands


def compare(cands: list[Ephemeris], jd: float | None = None) -> Comparison:
    """
    Choose the ephemeris to use, then lay every source's prediction for
    that same transit side by side and flag disagreement beyond what the
    stated uncertainties allow.

    The NASA Exoplanet Archive comes first: among its published
    ephemerides, the one that predicts the next transit most precisely by
    its own stated uncertainty. The archive is the reference the community
    cites, and its entries are peer-reviewed (Dennis Conti's advice). The
    other sources are cross-checks; only when the archive has no
    ephemeris for the planet does the most precise of them get used.
    That trades a little precision for provenance, and the comparison
    keeps it honest: on TOI-3757 b the archive's best entry was 16 min
    off and Exoplanet Watch was right, which the DISAGREES flag shows.
    """
    if not cands:
        raise ValueError("no ephemerides to compare")
    jd = now_jd() if jd is None else jd
    preds = [(c, *c.predict(jd)) for c in cands]
    nasa = [x for x in preds if x[0].source.startswith("NASA")]
    best, _, _, _ = min(nasa or preds, key=lambda x: x[3])
    sharpest = min(preds, key=lambda x: x[3])[0]
    # The next transit by the chosen ephemeris, then each source's
    # prediction of that same event.
    n = math.ceil((jd - best.t0) / best.period)
    tc_best = best.t0 + n * best.period
    rows = []
    for c in cands:
        tc, nn, unc = c.predict(tc_best)
        rows.append((c, tc, unc))
    u_best = next(u for c, _, u in rows if c is best)
    rows.sort(key=lambda r: r[2])

    def disagrees(tc, u):
        d = abs(tc - tc_best) * 1440
        return d > DISAGREE_MIN and d > 2 * math.hypot(u, u_best) * 1440

    # The warning looks at the three most precise informative sources only.
    # An old paper on a well-observed planet being stale is not news; the
    # leading candidates disagreeing is (TOI-3757 b: Exoplanet Watch vs the
    # discovery paper, 16 min). Listing all of them still shows the rest.
    top = [(c, tc, u) for c, tc, u in rows
           if u * 1440 <= INFORMATIVE_MIN][:3]
    if not any(c is best for c, _, _ in top):
        top.append(next(r for r in rows if r[0] is best))
    spread = ((max(tc for _, tc, _ in top) - min(tc for _, tc, _ in top))
              * 1440 if top else 0.0)
    disagree = any(disagrees(tc, u) for c, tc, u in top if c is not best)

    lines = [f"  ephemerides for the next transit (BJD_TDB):"]
    MAX_LINES = 8
    for c, tc, u in rows[:MAX_LINES]:
        d = (tc - tc_best) * 1440
        s = f"    {c.source[:34]:34} {tc:.5f}  ±{u * 1440:5.1f} min"
        if c is best:
            s += "   <- using"
            if c is not sharpest:
                s += " (NASA first)"
        elif u * 1440 > INFORMATIVE_MIN:
            s += "   (too uncertain to compare)"
        else:
            s += f"   {d:+6.1f} min"
            if disagrees(tc, u):
                s += "  DISAGREES"
        if c.note:
            s += f"  [{c.note.split(';')[0]}]"
        lines.append(s)
    if len(rows) > MAX_LINES:
        lines.append(f"    … and {len(rows) - MAX_LINES} less precise")
    if disagree:
        lines.append(
            f"  WARNING: the sources disagree by {spread:.0f} min for this "
            f"transit, more than their stated uncertainties allow — at least "
            f"one is stale. Extend the baseline on both sides by {spread:.0f} "
            f"min or more so ingress and egress are caught either way, and "
            f"say which ephemeris the O-C refers to.")
    return Comparison(best, tc_best, u_best * 1440, spread, lines, disagree)
