"""
N.I.N.A. Advanced API client — just the parts transitphot needs.

The Advanced API plugin (the same one Touch'N'Stars is built on) exposes
N.I.N.A.'s state over HTTP, by default on port 1888 under /v2/api. Two
things it knows that are otherwise guesswork:

* **When the sequence has finished.** Without the API, `night` waits for
  the capture folder to go quiet for 15 minutes. With it, the sequence's own
  status says when the run is over.
* **How each frame went.** The image history records HFR, star count and
  guiding RMS for every saved frame — failure modes (focus drift, a guiding
  glitch) that comparison-star flux alone does not catch.

Read-only by design: nothing here loads, starts or stops a sequence, or
moves equipment. Standard library only.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

HISTORY_FILE = "nina_image_history.json"

# N.I.N.A.'s sequence statuses. A container that has reached one of these
# will not run again without being reset.
TERMINAL = {"FINISHED", "FAILED", "SKIPPED"}


class NinaAPI:
    def __init__(self, base: str, timeout: float = 10.0):
        base = base.rstrip("/")
        if not base.startswith("http"):
            base = "http://" + base
        if not base.endswith("/v2/api"):
            base += "/v2/api"
        self.base = base
        self.timeout = timeout

    def _call(self, path: str, data: bytes | None = None, method: str = "GET"):
        url = f"{self.base}/{path.lstrip('/')}"
        headers = {"User-Agent": "transitphot"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.load(r)
        except urllib.error.HTTPError as exc:
            # N.I.N.A. explains refusals in the body ("Sequence is already
            # running", "Sequence is not initialized"); surface that rather
            # than a bare status code.
            try:
                body = json.load(exc)
            except Exception:                            # noqa: BLE001
                raise RuntimeError(f"{path}: HTTP {exc.code}") from exc
            raise RuntimeError(body.get("Error")
                               or f"{path}: HTTP {exc.code}") from exc
        if not body.get("Success", False):
            raise RuntimeError(body.get("Error") or f"{path} failed")
        return body.get("Response")

    def get(self, path: str):
        """GET an endpoint and return its Response field, or raise."""
        return self._call(path)

    def reachable(self) -> str | None:
        """The plugin version, or None if the API doesn't answer."""
        try:
            return str(self.get("version"))
        except Exception:                                # noqa: BLE001
            return None

    # ---------- sequence ----------
    def targets_status(self) -> tuple[str | None, str | None]:
        """
        Status of the sequence's target area, and the name of the item
        currently running (if any).

        The target area is where transitphot's exported sequences put
        everything — waits, flats, the transit run, shutdown — so its status
        is the status of the night.
        """
        state = self.get("sequence/state")
        targets = None
        for block in state or []:
            if isinstance(block, dict) and \
                    str(block.get("Name", "")).startswith("Targets"):
                targets = block
                break
        if targets is None:
            return None, None
        status = str(targets.get("Status") or "").upper() or None

        current = None
        for item in targets.get("Items") or []:
            # Items may arrive as dicts or, from some clients, as strings
            if isinstance(item, dict):
                if str(item.get("Status", "")).upper() == "RUNNING":
                    current = item.get("Name")
                    break
            elif isinstance(item, str) and "Status=RUNNING" in item:
                m = re.search(r"Name=([^;]+)", item)
                current = m.group(1).strip() if m else None
                break
        return status, current

    # ---------- image history ----------
    def image_history(self) -> list[dict]:
        resp = self.get("image-history?all=true")
        return [h for h in (resp or []) if isinstance(h, dict)]


def merge_history(path: Path, entries: list[dict]) -> int:
    """
    Merge history entries into a JSON file on disk, keyed by filename.

    N.I.N.A. keeps its image history in memory, so a restart mid-night
    loses it. Saving as we go means the frames recorded before a restart
    keep their quality data. Returns the number of entries now stored.
    """
    stored = {}
    if path.exists():
        try:
            for e in json.loads(path.read_text()):
                if e.get("Filename"):
                    stored[e["Filename"]] = e
        except Exception:                                # noqa: BLE001
            pass
    for e in entries:
        if e.get("Filename"):
            stored[e["Filename"]] = e
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(list(stored.values()), indent=1))
    tmp.replace(path)
    return len(stored)


def wait_for_sequence_end(api: NinaAPI, *, poll_s: float = 60.0,
                          history_path: Path | None = None,
                          max_unreachable: int = 10,
                          say=print) -> bool:
    """
    Block until N.I.N.A.'s sequence has finished.

    The end is recognized two ways: the target area reaching a terminal
    status, or having been seen running and then no longer running (which
    covers a sequence stopped by hand). A sequence that has not started yet
    is simply waited for.

    Returns True when the sequence ended, False if the API stopped
    answering — the caller then falls back to idle detection.
    """
    seen_running = False
    last_status, last_item = None, None
    misses = 0
    polls = 0
    while True:
        try:
            status, item = api.targets_status()
            misses = 0
        except Exception as exc:                         # noqa: BLE001
            misses += 1
            say(f"  N.I.N.A. API not answering ({exc}); "
                f"{max_unreachable - misses} tries left")
            if misses >= max_unreachable:
                return False
            time.sleep(poll_s)
            continue

        if status != last_status or item != last_item:
            say(f"  sequence {status or 'not loaded'}"
                + (f" — {item}" if item else ""))
            last_status, last_item = status, item

        # save the image history as we go, in case N.I.N.A. restarts
        if history_path is not None and polls % 5 == 0:
            try:
                n = merge_history(history_path, api.image_history())
                if n and polls % 30 == 0:
                    say(f"  image history: {n} frame(s) recorded")
            except Exception:                            # noqa: BLE001
                pass
        polls += 1

        if status == "RUNNING":
            seen_running = True
        elif status in TERMINAL:
            say(f"  sequence ended ({status})")
            return True
        elif seen_running and status is not None:
            say(f"  sequence stopped ({status}) after running")
            return True

        time.sleep(poll_s)


# ---------------------------------------------------------------- quality
def _rms_arcsec(text: str | None) -> float | None:
    """'Tot: 0.46 (0.51")' -> 0.51 — the total guiding RMS in arcseconds."""
    if not text:
        return None
    m = re.search(r"\(([\d.]+)\s*\"?\)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"Tot:\s*([\d.]+)", text)
    return float(m.group(1)) if m else None


def load_history(*folders: Path) -> dict[str, dict]:
    """History entries keyed by filename stem, from the first file found."""
    for f in folders:
        p = Path(f) / HISTORY_FILE
        if p.exists():
            try:
                return {Path(e["Filename"]).stem: e
                        for e in json.loads(p.read_text())
                        if e.get("Filename")}
            except Exception:                            # noqa: BLE001
                return {}
    return {}


def quality_flags(paths: list[Path], history: dict[str, dict],
                  hfr_k: float = 4.0, rms_factor: float = 2.5,
                  rms_floor: float = 1.5) -> dict[Path, str]:
    """
    Frames whose focus or guiding was clearly off, with the reason.

    HFR: flagged when more than hfr_k robust standard deviations above the
    session median — focus drift or passing thin cloud blurring the stars.
    Guiding: flagged when the total RMS exceeds both rms_factor times the
    session median and an absolute floor, so a night of uniformly soft
    guiding is not wholesale rejected.

    Deliberately conservative. The cloud filter and the fit's outlier
    handling already deal with ordinary scatter; this catches frames that
    are wrong for a reason the photometry can't see.
    """
    import numpy as np

    matched = {}
    for p in paths:
        name = p.stem
        entry = history.get(name)
        if entry is None:
            # calibrated copies carry a prefix; match on the original name
            for stem, e in history.items():
                if stem in name:
                    entry = e
                    break
        if entry is not None:
            matched[p] = entry
    if len(matched) < 10:
        return {}

    hfr = np.array([float(e.get("HFR") or np.nan) for e in matched.values()])
    rms = np.array([_rms_arcsec(e.get("RmsText")) or np.nan
                    for e in matched.values()])
    flags = {}

    ok = np.isfinite(hfr)
    if ok.sum() >= 10:
        med = np.median(hfr[ok])
        mad = 1.4826 * np.median(np.abs(hfr[ok] - med)) or 1e-9
        for (p, e), h in zip(matched.items(), hfr):
            if np.isfinite(h) and h > med + hfr_k * mad:
                flags[p] = f"HFR {h:.2f} vs session median {med:.2f}"

    ok = np.isfinite(rms)
    if ok.sum() >= 10:
        med = np.median(rms[ok])
        lim = max(rms_factor * med, rms_floor)
        for (p, e), r in zip(matched.items(), rms):
            if np.isfinite(r) and r > lim and p not in flags:
                flags[p] = f"guiding RMS {r:.2f}\" vs median {med:.2f}\""
    return flags


# ---------------------------------------------------------------------
# Commands. Everything above only reads; these change what N.I.N.A. does,
# and `start` moves equipment — callers should confirm with the observer
# first, and should not skip validation, which is what checks the gear is
# connected before an unattended run.
# ---------------------------------------------------------------------
def list_available(api: "NinaAPI") -> list[str]:
    return [str(x) for x in (api.get("sequence/list-available") or [])]


def load_json(api: "NinaAPI", body: bytes) -> str:
    """Load a sequence straight from JSON — no file on disk required."""
    return str(api._call("sequence/load", data=body, method="POST"))


def load_by_name(api: "NinaAPI", name: str) -> str:
    """Load one of the sequences in N.I.N.A.'s own folder."""
    import urllib.parse
    return str(api.get("sequence/load?sequenceName="
                       + urllib.parse.quote(name)))


def start(api: "NinaAPI", skip_validation: bool = False) -> str:
    """
    Start the loaded sequence.

    Validation stays on by default: it is what catches equipment that isn't
    connected, and an unattended transit run is the last place to skip it.
    """
    path = "sequence/start"
    if skip_validation:
        path += "?skipValidation=true"
    return str(api.get(path))
