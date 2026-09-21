"""
transitphot serve — drive the pipeline from a phone.

The processing computer holds the frames, so the work has to happen there.
This puts a small web page in front of it: pick a target, tap start, watch
the log, see the light curve. Reachable over the LAN at home and over
Tailscale from anywhere, which is how most people already reach their
imaging computers.

Deliberately not an app. The same page works on Android, iOS and a laptop,
there is nothing to install or sign, and it reuses the settings file the
desktop GUI already writes — so a rig configured once is configured for
both.

Security: this runs subprocesses on your computer. It binds to all
interfaces so a phone can reach it, which is safe behind Tailscale or on a
home LAN and emphatically unsafe on a public network. A token is required
unless you pass --no-token, and it is printed at startup.
"""

import asyncio
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)

SETTINGS = Path.home() / ".transitphot_gui.json"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:                                    # noqa: BLE001
        return {}


def save_settings(data: dict):
    try:
        cur = load_settings()
        cur.update(data)
        SETTINGS.write_text(json.dumps(cur, indent=2))
    except Exception:                                    # noqa: BLE001
        pass


PAGE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>TransitPlanner Processor</title>
<style>
  :root { --sky:#0B1220; --panel:#111B2E; --panel2:#0E1626; --ink:#E7E9F0;
          --dim:#8B94A7; --line:#1D2A42; --go:#7FD1A8; --warn:#E8C468;
          box-sizing:border-box; padding-top:env(safe-area-inset-top,0px);
          padding-bottom:env(safe-area-inset-bottom,0px); }
  *,*::before,*::after { box-sizing:inherit; }
  body { margin:0; background:var(--sky); color:var(--ink); font:16px/1.5
         system-ui,-apple-system,sans-serif; padding:0 14px 40px; }
  h1 { font-size:18px; font-weight:600; margin:18px 0 4px; }
  .sub { color:var(--dim); font-size:13.5px; margin:0 0 16px; }
  fieldset { border:1px solid var(--line); border-radius:8px; margin:0 0 14px;
             padding:12px 14px; }
  legend { color:var(--dim); font-size:13px; padding:0 6px; }
  label { display:block; font-size:13px; color:var(--dim); margin:10px 0 3px; }
  input,select { width:100%; background:var(--panel2); border:1px solid var(--line);
                 color:var(--ink); border-radius:6px; padding:11px 12px;
                 font:15px system-ui,sans-serif; }
  input:focus-visible,button:focus-visible { outline:2px solid var(--go);
                                             outline-offset:2px; }
  .row { display:flex; gap:10px; } .row > * { flex:1; }
  button { background:var(--panel); border:1px solid var(--line); color:var(--ink);
           border-radius:8px; padding:14px 18px; font:600 15px system-ui,sans-serif;
           width:100%; margin-top:14px; }
  button.go { border-color:var(--go); color:var(--go); }
  button:disabled { opacity:.45; }
  #log { background:#08111F; border:1px solid var(--line); border-radius:8px;
         padding:10px 12px; font:12.5px/1.45 ui-monospace,Menlo,monospace;
         white-space:pre-wrap; overflow-x:auto; min-height:180px;
         max-height:46vh; overflow-y:auto; margin-top:14px; }
  #status { font-size:13.5px; color:var(--warn); margin-top:8px; min-height:20px; }
  img { width:100%; border-radius:8px; margin-top:14px; display:none; }
  .hint { color:var(--dim); font-size:12px; margin-top:4px; }
</style></head><body>
<h1>TransitPlanner Processor</h1>
<p class="sub">Runs on __HOST__ — the frames stay there.</p>

<fieldset><legend>Send a sequence to N.I.N.A.</legend>
  <label for="nina_dir">N.I.N.A. sequence folder on the capture PC</label>
  <input id="nina_dir" placeholder="\\\\MELE-PC\\N.I.N.A">
  <div class="hint">A shared folder this computer can write to — the folder
    Touch'N'Stars lists sequences from.</div>
  <label for="seqfile">Sequence exported from TransitPlanner</label>
  <input id="seqfile" type="file" accept=".json,application/json">
  <button type="button" id="uploadbtn" onclick="uploadSeq()">Send to capture PC</button>
  <div id="uploadmsg" class="hint"></div>
</fieldset>

<fieldset><legend>Folders</legend>
  <label for="source">Capture folder (leave blank if already copied)</label>
  <input id="source" placeholder="\\\\MELE-PC\\n.i.n.a\\2026-09-16\\TARGET\\LIGHT">
  <label for="lights_root">Local folder for the lights</label>
  <input id="lights_root" placeholder="D:\\xfer\\TARGET">
  <label for="bias">Bias</label><input id="bias">
  <div class="row">
    <div><label for="darks">Darks</label><input id="darks"></div>
    <div><label for="flats">Flats</label><input id="flats"></div>
  </div>
  <div class="hint">Leave any calibration folder blank to skip it.</div>
</fieldset>

<fieldset><legend>Target</legend>
  <label for="target_name">Name</label>
  <input id="target_name" placeholder="TOI-3629 b">
  <button type="button" id="lookupbtn" onclick="lookup()">Look up in NASA archive</button>
  <div id="lookupmsg" class="hint"></div>
  <div class="row">
    <div><label for="ra">RA (deg)</label><input id="ra" inputmode="decimal"></div>
    <div><label for="dec">Dec (deg)</label><input id="dec" inputmode="decimal"></div>
  </div>
  <div class="row">
    <div><label for="target_mag">Mag (G)</label><input id="target_mag" inputmode="decimal"></div>
    <div><label for="depth_ppm">Depth (ppm)</label><input id="depth_ppm" inputmode="decimal"></div>
  </div>
  <div class="row">
    <div><label for="duration_hours">Duration (h)</label><input id="duration_hours" inputmode="decimal"></div>
    <div><label for="filter_band">Filter</label><input id="filter_band" placeholder="R"></div>
  </div>
  <label for="epoch_bjd">Epoch (BJD_TDB)</label><input id="epoch_bjd" inputmode="decimal">
  <label for="period">Period (days)</label><input id="period" inputmode="decimal">
</fieldset>

<fieldset><legend>Site and equipment</legend>
  <div class="row">
    <div><label for="lat">Latitude</label><input id="lat" inputmode="decimal"></div>
    <div><label for="lon">Longitude (E+)</label><input id="lon" inputmode="decimal"></div>
  </div>
  <div class="row">
    <div><label for="elevation">Elevation (m)</label><input id="elevation" inputmode="decimal"></div>
    <div><label for="binning">Binning</label><input id="binning" placeholder="1x1"></div>
  </div>
  <div class="row">
    <div><label for="aavso_obscode">AAVSO observer code</label><input id="aavso_obscode"></div>
    <div><label for="aavso_filter">AAVSO filter</label><input id="aavso_filter" placeholder="auto"></div>
  </div>
  <div class="hint">Saved on the processing computer and shared with the
    desktop app. Leave the AAVSO filter blank to derive it from the filter.</div>
</fieldset>

<fieldset><legend>Options</legend>
  <label for="after_idle">Start after the capture folder is idle (minutes)</label>
  <input id="after_idle" inputmode="numeric" value="15">
  <div class="hint">Leave the phone; it waits for the sequence to finish.</div>
</fieldset>

<button class="go" id="start" onclick="start()">Start the night</button>
<button id="stop" onclick="stop()" disabled>Stop</button>
<div id="status"></div>
<div id="log"></div>
<img id="plot" alt="Light curve">

<script>
const F = ["source","lights_root","bias","darks","flats","target_name","ra","dec",
           "target_mag","depth_ppm","duration_hours","filter_band","epoch_bjd",
           "period","after_idle","lat","lon","elevation","binning",
           "aavso_obscode","aavso_filter","nina_dir"];
const g = id => document.getElementById(id);
const say = t => { const l = g("log"); l.textContent += t;
                   l.scrollTop = l.scrollHeight; };

// Every API call must carry the token from the page URL — the server
// rejects anything without it.
const TOKEN = new URLSearchParams(location.search).get("t") || "";
function withToken(path) {
  if (!TOKEN) return path;
  return path + (path.includes("?") ? "&" : "?") + "t=" + encodeURIComponent(TOKEN);
}
async function api(path, opts = {}) {
  const r = await fetch(withToken(path), opts);
  let body = null;
  try { body = await r.json(); } catch (e) {}
  if (!r.ok) {
    const why = (body && (body.detail || body.error)) || r.statusText;
    throw new Error(r.status + " " + why);
  }
  return body || {};
}

async function boot() {
  try {
    const s = await api("api/settings");
    F.forEach(k => { if (s[k] != null && s[k] !== "") g(k).value = s[k]; });
  } catch (e) {
    g("status").textContent = "Could not load saved settings: " + e.message;
  }
}
boot();

function values() {
  const o = {}; F.forEach(k => o[k] = g(k).value.trim()); return o;
}

async function lookup() {
  const name = g("target_name").value.trim();
  const msg = g("lookupmsg");
  if (!name) { msg.textContent = "Enter a target name first."; return; }
  g("lookupbtn").disabled = true;
  msg.textContent = "Looking up " + name + "… (the archive can be slow)";
  try {
    const d = await api("api/lookup?name=" + encodeURIComponent(name));
    if (d.error) { msg.textContent = d.error; return; }
    const f = d.fields || {};
    let n = 0;
    for (const [k, v] of Object.entries(f)) if (g(k)) { g(k).value = v; n++; }
    msg.textContent = n ? (d.note || "Filled " + n + " fields.")
                        : "The archive returned no usable values.";
  } catch (e) {
    msg.textContent = "Lookup failed: " + e.message;
  } finally {
    g("lookupbtn").disabled = false;
  }
}

async function uploadSeq() {
  const msg = g("uploadmsg");
  const f = g("seqfile").files[0];
  const dir = g("nina_dir").value.trim();
  if (!dir) { msg.textContent = "Enter the N.I.N.A. sequence folder first."; return; }
  if (!f)   { msg.textContent = "Choose the sequence file first."; return; }
  g("uploadbtn").disabled = true;
  msg.textContent = "Sending " + f.name + "…";
  try {
    // Save the folder first so the server knows where to write.
    await api("api/settings", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({nina_dir: dir})});
    const d = await api("api/upload_sequence?name=" + encodeURIComponent(f.name),
      {method:"POST", headers:{"Content-Type":"application/json"}, body: f});
    msg.textContent = d.error ? d.error :
      "Sent. It's now in " + d.path + " — load it from Touch'N'Stars.";
  } catch (e) {
    msg.textContent = "Upload failed: " + e.message;
  } finally {
    g("uploadbtn").disabled = false;
  }
}

let es = null;
async function start() {
  g("log").textContent = ""; g("plot").style.display = "none";
  g("start").disabled = true; g("stop").disabled = false;
  g("status").textContent = "Starting…";
  let d;
  try {
    d = await api("api/start", {method:"POST",
      headers:{"Content-Type":"application/json"}, body:JSON.stringify(values())});
  } catch (e) { g("status").textContent = "Could not start: " + e.message;
                done(); return; }
  if (d.error) { g("status").textContent = d.error; done(); return; }
  g("status").textContent = "Running…";
  es = new EventSource(withToken("api/stream"));
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.line != null) say(m.line);
    if (m.done) {
      g("status").textContent = m.code === 0 ? "Finished." :
                                "Exited with code " + m.code;
      if (m.plot) { const i = g("plot");
        i.src = withToken("api/plot") + "&n=" + Date.now();
        i.style.display = "block"; }
      done();
    }
  };
  es.onerror = () => { g("status").textContent = "Connection lost."; done(); };
}
function done() { if (es) { es.close(); es = null; }
  g("start").disabled = false; g("stop").disabled = true; }
async function stop() {
  try { await api("api/stop", {method:"POST"}); } catch (e) {}
  g("status").textContent = "Stopping…";
}
</script></body></html>"""


def build_app(token: Optional[str], urls: Optional[list] = None):
    urls = urls or []
    app = FastAPI(title="transitphot serve")
    state = {"proc": None, "queue": None, "plot": None, "task": None}

    def check(request: Request):
        if token and request.query_params.get("t") != token \
                and request.headers.get("x-token") != token:
            raise HTTPException(401, "Missing or wrong token")

    @app.get("/qr", response_class=HTMLResponse)
    def qr_page(request: Request):
        """
        QR codes for getting the URL onto a phone.

        This page contains the token, so it is served ONLY to this computer.
        Anyone else on the network asking for it gets a 403 — otherwise the
        page that exists to share access would hand it to everyone.
        """
        client = request.client.host if request.client else ""
        if client not in ("127.0.0.1", "::1", "localhost"):
            raise HTTPException(403, "The QR page is only shown on the "
                                     "computer running the server.")
        cards = []
        for label, url in urls:
            svg = _qr_svg(url)
            code = (svg if svg else
                    "<p style='color:#8B94A7'>Install segno for a scannable "
                    "code: <code>pip install segno</code></p>")
            cards.append(
                f"<div class='card'><h2>{label}</h2>{code}"
                f"<p class='url'>{url}</p></div>")
        body = "".join(cards) or "<p>No reachable addresses found.</p>"
        return HTMLResponse(f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open on your phone</title>
<style>
 body {{ background:#0B1220; color:#E7E9F0; font:16px system-ui,sans-serif;
        margin:0; padding:28px; }}
 h1 {{ font-size:20px; font-weight:600; margin:0 0 6px; }}
 .sub {{ color:#8B94A7; margin:0 0 24px; }}
 .wrap {{ display:flex; gap:24px; flex-wrap:wrap; }}
 .card {{ background:#111B2E; border:1px solid #1D2A42; border-radius:10px;
         padding:18px; width:300px; }}
 .card h2 {{ font-size:15px; font-weight:600; margin:0 0 12px; }}
 .card svg {{ width:100%; height:auto; background:#fff; border-radius:6px;
             padding:10px; box-sizing:border-box; }}
 .url {{ font:12px ui-monospace,monospace; color:#8B94A7;
        word-break:break-all; margin:12px 0 0; }}
</style></head><body>
<h1>Open TransitPlanner Processor on your phone</h1>
<p class="sub">Scan with the phone camera, then bookmark or add it to your
home screen. The address stays the same between restarts.</p>
<div class="wrap">{body}</div></body></html>""")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        if token and request.query_params.get("t") != token:
            return HTMLResponse(
                "<body style='font:16px system-ui;padding:2rem'>"
                "Add the token to the URL: <code>?t=YOUR_TOKEN</code>"
                "</body>", status_code=401)
        host = request.headers.get("host", "this computer")
        return HTMLResponse(PAGE.replace("__HOST__", host))

    @app.get("/api/settings")
    def settings(request: Request):
        check(request)
        return JSONResponse(load_settings())

    @app.post("/api/settings")
    async def put_settings(request: Request):
        check(request)
        data = await request.json()
        if isinstance(data, dict):
            save_settings(data)
        return {"ok": True}

    @app.post("/api/upload_sequence")
    async def upload_sequence(name: str, request: Request):
        """
        Write an uploaded N.I.N.A. sequence into the capture PC's sequence
        folder, reached over its network share.

        Touch'N'Stars only lists sequences from that folder, and the phone
        cannot write there directly. The processing computer can, since it
        already reads frames from the capture PC for syncing.

        The upload is checked before anything is written: it must be valid
        JSON, look like a N.I.N.A. sequence, and be of a sane size. The
        filename is reduced to a plain name so it cannot address anywhere
        outside the configured folder.
        """
        check(request)
        dest_dir = (load_settings().get("nina_dir") or "").strip()
        if not dest_dir:
            return {"error": "Set the N.I.N.A. sequence folder first."}

        body = await request.body()
        if len(body) > 5_000_000:
            return {"error": "That file is too large to be a sequence."}
        try:
            seq = json.loads(body.decode("utf-8-sig"))
        except Exception:                                # noqa: BLE001
            return {"error": "That file isn't valid JSON."}
        if "SequenceRootContainer" not in str(seq.get("$type", "")):
            return {"error": "That JSON isn't a N.I.N.A. sequence (no "
                             "SequenceRootContainer at the top)."}

        import re as _re
        base = Path(name).name                      # drop any path parts
        base = _re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
        if not base.lower().endswith(".json"):
            base += ".json"
        if not base or base == ".json":
            base = "transit-sequence.json"

        folder = Path(dest_dir)
        try:
            if not folder.exists():
                return {"error": f"Can't reach {folder}. Check the share is "
                                 f"available from this computer, and that "
                                 f"it's shared with write access."}
            target = folder / base
            tmp = folder / (base + ".part")
            tmp.write_bytes(body)
            os.replace(tmp, target)
        except PermissionError:
            return {"error": f"No write permission on {folder}. Share it with "
                             f"write access for this computer's account."}
        except OSError as exc:
            return {"error": f"Couldn't write to {folder}: {exc}"}
        return {"ok": True, "path": str(target)}

    @app.get("/api/lookup")
    def lookup(name: str, request: Request):
        check(request)
        import urllib.parse
        import urllib.request as ur

        # Archive names are case- and space-sensitive ("KELT-16 b"), and
        # people rarely type them exactly. Try an exact match, then a
        # case-insensitive one, then a prefix — the same fallbacks the
        # desktop GUI uses.
        safe = name.strip().replace("'", "''")
        cols = ("pl_name, ra, dec, sy_gaiamag, sy_vmag, pl_orbper, "
                "pl_tranmid, pl_trandur, pl_trandep")
        queries = [
            f"SELECT {cols} FROM pscomppars WHERE pl_name = '{safe}'",
            f"SELECT {cols} FROM pscomppars "
            f"WHERE UPPER(pl_name) = UPPER('{safe}')",
            f"SELECT {cols} FROM pscomppars "
            f"WHERE UPPER(pl_name) LIKE UPPER('{safe}%')",
        ]
        rows, last_err = None, None
        for adql in queries:
            url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
                   + urllib.parse.quote(adql) + "&format=json")
            try:
                req = ur.Request(url, headers={"User-Agent": "transitphot"})
                with ur.urlopen(req, timeout=60) as r:
                    rows = json.load(r)
            except Exception as exc:                     # noqa: BLE001
                last_err = exc
                continue
            if rows:
                break
        if not rows:
            if last_err is not None and rows is None:
                return {"error": f"Archive lookup failed: {last_err}"}
            return {"error": f"No archive entry for '{name}'. Names look like "
                             f"'TOI-3629 b' or 'KELT-16 b' — a space before "
                             f"the planet letter."}
        r0 = rows[0]
        f = {}
        if r0.get("ra") is not None:
            f["ra"] = repr(float(r0["ra"]))
        if r0.get("dec") is not None:
            f["dec"] = repr(float(r0["dec"]))
        mag = r0.get("sy_gaiamag") or r0.get("sy_vmag")
        if mag is not None:
            f["target_mag"] = repr(float(mag))
        if r0.get("pl_trandep") is not None:
            f["depth_ppm"] = f"{float(r0['pl_trandep']) * 10000:.0f}"
        if r0.get("pl_trandur") is not None:
            f["duration_hours"] = repr(float(r0["pl_trandur"]))
        # full precision: an epoch is ~2.46e6 with six meaningful decimals
        if r0.get("pl_tranmid") is not None:
            f["epoch_bjd"] = repr(float(r0["pl_tranmid"]))
        if r0.get("pl_orbper") is not None:
            f["period"] = repr(float(r0["pl_orbper"]))
        canonical = r0.get("pl_name") or name
        f["target_name"] = canonical
        note = f"Filled from the archive for {canonical}."
        if canonical != name.strip():
            note += f" (matched '{name.strip()}')"
        return {"fields": f, "note": note}

    @app.post("/api/start")
    async def start(request: Request):
        check(request)
        if state["proc"] and state["proc"].returncode is None:
            return {"error": "A run is already in progress."}
        v = await request.json()
        save_settings(v)

        need = ["lights_root", "ra", "dec", "target_mag"]
        missing = [k for k in need if not v.get(k)]
        if missing:
            return {"error": "Missing: " + ", ".join(missing)}

        name = v.get("target_name") or "target"
        stem = name.replace(" ", "_")
        cmd = [sys.executable, "-m", "transitphot.cli", "night",
               "--lights-root", v["lights_root"],
               "--ra", v["ra"], "--dec", v["dec"],
               "--target-mag", v["target_mag"],
               "--target-name", name,
               "--out", f"{stem}.csv", "--plot", f"{stem}.png",
               "--fit", "--fix-duration", "--model", "both"]
        for key, flag in (("source", "--source"), ("bias", "--bias"),
                          ("darks", "--darks"),
                          ("flats", "--flats"),
                          ("filter_band", "--filter"),
                          ("depth_ppm", "--depth-ppm"),
                          ("duration_hours", "--duration-hours"),
                          ("epoch_bjd", "--epoch-bjd"), ("period", "--period"),
                          ("after_idle", "--after-idle")):
            if v.get(key):
                cmd += [flag, v[key]]
        # Site values come from the page (already saved above), falling back
        # to the settings file so a partly filled page still works.
        s = load_settings()
        for key, flag in (("lat", "--lat"), ("lon", "--lon"),
                          ("elevation", "--elevation"),
                          ("aavso_obscode", "--aavso-obscode"),
                          ("aavso_filter", "--aavso-filter"),
                          ("binning", "--binning")):
            val = v.get(key) or s.get(key)
            if val:
                cmd += [flag, str(val)]

        state["plot"] = Path.cwd() / name / f"{stem}.png"
        state["queue"] = asyncio.Queue()
        q = state["queue"]
        await q.put({"line": "$ " + " ".join(cmd) + "\n"})

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1",
                 "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        state["proc"] = proc

        async def pump():
            async for raw in proc.stdout:
                await q.put({"line": raw.decode("utf-8", "replace")})
            code = await proc.wait()
            plot = state["plot"]
            await q.put({"done": True, "code": code,
                         "plot": bool(plot and plot.exists())})

        state["task"] = asyncio.create_task(pump())
        return {"ok": True}

    @app.get("/api/stream")
    async def stream(request: Request):
        check(request)

        async def gen():
            q = state["queue"]
            if q is None:
                yield "data: " + json.dumps({"done": True, "code": -1}) + "\n\n"
                return
            while True:
                msg = await q.get()
                yield "data: " + json.dumps(msg) + "\n\n"
                if msg.get("done"):
                    return

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.post("/api/stop")
    def stop(request: Request):
        check(request)
        p = state["proc"]
        if p and p.returncode is None:
            p.terminate()
        return {"ok": True}

    @app.get("/api/plot")
    def plot(request: Request):
        check(request)
        p = state["plot"]
        if not (p and Path(p).exists()):
            raise HTTPException(404, "No plot yet")
        return FileResponse(p)

    return app


def _qr_svg(url: str) -> str:
    """An inline SVG QR code, or "" if segno isn't installed."""
    try:
        import segno
    except ImportError:
        return ""
    return segno.make(url, error="m").svg_inline(scale=6, dark="#0B1220",
                                                 light="#FFFFFF")


def _qr_terminal(url: str) -> bool:
    """Print a QR code to the terminal. False if segno isn't installed."""
    try:
        import segno
    except ImportError:
        return False
    try:
        segno.make(url, error="m").terminal(compact=True)
        return True
    except Exception:                                    # noqa: BLE001
        return False


def _persistent_token(renew: bool = False) -> str:
    """
    One token kept in the settings file, so the URL is the same every time
    the server starts and a phone bookmark keeps working. --new-token
    replaces it, which revokes every URL handed out before.
    """
    s = load_settings()
    tok = s.get("serve_token")
    if renew or not tok:
        tok = secrets.token_urlsafe(12)
        save_settings({"serve_token": tok})
    return tok


def _local_addresses():
    """
    The addresses a phone might use to reach this computer.

    gethostbyname(gethostname()) is unreliable on Windows: it often returns
    a Hyper-V, VPN or Tailscale adapter rather than the Wi-Fi one. Asking the
    OS which interface it would route an outbound packet through gives the
    real LAN address (no packet is actually sent). Tailscale's 100.64.0.0/10
    addresses are reported separately, since they only work over the tailnet.
    """
    import ipaddress
    import socket

    primary = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 80))        # TEST-NET; nothing is transmitted
        primary = s.getsockname()[0]
        s.close()
    except OSError:
        pass

    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    if primary:
        found.add(primary)

    cgnat = ipaddress.ip_network("100.64.0.0/10")
    lan, tailscale, others = None, None, []
    for a in sorted(found):
        ip = ipaddress.ip_address(a)
        if ip.is_loopback or ip.is_link_local:
            continue
        if ip in cgnat:
            tailscale = a
        elif a == primary:
            lan = a
        elif ip.is_private:
            others.append(a)
    if lan is None and others:
        lan = others.pop(0)
    return lan, tailscale, others


def serve(host: str = "0.0.0.0", port: int = 8765, use_token: bool = True,
          new_token: bool = False):
    import uvicorn

    token = _persistent_token(renew=new_token) if use_token else None

    lan, tailscale, others = _local_addresses()
    q = f"?t={token}" if token else ""
    urls = []
    if lan:
        urls.append(("On your home Wi-Fi", f"http://{lan}:{port}/{q}"))
    if tailscale:
        urls.append(("Over Tailscale", f"http://{tailscale}:{port}/{q}"))
    for a in others:
        urls.append((f"Also on {a}", f"http://{a}:{port}/{q}"))

    app = build_app(token, urls)

    print("transitphot serve")
    print(f"  on this machine : http://localhost:{port}/{q}")
    for label, url in urls:
        print(f"  {label:16s}: {url}")

    print(f"\nTo open it on a phone, visit http://localhost:{port}/qr on "
          f"this computer\nand scan the code. The address stays the same "
          f"between restarts, so\nbookmark it once.")

    if urls and _qr_terminal(urls[0][1]):
        print(f"(Or scan this — {urls[0][0].lower()}.)")
    elif urls:
        print("  (pip install segno for scannable QR codes)")

    if token:
        print("\nThe token in the address is required. Run with --new-token "
              "to replace it,\nwhich stops every earlier address working.")
    print("\nThis runs processes on this computer. Keep it behind Tailscale "
          "or a home LAN;\nnever expose the port to the internet.")
    print("\nIf a phone can't reach it, allow the port through Windows "
          "Firewall (network set to Private):")
    print(f'  New-NetFirewallRule -DisplayName "transitphot serve" -Direction '
          f'Inbound -Protocol TCP -LocalPort {port} -Action Allow -Profile Private\n')
    uvicorn.run(app, host=host, port=port, log_level="warning")
