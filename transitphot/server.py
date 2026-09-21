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

from __future__ import annotations

import asyncio
import json
import secrets
import subprocess
import sys
from pathlib import Path

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

<fieldset><legend>Folders</legend>
  <label for="source">Capture folder (leave blank if already copied)</label>
  <input id="source" placeholder="\\\\MELE-PC\\n.i.n.a\\2026-09-16\\TARGET\\LIGHT">
  <label for="lights_root">Local folder for the lights</label>
  <input id="lights_root" placeholder="D:\\xfer\\TARGET">
  <div class="row">
    <div><label for="darks">Darks</label><input id="darks"></div>
    <div><label for="flats">Flats</label><input id="flats"></div>
  </div>
</fieldset>

<fieldset><legend>Target</legend>
  <label for="target_name">Name</label>
  <input id="target_name" placeholder="TOI-3629 b">
  <button type="button" onclick="lookup()">Look up in NASA archive</button>
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
const F = ["source","lights_root","darks","flats","target_name","ra","dec",
           "target_mag","depth_ppm","duration_hours","filter_band","epoch_bjd",
           "period","after_idle"];
const g = id => document.getElementById(id);
const say = t => { const l = g("log"); l.textContent += t;
                   l.scrollTop = l.scrollHeight; };

async function boot() {
  const r = await fetch("api/settings");
  const s = await r.json();
  F.forEach(k => { if (s[k] != null && s[k] !== "") g(k).value = s[k]; });
}
boot();

function values() {
  const o = {}; F.forEach(k => o[k] = g(k).value.trim()); return o;
}

async function lookup() {
  const name = g("target_name").value.trim();
  if (!name) return;
  g("status").textContent = "Looking up " + name + "…";
  try {
    const r = await fetch("api/lookup?name=" + encodeURIComponent(name));
    const d = await r.json();
    if (d.error) { g("status").textContent = d.error; return; }
    for (const [k, v] of Object.entries(d.fields || {}))
      if (g(k)) g(k).value = v;
    g("status").textContent = d.note || "Filled from the archive.";
  } catch (e) { g("status").textContent = "Lookup failed: " + e.message; }
}

let es = null;
async function start() {
  g("log").textContent = ""; g("plot").style.display = "none";
  g("start").disabled = true; g("stop").disabled = false;
  g("status").textContent = "Running…";
  const r = await fetch("api/start", {method:"POST",
    headers:{"Content-Type":"application/json"}, body:JSON.stringify(values())});
  const d = await r.json();
  if (d.error) { g("status").textContent = d.error; done(); return; }
  es = new EventSource("api/stream");
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.line != null) say(m.line);
    if (m.done) {
      g("status").textContent = m.code === 0 ? "Finished." :
                                "Exited with code " + m.code;
      if (m.plot) { const i = g("plot");
        i.src = "api/plot?t=" + Date.now(); i.style.display = "block"; }
      done();
    }
  };
  es.onerror = () => { g("status").textContent = "Connection lost."; done(); };
}
function done() { if (es) { es.close(); es = null; }
  g("start").disabled = false; g("stop").disabled = true; }
async function stop() { await fetch("api/stop", {method:"POST"});
  g("status").textContent = "Stopping…"; }
</script></body></html>"""


def build_app(token: str | None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from fastapi.responses import FileResponse

    app = FastAPI(title="transitphot serve")
    state = {"proc": None, "queue": None, "plot": None, "task": None}

    def check(request: Request):
        if token and request.query_params.get("t") != token \
                and request.headers.get("x-token") != token:
            raise HTTPException(401, "Missing or wrong token")

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

    @app.get("/api/lookup")
    def lookup(name: str, request: Request):
        check(request)
        import urllib.parse
        import urllib.request as ur
        adql = ("SELECT pl_name, ra, dec, sy_gaiamag, sy_vmag, pl_orbper, "
                "pl_tranmid, pl_trandur, pl_trandep FROM pscomppars "
                f"WHERE pl_name = '{name.replace(chr(39), chr(39) * 2)}'")
        url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
               + urllib.parse.quote(adql) + "&format=json")
        try:
            req = ur.Request(url, headers={"User-Agent": "transitphot"})
            with ur.urlopen(req, timeout=60) as r:
                rows = json.load(r)
        except Exception as exc:                         # noqa: BLE001
            return {"error": f"Archive lookup failed: {exc}"}
        if not rows:
            return {"error": f"No archive entry for '{name}'. "
                             f"Names look like 'TOI-3629 b'."}
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
        return {"fields": f, "note": f"Filled from the archive for "
                                     f"{r0['pl_name']}."}

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
        for key, flag in (("source", "--source"), ("darks", "--darks"),
                          ("flats", "--flats"),
                          ("filter_band", "--filter"),
                          ("depth_ppm", "--depth-ppm"),
                          ("duration_hours", "--duration-hours"),
                          ("epoch_bjd", "--epoch-bjd"), ("period", "--period"),
                          ("after_idle", "--after-idle")):
            if v.get(key):
                cmd += [flag, v[key]]
        s = load_settings()
        for key, flag in (("lat", "--lat"), ("lon", "--lon"),
                          ("elevation", "--elevation"),
                          ("aavso_obscode", "--aavso-obscode"),
                          ("aavso_filter", "--aavso-filter"),
                          ("binning", "--binning")):
            if s.get(key):
                cmd += [flag, str(s[key])]

        state["plot"] = Path.cwd() / name / f"{stem}.png"
        state["queue"] = asyncio.Queue()
        q = state["queue"]
        await q.put({"line": "$ " + " ".join(cmd) + "\n"})

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**__import__("os").environ, "PYTHONUNBUFFERED": "1",
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


def serve(host: str = "0.0.0.0", port: int = 8765, use_token: bool = True):
    import uvicorn

    token = secrets.token_urlsafe(8) if use_token else None
    app = build_app(token)

    import socket
    try:
        lan = socket.gethostbyname(socket.gethostname())
    except Exception:                                    # noqa: BLE001
        lan = "your-computer"

    q = f"?t={token}" if token else ""
    print("transitphot serve")
    print(f"  on this machine : http://localhost:{port}/{q}")
    print(f"  on the LAN      : http://{lan}:{port}/{q}")
    print(f"  over Tailscale  : http://<tailscale-name>:{port}/{q}")
    if token:
        print("\nThe token in the URL is required. It changes each start;")
        print("pass --no-token to disable it on a trusted network.")
    print("\nThis runs processes on this computer. Keep it behind Tailscale")
    print("or a home LAN — never expose the port to the internet.\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
