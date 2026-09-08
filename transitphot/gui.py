"""
transitphot GUI — a small Tkinter front end.

Deliberately stdlib-only (Tkinter ships with Python on Windows), so the GUI
adds no dependencies to the package. It shells out to the CLI rather than
importing it, which keeps a long run from freezing the window and means the
GUI and the command line can never drift apart in behavior.

Launch with:  transitphot-gui
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

SETTINGS = Path.home() / ".transitphot_gui.json"

# Fields persisted between sessions. Site details rarely change; target
# details usually do, but keeping them saves retyping during a night of
# reprocessing.
FIELDS = [
    ("lights", "Lights folder", "dir"),
    ("bias", "Bias folder (optional)", "dir"),
    ("darks", "Darks folder (optional)", "dir"),
    ("flats", "Flats folder (optional)", "dir"),
    ("target_name", "Target name", "str"),
    ("ra", "RA (deg)", "float"),
    ("dec", "Dec (deg)", "float"),
    ("target_mag", "Target mag (Gaia G)", "float"),
    ("filter_band", "Filter (R, L, V...)", "str"),
    ("lat", "Site latitude (deg)", "float"),
    ("lon", "Site longitude (deg, E+)", "float"),
    ("elevation", "Site elevation (m)", "float"),
    ("depth_ppm", "Expected depth (ppm)", "float"),
    ("duration_hours", "Duration (hours)", "float"),
    ("epoch_bjd", "Epoch (BJD_TDB)", "float"),
    ("period", "Period (days)", "float"),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("transitphot")
        self.geometry("980x760")
        self.minsize(860, 620)

        self.vars: dict[str, tk.StringVar] = {}
        self.fix_duration = tk.BooleanVar(value=True)
        self.model = tk.StringVar(value="both")
        self.min_transparency = tk.StringVar(value="0.6")
        self.trim_start = tk.StringVar(value="0")
        self.trim_end = tk.StringVar(value="0")
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue[str] = queue.Queue()
        self.pending_plot: Path | None = None   # shown when the job succeeds
        self.plot_win: tk.Toplevel | None = None
        # Sync runs in its own slot: it can wait for hours, and locking out
        # calibrate/run for that whole time would make the GUI useless
        # overnight.
        self.sync_proc: subprocess.Popen | None = None
        self.sync_vars: dict[str, tk.StringVar] = {}

        self._build()
        self._load()
        self.after(100, self._drain)

    # ---------------- layout ----------------
    def _build(self):
        pad = dict(padx=6, pady=3)
        nb = ttk.Notebook(self)
        nb.pack(fill="x", padx=10, pady=(10, 4))

        # --- Sync tab ---
        f_sync = ttk.Frame(nb)
        nb.add(f_sync, text="Sync from capture device")
        for r, (key, label, default) in enumerate([
                ("source", "Capture folder (source)", ""),
                ("dest", "Local folder (destination)", "")]):
            ttk.Label(f_sync, text=label).grid(row=r, column=0, sticky="w", **pad)
            var = tk.StringVar(value=default)
            self.sync_vars[key] = var
            ttk.Entry(f_sync, textvariable=var, width=58).grid(
                row=r, column=1, columnspan=2, **pad)
            ttk.Button(f_sync, text="Browse…",
                       command=lambda k=key: self._pick_sync_dir(k)
                       ).grid(row=r, column=3, **pad)

        ttk.Label(f_sync, text="Start at (HH:MM, optional)").grid(
            row=2, column=0, sticky="w", **pad)
        self.sync_vars["start"] = tk.StringVar()
        ttk.Entry(f_sync, textvariable=self.sync_vars["start"], width=10).grid(
            row=2, column=1, sticky="w", **pad)

        ttk.Label(f_sync, text="Or start after idle (minutes)").grid(
            row=3, column=0, sticky="w", **pad)
        self.sync_vars["after_idle"] = tk.StringVar(value="15")
        ttk.Entry(f_sync, textvariable=self.sync_vars["after_idle"], width=10).grid(
            row=3, column=1, sticky="w", **pad)

        sbar = ttk.Frame(f_sync)
        sbar.grid(row=4, column=0, columnspan=4, sticky="w", pady=(10, 2))
        self.btn_sync = ttk.Button(sbar, text="Start sync", command=self._sync)
        self.btn_sync.pack(side="left", padx=6)
        ttk.Button(sbar, text="Dry run",
                   command=lambda: self._sync(dry=True)).pack(side="left", padx=6)
        self.btn_sync_stop = ttk.Button(sbar, text="Stop sync",
                                        command=self._stop_sync, state="disabled")
        self.btn_sync_stop.pack(side="left", padx=6)
        self.sync_status = ttk.Label(sbar, text="", foreground="#666")
        self.sync_status.pack(side="left", padx=12)

        ttk.Label(f_sync, foreground="#555", justify="left",
                  text="Copies after the session rather than during it: reading "
                       "large frames off the capture device\nwhile it is still "
                       "imaging competes with the camera and USB bus. Idle "
                       "detection starts the\ncopy once nothing has changed for "
                       "the given number of minutes.\n\n"
                       "The sync only runs while this window is open — use the "
                       "command line if you want to close it."
                  ).grid(row=5, column=0, columnspan=4, sticky="w", **pad)

        # --- Folders tab ---
        f1 = ttk.Frame(nb)
        nb.add(f1, text="Folders")
        for r, (key, label, _kind) in enumerate(FIELDS[:4]):
            ttk.Label(f1, text=label).grid(row=r, column=0, sticky="w", **pad)
            v = tk.StringVar()
            self.vars[key] = v
            ttk.Entry(f1, textvariable=v, width=68).grid(row=r, column=1, **pad)
            ttk.Button(f1, text="Browse…",
                       command=lambda k=key: self._pick_dir(k)
                       ).grid(row=r, column=2, **pad)
        ttk.Label(f1, foreground="#555",
                  text="Leave bias/darks/flats blank to skip them. "
                       "Calibrated frames are written to <lights>/calibrated."
                  ).grid(row=4, column=0, columnspan=3, sticky="w", **pad)

        # --- Target tab ---
        # Layout follows the order of work: name the target, look it up, and
        # everything the archive knows fills in. Site details live in their
        # own column because they are properties of the observer, not the
        # target, and change on a completely different timescale.
        f2 = ttk.Frame(nb)
        nb.add(f2, text="Target && site")

        ttk.Label(f2, text="Target name",
                  font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=0, sticky="w", **pad)
        v = tk.StringVar()
        self.vars["target_name"] = v
        e = ttk.Entry(f2, textvariable=v, width=30)
        e.grid(row=0, column=1, columnspan=2, sticky="w", **pad)
        e.bind("<Return>", lambda _ev: self._lookup())

        ttk.Button(f2, text="Look up target in NASA archive",
                   command=self._lookup).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(f2, foreground="#555", justify="left",
                  text=("Fills RA, Dec, magnitude, depth, duration, epoch and"
                        " period below.\nNames look like 'Kepler-17 b'.")
                  ).grid(row=1, column=2, columnspan=2, sticky="w", **pad)

        ttk.Separator(f2, orient="horizontal").grid(
            row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=(10, 6))

        ttk.Label(f2, text="Target", font=("TkDefaultFont", 9, "bold")).grid(
            row=3, column=0, sticky="w", **pad)
        target_fields = [("ra", "RA (deg)"), ("dec", "Dec (deg)"),
                         ("target_mag", "Target mag (Gaia G)"),
                         ("depth_ppm", "Expected depth (ppm)"),
                         ("duration_hours", "Duration (hours)"),
                         ("epoch_bjd", "Epoch (BJD_TDB)"),
                         ("period", "Period (days)")]
        for r, (key, label) in enumerate(target_fields, start=4):
            ttk.Label(f2, text=label).grid(row=r, column=0, sticky="w", **pad)
            var = tk.StringVar()
            self.vars[key] = var
            ttk.Entry(f2, textvariable=var, width=22).grid(
                row=r, column=1, sticky="w", **pad)

        ttk.Label(f2, text="Site && instrument",
                  font=("TkDefaultFont", 9, "bold")).grid(
            row=3, column=2, sticky="w", **pad)
        site_fields = [("lat", "Latitude (deg)"),
                       ("lon", "Longitude (deg, E+)"),
                       ("elevation", "Elevation (m)"),
                       ("filter_band", "Filter (R, L, V...)")]
        for r, (key, label) in enumerate(site_fields, start=4):
            ttk.Label(f2, text=label).grid(row=r, column=2, sticky="w", **pad)
            var = tk.StringVar()
            self.vars[key] = var
            ttk.Entry(f2, textvariable=var, width=22).grid(
                row=r, column=3, sticky="w", **pad)
        ttk.Label(f2, foreground="#555", justify="left",
                  text="Site values persist between sessions - enter them once."
                  ).grid(row=8, column=2, columnspan=2, sticky="w", **pad)

        # --- Options tab ---
        f3 = ttk.Frame(nb)
        nb.add(f3, text="Options")
        ttk.Label(f3, text="Transit model",
                  font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=0, sticky="w", **pad)
        mrow = ttk.Frame(f3)
        mrow.grid(row=0, column=1, columnspan=3, sticky="w", **pad)
        for val, label in (("trapezoid", "Trapezoid"),
                           ("ld", "Limb-darkened"),
                           ("both", "Both (compare)")):
            ttk.Radiobutton(mrow, text=label, value=val,
                            variable=self.model).pack(side="left", padx=(0, 14))
        ttk.Label(f3, foreground="#555", justify="left",
                  text="The trapezoid is robust but reads depths about 10-15% "
                       "low — its flat bottom sits above\nthe true centre of a "
                       "limb-darkened profile. The limb-darkened model gets "
                       "depth right;\n\"Both\" fits the same data twice so you "
                       "can compare. Needs a period to be set."
                  ).grid(row=1, column=0, columnspan=4, sticky="w", **pad)

        ttk.Separator(f3, orient="horizontal").grid(
            row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=(8, 6))

        ttk.Checkbutton(f3, text="Hold duration at the archive value "
                                 "(recommended)",
                        variable=self.fix_duration).grid(
            row=3, column=0, columnspan=2, sticky="w", **pad)
        for r, (label, var) in enumerate([
                ("Minimum transparency (0-1)", self.min_transparency),
                ("Trim from start (minutes)", self.trim_start),
                ("Trim from end (minutes)", self.trim_end)], start=4):
            ttk.Label(f3, text=label).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(f3, textvariable=var, width=10).grid(
                row=r, column=1, sticky="w", **pad)
        ttk.Label(f3, foreground="#555",
                  text="Free duration is usually a mistake on ground-based "
                       "data: the fit absorbs baseline curvature by\n"
                       "stretching the transit. Only release it with a long, "
                       "flat baseline on both sides."
                  ).grid(row=7, column=0, columnspan=3, sticky="w", **pad)

        # --- action buttons ---
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=10, pady=4)
        self.btn_cal = ttk.Button(bar, text="1. Calibrate", command=self._calibrate)
        self.btn_chk = ttk.Button(bar, text="2. Check WCS", command=self._check)
        self.btn_run = ttk.Button(bar, text="3. Run photometry", command=self._run)
        for b in (self.btn_cal, self.btn_chk, self.btn_run):
            b.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(bar, text="Stop", command=self._stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        ttk.Button(bar, text="Open results folder",
                   command=self._open_results).pack(side="right", padx=4)

        self.status = ttk.Label(self, text="Ready", foreground="#333")
        self.status.pack(fill="x", padx=12)

        # --- log ---
        wrap = ttk.Frame(self)
        wrap.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.log = tk.Text(wrap, wrap="word", height=20, bg="#0e1626",
                           fg="#e7e9f0", insertbackground="#e7e9f0")
        sb = ttk.Scrollbar(wrap, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # ---------------- helpers ----------------
    def _pick_dir(self, key):
        d = filedialog.askdirectory(title=f"Select {key} folder")
        if d:
            self.vars[key].set(d)

    def _say(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def _calibrated_dir(self) -> str:
        lights = self.vars["lights"].get().strip()
        cal = Path(lights) / "calibrated"
        return str(cal if cal.exists() else lights)

    def _results_dir(self) -> Path:
        name = self.vars["target_name"].get().strip() or "results"
        return Path.cwd() / name

    def _open_results(self):
        d = self._results_dir()
        if not d.exists():
            messagebox.showinfo("transitphot", f"No results yet in {d}")
            return
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", str(d)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(d)])
        else:
            subprocess.Popen(["xdg-open", str(d)])

    # ---------------- archive lookup ----------------
    def _lookup(self):
        name = self.vars["target_name"].get().strip()
        if not name:
            messagebox.showwarning("transitphot", "Enter a target name first.")
            return
        self._say(f"\nLooking up {name} in the NASA Exoplanet Archive…\n")
        threading.Thread(target=self._lookup_worker, args=(name,),
                         daemon=True).start()

    def _lookup_worker(self, name):
        """
        Query the NASA Exoplanet Archive TAP service.

        The service is often slow — 30 seconds is not unusual for a cold
        query — so the timeout is generous and failures explain what to do
        rather than just reporting the exception. Runs on a worker thread so
        the window stays responsive.
        """
        import urllib.error
        import urllib.parse
        import urllib.request

        wanted = ("pl_name, ra, dec, sy_vmag, sy_gaiamag, pl_orbper, "
                  "pl_tranmid, pl_trandur, pl_trandep")
        # Try the exact name first, then a case-insensitive match, then a
        # prefix match — archive names carry spaces and capitalisation that
        # are easy to get slightly wrong ("TrES-5b" vs "TrES-5 b").
        safe = name.replace("'", "''")
        queries = [
            f"SELECT {wanted} FROM pscomppars WHERE pl_name = '{safe}'",
            f"SELECT {wanted} FROM pscomppars "
            f"WHERE UPPER(pl_name) = UPPER('{safe}')",
            f"SELECT {wanted} FROM pscomppars "
            f"WHERE UPPER(pl_name) LIKE UPPER('{safe}%')",
        ]

        rows = None
        last_err = None
        for attempt, adql in enumerate(queries, 1):
            url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
                   + urllib.parse.quote(adql) + "&format=json")
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "transitphot"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    rows = json.load(r)
            except Exception as exc:                    # noqa: BLE001
                last_err = exc
                self.q.put(f"  attempt {attempt} failed ({exc})\n")
                continue
            if rows:
                break

        if not rows:
            if last_err is not None:
                self.q.put(
                    "Lookup failed. The archive service is often slow or "
                    "briefly unavailable — try again in a moment, or enter "
                    "the values by hand from\n"
                    "  https://exoplanetarchive.ipac.caltech.edu/\n")
            else:
                self.q.put(
                    f"No archive entry matching '{name}'. Names carry a space "
                    f"before the planet letter, e.g. 'TrES-5 b', "
                    f"'Kepler-17 b', 'WASP-10 b'.\n")
            return

        r0 = rows[0]

        def put(key, val, fmt="{:.6g}"):
            if val is not None:
                self.vars[key].set(fmt.format(val) if isinstance(val, float)
                                   else str(val))
        put("ra", r0.get("ra"))
        put("dec", r0.get("dec"))
        put("target_mag", r0.get("sy_gaiamag") or r0.get("sy_vmag"))
        put("period", r0.get("pl_orbper"), "{:.9g}")
        put("epoch_bjd", r0.get("pl_tranmid"), "{:.9g}")
        put("duration_hours", r0.get("pl_trandur"))
        if r0.get("pl_trandep") is not None:
            self.vars["depth_ppm"].set(f"{r0['pl_trandep'] * 10000:.0f}")
        if r0.get("pl_name") and r0["pl_name"] != name:
            self.vars["target_name"].set(r0["pl_name"])
        self.q.put(f"Filled parameters for {r0['pl_name']}.\n")

    # ---------------- running ----------------
    def _base_cmd(self, sub):
        return [sys.executable, "-m", "transitphot.cli", sub]

    def _calibrate(self):
        lights = self.vars["lights"].get().strip()
        if not lights:
            messagebox.showwarning("transitphot", "Choose a lights folder.")
            return
        cmd = self._base_cmd("calibrate") + ["--lights", lights]
        for key, flag in (("bias", "--bias"), ("darks", "--darks"),
                          ("flats", "--flats")):
            v = self.vars[key].get().strip()
            if v:
                cmd += [flag, v]
        self.pending_plot = None
        self._launch(cmd, "Calibrating…")

    def _check(self):
        self.pending_plot = None
        self._launch(self._base_cmd("check") + ["--lights", self._calibrated_dir()],
                     "Checking WCS…")

    def _run(self):
        need = ["lights", "ra", "dec", "target_mag"]
        missing = [k for k in need if not self.vars[k].get().strip()]
        if missing:
            messagebox.showwarning(
                "transitphot", "Missing: " + ", ".join(missing))
            return
        v = {k: var.get().strip() for k, var in self.vars.items()}
        name = v["target_name"] or "target"
        cmd = self._base_cmd("run") + [
            "--lights", self._calibrated_dir(),
            "--ra", v["ra"], "--dec", v["dec"],
            "--target-mag", v["target_mag"],
            "--target-name", name,
            "--out", f"{name.replace(' ', '_')}.csv",
            "--plot", f"{name.replace(' ', '_')}.png",
            "--fit",
        ]
        for key, flag in (("filter_band", "--filter"), ("lat", "--lat"),
                          ("lon", "--lon"), ("elevation", "--elevation"),
                          ("depth_ppm", "--depth-ppm"),
                          ("duration_hours", "--duration-hours"),
                          ("epoch_bjd", "--epoch-bjd"), ("period", "--period")):
            if v.get(key):
                cmd += [flag, v[key]]
        cmd += ["--model", self.model.get()]
        if self.fix_duration.get():
            cmd.append("--fix-duration")
        if self.min_transparency.get().strip():
            cmd += ["--min-transparency", self.min_transparency.get().strip()]
        for var, flag in ((self.trim_start, "--trim-start"),
                          (self.trim_end, "--trim-end")):
            val = var.get().strip()
            if val and float(val) > 0:
                cmd += [flag, val]
        self._save()
        self.pending_plot = (self._results_dir()
                             / f"{name.replace(' ', '_')}.png")
        self._launch(cmd, "Running photometry…")

    def _launch(self, cmd, status):
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("transitphot", "A job is already running.")
            return
        self._say("\n$ " + " ".join(cmd) + "\n")
        self.status.config(text=status)
        for b in (self.btn_cal, self.btn_chk, self.btn_run):
            b.config(state="disabled")
        self.btn_stop.config(state="normal")
        threading.Thread(target=self._worker, args=(cmd,), daemon=True).start()

    def _worker(self, cmd):
        try:
            # Force UTF-8 in the child and decode as UTF-8 here. Without
            # this, Windows hands the subprocess a cp1252 stdout and any
            # non-Latin-1 character in the output kills the run.
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
                env=env,
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform.startswith("win") else 0),
            )
            for line in self.proc.stdout:
                self.q.put(line)
            code = self.proc.wait()
            self.q.put(f"\n[finished, exit code {code}]\n")
            self.q.put(f"__CODE__{code}")
        except Exception as exc:                        # noqa: BLE001
            self.q.put(f"\n[error: {exc}]\n")
        finally:
            self.q.put("__DONE__")

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self._say("\n[stopped]\n")

    def _drain(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item == "__SYNCDONE__":
                    self.btn_sync.config(state="normal")
                    self.btn_sync_stop.config(state="disabled")
                    if self.sync_status.cget("text") != "Stopped":
                        self.sync_status.config(text="Done", foreground="#2e7d4f")
                elif item.startswith("__CODE__"):
                    self.last_code = int(item[len("__CODE__"):])
                elif item == "__DONE__":
                    self.status.config(text="Ready")
                    for b in (self.btn_cal, self.btn_chk, self.btn_run):
                        b.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    # Show the light curve as soon as it exists — the plot is
                    # the point of the run, and hunting for it in a folder
                    # afterwards is friction at 2am.
                    if (getattr(self, "last_code", 1) == 0
                            and self.pending_plot
                            and self.pending_plot.exists()):
                        self._show_plot(self.pending_plot)
                    self.pending_plot = None
                else:
                    self._say(item)
        except queue.Empty:
            pass
        self.after(100, self._drain)

    # ---------------- sync ----------------
    def _pick_sync_dir(self, key):
        d = filedialog.askdirectory(title=f"Select {key} folder")
        if d:
            self.sync_vars[key].set(d)

    def _sync(self, dry: bool = False):
        if self.sync_proc and self.sync_proc.poll() is None:
            messagebox.showinfo("transitphot", "A sync is already running.")
            return
        src = self.sync_vars["source"].get().strip()
        dst = self.sync_vars["dest"].get().strip()
        if not src or not dst:
            messagebox.showwarning(
                "transitphot", "Choose both a source and a destination folder.")
            return

        cmd = self._base_cmd("sync") + ["--source", src, "--dest", dst]
        start = self.sync_vars["start"].get().strip()
        idle = self.sync_vars["after_idle"].get().strip()
        if start:
            cmd += ["--start", start]
        if idle and float(idle) > 0:
            cmd += ["--after-idle", idle]
        if dry:
            cmd.append("--dry-run")

        self._say("\n$ " + " ".join(cmd) + "\n")
        self.sync_status.config(text="Dry run…" if dry else "Waiting…",
                                foreground="#b07d2b")
        self.btn_sync.config(state="disabled")
        self.btn_sync_stop.config(state="normal")
        self._save()
        threading.Thread(target=self._sync_worker, args=(cmd,),
                         daemon=True).start()

    def _sync_worker(self, cmd):
        try:
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            self.sync_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
                env=env,
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform.startswith("win") else 0),
            )
            for line in self.sync_proc.stdout:
                self.q.put("[sync] " + line)
            code = self.sync_proc.wait()
            self.q.put(f"[sync] finished, exit code {code}\n")
        except Exception as exc:                        # noqa: BLE001
            self.q.put(f"[sync] error: {exc}\n")
        finally:
            self.q.put("__SYNCDONE__")

    def _stop_sync(self):
        if self.sync_proc and self.sync_proc.poll() is None:
            self.sync_proc.terminate()
            self._say("[sync] stopped by user — files already copied are "
                      "complete and will be skipped on a later run\n")
        self.sync_status.config(text="Stopped", foreground="#666")

    # ---------------- plot viewer ----------------
    def _show_plot(self, path: Path):
        """
        Display the finished light curve in a window.

        Tkinter reads GIF/PNG via PhotoImage on Python 3.13+, but older
        builds and some PNG flavours fail — so fall back to the system image
        viewer rather than showing an error. Either way the astronomer sees
        their curve without going looking for it.
        """
        try:
            img = tk.PhotoImage(file=str(path))
        except Exception:                               # noqa: BLE001
            self._open_path(path)
            return

        # Downscale to fit the screen; PhotoImage only does integer factors.
        sw, sh = self.winfo_screenwidth() - 120, self.winfo_screenheight() - 200
        factor = 1
        while (img.width() // factor > sw or img.height() // factor > sh) \
                and factor < 6:
            factor += 1
        if factor > 1:
            img = img.subsample(factor, factor)

        if self.plot_win is not None and self.plot_win.winfo_exists():
            self.plot_win.destroy()
        win = tk.Toplevel(self)
        self.plot_win = win
        win.title(path.name)
        lbl = tk.Label(win, image=img, bd=0)
        lbl.image = img                                 # keep a reference
        lbl.pack()
        bar = ttk.Frame(win)
        bar.pack(fill="x", pady=4)
        ttk.Button(bar, text="Open in image viewer",
                   command=lambda: self._open_path(path)).pack(side="left", padx=6)
        ttk.Button(bar, text="Open results folder",
                   command=self._open_results).pack(side="left", padx=6)
        ttk.Button(bar, text="Close", command=win.destroy).pack(side="right", padx=6)
        self._say(f"\nOpened {path.name}\n")

    def _open_path(self, path: Path):
        if sys.platform.startswith("win"):
            import os as _os
            _os.startfile(str(path))                    # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    # ---------------- settings ----------------
    def _save(self):
        data = {k: v.get() for k, v in self.vars.items()}
        data.update({f"sync_{k}": v.get() for k, v in self.sync_vars.items()})
        data.update(model=self.model.get(),
                    fix_duration=self.fix_duration.get(),
                    min_transparency=self.min_transparency.get(),
                    trim_start=self.trim_start.get(),
                    trim_end=self.trim_end.get())
        try:
            SETTINGS.write_text(json.dumps(data, indent=2))
        except Exception:                               # noqa: BLE001
            pass

    def _load(self):
        if not SETTINGS.exists():
            return
        try:
            data = json.loads(SETTINGS.read_text())
        except Exception:                               # noqa: BLE001
            return
        for k, v in self.vars.items():
            if k in data:
                v.set(data[k])
        for k, v in self.sync_vars.items():
            if f"sync_{k}" in data:
                v.set(data[f"sync_{k}"])
        self.model.set(data.get("model", "both"))
        self.fix_duration.set(data.get("fix_duration", True))
        self.min_transparency.set(data.get("min_transparency", "0.6"))
        self.trim_start.set(data.get("trim_start", "0"))
        self.trim_end.set(data.get("trim_end", "0"))

    def destroy(self):
        for p in (self.proc, self.sync_proc):
            if p and p.poll() is None:
                p.terminate()
        self._save()
        super().destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
