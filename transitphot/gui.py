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
        self.min_transparency = tk.StringVar(value="0.6")
        self.trim_start = tk.StringVar(value="0")
        self.trim_end = tk.StringVar(value="0")
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue[str] = queue.Queue()

        self._build()
        self._load()
        self.after(100, self._drain)

    # ---------------- layout ----------------
    def _build(self):
        pad = dict(padx=6, pady=3)
        nb = ttk.Notebook(self)
        nb.pack(fill="x", padx=10, pady=(10, 4))

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
        f2 = ttk.Frame(nb)
        nb.add(f2, text="Target && site")
        for r, (key, label, _kind) in enumerate(FIELDS[4:]):
            col = 0 if r < 6 else 2
            row = r if r < 6 else r - 6
            ttk.Label(f2, text=label).grid(row=row, column=col, sticky="w", **pad)
            v = tk.StringVar()
            self.vars[key] = v
            ttk.Entry(f2, textvariable=v, width=22).grid(
                row=row, column=col + 1, sticky="w", **pad)
        ttk.Button(f2, text="Look up target in NASA archive",
                   command=self._lookup).grid(row=6, column=0, columnspan=2,
                                              sticky="w", **pad)
        ttk.Label(f2, foreground="#555",
                  text="Fills RA, Dec, magnitude, depth, duration, epoch and "
                       "period from the target name."
                  ).grid(row=6, column=2, columnspan=2, sticky="w", **pad)

        # --- Options tab ---
        f3 = ttk.Frame(nb)
        nb.add(f3, text="Options")
        ttk.Checkbutton(f3, text="Hold duration at the archive value "
                                 "(recommended)",
                        variable=self.fix_duration).grid(
            row=0, column=0, columnspan=2, sticky="w", **pad)
        for r, (label, var) in enumerate([
                ("Minimum transparency (0-1)", self.min_transparency),
                ("Trim from start (minutes)", self.trim_start),
                ("Trim from end (minutes)", self.trim_end)], start=1):
            ttk.Label(f3, text=label).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(f3, textvariable=var, width=10).grid(
                row=r, column=1, sticky="w", **pad)
        ttk.Label(f3, foreground="#555",
                  text="Free duration is usually a mistake on ground-based "
                       "data: the fit absorbs baseline curvature by\n"
                       "stretching the transit. Only release it with a long, "
                       "flat baseline on both sides."
                  ).grid(row=4, column=0, columnspan=3, sticky="w", **pad)

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
        import urllib.parse
        import urllib.request
        adql = (
            "SELECT pl_name, ra, dec, sy_vmag, sy_gaiamag, pl_orbper, "
            "pl_tranmid, pl_trandur, pl_trandep FROM pscomppars "
            f"WHERE pl_name = '{name}'"
        )
        url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
               + urllib.parse.quote(adql) + "&format=json")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                rows = json.load(r)
        except Exception as exc:                        # noqa: BLE001
            self.q.put(f"Lookup failed: {exc}\n")
            return
        if not rows:
            self.q.put(f"No archive entry for '{name}'. Names look like "
                       f"'Kepler-17 b' or 'TrES-3 b' (note the space).\n")
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
        self._launch(cmd, "Calibrating…")

    def _check(self):
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
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform.startswith("win") else 0),
            )
            for line in self.proc.stdout:
                self.q.put(line)
            code = self.proc.wait()
            self.q.put(f"\n[finished, exit code {code}]\n")
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
                if item == "__DONE__":
                    self.status.config(text="Ready")
                    for b in (self.btn_cal, self.btn_chk, self.btn_run):
                        b.config(state="normal")
                    self.btn_stop.config(state="disabled")
                else:
                    self._say(item)
        except queue.Empty:
            pass
        self.after(100, self._drain)

    # ---------------- settings ----------------
    def _save(self):
        data = {k: v.get() for k, v in self.vars.items()}
        data.update(fix_duration=self.fix_duration.get(),
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
        self.fix_duration.set(data.get("fix_duration", True))
        self.min_transparency.set(data.get("min_transparency", "0.6"))
        self.trim_start.set(data.get("trim_start", "0"))
        self.trim_end.set(data.get("trim_end", "0"))

    def destroy(self):
        self._save()
        super().destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
