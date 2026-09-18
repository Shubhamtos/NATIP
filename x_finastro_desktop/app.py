from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from scraper import ScrapeConfig, scrape_profile


APP_NAME = "X Financial Astrology Scraper"


def app_home() -> Path:
    return Path.home() / ".x_finastro_scraper"


class ScraperApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("820x600")
        self.minsize(760, 520)

        self.msg_queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()

        default_out = Path.home() / "Documents" / "X_FinAstro_Output"

        self.username_var = tk.StringVar(value="anandrathi12")
        self.output_var = tk.StringVar(value=str(default_out))
        self.status_var = tk.StringVar(value="Ready")

        self._build()
        self.after(200, self._pump_messages)

    def _build(self):
        pad = {"padx": 14, "pady": 8}

        title = ttk.Label(self, text="X Financial Astrology Scraper", font=("Arial", 19, "bold"))
        title.pack(pady=(18, 6))

        subtitle = ttk.Label(
            self,
            text="Scrapes a public X profile until the timeline stops yielding new posts, then exports only financial-astrology matches.",
            wraplength=760,
            justify="center",
        )
        subtitle.pack(pady=(0, 12))

        form = ttk.Frame(self)
        form.pack(fill="x", **pad)

        ttk.Label(form, text="X username").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.username_var, width=34).grid(row=0, column=1, sticky="ew", padx=(12, 0))

        ttk.Label(form, text="Output folder").grid(row=1, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(form, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=(12, 8), pady=(12, 0))
        ttk.Button(form, text="Browse", command=self._browse).grid(row=1, column=2, pady=(12, 0))

        form.columnconfigure(1, weight=1)

        controls = ttk.Frame(self)
        controls.pack(fill="x", **pad)

        self.start_btn = ttk.Button(controls, text="Start Scraping", command=self._start)
        self.start_btn.pack(side="left")

        self.stop_btn = ttk.Button(controls, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)

        ttk.Button(controls, text="Open Output Folder", command=self._open_output).pack(side="right")

        note = ttk.Label(
            self,
            text=(
                "First run: a Chromium window opens. If X requests login, log in normally there. "
                "The session is saved locally and reused. The scraper checkpoints continuously, "
                "so an interrupted run can resume without losing collected posts."
            ),
            wraplength=760,
            justify="left",
        )
        note.pack(fill="x", **pad)

        ttk.Separator(self).pack(fill="x", padx=14, pady=4)

        ttk.Label(self, text="Live progress", font=("Arial", 11, "bold")).pack(anchor="w", padx=14, pady=(6, 2))

        self.log_box = tk.Text(self, height=18, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=14, pady=6)

        status = ttk.Label(self, textvariable=self.status_var)
        status.pack(anchor="w", padx=14, pady=(0, 12))

    def _browse(self):
        path = filedialog.askdirectory(initialdir=self.output_var.get() or str(Path.home()))
        if path:
            self.output_var.set(path)

    def _append_log(self, msg: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        self.status_var.set(msg)

    def _pump_messages(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "done":
                    self._finish(payload)
                elif kind == "error":
                    self._error(payload)
        except queue.Empty:
            pass
        self.after(200, self._pump_messages)

    def _start(self):
        if self.worker and self.worker.is_alive():
            return

        username = self.username_var.get().strip().lstrip("@")
        if not username:
            messagebox.showerror(APP_NAME, "Enter an X username.")
            return

        output = Path(self.output_var.get()).expanduser()
        output.mkdir(parents=True, exist_ok=True)

        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._append_log(f"Starting scrape for @{username}...")

        cfg = ScrapeConfig(
            username=username,
            output_dir=output,
            profile_dir=app_home() / "browser_profile",
        )

        self.worker = threading.Thread(target=self._run_worker, args=(cfg,), daemon=True)
        self.worker.start()

    def _run_worker(self, cfg: ScrapeConfig):
        try:
            result = scrape_profile(
                cfg,
                progress=lambda m: self.msg_queue.put(("log", m)),
                should_stop=self.stop_event.is_set,
            )
            self.msg_queue.put(("done", result))
        except Exception as exc:
            self.msg_queue.put(("error", str(exc)))

    def _stop(self):
        self.stop_event.set()
        self._append_log("Stop requested. Saving checkpoint before closing...")

    def _finish(self, result: dict):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        if result.get("cancelled"):
            self._append_log("Scrape stopped safely. Checkpoint saved; next run will resume.")
            return

        msg = (
            f"COMPLETE — {result.get('scraped', 0)} total posts scraped; "
            f"{result.get('finastro', 0)} financial-astrology posts found.\n"
            f"Excel: {result.get('excel')}"
        )
        self._append_log(msg)
        messagebox.showinfo(APP_NAME, msg)

    def _error(self, error: str):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._append_log("ERROR: " + error)
        messagebox.showerror(
            APP_NAME,
            error + "\n\nIf Chromium is not installed for Playwright, run setup.py once.",
        )

    def _open_output(self):
        path = Path(self.output_var.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)

        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", str(path)])


if __name__ == "__main__":
    ScraperApp().mainloop()
