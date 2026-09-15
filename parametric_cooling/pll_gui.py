"""GUI for the Red Pitaya dual-PLL parametric-cooling design (pll_control.py).

A PyQt5 + pyqtgraph front-end exposing every parameter of the two FPGA PLLs,
plus live visualisation that the original ETH GUI never had:

  * Lock monitor   -- rolling plots of each PLL's tracked NCO frequency and
                      loop error vs. time (is it locked? is it drifting?), with
                      a live output-vector display of the (a, phi) drive.
  * Loop design    -- instant Bode plots of the phase-detector low-pass filter
                      (alpha/order) and the PI loop filter (kp/ki) so you can
                      tune the loop without trial-and-error on hardware.

Multiple Red Pitayas
--------------------
The window holds one **tab per board** (``BoardWidget``), each with its own
connection, parameters, plots and -- crucially -- its own SSH worker thread, so
a slow or stalled board never blocks the others or the UI. Add boards from the
toolbar; close a board with its tab's close button.

Why this doesn't freeze
-----------------------
ALL SSH/hardware access for a board runs on a dedicated worker thread
(``SSHWorker`` in its own ``QThread``); the Qt event loop is never blocked on
the network. Monitor polling is *self-paced* -- the next poll is only scheduled
after the previous result arrives -- so a slow link degrades the refresh rate
instead of backing up a queue and hanging. The driver keeps a local register
mirror and pipes all commands through one persistent remote shell, so each
edit is a single round-trip write (no per-command SSH channel + shell fork on
the board). Connect/exec timeouts mean a dead board errors out quickly rather
than locking the UI.

Run:
    python parametric_cooling/pll_gui.py --host 192.168.1.9
    python parametric_cooling/pll_gui.py --host 192.168.1.9 --host 192.168.1.10
    python parametric_cooling/pll_gui.py --mock      # no hardware
"""

import os
import sys
import time
import argparse
from collections import deque

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox,
    QDoubleSpinBox, QSpinBox, QMessageBox, QSplitter, QFrame, QTabWidget,
    QScrollArea, QFileDialog, QInputDialog, QToolBar, QAction, QProgressBar,
    QFormLayout,
)
from PyQt5.QtCore import Qt, QThread, QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor
import pyqtgraph as pg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pll_control as P  # noqa: E402

_PLL_COLORS = ["#00e5ff", "#feca57"]   # per-PLL trace colour
_HISTORY = 600                         # samples kept in rolling plots
_POLL_MS = 120                         # nominal monitor poll period
_REHEAT_POLL_MS = 1000                 # reheat-burst progress poll period


# ─────────────────────────────────────────────────────────────────────────────
# Mock backend (offline GUI testing; mirrors RedPitayaPLL's interface)
# ─────────────────────────────────────────────────────────────────────────────
class MockPLL:
    def __init__(self, hostname="mock", n_pll=2, **kw):
        self.hostname = hostname
        self.n_pll = n_pll
        self._t0 = time.time()
        self._g = {"output_1": "PLL1", "output_2": "PLL2",
                   "ext_pins_p": 0, "ext_pins_n": 0}
        self._p = [dict(P.COOLING_PRESET) for _ in range(n_pll)]
        for i, d in enumerate(self._p):
            d["f0"] = 130e3 + i * 20e3
        self._burst = None
        print("MockPLL: synthetic backend (no hardware).")

    # connection no-ops
    def close(self):
        print("[mock] closed")

    @property
    def connected(self):
        return True

    def upload_bitstream(self, *a, **k):
        print("[mock] upload+load bitstream")

    def sync_all(self):
        pass

    def get_param(self, name, pll=0):
        if name in P.GLOBAL_PARAM_KEYS:
            return self._g[name]
        return self._p[pll][name]

    def set_param(self, name, value, pll=0):
        if name in P.GLOBAL_PARAM_KEYS:
            self._g[name] = value if name.startswith("output") else int(value)
        else:
            self._p[pll][name] = (float(value) if name not in
                                  ("2nd_harm", "pid_en", "order")
                                  else int(float(value)))

    def all_params(self):
        return {"global": dict(self._g),
                "pll": [dict(d) for d in self._p]}

    def apply_cooling_preset(self, pll=0, preset=None):
        self._p[pll] = dict(preset or P.COOLING_PRESET)

    def save_params(self, fn):
        P.RedPitayaPLL.save_params(self, fn)

    def load_params(self, fn):
        P.RedPitayaPLL.load_params(self, fn)

    def read_monitors(self, plls=None):
        plls = range(self.n_pll) if plls is None else plls
        t = time.time() - self._t0
        out = {}
        for pll in plls:
            d = self._p[pll]
            harm = int(d["2nd_harm"])
            carrier = d["f0"] * (1 + harm)
            locked = int(d["pid_en"])
            # synthetic: small drift + noise; error decays when locked
            drift = (1 - locked) * 80 * np.sin(2 * np.pi * 0.2 * t + pll)
            freq = carrier + drift + np.random.randn() * 5
            err = (drift if not locked else
                   12 * np.exp(-t / 4) * np.sin(2 * np.pi * 1.5 * t)) \
                + np.random.randn() * 0.4
            out[pll] = {"nco_word": int(freq / P.NCO_HZ_PER_LSB),
                        "nco_freq_hz": freq, "err_counts": err * 40,
                        "err_deg": err}
        return out

    # reheating bursts -- replayed from the wall clock, no timing loop
    def start_reheat(self, t_off, t_on, cycles, **kw):
        self._burst = dict(t_off=float(t_off), t_on=float(t_on),
                           cycles=int(cycles), plls=kw.get("plls") or [0],
                           marker_bit=kw.get("marker_bit", 0),
                           settle=float(kw.get("settle", 1.0)), t0=time.time())
        self._burst["duration_s"] = (self._burst["settle"] +
                                     cycles * (t_on + t_off))
        print(f"[mock] reheat burst: {cycles} cycles")
        return dict(self._burst)

    def _mock_cycles_done(self):
        b = self._burst
        if b is None:
            return 0
        elapsed = time.time() - b["t0"] - b["settle"]
        period = b["t_on"] + b["t_off"]
        return int(max(0, min(b["cycles"], elapsed // period)))

    def reheat_running(self):
        return (self._burst is not None and
                self._mock_cycles_done() < self._burst["cycles"])

    def reheat_progress(self):
        done = self._mock_cycles_done()
        total = self._burst["cycles"] if self._burst else 0
        return {"running": self.reheat_running(), "cycles_done": done,
                "cycles": total, "fraction": done / total if total else 0.0}

    def stop_reheat(self, timeout=5.0):
        if self._burst is not None:
            self._burst["cycles"] = self._mock_cycles_done()
        print("[mock] reheat stopped")
        return True

    def fetch_reheat_log(self, local_path=None):
        b = self._burst or {"t_off": 0.02, "t_on": 0.5, "cycles": 0,
                            "settle": 1.0}
        n = self._mock_cycles_done()
        cycle = np.arange(n)
        period = b["t_on"] + b["t_off"]
        t_off = b["settle"] + cycle * period + b["t_on"]
        dur = b["t_off"] + np.random.randn(n) * 30e-6
        return {"cycle": cycle, "t_off_start": t_off, "t_on_start": t_off + dur,
                "off_duration": dur, "header": {"scheduling": "mock"},
                "warnings": [], "status": "done", "completed": n}


# ─────────────────────────────────────────────────────────────────────────────
# SSH worker — owns one board's driver, runs on its own thread
# ─────────────────────────────────────────────────────────────────────────────
class SSHWorker(QObject):
    connected = pyqtSignal(bool, int, str)   # ok, n_pll, message
    snapshot = pyqtSignal(object)            # all_params() dict
    monitors = pyqtSignal(object)            # read_monitors() dict
    reheat_state = pyqtSignal(object)        # reheat_progress() dict
    reheat_log = pyqtSignal(object)          # fetch_reheat_log() dict
    message = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.rp = None

    @pyqtSlot(str, str, str, int, bool, bool)
    def do_connect(self, host, user, pw, n_pll, mock, upload):
        try:
            if mock:
                self.rp = MockPLL(hostname=host, n_pll=n_pll)
            else:
                self.rp = P.RedPitayaPLL(host, user, pw, n_pll=n_pll)
                if upload:
                    self.message.emit("Uploading + loading bitstream…")
                    self.rp.upload_bitstream(load=True)
            self.connected.emit(True, self.rp.n_pll, f"Connected to {host}")
            self.snapshot.emit(self.rp.all_params())
        except Exception as exc:
            self.rp = None
            self.connected.emit(False, 0, f"Connection failed: {exc}")

    @pyqtSlot()
    def do_disconnect(self):
        if self.rp is not None:
            try:
                self.rp.close()
            except Exception:
                pass
            self.rp = None
        self.message.emit("Disconnected")

    @pyqtSlot(str, float, int)
    def do_set(self, name, value, pll):
        if self.rp is None:
            return
        try:
            self.rp.set_param(name, value, pll)
        except Exception as exc:
            self.error.emit(f"Set {name}[PLL{pll}] failed: {exc}")

    @pyqtSlot(str, str, int)
    def do_set_text(self, name, value, pll):
        """For string-valued params (output combos)."""
        if self.rp is None:
            return
        try:
            self.rp.set_param(name, value, pll)
        except Exception as exc:
            self.error.emit(f"Set {name} failed: {exc}")

    @pyqtSlot(int)
    def do_preset(self, pll):
        if self.rp is None:
            return
        try:
            self.rp.apply_cooling_preset(pll)
            self.snapshot.emit(self.rp.all_params())
            self.message.emit(f"Applied cooling preset to PLL{pll}")
        except Exception as exc:
            self.error.emit(f"Preset failed: {exc}")

    @pyqtSlot()
    def do_sync(self):
        if self.rp is None:
            return
        try:
            self.rp.sync_all()
            self.snapshot.emit(self.rp.all_params())
            self.message.emit("Resynced from device")
        except Exception as exc:
            self.error.emit(f"Resync failed: {exc}")

    @pyqtSlot(str)
    def do_save(self, fn):
        try:
            self.rp.save_params(fn)
            self.message.emit(f"Saved to {os.path.basename(fn)}")
        except Exception as exc:
            self.error.emit(f"Save failed: {exc}")

    @pyqtSlot(str)
    def do_load(self, fn):
        try:
            self.rp.load_params(fn)
            self.snapshot.emit(self.rp.all_params())
            self.message.emit(f"Loaded {os.path.basename(fn)}")
        except Exception as exc:
            self.error.emit(f"Load failed: {exc}")

    @pyqtSlot()
    def do_poll(self):
        if self.rp is None:
            return
        try:
            self.monitors.emit(self.rp.read_monitors())
        except Exception as exc:
            self.error.emit(f"Monitor read failed: {exc}")

    # ── reheating bursts ────────────────────────────────────────────────────
    @pyqtSlot(object)
    def do_reheat_start(self, cfg):
        if self.rp is None:
            return
        try:
            info = self.rp.start_reheat(**cfg)
            self.message.emit(
                f"Reheat burst running: {info['cycles']} cycles, "
                f"~{info['duration_s']:.0f} s")
            self.reheat_state.emit(self.rp.reheat_progress())
        except Exception as exc:
            self.error.emit(f"Reheat start failed: {exc}")
            self.reheat_state.emit({"running": False, "cycles_done": 0,
                                    "cycles": 0, "fraction": 0.0})

    @pyqtSlot()
    def do_reheat_stop(self):
        if self.rp is None:
            return
        try:
            self.rp.stop_reheat()
            self.message.emit("Reheat burst stopped; drive restored")
        except Exception as exc:
            self.error.emit(f"Reheat stop failed: {exc}")
        try:
            self.reheat_state.emit(self.rp.reheat_progress())
            self.snapshot.emit(self.rp.all_params())
        except Exception:
            pass

    @pyqtSlot()
    def do_reheat_poll(self):
        if self.rp is None:
            return
        try:
            self.reheat_state.emit(self.rp.reheat_progress())
        except Exception as exc:
            self.error.emit(f"Reheat poll failed: {exc}")

    @pyqtSlot(str)
    def do_reheat_fetch(self, fn):
        if self.rp is None:
            return
        try:
            self.reheat_log.emit(self.rp.fetch_reheat_log(fn or None))
        except Exception as exc:
            self.error.emit(f"Reheat log fetch failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# small widget helpers
# ─────────────────────────────────────────────────────────────────────────────
def _dspin(lo, hi, val, dec=4, step=None, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(val)
    if step is not None:
        s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    s.setKeyboardTracking(False)   # emit only on Enter/focus-out -> fewer writes
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Per-PLL control panel
# ─────────────────────────────────────────────────────────────────────────────
class PllPanel(QGroupBox):
    """All parameters for one PLL. Emits (name, value, pll) on user edits."""

    changed = pyqtSignal(str, float, int)

    def __init__(self, pll, win):
        super().__init__(f"PLL {pll}")
        self.pll = pll
        self._win = win
        self._loading = False
        col = _PLL_COLORS[pll % len(_PLL_COLORS)]
        self.setStyleSheet(f"QGroupBox{{font-weight:bold;}}"
                           f"QGroupBox::title{{color:{col};}}")
        g = QGridLayout(self)
        g.setVerticalSpacing(3)

        self.cb_harm = QCheckBox("2nd harmonic")
        self.cb_pid = QCheckBox("PID lock (engage)")
        g.addWidget(self.cb_harm, 0, 0, 1, 2)
        g.addWidget(self.cb_pid, 1, 0, 1, 2)

        self.f0 = _dspin(0.0, 10e6, 130e3, dec=2, step=1e3, suffix=" Hz")
        self.bw = _dspin(0.0, 5e6, 5e3, dec=2, step=100.0, suffix=" Hz")
        self.a = _dspin(0.0, 127.0, 30.0, dec=2, step=1.0)
        self.phi = _dspin(-180.0, 180.0, 0.0, dec=2, step=1.0, suffix=" °")
        self.kp = _dspin(-1e5, 1e5, -40.0, dec=4, step=1.0)
        self.ki = _dspin(-1e5, 1e5, -0.128, dec=5, step=0.01)
        self.alpha = _dspin(0.0, 0.999999, 0.39, dec=5, step=0.01)
        self.order = QSpinBox(); self.order.setRange(1, 8); self.order.setValue(4)

        rows = [("f0 (setpoint):", self.f0), ("bw (drift limit):", self.bw),
                ("a (amplitude 0–127):", self.a), ("phi (phase):", self.phi),
                ("kp (neg!):", self.kp), ("ki (neg!):", self.ki),
                ("alpha (LPF):", self.alpha), ("order (1–8):", self.order)]
        r = 2
        for label, w in rows:
            g.addWidget(QLabel(label), r, 0)
            g.addWidget(w, r, 1)
            r += 1

        self.preset_btn = QPushButton("Apply cooling preset")
        self.preset_btn.clicked.connect(lambda: self._win.apply_preset(self.pll))
        g.addWidget(self.preset_btn, r, 0, 1, 2)

        # wire signals
        self.cb_harm.toggled.connect(
            lambda v: self._emit("2nd_harm", 1.0 if v else 0.0))
        self.cb_pid.toggled.connect(
            lambda v: self._emit("pid_en", 1.0 if v else 0.0))
        self.f0.valueChanged.connect(lambda v: self._emit("f0", v))
        self.bw.valueChanged.connect(lambda v: self._emit("bw", v))
        self.a.valueChanged.connect(lambda v: self._emit("a", v))
        self.phi.valueChanged.connect(lambda v: self._emit("phi", v))
        self.kp.valueChanged.connect(lambda v: self._emit("kp", v))
        self.ki.valueChanged.connect(lambda v: self._emit("ki", v))
        self.alpha.valueChanged.connect(lambda v: self._emit("alpha", v))
        self.order.valueChanged.connect(lambda v: self._emit("order", float(v)))

    def _emit(self, name, value):
        if not self._loading:
            self.changed.emit(name, float(value), self.pll)

    def load_values(self, d):
        """Populate widgets from an all_params()['pll'][i] dict without emitting."""
        self._loading = True
        try:
            self.cb_harm.setChecked(bool(int(d["2nd_harm"])))
            self.cb_pid.setChecked(bool(int(d["pid_en"])))
            self.f0.setValue(float(d["f0"]))
            self.bw.setValue(float(d["bw"]))
            self.a.setValue(float(d["a"]))
            self.phi.setValue(float(d["phi"]))
            self.kp.setValue(float(d["kp"]))
            self.ki.setValue(float(d["ki"]))
            self.alpha.setValue(float(d["alpha"]))
            self.order.setValue(int(d["order"]))
        finally:
            self._loading = False


# ─────────────────────────────────────────────────────────────────────────────
# Reheating-burst panel
# ─────────────────────────────────────────────────────────────────────────────
class ReheatPanel(QWidget):
    """Gate the cooling drive off and on, many times, for reheating runs.

    The cycling itself happens on the board (``rp_reheat.py``) -- this panel
    only launches it, shows progress, and plots the achieved off-window
    durations afterwards so cycles the board's scheduler disturbed can be
    spotted (and cut in analysis).
    """

    start_requested = pyqtSignal(object)   # kwargs for start_reheat()
    stop_requested = pyqtSignal()
    fetch_requested = pyqtSignal(str)      # local filename ('' = don't save)

    def __init__(self, n_pll):
        super().__init__()
        self._n_pll = n_pll
        self._running = False
        self._build()

    def _build(self):
        root = QHBoxLayout(self)

        box = QGroupBox("Burst")
        form = QFormLayout(box)
        self.t_on = _dspin(0.1, 600000, 500.0, dec=2, step=10, suffix=" ms")
        self.t_off = _dspin(0.1, 10000, 20.0, dec=2, step=1, suffix=" ms")
        self.cycles = QSpinBox(); self.cycles.setRange(1, 1000000)
        self.cycles.setValue(1000)
        form.addRow("Feedback ON (cool):", self.t_on)
        form.addRow("Feedback OFF (dark):", self.t_off)
        form.addRow("Cycles:", self.cycles)

        self.pll_checks = []
        pll_row = QHBoxLayout()
        for pll in range(self._n_pll):
            cb = QCheckBox(f"PLL{pll}")
            cb.setChecked(pll == 0)
            self.pll_checks.append(cb)
            pll_row.addWidget(cb)
        pll_row.addStretch()
        wrap = QWidget(); wrap.setLayout(pll_row)
        form.addRow("Gate:", wrap)

        marker_row = QHBoxLayout()
        self.marker_on = QCheckBox("ext_pins_p bit")
        self.marker_on.setChecked(True)
        self.marker_bit = QSpinBox(); self.marker_bit.setRange(0, 7)
        marker_row.addWidget(self.marker_on)
        marker_row.addWidget(self.marker_bit)
        marker_row.addStretch()
        mwrap = QWidget(); mwrap.setLayout(marker_row)
        mwrap.setToolTip("Digital marker on the E1 header, high while the "
                         "feedback is off — record it on a spare DAQ channel "
                         "to timestamp every dark window.")
        form.addRow("Marker:", mwrap)

        self.settle = _dspin(0, 60, 1.0, dec=2, step=0.5, suffix=" s")
        self.spin_ms = _dspin(0.1, 20, 1.5, dec=2, step=0.5, suffix=" ms")
        self.spin_ms.setToolTip("Busy-wait window before each edge. Larger = "
                                "sharper edges, more CPU on the board.")
        self.realtime = QCheckBox("SCHED_FIFO on the board")
        self.realtime.setChecked(True)
        form.addRow("Settle first:", self.settle)
        form.addRow("Busy-wait:", self.spin_ms)
        form.addRow("", self.realtime)

        self.duration_lbl = QLabel("")
        form.addRow("Total:", self.duration_lbl)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("Start burst")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        btns.addWidget(self.start_btn); btns.addWidget(self.stop_btn)
        form.addRow(btns)

        self.progress = QProgressBar(); self.progress.setRange(0, 100)
        self.progress_lbl = QLabel("idle")
        form.addRow(self.progress)
        form.addRow(self.progress_lbl)

        fetch = QHBoxLayout()
        self.fetch_btn = QPushButton("Fetch timing log")
        self.save_btn = QPushButton("Fetch + save…")
        self.fetch_btn.clicked.connect(lambda: self.fetch_requested.emit(""))
        self.save_btn.clicked.connect(self._on_save)
        fetch.addWidget(self.fetch_btn); fetch.addWidget(self.save_btn)
        form.addRow(fetch)

        box.setMaximumWidth(340)
        root.addWidget(box)

        right = QVBoxLayout()
        self.plot = pg.PlotWidget(); self.plot.setBackground("#1a1a1a")
        self.plot.setLabel("left", "Achieved OFF window", units="s")
        self.plot.setLabel("bottom", "Cycle")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen=None, symbol="o", symbolSize=3,
                                    symbolBrush="#00e5ff", symbolPen=None)
        self.target_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(
            "#feca57", style=Qt.DashLine))
        self.plot.addItem(self.target_line)
        right.addWidget(self.plot, 1)
        self.stats_lbl = QLabel("No timing log fetched yet.")
        self.stats_lbl.setStyleSheet("font-family:monospace; color:#e0e0e0;")
        right.addWidget(self.stats_lbl)
        root.addLayout(right, 1)

        for w in (self.t_on, self.t_off):
            w.valueChanged.connect(self._update_duration)
        self.cycles.valueChanged.connect(self._update_duration)
        self.settle.valueChanged.connect(self._update_duration)
        self._update_duration()

    # ── helpers ─────────────────────────────────────────────────────────────
    def _update_duration(self):
        total = (self.settle.value() +
                 self.cycles.value() *
                 (self.t_on.value() + self.t_off.value()) * 1e-3)
        self.duration_lbl.setText(f"{total:.1f} s  ({total/60:.1f} min)")

    def config(self):
        plls = [i for i, cb in enumerate(self.pll_checks) if cb.isChecked()]
        return dict(t_off=self.t_off.value() * 1e-3,
                    t_on=self.t_on.value() * 1e-3,
                    cycles=int(self.cycles.value()),
                    plls=plls,
                    marker_bit=(int(self.marker_bit.value())
                                if self.marker_on.isChecked() else None),
                    settle=self.settle.value(),
                    spin=self.spin_ms.value() * 1e-3,
                    realtime=self.realtime.isChecked())

    def _on_start(self):
        cfg = self.config()
        if not cfg["plls"]:
            QMessageBox.warning(self, "Reheat burst",
                                "Select at least one PLL to gate.")
            return
        self.target_line.setValue(cfg["t_off"])
        self.start_requested.emit(cfg)

    def _on_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save reheat timing log",
                                            "reheat_timing.log", "Log (*.log)")
        if fn:
            self.fetch_requested.emit(fn)

    # ── state from the worker ───────────────────────────────────────────────
    def set_running(self, running, done=0, total=0):
        self._running = running
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if total:
            self.progress.setValue(int(100 * done / total))
        if running:
            self.progress_lbl.setText(f"running — cycle {done}/{total}")
        elif total:
            self.progress_lbl.setText(f"finished — {done}/{total} cycles")
        else:
            self.progress_lbl.setText("idle")

    def show_log(self, log):
        dur = np.asarray(log["off_duration"], dtype=float)
        self.curve.setData(np.asarray(log["cycle"], dtype=float), dur)
        if dur.size:
            target = self.target_line.value() or float(np.median(dur))
            excess_ms = (dur - target) * 1e3
            self.stats_lbl.setText(
                f"n = {dur.size}   status: {log['status']}\n"
                f"OFF window: mean {dur.mean()*1e3:.3f} ms   "
                f"std {dur.std()*1e3:.3f} ms\n"
                f"deviation from target: max {np.abs(excess_ms).max():.3f} ms   "
                f"|dev| > 1 ms in {int((np.abs(excess_ms) > 1).sum())} cycles")
        else:
            self.stats_lbl.setText(f"Empty log (status: {log['status']}).")
        for warning in log.get("warnings", []):
            self.stats_lbl.setText(self.stats_lbl.text() + f"\n⚠ {warning}")


# ─────────────────────────────────────────────────────────────────────────────
# One board (connection + controls + plots + its own worker thread)
# ─────────────────────────────────────────────────────────────────────────────
class BoardWidget(QWidget):
    # request signals -> this board's worker (queued across threads)
    req_connect = pyqtSignal(str, str, str, int, bool, bool)
    req_disconnect = pyqtSignal()
    req_set = pyqtSignal(str, float, int)
    req_set_text = pyqtSignal(str, str, int)
    req_preset = pyqtSignal(int)
    req_sync = pyqtSignal()
    req_save = pyqtSignal(str)
    req_load = pyqtSignal(str)
    req_poll = pyqtSignal()
    req_reheat_start = pyqtSignal(object)
    req_reheat_stop = pyqtSignal()
    req_reheat_poll = pyqtSignal()
    req_reheat_fetch = pyqtSignal(str)

    # -> container: ask it to refresh this board's tab label/colour
    tab_dirty = pyqtSignal(object)

    def __init__(self, host="192.168.1.9", n_pll=2, use_mock=False):
        super().__init__()
        self._n_pll = n_pll
        self._use_mock = use_mock
        self._connected = False
        self._panels = []
        self._global_loading = False

        self._hist = [dict(t=deque(maxlen=_HISTORY), f=deque(maxlen=_HISTORY),
                           e=deque(maxlen=_HISTORY)) for _ in range(n_pll)]
        self._t0 = time.time()

        self._build_ui(host)
        self._start_worker()

        self._poll_timer = QTimer(self)
        self._poll_timer.setSingleShot(True)
        self._poll_timer.timeout.connect(self.req_poll.emit)

        # Reheat progress is polled far more slowly than the lock monitor.
        self._reheat_timer = QTimer(self)
        self._reheat_timer.setSingleShot(True)
        self._reheat_timer.timeout.connect(self.req_reheat_poll.emit)
        self._reheat_was_running = False

    # ── public helpers for the container ────────────────────────────────────
    def host(self):
        return self._host_edit.text().strip() or "(no host)"

    def is_connected(self):
        return self._connected

    def shutdown(self):
        """Stop polling, disconnect, and tear down the worker thread."""
        self._poll_timer.stop()
        self._reheat_timer.stop()
        try:
            self.req_disconnect.emit()
            self._thread.quit()
            self._thread.wait(2000)
        except Exception:
            pass

    # ── worker thread setup ──────────────────────────────────────────────────
    def _start_worker(self):
        self._thread = QThread(self)
        self._worker = SSHWorker()
        self._worker.moveToThread(self._thread)
        self.req_connect.connect(self._worker.do_connect)
        self.req_disconnect.connect(self._worker.do_disconnect)
        self.req_set.connect(self._worker.do_set)
        self.req_set_text.connect(self._worker.do_set_text)
        self.req_preset.connect(self._worker.do_preset)
        self.req_sync.connect(self._worker.do_sync)
        self.req_save.connect(self._worker.do_save)
        self.req_load.connect(self._worker.do_load)
        self.req_poll.connect(self._worker.do_poll)
        self.req_reheat_start.connect(self._worker.do_reheat_start)
        self.req_reheat_stop.connect(self._worker.do_reheat_stop)
        self.req_reheat_poll.connect(self._worker.do_reheat_poll)
        self.req_reheat_fetch.connect(self._worker.do_reheat_fetch)
        self._worker.connected.connect(self._on_connected)
        self._worker.snapshot.connect(self._on_snapshot)
        self._worker.monitors.connect(self._on_monitors)
        self._worker.reheat_state.connect(self._on_reheat_state)
        self._worker.reheat_log.connect(self._on_reheat_log)
        self._worker.message.connect(self._set_status)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self, host):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(self._build_connect_bar(host))
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_controls())
        split.addWidget(self._build_tabs())
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)
        root.addWidget(self._build_status_bar())
        self._set_controls_enabled(False)

    def _build_connect_bar(self, host):
        bar = QFrame(); bar.setFrameShape(QFrame.StyledPanel)
        hl = QHBoxLayout(bar); hl.setContentsMargins(8, 4, 8, 4)
        hl.addWidget(QLabel("Host:"))
        self._host_edit = QLineEdit(host); self._host_edit.setMaximumWidth(140)
        self._host_edit.textChanged.connect(lambda _: self.tab_dirty.emit(self))
        hl.addWidget(self._host_edit)
        hl.addWidget(QLabel("User:"))
        self._user_edit = QLineEdit("root"); self._user_edit.setMaximumWidth(70)
        hl.addWidget(self._user_edit)
        hl.addWidget(QLabel("Pass:"))
        self._pass_edit = QLineEdit("root"); self._pass_edit.setMaximumWidth(70)
        self._pass_edit.setEchoMode(QLineEdit.Password)
        hl.addWidget(self._pass_edit)
        self._upload_check = QCheckBox("Load bitstream on connect")
        hl.addWidget(self._upload_check)
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        hl.addWidget(self._connect_btn)
        self._led = QLabel("●"); self._led.setStyleSheet("color:#c0392b;")
        hl.addWidget(self._led)
        hl.addStretch()
        return bar

    def _build_controls(self):
        wrap = QWidget()
        v = QVBoxLayout(wrap); v.setContentsMargins(4, 4, 4, 4)
        for pll in range(self._n_pll):
            panel = PllPanel(pll, self)
            panel.changed.connect(self.req_set.emit)
            self._panels.append(panel)
            v.addWidget(panel)
        v.addWidget(self._build_global_box())

        btns = QHBoxLayout()
        sync_btn = QPushButton("Resync"); sync_btn.clicked.connect(self.req_sync.emit)
        save_btn = QPushButton("Save…"); save_btn.clicked.connect(self._on_save)
        load_btn = QPushButton("Load…"); load_btn.clicked.connect(self._on_load)
        for b in (sync_btn, save_btn, load_btn):
            btns.addWidget(b)
        v.addLayout(btns)
        v.addStretch()

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(wrap)
        scroll.setMinimumWidth(330); scroll.setMaximumWidth(400)
        return scroll

    def _build_global_box(self):
        box = QGroupBox("Global / outputs")
        g = QGridLayout(box)
        self._out1 = QComboBox(); self._out1.addItems(P.OUTPUT_OPTIONS.keys())
        self._out2 = QComboBox(); self._out2.addItems(P.OUTPUT_OPTIONS.keys())
        self._extp = QSpinBox(); self._extp.setRange(0, 255)
        self._extn = QSpinBox(); self._extn.setRange(0, 255)
        self._out1.activated[str].connect(
            lambda val: self.req_set_text.emit("output_1", val, 0))
        self._out2.activated[str].connect(
            lambda val: self.req_set_text.emit("output_2", val, 0))
        self._extp.valueChanged.connect(
            lambda v: self._global_changed("ext_pins_p", v))
        self._extn.valueChanged.connect(
            lambda v: self._global_changed("ext_pins_n", v))
        rows = [("RF out 1:", self._out1), ("RF out 2:", self._out2),
                ("ext pins P (0–255):", self._extp),
                ("ext pins N (0–255):", self._extn)]
        for r, (lbl, w) in enumerate(rows):
            g.addWidget(QLabel(lbl), r, 0); g.addWidget(w, r, 1)
        return box

    def _global_changed(self, name, value):
        if not self._global_loading:
            self.req_set.emit(name, float(value), 0)

    def _build_tabs(self):
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_monitor_tab(), "Lock monitor")
        self._tabs.addTab(self._build_bode_tab(), "Loop design (Bode)")
        self._tabs.addTab(self._build_reheat_tab(), "Reheat burst")
        return self._tabs

    def _build_reheat_tab(self):
        self._reheat = ReheatPanel(self._n_pll)
        self._reheat.start_requested.connect(self._on_reheat_start)
        self._reheat.stop_requested.connect(self.req_reheat_stop.emit)
        self._reheat.fetch_requested.connect(self.req_reheat_fetch.emit)
        return self._reheat

    def _build_monitor_tab(self):
        w = QWidget(); hl = QHBoxLayout(w)
        glw = pg.GraphicsLayoutWidget(); glw.setBackground("#1a1a1a")

        self._p_freq = glw.addPlot(row=0, col=0)
        self._p_freq.setLabel("left", "NCO frequency", units="Hz")
        self._p_freq.showGrid(x=True, y=True, alpha=0.3)
        self._p_freq.addLegend(offset=(-10, 10))
        self._p_err = glw.addPlot(row=1, col=0)
        self._p_err.setLabel("left", "Loop error", units="deg")
        self._p_err.setLabel("bottom", "Time", units="s")
        self._p_err.showGrid(x=True, y=True, alpha=0.3)
        self._p_err.setXLink(self._p_freq)

        self._freq_curves, self._err_curves = [], []
        for pll in range(self._n_pll):
            pen = pg.mkPen(_PLL_COLORS[pll % len(_PLL_COLORS)], width=2)
            self._freq_curves.append(
                self._p_freq.plot(pen=pen, name=f"PLL{pll}"))
            self._err_curves.append(self._p_err.plot(pen=pen))
        hl.addWidget(glw, 1)

        side = QWidget(); sv = QVBoxLayout(side); side.setMaximumWidth(260)
        self._readout = QLabel("—"); self._readout.setStyleSheet(
            "font-family:monospace; color:#e0e0e0;")
        self._readout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        sv.addWidget(self._readout)
        vec = pg.PlotWidget(); vec.setBackground("#1a1a1a")
        vec.setAspectLocked(True); vec.setXRange(-130, 130); vec.setYRange(-130, 130)
        vec.setLabel("bottom", "w_b (a·cosφ)"); vec.setLabel("left", "w_a (a·sinφ)")
        vec.showGrid(x=True, y=True, alpha=0.3)
        th = np.linspace(0, 2 * np.pi, 100)
        vec.plot(127 * np.cos(th), 127 * np.sin(th),
                 pen=pg.mkPen("#555", style=Qt.DashLine))
        self._vec_curves = []
        for pll in range(self._n_pll):
            pen = pg.mkPen(_PLL_COLORS[pll % len(_PLL_COLORS)], width=2)
            self._vec_curves.append(vec.plot(pen=pen, symbol="o", symbolSize=6,
                                             symbolBrush=_PLL_COLORS[pll % 2]))
        sv.addWidget(QLabel("Output vector (a, φ):"))
        sv.addWidget(vec)
        hl.addWidget(side)
        return w

    def _build_bode_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        glw = pg.GraphicsLayoutWidget(); glw.setBackground("#1a1a1a")
        self._b_lpf = glw.addPlot(row=0, col=0)
        self._b_lpf.setTitle("Phase-detector low-pass (alpha / order)")
        self._b_lpf.setLabel("left", "Mag", units="dB")
        self._b_lpf.setLabel("bottom", "Frequency", units="Hz")
        self._b_lpf.setLogMode(x=True, y=False); self._b_lpf.showGrid(True, True, 0.3)
        self._b_lpf.addLegend(offset=(-10, 10))
        self._b_pi = glw.addPlot(row=1, col=0)
        self._b_pi.setTitle("PI loop filter |P + I/(iω)|  (Hz/deg)")
        self._b_pi.setLabel("left", "Gain", units="Hz/deg")
        self._b_pi.setLabel("bottom", "Frequency", units="Hz")
        self._b_pi.setLogMode(x=True, y=True); self._b_pi.showGrid(True, True, 0.3)
        self._b_pi.addLegend(offset=(-10, 10))
        v.addWidget(glw, 1)

        self._lpf_curves, self._pi_curves = [], []
        for pll in range(self._n_pll):
            pen = pg.mkPen(_PLL_COLORS[pll % len(_PLL_COLORS)], width=2)
            self._lpf_curves.append(self._b_lpf.plot(pen=pen, name=f"PLL{pll}"))
            self._pi_curves.append(self._b_pi.plot(pen=pen, name=f"PLL{pll}"))

        btn = QPushButton("Update Bode from current values")
        btn.clicked.connect(self._update_bode)
        v.addWidget(btn)
        return w

    def _build_status_bar(self):
        w = QFrame(); w.setFrameShape(QFrame.StyledPanel); w.setFixedHeight(26)
        hl = QHBoxLayout(w); hl.setContentsMargins(6, 2, 6, 2)
        self._rate_label = QLabel("")
        self._status_label = QLabel("Not connected")
        hl.addWidget(self._rate_label)
        sep = QFrame(); sep.setFrameShape(QFrame.VLine); hl.addWidget(sep)
        hl.addStretch(); hl.addWidget(self._status_label)
        return w

    # ── enable/disable ───────────────────────────────────────────────────────
    def _set_controls_enabled(self, on):
        for p in self._panels:
            p.setEnabled(on)
        self._reheat.setEnabled(on)

    # ── connection handlers ──────────────────────────────────────────────────
    def _on_connect_clicked(self):
        if self._connected:
            self._poll_timer.stop()
            self._reheat_timer.stop()
            if self._reheat_was_running:
                QMessageBox.information(
                    self, "Reheat burst",
                    "A reheating burst is still cycling on the board. It runs "
                    "independently of this GUI and restores the drive when it "
                    "finishes; reconnect to watch or stop it.")
                self._reheat_was_running = False
            self._reheat.set_running(False)
            self.req_disconnect.emit()
            self._connected = False
            self._connect_btn.setText("Connect")
            self._led.setStyleSheet("color:#c0392b;")
            self._set_controls_enabled(False)
            self._set_status("Disconnected")
            self.tab_dirty.emit(self)
            return
        host = self.host()
        mock = self._use_mock or host == "mock"
        self._connect_btn.setEnabled(False)
        self._set_status(f"Connecting to {host}…")
        self.req_connect.emit(host, self._user_edit.text(),
                              self._pass_edit.text(), self._n_pll, mock,
                              self._upload_check.isChecked())

    @pyqtSlot(bool, int, str)
    def _on_connected(self, ok, n_pll, msg):
        self._connect_btn.setEnabled(True)
        self._set_status(msg)
        if not ok:
            QMessageBox.critical(self, "Connection", msg)
            self.tab_dirty.emit(self)
            return
        self._connected = True
        self._connect_btn.setText("Disconnect")
        self._led.setStyleSheet("color:#27ae60;")
        self._set_controls_enabled(True)
        for h in self._hist:
            h["t"].clear(); h["f"].clear(); h["e"].clear()
        self._t0 = time.time()
        self._update_bode()
        self._poll_timer.start(_POLL_MS)
        # A burst launched earlier may still be cycling on the board.
        self.req_reheat_poll.emit()
        self.tab_dirty.emit(self)

    @pyqtSlot(object)
    def _on_snapshot(self, snap):
        for pll, d in enumerate(snap["pll"]):
            if pll < len(self._panels):
                self._panels[pll].load_values(d)
        g = snap["global"]
        self._global_loading = True
        try:
            self._out1.setCurrentText(g["output_1"])
            self._out2.setCurrentText(g["output_2"])
            self._extp.setValue(int(g["ext_pins_p"]))
            self._extn.setValue(int(g["ext_pins_n"]))
        finally:
            self._global_loading = False
        self._update_bode()

    @pyqtSlot(object)
    def _on_monitors(self, mons):
        now = time.time() - self._t0
        lines = []
        for pll, m in mons.items():
            h = self._hist[pll]
            h["t"].append(now); h["f"].append(m["nco_freq_hz"])
            h["e"].append(m["err_deg"])
            self._freq_curves[pll].setData(list(h["t"]), list(h["f"]))
            self._err_curves[pll].setData(list(h["t"]), list(h["e"]))
            a = self._panels[pll].a.value()
            phi = np.deg2rad(self._panels[pll].phi.value())
            self._vec_curves[pll].setData([0, a * np.cos(phi)],
                                          [0, a * np.sin(phi)])
            lines.append(f"PLL{pll}:\n  NCO  {m['nco_freq_hz']:>12.2f} Hz\n"
                         f"  err  {m['err_deg']:>+12.3f} °")
        self._readout.setText("\n".join(lines))
        if self._connected:
            self._poll_timer.start(_POLL_MS)

    # ── reheating bursts ─────────────────────────────────────────────────────
    def _on_reheat_start(self, cfg):
        if not self._connected:
            return
        self._reheat.set_running(True, 0, cfg["cycles"])
        self.req_reheat_start.emit(cfg)

    @pyqtSlot(object)
    def _on_reheat_state(self, state):
        running = bool(state.get("running"))
        self._reheat.set_running(running, state.get("cycles_done", 0),
                                 state.get("cycles", 0))
        if running:
            self._reheat_timer.start(_REHEAT_POLL_MS)
        elif self._reheat_was_running:
            # Burst just finished -- pull the timing log without being asked.
            self._set_status("Reheat burst finished; fetching timing log…")
            self.req_reheat_fetch.emit("")
        self._reheat_was_running = running

    @pyqtSlot(object)
    def _on_reheat_log(self, log):
        self._reheat.show_log(log)
        self._set_status(f"Reheat log: {log['completed']} cycles "
                         f"({log['status']})")

    @pyqtSlot(str)
    def _on_error(self, msg):
        self._set_status(msg)

    # ── actions ──────────────────────────────────────────────────────────────
    def apply_preset(self, pll):
        if self._connected:
            self.req_preset.emit(pll)

    def _on_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save PLL parameters",
                                            f"{self.host()}_pll.txt", "Text (*.txt)")
        if fn:
            self.req_save.emit(fn)

    def _on_load(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load PLL parameters",
                                            "", "Text (*.txt)")
        if fn:
            self.req_load.emit(fn)

    def _update_bode(self):
        f = np.logspace(1, np.log10(P.CLK_HZ / 2), 400)
        for pll, panel in enumerate(self._panels):
            H = P.RedPitayaPLL.phase_detector_lpf(panel.alpha.value(),
                                                  panel.order.value(), f)
            mag = 20 * np.log10(np.abs(H) + 1e-30)
            self._lpf_curves[pll].setData(f, mag)
            G = P.RedPitayaPLL.pi_loop_tf(panel.kp.value(), panel.ki.value(), f)
            self._pi_curves[pll].setData(f, np.abs(G) + 1e-30)

    def _set_status(self, msg):
        self._status_label.setText(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Main window — a tab per Red Pitaya
# ─────────────────────────────────────────────────────────────────────────────
class PllMainWindow(QMainWindow):

    def __init__(self, n_pll=2, use_mock=False):
        super().__init__()
        self.setWindowTitle("Red Pitaya — Parametric-cooling PLLs")
        self.setMinimumSize(1420, 840)
        self._n_pll = n_pll
        self._use_mock = use_mock

        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)
        self.setCentralWidget(self._tabs)

        tb = QToolBar("Boards"); tb.setMovable(False)
        self.addToolBar(tb)
        add_act = QAction("➕ Add Red Pitaya", self)
        add_act.triggered.connect(self._add_board_dialog)
        tb.addAction(add_act)

    # ── board management ─────────────────────────────────────────────────────
    def add_board(self, host="192.168.1.9"):
        board = BoardWidget(host=host, n_pll=self._n_pll,
                            use_mock=self._use_mock)
        board.tab_dirty.connect(self._refresh_tab)
        idx = self._tabs.addTab(board, host)
        self._tabs.setCurrentIndex(idx)
        self._refresh_tab(board)
        return board

    def _add_board_dialog(self):
        default = "mock" if self._use_mock else "192.168.1.9"
        host, ok = QInputDialog.getText(self, "Add Red Pitaya",
                                        "Hostname / IP:", text=default)
        if ok and host.strip():
            self.add_board(host.strip())

    def _refresh_tab(self, board):
        idx = self._tabs.indexOf(board)
        if idx < 0:
            return
        dot = "●" if board.is_connected() else "○"
        self._tabs.setTabText(idx, f"{dot} {board.host()}")
        color = QColor("#27ae60") if board.is_connected() else QColor("#c0392b")
        self._tabs.tabBar().setTabTextColor(idx, color)

    def _close_tab(self, idx):
        board = self._tabs.widget(idx)
        if board is None:
            return
        if board.is_connected():
            if QMessageBox.question(
                    self, "Close board",
                    f"Disconnect and close '{board.host()}'?") != \
                    QMessageBox.Yes:
                return
        board.shutdown()
        self._tabs.removeTab(idx)
        board.deleteLater()

    # ── shutdown ─────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        for i in range(self._tabs.count()):
            board = self._tabs.widget(i)
            if board is not None:
                board.shutdown()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(
        description="Red Pitaya parametric-cooling PLL GUI (multi-board)")
    parser.add_argument("--host", action="append", default=None,
                        help="Red Pitaya IP / hostname (repeat for several "
                             "boards, e.g. --host A --host B)")
    parser.add_argument("--n-pll", type=int, default=2, choices=(1, 2))
    parser.add_argument("--mock", action="store_true",
                        help="Run with synthetic backend(s), no hardware")
    args = parser.parse_args()

    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = PllMainWindow(n_pll=args.n_pll, use_mock=args.mock)

    hosts = args.host
    if not hosts:
        hosts = ["mock"] if args.mock else ["192.168.1.9"]
    for h in hosts:
        win.add_board(h)

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
