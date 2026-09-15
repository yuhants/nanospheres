"""
Red Pitaya PI locks for the trap monitor: laser power stabilization (with
particle drop) and homodyne local-oscillator phase lock.

Both loops run inside ONE Red Pitaya FPGA (PyRPL bitstream), one on each
output.  The GUI only sets registers and reads monitor values, so it can
never stall a lock.

    laser    : photodiode -> in1 -> pid0 -> out1 -> HV amp -> EOM -> PBS
    homodyne : homodyne signal -> in2 -> pid1 -> out2 -> fiber stretcher

Layers
------
    PyRPLBackend / MockBackend   raw access to PID modules + signal sampler
    SharedLink                   the single PyRPL connection, shared by all tabs
    PiLock                       lock / hold / drop / sweep logic for one pid module
    PiLockPanel                  generic tab (connection, loop, readout, sweep, plots)
      LaserStabPanel             + particle-drop section
      HomodynePanel              + rail auto-reset (fiber stretcher runs out of range)

Drop mechanism (open-loop hold, no integrator wind-up):
    1. remember p, i and the integrator value ``ival`` (= the output voltage
       that currently holds the trap)
    2. set p = i = 0 and preset ``ival`` to the calibrated extinction voltage
    3. after the dwell restore ``ival`` first, then p and i

PyRPL sign convention: output = p*(in - setpoint) + integral(...).  If the
error signal RISES with output voltage the gains must be NEGATIVE.  The
calibration sweep measures that slope and shows the required sign.

Notes
-----
* Connecting with "Reload FPGA" replaces whatever bitstream the board is
  running (e.g. the Oscilloscope+Lock-in+PID web app).  It also zeroes every
  PID register, so a lock running on the board is lost and a trapped
  particle is dropped.  Whether the PyRPL bitstream has been loaded is
  therefore remembered in the settings file (``link.fpga_loaded``), not just
  for the session: after a PC reboot / GUI crash the default is to reconnect
  WITHOUT reloading, and a reload always asks for confirmation.
* On reconnect a lock found running on the board is adopted as-is: the
  panel takes setpoint / gains / limits / routing from the registers instead
  of writing its own remembered values.
* PyRPL is not thread-safe -> every board access happens on the GUI thread,
  driven by QTimers (same pattern as pyrpl_gui.py).
* Disconnecting does NOT touch the FPGA registers: engaged locks keep running
  standalone on the board.
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import datetime
import collections

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QMessageBox, QSplitter, QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont

try:
    import pyqtgraph as pg
    _HAS_PG = True
except Exception:          # pragma: no cover - plotting is optional
    pg = None
    _HAS_PG = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

SETTINGS_PATH = os.path.join(_HERE, 'laser_stab_settings.json')

PID_MODULES = ['pid0', 'pid1', 'pid2']
INPUTS      = ['in1', 'in2']
OUTPUTS     = ['out1', 'out2']

# Loop states reported by PiLock.state
STATE_DISCONNECTED = 'disconnected'
STATE_OFF          = 'off'        # output_direct = off (0 V out)
STATE_HOLD         = 'hold'       # output on, gains zero, ival held
STATE_LOCKED       = 'locked'     # PI loop closed
STATE_DROPPING     = 'dropping'   # open-loop at extinction voltage
STATE_SWEEP        = 'sweep'      # calibration sweep in progress

_STATE_COLORS = {
    STATE_DISCONNECTED: '#666666',
    STATE_OFF:          '#888888',
    STATE_HOLD:         '#ffaa00',
    STATE_LOCKED:       '#00cc44',
    STATE_DROPPING:     '#ff3333',
    STATE_SWEEP:        '#00e5ff',
}

POLL_HZ = 5

LINK_DEFAULTS = {'host': '192.168.1.100', 'config': 'laser_stab',
                 'fpga_loaded': False}   # PyRPL bitstream on the board?

LASER_DEFAULTS = {
    'module': 'pid0', 'input': 'in1', 'output': 'out1',
    'setpoint': 0.500, 'p': -0.05, 'i': -200.0,
    'vmin': -1.0, 'vmax': 1.0, 'lowpass_hz': 0.0, 'start_v': 0.0,
    'hold_v': -0.9, 'dwell_ms': 300, 'confirm_drop': True,
}

HOMODYNE_DEFAULTS = {
    'module': 'pid1', 'input': 'in2', 'output': 'out2',
    'setpoint': 0.0, 'p': -0.1, 'i': -500.0,
    'vmin': -1.0, 'vmax': 1.0, 'lowpass_hz': 0.0, 'start_v': 0.0,
    'rail_reset': True, 'rail_margin': 0.05, 'v_2pi': 0.0,
}

_FLAT_KEYS = set(LASER_DEFAULTS) | {'host', 'config'}


def load_settings() -> dict:
    """{'link': {...}, 'laser': {...}, 'homodyne': {...}} with migration of
    the older flat (laser-only) file."""
    s = {'link': dict(LINK_DEFAULTS), 'laser': dict(LASER_DEFAULTS),
         'homodyne': dict(HOMODYNE_DEFAULTS)}
    try:
        with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except Exception:
        return s
    if isinstance(raw, dict) and 'link' not in raw and 'laser' not in raw \
            and _FLAT_KEYS & set(raw):
        # legacy flat file
        for k in ('host', 'config'):
            if k in raw:
                s['link'][k] = raw[k]
        s['laser'].update({k: v for k, v in raw.items() if k in LASER_DEFAULTS})
        return s
    for key in ('link', 'laser', 'homodyne'):
        if isinstance(raw.get(key), dict):
            s[key].update(raw[key])
    return s


def save_settings(s: dict) -> None:
    try:
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(s, f, indent=2)
    except Exception:
        pass


# ── Backends ──────────────────────────────────────────────────────────────────

class PyRPLBackend:
    """One PyRPL connection; exposes the PID modules and the signal sampler."""

    def __init__(self, host: str, config: str = 'laser_stab',
                 reloadfpga: bool = True):
        from control.src.Red_pitaya_control.pyrpl_control import RedPitayaPyRPL
        self._rp = RedPitayaPyRPL(hostname=host, config=config, gui=False,
                                  reloadfpga=reloadfpga)
        self.host = host

    def pid(self, module: str):
        return getattr(self._rp.rp, module)

    def read_signal(self, name: str, t: float = 0.005) -> tuple[float, float]:
        """(mean, std) of a bus signal ('in1', 'pid0', ...) over ``t`` seconds."""
        m, s = self._rp.rp.sampler.mean_stddev(name, t)
        return float(m), float(s)

    def step(self) -> None:          # hardware advances itself
        pass

    def close(self) -> None:
        try:
            self._rp.close()
        except Exception:
            pass


class _MockPid:
    def __init__(self):
        self.input = 'in1'
        self.output_direct = 'off'
        self.setpoint = 0.0
        self.p = 0.0
        self.i = 0.0
        self.ival = 0.0
        self.max_voltage = 1.0
        self.min_voltage = -1.0
        self.inputfilter = [0, 0, 0, 0]


class MockBackend:
    """Two simulated plants so the tabs run without hardware:

    out1 -> EOM + PBS -> photodiode on in1:  A·sin²(π/2·(V−V0)/Vπ), drifting laser
    out2 -> fiber stretcher -> homodyne on in2: B·sin(2π·V/Vper + φ(t)), drifting φ
    """

    AMP, VPI, V0, OFFSET, NOISE = 0.80, 1.20, -0.90, 0.02, 0.004
    HAMP, VPER, PHASE_RATE, HNOISE = 0.30, 1.50, 0.25, 0.003   # rad/s drift

    def __init__(self, host='mock', **kw):
        self.host = host
        self._pids = {m: _MockPid() for m in PID_MODULES}
        self._t0 = time.time()
        self._t_last = self._t0
        self._out = {'out1': 0.0, 'out2': 0.0}
        self._drift = 1.0
        self._phi = 0.0
        print('MockBackend: synthetic two-loop backend (laser + homodyne).')

    def pid(self, module):
        return self._pids[module]

    # --- plants ----------------------------------------------------------
    def _pd_of(self, v: float, noise=True) -> float:
        x = (v - self.V0) / self.VPI
        pd = self._drift * self.AMP * math.sin(0.5 * math.pi * x) ** 2 + self.OFFSET
        return pd + (self.NOISE * np.random.randn() if noise else 0.0)

    def _hom_of(self, v: float, noise=True) -> float:
        h = self.HAMP * math.sin(2 * math.pi * v / self.VPER + self._phi)
        return h + (self.HNOISE * np.random.randn() if noise else 0.0)

    def _input_value(self, name: str, noise=False) -> float:
        if name == 'in1':
            return self._pd_of(self._out['out1'], noise)
        if name == 'in2':
            return self._hom_of(self._out['out2'], noise)
        return 0.0

    def step(self) -> None:
        """Advance the simulation by the wall-clock time since the last call."""
        now = time.time()
        dt = min(0.5, max(1e-3, now - self._t_last))
        self._t_last = now
        self._drift = 1.0 + 0.05 * math.sin(2 * math.pi * 0.07 * (now - self._t0)) \
                          + 0.01 * np.random.randn() * math.sqrt(dt)
        self._phi += self.PHASE_RATE * dt + 0.05 * np.random.randn() * math.sqrt(dt)
        drivers = {}
        for name, pid in self._pids.items():
            if pid.output_direct in self._out:
                drivers.setdefault(pid.output_direct, pid)
        for out in self._out:
            if out not in drivers:
                self._out[out] = 0.0
        n = max(1, int(dt / 1e-3))
        h = dt / n
        for _ in range(n):
            for out, pid in drivers.items():
                err = self._input_value(pid.input) - pid.setpoint
                pid.ival += 2 * math.pi * pid.i * err * h
                pid.ival = min(pid.max_voltage, max(pid.min_voltage, pid.ival))
                v = pid.p * err + pid.ival
                self._out[out] = min(pid.max_voltage, max(pid.min_voltage, v))

    def read_signal(self, name: str, t: float = 0.005):
        if name.startswith('in'):
            return self._input_value(name, noise=True), \
                (self.NOISE if name == 'in1' else self.HNOISE)
        if name in self._out:
            return self._out[name], 0.0
        if name.startswith('pid'):
            pid = self._pids[name]
            if pid.output_direct in self._out:
                return self._out[pid.output_direct], 0.0
            return 0.0, 0.0
        return 0.0, 0.0

    def close(self):
        pass


# ── Shared connection ────────────────────────────────────────────────────────

class SharedLink(QObject):
    """The one PyRPL connection to the board, shared by every lock tab."""

    connected_changed = pyqtSignal(bool)
    message           = pyqtSignal(str)

    def __init__(self, use_mock: bool = False, settings: dict | None = None):
        super().__init__()
        self.mock = use_mock
        self.backend = None
        self.settings = settings if settings is not None else load_settings()

    @property
    def connected(self) -> bool:
        return self.backend is not None

    @property
    def host(self) -> str:
        return self.settings['link']['host']

    @property
    def fpga_loaded(self) -> bool:
        """True once the PyRPL bitstream has been uploaded to this host
        (persisted across GUI restarts and PC reboots - the board keeps
        running it until it is power-cycled or another app is started)."""
        return bool(self.settings['link'].get('fpga_loaded', False))

    def connect_board(self, host: str, config: str, reloadfpga: bool) -> None:
        if self.connected:
            return
        self.backend = (MockBackend(host) if self.mock
                        else PyRPLBackend(host, config, reloadfpga))
        if host != self.settings['link'].get('host'):
            self.settings['link']['fpga_loaded'] = False   # different board
        self.settings['link']['host'] = host
        self.settings['link']['config'] = config
        if reloadfpga:
            self.settings['link']['fpga_loaded'] = True
        save_settings(self.settings)
        self.connected_changed.emit(True)
        self.message.emit(f'Red Pitaya: connected to {host}'
                          + (' (mock)' if self.mock else '')
                          + (' - FPGA reloaded' if reloadfpga and not self.mock else ''))

    def disconnect_board(self) -> None:
        if not self.connected:
            return
        self.backend.close()
        self.backend = None
        self.connected_changed.emit(False)
        self.message.emit('Red Pitaya: disconnected (locks keep running on the board)')

    def step(self) -> None:
        if self.backend is not None:
            self.backend.step()

    def save(self) -> None:
        save_settings(self.settings)


# ── Lock logic ────────────────────────────────────────────────────────────────

class PiLock:
    """Lock / hold / drop / sweep for one pid module on a SharedLink.  No Qt."""

    def __init__(self, link: SharedLink, module: str = 'pid0'):
        self.link = link
        self.module = module
        self.state = STATE_DISCONNECTED
        self.output = 'out1'
        self._saved = None          # (p, i, ival, state) around drop / sweep

    @property
    def connected(self) -> bool:
        return self.link.connected

    @property
    def backend(self):
        return self.link.backend

    @property
    def pid(self):
        return self.backend.pid(self.module)

    def sync_state(self) -> None:
        """Infer the loop state from the registers (e.g. after reconnecting)."""
        if not self.connected:
            self.state = STATE_DISCONNECTED
            return
        pid = self.pid
        if pid.output_direct == 'off':
            self.state = STATE_OFF
        elif pid.p == 0 and pid.i == 0:
            self.state = STATE_HOLD
        else:
            self.state = STATE_LOCKED

    # --- configuration -------------------------------------------------
    def configure(self, module: str, input: str, output: str, setpoint: float,
                  vmin: float, vmax: float, lowpass_hz: float) -> None:
        """Write routing / setpoint / limits.  Gains are written by engage()."""
        self.module = module
        self.output = output
        pid = self.pid
        pid.input = input
        pid.setpoint = setpoint
        pid.min_voltage = vmin
        pid.max_voltage = vmax
        pid.inputfilter = [lowpass_hz if lowpass_hz > 0 else 0, 0, 0, 0]
        if pid.output_direct not in ('off', output):
            pid.output_direct = output       # moved to another output
        # Make sure no other PID drives the same output.
        for other in PID_MODULES:
            if other != module:
                o = self.backend.pid(other)
                if o.output_direct == output:
                    o.output_direct = 'off'

    def set_setpoint(self, setpoint: float) -> None:
        self.pid.setpoint = setpoint

    def set_gains(self, p: float, i: float) -> None:
        pid = self.pid
        pid.p = p
        pid.i = i

    # --- loop control --------------------------------------------------
    def engage(self, p: float, i: float, start_v: float | None = None) -> None:
        """Close the loop.  Output enabled at ``start_v`` (or the present
        integrator value) with zero gains first, then the gains are applied."""
        pid = self.pid
        pid.p = 0.0
        pid.i = 0.0
        if start_v is not None:
            pid.ival = float(np.clip(start_v, pid.min_voltage, pid.max_voltage))
        pid.output_direct = self.output
        pid.p = p
        pid.i = i
        self.state = STATE_LOCKED

    def hold(self) -> None:
        """Freeze the integrator: output stays at its present DC value."""
        pid = self.pid
        pid.p = 0.0
        pid.i = 0.0
        if pid.output_direct != 'off':
            self.state = STATE_HOLD

    def set_output_voltage(self, v: float) -> None:
        """Open-loop: drive a constant voltage (gains zero)."""
        pid = self.pid
        pid.p = 0.0
        pid.i = 0.0
        pid.ival = float(np.clip(v, pid.min_voltage, pid.max_voltage))
        pid.output_direct = self.output
        self.state = STATE_HOLD

    def output_off(self) -> None:
        self.pid.output_direct = 'off'
        self.state = STATE_OFF

    def reset_integrator(self, v: float) -> None:
        """Jump the integrator (used by the rail auto-reset while locked)."""
        pid = self.pid
        pid.ival = float(np.clip(v, pid.min_voltage, pid.max_voltage))

    # --- drop -----------------------------------------------------------
    def drop_begin(self, hold_v: float) -> dict:
        pid = self.pid
        if pid.output_direct == 'off':
            raise RuntimeError('Output is off - nothing to drop from.')
        self._saved = (pid.p, pid.i, pid.ival, self.state)
        pid.p = 0.0
        pid.i = 0.0
        pid.ival = float(np.clip(hold_v, pid.min_voltage, pid.max_voltage))
        self.state = STATE_DROPPING
        return {'saved_ival': self._saved[2], 'hold_v': pid.ival}

    def drop_end(self) -> None:
        self._restore()

    def _restore(self) -> None:
        if self._saved is None:
            return
        p, i, ival, state = self._saved
        pid = self.pid
        pid.ival = ival
        pid.p = p
        pid.i = i
        self.state = state
        self._saved = None

    # --- calibration sweep ---------------------------------------------
    def sweep_begin(self) -> None:
        pid = self.pid
        self._saved = (pid.p, pid.i, pid.ival, self.state)
        pid.p = 0.0
        pid.i = 0.0
        if pid.output_direct == 'off':
            pid.output_direct = self.output
            self._saved = (0.0, 0.0, pid.ival, STATE_OFF)   # go back to OFF after
        self.state = STATE_SWEEP

    def sweep_set(self, v: float) -> None:
        self.pid.ival = float(v)

    def sweep_end(self) -> None:
        saved = self._saved
        self._restore()
        if saved is not None and saved[3] == STATE_OFF:
            self.pid.output_direct = 'off'
            self.state = STATE_OFF

    # --- readout -------------------------------------------------------
    def status(self, t: float = 0.005) -> dict:
        pid = self.pid
        in_m, in_s = self.backend.read_signal(pid.input, t)
        out_m, _ = self.backend.read_signal(self.module, t)
        return {'in': in_m, 'in_std': in_s, 'out': out_m,
                'ival': float(pid.ival), 'setpoint': float(pid.setpoint),
                'vmin': float(pid.min_voltage), 'vmax': float(pid.max_voltage),
                'state': self.state}


# Backwards-compatible name
LaserStabilizer = PiLock


# ── Sweep analysis ────────────────────────────────────────────────────────────

def analyze_sweep(volts, sig, setpoint: float) -> dict:
    """Minimum / maximum, setpoint crossing (steepest), required gain sign,
    and for a periodic (fringe) signal an estimate of the period in volts."""
    v = np.asarray(volts, float)
    y = np.asarray(sig, float)
    out: dict = {}
    if len(v) < 3:
        return out
    k = int(np.argmin(y))
    out['v_min'] = float(v[k]); out['sig_min'] = float(y[k])
    k = int(np.argmax(y))
    out['v_max'] = float(v[k]); out['sig_max'] = float(y[k])
    d = y - setpoint
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    best = None
    crossings = []
    for j in idx:
        dv = v[j + 1] - v[j]
        dy = y[j + 1] - y[j]
        if dv == 0 or dy == 0:
            continue
        slope = dy / dv
        vc = v[j] - d[j] * dv / dy
        crossings.append((vc, slope))
        if best is None or abs(slope) > abs(best[1]):
            best = (vc, slope)
    if best is not None:
        out['v_start'] = float(best[0])
        out['slope'] = float(best[1])
        out['gain_sign'] = -1 if best[1] > 0 else 1
    same = [c for c in crossings if best is not None and np.sign(c[1]) == np.sign(best[1])]
    if len(same) >= 2:
        vs = np.sort([c[0] for c in same])
        out['v_period'] = float(np.median(np.diff(vs)))
    return out


# ── Widgets ───────────────────────────────────────────────────────────────────

def _dspin(lo, hi, val, dec=3, step=None, suffix=''):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(val)
    if step is not None:
        s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    s.setMinimumHeight(26)
    return s


class _StripChart(QWidget):
    """Two stacked time traces (error signal, output) over the last N seconds."""

    def __init__(self, signal_label: str, seconds: float = 60.0):
        super().__init__()
        self.seconds = seconds
        self._t   = collections.deque()
        self._sig = collections.deque()
        self._out = collections.deque()
        self._sp  = collections.deque()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if not _HAS_PG:
            lay.addWidget(QLabel('pyqtgraph not installed - no live plot'))
            return
        self._w1 = pg.PlotWidget(title=f'{signal_label} (V)')
        self._w2 = pg.PlotWidget(title='PID output (V)')
        for w in (self._w1, self._w2):
            w.showGrid(x=True, y=True, alpha=0.3)
            w.setLabel('bottom', 'time', units='s')
        self._w2.setXLink(self._w1)
        self._c_sig = self._w1.plot(pen=pg.mkPen('#00e5ff', width=1.5))
        self._c_sp  = self._w1.plot(pen=pg.mkPen('#ffaa00', width=1,
                                                 style=Qt.DashLine))
        self._c_out = self._w2.plot(pen=pg.mkPen('#e056fd', width=1.5))
        self._rail_lo = pg.InfiniteLine(angle=0, pen=pg.mkPen('#ff3333', style=Qt.DotLine))
        self._rail_hi = pg.InfiniteLine(angle=0, pen=pg.mkPen('#ff3333', style=Qt.DotLine))
        self._w2.addItem(self._rail_lo); self._w2.addItem(self._rail_hi)
        lay.addWidget(self._w1)
        lay.addWidget(self._w2)

    def set_rails(self, lo: float, hi: float) -> None:
        if _HAS_PG:
            self._rail_lo.setPos(lo); self._rail_hi.setPos(hi)

    def append(self, t: float, sig: float, out: float, sp: float) -> None:
        self._t.append(t); self._sig.append(sig); self._out.append(out); self._sp.append(sp)
        while self._t and t - self._t[0] > self.seconds:
            for q in (self._t, self._sig, self._out, self._sp):
                q.popleft()
        if _HAS_PG and self._t:
            tt = np.asarray(self._t) - t
            self._c_sig.setData(tt, np.asarray(self._sig))
            self._c_sp.setData(tt, np.asarray(self._sp))
            self._c_out.setData(tt, np.asarray(self._out))

    def clear(self) -> None:
        for q in (self._t, self._sig, self._out, self._sp):
            q.clear()
        if _HAS_PG:
            for c in (self._c_sig, self._c_sp, self._c_out):
                c.setData([], [])


class _SweepPlot(QWidget):
    def __init__(self, signal_label: str):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if not _HAS_PG:
            lay.addWidget(QLabel('pyqtgraph not installed'))
            return
        self._w = pg.PlotWidget(title=f'Calibration: {signal_label} vs output voltage')
        self._w.showGrid(x=True, y=True, alpha=0.3)
        self._w.setLabel('bottom', 'output', units='V')
        self._w.setLabel('left', signal_label, units='V')
        self._curve = self._w.plot(pen=pg.mkPen('#9ae84f', width=1.5),
                                   symbol='o', symbolSize=4,
                                   symbolBrush='#9ae84f')
        self._sp_line    = pg.InfiniteLine(angle=0, pen=pg.mkPen('#ffaa00', style=Qt.DashLine))
        self._hold_line  = pg.InfiniteLine(angle=90, pen=pg.mkPen('#ff3333', style=Qt.DashLine))
        self._start_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('#00cc44', style=Qt.DashLine))
        for l in (self._sp_line, self._hold_line, self._start_line):
            self._w.addItem(l)
        self._hold_line.hide()
        lay.addWidget(self._w)

    def set_data(self, v, sig) -> None:
        if _HAS_PG:
            self._curve.setData(np.asarray(v), np.asarray(sig))

    def set_markers(self, setpoint=None, hold_v=None, start_v=None) -> None:
        if not _HAS_PG:
            return
        if setpoint is not None:
            self._sp_line.setPos(setpoint)
        if hold_v is not None:
            self._hold_line.setPos(hold_v); self._hold_line.show()
        if start_v is not None:
            self._start_line.setPos(start_v)


# ── Generic PI-lock tab ───────────────────────────────────────────────────────

class PiLockPanel(QWidget):
    """One PI lock tab.  Subclasses add feature sections and set labels.

    Signals
    -------
    state_changed(str)      one of the STATE_* constants
    message(str)            status-bar text for the host window
    """

    state_changed = pyqtSignal(str)
    message       = pyqtSignal(str)

    SETTINGS_KEY  = 'laser'
    DEFAULTS      = LASER_DEFAULTS
    SIGNAL_LABEL  = 'Photodiode'
    TITLE         = 'PI loop'

    def __init__(self, link: SharedLink, parent=None):
        super().__init__(parent)
        self.link = link
        self._mock = link.mock
        self._lock = PiLock(link)
        self._s = link.settings.setdefault(self.SETTINGS_KEY, dict(self.DEFAULTS))
        for k, v in self.DEFAULTS.items():
            self._s.setdefault(k, v)

        self._poll = QTimer(self)
        self._poll.timeout.connect(self._on_poll)

        self._sweep_timer = QTimer(self)
        self._sweep_timer.timeout.connect(self._sweep_tick)
        self._sweep_v: np.ndarray | None = None
        self._sweep_i = 0
        self._sweep_sig: list = []
        self._sweep_phase = 0
        self._sweep_result: dict = {}
        self._last_status: dict | None = None

        self._build_ui()
        self._apply_settings(self._s)
        link.connected_changed.connect(self._on_link_changed)
        link.message.connect(self.message)
        self._on_link_changed(link.connected)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        col = QWidget()
        self._col_layout = QVBoxLayout(col)
        self._col_layout.setContentsMargins(2, 2, 6, 2)
        self._col_layout.setSpacing(8)
        self._col_layout.addWidget(self._build_connection_group())
        self._col_layout.addWidget(self._build_loop_group())
        self._col_layout.addWidget(self._build_readout_group())
        self._build_extra_groups()          # subclass sections
        self._col_layout.addWidget(self._build_sweep_group())
        self._col_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(col)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(460)

        right = QSplitter(Qt.Vertical)
        self._strip = _StripChart(self.SIGNAL_LABEL, seconds=60.0)
        self._sweep_plot = _SweepPlot(self.SIGNAL_LABEL)
        right.addWidget(self._strip)
        right.addWidget(self._sweep_plot)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

        root.addWidget(scroll)
        root.addWidget(right, 1)

    def _build_extra_groups(self) -> None:      # overridden by subclasses
        pass

    def _build_connection_group(self) -> QGroupBox:
        g = QGroupBox('Red Pitaya (PyRPL) — shared by all lock tabs')
        lay = QGridLayout(g)
        self._host_edit = QLineEdit(self.link.settings['link']['host'])
        self._config_edit = QLineEdit(self.link.settings['link']['config'])
        self._config_edit.setToolTip('PyRPL config name (stored in the PyRPL '
                                     'user dir).')
        self._reload_chk = QCheckBox('Reload FPGA bitstream on connect')
        self._reload_chk.setChecked(not self.link.fpga_loaded)
        self._reload_chk.setToolTip(
            'Required the first time after the board ran another app\n'
            '(e.g. Oscilloscope+Lock-in+PID) or was power-cycled.\n'
            'Reloading ZEROES all PID outputs: a running lock is lost and\n'
            'a trapped particle is dropped. Leave unchecked to reconnect\n'
            'to running PyRPL locks (the default once the bitstream has\n'
            'been loaded, also after a PC reboot).')
        self._connect_btn = QPushButton('Connect')
        self._disconnect_btn = QPushButton('Disconnect')
        self._connect_btn.clicked.connect(self._on_connect)
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._conn_lbl = QLabel('Not connected')
        self._conn_lbl.setStyleSheet('color:#888;')
        r = 0
        lay.addWidget(QLabel('Host:'), r, 0); lay.addWidget(self._host_edit, r, 1, 1, 2); r += 1
        lay.addWidget(QLabel('Config:'), r, 0); lay.addWidget(self._config_edit, r, 1, 1, 2); r += 1
        lay.addWidget(self._reload_chk, r, 0, 1, 3); r += 1
        lay.addWidget(self._connect_btn, r, 0, 1, 2); lay.addWidget(self._disconnect_btn, r, 2); r += 1
        lay.addWidget(self._conn_lbl, r, 0, 1, 3)
        return g

    def _build_loop_group(self) -> QGroupBox:
        g = QGroupBox(self.TITLE)
        lay = QGridLayout(g)
        lay.setVerticalSpacing(6)
        self._module_cb = QComboBox(); self._module_cb.addItems(PID_MODULES)
        self._input_cb  = QComboBox(); self._input_cb.addItems(INPUTS)
        self._output_cb = QComboBox(); self._output_cb.addItems(OUTPUTS)

        big = QFont('Consolas', 12, QFont.Bold)
        self._setpoint_sp = _dspin(-1.0, 1.0, 0.0, 4, 0.005, ' V')
        self._setpoint_sp.setFont(big)
        self._setpoint_sp.setMinimumHeight(32)
        self._setpoint_sp.setToolTip(f'{self.SIGNAL_LABEL} voltage to hold. '
                                     'Arrow keys / wheel step by 5 mV.')
        self._setpoint_sp.valueChanged.connect(self._on_setpoint_edited)
        self._sp_now_btn = QPushButton('← current')
        self._sp_now_btn.setToolTip(f'Set the setpoint to the present '
                                    f'{self.SIGNAL_LABEL.lower()} reading')
        self._sp_now_btn.clicked.connect(self._on_setpoint_from_current)

        self._p_sp = _dspin(-1000.0, 1000.0, 0.0, 4, 0.01)
        self._p_sp.setToolTip('Proportional gain (dimensionless). Negative if '
                              f'the {self.SIGNAL_LABEL.lower()} rises with output voltage.')
        self._i_sp = _dspin(-1e6, 1e6, 0.0, 1, 10.0, ' Hz')
        self._i_sp.setToolTip('Integrator unity-gain frequency. Same sign as P.')
        self._vmin_sp = _dspin(-1.0, 1.0, -1.0, 3, 0.05, ' V')
        self._vmax_sp = _dspin(-1.0, 1.0,  1.0, 3, 0.05, ' V')
        self._lp_sp = _dspin(0.0, 1e6, 0.0, 0, 100.0, ' Hz')
        self._lp_sp.setToolTip('Input low-pass corner (0 = off)')
        self._start_sp = _dspin(-1.0, 1.0, 0.0, 3, 0.01, ' V')
        self._start_sp.setToolTip('Output voltage the loop starts from when '
                                  'engaged from OFF (use the sweep to find it)')

        self._engage_btn = QPushButton('Engage lock')
        self._engage_btn.setCheckable(True)
        self._engage_btn.setMinimumHeight(34)
        self._engage_btn.setFont(QFont('Arial', 10, QFont.Bold))
        self._engage_btn.setStyleSheet(
            'QPushButton { background:#2a5c2a; color:#ccc; border-radius:4px; }'
            'QPushButton:checked { background:#00802b; color:white; }'
            'QPushButton:disabled { background:#333; color:#666; }')
        self._engage_btn.clicked.connect(self._on_engage_clicked)
        self._hold_btn = QPushButton('Hold output')
        self._hold_btn.setToolTip('Freeze the integrator: output stays at its '
                                  'present value, loop open')
        self._hold_btn.clicked.connect(self._on_hold)
        self._set_v_btn = QPushButton('Drive start V')
        self._set_v_btn.setToolTip('Open loop: drive the start voltage as a '
                                   'constant DC output')
        self._set_v_btn.clicked.connect(self._on_set_start_v)
        self._off_btn = QPushButton('Output OFF')
        self._off_btn.setToolTip('Output to 0 V')
        self._off_btn.clicked.connect(self._on_output_off)
        self._apply_btn = QPushButton('Apply gains / limits / filter')
        self._apply_btn.setToolTip('Write setpoint, gains, limits and filter '
                                   'to the board without changing the loop state')
        self._apply_btn.clicked.connect(self._on_apply)

        r = 0
        lay.addWidget(QLabel('Module:'), r, 0); lay.addWidget(self._module_cb, r, 1)
        lay.addWidget(QLabel('In:'), r, 2); lay.addWidget(self._input_cb, r, 3)
        lay.addWidget(QLabel('Out:'), r, 4); lay.addWidget(self._output_cb, r, 5); r += 1
        lay.addWidget(QLabel('Setpoint:'), r, 0)
        lay.addWidget(self._setpoint_sp, r, 1, 1, 3)
        lay.addWidget(self._sp_now_btn, r, 4, 1, 2); r += 1
        lay.addWidget(QLabel('P gain:'), r, 0); lay.addWidget(self._p_sp, r, 1, 1, 5); r += 1
        lay.addWidget(QLabel('I (Hz):'), r, 0); lay.addWidget(self._i_sp, r, 1, 1, 5); r += 1
        lay.addWidget(QLabel('Vmin:'), r, 0); lay.addWidget(self._vmin_sp, r, 1, 1, 2)
        lay.addWidget(QLabel('Vmax:'), r, 3); lay.addWidget(self._vmax_sp, r, 4, 1, 2); r += 1
        lay.addWidget(QLabel('Low-pass:'), r, 0); lay.addWidget(self._lp_sp, r, 1, 1, 5); r += 1
        lay.addWidget(QLabel('Start V:'), r, 0); lay.addWidget(self._start_sp, r, 1, 1, 2)
        lay.addWidget(self._set_v_btn, r, 3, 1, 3); r += 1
        lay.addWidget(self._apply_btn, r, 0, 1, 6); r += 1
        lay.addWidget(self._engage_btn, r, 0, 1, 6); r += 1
        lay.addWidget(self._hold_btn, r, 0, 1, 3); lay.addWidget(self._off_btn, r, 3, 1, 3)
        for c in range(6):
            lay.setColumnStretch(c, 1 if c % 2 else 0)
        return g

    def _build_readout_group(self) -> QGroupBox:
        g = QGroupBox('Readout')
        lay = QGridLayout(g)
        big = QFont('Consolas', 13, QFont.Bold)
        self._state_lbl = QLabel('—')
        self._state_lbl.setFont(QFont('Arial', 11, QFont.Bold))
        self._state_lbl.setAlignment(Qt.AlignCenter)
        self._state_lbl.setMinimumHeight(28)
        self._sig_lbl = QLabel('—'); self._sig_lbl.setFont(big)
        self._err_lbl = QLabel('—')
        self._out_lbl = QLabel('—'); self._out_lbl.setFont(big)
        r = 0
        lay.addWidget(self._state_lbl, r, 0, 1, 2); r += 1
        lay.addWidget(QLabel(f'{self.SIGNAL_LABEL}:'), r, 0); lay.addWidget(self._sig_lbl, r, 1); r += 1
        lay.addWidget(QLabel('Error / rms:'), r, 0); lay.addWidget(self._err_lbl, r, 1); r += 1
        lay.addWidget(QLabel('Output:'), r, 0); lay.addWidget(self._out_lbl, r, 1)
        lay.setColumnStretch(1, 1)
        return g

    def _build_sweep_group(self) -> QGroupBox:
        g = QGroupBox('Calibration sweep')
        lay = QGridLayout(g)
        self._sweep_btn = QPushButton('Sweep output Vmin → Vmax')
        self._sweep_btn.setToolTip(
            f'Open the loop and step the output while reading the '
            f'{self.SIGNAL_LABEL.lower()}.\nFinds the setpoint crossing (lock '
            'start voltage) and the gain sign.\nWARNING: this ramps the output '
            'over its full range.')
        self._sweep_btn.setCheckable(True)
        self._sweep_btn.clicked.connect(self._on_sweep_clicked)
        self._sweep_pts_sp = QSpinBox()
        self._sweep_pts_sp.setRange(11, 401); self._sweep_pts_sp.setValue(81)
        self._sweep_pts_sp.setSuffix(' pts')
        self._sweep_lbl = QLabel('No sweep yet')
        self._sweep_lbl.setWordWrap(True)
        self._sweep_lbl.setStyleSheet('color:#aaa; font-size:9pt;')
        self._use_start_btn = QPushButton('Start V ← crossing')
        self._use_start_btn.setEnabled(False)
        self._use_start_btn.clicked.connect(self._on_use_start)
        self._use_sign_btn = QPushButton('Fix gain sign')
        self._use_sign_btn.setEnabled(False)
        self._use_sign_btn.setToolTip('Flip P and I to the sign the sweep requires')
        self._use_sign_btn.clicked.connect(self._on_use_sign)
        r = 0
        lay.addWidget(self._sweep_btn, r, 0, 1, 2); lay.addWidget(self._sweep_pts_sp, r, 2); r += 1
        lay.addWidget(self._sweep_lbl, r, 0, 1, 3); r += 1
        lay.addWidget(self._use_start_btn, r, 0, 1, 2); lay.addWidget(self._use_sign_btn, r, 2); r += 1
        self._sweep_extra_row = (lay, r)
        return g

    # ── settings ──────────────────────────────────────────────────────────
    def _apply_settings(self, s: dict) -> None:
        self._module_cb.setCurrentText(s['module'])
        self._input_cb.setCurrentText(s['input'])
        self._output_cb.setCurrentText(s['output'])
        self._setpoint_sp.blockSignals(True)
        self._setpoint_sp.setValue(float(s['setpoint']))
        self._setpoint_sp.blockSignals(False)
        self._p_sp.setValue(float(s['p']))
        self._i_sp.setValue(float(s['i']))
        self._vmin_sp.setValue(float(s['vmin']))
        self._vmax_sp.setValue(float(s['vmax']))
        self._lp_sp.setValue(float(s['lowpass_hz']))
        self._start_sp.setValue(float(s['start_v']))
        self._sweep_plot.set_markers(setpoint=self._setpoint_sp.value(),
                                     start_v=self._start_sp.value())
        self._strip.set_rails(self._vmin_sp.value(), self._vmax_sp.value())

    def _collect_settings(self) -> dict:
        return {
            'module': self._module_cb.currentText(),
            'input': self._input_cb.currentText(),
            'output': self._output_cb.currentText(),
            'setpoint': self._setpoint_sp.value(),
            'p': self._p_sp.value(), 'i': self._i_sp.value(),
            'vmin': self._vmin_sp.value(), 'vmax': self._vmax_sp.value(),
            'lowpass_hz': self._lp_sp.value(),
            'start_v': self._start_sp.value(),
        }

    def save(self) -> None:
        self._s.update(self._collect_settings())
        self.link.settings['link']['host'] = self._host_edit.text().strip()
        self.link.settings['link']['config'] = self._config_edit.text().strip() or 'laser_stab'
        self.link.save()

    # ── connection (shared) ───────────────────────────────────────────────
    def _on_connect(self) -> None:
        host = self._host_edit.text().strip()
        cfg = self._config_edit.text().strip() or 'laser_stab'
        reload = self._reload_chk.isChecked()
        if reload and not self._mock and not self._confirm_reload(host):
            return
        self._conn_lbl.setText('Connecting…')
        self._conn_lbl.repaint()
        try:
            self.link.connect_board(host, cfg, reload)
        except Exception as exc:
            self.link.backend = None
            self._conn_lbl.setText(f'Connect failed: {exc}')
            self._conn_lbl.setStyleSheet('color:#ff5555;')
            self.message.emit(f'Red Pitaya: connect failed: {exc}')

    def _confirm_reload(self, host: str) -> bool:
        """Reloading the bitstream zeroes every PID register: any lock that
        is still running on the board (e.g. after a PC reboot) is lost and
        the particle drops.  Make the operator choose explicitly."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle('Reload FPGA bitstream?')
        box.setText(f'Reload the PyRPL FPGA bitstream on {host}?')
        box.setInformativeText(
            'Reloading resets ALL PID registers: every output goes to 0 V.\n'
            'If a lock is still running on the board (laser power lock, '
            'homodyne lock) it will be lost and a trapped particle will be '
            'dropped.\n\n'
            + ('The bitstream was already loaded on this host before - '
               'reloading is normally NOT needed.'
               if self.link.fpga_loaded else
               'Only needed if the board was power-cycled or ran another '
               'app (e.g. Oscilloscope+Lock-in+PID) since.'))
        reload_btn = box.addButton('Reload (resets outputs)',
                                   QMessageBox.DestructiveRole)
        keep_btn = box.addButton('Connect without reload',
                                 QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(keep_btn)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is keep_btn:
            self._reload_chk.setChecked(False)
            return True
        return clicked is reload_btn

    def _on_disconnect(self) -> None:
        self.link.disconnect_board()

    def _on_link_changed(self, on: bool) -> None:
        """Every panel reacts to the shared connection coming or going."""
        self._connect_btn.setEnabled(not on)
        self._disconnect_btn.setEnabled(on)
        self._host_edit.setEnabled(not on)
        self._config_edit.setEnabled(not on)
        if on:
            self._host_edit.setText(self.link.settings['link']['host'])
            self._config_edit.setText(self.link.settings['link']['config'])
            self._reload_chk.setChecked(False)     # a reconnect must not reload
            self._conn_lbl.setText(f'Connected: {self.link.backend.host}'
                                   + ('  [mock]' if self._mock else ''))
            self._conn_lbl.setStyleSheet('color:#00cc44;')
            self._lock.module = self._module_cb.currentText()
            self._lock.output = self._output_cb.currentText()
            try:
                self._lock.sync_state()
                if self._lock.state in (STATE_LOCKED, STATE_HOLD):
                    # A lock is running on the board (GUI restart / PC
                    # reboot): take its settings, do not write ours.
                    self._adopt_board_state()
                    self.message.emit(
                        f'{self.TITLE}: lock already running on the board '
                        f'({self._lock.state}) - adopted its settings, '
                        f'nothing written')
                else:
                    # Output is off: routing / setpoint / limits follow the
                    # panel; gains are written by engage().
                    self._write_config()
            except Exception as exc:
                self.message.emit(f'{self.TITLE}: config write failed: {exc}')
            self._strip.clear()
            self._poll.start(int(1000 / POLL_HZ))
            self.save()
        else:
            self._poll.stop()
            self._abort_sweep()
            self._on_disconnected_extra()
            self._lock.state = STATE_DISCONNECTED
            self._conn_lbl.setText('Not connected')
            self._conn_lbl.setStyleSheet('color:#888;')
            self._state_lbl.setText('—')
            self._state_lbl.setStyleSheet('')
            self._sweep_btn.setChecked(False)
            self.save()
        self._update_state_widgets()

    def _on_disconnected_extra(self) -> None:   # subclass hook
        pass

    def _adopt_board_state(self) -> None:
        """Read routing / setpoint / gains / limits / filter of a running
        lock into the panel widgets without touching the registers."""
        pid = self._lock.pid
        self._lock.output = str(pid.output_direct)
        widgets = (self._setpoint_sp, self._p_sp, self._i_sp,
                   self._vmin_sp, self._vmax_sp, self._lp_sp,
                   self._input_cb, self._output_cb)
        for w in widgets:
            w.blockSignals(True)
        try:
            self._input_cb.setCurrentText(str(pid.input))
            self._output_cb.setCurrentText(str(pid.output_direct))
            self._setpoint_sp.setValue(float(pid.setpoint))
            if self._lock.state == STATE_LOCKED:      # HOLD has p = i = 0
                self._p_sp.setValue(float(pid.p))
                self._i_sp.setValue(float(pid.i))
            self._vmin_sp.setValue(float(pid.min_voltage))
            self._vmax_sp.setValue(float(pid.max_voltage))
            try:
                lp = float(pid.inputfilter[0])
            except Exception:
                lp = 0.0
            self._lp_sp.setValue(max(lp, 0.0))
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._sweep_plot.set_markers(setpoint=self._setpoint_sp.value())
        self._strip.set_rails(self._vmin_sp.value(), self._vmax_sp.value())

    # ── loop actions ──────────────────────────────────────────────────────
    def _write_config(self) -> None:
        self._lock.configure(
            module=self._module_cb.currentText(),
            input=self._input_cb.currentText(),
            output=self._output_cb.currentText(),
            setpoint=self._setpoint_sp.value(),
            vmin=self._vmin_sp.value(), vmax=self._vmax_sp.value(),
            lowpass_hz=self._lp_sp.value())
        self._strip.set_rails(self._vmin_sp.value(), self._vmax_sp.value())

    def _busy(self) -> bool:
        return self._lock.state in (STATE_DROPPING, STATE_SWEEP)

    def _guarded(self, fn, label: str) -> bool:
        try:
            fn()
            return True
        except Exception as exc:
            self.message.emit(f'{self.TITLE}: {label} failed: {exc}')
            QMessageBox.warning(self, self.TITLE, f'{label} failed:\n{exc}')
            return False
        finally:
            self._update_state_widgets()

    def _on_setpoint_edited(self, v: float) -> None:
        """Setpoint changes are written live (while connected, not busy)."""
        self._sweep_plot.set_markers(setpoint=v)
        if self._lock.connected and not self._busy():
            try:
                self._lock.set_setpoint(v)
            except Exception as exc:
                self.message.emit(f'{self.TITLE}: setpoint write failed: {exc}')
        self.save()

    def _on_setpoint_from_current(self) -> None:
        if self._last_status is not None:
            self._setpoint_sp.setValue(round(self._last_status['in'], 4))

    def _on_apply(self) -> None:
        def do():
            self._write_config()
            if self._lock.state == STATE_LOCKED:
                self._lock.set_gains(self._p_sp.value(), self._i_sp.value())
            self.save()
        if self._guarded(do, 'Apply'):
            self.message.emit(f'{self.TITLE}: settings written')

    def _on_engage_clicked(self, checked: bool) -> None:
        if self._busy():
            self._engage_btn.setChecked(not checked)
            return
        if checked:
            def do():
                self._write_config()
                start = self._start_sp.value() if self._lock.state == STATE_OFF else None
                self._lock.engage(self._p_sp.value(), self._i_sp.value(), start_v=start)
                self.save()
            if self._guarded(do, 'Engage'):
                self.message.emit(f'{self.TITLE}: lock engaged')
        else:
            if self._guarded(self._lock.hold, 'Hold'):
                self.message.emit(f'{self.TITLE}: lock opened (output held)')

    def _on_hold(self) -> None:
        if self._busy():
            return
        if self._guarded(self._lock.hold, 'Hold'):
            self.message.emit(f'{self.TITLE}: output held')

    def _on_set_start_v(self) -> None:
        if self._busy():
            return
        def do():
            self._write_config()
            self._lock.set_output_voltage(self._start_sp.value())
        if self._guarded(do, 'Drive output'):
            self.message.emit(f'{self.TITLE}: output = {self._start_sp.value():.3f} V (open loop)')

    def _on_output_off(self) -> None:
        if self._busy():
            return
        if self._guarded(self._lock.output_off, 'Output off'):
            self.message.emit(f'{self.TITLE}: output OFF')

    # ── calibration sweep ─────────────────────────────────────────────────
    def _sweep_warning(self) -> str:
        return ('The loop is locked. The sweep opens the loop and ramps the\n'
                'output over its full range.\nContinue?')

    def _on_sweep_clicked(self, checked: bool) -> None:
        if not checked:
            self._abort_sweep()
            return
        if self._lock.state == STATE_DROPPING:
            self._sweep_btn.setChecked(False)
            return
        if self._lock.state == STATE_LOCKED:
            ans = QMessageBox.question(self, 'Calibration sweep', self._sweep_warning(),
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                self._sweep_btn.setChecked(False)
                return
        def do():
            self._write_config()
            self._lock.sweep_begin()
        if not self._guarded(do, 'Sweep'):
            self._sweep_btn.setChecked(False)
            return
        n = self._sweep_pts_sp.value()
        self._sweep_v = np.linspace(self._vmin_sp.value(), self._vmax_sp.value(), n)
        self._sweep_i = 0
        self._sweep_sig = []
        self._sweep_phase = 0
        self._sweep_plot.set_data([], [])
        self._sweep_lbl.setText('Sweeping…')
        self._sweep_timer.start(25)

    def _sweep_tick(self) -> None:
        try:
            if self._sweep_phase == 0:                       # set voltage
                self._lock.sweep_set(float(self._sweep_v[self._sweep_i]))
                self._sweep_phase = 1
                return
            self.link.step()
            sig, _ = self._lock.backend.read_signal(self._input_cb.currentText(), 0.005)
            self._sweep_sig.append(sig)
            self._sweep_plot.set_data(self._sweep_v[:len(self._sweep_sig)], self._sweep_sig)
            self._sweep_i += 1
            self._sweep_phase = 0
            if self._sweep_i >= len(self._sweep_v):
                self._end_sweep(completed=True)
        except Exception as exc:
            self.message.emit(f'{self.TITLE}: sweep error: {exc}')
            self._end_sweep(completed=False)

    def _abort_sweep(self) -> None:
        if self._sweep_timer.isActive():
            self._end_sweep(completed=False)

    def _end_sweep(self, completed: bool) -> None:
        self._sweep_timer.stop()
        if self._lock.connected:
            try:
                self._lock.sweep_end()
            except Exception as exc:
                self.message.emit(f'{self.TITLE}: restore after sweep failed: {exc}')
        self._sweep_btn.setChecked(False)
        self._update_state_widgets()
        if not completed:
            self._sweep_lbl.setText('Sweep aborted')
            return
        res = analyze_sweep(self._sweep_v[:len(self._sweep_sig)], self._sweep_sig,
                            self._setpoint_sp.value())
        self._sweep_result = res
        if not res:
            self._sweep_lbl.setText('Sweep too short')
            return
        self._sweep_lbl.setText(self._describe_sweep(res))
        self._use_start_btn.setEnabled('v_start' in res)
        wrong = ('gain_sign' in res and self._p_sp.value() != 0
                 and np.sign(self._p_sp.value()) != res['gain_sign'])
        self._use_sign_btn.setEnabled(bool(wrong))
        self._on_sweep_done(res)
        self.message.emit(f'{self.TITLE}: sweep complete')

    def _describe_sweep(self, res: dict) -> str:
        txt = (f'Min {res["sig_min"]:.3f} V at {res["v_min"]:+.3f} V; '
               f'max {res["sig_max"]:.3f} V at {res["v_max"]:+.3f} V.')
        if 'v_start' in res:
            sign = 'NEGATIVE' if res['gain_sign'] < 0 else 'POSITIVE'
            txt += (f'  Setpoint crossing at {res["v_start"]:+.3f} V, slope '
                    f'{res["slope"]:+.2f} V/V → gains must be {sign}.')
            cur = self._p_sp.value()
            if cur != 0 and np.sign(cur) != res['gain_sign']:
                txt += '  ⚠ current P/I have the wrong sign!'
        else:
            txt += '  Setpoint outside the sweep range - no lock point.'
        return txt

    def _on_sweep_done(self, res: dict) -> None:   # subclass hook
        pass

    def _on_use_start(self) -> None:
        if 'v_start' in self._sweep_result:
            self._start_sp.setValue(self._sweep_result['v_start'])
            self._sweep_plot.set_markers(start_v=self._start_sp.value())
            self.save()

    def _on_use_sign(self) -> None:
        sgn = self._sweep_result.get('gain_sign')
        if sgn:
            self._p_sp.setValue(sgn * abs(self._p_sp.value()))
            self._i_sp.setValue(sgn * abs(self._i_sp.value()))
            self._use_sign_btn.setEnabled(False)
            self.save()
            if self._lock.state == STATE_LOCKED:
                self._guarded(lambda: self._lock.set_gains(self._p_sp.value(), self._i_sp.value()),
                              'Gain update')

    # ── polling ───────────────────────────────────────────────────────────
    def _on_poll(self) -> None:
        if not self._lock.connected:
            return
        now = time.time()
        try:
            self.link.step()
            st = self._lock.status(t=0.005)
        except Exception as exc:
            self.message.emit(f'{self.TITLE}: readout error: {exc}')
            return
        self._last_status = st
        sig, out, sp = st['in'], st['out'], st['setpoint']
        self._sig_lbl.setText(f'{sig:+.4f} V')
        self._err_lbl.setText(f'{sig - sp:+.4f} V / {st["in_std"] * 1e3:.1f} mV')
        self._out_lbl.setText(f'{out:+.4f} V')
        self._strip.append(now, sig, out, sp)
        self._sp_now_btn.setEnabled(True)
        self._on_status(st)
        if not self._sweep_timer.isActive():
            self._update_state_widgets()

    def _on_status(self, st: dict) -> None:       # subclass hook
        pass

    def _update_state_widgets(self) -> None:
        st = self._lock.state if self._lock.connected else STATE_DISCONNECTED
        self._state_lbl.setText(st.upper())
        self._state_lbl.setStyleSheet(
            f'color:white; background:{_STATE_COLORS.get(st, "#444")}; '
            f'border-radius:4px; padding:3px;')
        busy = st in (STATE_DROPPING, STATE_SWEEP)
        self._engage_btn.blockSignals(True)
        self._engage_btn.setChecked(st == STATE_LOCKED)
        self._engage_btn.setText('Lock engaged' if st == STATE_LOCKED else 'Engage lock')
        self._engage_btn.blockSignals(False)
        conn = self._lock.connected
        for w in (self._engage_btn, self._hold_btn, self._off_btn,
                  self._set_v_btn, self._apply_btn):
            w.setEnabled(conn and not busy)
        self._sp_now_btn.setEnabled(conn and self._last_status is not None)
        self._sweep_btn.setEnabled(conn and st != STATE_DROPPING)
        self._update_extra_widgets(st)
        self.state_changed.emit(st)

    def _update_extra_widgets(self, st: str) -> None:   # subclass hook
        pass

    # ── lifecycle ─────────────────────────────────────────────────────────
    def shutdown(self) -> None:
        """Leaves the board's registers untouched (engaged locks keep running)."""
        self._poll.stop()
        if self._sweep_timer.isActive():
            self._end_sweep(completed=False)
        self.save()


# ── Laser power lock + particle drop ──────────────────────────────────────────

class LaserStabPanel(PiLockPanel):
    """Laser power stabilization (EOM + PBS) with the particle-drop feature."""

    dropped = pyqtSignal(dict)

    SETTINGS_KEY = 'laser'
    DEFAULTS     = LASER_DEFAULTS
    SIGNAL_LABEL = 'Photodiode'
    TITLE        = 'Laser power PI loop'

    def __init__(self, link: SharedLink, parent=None, embed_drop: bool = True):
        """``embed_drop=False`` builds the drop group but leaves it out of this
        tab so the host window can place ``self.drop_group`` elsewhere (the
        trap monitor puts it under the live camera view)."""
        self._embed_drop = embed_drop
        self._drop_timer = QTimer()
        self._drop_timer.setSingleShot(True)
        self._drop_info: dict | None = None
        super().__init__(link, parent)
        self._drop_timer.setParent(self)
        self._drop_timer.timeout.connect(self._finish_drop)

    def _build_extra_groups(self) -> None:
        g = QGroupBox('Particle drop (laser lock)')
        self.drop_group = g
        lay = QGridLayout(g)
        lay.setHorizontalSpacing(10)
        self._lock_state_lbl = QLabel('LASER LOCK: —')
        self._lock_state_lbl.setFont(QFont('Arial', 9, QFont.Bold))
        self._lock_state_lbl.setAlignment(Qt.AlignCenter)
        self._lock_state_lbl.setMinimumWidth(150)
        self._lock_state_lbl.setToolTip('State of the laser power lock '
                                        '(Laser stabilization tab)')
        self._hold_v_sp = _dspin(-1.0, 1.0, -0.9, 3, 0.01, ' V')
        self._hold_v_sp.setToolTip('Output voltage giving minimum transmission '
                                   '(extinction). Found by the sweep.')
        self._dwell_sp = QSpinBox()
        self._dwell_sp.setRange(10, 60000); self._dwell_sp.setValue(300)
        self._dwell_sp.setSuffix(' ms'); self._dwell_sp.setSingleStep(50)
        self._dwell_sp.setMinimumHeight(26)
        self._dwell_sp.setToolTip('How long the trap stays dark')
        self._confirm_chk = QCheckBox('Confirm before drop')
        self._use_hold_btn = QPushButton('Hold V ← sweep minimum')
        self._use_hold_btn.setEnabled(False)
        self._use_hold_btn.clicked.connect(self._on_use_hold)
        self._drop_btn = QPushButton('DROP PARTICLE')
        self._drop_btn.setMinimumHeight(42)
        self._drop_btn.setFont(QFont('Arial', 11, QFont.Bold))
        self._drop_btn.setStyleSheet(
            'QPushButton { background:#8b0000; color:white; border-radius:4px; }'
            'QPushButton:pressed { background:#ff3333; }'
            'QPushButton:disabled { background:#333; color:#666; }')
        self._drop_btn.clicked.connect(self.request_drop)
        if self._embed_drop:
            # vertical, in the tab's control column
            r = 0
            lay.addWidget(self._lock_state_lbl, r, 0, 1, 3); r += 1
            lay.addWidget(QLabel('Hold V:'), r, 0); lay.addWidget(self._hold_v_sp, r, 1)
            lay.addWidget(self._use_hold_btn, r, 2); r += 1
            lay.addWidget(QLabel('Dwell:'), r, 0); lay.addWidget(self._dwell_sp, r, 1)
            lay.addWidget(self._confirm_chk, r, 2); r += 1
            lay.addWidget(self._drop_btn, r, 0, 1, 3)
            lay.setColumnStretch(1, 1); lay.setColumnStretch(2, 1)
            self._col_layout.addWidget(g)
        else:
            # one wide row, meant to sit under the live camera view
            self._drop_btn.setMinimumWidth(200)
            lay.addWidget(self._lock_state_lbl, 0, 0, 2, 1)
            lay.addWidget(QLabel('Hold V:'), 0, 1); lay.addWidget(self._hold_v_sp, 0, 2)
            lay.addWidget(QLabel('Dwell:'), 1, 1); lay.addWidget(self._dwell_sp, 1, 2)
            lay.addWidget(self._use_hold_btn, 0, 3)
            lay.addWidget(self._confirm_chk, 1, 3)
            lay.addWidget(self._drop_btn, 0, 4, 2, 1)
            lay.setColumnStretch(4, 1)
            # the sweep tab keeps a pointer to Hold V via _use_hold_btn/_hold_v_sp;
            # the group itself is placed by the host window.

    def _apply_settings(self, s: dict) -> None:
        super()._apply_settings(s)
        self._hold_v_sp.setValue(float(s['hold_v']))
        self._dwell_sp.setValue(int(s['dwell_ms']))
        self._confirm_chk.setChecked(bool(s['confirm_drop']))
        self._sweep_plot.set_markers(hold_v=self._hold_v_sp.value())

    def _collect_settings(self) -> dict:
        d = super()._collect_settings()
        d.update({'hold_v': self._hold_v_sp.value(),
                  'dwell_ms': self._dwell_sp.value(),
                  'confirm_drop': self._confirm_chk.isChecked()})
        return d

    def _sweep_warning(self) -> str:
        return ('The loop is locked. The sweep opens the loop and ramps the\n'
                'laser power over its full range - a trapped particle will be\n'
                'lost. Continue?')

    def _describe_sweep(self, res: dict) -> str:
        txt = super()._describe_sweep(res)
        return txt.replace('Min ', 'Extinction: PD ', 1)

    def _on_sweep_done(self, res: dict) -> None:
        self._use_hold_btn.setEnabled('v_min' in res)

    def _on_use_hold(self) -> None:
        if 'v_min' in self._sweep_result:
            self._hold_v_sp.setValue(self._sweep_result['v_min'])
            self._sweep_plot.set_markers(hold_v=self._hold_v_sp.value())
            self.save()

    def _update_extra_widgets(self, st: str) -> None:
        self._drop_btn.setEnabled(self.can_drop())
        self._drop_btn.setText('DROPPING…' if st == STATE_DROPPING else 'DROP PARTICLE')
        self._lock_state_lbl.setText(f'LASER LOCK: {st.upper()}')
        self._lock_state_lbl.setStyleSheet(
            f'color:white; background:{_STATE_COLORS.get(st, "#444")}; '
            f'border-radius:4px; padding:3px;')
        if st == STATE_DISCONNECTED:
            self._drop_btn.setToolTip('Connect the Red Pitaya and engage the '
                                      'laser lock on the Laser stabilization tab')
        elif st == STATE_OFF:
            self._drop_btn.setToolTip('Laser lock output is OFF - engage the lock first')
        else:
            self._drop_btn.setToolTip('Hold the laser at the extinction voltage '
                                      'for the dwell time, then restore the lock')

    def _on_disconnected_extra(self) -> None:
        if self._drop_timer.isActive():
            self._drop_timer.stop()
            self._drop_info = None

    # --- drop ------------------------------------------------------------
    def can_drop(self) -> bool:
        return self._lock.connected and self._lock.state in (STATE_LOCKED, STATE_HOLD)

    def request_drop(self) -> None:
        """Public entry point (also used by the trap-monitor Drop button)."""
        if not self.can_drop():
            self.message.emit('Laser: cannot drop - loop output not active')
            return
        if self._confirm_chk.isChecked():
            ans = QMessageBox.question(
                self, 'Drop particle',
                f'Set output to {self._hold_v_sp.value():.3f} V for '
                f'{self._dwell_sp.value()} ms, then restore the lock?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return
        try:
            info = self._lock.drop_begin(self._hold_v_sp.value())
        except Exception as exc:
            self.message.emit(f'Laser: drop failed: {exc}')
            self._update_state_widgets()
            return
        info.update({'t_start': time.time(),
                     'timestamp': datetime.datetime.now().isoformat(timespec='milliseconds'),
                     'dwell_ms': self._dwell_sp.value()})
        self._drop_info = info
        self._drop_timer.start(int(self._dwell_sp.value()))
        self._update_state_widgets()
        self.message.emit('Laser: DROPPING…')

    def _finish_drop(self) -> None:
        info = self._drop_info or {}
        self._drop_info = None
        if self._lock.connected:
            try:
                self._lock.drop_end()
            except Exception as exc:
                self.message.emit(f'Laser: restore after drop failed: {exc}')
        info['t_end'] = time.time()
        self._update_state_widgets()
        self.message.emit(f'Laser: drop done, lock restored '
                          f'({info.get("dwell_ms", 0)} ms at '
                          f'{info.get("hold_v", float("nan")):.3f} V)')
        self.dropped.emit(info)

    def shutdown(self) -> None:
        if self._drop_timer.isActive():
            self._drop_timer.stop()
            try:
                self._lock.drop_end()
            except Exception:
                pass
        super().shutdown()


# ── Homodyne LO phase lock ────────────────────────────────────────────────────

class HomodynePanel(PiLockPanel):
    """Locks the homodyne output (LO phase) with a fiber stretcher.  The
    stretcher has a finite range, so when the output nears a rail the
    integrator is jumped back: by one fringe period (V_2π) if known, else to
    the centre of the range.  The lock re-acquires on the next fringe."""

    SETTINGS_KEY = 'homodyne'
    DEFAULTS     = HOMODYNE_DEFAULTS
    SIGNAL_LABEL = 'Homodyne signal'
    TITLE        = 'Homodyne phase PI loop'

    def __init__(self, link: SharedLink, parent=None):
        self._n_resets = 0
        self._last_reset_t = 0.0
        super().__init__(link, parent)

    def _build_extra_groups(self) -> None:
        g = QGroupBox('Fiber stretcher range')
        lay = QGridLayout(g)
        self._rail_chk = QCheckBox('Auto-reset integrator at rail')
        self._rail_chk.setToolTip(
            'While locked: when the output comes within the margin of Vmin or\n'
            'Vmax, jump the integrator by one fringe (V_2π) towards the centre,\n'
            'or to the centre of the range if V_2π is 0.')
        self._rail_chk.stateChanged.connect(lambda _: self.save())
        self._margin_sp = _dspin(0.01, 0.5, 0.05, 3, 0.01, ' V')
        self._margin_sp.setToolTip('Distance from Vmin / Vmax that counts as the rail')
        self._v2pi_sp = _dspin(0.0, 2.0, 0.0, 3, 0.01, ' V')
        self._v2pi_sp.setToolTip('Output voltage per 2π of LO phase (fringe '
                                 'period). 0 = unknown → reset to centre.')
        self._use_period_btn = QPushButton('V_2π ← sweep')
        self._use_period_btn.setEnabled(False)
        self._use_period_btn.clicked.connect(self._on_use_period)
        self._reset_now_btn = QPushButton('Re-centre now')
        self._reset_now_btn.setToolTip('Jump the integrator to the centre of '
                                       'the range (lock re-acquires)')
        self._reset_now_btn.clicked.connect(self._on_reset_now)
        self._reset_lbl = QLabel('0 resets')
        self._reset_lbl.setStyleSheet('color:#aaa; font-size:9pt;')
        r = 0
        lay.addWidget(self._rail_chk, r, 0, 1, 3); r += 1
        lay.addWidget(QLabel('Margin:'), r, 0); lay.addWidget(self._margin_sp, r, 1, 1, 2); r += 1
        lay.addWidget(QLabel('V_2π:'), r, 0); lay.addWidget(self._v2pi_sp, r, 1)
        lay.addWidget(self._use_period_btn, r, 2); r += 1
        lay.addWidget(self._reset_now_btn, r, 0, 1, 2); lay.addWidget(self._reset_lbl, r, 2)
        lay.setColumnStretch(1, 1); lay.setColumnStretch(2, 1)
        self._col_layout.addWidget(g)

    def _apply_settings(self, s: dict) -> None:
        super()._apply_settings(s)
        self._rail_chk.setChecked(bool(s['rail_reset']))
        self._margin_sp.setValue(float(s['rail_margin']))
        self._v2pi_sp.setValue(float(s['v_2pi']))

    def _collect_settings(self) -> dict:
        d = super()._collect_settings()
        d.update({'rail_reset': self._rail_chk.isChecked(),
                  'rail_margin': self._margin_sp.value(),
                  'v_2pi': self._v2pi_sp.value()})
        return d

    def _describe_sweep(self, res: dict) -> str:
        txt = super()._describe_sweep(res)
        if 'v_period' in res:
            txt += f'  Fringe period ≈ {res["v_period"]:.3f} V per 2π.'
        return txt

    def _on_sweep_done(self, res: dict) -> None:
        self._use_period_btn.setEnabled('v_period' in res)

    def _on_use_period(self) -> None:
        if 'v_period' in self._sweep_result:
            self._v2pi_sp.setValue(self._sweep_result['v_period'])
            self.save()

    def _update_extra_widgets(self, st: str) -> None:
        self._reset_now_btn.setEnabled(self._lock.connected and st == STATE_LOCKED)

    def _centre(self) -> float:
        return 0.5 * (self._vmin_sp.value() + self._vmax_sp.value())

    def _do_reset(self, direction: int, reason: str) -> None:
        """direction: -1 = output near Vmax (jump down), +1 = near Vmin."""
        v2pi = self._v2pi_sp.value()
        if v2pi > 0:
            target = self._lock.pid.ival + direction * v2pi
            lo, hi = self._vmin_sp.value(), self._vmax_sp.value()
            if not (lo < target < hi):
                target = self._centre()
        else:
            target = self._centre()
        try:
            self._lock.reset_integrator(target)
        except Exception as exc:
            self.message.emit(f'Homodyne: integrator reset failed: {exc}')
            return
        self._n_resets += 1
        self._last_reset_t = time.time()
        self._reset_lbl.setText(
            f'{self._n_resets} resets, last {datetime.datetime.now():%H:%M:%S}')
        self.message.emit(f'Homodyne: {reason} → integrator to {target:+.3f} V')

    def _on_reset_now(self) -> None:
        if self._lock.state == STATE_LOCKED:
            self._do_reset(0, 'manual re-centre')

    def _on_status(self, st: dict) -> None:
        if (not self._rail_chk.isChecked() or st['state'] != STATE_LOCKED
                or time.time() - self._last_reset_t < 1.0):
            return
        m = self._margin_sp.value()
        if st['out'] >= st['vmax'] - m:
            self._do_reset(-1, 'output at upper rail')
        elif st['out'] <= st['vmin'] + m:
            self._do_reset(+1, 'output at lower rail')


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
    app = QApplication(sys.argv)
    link = SharedLink(use_mock='--mock' in sys.argv)
    win = QMainWindow()
    tabs = QTabWidget()
    laser = LaserStabPanel(link)
    hom = HomodynePanel(link)
    for p in (laser, hom):
        p.message.connect(lambda s: win.statusBar().showMessage(s))
    tabs.addTab(laser, 'Laser stabilization')
    tabs.addTab(hom, 'Homodyne phase lock')
    win.setCentralWidget(tabs)
    win.setWindowTitle('Red Pitaya locks (standalone)')
    win.resize(1200, 760)
    win.show()
    rc = app.exec_()
    laser.shutdown(); hom.shutdown(); link.disconnect_board()
    sys.exit(rc)
