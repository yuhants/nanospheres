"""
Red Pitaya signal-chain GUI — flexible PyQt5 front-end for pyrpl_control.

The Red Pitaya FPGA is a signal *bus*: every DSP module's output is a named
signal, and any module's input can be set to any other signal (a physical
input 'in1'/'in2' or another module's output). This GUI exposes that directly,
so you can freely combine and re-wire modules into arbitrary chains, e.g.

    in1 -> pid0 (input filters + PI) -> pid1 (input filters) -> out1

Layout
------
  Top    — connection bar (always visible)
  Tab "Patch & Modules":
      Left   — a rack of module blocks (iq0-2, pid0-2, iir, asg0-1). Each block
               picks its own input source, parameters/filters, and output.
      Right  — a live routing diagram of the chain you have built.
  Tab "Scope":
      Live oscilloscope that can tap any signal on the bus (incl. internal
      module outputs), so you can probe the chain at any point.

Design notes
------------
* All hardware access runs in the main thread (PyRPL owns its own Qt/async loop
  and is not thread-safe); the scope refreshes via a QTimer.
* A ``--mock`` synthetic backend lets the whole GUI run with no hardware.

Run:
    python control/src/Red_pitaya_control/pyrpl_gui.py --host 192.168.1.100
    python control/src/Red_pitaya_control/pyrpl_gui.py --mock
"""

import os
import sys
import time
import argparse
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QMessageBox, QSplitter, QFrame,
    QTabWidget, QScrollArea, QGraphicsView, QGraphicsScene,
)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import QFont, QColor, QPen, QBrush, QPolygonF, QPainter

import pyqtgraph as pg

# Same-directory import (matches redpitaya_control.py importing redpitaya_scpi)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Signal-routing option lists
_PHYS_INPUTS = ['in1', 'in2']
_MODULE_NODES = ['iq0', 'iq1', 'iq2', 'pid0', 'pid1', 'pid2', 'iir',
                 'asg0', 'asg1']
# valid sources for any module's `input`
_INPUT_SOURCES = _PHYS_INPUTS + ['iq0', 'iq1', 'iq2', 'pid0', 'pid1', 'pid2',
                                 'asg0', 'asg1', 'iir', 'off']
_SCOPE_INPUTS = _PHYS_INPUTS + ['iq0', 'iq1', 'iq2', 'pid0', 'pid1', 'pid2',
                                'asg0', 'asg1', 'iir', 'out1', 'out2', 'off']
_OUTPUTS = ['off', 'out1', 'out2']
_TRIGGERS = ['immediately', 'ch1_positive_edge', 'ch1_negative_edge',
             'ch2_positive_edge', 'ch2_negative_edge',
             'ext_positive_edge', 'ext_negative_edge', 'asg0', 'asg1']
_IQ_OUTPUT_SIGNALS = ['quadrature', 'output_direct', 'pfd', 'off']
_ASG_WAVEFORMS = ['sin', 'cos', 'ramp', 'halframp', 'square', 'dc', 'noise']

# Border colour per module family (used by the routing diagram)
_NODE_COLORS = {
    'in': '#9ae84f', 'out': '#ff9f43', 'iq': '#00e5ff',
    'pid': '#e056fd', 'iir': '#feca57', 'asg': '#54a0ff',
}

_REFRESH_HZ = 8


# ── Mock backend (for --mock / offline GUI testing) ───────────────────────────

class MockRedPitaya:
    """Synthetic stand-in for RedPitayaPyRPL: tracks routing, returns fake data."""

    DSP_MODULES = list(_MODULE_NODES)

    def __init__(self, hostname='mock', **kwargs):
        self.hostname = hostname
        self._t0 = time.time()
        self._state = {n: {'input': 'off', 'output_direct': 'off'}
                       for n in self.DSP_MODULES}
        print('MockRedPitaya: synthetic backend (no hardware).')

    def _store(self, module, kw):
        st = self._state.setdefault(module, {})
        st.update(kw)

    def acquire_scope(self, duration=0.01, input1='in1', input2='in2',
                      decimation=None, trigger_source='immediately',
                      trigger_delay=0.0, threshold=0.0, hysteresis=0.01,
                      average=True, timeout=5.0):
        n = 16384
        t = np.linspace(0, duration, n)
        phase = time.time() - self._t0
        f = 5.0 / duration if duration > 0 else 5e3

        def synth(src, amp, noise):
            if src == 'off':
                return np.zeros(n)
            if src.startswith('iq'):
                return amp * 0.3 * np.ones(n) + noise * np.random.randn(n)
            return amp * np.sin(2 * np.pi * f * t + phase) + noise * np.random.randn(n)

        return t, synth(input1, 0.4, 0.02), synth(input2, 0.25, 0.02)

    def setup_lockin(self, module='iq0', input='in1', output_direct='off', **kw):
        self._store(module, {'input': input, 'output_direct': output_direct, **kw})
        print(f'[mock] {module} lock-in <- in={input} out={output_direct} {kw}')

    def lockin_amplitude_phase(self, module='iq0', duration=0.01, timeout=5.0):
        return 0.3 + 0.01 * np.random.randn(), 37.0 + np.random.randn()

    def setup_pid(self, module='pid0', input='in1', output_direct='off', **kw):
        self._store(module, {'input': input, 'output_direct': output_direct, **kw})
        print(f'[mock] {module} PID <- in={input} out={output_direct} {kw}')

    def pause_pid(self, module='pid0'):
        self._store(module, {'output_direct': 'off'})
        print(f'[mock] {module} paused')

    def reset_pid_integrator(self, module='pid0', ival=0.0):
        print(f'[mock] {module} integrator reset -> {ival}')

    def setup_iir(self, module='iir', input='in1', output_direct='off', **kw):
        self._store(module, {'input': input, 'output_direct': output_direct, **kw})
        print(f'[mock] {module} IIR <- in={input} out={output_direct} {kw}')

    def setup_output(self, module='asg0', output_direct='out1', **kw):
        self._store(module, {'input': 'off', 'output_direct': output_direct, **kw})
        print(f'[mock] {module} ASG <- out={output_direct} {kw}')

    def transfer_function(self, module, frequencies):
        from pyrpl_control import (inputfilter_tf, pi_transfer_function,
                                   iir_transfer_function)
        f = np.asarray(frequencies, dtype=float)
        st = self._state.get(module, {})
        if module.startswith('pid'):
            return (pi_transfer_function(st.get('p', 0.0), st.get('i', 0.0), f)
                    * inputfilter_tf(st.get('inputfilter') or [], f))
        if module.startswith('iq'):
            flt = []
            if st.get('bandwidth'):
                flt.append(st['bandwidth'])
            if st.get('acbandwidth'):
                flt.append(-st['acbandwidth'])
            return inputfilter_tf(flt, f)
        if module == 'iir':
            return iir_transfer_function(st.get('zeros') or [],
                                         st.get('poles') or [],
                                         st.get('gain', 1.0), f)
        return np.ones_like(f, dtype=complex)

    def network_analyzer(self, input='in1', output_direct='out1',
                         start_freq=1e3, stop_freq=1e6, points=201, rbw=100.0,
                         amplitude=0.1, acbandwidth=0.0, logscale=True,
                         avg_per_point=1):
        if logscale:
            f = np.logspace(np.log10(start_freq), np.log10(stop_freq), points)
        else:
            f = np.linspace(start_freq, stop_freq, points)
        if input in self._state and self._state[input].get('input') not in (None, 'off'):
            H = self.transfer_function(input, f)
        else:
            H = np.ones(points, dtype=complex)
        H = H * (1 + 0.03 * np.random.randn(points))   # synthetic measurement noise
        return f, H

    def route(self, module, input=None, output_direct=None):
        upd = {}
        if input is not None:
            upd['input'] = input
        if output_direct is not None:
            upd['output_direct'] = output_direct
        self._store(module, upd)

    def set_filter(self, module, inputfilter):
        print(f'[mock] {module} filter <- {inputfilter}')

    def routing_summary(self):
        return {n: dict(self._state.get(n, {'input': 'off',
                                            'output_direct': 'off'}))
                for n in self.DSP_MODULES}

    def close(self):
        print('[mock] closed')


# ── small widget helpers ──────────────────────────────────────────────────────

def _dspin(lo, hi, val, dec=4, step=None, suffix='', special=None):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(val)
    if step is not None:
        s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    if special is not None:
        s.setSpecialValueText(special)
    s.setKeyboardTracking(False)
    return s


def _combo(items, current=None):
    c = QComboBox()
    c.addItems(items)
    if current is not None and current in items:
        c.setCurrentText(current)
    return c


# ── Module panels ─────────────────────────────────────────────────────────────

class _ModulePanel(QGroupBox):
    """Base block: input combo + params + output combo + Apply, for one module."""

    def __init__(self, title, module, win):
        super().__init__(title)
        self.module = module
        self._win = win
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(8, 6, 8, 6)
        self.grid.setVerticalSpacing(3)
        self._row = 0
        self._filter_spins = []

    # layout helpers
    def add(self, label, widget):
        if label:
            self.grid.addWidget(QLabel(label), self._row, 0)
            self.grid.addWidget(widget, self._row, 1)
        else:
            self.grid.addWidget(widget, self._row, 0, 1, 2)
        self._row += 1

    def add_two(self, w1, w2):
        self.grid.addWidget(w1, self._row, 0)
        self.grid.addWidget(w2, self._row, 1)
        self._row += 1

    def add_filter_bank(self, n=4):
        """A row of n corner-frequency spin boxes (Hz; +LP, -HP, 0=off)."""
        wrap = QWidget()
        hl = QHBoxLayout(wrap)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(2)
        for _ in range(n):
            s = _dspin(-1e6, 1e6, 0.0, dec=1, step=100.0, special='off')
            s.setMinimumWidth(64)
            self._filter_spins.append(s)
            hl.addWidget(s)
        lbl = QLabel('Filters:')
        lbl.setToolTip('Input filter corners in Hz: positive = low-pass, '
                       'negative = high-pass, 0/off = bypass that stage.')
        self.grid.addWidget(lbl, self._row, 0)
        self.grid.addWidget(wrap, self._row, 1)
        self._row += 1

    def filters(self):
        return [w.value() for w in self._filter_spins if w.value() != 0.0]

    def rp(self):
        return self._win.backend()

    def add_bode_button(self):
        b = QPushButton('Bode ▸')
        b.setToolTip('Apply this module, then plot its designed transfer '
                     'function in the Transfer-function tab.')
        b.clicked.connect(self.bode)
        self.add('', b)

    def bode(self):
        self.apply()
        if self.rp() is not None:
            self._win.show_bode(self.module)

    def _done(self, msg):
        self._win.after_change(msg)

    def _fail(self, exc):
        QMessageBox.critical(self, f'{self.module} error', str(exc))


class IqPanel(_ModulePanel):
    """IQ module: lock-in demodulator, or band-pass / frequency-shift filter."""

    def __init__(self, module, win):
        super().__init__(f'{module} — IQ / lock-in', module, win)
        self.c_in = _combo(_INPUT_SOURCES, 'in1')
        self.freq = _dspin(0.0, 62.5e6, 1e3, dec=2, step=1e3, suffix=' Hz')
        self.bw = _dspin(0.0, 1e6, 100.0, dec=2, step=10.0, suffix=' Hz')
        self.acbw = _dspin(0.0, 1e6, 0.0, dec=1, step=10.0, suffix=' Hz', special='off')
        self.phase = _dspin(-360.0, 360.0, 0.0, dec=2, step=1.0, suffix=' °')
        self.gain = _dspin(-1e6, 1e6, 0.0, dec=4, step=0.01)
        self.amp = _dspin(0.0, 1.0, 0.0, dec=4, step=0.01, suffix=' V')
        self.c_out = _combo(_OUTPUTS, 'off')
        self.c_sig = _combo(_IQ_OUTPUT_SIGNALS, 'quadrature')

        self.add('Input:', self.c_in)
        self.add('Frequency:', self.freq)
        self.add('Bandwidth (LP):', self.bw)
        self.add('AC band (HP):', self.acbw)
        self.add('Phase:', self.phase)
        self.add('Gain:', self.gain)
        self.add('Drive amp:', self.amp)
        self.add('Out signal:', self.c_sig)
        self.add('Output:', self.c_out)

        apply_btn = QPushButton('Apply')
        apply_btn.clicked.connect(self.apply)
        read_btn = QPushButton('Read R/θ')
        read_btn.clicked.connect(self.read)
        self.add_two(apply_btn, read_btn)

        self.readout = QLabel('R = —  θ = —')
        self.readout.setAlignment(Qt.AlignCenter)
        self.readout.setStyleSheet('color:#00e5ff;')
        self.add('', self.readout)
        self.add_bode_button()

    def apply(self):
        rp = self.rp()
        if rp is None:
            return
        try:
            rp.setup_lockin(
                module=self.module, frequency=self.freq.value(),
                bandwidth=self.bw.value(), input=self.c_in.currentText(),
                phase=self.phase.value(), gain=self.gain.value(),
                amplitude=self.amp.value(),
                acbandwidth=(self.acbw.value() or None),
                output_direct=self.c_out.currentText(),
                output_signal=self.c_sig.currentText())
            self._done(f'{self.module} applied')
        except Exception as exc:
            self._fail(exc)

    def read(self):
        rp = self.rp()
        if rp is None:
            return
        try:
            R, theta = self._win.read_lockin(self.module)
            self.readout.setText(f'R = {R:.4f} V  θ = {theta:+.1f}°')
        except Exception as exc:
            self._fail(exc)


class PidPanel(_ModulePanel):
    """PID module: input filter bank + PI controller (set I=0 for a pure filter)."""

    def __init__(self, module, win):
        super().__init__(f'{module} — PID / filter', module, win)
        self.c_in = _combo(_INPUT_SOURCES, 'in1')
        self.add('Input:', self.c_in)
        self.add_filter_bank(4)
        self.sp = _dspin(-1.0, 1.0, 0.0, dec=4, step=0.01, suffix=' V')
        self.p = _dspin(-1e6, 1e6, 0.0, dec=4, step=0.01)
        self.i = _dspin(-1e7, 1e7, 0.0, dec=2, step=10.0, suffix=' Hz')
        self.vmax = _dspin(-1.0, 1.0, 1.0, dec=3, step=0.05, suffix=' V')
        self.vmin = _dspin(-1.0, 1.0, -1.0, dec=3, step=0.05, suffix=' V')
        self.c_out = _combo(_OUTPUTS, 'off')
        self.add('Setpoint:', self.sp)
        self.add('P (gain):', self.p)
        self.add('I (Hz):', self.i)
        self.add('V max:', self.vmax)
        self.add('V min:', self.vmin)
        self.add('Output:', self.c_out)

        apply_btn = QPushButton('Apply / Engage')
        apply_btn.clicked.connect(self.apply)
        pause_btn = QPushButton('Pause')
        pause_btn.clicked.connect(self.pause)
        reset_btn = QPushButton('Reset ∫')
        reset_btn.clicked.connect(self.reset)
        self.grid.addWidget(apply_btn, self._row, 0, 1, 2); self._row += 1
        self.add_two(pause_btn, reset_btn)
        self.add_bode_button()

    def apply(self):
        rp = self.rp()
        if rp is None:
            return
        try:
            rp.setup_pid(
                module=self.module, input=self.c_in.currentText(),
                output_direct=self.c_out.currentText(),
                setpoint=self.sp.value(), p=self.p.value(), i=self.i.value(),
                max_voltage=self.vmax.value(), min_voltage=self.vmin.value(),
                inputfilter=(self.filters() or None))
            self._done(f'{self.module} engaged')
        except Exception as exc:
            self._fail(exc)

    def pause(self):
        rp = self.rp()
        if rp is None:
            return
        try:
            rp.pause_pid(module=self.module)
            self.c_out.setCurrentText('off')
            self._done(f'{self.module} paused')
        except Exception as exc:
            self._fail(exc)

    def reset(self):
        rp = self.rp()
        if rp is None:
            return
        try:
            rp.reset_pid_integrator(module=self.module)
            self._done(f'{self.module} integrator reset')
        except Exception as exc:
            self._fail(exc)


class IirPanel(_ModulePanel):
    """IIR module: arbitrary transfer function from poles/zeros (Hz, complex)."""

    def __init__(self, module, win):
        super().__init__(f'{module} — IIR filter', module, win)
        self.c_in = _combo(_INPUT_SOURCES, 'in1')
        self.zeros = QLineEdit('')
        self.zeros.setPlaceholderText('e.g. -1e3, -5e3+2e4j')
        self.poles = QLineEdit('-1e3, -2e4')
        self.poles.setPlaceholderText('e.g. -1e3, -2e4+1e4j')
        self.gain = _dspin(-1e6, 1e6, 1.0, dec=4, step=0.1)
        self.loops = QSpinBox(); self.loops.setRange(0, 1023)
        self.loops.setSpecialValueText('auto')
        self.c_out = _combo(_OUTPUTS, 'off')
        self.on = QCheckBox('On'); self.on.setChecked(True)
        self.bypass = QCheckBox('Bypass')

        self.add('Input:', self.c_in)
        self.add('Zeros (Hz):', self.zeros)
        self.add('Poles (Hz):', self.poles)
        self.add('Gain:', self.gain)
        self.add('Loops:', self.loops)
        self.add('Output:', self.c_out)
        self.add_two(self.on, self.bypass)
        apply_btn = QPushButton('Apply')
        apply_btn.clicked.connect(self.apply)
        self.add('', apply_btn)
        self.add_bode_button()

    @staticmethod
    def _parse(text):
        text = text.strip()
        if not text:
            return []
        out = []
        for tok in text.replace(';', ',').split(','):
            tok = tok.strip().replace(' ', '')
            if tok:
                out.append(complex(tok))
        return out

    def apply(self):
        rp = self.rp()
        if rp is None:
            return
        try:
            zeros = self._parse(self.zeros.text())
            poles = self._parse(self.poles.text())
            loops = self.loops.value() or None
            rp.setup_iir(
                module=self.module, zeros=zeros, poles=poles,
                gain=self.gain.value(), input=self.c_in.currentText(),
                output_direct=self.c_out.currentText(), loops=loops,
                on=self.on.isChecked(), bypass=self.bypass.isChecked())
            self._done(f'{self.module} applied')
        except ValueError as exc:
            QMessageBox.warning(self, 'IIR parse error',
                                f'Could not parse poles/zeros: {exc}')
        except Exception as exc:
            self._fail(exc)


class AsgPanel(_ModulePanel):
    """Arbitrary signal generator: a source (no input) feeding an output."""

    def __init__(self, module, win):
        super().__init__(f'{module} — signal generator', module, win)
        self.wave = _combo(_ASG_WAVEFORMS, 'sin')
        self.freq = _dspin(0.0, 62.5e6, 1e3, dec=2, step=1e3, suffix=' Hz')
        self.amp = _dspin(0.0, 1.0, 0.1, dec=4, step=0.01, suffix=' V')
        self.off = _dspin(-1.0, 1.0, 0.0, dec=4, step=0.01, suffix=' V')
        self.c_out = _combo(_OUTPUTS, 'off')
        self.add('Waveform:', self.wave)
        self.add('Frequency:', self.freq)
        self.add('Amplitude:', self.amp)
        self.add('Offset:', self.off)
        self.add('Output:', self.c_out)
        apply_btn = QPushButton('Apply')
        apply_btn.clicked.connect(self.apply)
        self.add('', apply_btn)

    def apply(self):
        rp = self.rp()
        if rp is None:
            return
        try:
            rp.setup_output(
                module=self.module, waveform=self.wave.currentText(),
                frequency=self.freq.value(), amplitude=self.amp.value(),
                offset=self.off.value(),
                output_direct=self.c_out.currentText())
            self._done(f'{self.module} output applied')
        except Exception as exc:
            self._fail(exc)


# ── Routing diagram ───────────────────────────────────────────────────────────

class RoutingDiagram(QGraphicsView):
    """Draws the current signal flow: inputs → modules → outputs."""

    BOX_W, BOX_H, V_GAP = 132, 34, 14

    def __init__(self):
        self._scene = QGraphicsScene()
        super().__init__(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor('#1a1a1a')))
        self.setMinimumWidth(360)

    @staticmethod
    def _color(name):
        for pre, col in _NODE_COLORS.items():
            if name.startswith(pre):
                return col
        return '#888888'

    def _box(self, name, x, y, sub=''):
        rect = QRectF(x, y, self.BOX_W, self.BOX_H)
        col = QColor(self._color(name))
        self._scene.addRect(rect, QPen(col, 2), QBrush(QColor('#262626')))
        txt = self._scene.addText(name if not sub else f'{name}\n{sub}',
                                  QFont('Segoe UI', 8))
        txt.setDefaultTextColor(QColor('#e0e0e0'))
        txt.setPos(x + 6, y + 3)
        return {'rect': rect,
                'left': QPointF(x, y + self.BOX_H / 2),
                'right': QPointF(x + self.BOX_W, y + self.BOX_H / 2)}

    def _arrow(self, p0, p1, color='#aaaaaa'):
        pen = QPen(QColor(color), 1.6)
        self._scene.addLine(p0.x(), p0.y(), p1.x(), p1.y(), pen)
        # arrowhead
        import math
        ang = math.atan2(p1.y() - p0.y(), p1.x() - p0.x())
        size = 8
        a = QPointF(p1.x() - size * math.cos(ang - 0.4),
                    p1.y() - size * math.sin(ang - 0.4))
        b = QPointF(p1.x() - size * math.cos(ang + 0.4),
                    p1.y() - size * math.sin(ang + 0.4))
        head = QPolygonF([p1, a, b])
        self._scene.addPolygon(head, QPen(QColor(color)), QBrush(QColor(color)))

    def update_routing(self, summary):
        self._scene.clear()
        if not summary:
            t = self._scene.addText('Not connected', QFont('Segoe UI', 10))
            t.setDefaultTextColor(QColor('#777'))
            return

        # Which modules are wired in?
        active = set()
        for name, info in summary.items():
            if info.get('input') not in (None, 'off') or \
               info.get('output_direct') not in (None, 'off'):
                active.add(name)
        # also place any module used as another's source
        for info in summary.values():
            src = info.get('input')
            if src in summary:
                active.add(src)

        col_x = {'in': 20, 'mod': 200, 'out': 380}
        nodes = {}

        # input column
        y = 20
        for n in _PHYS_INPUTS:
            nodes[n] = self._box(n, col_x['in'], y)
            y += self.BOX_H + self.V_GAP

        # module column (stable order)
        y = 20
        ordered = [m for m in _MODULE_NODES if m in active]
        for n in ordered:
            info = summary.get(n, {})
            sub = f"{info.get('input','?')}→{info.get('output_direct','off')}"
            nodes[n] = self._box(n, col_x['mod'], y, sub)
            y += self.BOX_H + self.V_GAP
        if not ordered:
            t = self._scene.addText('No modules wired yet —\nset a module input/output',
                                    QFont('Segoe UI', 9))
            t.setDefaultTextColor(QColor('#777'))
            t.setPos(col_x['mod'], 30)

        # output column
        y = 20
        for n in ['out1', 'out2']:
            nodes[n] = self._box(n, col_x['out'], y)
            y += self.BOX_H + self.V_GAP

        # arrows
        for n in ordered:
            info = summary.get(n, {})
            src = info.get('input')
            if src in nodes:
                self._arrow(nodes[src]['right'], nodes[n]['left'],
                            self._color(n))
            dst = info.get('output_direct')
            if dst in nodes:
                self._arrow(nodes[n]['right'], nodes[dst]['left'],
                            self._color(n))

        self.setSceneRect(self._scene.itemsBoundingRect().adjusted(-10, -10, 10, 10))


# ── Bode plot ─────────────────────────────────────────────────────────────────

class BodePlot(QWidget):
    """Stacked magnitude (dB) + phase (deg) vs log-frequency, multi-trace."""

    _PALETTE = ['#00e5ff', '#feca57', '#e056fd', '#9ae84f', '#ff9f43', '#ff6b6b']

    def __init__(self):
        super().__init__()
        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground('#1a1a1a')
        self.p_mag = self._glw.addPlot(row=0, col=0)
        self.p_mag.setLabel('left', 'Magnitude', units='dB')
        self.p_mag.showGrid(x=True, y=True, alpha=0.3)
        self.p_mag.setLogMode(x=True, y=False)
        self.p_mag.addLegend(offset=(-10, 10))
        self.p_ph = self._glw.addPlot(row=1, col=0)
        self.p_ph.setLabel('left', 'Phase', units='deg')
        self.p_ph.setLabel('bottom', 'Frequency', units='Hz')
        self.p_ph.showGrid(x=True, y=True, alpha=0.3)
        self.p_ph.setLogMode(x=True, y=False)
        self.p_ph.setXLink(self.p_mag)
        vl.addWidget(self._glw)
        self._traces = {}     # name -> (mag_curve, ph_curve)
        self._n = 0

    def plot(self, name, freqs, H, color=None):
        freqs = np.asarray(freqs, dtype=float)
        H = np.asarray(H, dtype=complex)
        mag = 20.0 * np.log10(np.abs(H) + 1e-30)
        ph = np.degrees(np.unwrap(np.angle(H)))
        if name not in self._traces:
            if color is None:
                color = self._PALETTE[self._n % len(self._PALETTE)]
            self._n += 1
            pen = pg.mkPen(color, width=2)
            mc = self.p_mag.plot(freqs, mag, pen=pen, name=name)
            pc = self.p_ph.plot(freqs, ph, pen=pen)
            self._traces[name] = (mc, pc)
        else:
            mc, pc = self._traces[name]
            mc.setData(freqs, mag)
            pc.setData(freqs, ph)

    def clear(self):
        self.p_mag.clear()
        self.p_ph.clear()
        self.p_mag.addLegend(offset=(-10, 10))
        self._traces = {}
        self._n = 0


# ── Main window ───────────────────────────────────────────────────────────────

class PyRPLMainWindow(QMainWindow):

    def __init__(self, host='192.168.1.100', use_mock=False):
        super().__init__()
        self.setWindowTitle('Red Pitaya — Signal-chain Builder')
        self.setMinimumSize(1500, 820)

        self._host = host
        self._use_mock = use_mock
        self._rp = None
        self._running = False
        self._acquiring = False
        self._panels = []

        self._scope_timer = QTimer(self)
        self._scope_timer.timeout.connect(self._poll_scope)

        self._build_ui()
        if use_mock:
            self._host_edit.setText('mock')

    # ── public accessors used by panels ────────────────────────────────────────

    def backend(self):
        return self._rp

    def after_change(self, msg):
        self._status_label.setText(msg)
        self._refresh_diagram()

    def read_lockin(self, module):
        """Pause continuous scope, read R/θ, resume. Returns (R, theta)."""
        was_running = self._running
        if was_running:
            self._scope_timer.stop()
        try:
            return self._rp.lockin_amplitude_phase(
                module=module, duration=self._sc_dur.value())
        finally:
            if was_running:
                self._scope_timer.start(int(1000 / _REFRESH_HZ))

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.addWidget(self._build_connect_bar())

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_patch_tab(), 'Patch && Modules')
        self._tabs.addTab(self._build_scope_tab(), 'Scope')
        self._tabs.addTab(self._build_tf_tab(), 'Transfer function')
        root.addWidget(self._tabs, 1)
        root.addWidget(self._build_status_bar())
        self.setCentralWidget(central)
        self._set_connected(False)

    def _build_connect_bar(self):
        bar = QFrame()
        bar.setFrameShape(QFrame.StyledPanel)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(8, 4, 8, 4)
        hl.addWidget(QLabel('Host:'))
        self._host_edit = QLineEdit(self._host)
        self._host_edit.setMaximumWidth(180)
        hl.addWidget(self._host_edit)
        self._reload_check = QCheckBox('Reload FPGA')
        self._reload_check.setChecked(True)
        hl.addWidget(self._reload_check)
        self._connect_btn = QPushButton('Connect')
        self._connect_btn.clicked.connect(self._on_connect)
        hl.addWidget(self._connect_btn)
        self._disconnect_btn = QPushButton('Disconnect')
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._disconnect_btn.setEnabled(False)
        hl.addWidget(self._disconnect_btn)
        hl.addStretch()
        return bar

    def _build_patch_tab(self):
        splitter = QSplitter(Qt.Horizontal)

        # module rack (scrollable)
        rack = QWidget()
        rv = QVBoxLayout(rack)
        rv.setContentsMargins(4, 4, 4, 4)
        for mod in ['iq0', 'iq1', 'iq2']:
            self._panels.append(IqPanel(mod, self))
        for mod in ['pid0', 'pid1', 'pid2']:
            self._panels.append(PidPanel(mod, self))
        self._panels.append(IirPanel('iir', self))
        for mod in ['asg0', 'asg1']:
            self._panels.append(AsgPanel(mod, self))
        for p in self._panels:
            rv.addWidget(p)
        rv.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rack)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(440)

        # diagram side
        right = QWidget()
        rl = QVBoxLayout(right)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel('Signal flow'))
        hdr.addStretch()
        refresh_btn = QPushButton('Refresh')
        refresh_btn.clicked.connect(self._refresh_diagram)
        hdr.addWidget(refresh_btn)
        rl.addLayout(hdr)
        self._diagram = RoutingDiagram()
        rl.addWidget(self._diagram, 1)
        hint = QLabel('Tip: set each module\'s Input to a physical input or '
                      'another module to chain them. Probe any point in the '
                      'Scope tab.')
        hint.setWordWrap(True)
        hint.setStyleSheet('color:#888;')
        rl.addWidget(hint)

        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        return splitter

    def _build_scope_tab(self):
        w = QWidget()
        hl = QHBoxLayout(w)

        ctl = QGroupBox('Scope')
        g = QGridLayout(ctl)
        self._sc_in1 = _combo(_SCOPE_INPUTS, 'in1')
        self._sc_in2 = _combo(_SCOPE_INPUTS, 'in2')
        self._sc_dur = _dspin(1e-6, 8.0, 1e-3, dec=6, step=1e-3, suffix=' s')
        self._sc_trig = _combo(_TRIGGERS, 'immediately')
        self._sc_thresh = _dspin(-20.0, 20.0, 0.0, dec=3, step=0.01, suffix=' V')
        self._autoscale = QCheckBox('Auto Y'); self._autoscale.setChecked(True)
        self._run_btn = QPushButton('Run'); self._run_btn.setCheckable(True)
        self._run_btn.clicked.connect(self._on_run_toggle)
        self._single_btn = QPushButton('Single')
        self._single_btn.clicked.connect(self._on_single)
        r = 0
        for lbl, wid in [('Ch1 in:', self._sc_in1), ('Ch2 in:', self._sc_in2),
                         ('Duration:', self._sc_dur), ('Trigger:', self._sc_trig),
                         ('Threshold:', self._sc_thresh)]:
            g.addWidget(QLabel(lbl), r, 0); g.addWidget(wid, r, 1); r += 1
        g.addWidget(self._autoscale, r, 0, 1, 2); r += 1
        g.addWidget(self._run_btn, r, 0); g.addWidget(self._single_btn, r, 1); r += 1
        g.setRowStretch(r, 1)
        ctl.setMaximumWidth(260)

        self._plot = pg.PlotWidget()
        self._plot.setBackground('#1a1a1a')
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel('bottom', 'Time', units='s')
        self._plot.setLabel('left', 'Voltage', units='V')
        self._plot.addLegend(offset=(-10, 10))
        self._curve1 = self._plot.plot(pen=pg.mkPen('#9ae84f', width=1), name='ch1')
        self._curve2 = self._plot.plot(pen=pg.mkPen('#00e5ff', width=1), name='ch2')

        hl.addWidget(ctl)
        hl.addWidget(self._plot, 1)
        return w

    def _build_tf_tab(self):
        w = QWidget()
        hl = QHBoxLayout(w)

        ctl = QWidget()
        ctl.setMaximumWidth(280)
        cv = QVBoxLayout(ctl)
        cv.setContentsMargins(2, 2, 2, 2)

        # shared frequency range
        rng = QGroupBox('Frequency range')
        rg = QGridLayout(rng)
        self._tf_start = _dspin(1e-3, 62.5e6, 1e3, dec=2, step=100.0, suffix=' Hz')
        self._tf_stop = _dspin(1.0, 62.5e6, 1e6, dec=2, step=1e3, suffix=' Hz')
        self._tf_points = QSpinBox(); self._tf_points.setRange(11, 4001)
        self._tf_points.setValue(401)
        self._tf_log = QCheckBox('Log sweep'); self._tf_log.setChecked(True)
        rg.addWidget(QLabel('Start:'), 0, 0); rg.addWidget(self._tf_start, 0, 1)
        rg.addWidget(QLabel('Stop:'), 1, 0); rg.addWidget(self._tf_stop, 1, 1)
        rg.addWidget(QLabel('Points:'), 2, 0); rg.addWidget(self._tf_points, 2, 1)
        rg.addWidget(self._tf_log, 3, 0, 1, 2)
        cv.addWidget(rng)

        # designed (computed) transfer function
        comp = QGroupBox('Designed (computed)')
        cg = QGridLayout(comp)
        self._tf_mod = _combo(['pid0', 'pid1', 'pid2', 'iq0', 'iq1', 'iq2', 'iir'],
                              'pid0')
        plot_btn = QPushButton('Plot designed TF')
        plot_btn.clicked.connect(self._on_plot_designed)
        cg.addWidget(QLabel('Module:'), 0, 0); cg.addWidget(self._tf_mod, 0, 1)
        cg.addWidget(plot_btn, 1, 0, 1, 2)
        cv.addWidget(comp)

        # measured (network analyzer)
        na = QGroupBox('Measured (network analyzer)')
        ng = QGridLayout(na)
        self._na_meas = _combo(_SCOPE_INPUTS, 'in1')
        self._na_ref = _combo(['drive'] + _SCOPE_INPUTS, 'drive')
        self._na_drive = _combo(_OUTPUTS, 'out1')
        self._na_amp = _dspin(0.0, 1.0, 0.05, dec=4, step=0.01, suffix=' V')
        self._na_rbw = _dspin(0.1, 1e6, 100.0, dec=1, step=10.0, suffix=' Hz')
        self._na_acbw = _dspin(0.0, 1e6, 0.0, dec=1, step=10.0, suffix=' Hz',
                               special='off')
        run_btn = QPushButton('Run sweep')
        run_btn.clicked.connect(self._on_run_na)
        r = 0
        for lbl, wid in [('Measure:', self._na_meas), ('Reference:', self._na_ref),
                         ('Drive out:', self._na_drive), ('Amplitude:', self._na_amp),
                         ('RBW:', self._na_rbw), ('AC band:', self._na_acbw)]:
            ng.addWidget(QLabel(lbl), r, 0); ng.addWidget(wid, r, 1); r += 1
        ng.addWidget(run_btn, r, 0, 1, 2)
        self._na_ref.setToolTip('Reference for the ratio H = measure / reference. '
                                '"drive" uses the excitation itself (one sweep); '
                                'any other point runs a second sweep and divides, '
                                'giving an arbitrary point-vs-point transfer function.')
        cv.addWidget(na)

        clear_btn = QPushButton('Clear plot')
        clear_btn.clicked.connect(lambda: self._bode.clear())
        cv.addWidget(clear_btn)
        cv.addStretch()

        self._bode = BodePlot()
        hl.addWidget(ctl)
        hl.addWidget(self._bode, 1)
        return w

    def _tf_freqs(self):
        a, b = self._tf_start.value(), self._tf_stop.value()
        n = self._tf_points.value()
        if self._tf_log.isChecked() and a > 0:
            return np.logspace(np.log10(a), np.log10(b), n)
        return np.linspace(a, b, n)

    def show_bode(self, module):
        """Switch to the TF tab and plot a module's designed transfer function."""
        if module in [self._tf_mod.itemText(i) for i in range(self._tf_mod.count())]:
            self._tf_mod.setCurrentText(module)
        self._tabs.setCurrentWidget(self._tabs.widget(2))
        self._on_plot_designed()

    def _on_plot_designed(self):
        if self._rp is None:
            return
        module = self._tf_mod.currentText()
        try:
            freqs = self._tf_freqs()
            H = self._rp.transfer_function(module, freqs)
            self._bode.plot(f'{module} (designed)', freqs, H)
            self._status_label.setText(f'{module} transfer function plotted')
        except Exception as exc:
            QMessageBox.critical(self, 'Transfer function error', str(exc))

    def _on_run_na(self):
        if self._rp is None:
            return
        was_running = self._running
        if was_running:
            self._scope_timer.stop()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            kw = dict(output_direct=self._na_drive.currentText(),
                      start_freq=self._tf_start.value(),
                      stop_freq=self._tf_stop.value(),
                      points=self._tf_points.value(),
                      rbw=self._na_rbw.value(),
                      amplitude=self._na_amp.value(),
                      acbandwidth=(self._na_acbw.value() or 0.0),
                      logscale=self._tf_log.isChecked())
            meas = self._na_meas.currentText()
            ref = self._na_ref.currentText()
            freqs, H = self._rp.network_analyzer(input=meas, **kw)
            if ref == 'drive':
                name = f'{meas}/drive (meas)'
            else:
                _, Href = self._rp.network_analyzer(input=ref, **kw)
                H = H / Href
                name = f'{meas}/{ref} (meas)'
            self._bode.plot(name, freqs, H)
            self._status_label.setText('Network-analyzer sweep done')
        except Exception as exc:
            QMessageBox.critical(self, 'Network analyzer error', str(exc))
        finally:
            QApplication.restoreOverrideCursor()
            if was_running:
                self._scope_timer.start(int(1000 / _REFRESH_HZ))

    def _build_status_bar(self):
        w = QWidget(); w.setFixedHeight(26)
        hl = QHBoxLayout(w); hl.setContentsMargins(6, 2, 6, 2)
        self._rate_label = QLabel('')
        self._status_label = QLabel('Not connected')
        hl.addWidget(self._rate_label)
        sep = QFrame(); sep.setFrameShape(QFrame.VLine); hl.addWidget(sep)
        hl.addStretch()
        hl.addWidget(self._status_label)
        return w

    # ── enable/disable ─────────────────────────────────────────────────────────

    def _set_connected(self, on):
        for p in self._panels:
            p.setEnabled(on)
        for b in (self._run_btn, self._single_btn):
            b.setEnabled(on)

    # ── connection ─────────────────────────────────────────────────────────────

    def _on_connect(self):
        host = self._host_edit.text().strip()
        if not host:
            QMessageBox.warning(self, 'Connect', 'Enter a host/IP.')
            return
        self._connect_btn.setEnabled(False)
        self._status_label.setText(f'Connecting to {host}…')
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.repaint()
        try:
            if self._use_mock or host == 'mock':
                self._rp = MockRedPitaya(hostname=host)
            else:
                from pyrpl_control import RedPitayaPyRPL
                self._rp = RedPitayaPyRPL(
                    hostname=host, gui=False,
                    reloadfpga=self._reload_check.isChecked())
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self._connect_btn.setEnabled(True)
            self._status_label.setText('Connection failed')
            QMessageBox.critical(self, 'Connection error', str(exc))
            return
        QApplication.restoreOverrideCursor()
        self._disconnect_btn.setEnabled(True)
        self._set_connected(True)
        self._status_label.setText(f'Connected to {host}')
        self._refresh_diagram()

    def _on_disconnect(self):
        self._stop_running()
        if self._rp is not None:
            try:
                self._rp.close()
            except Exception:
                pass
            self._rp = None
        self._set_connected(False)
        self._disconnect_btn.setEnabled(False)
        self._connect_btn.setEnabled(True)
        self._run_btn.setChecked(False)
        self._status_label.setText('Disconnected')
        self._diagram.update_routing(None)

    # ── routing diagram ────────────────────────────────────────────────────────

    def _refresh_diagram(self):
        if self._rp is None:
            self._diagram.update_routing(None)
            return
        try:
            self._diagram.update_routing(self._rp.routing_summary())
        except Exception:
            pass

    # ── scope ──────────────────────────────────────────────────────────────────

    def _scope_kwargs(self):
        return dict(duration=self._sc_dur.value(),
                    input1=self._sc_in1.currentText(),
                    input2=self._sc_in2.currentText(),
                    trigger_source=self._sc_trig.currentText(),
                    threshold=self._sc_thresh.value())

    def _on_run_toggle(self, checked):
        if checked:
            if self._rp is None:
                self._run_btn.setChecked(False)
                return
            self._running = True
            self._run_btn.setText('Stop')
            self._scope_timer.start(int(1000 / _REFRESH_HZ))
        else:
            self._stop_running()

    def _stop_running(self):
        self._running = False
        self._scope_timer.stop()
        self._run_btn.setText('Run')

    def _on_single(self):
        if self._rp is not None:
            self._acquire_once()

    def _poll_scope(self):
        if self._running and not self._acquiring:
            self._acquire_once()

    def _acquire_once(self):
        if self._rp is None or self._acquiring:
            return
        self._acquiring = True
        t0 = time.time()
        try:
            t, ch1, ch2 = self._rp.acquire_scope(**self._scope_kwargs())
        except Exception as exc:
            self._stop_running()
            self._run_btn.setChecked(False)
            QMessageBox.critical(self, 'Scope error', str(exc))
            return
        finally:
            self._acquiring = False
        self._curve1.setData(t, ch1)
        if ch2 is not None and self._sc_in2.currentText() != 'off':
            self._curve2.setData(t, ch2)
        else:
            self._curve2.clear()
        if self._autoscale.isChecked():
            self._plot.enableAutoRange('y', True)
        self._rate_label.setText(f'acq {(time.time()-t0)*1e3:.0f} ms')

    # ── shutdown ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._stop_running()
        if self._rp is not None:
            try:
                self._rp.close()
            except Exception:
                pass
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description='Red Pitaya signal-chain GUI')
    parser.add_argument('--host', default='192.168.1.100',
                        help='Red Pitaya IP / hostname')
    parser.add_argument('--mock', action='store_true',
                        help='Run with a synthetic backend (no hardware)')
    args = parser.parse_args()

    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = PyRPLMainWindow(host=args.host, use_mock=args.mock)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
