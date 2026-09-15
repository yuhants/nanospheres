"""
Shared pulse-generator widgets and sequencing, for the Quantum Composers 9614.

Two GUIs use this module:

  ``pulse_gen/pulse_gui.py``   pulses on their own, stepped on a wall clock
  ``daq/daq_gui.py``           pulses stepped by the recording — one amplitude
                               per N files, which is what the calibration script
                               actually did

What lives here:

  PulseControlPanel   connection, pulse shape, amplitude list, manual output
                      gate, schematic and log — the half of the GUI that is the
                      same in both hosts.  It owns the PulseGenerator9614.
  SweepWorker         wall-clock sweep (dwell per amplitude) in a QThread.
  PulseSequencer      file-driven stepping, called from inside the DAQ's
                      acquisition thread between files, so an amplitude change
                      can never land in the middle of a capture.

The two steppers are deliberately separate: one owns the clock, the other is
owned by the DAQ.  Both leave the output off when they finish, including on
error.
"""

import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox, QTextEdit,
)
from PyQt5.QtCore import QThread, pyqtSignal

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from control.src.quantum_composers_9614_control import (
    PulseGenerator9614, DEFAULT_PORT, DEFAULT_BAUDRATE,
)

try:
    from serial.tools import list_ports as _list_ports
except Exception:
    _list_ports = None


PRESETS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presets')
BAUD_OPTIONS = ['9600', '19200', '38400', '57600', '115200']
TIME_UNITS   = {'ns': 1e-9, 'µs': 1e-6, 'ms': 1e-3, 's': 1.0}
POLARITIES   = ['NORM', 'COMP', 'INV']
OUTPUT_MODES = ['leave as is', 'Adjustable (ADJ)', 'TTL']
DWELL_MODES  = ['seconds', 'pulses']

# The 9614's adjustable output tops out at 20 V; the GUI refuses to ask for more.
MAX_AMPLITUDE_V = 20.0

# Returned by PulseControlPanel._safe when the call raised.  A sentinel, not
# None: the setter methods return None on success, so None cannot mean failure.
FAILED = object()


def fmt_time(seconds: float) -> str:
    if seconds >= 3600:
        return f'{seconds / 3600:.3g} h'
    if seconds >= 120:
        return f'{seconds / 60:.3g} min'
    if seconds >= 1.0:
        return f'{seconds:.3g} s'
    if seconds >= 1e-3:
        return f'{seconds * 1e3:.3g} ms'
    if seconds >= 1e-6:
        return f'{seconds * 1e6:.3g} µs'
    return f'{seconds * 1e9:.3g} ns'


def fmt_clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f'{seconds // 60:02d}:{seconds % 60:02d}'


def parse_amplitudes(text: str) -> List[float]:
    """
    Parse an amplitude list: comma- or whitespace-separated volts.

    Raises ValueError naming the offending entry, so a GUI can say which one.
    """
    amps = []
    for token in text.replace(',', ' ').split():
        try:
            amps.append(float(token))
        except ValueError:
            raise ValueError(f'not a number: {token!r}')
    return amps


def amplitude_tag(amplitude: float) -> str:
    """Filename fragment for an amplitude, as in '..._2.5v_0.hdf5'."""
    return f'{amplitude:g}v_'


# ── Sequencing ─────────────────────────────────────────────────────────────────

@dataclass
class SweepStep:
    amplitude: float      # V
    dwell: float          # s


class PulseSequencer:
    """
    Steps the amplitude with the DAQ's file counter instead of a clock.

    Called from the acquisition thread: ``before_file(i)`` runs while the scope
    is idle, so the amplitude only ever changes between files and every file
    belongs to exactly one amplitude.  ``files_per_amplitude`` files are taken at
    each entry of ``amplitudes``, reproducing

        for amp in amps:
            set_pulse(...); turn_on(); <nfiles files>; turn_off()

    from dm_search/impulse_calibration.py.

    A run longer than the list either cycles back to the start or holds the last
    amplitude, depending on ``cycle``.  ``finish()`` is idempotent and must be
    called on every exit path — AcquisitionWorker does so in a finally block.
    """

    def __init__(self, gen: PulseGenerator9614, channel: int,
                 amplitudes: List[float], files_per_amplitude: int,
                 output_off_between: bool = True, cycle: bool = False,
                 log: Optional[Callable[[str], None]] = None):
        if not amplitudes:
            raise ValueError('no amplitudes given')
        if files_per_amplitude < 1:
            raise ValueError('files per amplitude must be at least 1')
        self.gen                 = gen
        self.channel             = channel
        self.amplitudes          = list(amplitudes)
        self.files_per_amplitude = int(files_per_amplitude)
        self.output_off_between  = output_off_between
        self.cycle               = cycle
        self._log                = log
        self._current_step: Optional[int] = None
        self._index_in_step      = 0
        self._counts: dict       = {}     # amplitude index → files taken so far
        self._finished           = False

    @property
    def n_files(self) -> int:
        """Files needed to walk the list exactly once."""
        return len(self.amplitudes) * self.files_per_amplitude

    def set_log(self, log: Optional[Callable[[str], None]]) -> None:
        """
        Redirect the log after construction.

        The DAQ GUI points this at its acquisition worker's ``message`` signal,
        which only exists once the worker has been built around the sequencer.
        """
        self._log = log

    def step_index(self, file_index: int) -> int:
        step = file_index // self.files_per_amplitude
        if step >= len(self.amplitudes):
            step = step % len(self.amplitudes) if self.cycle else len(self.amplitudes) - 1
        return step

    def amplitude_for_file(self, file_index: int) -> float:
        return self.amplitudes[self.step_index(file_index)]

    def file_tag(self, file_index: int) -> str:
        return amplitude_tag(self.amplitude_for_file(file_index))

    @property
    def index_in_step(self) -> int:
        """
        Position of the current file within its amplitude, counting from 0.

        Valid after before_file(); this is what gives per-amplitude file
        numbering (``..._2.5v_0``, ``..._2.5v_1``, ``..._5v_0``, …) as in
        dm_search/impulse_calibration.py.

        It counts per amplitude rather than per visit, so a cycling run resumes
        an amplitude's numbering on its second pass instead of overwriting the
        files from its first.
        """
        return self._index_in_step

    def before_file(self, file_index: int) -> float:
        """
        Make sure the right amplitude is on the output, and return it.

        The generator is only touched when the step changes, so the train runs
        undisturbed through the files of one amplitude.
        """
        step = self.step_index(file_index)
        amplitude = self.amplitudes[step]
        if step != self._current_step:
            if self.output_off_between and self._current_step is not None:
                self.gen.output_off()
            self.gen.set_amplitude(self.channel, amplitude)
            self.gen.output_on()
            self._current_step = step
            self._finished = False
            self._say(f'pulse step {step + 1}/{len(self.amplitudes)}: {amplitude:g} V')
        self._index_in_step = self._counts.get(step, 0)
        self._counts[step] = self._index_in_step + 1
        return amplitude

    def finish(self) -> None:
        """Turn the output off at the end of a run (safe to call twice)."""
        if self._finished:
            return
        self._finished = True
        self._current_step = None
        try:
            self.gen.output_off()
            self._say('pulse output off')
        except Exception as exc:
            self._say(f'could not turn the pulse output off: {exc}')

    def _say(self, message: str) -> None:
        if self._log is not None:
            self._log(message)


class SweepWorker(QThread):
    """
    Walk an amplitude list on a wall clock, holding the output on for each dwell.

    Used by the standalone pulse GUI, where no DAQ is driving the timing.

    Signals
    -------
    step_started : (index, amplitude) — output is on at that amplitude
    step_tick    : (index, elapsed, dwell) — ~10 Hz, for the countdown
    step_done    : (index)
    message      : log line
    done         : (completed) — False if stopped early
    error        : the generator or the port failed; the output has been turned
                   off if that was still possible

    Stop and skip are cooperative and are checked while waiting, so both take
    effect within ~50 ms.
    """
    step_started = pyqtSignal(int, float)
    step_tick    = pyqtSignal(int, float, float)
    step_done    = pyqtSignal(int)
    message      = pyqtSignal(str)
    done         = pyqtSignal(bool)
    error        = pyqtSignal(str)

    def __init__(self, gen: PulseGenerator9614, channel: int, steps: List[SweepStep],
                 gap: float, output_off_between: bool,
                 read_pressure: bool = False, pressure_port: str = 'COM7',
                 pressure_reader: Optional[Callable] = None):
        super().__init__()
        self._gen                = gen
        self._channel            = channel
        self._steps              = steps
        self._gap                = gap
        self._output_off_between = output_off_between
        self._read_pressure      = read_pressure
        self._pressure_port      = pressure_port
        self._pressure_reader    = pressure_reader
        self._stop_requested     = False
        self._skip_requested     = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def request_skip(self) -> None:
        self._skip_requested = True

    def _wait(self, duration: float, index: Optional[int] = None) -> None:
        """Sleep in slices, emitting ticks and honouring stop/skip."""
        start = time.time()
        while True:
            elapsed = time.time() - start
            if index is not None:
                self.step_tick.emit(index, min(elapsed, duration), duration)
            if elapsed >= duration or self._stop_requested or self._skip_requested:
                return
            time.sleep(min(0.05, duration - elapsed))

    def run(self) -> None:
        try:
            for i, step in enumerate(self._steps):
                if self._stop_requested:
                    break
                self._skip_requested = False

                self._gen.set_amplitude(self._channel, step.amplitude)
                self._gen.output_on()
                self.step_started.emit(i, step.amplitude)

                if self._read_pressure and self._pressure_reader is not None:
                    try:
                        pressure = self._pressure_reader(port=self._pressure_port)
                        self.message.emit(f'pressure {pressure:.2e} mbar')
                    except Exception as exc:
                        self.message.emit(f'pressure read failed: {exc}')

                self._wait(step.dwell, i)

                if self._output_off_between:
                    self._gen.output_off()
                self.step_done.emit(i)

                if self._skip_requested:
                    self.message.emit(f'step {i + 1} skipped')
                if self._gap > 0 and i < len(self._steps) - 1 and not self._stop_requested:
                    self._wait(self._gap)

            self._gen.output_off()
            self.done.emit(not self._stop_requested)
        except Exception as exc:
            try:
                self._gen.output_off()
            except Exception:
                pass
            self.error.emit(str(exc))


# ── Schematic ──────────────────────────────────────────────────────────────────

class PulseSchematic(FigureCanvasQTAgg):
    """
    Two-panel picture of the train being requested: a few periods on the left,
    one pulse on the right.  The pulses are drawn at an exaggerated width on the
    left panel — at 200 ns in 0.15 s they would otherwise be invisible — so read
    the width off the right panel, not the left.
    """

    def __init__(self):
        self._figure = Figure(figsize=(6, 2.2), tight_layout=True)
        super().__init__(self._figure)
        self._ax_train = self._figure.add_subplot(1, 2, 1)
        self._ax_pulse = self._figure.add_subplot(1, 2, 2)
        self.setMinimumHeight(180)
        self.setMaximumHeight(260)

    def update_schematic(self, amp: float, width: float, period: float,
                         delay: float, polarity: str) -> None:
        sign = -1.0 if polarity == 'INV' else 1.0
        level = sign * max(amp, 1e-9)

        ax = self._ax_train
        ax.clear()
        n_shown = 3
        span = n_shown * period
        draw_width = max(width, 0.012 * span)      # keep the pulse visible
        t, v = [0.0], [0.0]
        for k in range(n_shown + 1):
            edge = k * period + delay
            t += [edge, edge, edge + draw_width, edge + draw_width]
            v += [0.0, level, level, 0.0]
        t.append(span)
        v.append(0.0)
        ax.plot(t, v, color='C0', lw=1.2)
        ax.set_xlim(0, span)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('amplitude (V)')
        ax.set_title(f'train — period {fmt_time(period)} '
                     f'({1.0 / period:.4g} Hz)', fontsize=9)
        ax.annotate('', xy=(delay, level * 1.12), xytext=(period + delay, level * 1.12),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=0.8))
        ax.text(period / 2 + delay, level * 1.18, 'period', ha='center',
                fontsize=8, color='gray')

        ax = self._ax_pulse
        ax.clear()
        pad = max(width, 1e-12)
        ax.plot([-pad, 0, 0, width, width, width + pad],
                [0, 0, level, level, 0, 0], color='C1', lw=1.4)
        ax.set_xlim(-pad, width + pad)
        ax.set_xlabel('time from pulse start (s)')
        ax.set_title(f'single pulse — {fmt_time(width)} @ {amp:.2f} V', fontsize=9)

        for ax in (self._ax_train, self._ax_pulse):
            ax.grid(alpha=0.25)
            lo, hi = sorted((0.0, level * 1.35))
            ax.set_ylim(lo - 0.05 * abs(level), hi + 0.05 * abs(level))
        self.draw_idle()


# ── Control panel ──────────────────────────────────────────────────────────────

class PulseControlPanel(QWidget):
    """
    Connection, pulse shape, amplitude list and the manual output gate.

    Owns the PulseGenerator9614 and is the only thing that opens or closes the
    port, so a host GUI never touches serial directly — it asks the panel for
    ``generator`` and hands that to a SweepWorker or a PulseSequencer.  Since the
    generator object is internally locked, the panel's manual controls remain
    safe to use while a worker holds it; the host is expected to call
    ``set_busy(True)`` anyway so the two cannot fight over the amplitude.

    Signals
    -------
    log_message      : a line for the host's log
    state_changed    : connection or output state changed (hosts refresh buttons)
    settings_changed : a shape or list edit (hosts refresh their summaries)
    """
    log_message      = pyqtSignal(str)
    state_changed    = pyqtSignal()
    settings_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None, show_schematic: bool = True,
                 show_log: bool = True):
        super().__init__(parent)
        self._gen: Optional[PulseGenerator9614] = None
        self._output_on = False
        self._busy      = False
        self._live_amplitude: Optional[float] = None   # set by a sweep/sequenced run

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_shape_group())
        layout.addWidget(self._build_list_group())

        self._schematic = PulseSchematic() if show_schematic else None
        if self._schematic is not None:
            layout.addWidget(self._schematic)

        self._log = None
        if show_log:
            group = QGroupBox('Pulse log')
            log_layout = QVBoxLayout(group)
            self._log = QTextEdit()
            self._log.setReadOnly(True)
            self._log.setMaximumHeight(110)
            log_layout.addWidget(self._log)
            layout.addWidget(group)
            self.log_message.connect(self._append_log)

        self._on_settings_changed()
        self._update_widgets()

    # ── Construction ────────────────────────────────────────────────────────────

    def _build_connection_group(self) -> QGroupBox:
        group  = QGroupBox('Pulse generator (Quantum Composers 9614)')
        layout = QGridLayout(group)

        self._port_combo = QComboBox()
        self._port_combo.setEditable(True)
        self.refresh_ports()

        refresh_btn = QPushButton('⟳')
        refresh_btn.setMaximumWidth(28)
        refresh_btn.setToolTip('Rescan serial ports')
        refresh_btn.clicked.connect(self.refresh_ports)

        self._baud_combo = QComboBox()
        self._baud_combo.addItems(BAUD_OPTIONS)
        self._baud_combo.setCurrentText(str(DEFAULT_BAUDRATE))

        self._connect_btn    = QPushButton('Connect')
        self._disconnect_btn = QPushButton('Disconnect')
        self._identify_btn   = QPushButton('Identify')
        self._connect_btn.clicked.connect(self.connect_generator)
        self._disconnect_btn.clicked.connect(self.disconnect_generator)
        self._identify_btn.clicked.connect(self.identify)

        self._idn_label = QLabel('not connected')
        self._idn_label.setStyleSheet('color: gray;')
        self._idn_label.setWordWrap(True)

        layout.addWidget(QLabel('Port:'),      0, 0)
        layout.addWidget(self._port_combo,     0, 1)
        layout.addWidget(refresh_btn,          0, 2)
        layout.addWidget(QLabel('Baud:'),      1, 0)
        layout.addWidget(self._baud_combo,     1, 1, 1, 2)
        layout.addWidget(self._connect_btn,    2, 0)
        layout.addWidget(self._disconnect_btn, 2, 1)
        layout.addWidget(self._identify_btn,   2, 2)
        layout.addWidget(self._idn_label,      3, 0, 1, 3)
        return group

    def _build_shape_group(self) -> QGroupBox:
        group  = QGroupBox('Pulse shape')
        layout = QGridLayout(group)

        self._channel_spin = QSpinBox()
        self._channel_spin.setRange(1, 8)
        self._channel_spin.setValue(1)
        self._channel_spin.setToolTip('Output channel (PULSE1 … PULSE8)')

        self._amp_spin = QDoubleSpinBox()
        self._amp_spin.setRange(0.0, MAX_AMPLITUDE_V)
        self._amp_spin.setDecimals(2)
        self._amp_spin.setSingleStep(0.5)
        self._amp_spin.setValue(2.5)
        self._amp_spin.setSuffix(' V')
        self._amp_spin.setToolTip('Amplitude for manual output;\n'
                                  'a sweep or a sequenced run overwrites it as it steps.')

        self._width_spin, self._width_unit = self._value_with_unit(200.0, 'ns')
        self._width_spin.setToolTip('Pulse width — 200 ns in the calibration script')

        self._period_spin, self._period_unit = self._value_with_unit(150.0, 'ms')
        self._period_spin.setToolTip('Pulse-to-pulse period (T0)')
        self._rate_label = QLabel('')
        self._rate_label.setStyleSheet('color: gray;')

        self._delay_spin, self._delay_unit = self._value_with_unit(0.0, 'ns')

        self._polarity_combo = QComboBox()
        self._polarity_combo.addItems(POLARITIES)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(OUTPUT_MODES)
        self._mode_combo.setToolTip(
            'Adjustable makes the amplitude setting reach the output.\n'
            '"leave as is" sends no mode command, matching impulse_calibration.py.')

        self._apply_btn = QPushButton('Apply to generator')
        self._apply_btn.setToolTip('Push channel state, amplitude, width, period,\n'
                                   'delay and polarity to the controller')
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        self._read_btn = QPushButton('Read back')
        self._read_btn.setToolTip('Fill the panel from the controller')
        self._read_btn.clicked.connect(self.read_back)

        # The button says what pressing it does; the label beside it says what the
        # output is doing.  The same button turns the output back off.
        self._output_btn = QPushButton('⏻  Turn output ON')
        self._output_btn.setToolTip(
            'Master gate (:PULSE0:STATE) at the amplitude above — press again to\n'
            'turn it off.  Independent of the scope, so it works while monitoring.')
        self._output_btn.clicked.connect(self.toggle_output)

        self._output_state_label = QLabel('output off')
        self._output_state_label.setStyleSheet('color: gray;')

        for widget in (self._amp_spin, self._width_spin, self._period_spin,
                       self._delay_spin):
            widget.valueChanged.connect(self._on_settings_changed)
        for widget in (self._width_unit, self._period_unit, self._delay_unit,
                       self._polarity_combo):
            widget.currentIndexChanged.connect(self._on_settings_changed)

        layout.addWidget(QLabel('Channel:'),     0, 0)
        layout.addWidget(self._channel_spin,     0, 1, 1, 2)
        layout.addWidget(QLabel('Amplitude:'),   1, 0)
        layout.addWidget(self._amp_spin,         1, 1, 1, 2)
        layout.addWidget(QLabel('Width:'),       2, 0)
        layout.addWidget(self._width_spin,       2, 1)
        layout.addWidget(self._width_unit,       2, 2)
        layout.addWidget(QLabel('Period:'),      3, 0)
        layout.addWidget(self._period_spin,      3, 1)
        layout.addWidget(self._period_unit,      3, 2)
        layout.addWidget(self._rate_label,       4, 1, 1, 2)
        layout.addWidget(QLabel('Delay:'),       5, 0)
        layout.addWidget(self._delay_spin,       5, 1)
        layout.addWidget(self._delay_unit,       5, 2)
        layout.addWidget(QLabel('Polarity:'),    6, 0)
        layout.addWidget(self._polarity_combo,   6, 1, 1, 2)
        layout.addWidget(QLabel('Output mode:'), 7, 0)
        layout.addWidget(self._mode_combo,       7, 1, 1, 2)
        layout.addWidget(self._apply_btn,        8, 0, 1, 2)
        layout.addWidget(self._read_btn,         8, 2)
        layout.addWidget(self._output_btn,       9, 0, 1, 2)
        layout.addWidget(self._output_state_label, 9, 2)
        return group

    def _value_with_unit(self, value: float, unit: str) -> Tuple[QDoubleSpinBox, QComboBox]:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(0.0, 1e6)
        spin.setValue(value)
        combo = QComboBox()
        combo.addItems(list(TIME_UNITS))
        combo.setCurrentText(unit)
        return spin, combo

    def _build_list_group(self) -> QGroupBox:
        group  = QGroupBox('Amplitude list')
        layout = QGridLayout(group)

        self._amps_edit = QLineEdit('2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20')
        self._amps_edit.setToolTip('Amplitudes in volts, comma or space separated')
        self._amps_edit.textChanged.connect(self._on_settings_changed)

        self._range_start = QDoubleSpinBox()
        self._range_stop  = QDoubleSpinBox()
        self._range_step  = QDoubleSpinBox()
        for spin, value in ((self._range_start, 2.5), (self._range_stop, 20.0),
                            (self._range_step, 2.5)):
            spin.setRange(0.01 if spin is self._range_step else 0.0, MAX_AMPLITUDE_V)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setValue(value)
        fill_btn = QPushButton('Fill list')
        fill_btn.setToolTip('Replace the list with start → stop in steps of…')
        fill_btn.clicked.connect(self._on_fill_range)
        reverse_btn = QPushButton('Reverse')
        reverse_btn.clicked.connect(self._on_reverse_list)

        layout.addWidget(QLabel('Amplitudes (V):'), 0, 0, 1, 4)
        layout.addWidget(self._amps_edit,           1, 0, 1, 4)
        layout.addWidget(QLabel('From:'),           2, 0)
        layout.addWidget(self._range_start,         2, 1)
        layout.addWidget(QLabel('to:'),             2, 2)
        layout.addWidget(self._range_stop,          2, 3)
        layout.addWidget(QLabel('step:'),           3, 0)
        layout.addWidget(self._range_step,          3, 1)
        layout.addWidget(fill_btn,                  3, 2)
        layout.addWidget(reverse_btn,               3, 3)
        return group

    # ── State ───────────────────────────────────────────────────────────────────

    @property
    def generator(self) -> Optional[PulseGenerator9614]:
        return self._gen

    @property
    def is_connected(self) -> bool:
        return self._gen is not None and self._gen.connected

    @property
    def output_is_on(self) -> bool:
        return self._output_on

    def set_busy(self, busy: bool) -> None:
        """
        Lock out the controls a running sweep or recording owns.

        The port stays open and the log keeps updating; only the knobs that
        would race the stepper are disabled.
        """
        self._busy = busy
        self._update_widgets()

    def _update_widgets(self) -> None:
        connected = self.is_connected
        free      = not self._busy
        self._connect_btn.setEnabled(not connected and free)
        self._disconnect_btn.setEnabled(connected and free)
        self._identify_btn.setEnabled(connected and free)
        self._apply_btn.setEnabled(connected and free)
        self._read_btn.setEnabled(connected and free)
        self._output_btn.setEnabled(connected and free)
        self._amp_spin.setEnabled(free)
        self._output_btn.setText(
            '⏻  Turn output OFF' if self._output_on else '⏻  Turn output ON')
        self._output_btn.setStyleSheet(
            'font-weight: bold; color: #b00;' if self._output_on else '')
        if not connected:
            self._output_state_label.setText('not connected')
            self._output_state_label.setStyleSheet('color: gray;')
        elif self._output_on:
            amplitude = (self._live_amplitude if self._live_amplitude is not None
                         else self.amplitude())
            self._output_state_label.setText(
                f'PULSING at {amplitude:.2f} V' + (' (run)' if self._busy else ''))
            self._output_state_label.setStyleSheet('color: #b00; font-weight: bold;')
        else:
            self._output_state_label.setText('output off')
            self._output_state_label.setStyleSheet('color: gray;')

    def note_output_state(self, on: bool, amplitude: Optional[float] = None) -> None:
        """
        Tell the panel the output was changed by a worker rather than by hand.

        ``amplitude`` is the value the worker has just set, so the readout tracks
        a sweep or a sequenced run instead of showing the manual spin box.
        """
        self._output_on = on
        self._live_amplitude = amplitude if on else None
        self._update_widgets()
        self.state_changed.emit()

    def _append_log(self, text: str) -> None:
        if self._log is not None:
            self._log.append(f'[{time.strftime("%H:%M:%S")}] {text}')

    # ── Values ──────────────────────────────────────────────────────────────────

    def channel(self) -> int:
        return self._channel_spin.value()

    def amplitude(self) -> float:
        return self._amp_spin.value()

    def width(self) -> float:
        return self._width_spin.value() * TIME_UNITS[self._width_unit.currentText()]

    def period(self) -> float:
        return self._period_spin.value() * TIME_UNITS[self._period_unit.currentText()]

    def delay(self) -> float:
        return self._delay_spin.value() * TIME_UNITS[self._delay_unit.currentText()]

    def polarity(self) -> str:
        return self._polarity_combo.currentText()

    def output_mode(self) -> Optional[str]:
        text = self._mode_combo.currentText()
        if text.startswith('Adjustable'):
            return 'ADJ'
        if text == 'TTL':
            return 'TTL'
        return None

    def amplitudes(self) -> List[float]:
        """The amplitude list; raises ValueError if it does not parse."""
        return parse_amplitudes(self._amps_edit.text())

    def validated_amplitudes(self, parent: Optional[QWidget] = None) -> Optional[List[float]]:
        """Parse and range-check the list, reporting problems in a message box."""
        try:
            amps = self.amplitudes()
        except ValueError as exc:
            QMessageBox.warning(parent or self, 'Amplitude list', str(exc))
            return None
        if not amps:
            QMessageBox.warning(parent or self, 'Amplitude list', 'No amplitudes given.')
            return None
        bad = [a for a in amps if a > MAX_AMPLITUDE_V or a < 0]
        if bad:
            QMessageBox.warning(
                parent or self, 'Amplitude list',
                f'Out of range (0–{MAX_AMPLITUDE_V:g} V): '
                + ', '.join(f'{a:g}' for a in bad))
            return None
        return amps

    def pulse_attrs(self) -> dict:
        """Pulse settings to record alongside the data (HDF5 attributes)."""
        return {
            'pulse_channel':  self.channel(),
            'pulse_width_s':  self.width(),
            'pulse_period_s': self.period(),
            'pulse_delay_s':  self.delay(),
            'pulse_polarity': self.polarity(),
        }

    # ── Actions ─────────────────────────────────────────────────────────────────

    def refresh_ports(self) -> None:
        current = self._port_combo.currentText() or DEFAULT_PORT
        self._port_combo.clear()
        ports = []
        if _list_ports is not None:
            try:
                ports = [p.device for p in _list_ports.comports()]
            except Exception:
                ports = []
        if current not in ports:
            ports.insert(0, current)
        self._port_combo.addItems(ports)
        self._port_combo.setCurrentText(current)

    def connect_generator(self) -> bool:
        if self.is_connected:
            return True
        try:
            gen = PulseGenerator9614(port=self._port_combo.currentText(),
                                     baudrate=int(self._baud_combo.currentText()))
            gen.connect()
        except Exception as exc:
            QMessageBox.critical(self, 'Pulse generator',
                                 f'Could not open the port:\n{exc}')
            return False
        self._gen = gen
        self.log_message.emit(f'connected on {gen.port} @ {gen.baudrate}')
        self.identify()
        state = self._safe(gen.is_output_on, 'output state')
        if state is not FAILED and state is not None:
            self._output_on = bool(state)
            self.log_message.emit(f'output is currently {"ON" if state else "OFF"}')
        self._update_widgets()
        self.state_changed.emit()
        return True

    def disconnect_generator(self) -> None:
        if self._gen is None:
            return
        self._safe(self._gen.output_off, 'output off')
        self._gen.close()
        self._gen = None
        self._output_on = False
        self._idn_label.setText('not connected')
        self.log_message.emit('disconnected')
        self._update_widgets()
        self.state_changed.emit()

    def identify(self) -> None:
        if self._gen is None:
            return
        idn = self._safe(self._gen.idn, 'identify')
        if idn is FAILED:
            return
        if idn:
            self._idn_label.setText(idn)
            self.log_message.emit(f'*IDN? → {idn}')
        else:
            self._idn_label.setText('connected (no reply to *IDN?)')

    def _safe(self, func, what: str):
        """
        Run a generator call, reporting failures to the log instead of raising.

        Returns the call's value, or FAILED if it raised — never None-for-failure,
        since the setters legitimately return None when they succeed.
        """
        try:
            return func()
        except Exception as exc:
            self.log_message.emit(f'{what} failed: {exc}')
            return FAILED

    def _on_apply_clicked(self) -> None:
        self.apply_shape()

    def apply_shape(self, amplitude: Optional[float] = None,
                    parent: Optional[QWidget] = None) -> bool:
        """
        Push the whole pulse setup to the controller.  True if it went through.

        Hosts call this once before a sweep or a run, after which only the
        amplitude changes.
        """
        if self._gen is None:
            QMessageBox.warning(parent or self, 'Pulse generator',
                                'Connect to the pulse generator first.')
            return False
        amp = self.amplitude() if amplitude is None else amplitude
        try:
            self._gen.configure(
                channel=self.channel(),
                amp=amp,
                width=self.width(),
                period=self.period(),
                delay=self.delay(),
                polarity=self.polarity(),
                output_mode=self.output_mode(),
            )
        except Exception as exc:
            QMessageBox.critical(parent or self, 'Pulse generator', str(exc))
            self.log_message.emit(f'apply failed: {exc}')
            return False
        self.log_message.emit(
            f'applied: ch{self.channel()} {amp:.2f} V, width {fmt_time(self.width())}, '
            f'period {fmt_time(self.period())}')
        return True

    def read_back(self) -> None:
        if self._gen is None:
            return
        settings = self._safe(lambda: self._gen.read_settings(self.channel()),
                              'read back')
        if settings is FAILED or not settings:
            return
        if settings['amp'] is not None:
            self._amp_spin.setValue(min(settings['amp'], MAX_AMPLITUDE_V))
        if settings['width'] is not None:
            self._set_time(self._width_spin, self._width_unit, settings['width'])
        if settings['period'] is not None:
            self._set_time(self._period_spin, self._period_unit, settings['period'])
        if settings['delay'] is not None:
            self._set_time(self._delay_spin, self._delay_unit, settings['delay'])
        if settings['polarity'] in POLARITIES:
            self._polarity_combo.setCurrentText(settings['polarity'])
        self.log_message.emit('read back: ' + ', '.join(
            f'{k}={v}' for k, v in settings.items() if v is not None))
        self._on_settings_changed()

    @staticmethod
    def _set_time(spin: QDoubleSpinBox, unit: QComboBox, seconds: float) -> None:
        """Show a time in whichever unit keeps it a readable number."""
        for name in ('s', 'ms', 'µs', 'ns'):
            scale = TIME_UNITS[name]
            if seconds >= scale or name == 'ns':
                unit.setCurrentText(name)
                spin.setValue(seconds / scale)
                return

    def toggle_output(self) -> None:
        """
        Flip the master gate.  The same button turns the output back off.

        The panel's state only moves if the controller accepted the command, so
        a failed write leaves the button showing what the hardware is doing.
        """
        if self._gen is None:
            return
        if self._output_on:
            if self._safe(self._gen.output_off, 'output off') is FAILED:
                return
            self._output_on = False
            self.log_message.emit('output OFF')
        else:
            if self._safe(lambda: self._gen.set_amplitude(
                    self.channel(), self.amplitude()), 'set amplitude') is FAILED:
                return
            if self._safe(self._gen.output_on, 'output on') is FAILED:
                return
            self._output_on = True
            self.log_message.emit(f'output ON at {self.amplitude():.2f} V')
        self._update_widgets()
        self.state_changed.emit()

    def output_off(self) -> None:
        """Turn the output off if it is on (used on shutdown and error paths)."""
        if self._gen is not None and self._output_on:
            self._safe(self._gen.output_off, 'output off')
        self._output_on = False
        self._update_widgets()
        self.state_changed.emit()

    def shutdown(self) -> None:
        if self._gen is not None:
            try:
                self._gen.output_off()
            except Exception:
                pass
            self._gen.close()
            self._gen = None
        self._output_on = False

    def _on_fill_range(self) -> None:
        start, stop, step = (self._range_start.value(), self._range_stop.value(),
                             self._range_step.value())
        if step <= 0:
            QMessageBox.warning(self, 'Fill list', 'Step must be positive.')
            return
        direction = 1 if stop >= start else -1
        values, value = [], start
        # Half a step of slack so a stop that lands on the grid is included
        # despite floating-point drift.
        while (stop - value) * direction >= -step / 2 and len(values) < 1000:
            values.append(round(value, 4))
            value += step * direction
        self._amps_edit.setText(', '.join(f'{v:g}' for v in values))

    def _on_reverse_list(self) -> None:
        try:
            amps = self.amplitudes()
        except ValueError as exc:
            QMessageBox.warning(self, 'Amplitude list', str(exc))
            return
        self._amps_edit.setText(', '.join(f'{a:g}' for a in reversed(amps)))

    def _on_settings_changed(self) -> None:
        period = max(self.period(), 1e-12)
        self._rate_label.setText(f'= {1.0 / period:.4g} Hz')
        if self._schematic is not None:
            self._schematic.update_schematic(
                self.amplitude(), max(self.width(), 1e-12), period,
                self.delay(), self.polarity())
        self.settings_changed.emit()

    # ── Presets ─────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            'port':        self._port_combo.currentText(),
            'baudrate':    self._baud_combo.currentText(),
            'channel':     self._channel_spin.value(),
            'amplitude':   self._amp_spin.value(),
            'width':       [self._width_spin.value(), self._width_unit.currentText()],
            'period':      [self._period_spin.value(), self._period_unit.currentText()],
            'delay':       [self._delay_spin.value(), self._delay_unit.currentText()],
            'polarity':    self._polarity_combo.currentText(),
            'output_mode': self._mode_combo.currentText(),
            'amplitudes':  self._amps_edit.text(),
        }

    def from_dict(self, d: dict) -> None:
        def value_unit(key, spin, unit):
            if key in d:
                spin.setValue(float(d[key][0]))
                unit.setCurrentText(d[key][1])

        if 'port' in d:
            self._port_combo.setCurrentText(d['port'])
        if 'baudrate' in d:
            self._baud_combo.setCurrentText(str(d['baudrate']))
        if 'channel' in d:
            self._channel_spin.setValue(int(d['channel']))
        if 'amplitude' in d:
            self._amp_spin.setValue(float(d['amplitude']))
        value_unit('width',  self._width_spin,  self._width_unit)
        value_unit('period', self._period_spin, self._period_unit)
        value_unit('delay',  self._delay_spin,  self._delay_unit)
        if 'polarity' in d:
            self._polarity_combo.setCurrentText(d['polarity'])
        if 'output_mode' in d:
            self._mode_combo.setCurrentText(d['output_mode'])
        if 'amplitudes' in d:
            self._amps_edit.setText(d['amplitudes'])
        self._on_settings_changed()
