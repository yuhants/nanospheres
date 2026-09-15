"""
Calibration pulse GUI — PyQt5 front end for the Quantum Composers 9614.

This is the pulse-generator half of ``dm_search/impulse_calibration.py`` made
interactive: set the pulse shape, then step through a list of amplitudes,
holding each one for a set dwell.  The script's inner loop

    for amp in amps:
        set_pulse(channel=1, amp=amp, width='0.0000002', period='0.15')
        turn_on()
        ... take nfiles files ...
        turn_off()

becomes: type the amplitude list, set the dwell to however long those files
take, press Start.

Nothing here writes data files.  Two ways to record alongside it:

  * run the DAQ separately (``python daq/take_data_gui.py``, or a script) — the
    two are independent, and the dwell is a wall clock you match to the run by
    hand; or
  * use the DAQ GUI's own "Pulses" tab, which embeds the same panel and steps
    the amplitude every N *files* instead of every N seconds, tags the filenames
    with the amplitude and records the pulse settings into each HDF5 file.  That
    is the faithful version of the script; this GUI is the bench tool for
    setting pulses up and trying them out.

Layout
  Left    pulse-generator panel (connection, shape, amplitude list, schematic)
  Right   sweep settings, per-step table, log
  Bottom  Start / Skip / Stop, status and countdown

Notes
  - Output ON/OFF drives T0 (``:PULSE0:STATE``), the same global gate the script
    toggles.  It is the master switch: the panel settings mean nothing until it
    is on.
  - Dwell can be given in seconds or in pulses; pulses are converted with the
    current period, so changing the period rescales the sweep.
  - "Output off between steps" reproduces the script (off during the amplitude
    change, on for the dwell).  Unticked, the train runs continuously and only
    the amplitude steps, which avoids the on/off transient.
  - Amplitude only reaches the output if the channel is in adjustable mode; set
    "Output mode" to Adjustable once if you are not sure ("leave as is" sends no
    mode command, like the script).

Run:
    python pulse_gen/pulse_gui.py

Headless (no GUI):

    from control.src.quantum_composers_9614_control import PulseGenerator9614

    with PulseGenerator9614(port='COM9') as gen:
        gen.configure(channel=1, amp=2.5, width=200e-9, period=0.15)
        gen.output_on()
        ...
        gen.output_off()
"""

import sys
import os
import json
import time
from typing import List, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox, QSplitter,
    QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QTextEdit,
    QHeaderView, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pulse_gen.pulse_panel import (
    PulseControlPanel, SweepWorker, SweepStep, PRESETS_DIR, DWELL_MODES,
    fmt_time, fmt_clock,
)

try:
    from control.src.agilent_twisstorr_84fsag_control import read_pressure as _hw_read_pressure
    _HAS_PRESSURE = True
except Exception:
    _hw_read_pressure = None
    _HAS_PRESSURE = False


class PulseMainWindow(QMainWindow):
    """
    States: 'disconnected', 'idle' (port open, sweep not running), 'sweeping'.

    The generator object is shared with SweepWorker and is internally locked, so
    the port stays usable during a sweep — but the panel controls that would
    fight the sweep are disabled (``panel.set_busy``) rather than left to race it.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Calibration pulses — Quantum Composers 9614')
        self.setMinimumSize(1150, 720)

        self._worker: Optional[SweepWorker] = None
        self._steps:  List[SweepStep]       = []
        self._sweep_start: Optional[float]  = None

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        self._build_ui()
        self._on_settings_changed()
        self._update_buttons()

    # ── UI construction ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        self._panel = PulseControlPanel()
        self._panel.log_message.connect(self._log_line)
        self._panel.state_changed.connect(self._update_buttons)
        self._panel.settings_changed.connect(self._on_settings_changed)

        panel_holder = QWidget()
        holder_layout = QVBoxLayout(panel_holder)
        holder_layout.addWidget(self._panel)
        holder_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(panel_holder)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(440)
        scroll.setMaximumWidth(520)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._build_sweep_group())
        right_layout.addWidget(self._build_table_group(), stretch=1)
        right_layout.addWidget(self._build_log_group())
        right_layout.addWidget(self._build_control_bar())

        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def _build_sweep_group(self) -> QGroupBox:
        group  = QGroupBox('Sweep (timed)')
        layout = QGridLayout(group)

        self._dwell_spin = QDoubleSpinBox()
        self._dwell_spin.setRange(0.1, 1e6)
        self._dwell_spin.setDecimals(2)
        self._dwell_spin.setValue(60.0)
        self._dwell_spin.setToolTip(
            'How long to hold each amplitude — long enough for the DAQ to take\n'
            'the files you want at that amplitude.')
        self._dwell_mode = QComboBox()
        self._dwell_mode.addItems(DWELL_MODES)

        self._gap_spin = QDoubleSpinBox()
        self._gap_spin.setRange(0.0, 3600.0)
        self._gap_spin.setDecimals(1)
        self._gap_spin.setValue(0.0)
        self._gap_spin.setSuffix(' s')
        self._gap_spin.setToolTip('Output-off pause between steps')

        self._repeat_spin = QSpinBox()
        self._repeat_spin.setRange(1, 999)
        self._repeat_spin.setValue(1)
        self._repeat_spin.setToolTip('Passes through the whole list')

        self._off_between_check = QCheckBox('Output off between steps')
        self._off_between_check.setChecked(True)

        self._pressure_check = QCheckBox('Log pressure at each step')
        self._pressure_check.setChecked(_HAS_PRESSURE)
        self._pressure_check.setEnabled(_HAS_PRESSURE)
        if not _HAS_PRESSURE:
            self._pressure_check.setToolTip('Pressure driver not importable')
        self._pressure_port = QLineEdit('COM7')
        self._pressure_port.setMaximumWidth(70)

        self._summary_label = QLabel('')
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet('color: gray;')

        for widget in (self._dwell_spin, self._gap_spin, self._repeat_spin):
            widget.valueChanged.connect(self._on_settings_changed)
        self._dwell_mode.currentIndexChanged.connect(self._on_settings_changed)

        layout.addWidget(QLabel('Dwell:'),        0, 0)
        layout.addWidget(self._dwell_spin,        0, 1)
        layout.addWidget(self._dwell_mode,        0, 2)
        layout.addWidget(QLabel('Gap:'),          1, 0)
        layout.addWidget(self._gap_spin,          1, 1)
        layout.addWidget(QLabel('Repeats:'),      1, 2)
        layout.addWidget(self._repeat_spin,       1, 3)
        layout.addWidget(self._off_between_check, 2, 0, 1, 4)
        layout.addWidget(self._pressure_check,    3, 0, 1, 3)
        layout.addWidget(self._pressure_port,     3, 3)
        layout.addWidget(self._summary_label,     4, 0, 1, 4)
        return group

    def _build_table_group(self) -> QGroupBox:
        group  = QGroupBox('Sweep steps')
        layout = QVBoxLayout(group)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ['Amplitude (V)', 'Dwell', 'Pulses', 'Status'])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self._table)
        return group

    def _build_log_group(self) -> QGroupBox:
        group  = QGroupBox('Log')
        layout = QVBoxLayout(group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(110)
        layout.addWidget(self._log)
        return group

    def _build_control_bar(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)

        self._start_btn = QPushButton('▶  Start sweep')
        self._skip_btn  = QPushButton('⏭  Skip step')
        self._stop_btn  = QPushButton('■  Stop')
        self._start_btn.clicked.connect(self._on_start)
        self._skip_btn.clicked.connect(self._on_skip)
        self._stop_btn.clicked.connect(self._on_stop)

        self._state_label = QLabel('Not connected')
        self._state_label.setStyleSheet('font-weight: bold;')
        self._step_label      = QLabel('')
        self._countdown_label = QLabel('')
        self._elapsed_label   = QLabel('00:00')

        for item in (self._start_btn, self._skip_btn, self._stop_btn,
                     self._state_label, self._step_label):
            layout.addWidget(item)
        layout.addStretch()
        layout.addWidget(self._countdown_label)
        layout.addWidget(self._elapsed_label)
        for label, slot in [('Save preset', self._save_preset),
                            ('Load preset', self._load_preset)]:
            button = QPushButton(label)
            button.clicked.connect(slot)
            layout.addWidget(button)
        return widget

    # ── Settings ────────────────────────────────────────────────────────────────

    def _dwell_seconds(self) -> float:
        """Dwell in seconds, converting a pulse count with the current period."""
        if self._dwell_mode.currentText() == 'pulses':
            return self._dwell_spin.value() * self._panel.period()
        return self._dwell_spin.value()

    def _sweeping(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _on_settings_changed(self) -> None:
        if self._sweeping():
            return
        try:
            amps = self._panel.amplitudes()
        except ValueError:
            amps = []
        dwell  = self._dwell_seconds()
        period = max(self._panel.period(), 1e-12)
        self._steps = [SweepStep(a, dwell) for _ in range(self._repeat_spin.value())
                       for a in amps]

        self._table.setRowCount(len(self._steps))
        for row, step in enumerate(self._steps):
            self._set_row(row, step, period, 'pending')

        total = (sum(s.dwell for s in self._steps)
                 + self._gap_spin.value() * max(len(self._steps) - 1, 0))
        if self._steps:
            self._summary_label.setText(
                f'{len(self._steps)} steps · {fmt_time(dwell)} each '
                f'(≈{dwell / period:.0f} pulses) · total {fmt_time(total)}')
        else:
            self._summary_label.setText('no amplitudes')

    def _set_row(self, row: int, step: SweepStep, period: float, status: str) -> None:
        values = [f'{step.amplitude:g}', fmt_time(step.dwell),
                  f'{step.dwell / period:.0f}', status]
        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, col, item)

    def _set_status(self, row: int, status: str, colour: Optional[str] = None) -> None:
        item = self._table.item(row, 3)
        if item is None:
            return
        item.setText(status)
        if colour:
            item.setForeground(QColor(colour))

    def _log_line(self, text: str) -> None:
        self._log.append(f'[{time.strftime("%H:%M:%S")}] {text}')

    def _update_buttons(self) -> None:
        connected = self._panel.is_connected
        sweeping  = self._sweeping()

        self._start_btn.setEnabled(connected and not sweeping)
        self._skip_btn.setEnabled(sweeping)
        self._stop_btn.setEnabled(sweeping or (connected and self._panel.output_is_on))

        if not connected:
            self._state_label.setText('Not connected')
        elif sweeping:
            self._state_label.setText('Sweeping')
        elif self._panel.output_is_on:
            self._state_label.setText('Output ON')
        else:
            self._state_label.setText('Idle')

    # ── Sweep control ───────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        gen = self._panel.generator
        if gen is None:
            return
        amps = self._panel.validated_amplitudes(self)
        if amps is None:
            return
        dwell = self._dwell_seconds()
        steps = [SweepStep(a, dwell) for _ in range(self._repeat_spin.value())
                 for a in amps]

        # The shape (width, period, polarity, mode) is set once here; the sweep
        # then changes nothing but the amplitude.
        if not self._panel.apply_shape(steps[0].amplitude, parent=self):
            return

        self._steps = steps
        period = max(self._panel.period(), 1e-12)
        self._table.setRowCount(len(steps))
        for row, step in enumerate(steps):
            self._set_row(row, step, period, 'pending')

        self._worker = SweepWorker(
            gen, self._panel.channel(), steps,
            gap=self._gap_spin.value(),
            output_off_between=self._off_between_check.isChecked(),
            read_pressure=self._pressure_check.isChecked(),
            pressure_port=self._pressure_port.text().strip() or 'COM7',
            pressure_reader=_hw_read_pressure,
        )
        self._worker.step_started.connect(self._on_step_started)
        self._worker.step_tick.connect(self._on_step_tick)
        self._worker.step_done.connect(self._on_step_done)
        self._worker.message.connect(self._log_line)
        self._worker.done.connect(self._on_sweep_done)
        self._worker.error.connect(self._on_sweep_error)

        total = (sum(s.dwell for s in steps)
                 + self._gap_spin.value() * max(len(steps) - 1, 0))
        self._log_line(f'sweep started: {len(steps)} steps, total {fmt_time(total)}')
        self._sweep_start = time.time()
        self._elapsed_timer.start(500)
        self._panel.set_busy(True)
        self._panel.note_output_state(True, steps[0].amplitude)
        self._worker.start()
        self._update_buttons()

    def _on_skip(self) -> None:
        if self._worker is not None:
            self._worker.request_skip()

    def _on_stop(self) -> None:
        if self._sweeping():
            self._worker.request_stop()
            self._log_line('stop requested')
        else:
            self._panel.output_off()

    def _on_step_started(self, index: int, amplitude: float) -> None:
        self._panel.note_output_state(True, amplitude)
        self._set_status(index, 'running', '#0a0')
        self._step_label.setText(
            f'step {index + 1}/{len(self._steps)} — {amplitude:g} V')
        self._table.scrollToItem(self._table.item(index, 0))
        self._log_line(f'step {index + 1}/{len(self._steps)}: {amplitude:g} V')

    def _on_step_tick(self, index: int, elapsed: float, dwell: float) -> None:
        self._countdown_label.setText(f'step {fmt_clock(dwell - elapsed)} left')

    def _on_step_done(self, index: int) -> None:
        self._set_status(index, 'done', '#666')

    def _on_sweep_done(self, completed: bool) -> None:
        self._finish_sweep()
        self._log_line('sweep finished' if completed else 'sweep stopped')
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 3)
            if item is not None and item.text() in ('running', 'pending'):
                self._set_status(row, 'not run', '#a60')

    def _on_sweep_error(self, message: str) -> None:
        self._finish_sweep()
        self._log_line(f'sweep error: {message}')
        QMessageBox.critical(self, 'Sweep', f'{message}\n\nThe output was turned off.')

    def _finish_sweep(self) -> None:
        self._elapsed_timer.stop()
        self._worker = None
        self._step_label.setText('')
        self._countdown_label.setText('')
        self._panel.set_busy(False)
        self._panel.note_output_state(False)
        self._update_buttons()

    def _tick_elapsed(self) -> None:
        if self._sweep_start is not None:
            self._elapsed_label.setText(fmt_clock(time.time() - self._sweep_start))

    # ── Presets ─────────────────────────────────────────────────────────────────

    def _to_dict(self) -> dict:
        d = self._panel.to_dict()
        d.update({
            'dwell':         self._dwell_spin.value(),
            'dwell_mode':    self._dwell_mode.currentText(),
            'gap':           self._gap_spin.value(),
            'repeats':       self._repeat_spin.value(),
            'off_between':   self._off_between_check.isChecked(),
            'read_pressure': self._pressure_check.isChecked(),
            'pressure_port': self._pressure_port.text(),
        })
        return d

    def _from_dict(self, d: dict) -> None:
        self._panel.from_dict(d)
        if 'dwell' in d:
            self._dwell_spin.setValue(float(d['dwell']))
        if 'dwell_mode' in d:
            self._dwell_mode.setCurrentText(d['dwell_mode'])
        if 'gap' in d:
            self._gap_spin.setValue(float(d['gap']))
        if 'repeats' in d:
            self._repeat_spin.setValue(int(d['repeats']))
        if 'off_between' in d:
            self._off_between_check.setChecked(bool(d['off_between']))
        if _HAS_PRESSURE and 'read_pressure' in d:
            self._pressure_check.setChecked(bool(d['read_pressure']))
        if 'pressure_port' in d:
            self._pressure_port.setText(d['pressure_port'])
        self._on_settings_changed()

    def _save_preset(self) -> None:
        os.makedirs(PRESETS_DIR, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save preset', PRESETS_DIR, 'JSON (*.json)')
        if path:
            with open(path, 'w') as f:
                json.dump(self._to_dict(), f, indent=2)
            self._log_line(f'preset saved to {path}')

    def _load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load preset', PRESETS_DIR, 'JSON (*.json)')
        if path:
            with open(path) as f:
                self._from_dict(json.load(f))
            self._log_line(f'preset loaded from {path}')

    # ── Shutdown ────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._sweeping():
            self._worker.request_stop()
            self._worker.wait(5000)
        self._panel.shutdown()
        super().closeEvent(event)


# ── Entry points ───────────────────────────────────────────────────────────────

def launch_gui() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = PulseMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    launch_gui()
