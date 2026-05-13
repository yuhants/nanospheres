"""
Picoscope DAQ GUI — PyQt5 interface for PicoscopeDAQ.

Layout:
  Left panel  — hardware selector, channel table, acquisition settings, output path
  Right panel — display channel selector + dual plot (time series top, PSD bottom)
  Bottom bar  — Start / Stop, file counter, pressure, elapsed time

The worker thread computes a downsampled time trace and Welch PSD for each
recorded channel after every file, then emits the small result dict.  The main
thread updates the plots and caches the last result so toggling display channels
re-renders immediately without waiting for the next file.

Run directly:
    python daq/daq_gui.py
or from the repo root:
    python take_data_gui.py
"""

import sys
import os
import json
import time
import numpy as np
from scipy.signal import welch
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox, QSplitter,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daq.picoscope_daq import (
    PicoscopeDAQ, AcquisitionConfig, ChannelConfig,
    CHANNEL_INPUT_RANGES_MV, RANGE_LABELS, TIME_UNITS_SEC, PICO_SERIALS,
)

try:
    from control.src.agilent_twisstorr_84fsag_control import read_pressure as _hw_read_pressure
    _HAS_PRESSURE = True
except Exception:
    _HAS_PRESSURE = False

ALL_CHANNELS      = list('ABCDEFGH')
TIME_UNIT_OPTIONS = ['PS4000A_NS', 'PS4000A_US', 'PS4000A_MS']
PRESETS_DIR       = os.path.join(os.path.dirname(__file__), 'presets')
_DISPLAY_N_PTS    = 10_000   # max points shown in time-series panel


def _fmt_range(range_mv: int) -> str:
    """Format a range value (in mV) as a tidy label, e.g. '±1 V' or '±500 mV'."""
    if range_mv >= 1000:
        return f'±{range_mv // 1000} V'
    return f'±{range_mv} mV'


# ── Worker thread ──────────────────────────────────────────────────────────────

class AcquisitionWorker(QThread):
    """
    Acquire → save → compute display data, in a background thread.

    Signals
    -------
    preview_ready : emitted during acquisition every ~chunk_period seconds with a
                    rolling time-series window.  Only the top plot panel is updated.
    data_ready    : emitted after each complete file with full time series + PSD.
    progress      : file count + pressure update.

    Only small display arrays cross the thread boundary, not the raw int16 buffer.
    Stop is cooperative (only checked between files; acquire_one is blocking).
    """
    preview_ready = pyqtSignal(dict)          # {ch: {'tt', 'trace_mv', 'range_mv'}}
    data_ready    = pyqtSignal(int, dict)     # file_idx, full display_data
    progress      = pyqtSignal(int, int, float)
    done          = pyqtSignal()              # avoid shadowing QThread.finished
    error         = pyqtSignal(str)

    def __init__(self, daq: PicoscopeDAQ, file_directory: str, file_prefix: str,
                 idx_start: int, n_files: int, read_pressure: bool, pressure_port: str,
                 chunk_period: float = 0.5, window_duration: float = 2.0):
        super().__init__()
        self._daq              = daq
        self._file_directory   = file_directory
        self._file_prefix      = file_prefix
        self._idx_start        = idx_start
        self._n_files          = n_files
        self._read_pressure    = read_pressure
        self._pressure_port    = pressure_port
        self._chunk_period     = chunk_period
        self._window_duration  = window_duration
        self._stop_requested   = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            os.makedirs(self._file_directory, exist_ok=True)
            for i in range(self._n_files):
                if self._stop_requested:
                    break

                pressure = None
                if self._read_pressure and _HAS_PRESSURE:
                    try:
                        pressure = _hw_read_pressure(port=self._pressure_port)
                    except Exception:
                        pressure = -1.0

                timestamp, dt, adc2mvs, data = self._daq.acquire_one(
                    on_chunk=self._on_preview_chunk,
                    chunk_period=self._chunk_period,
                    window_duration=self._window_duration,
                )

                fname = f'{self._file_prefix}{i + self._idx_start}.hdf5'
                self._daq.write_hdf5(
                    os.path.join(self._file_directory, fname),
                    timestamp, dt, adc2mvs, data, pressure,
                )

                display = self._prepare_display(data, dt, adc2mvs)
                self.data_ready.emit(i, display)
                self.progress.emit(i + 1, self._n_files,
                                   pressure if pressure is not None else -1.0)

            self.done.emit()
        except Exception as exc:
            self.error.emit(str(exc))

    def _on_preview_chunk(self, window: np.ndarray,
                          adc2mvs: np.ndarray, dt: float) -> None:
        """
        Called from inside the streaming callback (still in this worker thread).
        Converts the rolling window to mV, downsamples, and emits preview_ready.
        Qt queued connections handle thread-safe delivery to the main thread.
        """
        n        = window.shape[1]
        decimate = max(1, n // _DISPLAY_N_PTS)
        display  = {}
        for i, ch in enumerate(self._daq.config.channels):
            trace = window[i, ::decimate].astype(np.float32) * float(adc2mvs[i])
            tt    = np.arange(len(trace)) * (decimate * dt)
            display[ch.name] = {
                'tt':       tt,
                'trace_mv': trace,
                'range_mv': int(CHANNEL_INPUT_RANGES_MV[ch.range_idx]),
            }
        self.preview_ready.emit(display)

    def _prepare_display(self, data: np.ndarray, dt: float,
                         adc2mvs: np.ndarray) -> Dict:
        """Full time series + PSD for the completed file."""
        fs       = 1.0 / dt
        nperseg  = min(int(fs / 10), 65536)
        n_total  = data.shape[1]
        decimate = max(1, n_total // _DISPLAY_N_PTS)

        display = {}
        for i, ch in enumerate(self._daq.config.channels):
            trace_mv     = data[i, ::decimate].astype(np.float32) * float(adc2mvs[i])
            tt           = np.arange(len(trace_mv)) * (decimate * dt)
            signal_mv    = data[i].astype(np.float32) * float(adc2mvs[i])
            ff, pp_mv2hz = welch(signal_mv, fs=fs, nperseg=nperseg)
            display[ch.name] = {
                'tt':       tt,
                'trace_mv': trace_mv,
                'ff':       ff,
                'pp':       pp_mv2hz * 1e-6,
                'range_mv': int(CHANNEL_INPUT_RANGES_MV[ch.range_idx]),
            }
        return display


# ── Dual-panel plot canvas ─────────────────────────────────────────────────────

class DaqPlotCanvas(FigureCanvasQTAgg):
    """
    Two stacked subplots sharing independent y-axes:
      top    — time series (mV), one trace per selected channel
      bottom — PSD (V²/Hz, log scale), one trace per selected channel

    Each channel is drawn with a consistent colour from the default cycle so
    the legend matches between the two panels.  The legend entries include the
    digitization range so scale differences between channels are immediately
    visible (e.g. "Ch D  ±1 V", "Ch C  ±5 V").
    """

    def __init__(self):
        self.fig = Figure(tight_layout=True)
        super().__init__(self.fig)
        self._ax_time: Optional[object] = None
        self._ax_psd:  Optional[object] = None
        self._color_map: Dict[str, str] = {}
        self.setMinimumSize(480, 400)
        self._init_axes()

    def _init_axes(self) -> None:
        self.fig.clf()
        self._ax_time = self.fig.add_subplot(2, 1, 1)
        self._ax_psd  = self.fig.add_subplot(2, 1, 2)
        self._style_axes()
        self.draw()

    def _style_axes(self) -> None:
        self._ax_time.set_ylabel('Signal (mV)')
        self._ax_time.set_xlabel('Time (s)')
        self._ax_time.grid(True, alpha=0.3)

        self._ax_psd.set_ylabel('PSD (V²/Hz)')
        self._ax_psd.set_xlabel('Frequency (Hz)')
        self._ax_psd.set_yscale('log')
        self._ax_psd.grid(True, which='both', alpha=0.3)

    def assign_colors(self, channel_names: List[str]) -> None:
        """Pre-assign consistent colours to channels from the default cycle."""
        colours = [f'C{i}' for i in range(len(channel_names))]
        self._color_map = {ch: colours[i] for i, ch in enumerate(channel_names)}

    def render_all(self, display_data: Dict, selected: List[str]) -> None:
        """Redraw both panels for the given selected channels."""
        if self._ax_time is None:
            return
        self._ax_time.cla()
        self._ax_psd.cla()
        for ch in selected:
            if ch not in display_data:
                continue
            d     = display_data[ch]
            color = self._color_map.get(ch, None)
            label = f'Ch {ch}  {_fmt_range(d["range_mv"])}'
            self._ax_time.plot(d['tt'], d['trace_mv'],
                               label=label, color=color, lw=0.8)
            self._ax_psd.semilogy(d['ff'], d['pp'],
                                  label=label, color=color, lw=0.8)
        self._style_axes()
        if selected:
            self._ax_time.legend(fontsize=8, loc='upper right', framealpha=0.7)
            self._ax_psd.legend(fontsize=8, loc='upper right', framealpha=0.7)
        self.draw_idle()

    def update_timeseries(self, display_data: Dict, selected: List[str]) -> None:
        """Redraw only the top panel — called on every preview chunk during acquisition."""
        if self._ax_time is None:
            return
        self._ax_time.cla()
        for ch in selected:
            if ch not in display_data:
                continue
            d     = display_data[ch]
            color = self._color_map.get(ch, None)
            label = f'Ch {ch}  {_fmt_range(d["range_mv"])}'
            self._ax_time.plot(d['tt'], d['trace_mv'],
                               label=label, color=color, lw=0.8)
        self._ax_time.set_ylabel('Signal (mV)')
        self._ax_time.set_xlabel('Time (s)')
        self._ax_time.grid(True, alpha=0.3)
        if selected:
            self._ax_time.legend(fontsize=8, loc='upper right', framealpha=0.7)
        self.draw_idle()


# ── Main window ────────────────────────────────────────────────────────────────

class DaqMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Picoscope DAQ')
        self.setMinimumSize(1150, 720)

        self._daq:               Optional[PicoscopeDAQ]        = None
        self._worker:            Optional[AcquisitionWorker]   = None
        self._last_display_data: Optional[Dict]                = None
        self._start_time:        Optional[float]               = None

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # Left: config panel
        config_widget = QWidget()
        config_widget.setMaximumWidth(390)
        cfg_layout = QVBoxLayout(config_widget)
        cfg_layout.addWidget(self._build_hardware_group())
        cfg_layout.addWidget(self._build_channel_group())
        cfg_layout.addWidget(self._build_acquisition_group())
        cfg_layout.addWidget(self._build_output_group())

        btn_row = QHBoxLayout()
        for label, slot in [('Save config', self._save_config),
                             ('Load config', self._load_config)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        cfg_layout.addLayout(btn_row)
        cfg_layout.addStretch()

        # Right: display selector + dual plot + control bar
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self._build_display_selector())
        self._canvas = DaqPlotCanvas()
        right_layout.addWidget(self._canvas)
        right_layout.addWidget(self._build_control_bar())

        splitter.addWidget(config_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def _build_hardware_group(self) -> QGroupBox:
        group  = QGroupBox('Hardware')
        layout = QGridLayout(group)

        self._pico_combo = QComboBox()
        for label in PICO_SERIALS:
            self._pico_combo.addItem(label)

        self._connect_btn    = QPushButton('Connect')
        self._disconnect_btn = QPushButton('Disconnect')
        self._disconnect_btn.setEnabled(False)
        self._connect_btn.clicked.connect(self._on_connect)
        self._disconnect_btn.clicked.connect(self._on_disconnect)

        layout.addWidget(QLabel('Scope:'),     0, 0)
        layout.addWidget(self._pico_combo,     0, 1, 1, 2)
        layout.addWidget(self._connect_btn,    1, 0)
        layout.addWidget(self._disconnect_btn, 1, 1)
        return group

    def _build_channel_group(self) -> QGroupBox:
        group  = QGroupBox('Channels')
        layout = QGridLayout(group)

        for col, header in enumerate(['', 'Ch', 'Range', 'Coupling', 'Save']):
            layout.addWidget(QLabel(f'<b>{header}</b>'), 0, col)

        self._ch_rows: Dict[str, dict] = {}
        for row, ch in enumerate(ALL_CHANNELS, start=1):
            enable      = QCheckBox()
            range_cb    = QComboBox()
            for lbl in RANGE_LABELS:
                range_cb.addItem(lbl)
            range_cb.setCurrentIndex(7)   # 2 V default
            coupling_cb = QComboBox()
            coupling_cb.addItems(['DC', 'AC'])
            save_cb     = QComboBox()
            save_cb.addItems(['Full', 'Mean'])

            layout.addWidget(enable,      row, 0)
            layout.addWidget(QLabel(ch),  row, 1)
            layout.addWidget(range_cb,    row, 2)
            layout.addWidget(coupling_cb, row, 3)
            layout.addWidget(save_cb,     row, 4)

            self._ch_rows[ch] = {
                'enable':   enable,
                'range':    range_cb,
                'coupling': coupling_cb,
                'save':     save_cb,
            }
        return group

    def _build_acquisition_group(self) -> QGroupBox:
        group  = QGroupBox('Acquisition')
        layout = QGridLayout(group)

        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 999999)
        self._interval_spin.setValue(200)

        self._units_combo = QComboBox()
        self._units_combo.addItems(TIME_UNIT_OPTIONS)

        self._buffer_edit = QLineEdit('33554432')   # 2**25

        self._nfiles_spin = QSpinBox()
        self._nfiles_spin.setRange(1, 999999)
        self._nfiles_spin.setValue(100)

        self._idxstart_spin = QSpinBox()
        self._idxstart_spin.setRange(0, 999999)

        self._pressure_check    = QCheckBox('Read pressure')
        self._pressure_port_edit = QLineEdit('COM7')
        self._pressure_port_edit.setMaximumWidth(60)
        self._pressure_check.setChecked(True)
        self._pressure_check.setEnabled(_HAS_PRESSURE)
        if not _HAS_PRESSURE:
            self._pressure_check.setToolTip('Pressure driver not importable')

        layout.addWidget(QLabel('Interval:'),       0, 0)
        layout.addWidget(self._interval_spin,        0, 1)
        layout.addWidget(self._units_combo,          0, 2)
        layout.addWidget(QLabel('Buffer size:'),     1, 0)
        layout.addWidget(self._buffer_edit,          1, 1, 1, 2)
        layout.addWidget(QLabel('N files:'),         2, 0)
        layout.addWidget(self._nfiles_spin,          2, 1)
        layout.addWidget(QLabel('Start index:'),     3, 0)
        layout.addWidget(self._idxstart_spin,        3, 1)
        layout.addWidget(self._pressure_check,       4, 0, 1, 2)
        layout.addWidget(self._pressure_port_edit,   4, 2)
        return group

    def _build_output_group(self) -> QGroupBox:
        group  = QGroupBox('Output')
        layout = QGridLayout(group)

        self._dir_edit    = QLineEdit(r'E:\data')
        browse_btn        = QPushButton('…')
        browse_btn.setMaximumWidth(28)
        browse_btn.clicked.connect(self._browse_dir)
        self._prefix_edit = QLineEdit('YYYYMMDD_abc_')

        layout.addWidget(QLabel('Directory:'), 0, 0)
        layout.addWidget(self._dir_edit,       0, 1)
        layout.addWidget(browse_btn,           0, 2)
        layout.addWidget(QLabel('Prefix:'),    1, 0)
        layout.addWidget(self._prefix_edit,    1, 1, 1, 2)
        return group

    def _build_display_selector(self) -> QWidget:
        """
        A row of checkboxes (one per recorded channel) plus update-rate controls.
        Hidden until connected; toggling re-renders from the cached last buffer.
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(QLabel('Display:'))

        self._display_checks: Dict[str, QCheckBox] = {}
        for ch in ALL_CHANNELS:
            cb = QCheckBox(ch)
            cb.setVisible(False)
            cb.stateChanged.connect(self._on_display_selection_changed)
            self._display_checks[ch] = cb
            layout.addWidget(cb)

        layout.addStretch()

        # Preview timing controls
        layout.addWidget(QLabel('Update:'))
        self._chunk_period_spin = QDoubleSpinBox()
        self._chunk_period_spin.setRange(0.1, 30.0)
        self._chunk_period_spin.setSingleStep(0.5)
        self._chunk_period_spin.setValue(0.5)
        self._chunk_period_spin.setSuffix(' s')
        self._chunk_period_spin.setToolTip(
            'How often to refresh the time-series panel during acquisition')
        self._chunk_period_spin.setMaximumWidth(80)
        layout.addWidget(self._chunk_period_spin)

        layout.addWidget(QLabel('Window:'))
        self._window_duration_spin = QDoubleSpinBox()
        self._window_duration_spin.setRange(0.1, 120.0)
        self._window_duration_spin.setSingleStep(1.0)
        self._window_duration_spin.setValue(2.0)
        self._window_duration_spin.setSuffix(' s')
        self._window_duration_spin.setToolTip(
            'How many seconds of history to show in the time-series panel')
        self._window_duration_spin.setMaximumWidth(80)
        layout.addWidget(self._window_duration_spin)

        return widget

    def _build_control_bar(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)

        self._start_btn = QPushButton('▶  Start')
        self._stop_btn  = QPushButton('■  Stop')
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)

        self._progress_label = QLabel('Not connected')
        self._pressure_label = QLabel('')
        self._elapsed_label  = QLabel('00:00')

        layout.addWidget(self._start_btn)
        layout.addWidget(self._stop_btn)
        layout.addWidget(self._progress_label)
        layout.addStretch()
        layout.addWidget(self._pressure_label)
        layout.addWidget(self._elapsed_label)
        return widget

    # ── Hardware actions ────────────────────────────────────────────────────────

    def _on_connect(self) -> None:
        config = self._build_daq_config()
        if config is None:
            return
        serial = PICO_SERIALS[self._pico_combo.currentText()]
        try:
            self._daq = PicoscopeDAQ(config)
            self._daq.connect(serial)
        except Exception as exc:
            QMessageBox.critical(self, 'Connection error', str(exc))
            self._daq = None
            return

        enabled_channels = [ch for ch in ALL_CHANNELS
                            if self._ch_rows[ch]['enable'].isChecked()]
        self._canvas.assign_colors(enabled_channels)
        self._canvas._init_axes()
        self._show_display_checkboxes(enabled_channels)

        self._connect_btn.setEnabled(False)
        self._disconnect_btn.setEnabled(True)
        self._start_btn.setEnabled(True)
        self._progress_label.setText('Connected — ready')

    def _on_disconnect(self) -> None:
        if self._daq:
            try:
                self._daq.disconnect()
            except Exception:
                pass
            self._daq = None
        self._hide_display_checkboxes()
        self._connect_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)
        self._start_btn.setEnabled(False)
        self._progress_label.setText('Disconnected')

    def _show_display_checkboxes(self, enabled_channels: List[str]) -> None:
        for ch, cb in self._display_checks.items():
            cb.setVisible(ch in enabled_channels)
            cb.setChecked(ch in enabled_channels)

    def _hide_display_checkboxes(self) -> None:
        for cb in self._display_checks.values():
            cb.setVisible(False)

    # ── Acquisition actions ─────────────────────────────────────────────────────

    def _on_start(self) -> None:
        self._last_display_data = None
        self._worker = AcquisitionWorker(
            daq=self._daq,
            file_directory=self._dir_edit.text(),
            file_prefix=self._prefix_edit.text(),
            idx_start=self._idxstart_spin.value(),
            n_files=self._nfiles_spin.value(),
            read_pressure=self._pressure_check.isChecked(),
            pressure_port=self._pressure_port_edit.text(),
            chunk_period=self._chunk_period_spin.value(),
            window_duration=self._window_duration_spin.value(),
        )
        self._worker.preview_ready.connect(self._on_preview_ready)
        self._worker.data_ready.connect(self._on_data_ready)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_error)
        self._worker.done.connect(self._on_finished)

        self._start_time = time.time()
        self._elapsed_timer.start(1000)
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress_label.setText('Running…')
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.request_stop()
        self._stop_btn.setEnabled(False)
        self._progress_label.setText('Stopping after current file…')

    def _on_preview_ready(self, preview_data: Dict) -> None:
        """Semi-realtime update: redraw only the time-series panel."""
        selected = [ch for ch, cb in self._display_checks.items() if cb.isChecked()]
        self._canvas.update_timeseries(preview_data, selected)

    def _on_data_ready(self, _idx: int, display_data: Dict) -> None:
        """End-of-file update: redraw both panels (time series + PSD)."""
        self._last_display_data = display_data
        self._redraw_plots()

    def _on_display_selection_changed(self) -> None:
        """Re-render immediately when the user toggles a display checkbox."""
        if self._last_display_data:
            self._redraw_plots()

    def _redraw_plots(self) -> None:
        selected = [ch for ch, cb in self._display_checks.items()
                    if cb.isChecked()]
        self._canvas.render_all(self._last_display_data, selected)

    def _on_progress(self, n_done: int, n_total: int, pressure: float) -> None:
        self._progress_label.setText(f'Files: {n_done} / {n_total}')
        if pressure > 0:
            self._pressure_label.setText(f'{pressure:.2e} mbar')

    def _on_finished(self) -> None:
        self._elapsed_timer.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress_label.setText(f'Done — {self._nfiles_spin.value()} files')

    def _on_error(self, msg: str) -> None:
        self._elapsed_timer.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        QMessageBox.critical(self, 'Acquisition error', msg)

    def _tick_elapsed(self) -> None:
        if self._start_time:
            elapsed = int(time.time() - self._start_time)
            m, s = divmod(elapsed, 60)
            self._elapsed_label.setText(f'{m:02d}:{s:02d}')

    # ── Config helpers ──────────────────────────────────────────────────────────

    def _build_daq_config(self) -> Optional[AcquisitionConfig]:
        channels = []
        for ch_name in ALL_CHANNELS:
            row = self._ch_rows[ch_name]
            if not row['enable'].isChecked():
                continue
            channels.append(ChannelConfig(
                name=ch_name,
                range_idx=row['range'].currentIndex(),
                coupling=row['coupling'].currentText(),
                save_mean_only=(row['save'].currentText() == 'Mean'),
            ))
        if not channels:
            QMessageBox.warning(self, 'No channels', 'Enable at least one channel.')
            return None
        try:
            buffer_size = int(self._buffer_edit.text())
        except ValueError:
            QMessageBox.warning(self, 'Invalid buffer size', 'Must be an integer.')
            return None
        return AcquisitionConfig(
            channels=channels,
            sample_interval=self._interval_spin.value(),
            sample_units=self._units_combo.currentText(),
            buffer_size=buffer_size,
        )

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, 'Select output directory', self._dir_edit.text())
        if d:
            self._dir_edit.setText(d)

    def _to_dict(self) -> dict:
        channels = {}
        for ch, row in self._ch_rows.items():
            channels[ch] = {
                'enabled':   row['enable'].isChecked(),
                'range_idx': row['range'].currentIndex(),
                'coupling':  row['coupling'].currentText(),
                'save_mode': row['save'].currentText(),
            }
        return {
            'picoscope':       self._pico_combo.currentText(),
            'channels':        channels,
            'sample_interval': self._interval_spin.value(),
            'sample_units':    self._units_combo.currentText(),
            'buffer_size':     self._buffer_edit.text(),
            'n_files':         self._nfiles_spin.value(),
            'idx_start':       self._idxstart_spin.value(),
            'file_directory':  self._dir_edit.text(),
            'file_prefix':     self._prefix_edit.text(),
            'read_pressure':   self._pressure_check.isChecked(),
            'pressure_port':   self._pressure_port_edit.text(),
        }

    def _from_dict(self, d: dict) -> None:
        if 'picoscope' in d:
            idx = self._pico_combo.findText(d['picoscope'])
            if idx >= 0:
                self._pico_combo.setCurrentIndex(idx)
        for ch, cfg in d.get('channels', {}).items():
            if ch not in self._ch_rows:
                continue
            row = self._ch_rows[ch]
            row['enable'].setChecked(cfg.get('enabled', False))
            row['range'].setCurrentIndex(cfg.get('range_idx', 7))
            for widget, key, default in [
                (row['coupling'], 'coupling',  'DC'),
                (row['save'],     'save_mode', 'Full'),
            ]:
                idx = widget.findText(cfg.get(key, default))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
        for attr, widget in [
            ('sample_interval', self._interval_spin),
            ('n_files',         self._nfiles_spin),
            ('idx_start',       self._idxstart_spin),
        ]:
            if attr in d:
                widget.setValue(d[attr])
        if 'sample_units' in d:
            idx = self._units_combo.findText(d['sample_units'])
            if idx >= 0:
                self._units_combo.setCurrentIndex(idx)
        for attr, widget in [
            ('buffer_size',    self._buffer_edit),
            ('file_directory', self._dir_edit),
            ('file_prefix',    self._prefix_edit),
            ('pressure_port',  self._pressure_port_edit),
        ]:
            if attr in d:
                widget.setText(str(d[attr]))
        if 'read_pressure' in d:
            self._pressure_check.setChecked(d['read_pressure'])

    def _save_config(self) -> None:
        os.makedirs(PRESETS_DIR, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save config', PRESETS_DIR, 'JSON (*.json)')
        if path:
            with open(path, 'w') as f:
                json.dump(self._to_dict(), f, indent=2)

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load config', PRESETS_DIR, 'JSON (*.json)')
        if path:
            with open(path) as f:
                self._from_dict(json.load(f))

    def closeEvent(self, event) -> None:
        if self._daq:
            try:
                self._daq.disconnect()
            except Exception:
                pass
        super().closeEvent(event)


# ── Entry points ───────────────────────────────────────────────────────────────

def launch_gui() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = DaqMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    launch_gui()
