"""
Qt side of the slow data: the settings tab and the strip chart.

The arithmetic and the file formats live in slow_data.py; daq_gui.py owns the
workers and decides when a stream starts.  This module is only widgets:

  SlowSettingsPanel  the "Slow data" tab — rate, lock-in table, what to write,
                     which series to plot
  SlowPlotCanvas     stacked strip charts against wall-clock time
"""

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QCheckBox, QComboBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QListWidget, QListWidgetItem,
    QHeaderView, QAbstractItemView, QScrollArea, QFrame,
)
from PyQt5.QtCore import Qt, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates

from daq.slow_data import SlowConfig, LockinConfig, SlowHistory, safe_name

ALL_CHANNELS = list('ABCDEFGH')
NO_REFERENCE = '—'
MAX_PLOTS    = 6
MAX_DRAW_POINTS = 20000


class SlowSettingsPanel(QWidget):
    """
    Everything about the slow data that the user sets.

    Reduction settings (rate, statistics, lock-ins) only take hold when a stream
    starts, because the block length and the columns are fixed for a stream —
    ``apply_requested`` asks the main window to restart the monitor with them.
    Chart settings act at once through ``view_changed``.
    """
    apply_requested   = pyqtSignal()
    view_changed      = pyqtSignal()
    clear_requested   = pyqtSignal()
    new_log_requested = pyqtSignal()
    log_toggled       = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._units: Dict[str, str] = {}
        self._build_ui()
        self._update_rate_label()

    # ── construction ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.addWidget(self._build_reduction_group())
        layout.addWidget(self._build_lockin_group())
        layout.addWidget(self._build_record_group())
        layout.addWidget(self._build_chart_group())
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _build_reduction_group(self) -> QGroupBox:
        group  = QGroupBox('Slow data')
        layout = QGridLayout(group)

        self._enable_check = QCheckBox('Take slow data')
        self._enable_check.setToolTip(
            'Slow inputs are sampled with the fast ones and averaged, block by\n'
            'block, into one row per 1/rate — while monitoring and recording.\n'
            'Mark a slow input with Save = "Slow" in the channel table; doing\n'
            'so switches this on by itself.')

        self._rate_spin = QDoubleSpinBox()
        self._rate_spin.setDecimals(3)
        self._rate_spin.setRange(0.01, 100.0)
        self._rate_spin.setValue(10.0)
        self._rate_spin.setSuffix(' Hz')
        self._rate_spin.setToolTip(
            'Slow rows per second.  Each row integrates 1/rate seconds, so a\n'
            'lower rate is also a narrower lock-in bandwidth.')
        self._rate_spin.valueChanged.connect(self._update_rate_label)

        self._rate_label = QLabel('')
        self._rate_label.setStyleSheet('color: gray;')

        self._stats_check = QCheckBox('Also mean and RMS of the fast channels')
        self._stats_check.setToolTip(
            'Channels set to Save = "Slow" in the channel table always get\n'
            'mean_<ch> and rms_<ch>.  Tick this to get them for every enabled\n'
            'channel as well (e.g. to follow the size of the motion).')

        self._apply_btn = QPushButton('Apply')
        self._apply_btn.setToolTip(
            'Use these settings now: restarts the monitor.  A recording run\n'
            'picks them up when it starts.')
        self._apply_btn.clicked.connect(self.apply_requested.emit)

        self._status_label = QLabel('')
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet('color: gray;')

        layout.addWidget(self._enable_check, 0, 0, 1, 3)
        layout.addWidget(QLabel('Rate:'),    1, 0)
        layout.addWidget(self._rate_spin,    1, 1)
        layout.addWidget(self._apply_btn,    1, 2)
        layout.addWidget(self._rate_label,   2, 0, 1, 3)
        layout.addWidget(self._stats_check,  3, 0, 1, 3)
        layout.addWidget(self._status_label, 4, 0, 1, 3)
        return group

    def _build_lockin_group(self) -> QGroupBox:
        group  = QGroupBox('Lock-in demodulators')
        layout = QVBoxLayout(group)

        note = QLabel(
            'Optional software lock-in, for when the response is not already '
            'demodulated by an instrument: amplitude of the signal channel at '
            'the drive frequency.  With a reference channel (drive monitor or sync) '
            'x is the part in phase with the drive, its sign follows the sign '
            'of the charge, and the frequency follows the drive.')
        note.setWordWrap(True)
        note.setStyleSheet('color: gray;')
        layout.addWidget(note)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ['Name', 'Sig', 'Freq (Hz)', 'Ref', 'Phase °', 'Scale', 'Unit'])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        for col, width in enumerate([70, 46, 100, 46, 62, 60, 40]):
            self._table.setColumnWidth(col, width)
        header.setStretchLastSection(True)
        self._table.setMinimumHeight(120)
        self._table.setToolTip(
            'Phase rotates the output so the response sits in x.\n'
            'Scale and Unit calibrate it: x, y and r are mV × Scale, labelled Unit\n'
            '(e.g. elementary charges per mV → "e").')
        layout.addWidget(self._table)

        row = QHBoxLayout()
        add_btn = QPushButton('Add')
        del_btn = QPushButton('Remove')
        add_btn.clicked.connect(lambda: self._add_row())
        del_btn.clicked.connect(self._remove_row)
        row.addWidget(add_btn)
        row.addWidget(del_btn)
        row.addStretch()
        layout.addLayout(row)
        return group

    def _build_record_group(self) -> QGroupBox:
        group  = QGroupBox('Writing')
        layout = QVBoxLayout(group)

        self._embed_check = QCheckBox('Store in each recorded file (/slow group)')
        self._embed_check.setChecked(True)
        self._embed_check.setToolTip(
            'The slow rows reduced from a file\'s own samples are written into it,\n'
            'so the fast waveform and the slow series cannot be separated.')

        self._log_check = QCheckBox('Continuous slow log (also while monitoring)')
        self._log_check.setToolTip(
            'One HDF5 file in the output directory that keeps growing across\n'
            'monitoring, recording and the gaps between files:\n'
            '<prefix>slow_<date>_<time>.hdf5.  Lets a long charge record be kept\n'
            'without writing any fast data.')
        self._log_check.stateChanged.connect(lambda _s: self.log_toggled.emit())

        self._log_label = QLabel('not logging')
        self._log_label.setWordWrap(True)
        self._log_label.setStyleSheet('color: gray;')

        new_btn = QPushButton('Start a new log file')
        new_btn.clicked.connect(self.new_log_requested.emit)

        layout.addWidget(self._embed_check)
        layout.addWidget(self._log_check)
        layout.addWidget(self._log_label)
        layout.addWidget(new_btn, alignment=Qt.AlignLeft)
        return group

    def _build_chart_group(self) -> QGroupBox:
        group  = QGroupBox('Strip chart')
        layout = QGridLayout(group)

        self._span_spin = QDoubleSpinBox()
        self._span_spin.setDecimals(1)
        self._span_spin.setRange(0.0, 100000.0)
        self._span_spin.setValue(10.0)
        self._span_spin.setSuffix(' min')
        self._span_spin.setSpecialValueText('everything kept')
        self._span_spin.setToolTip('How far back the chart shows (0 = all of it)')
        self._span_spin.valueChanged.connect(self.view_changed.emit)

        self._keep_spin = QDoubleSpinBox()
        self._keep_spin.setDecimals(1)
        self._keep_spin.setRange(0.1, 1000.0)
        self._keep_spin.setValue(12.0)
        self._keep_spin.setSuffix(' h')
        self._keep_spin.setToolTip(
            'How much history is kept in memory for the chart.  The log file,\n'
            'if enabled, keeps everything regardless.')
        self._keep_spin.valueChanged.connect(self.view_changed.emit)

        self._series_list = QListWidget()
        self._series_list.setMinimumHeight(140)
        self._series_list.setToolTip(
            f'Tick the series to plot (up to {MAX_PLOTS}, one panel each)')
        self._series_list.itemChanged.connect(lambda _item: self.view_changed.emit())

        clear_btn = QPushButton('Clear chart')
        clear_btn.setToolTip('Forget the history on screen (files are untouched)')
        clear_btn.clicked.connect(self.clear_requested.emit)

        layout.addWidget(QLabel('Show last:'), 0, 0)
        layout.addWidget(self._span_spin,      0, 1)
        layout.addWidget(QLabel('Keep:'),      1, 0)
        layout.addWidget(self._keep_spin,      1, 1)
        layout.addWidget(self._series_list,    2, 0, 1, 2)
        layout.addWidget(clear_btn,            3, 0)
        return group

    # ── lock-in table ──────────────────────────────────────────────────────────

    def _add_row(self, lk: Optional[LockinConfig] = None) -> None:
        if lk is None:
            lk = LockinConfig(name=f'lockin{self._table.rowCount() + 1}'
                              if self._table.rowCount() else 'charge',
                              channel='A', freq=87000.0)
        row = self._table.rowCount()
        self._table.insertRow(row)

        self._table.setItem(row, 0, QTableWidgetItem(lk.name))

        sig = QComboBox()
        sig.addItems(ALL_CHANNELS)
        sig.setCurrentText(lk.channel)
        self._table.setCellWidget(row, 1, sig)

        freq = QDoubleSpinBox()
        freq.setDecimals(3)
        freq.setRange(0.001, 40e6)
        freq.setValue(lk.freq)
        freq.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self._table.setCellWidget(row, 2, freq)

        ref = QComboBox()
        ref.addItems([NO_REFERENCE] + ALL_CHANNELS)
        ref.setCurrentText(lk.reference or NO_REFERENCE)
        self._table.setCellWidget(row, 3, ref)

        phase = QDoubleSpinBox()
        phase.setDecimals(1)
        phase.setRange(-360.0, 360.0)
        phase.setValue(lk.phase_deg)
        phase.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self._table.setCellWidget(row, 4, phase)

        self._table.setItem(row, 5, QTableWidgetItem(f'{lk.scale:g}'))
        self._table.setItem(row, 6, QTableWidgetItem(lk.unit))

    def _remove_row(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            row = self._table.rowCount() - 1
        if row >= 0:
            self._table.removeRow(row)

    def _lockins(self) -> List[LockinConfig]:
        out = []
        for row in range(self._table.rowCount()):
            def text(col, default=''):
                item = self._table.item(row, col)
                return item.text().strip() if item is not None else default
            try:
                scale = float(text(5, '1') or 1)
            except ValueError:
                scale = 1.0
            ref = self._table.cellWidget(row, 3).currentText()
            out.append(LockinConfig(
                name=safe_name(text(0) or f'lockin{row + 1}'),
                channel=self._table.cellWidget(row, 1).currentText(),
                freq=self._table.cellWidget(row, 2).value(),
                reference='' if ref == NO_REFERENCE else ref,
                phase_deg=self._table.cellWidget(row, 4).value(),
                scale=scale,
                unit=text(6) or 'mV',
            ))
        return out

    # ── state read by the main window ──────────────────────────────────────────

    def config(self) -> SlowConfig:
        return SlowConfig(enabled=self._enable_check.isChecked(),
                          rate_hz=self._rate_spin.value(),
                          stats=self._stats_check.isChecked(),
                          lockins=self._lockins())

    def embed_in_files(self) -> bool:
        return self._embed_check.isChecked()

    def log_enabled(self) -> bool:
        return self._log_check.isChecked()

    def span_seconds(self) -> float:
        return self._span_spin.value() * 60.0

    def keep_points(self) -> int:
        return int(self._keep_spin.value() * 3600.0 * self._rate_spin.value())

    def set_sample_period(self, dt: float) -> None:
        self._dt = dt
        self._update_rate_label()

    def _update_rate_label(self) -> None:
        dt = getattr(self, '_dt', None)
        period = 1.0 / self._rate_spin.value()
        text = f'one row per {period:g} s'
        if dt:
            block = max(1, int(round(period / dt)))
            text += f' = {block:,} samples'
        self._rate_label.setText(text)

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_log_text(self, text: str) -> None:
        self._log_label.setText(text)

    def set_busy(self, recording: bool) -> None:
        """Settings cannot change under a run; the chart controls stay live."""
        self._apply_btn.setEnabled(not recording)

    # ── series checklist ───────────────────────────────────────────────────────

    def set_columns(self, columns: Sequence[str], units: Dict[str, str]) -> None:
        """
        Offer ``columns`` for plotting.  Ticks survive a change of configuration
        where the name survives; new lock-in outputs come pre-ticked (x and r),
        statistics do not — there are two per channel and they are rarely what
        the chart is for.
        """
        previous = {self._series_list.item(i).data(Qt.UserRole):
                    self._series_list.item(i).checkState() == Qt.Checked
                    for i in range(self._series_list.count())}
        self._units = dict(units)
        self._series_list.blockSignals(True)
        self._series_list.clear()
        any_lockin = any(not c.startswith(('mean_', 'rms_')) for c in columns)
        for col in columns:
            unit = units.get(col, '')
            item = QListWidgetItem(f'{col}  [{unit}]' if unit else col)
            item.setData(Qt.UserRole, col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if col in previous:
                on = previous[col]
            elif any_lockin:
                on = col.endswith(('_x', '_r')) and not col.startswith(('mean_', 'rms_'))
            else:
                on = col.startswith('mean_')
            item.setCheckState(Qt.Checked if on else Qt.Unchecked)
            self._series_list.addItem(item)
        self._series_list.blockSignals(False)
        self.view_changed.emit()

    def selected_series(self) -> List[str]:
        out = []
        for i in range(self._series_list.count()):
            item = self._series_list.item(i)
            if item.checkState() == Qt.Checked:
                out.append(item.data(Qt.UserRole))
        return out[:MAX_PLOTS]

    # ── presets ────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = self.config().to_dict()
        d.update(embed=self._embed_check.isChecked(),
                 log=self._log_check.isChecked(),
                 span_min=self._span_spin.value(),
                 keep_h=self._keep_spin.value(),
                 plotted=self.selected_series())
        return d

    def from_dict(self, d: dict) -> None:
        cfg = SlowConfig.from_dict(d)
        widgets = (self._enable_check, self._log_check)
        for w in widgets:
            w.blockSignals(True)
        self._enable_check.setChecked(cfg.enabled)
        self._rate_spin.setValue(cfg.rate_hz)
        self._stats_check.setChecked(cfg.stats)
        self._table.setRowCount(0)
        for lk in cfg.lockins:
            self._add_row(lk)
        self._embed_check.setChecked(bool(d.get('embed', True)))
        self._log_check.setChecked(bool(d.get('log', False)))
        for w in widgets:
            w.blockSignals(False)
        if 'span_min' in d:
            self._span_spin.setValue(float(d['span_min']))
        if 'keep_h' in d:
            self._keep_spin.setValue(float(d['keep_h']))
        # Remembered ticks are applied when the columns next appear.
        plotted = d.get('plotted')
        if isinstance(plotted, list):
            self._series_list.blockSignals(True)
            self._series_list.clear()
            for col in plotted:
                item = QListWidgetItem(str(col))
                item.setData(Qt.UserRole, str(col))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self._series_list.addItem(item)
            self._series_list.blockSignals(False)


class SlowPlotCanvas(FigureCanvasQTAgg):
    """
    Stacked strip charts, one panel per selected series, sharing a wall-clock
    time axis — a charge step is read off against the time on the lab clock.

    Redrawn in full a few times a second at most (the main window throttles it);
    at slow-data rates that is cheap and needs none of the blitting the fast
    canvas depends on.
    """

    def __init__(self):
        self._fig = Figure(figsize=(8, 2.6))
        super().__init__(self._fig)
        self.setMinimumHeight(120)
        self._columns: List[str] = []
        self._axes: Dict[str, object] = {}
        self._lines: Dict[str, object] = {}
        self._texts: Dict[str, object] = {}
        self._rebuild([], {})

    def _rebuild(self, columns: List[str], units: Dict[str, str]) -> None:
        self._fig.clear()
        self._columns = list(columns)
        self._axes, self._lines, self._texts = {}, {}, {}
        if not columns:
            ax = self._fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.5, 'no slow series selected', ha='center', va='center',
                    color='gray', transform=ax.transAxes)
            return
        axes = self._fig.subplots(len(columns), 1, sharex=True, squeeze=False)[:, 0]
        for i, (ax, col) in enumerate(zip(axes, columns)):
            (line,) = ax.plot([], [], lw=1.0, color=f'C{i}')
            unit = units.get(col, '')
            ax.set_ylabel(f'{col}\n[{unit}]' if unit else col, fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(axis='y', useOffset=False)
            self._texts[col] = ax.text(
                0.995, 0.93, '', transform=ax.transAxes, ha='right', va='top',
                fontsize=8, family='monospace',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))
            self._axes[col], self._lines[col] = ax, line
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        self._fig.subplots_adjust(left=0.09, right=0.99, top=0.97,
                                  bottom=0.16 if len(columns) < 3 else 0.1,
                                  hspace=0.12)

    def render(self, history: SlowHistory, selected: Sequence[str],
               span: float) -> None:
        selected = [c for c in selected if c in history.columns]
        if selected != self._columns:
            self._rebuild(selected, history.units)
        if not selected:
            self.draw_idle()
            return

        t_all, _ = history.series(selected[0])
        if len(t_all) == 0:
            for col in selected:
                self._lines[col].set_data([], [])
                self._texts[col].set_text('')
            self.draw_idle()
            return
        t_last = float(t_all[-1])
        since  = t_last - span if span > 0 else None
        # Matplotlib dates are days in UTC; shift so the axis reads local time.
        offset = time.localtime(t_last).tm_gmtoff

        x_lo = x_hi = None
        for col in selected:
            t, y = history.series(col, since)
            if len(t) > MAX_DRAW_POINTS:
                step = int(np.ceil(len(t) / MAX_DRAW_POINTS))
                t, y = t[::step], y[::step]
            x = (t + offset) / 86400.0
            self._lines[col].set_data(x, y)
            ax = self._axes[col]
            finite = y[np.isfinite(y)]
            if len(finite):
                lo, hi = float(finite.min()), float(finite.max())
                pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi) * 0.05, 1e-9)
                ax.set_ylim(lo - pad, hi + pad)
            latest = history.latest(col)
            unit = history.units.get(col, '')
            self._texts[col].set_text(
                '' if latest is None else f'{latest:.5g} {unit}')
            if len(x):
                x_lo, x_hi = float(x[0]), float(x[-1])
        if x_lo is not None:
            if since is not None:
                x_lo = (since + offset) / 86400.0
            if x_hi <= x_lo:
                x_hi = x_lo + 1.0 / 86400.0
            self._axes[selected[0]].set_xlim(x_lo, x_hi)
        self.draw_idle()
