"""Live time-series plots: count rate, dose rate, temperature, battery vs time."""

import time
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from radiacode.types import RareData, RealTimeData

# Selectable display windows: label -> seconds (None = all data)
_WINDOWS = [
    ('1 min',  60),
    ('5 min',  300),
    ('15 min', 900),
    ('1 hour', 3600),
    ('All',    None),
]

# Keep at most this many samples (1 Hz realtime → ~2 h; rare data is slower)
_MAXLEN = 7200

# Don't redraw more often than this (seconds) to keep the GUI responsive
_REDRAW_MIN_INTERVAL = 1.0


class TimeSeriesWidget(QWidget):
    """Rolling history of count rate, dose rate, temperature and battery."""

    def __init__(self, color: str = '#2196F3', parent=None):
        super().__init__(parent)
        self._color = color

        # Realtime series (count rate, dose rate) — arrive ~1 Hz
        self._t_rt = deque(maxlen=_MAXLEN)
        self._cr   = deque(maxlen=_MAXLEN)
        self._dr   = deque(maxlen=_MAXLEN)

        # Rare series (temperature, battery) — arrive slowly
        self._t_rare = deque(maxlen=_MAXLEN)
        self._temp   = deque(maxlen=_MAXLEN)
        self._bat    = deque(maxlen=_MAXLEN)

        self._window_s = 300            # default 5 min
        self._last_draw = 0.0
        self._build_ui()

    # ------------------------------------------------------------------ #
    #  UI                                                                  #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel('Window:'))
        self._window_cb = QComboBox()
        for label, _ in _WINDOWS:
            self._window_cb.addItem(label)
        self._window_cb.setCurrentText('5 min')
        self._window_cb.currentIndexChanged.connect(self._on_window_change)
        ctrl.addWidget(self._window_cb)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._fig = Figure(figsize=(5, 4), tight_layout=True)
        self._fig.patch.set_facecolor('#f8f8f8')
        self._ax_cr   = self._fig.add_subplot(311)
        self._ax_dr   = self._fig.add_subplot(312, sharex=self._ax_cr)
        self._ax_env  = self._fig.add_subplot(313, sharex=self._ax_cr)
        self._ax_bat  = self._ax_env.twinx()

        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._canvas)

        self._redraw(force=True)

    def _on_window_change(self, idx: int):
        self._window_s = _WINDOWS[idx][1]
        self._redraw(force=True)

    # ------------------------------------------------------------------ #
    #  Data ingestion (called from MainWindow slots)                       #
    # ------------------------------------------------------------------ #

    def add_realtime(self, item: RealTimeData):
        ts = item.dt.timestamp() if item.dt else time.time()
        self._t_rt.append(ts)
        self._cr.append(item.count_rate)
        self._dr.append(item.dose_rate)
        self._redraw()

    def add_rare(self, item: RareData):
        ts = item.dt.timestamp() if item.dt else time.time()
        self._t_rare.append(ts)
        temp = getattr(item, 'temperature', None)
        bat = getattr(item, 'charge_level', None)
        self._temp.append(temp if temp is not None else np.nan)
        # charge_level is already a percentage
        self._bat.append(
            max(0.0, min(100.0, bat)) if bat is not None else np.nan
        )
        self._redraw()

    def clear(self):
        for d in (self._t_rt, self._cr, self._dr,
                  self._t_rare, self._temp, self._bat):
            d.clear()
        self._redraw(force=True)

    # ------------------------------------------------------------------ #
    #  Drawing                                                             #
    # ------------------------------------------------------------------ #

    def _filter_window(self, t, *series):
        """Return data within the selected time window (in matplotlib dates)."""
        if not t:
            return (np.array([]),) + tuple(np.array([]) for _ in series)
        t = np.asarray(t, dtype=float)
        if self._window_s is not None:
            cutoff = t[-1] - self._window_s
            mask = t >= cutoff
            t = t[mask]
            series = tuple(np.asarray(s, dtype=float)[mask] for s in series)
        else:
            series = tuple(np.asarray(s, dtype=float) for s in series)
        dates = mdates.date2num([datetime.fromtimestamp(x) for x in t])
        return (dates,) + series

    def _redraw(self, force: bool = False):
        now = time.monotonic()
        if not force and (now - self._last_draw) < _REDRAW_MIN_INTERVAL:
            return
        self._last_draw = now

        t_cr, cr, dr = self._filter_window(self._t_rt, self._cr, self._dr)
        t_env, temp, bat = self._filter_window(self._t_rare, self._temp, self._bat)

        for ax in (self._ax_cr, self._ax_dr, self._ax_env):
            ax.clear()
            ax.set_facecolor('#fafafa')
            ax.grid(True, alpha=0.3, linestyle='--')
        self._ax_bat.clear()

        # Count rate
        if len(t_cr):
            self._ax_cr.plot(t_cr, cr, color=self._color, linewidth=1.0)
        self._ax_cr.set_ylabel('Count rate\n(cps)', fontsize=8)
        self._ax_cr.tick_params(labelbottom=False)

        # Dose rate
        if len(t_cr):
            self._ax_dr.plot(t_cr, dr, color='#d32f2f', linewidth=1.0)
        self._ax_dr.set_ylabel('Dose rate\n(µSv/h)', fontsize=8)
        self._ax_dr.tick_params(labelbottom=False)

        # Temperature + battery (twin axis)
        if len(t_env):
            self._ax_env.plot(t_env, temp, color='#f57c00', linewidth=1.0,
                              label='Temp')
            self._ax_bat.plot(t_env, bat, color='#388e3c', linewidth=1.0,
                              linestyle='--', label='Battery')
        self._ax_env.set_ylabel('Temp (°C)', fontsize=8, color='#f57c00')
        self._ax_bat.set_ylabel('Battery (%)', fontsize=8, color='#388e3c')
        self._ax_bat.set_ylim(0, 100)
        self._ax_env.tick_params(axis='y', labelcolor='#f57c00')
        self._ax_bat.tick_params(axis='y', labelcolor='#388e3c')

        # X axis (time) only on bottom plot
        self._ax_env.xaxis_date()
        self._ax_env.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        for lbl in self._ax_env.get_xticklabels():
            lbl.set_rotation(0)
            lbl.set_fontsize(7)

        if not len(t_cr) and not len(t_env):
            self._ax_cr.text(
                0.5, 0.5, 'No data yet', transform=self._ax_cr.transAxes,
                ha='center', va='center', color='#aaa', fontsize=11,
            )

        self._canvas.draw_idle()
