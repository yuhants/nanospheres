"""Waterfall (spectrogram) view: differential spectrum vs time.

The device reports a *cumulative* spectrum, so each incoming snapshot is
differenced against the previous one to obtain counts accumulated during the
interval. Rows are stacked over time and shown as a 2-D heatmap.
"""

import time
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, Normalize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from radiacode.types import Spectrum

_WINDOWS = [
    ('5 min',  300),
    ('15 min', 900),
    ('1 hour', 3600),
    ('All',    None),
]

# Max number of stacked rows kept in memory
_MAXROWS = 1024


class SpectrumWaterfall(QWidget):
    """Time-vs-energy heatmap of per-interval (differential) spectra."""

    def __init__(self, color: str = '#2196F3', parent=None):
        super().__init__(parent)
        self._times = deque(maxlen=_MAXROWS)     # row timestamps (epoch s)
        self._rows = deque(maxlen=_MAXROWS)      # differential count arrays
        self._prev_counts = None                 # last cumulative snapshot
        self._calib = (0.0, 1.0, 0.0)            # a0, a1, a2

        self._window_s = 900
        self._log = True
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
        self._window_cb.setCurrentText('15 min')
        self._window_cb.currentIndexChanged.connect(self._on_window_change)
        ctrl.addWidget(self._window_cb)
        ctrl.addStretch()
        self._log_cb = QCheckBox('Log color')
        self._log_cb.setChecked(True)
        self._log_cb.toggled.connect(self._on_log_toggle)
        ctrl.addWidget(self._log_cb)
        layout.addLayout(ctrl)

        self._fig = Figure(figsize=(5, 4), tight_layout=True)
        self._fig.patch.set_facecolor('#f8f8f8')
        self._ax = self._fig.add_subplot(111)
        self._cbar = None
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._canvas)

        self._draw_empty()

    def _on_window_change(self, idx: int):
        self._window_s = _WINDOWS[idx][1]
        self._redraw()

    def _on_log_toggle(self, checked: bool):
        self._log = checked
        self._redraw()

    # ------------------------------------------------------------------ #
    #  Data ingestion                                                      #
    # ------------------------------------------------------------------ #

    def add_spectrum(self, spectrum: Spectrum):
        counts = np.asarray(spectrum.counts, dtype=np.float64)
        self._calib = (spectrum.a0, spectrum.a1, spectrum.a2)

        if self._prev_counts is not None and len(self._prev_counts) == len(counts):
            diff = counts - self._prev_counts
            # A spectrum reset makes cumulative counts drop — treat as a fresh
            # baseline and skip this (meaningless) negative row.
            if np.any(diff < 0):
                diff = None
        else:
            diff = None

        self._prev_counts = counts
        if diff is not None:
            self._times.append(time.time())
            self._rows.append(diff)
            self._redraw()

    def clear(self):
        self._times.clear()
        self._rows.clear()
        self._prev_counts = None
        self._draw_empty()

    # ------------------------------------------------------------------ #
    #  Drawing                                                             #
    # ------------------------------------------------------------------ #

    def _draw_empty(self):
        self._ax.clear()
        self._ax.set_xlabel('Energy (keV)')
        self._ax.set_ylabel('Time')
        self._ax.text(
            0.5, 0.5, 'Accumulating spectra…\n(needs ≥2 snapshots)',
            transform=self._ax.transAxes,
            ha='center', va='center', color='#aaa', fontsize=11,
        )
        self._canvas.draw_idle()

    def _redraw(self):
        if not self._rows:
            self._draw_empty()
            return

        times = np.asarray(self._times, dtype=float)
        if self._window_s is not None:
            mask = times >= (times[-1] - self._window_s)
        else:
            mask = np.ones(len(times), dtype=bool)

        sel_times = times[mask]
        rows = [r for r, m in zip(self._rows, mask) if m]
        data = np.vstack(rows)               # shape (n_rows, n_channels)

        a0, a1, a2 = self._calib
        n_chan = data.shape[1]
        ch = np.arange(n_chan)
        energy = a0 + a1 * ch + a2 * ch ** 2

        t_dates = mdates.date2num([datetime.fromtimestamp(t) for t in sel_times])
        # Guard against a singular time extent (single row / identical stamps)
        t0, t1 = float(t_dates[0]), float(t_dates[-1])
        if t1 <= t0:
            t1 = t0 + (5.0 / 86400.0)   # ~5 s in matplotlib date units

        self._ax.clear()
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        if self._log:
            vmax = max(1.0, data.max())
            norm = LogNorm(vmin=1.0, vmax=vmax)
            plot_data = np.clip(data, 1.0, None)
        else:
            norm = Normalize(vmin=0, vmax=max(1.0, data.max()))
            plot_data = data

        # imshow needs uniform spacing; energy axis is near-linear so use extent
        extent = [energy[0], energy[-1], t0, t1]
        im = self._ax.imshow(
            plot_data, aspect='auto', origin='lower',
            extent=extent, norm=norm, cmap='viridis',
            interpolation='nearest',
        )
        self._cbar = self._fig.colorbar(im, ax=self._ax, label='Counts / interval')

        self._ax.set_xlabel('Energy (keV)')
        self._ax.set_ylabel('Time')
        self._ax.set_xlim(0, min(3000.0, float(energy[-1])))
        self._ax.yaxis_date()
        self._ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        for lbl in self._ax.get_yticklabels():
            lbl.set_fontsize(7)

        self._canvas.draw_idle()
