"""Matplotlib-based spectrum display widget."""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox

from radiacode.types import Spectrum


def _channel_to_energy(channels: np.ndarray, a0: float, a1: float, a2: float) -> np.ndarray:
    return a0 + a1 * channels + a2 * channels ** 2


class SpectrumWidget(QWidget):
    """Displays an energy-calibrated gamma spectrum with log/linear y-axis toggle."""

    def __init__(self, color: str = '#2196F3', parent=None):
        super().__init__(parent)
        self._color = color
        self._spectrum: Spectrum = None
        self._log_scale = True
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.addStretch()
        self._log_cb = QCheckBox('Log scale')
        self._log_cb.setChecked(True)
        self._log_cb.toggled.connect(self._on_log_toggle)
        ctrl.addWidget(self._log_cb)
        layout.addLayout(ctrl)

        # Matplotlib canvas
        self._fig = Figure(figsize=(5, 3), tight_layout=True)
        self._fig.patch.set_facecolor('#f8f8f8')
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._canvas)

        self._draw_empty()

    def _draw_empty(self):
        self._ax.clear()
        self._ax.set_xlabel('Energy (keV)')
        self._ax.set_ylabel('Counts')
        self._ax.set_facecolor('#fafafa')
        self._ax.text(
            0.5, 0.5, 'No spectrum data',
            transform=self._ax.transAxes,
            ha='center', va='center', color='#aaa', fontsize=11,
        )
        self._canvas.draw_idle()

    def update_spectrum(self, spectrum) -> None:
        self._spectrum = spectrum
        self._redraw()

    def _on_log_toggle(self, checked: bool):
        self._log_scale = checked
        self._redraw()

    def _redraw(self):
        if self._spectrum is None:
            self._draw_empty()
            return

        counts = np.array(self._spectrum.counts, dtype=np.float64)
        channels = np.arange(len(counts))
        energies = _channel_to_energy(
            channels,
            self._spectrum.a0,
            self._spectrum.a1,
            self._spectrum.a2,
        )

        self._ax.clear()
        self._ax.set_facecolor('#fafafa')

        # Replace zeros with NaN for log scale so they don't distort the plot
        plot_counts = counts.copy()
        if self._log_scale:
            plot_counts[plot_counts <= 0] = np.nan

        self._ax.fill_between(
            energies, plot_counts,
            step='mid', alpha=0.4, color=self._color,
        )
        self._ax.step(energies, plot_counts, where='mid', color=self._color, linewidth=0.8)

        self._ax.set_xlabel('Energy (keV)')
        self._ax.set_ylabel('Counts')
        duration_s = self._spectrum.duration.total_seconds()
        self._ax.set_title(
            f'Live time: {duration_s:.0f} s', fontsize=9, color='#555'
        )

        if self._log_scale:
            self._ax.set_yscale('log')
        else:
            self._ax.set_yscale('linear')
            self._ax.set_ylim(bottom=0)

        # Sensible x range: stop at ~3000 keV
        max_e = min(3000.0, float(energies[-1]))
        self._ax.set_xlim(0, max_e)
        self._ax.grid(True, alpha=0.3, linestyle='--')

        self._canvas.draw_idle()
