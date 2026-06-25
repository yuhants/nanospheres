"""Per-device panel showing live dose rate, count rate, and spectrum."""

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
)

from radiacode.types import RareData, RealTimeData, Spectrum

from .spectrum_widget import SpectrumWidget
from .time_series_widget import TimeSeriesWidget
from .spectrum_waterfall import SpectrumWaterfall

# Dose-rate thresholds (µSv/h) for colour coding
_LOW  = 0.3
_HIGH = 1.0

_COLORS = [
    '#2196F3', '#4CAF50', '#FF9800',
    '#E91E63', '#9C27B0', '#00BCD4',
]
_color_idx = 0


def _next_color() -> str:
    global _color_idx
    c = _COLORS[_color_idx % len(_COLORS)]
    _color_idx += 1
    return c


def _dose_color(dose_rate: float) -> str:
    if dose_rate >= _HIGH:
        return '#d32f2f'   # red
    if dose_rate >= _LOW:
        return '#f57c00'   # orange
    return '#388e3c'       # green


class _BigLabel(QLabel):
    """Large centred value label."""

    def __init__(self, text: str = '—', parent=None):
        super().__init__(text, parent)
        font = QFont()
        font.setPointSize(28)
        font.setBold(True)
        self.setFont(font)
        self.setAlignment(Qt.AlignCenter)

    def set_value(self, value: str, color: str = '#212121'):
        self.setText(value)
        self.setStyleSheet(f'color: {color};')


class _SmallLabel(QLabel):
    """Smaller unit/caption label."""

    def __init__(self, text: str = '', parent=None):
        super().__init__(text, parent)
        font = QFont()
        font.setPointSize(10)
        self.setFont(font)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('color: #757575;')


class DevicePanel(QWidget):
    """Full detail view for one connected RadiaCode device."""

    def __init__(self, address: str, name: str, serial: str, firmware: str, parent=None):
        super().__init__(parent)
        self.address = address
        self.name = name
        self.serial = serial
        self.firmware = firmware
        self._color = _next_color()
        self._last_rare: RareData = None
        self._device_manager = None   # set by MainWindow
        self._connected = True

        self._build_ui()

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # Header
        header = QHBoxLayout()
        title = QLabel(f'<b>{self.name}</b>')
        title.setStyleSheet('font-size: 14pt;')
        header.addWidget(title)
        self._status_lbl = QLabel('● Connected')
        self._status_lbl.setStyleSheet('color: #388e3c; font-size: 10pt;')
        header.addWidget(self._status_lbl)
        header.addStretch()
        info = QLabel(
            f'Serial: {self.serial}  &nbsp;|&nbsp;  FW: {self.firmware}  '
            f'&nbsp;|&nbsp;  <span style="color:#aaa">{self.address}</span>'
        )
        info.setTextFormat(Qt.RichText)
        header.addWidget(info)
        root.addLayout(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color: #ddd;')
        root.addWidget(sep)

        # Live metrics row
        metrics = QHBoxLayout()
        metrics.setSpacing(0)

        self._dose_val  = _BigLabel('—')
        self._dose_unit = _SmallLabel('µSv/h')
        self._cr_val    = _BigLabel('—')
        self._cr_unit   = _SmallLabel('cps')
        self._temp_val  = _BigLabel('—')
        self._temp_unit = _SmallLabel('°C')
        self._bat_val   = _BigLabel('—')
        self._bat_unit  = _SmallLabel('battery')

        for val, unit, title_text in [
            (self._dose_val,  self._dose_unit, 'Dose Rate'),
            (self._cr_val,    self._cr_unit,   'Count Rate'),
            (self._temp_val,  self._temp_unit, 'Temperature'),
            (self._bat_val,   self._bat_unit,  'Battery'),
        ]:
            box = QGroupBox(title_text)
            box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            vb = QVBoxLayout(box)
            vb.addWidget(val)
            vb.addWidget(unit)
            metrics.addWidget(box)

        root.addLayout(metrics)

        # Plot tabs: live spectrum, time series, spectrum waterfall
        self._plot_tabs = QTabWidget()
        self._spectrum = SpectrumWidget(color=self._color)
        self._time_series = TimeSeriesWidget(color=self._color)
        self._waterfall = SpectrumWaterfall(color=self._color)
        self._plot_tabs.addTab(self._spectrum, 'Spectrum')
        self._plot_tabs.addTab(self._time_series, 'Time Series')
        self._plot_tabs.addTab(self._waterfall, 'Waterfall')
        root.addWidget(self._plot_tabs, stretch=1)

        # Button row
        btn_row = QHBoxLayout()

        self._btn_conn = QPushButton('Disconnect')
        self._btn_conn.setToolTip(
            'Disconnect the device but keep this dashboard and its data'
        )
        self._btn_conn.clicked.connect(self._on_toggle_connection)
        btn_row.addWidget(self._btn_conn)

        btn_row.addStretch()

        self._btn_reset_spec = QPushButton('Reset Spectrum')
        self._btn_reset_spec.clicked.connect(self._on_reset_spectrum)
        btn_row.addWidget(self._btn_reset_spec)

        self._btn_reset_dose = QPushButton('Reset Dose')
        self._btn_reset_dose.clicked.connect(self._on_reset_dose)
        btn_row.addWidget(self._btn_reset_dose)

        root.addLayout(btn_row)

        # Stale-data indicator: grey out if no update for 5 s
        self._stale_timer = QTimer(self)
        self._stale_timer.setInterval(5000)
        self._stale_timer.timeout.connect(self._mark_stale)

    # ------------------------------------------------------------------ #
    #  Data update slots (connected in MainWindow)                         #
    # ------------------------------------------------------------------ #

    def on_realtime(self, item: RealTimeData):
        dr = item.dose_rate
        cr = item.count_rate
        color = _dose_color(dr)

        self._dose_val.set_value(f'{dr:.4f}', color)
        self._cr_val.set_value(f'{cr:.1f}')

        # Reset stale timer
        self._stale_timer.start()
        self._dose_val.setStyleSheet(f'color: {color};')
        self._cr_val.setStyleSheet('color: #212121;')

        self._time_series.add_realtime(item)

    def on_rare(self, item: RareData):
        self._last_rare = item
        if hasattr(item, 'temperature') and item.temperature is not None:
            self._temp_val.set_value(f'{item.temperature:.1f}')
        if hasattr(item, 'charge_level') and item.charge_level is not None:
            # charge_level is already a percentage (decoder divides raw by 100)
            pct = max(0.0, min(100.0, item.charge_level))
            self._bat_val.set_value(f'{pct:.0f}%')
        self._time_series.add_rare(item)

    def on_spectrum(self, spectrum: Spectrum):
        self._spectrum.update_spectrum(spectrum)
        self._waterfall.add_spectrum(spectrum)

    def _mark_stale(self):
        self._dose_val.setStyleSheet('color: #bdbdbd;')
        self._cr_val.setStyleSheet('color: #bdbdbd;')
        self._stale_timer.stop()

    # ------------------------------------------------------------------ #
    #  Connection state                                                    #
    # ------------------------------------------------------------------ #

    def set_connected(self, connected: bool):
        """Update the panel chrome without discarding any displayed data."""
        self._connected = connected
        if connected:
            self._status_lbl.setText('● Connected')
            self._status_lbl.setStyleSheet('color: #388e3c; font-size: 10pt;')
            self._btn_conn.setText('Disconnect')
        else:
            self._status_lbl.setText('● Disconnected')
            self._status_lbl.setStyleSheet('color: #bdbdbd; font-size: 10pt;')
            self._btn_conn.setText('Reconnect')
            self._stale_timer.stop()
            # Grey the live values; plots keep their accumulated history.
            self._dose_val.setStyleSheet('color: #bdbdbd;')
            self._cr_val.setStyleSheet('color: #bdbdbd;')
        # Live-control buttons only make sense while connected
        self._btn_reset_spec.setEnabled(connected)
        self._btn_reset_dose.setEnabled(connected)

    # ------------------------------------------------------------------ #
    #  Button handlers                                                     #
    # ------------------------------------------------------------------ #

    def set_device_manager(self, mgr):
        self._device_manager = mgr

    def _on_toggle_connection(self):
        if not self._device_manager:
            return
        if self._connected:
            self._device_manager.disconnect_device(self.address)
        else:
            self._device_manager.connect_device(self.address, self.name)

    def _on_reset_spectrum(self):
        if self._device_manager:
            self._device_manager.request_spectrum_reset(self.address)
            self._spectrum.update_spectrum(None)
            self._waterfall.clear()

    def _on_reset_dose(self):
        if self._device_manager:
            self._device_manager.request_dose_reset(self.address)
