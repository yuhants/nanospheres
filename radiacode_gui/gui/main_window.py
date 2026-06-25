"""Main application window."""

import logging
import os
from typing import Dict

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QIcon
from PyQt5.QtWidgets import (
    QAction, QApplication, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSplitter,
    QStatusBar, QTabWidget, QToolBar, QVBoxLayout, QWidget,
)

from radiacode.types import RareData, RealTimeData, Spectrum

from ..device_manager import DeviceManager
from .device_panel import DevicePanel
from .scan_dialog import ScanDialog
from .settings_dialog import LoggingSettingsDialog

log = logging.getLogger(__name__)


class _DeviceListItem(QWidget):
    """Compact device entry shown in the left sidebar."""

    def __init__(self, address: str, name: str, parent=None):
        super().__init__(parent)
        self.address = address

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        self._dot = QLabel('●')
        self._dot.setFixedWidth(14)
        self._dot.setStyleSheet('color: #4CAF50; font-size: 10pt;')
        layout.addWidget(self._dot)

        info = QVBoxLayout()
        info.setSpacing(1)
        self._name_lbl = QLabel(f'<b>{name}</b>')
        self._name_lbl.setTextFormat(Qt.RichText)
        self._dose_lbl = QLabel('—')
        self._dose_lbl.setStyleSheet('font-size: 9pt; color: #555;')
        info.addWidget(self._name_lbl)
        info.addWidget(self._dose_lbl)
        layout.addLayout(info)
        layout.addStretch()

    def update_dose(self, dose_rate: float):
        self._dose_lbl.setText(f'{dose_rate:.4f} µSv/h')

    def set_disconnected(self):
        self._dot.setStyleSheet('color: #bdbdbd; font-size: 10pt;')
        self._dose_lbl.setText('disconnected')
        self._dose_lbl.setStyleSheet('font-size: 9pt; color: #bdbdbd;')

    def set_connected(self):
        self._dot.setStyleSheet('color: #4CAF50; font-size: 10pt;')
        self._dose_lbl.setText('—')
        self._dose_lbl.setStyleSheet('font-size: 9pt; color: #555;')


class MainWindow(QMainWindow):

    def __init__(self, device_manager: DeviceManager, parent=None):
        super().__init__(parent)
        self._mgr = device_manager
        self._panels: Dict[str, DevicePanel]           = {}
        self._list_widgets: Dict[str, _DeviceListItem] = {}
        self._list_items:   Dict[str, QListWidgetItem] = {}
        self._output_dir: str = os.path.join(os.getcwd(), 'radiacode_data')

        self.setWindowTitle('RadiaCode Multi-Device Monitor')
        self.setMinimumSize(1100, 680)

        self._build_ui()
        self._connect_manager()

    # ------------------------------------------------------------------ #
    #  UI                                                                  #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        # Toolbar
        tb = QToolBar('Main')
        tb.setMovable(False)
        self.addToolBar(tb)
        self._toolbar = tb

        self._act_scan = QAction('⟳  Scan', self)
        self._act_scan.setToolTip('Scan for RadiaCode BLE devices')
        self._act_scan.triggered.connect(self._open_scan_dialog)
        tb.addAction(self._act_scan)

        tb.addSeparator()

        self._act_record = QAction('⏺  Record OFF', self)
        self._act_record.setCheckable(True)
        self._act_record.setToolTip('Toggle HDF5 data recording')
        self._act_record.triggered.connect(self._toggle_recording)
        tb.addAction(self._act_record)

        self._act_dir = QAction('📁  Output Dir', self)
        self._act_dir.setToolTip('Choose output directory for recorded files')
        self._act_dir.triggered.connect(self._choose_output_dir)
        tb.addAction(self._act_dir)

        self._act_settings = QAction('⚙  Logging Settings', self)
        self._act_settings.setToolTip('Configure file format and what to save')
        self._act_settings.triggered.connect(self._open_settings_dialog)
        tb.addAction(self._act_settings)

        tb.addSeparator()
        self._dir_label = QLabel(f'  {self._output_dir}')
        self._dir_label.setStyleSheet('color: #555; font-size: 9pt;')
        tb.addWidget(self._dir_label)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage('Ready — click Scan to discover devices.')

        # Central splitter
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Left: device list
        left = QWidget()
        left.setFixedWidth(230)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        hdr = QLabel('Connected Devices')
        hdr.setStyleSheet(
            'background: #1565C0; color: white; padding: 6px; font-weight: bold;'
        )
        left_layout.addWidget(hdr)

        self._device_list = QListWidget()
        self._device_list.setStyleSheet('QListWidget { border: none; }')
        self._device_list.currentRowChanged.connect(self._on_list_select)
        left_layout.addWidget(self._device_list)

        scan_btn = QPushButton('Scan for Devices')
        scan_btn.clicked.connect(self._open_scan_dialog)
        left_layout.addWidget(scan_btn)
        splitter.addWidget(left)

        # Right: tabs
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._on_tab_close)
        self._tabs.setStyleSheet('QTabWidget::pane { border: none; }')
        splitter.addWidget(self._tabs)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._show_placeholder()

    def _show_placeholder(self):
        if self._tabs.count() == 0:
            ph = QWidget()
            ph.setObjectName('placeholder')
            lyt = QVBoxLayout(ph)
            lbl = QLabel(
                'No devices connected.\n\n'
                'Click  ⟳ Scan  in the toolbar or the button below\n'
                'to discover RadiaCode BLE devices.'
            )
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet('color: #9e9e9e; font-size: 13pt;')
            lyt.addWidget(lbl)
            self._tabs.addTab(ph, 'Welcome')
            self._tabs.tabBar().setTabButton(0, self._tabs.tabBar().RightSide, None)

    def _remove_placeholder(self):
        for i in range(self._tabs.count()):
            if self._tabs.widget(i).objectName() == 'placeholder':
                self._tabs.removeTab(i)
                return

    # ------------------------------------------------------------------ #
    #  DeviceManager signal handlers                                       #
    # ------------------------------------------------------------------ #

    def _connect_manager(self):
        m = self._mgr
        m.device_connected.connect(self._on_device_connected)
        m.device_disconnected.connect(self._on_device_disconnected)
        m.device_connect_failed.connect(self._on_connect_failed)
        m.realtime_data.connect(self._on_realtime)
        m.spectrum_data.connect(self._on_spectrum)
        m.rare_data.connect(self._on_rare)
        m.scan_started.connect(lambda: self._status.showMessage('Scanning…'))
        m.scan_finished.connect(
            lambda r: self._status.showMessage(
                f'Scan complete — {len(r)} device(s) found.'
            )
        )
        m.scan_error.connect(
            lambda e: self._status.showMessage(f'Scan error: {e}')
        )

    def _on_device_connected(self, address: str, name: str, serial: str, firmware: str):
        self._remove_placeholder()

        # Reconnect to an existing dashboard: keep the panel and its data.
        existing = self._panels.get(address)
        if existing is not None:
            existing.set_connected(True)
            for i in range(self._tabs.count()):
                if self._tabs.widget(i) is existing:
                    self._tabs.setTabText(i, existing.name)
                    self._tabs.setCurrentIndex(i)
                    break
            widget = self._list_widgets.get(address)
            if widget:
                widget.set_connected()
            self._status.showMessage(f'Reconnected: {existing.name}  |  {serial}')
            return

        panel = DevicePanel(address, name, serial, firmware, self)
        panel.set_device_manager(self._mgr)
        self._panels[address] = panel

        tab_idx = self._tabs.addTab(panel, name)
        self._tabs.setCurrentIndex(tab_idx)

        # Sidebar entry
        list_item = QListWidgetItem(self._device_list)
        widget = _DeviceListItem(address, name)
        list_item.setSizeHint(widget.sizeHint())
        self._device_list.setItemWidget(list_item, widget)
        self._list_widgets[address] = widget
        self._list_items[address]   = list_item

        self._status.showMessage(
            f'Connected: {name}  |  {serial}  |  FW {firmware}'
        )

    def _on_device_disconnected(self, address: str):
        widget = self._list_widgets.get(address)
        if widget:
            widget.set_disconnected()

        # Keep the dashboard/data; just mark it disconnected.
        panel = self._panels.get(address)
        if panel:
            panel.set_connected(False)
            for i in range(self._tabs.count()):
                if self._tabs.widget(i) is panel:
                    self._tabs.setTabText(i, panel.name + ' [off]')
                    break

        self._status.showMessage(f'Device disconnected: {address}')

    def _on_connect_failed(self, address: str, error: str):
        QMessageBox.warning(
            self, 'Connection Failed',
            f'Could not connect to {address}:\n\n{error}'
        )
        self._status.showMessage(f'Connect failed: {address}')

    def _on_realtime(self, address: str, item: RealTimeData):
        panel = self._panels.get(address)
        if panel:
            panel.on_realtime(item)
        widget = self._list_widgets.get(address)
        if widget:
            widget.update_dose(item.dose_rate)

    def _on_spectrum(self, address: str, spectrum: Spectrum):
        panel = self._panels.get(address)
        if panel:
            panel.on_spectrum(spectrum)

    def _on_rare(self, address: str, item: RareData):
        panel = self._panels.get(address)
        if panel:
            panel.on_rare(item)

    # ------------------------------------------------------------------ #
    #  UI event handlers                                                   #
    # ------------------------------------------------------------------ #

    def _open_scan_dialog(self):
        dlg = ScanDialog(self._mgr, self._mgr.connected_addresses(), self)
        dlg.connect_requested.connect(self._mgr.connect_device)
        dlg.exec_()

    def _on_list_select(self, row: int):
        item = self._device_list.item(row)
        if item is None:
            return
        widget = self._device_list.itemWidget(item)
        if widget is None:
            return
        address = widget.address
        panel = self._panels.get(address)
        if panel:
            for i in range(self._tabs.count()):
                if self._tabs.widget(i) is panel:
                    self._tabs.setCurrentIndex(i)
                    break

    def _on_tab_close(self, index: int):
        panel = self._tabs.widget(index)
        if not isinstance(panel, DevicePanel):
            self._tabs.removeTab(index)
            return
        address = panel.address
        reply = QMessageBox.question(
            self, 'Disconnect',
            f'Disconnect from {panel.name}?',
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._mgr.disconnect_device(address)
            self._tabs.removeTab(index)
            self._panels.pop(address, None)
            # Remove from sidebar
            item = self._list_items.pop(address, None)
            if item:
                row = self._device_list.row(item)
                self._device_list.takeItem(row)
            self._list_widgets.pop(address, None)
            if self._tabs.count() == 0:
                self._show_placeholder()

    def _toggle_recording(self, checked: bool):
        if checked:
            if not os.path.isdir(self._output_dir):
                try:
                    os.makedirs(self._output_dir, exist_ok=True)
                except Exception as e:
                    QMessageBox.critical(self, 'Error', f'Cannot create output dir:\n{e}')
                    self._act_record.setChecked(False)
                    return
            self._mgr.set_output_dir(self._output_dir)
            self._mgr.set_logging_enabled(True)
            self._act_record.setText('⏺  Record ON')
            self._style_record_button('color: #d32f2f; font-weight: bold;')
            self._status.showMessage(f'Recording to {self._output_dir}')
        else:
            self._mgr.set_logging_enabled(False)
            self._act_record.setText('⏺  Record OFF')
            self._style_record_button('')
            self._status.showMessage('Recording stopped.')

    def _style_record_button(self, style: str):
        # QAction has no setStyleSheet; style the toolbar button widget instead.
        btn = self._toolbar.widgetForAction(self._act_record)
        if btn is not None:
            btn.setStyleSheet(style)

    def _open_settings_dialog(self):
        dlg = LoggingSettingsDialog(self._mgr.get_log_config(), self)
        if dlg.exec_() == dlg.Accepted:
            cfg = dlg.config()
            self._mgr.set_log_config(cfg)
            streams = [s for s, on in (
                ('realtime', cfg.save_realtime),
                ('status', cfg.save_status),
                ('spectrum', cfg.save_spectrum),
            ) if on]
            self._status.showMessage(
                f'Logging: {cfg.fmt.upper()} — saving: '
                f'{", ".join(streams) or "nothing"}'
            )

    def _choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, 'Select Output Directory', self._output_dir
        )
        if path:
            self._output_dir = path
            self._dir_label.setText(f'  {path}')
            self._mgr.set_output_dir(path)

    def closeEvent(self, event):
        self._mgr.stop()
        event.accept()
