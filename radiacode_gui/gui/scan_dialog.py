"""BLE scan dialog — shows all discovered devices and lets the user connect."""

from typing import List

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QVBoxLayout,
)

_RC_NAMES = ('rc-', 'radiacode')


def _is_radiacode(name: str) -> bool:
    return any(k in (name or '').lower() for k in _RC_NAMES)


class ScanDialog(QDialog):
    """Modal dialog that triggers a BLE scan and returns selected device(s)."""

    connect_requested = pyqtSignal(str, str)   # address, name

    def __init__(self, device_manager, already_connected: List[str], parent=None):
        super().__init__(parent)
        self._mgr = device_manager
        self._already_connected = set(already_connected)
        self._scanning = False

        self.setWindowTitle('Connect to RadiaCode Device')
        self.setMinimumWidth(520)
        self.setMinimumHeight(420)
        self._build_ui()
        self._connect_signals()
        QTimer.singleShot(100, self._start_scan)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Status label
        self._status_label = QLabel('Scanning for BLE devices…')
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet('color: #1565C0;')
        layout.addWidget(self._status_label)

        # Device list
        list_box = QGroupBox('Discovered BLE Devices  (RadiaCode devices shown first)')
        list_layout = QVBoxLayout(list_box)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setMinimumHeight(180)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.itemSelectionChanged.connect(self._update_connect_btn)
        list_layout.addWidget(self._list)

        rescan_row = QHBoxLayout()
        rescan_row.addStretch()
        self._scan_btn = QPushButton('Re-scan  (5 s)')
        self._scan_btn.clicked.connect(self._start_scan)
        self._scan_btn.setEnabled(False)
        rescan_row.addWidget(self._scan_btn)
        list_layout.addLayout(rescan_row)
        layout.addWidget(list_box)

        # Manual entry
        manual_box = QGroupBox('Or enter MAC address manually')
        manual_box.setToolTip(
            'Find the MAC address in:\n'
            '• Windows Settings › Bluetooth & devices › View more devices\n'
            '• The RadiaCode device screen (Info / About)\n'
            '• Format: AA:BB:CC:DD:EE:FF'
        )
        manual_layout = QHBoxLayout(manual_box)
        self._mac_edit = QLineEdit()
        self._mac_edit.setPlaceholderText('AA:BB:CC:DD:EE:FF')
        self._mac_edit.setMaxLength(17)
        self._mac_edit.textChanged.connect(self._update_connect_btn)
        manual_layout.addWidget(self._mac_edit)
        layout.addWidget(manual_box)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._connect_btn = QPushButton('Connect')
        self._connect_btn.setDefault(True)
        self._connect_btn.setEnabled(False)
        self._connect_btn.clicked.connect(self._on_connect)
        btn_row.addWidget(self._connect_btn)

        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Hint
        hint = QLabel(
            'Important: RadiaCode devices can only connect to one host at a time. '
            'If the phone Radiacode app is running, force-quit it first — the device '
            'stops advertising while connected to the phone. Wait ~15 s after '
            'closing the phone app, then Re-scan.'
        )
        hint.setWordWrap(True)
        hint.setStyleSheet('color: #757575; font-size: 9pt;')
        layout.addWidget(hint)

    def _connect_signals(self):
        self._mgr.scan_finished.connect(self._on_scan_finished)
        self._mgr.scan_error.connect(self._on_scan_error)
        self._mgr.device_connected.connect(self._on_device_connected)

    def _start_scan(self):
        if self._scanning:
            return
        self._scanning = True
        self._list.clear()
        self._status_label.setText('Scanning for BLE devices…  (5 s)')
        self._status_label.setStyleSheet('color: #1565C0;')
        self._scan_btn.setEnabled(False)
        self._connect_btn.setEnabled(False)
        self._mgr.start_scan(timeout=5.0)

    def _on_scan_finished(self, results: list):
        self._scanning = False
        self._list.clear()

        rc_found = [r for r in results if _is_radiacode(r[1])]
        other    = [r for r in results if not _is_radiacode(r[1])]

        if not results:
            self._status_label.setText(
                'No BLE devices found.  Make sure RadiaCode devices are powered on, '
                'then click Re-scan.  Or enter a MAC address manually below.'
            )
            self._status_label.setStyleSheet('color: #c62828;')
        else:
            n_rc = len(rc_found)
            msg = f'Found {len(results)} BLE device(s)'
            if n_rc:
                msg += f', {n_rc} RadiaCode'
            self._status_label.setText(msg)
            self._status_label.setStyleSheet('color: #2e7d32;')

        for address, name, rssi in results:
            connected = address in self._already_connected
            is_rc = _is_radiacode(name)

            rssi_str = f'{rssi} dBm' if rssi != -999 else '?'
            label = f'{name}  —  {address}  ({rssi_str})'
            if not is_rc:
                label = f'[other]  {label}'
            if connected:
                label += '  [connected]'

            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, (address, name))
            if is_rc and not connected:
                item.setForeground(Qt.darkBlue)
            if connected:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self._list.addItem(item)

        self._scan_btn.setEnabled(True)
        self._update_connect_btn()

    def _on_scan_error(self, msg: str):
        self._scanning = False
        self._status_label.setText(f'Scan error: {msg}')
        self._status_label.setStyleSheet('color: #c62828;')
        self._scan_btn.setEnabled(True)

    def _update_connect_btn(self):
        list_ok = False
        item = self._list.currentItem()
        if item and (item.flags() & Qt.ItemIsEnabled):
            list_ok = True
        mac_ok = len(self._mac_edit.text().strip()) == 17
        self._connect_btn.setEnabled(list_ok or mac_ok)

    def _on_double_click(self, item: QListWidgetItem):
        if item.flags() & Qt.ItemIsEnabled:
            self._emit_connect_item(item)

    def _on_connect(self):
        mac_text = self._mac_edit.text().strip()
        if mac_text and len(mac_text) == 17:
            self._emit_connect_address(mac_text, f'RC [{mac_text[-8:]}]')
            return
        item = self._list.currentItem()
        if item and (item.flags() & Qt.ItemIsEnabled):
            self._emit_connect_item(item)

    def _emit_connect_item(self, item: QListWidgetItem):
        address, name = item.data(Qt.UserRole)
        self._already_connected.add(address)
        item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
        item.setText(item.text() + '  [connecting…]')
        self._connect_btn.setEnabled(False)
        self.connect_requested.emit(address, name)

    def _emit_connect_address(self, address: str, name: str):
        self._already_connected.add(address)
        self._mac_edit.setEnabled(False)
        self._connect_btn.setEnabled(False)
        self.connect_requested.emit(address, name)

    def _on_device_connected(self, address: str, *_):
        self._already_connected.add(address)
        for i in range(self._list.count()):
            item = self._list.item(i)
            data = item.data(Qt.UserRole)
            if data and data[0] == address and '[connecting…]' in item.text():
                item.setText(item.text().replace('[connecting…]', '[connected]'))
