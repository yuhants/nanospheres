"""
Shared building blocks for the lab-task GUI.

Every tab in ``lab_gui/lab_gui.py`` is a ``TaskPanel``: a widget that owns one
instrument, knows how to make it safe again, and can save/restore its settings.
The main window knows nothing about any particular instrument — it only calls
the handful of methods defined here — so adding a task means writing one panel
module and adding one line to ``lab_gui/registry.py``.

Also here, because more than one panel needs them:

  ConnectionBar   VISA address combo + Connect/Disconnect + status line
  AutoOffBox      "turn the output off after N minutes", with a countdown
  SectionFrame    the collapsible box a panel sits in on the single-page window

None of them touch hardware themselves; they are widgets that emit signals.
"""

import os
import time
from typing import List, Optional

from PyQt5.QtWidgets import (
    QWidget, QGroupBox, QGridLayout, QHBoxLayout, QVBoxLayout, QComboBox,
    QPushButton, QLabel, QCheckBox, QDoubleSpinBox,
)
from PyQt5.QtCore import pyqtSignal, QTimer


PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presets')


def visa_addresses() -> List[str]:
    """Every VISA resource currently visible, or [] if pyvisa cannot look."""
    try:
        import pyvisa
        return list(pyvisa.ResourceManager().list_resources())
    except Exception:
        return []


def fmt_clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f'{seconds // 60:02d}:{seconds % 60:02d}'


# ── Task panel ─────────────────────────────────────────────────────────────────

class TaskPanel(QWidget):
    """
    Base class for one routine task.

    Subclasses set ``title`` and implement the hardware bits.  The contract the
    main window relies on:

      is_active()   is this panel currently driving something into the trap?
      stop()        make it safe — outputs off.  Must not raise, and must be
                    safe to call when nothing is on or nothing is connected.
      shutdown()    stop() and close the port; called once, on exit.

    ``stop`` is what the window's "All outputs off" button and its close handler
    call on every panel in turn, so a panel that cannot be made safe by that one
    call is a panel that can be left driving the sphere by accident.

    Signals
    -------
    log_message   : a line for the shared log
    state_changed : connection or output state moved (window refreshes its
                    status bar and the tab colour)
    """
    log_message   = pyqtSignal(str)
    state_changed = pyqtSignal()

    title = 'Task'

    # ── For the main window ─────────────────────────────────────────────────────

    def is_active(self) -> bool:
        """True while an output is on."""
        return False

    def status_text(self) -> str:
        """One short line for the status bar, e.g. 'E field: 5 Vpp @ 87 kHz'."""
        return ''

    def stop(self) -> None:
        """Turn every output off.  Never raises."""

    def shutdown(self) -> None:
        """Stop and release the instrument.  Never raises."""
        self.stop()

    # ── Presets ─────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {}

    def from_dict(self, d: dict) -> None:
        pass

    # ── Helpers for subclasses ──────────────────────────────────────────────────

    def log(self, message: str) -> None:
        self.log_message.emit(message)

    def safe(self, func, what: str):
        """
        Run a hardware call, logging failures instead of raising.

        Returns ``FAILED`` if the call raised — a sentinel rather than None,
        because setters legitimately return None when they succeed.
        """
        try:
            return func()
        except Exception as exc:
            self.log(f'{what} failed: {exc}')
            return FAILED


FAILED = object()


# ── Connection ─────────────────────────────────────────────────────────────────

class ConnectionBar(QGroupBox):
    """
    Instrument connection group: an editable VISA address, rescan, connect,
    disconnect, and a status line.

    The address combo is editable and keeps whatever was typed after a rescan,
    so a hard-coded address from a script still works when the instrument is
    switched off and does not enumerate.

    Signals
    -------
    connect_requested    : (address)
    disconnect_requested : ()
    """
    connect_requested    = pyqtSignal(str)
    disconnect_requested = pyqtSignal()

    def __init__(self, title: str, default_address: str = '',
                 parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        layout = QGridLayout(self)

        self._address = QComboBox()
        self._address.setEditable(True)
        self._address.setMinimumWidth(320)
        self._default_address = default_address
        self.refresh()

        refresh_btn = QPushButton('⟳')
        refresh_btn.setMaximumWidth(28)
        refresh_btn.setToolTip('Rescan VISA resources')
        refresh_btn.clicked.connect(self.refresh)

        self._connect_btn    = QPushButton('Connect')
        self._disconnect_btn = QPushButton('Disconnect')
        self._connect_btn.clicked.connect(
            lambda: self.connect_requested.emit(self.address()))
        self._disconnect_btn.clicked.connect(self.disconnect_requested.emit)

        self._status = QLabel('not connected')
        self._status.setStyleSheet('color: gray;')
        self._status.setWordWrap(True)

        layout.addWidget(QLabel('VISA:'),      0, 0)
        layout.addWidget(self._address,        0, 1, 1, 2)
        layout.addWidget(refresh_btn,          0, 3)
        layout.addWidget(self._connect_btn,    1, 1)
        layout.addWidget(self._disconnect_btn, 1, 2)
        layout.addWidget(self._status,         2, 0, 1, 4)
        self.set_connected(False)

    def address(self) -> str:
        return self._address.currentText().strip()

    def set_address(self, address: str) -> None:
        self._address.setCurrentText(address)

    def refresh(self) -> None:
        current = self.address() or self._default_address
        found = visa_addresses()
        if current and current not in found:
            found.insert(0, current)
        self._address.clear()
        self._address.addItems(found)
        self._address.setCurrentText(current)

    def set_connected(self, connected: bool, description: str = '') -> None:
        self._connect_btn.setEnabled(not connected)
        self._disconnect_btn.setEnabled(connected)
        self._address.setEnabled(not connected)
        self._status.setText(description or
                             ('connected' if connected else 'not connected'))
        self._status.setStyleSheet('color: #060;' if connected else 'color: gray;')


# ── Auto-off ───────────────────────────────────────────────────────────────────

class AutoOffBox(QGroupBox):
    """
    Optional dead-man timer: turn the output off by itself after a set time.

    ``apply_hv.py`` had this as a hard-coded 600 s loop and
    ``charge_measurement.py`` ran until Ctrl-C; here it is a tick box, so a tab
    left alone does not keep driving the trap indefinitely.

    The host arms it when it turns the output on and cancels it when the output
    goes off; ``expired`` means "turn your output off now".
    """
    expired = pyqtSignal()

    def __init__(self, minutes: float = 5.0, checked: bool = True,
                 parent: Optional[QWidget] = None):
        super().__init__('Auto off', parent)
        layout = QHBoxLayout(self)

        self._check = QCheckBox('Turn output off after')
        self._check.setChecked(checked)

        self._minutes = QDoubleSpinBox()
        self._minutes.setRange(0.1, 24 * 60.0)
        self._minutes.setDecimals(1)
        self._minutes.setValue(minutes)
        self._minutes.setSuffix(' min')

        self._countdown = QLabel('')
        self._countdown.setStyleSheet('font-weight: bold;')

        layout.addWidget(self._check)
        layout.addWidget(self._minutes)
        layout.addStretch()
        layout.addWidget(self._countdown)

        self._deadline: Optional[float] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def is_armed(self) -> bool:
        return self._check.isChecked()

    def minutes(self) -> float:
        return self._minutes.value()

    def start(self) -> None:
        """Begin counting down, if the box is ticked."""
        self.cancel()
        if not self.is_armed():
            return
        self._deadline = time.time() + self._minutes.value() * 60.0
        self._timer.start(500)
        self._tick()

    def cancel(self) -> None:
        self._timer.stop()
        self._deadline = None
        self._countdown.setText('')

    def _tick(self) -> None:
        if self._deadline is None:
            return
        remaining = self._deadline - time.time()
        if remaining <= 0:
            self.cancel()
            self.expired.emit()
            return
        self._countdown.setText(f'{fmt_clock(remaining)} left')

    def to_dict(self) -> dict:
        return {'auto_off': self._check.isChecked(),
                'auto_off_minutes': self._minutes.value()}

    def from_dict(self, d: dict) -> None:
        if 'auto_off' in d:
            self._check.setChecked(bool(d['auto_off']))
        if 'auto_off_minutes' in d:
            self._minutes.setValue(float(d['auto_off_minutes']))


# ── Sections ───────────────────────────────────────────────────────────────────

class SectionFrame(QGroupBox):
    """
    One task's box on the single-page window: a header you can click to fold the
    section away, and the panel underneath it.

    The header carries the panel's own status line, so a folded section still
    says whether it is driving something — folding is for screen space, not for
    hiding a live output.
    """

    def __init__(self, panel: 'TaskPanel', collapsed: bool = False,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.panel = panel

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QHBoxLayout()
        self._toggle = QPushButton()
        self._toggle.setFlat(True)
        self._toggle.setMaximumWidth(24)
        self._toggle.clicked.connect(lambda: self.set_collapsed(not self._collapsed))

        self._title = QLabel(panel.title)
        self._title.setStyleSheet('font-weight: bold;')
        self._status = QLabel('')

        header.addWidget(self._toggle)
        header.addWidget(self._title)
        header.addStretch()
        header.addWidget(self._status)
        layout.addLayout(header)
        layout.addWidget(panel)

        self._collapsed = False
        self.set_collapsed(collapsed)
        self.refresh_status()

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self.panel.setVisible(not collapsed)
        self._toggle.setText('▸' if collapsed else '▾')

    def refresh_status(self) -> None:
        """Repaint the header from the panel's state (the window calls this)."""
        active = self.panel.is_active()
        self._status.setText(self.panel.status_text() if active else '')
        self._status.setStyleSheet('color: #b00; font-weight: bold;')
        self._title.setText(('● ' if active else '') + self.panel.title)
        self._title.setStyleSheet(
            'font-weight: bold; color: #b00;' if active else 'font-weight: bold;')
