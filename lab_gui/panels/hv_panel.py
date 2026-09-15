"""
HV trigger — ``control/apply_hv.py`` as a panel.

The script pulses the RIGOL DG822 into the Spellman supply's control input to
strike the needle and change the sphere's charge state:

    DG822.pulse(amp=VOLT, duty=90, freq=FREQ_PULSE, off=-VOLT/2)
    DG822.turn_on()
    ... 600 s or Ctrl-C ...
    DG822.turn_off()

Same thing here, with the 600 s timeout as an editable auto-off (ticked by
default — this one drives a kilovolt supply, so leaving it running is the
failure mode worth designing against).

The control amplitude is in kV because that is what the supply makes of it:
1 V out of the DG822 ≈ 1 kV at the needle.  Below roughly 1 kV it will not
ionise the air, and much above that the supply cannot deliver the current, so
the useful range is narrow.

Offset follows the script's ``-amp/2``, which puts the pulse train between 0 and
+amp rather than straddling zero.  Untick "offset = −amp/2" to set it by hand.
"""

from typing import Optional

from PyQt5.QtWidgets import (
    QVBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton, QCheckBox,
    QDoubleSpinBox, QSpinBox,
)

import control.src.RIGOL_control.DG822.DG822_control as rig

from lab_gui.task_panel import TaskPanel, ConnectionBar, AutoOffBox


DEFAULT_ADDRESS = 'USB0::0x1AB1::0x0643::DG8A204201834::INSTR'

WIRING_NOTE = (
    'Connect DG822 channel 1 to pin B on the Spellman J2 connector:\n'
    '     o  x\n'
    '   o  o  o\n'
    '    o o o')


class HVPanel(TaskPanel):
    """Pulse train that triggers the needle HV supply."""

    title = 'High voltage'

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gen: Optional[rig.FuncGen] = None
        self._output_on = False

        layout = QVBoxLayout(self)
        self._connection = ConnectionBar('Function generator (RIGOL DG822)',
                                         DEFAULT_ADDRESS)
        self._connection.connect_requested.connect(self._connect)
        self._connection.disconnect_requested.connect(self._disconnect)
        layout.addWidget(self._connection)
        layout.addWidget(self._build_pulse_group())

        self._auto_off = AutoOffBox(minutes=10.0, checked=True)
        self._auto_off.expired.connect(self._on_auto_off)
        layout.addWidget(self._auto_off)

        note = QLabel(WIRING_NOTE)
        note.setStyleSheet('color: gray; font-family: monospace;')
        layout.addWidget(note)
        layout.addStretch()

        self._update_widgets()

    # ── Construction ────────────────────────────────────────────────────────────

    def _build_pulse_group(self) -> QGroupBox:
        group  = QGroupBox('Trigger pulse')
        layout = QGridLayout(group)

        self._channel = QSpinBox()
        self._channel.setRange(1, 2)
        self._channel.setValue(1)

        self._volt = QDoubleSpinBox()
        self._volt.setRange(0.0, 2.0)
        self._volt.setDecimals(2)
        self._volt.setSingleStep(0.05)
        self._volt.setValue(1.05)
        self._volt.setSuffix(' kV')
        self._volt.setToolTip('Control amplitude, sent as Vpp to the DG822.\n'
                              'Below ~1 kV the air will not ionise.')

        self._freq = QDoubleSpinBox()
        self._freq.setRange(0.001, 1000.0)
        self._freq.setDecimals(3)
        self._freq.setSingleStep(0.1)
        self._freq.setValue(0.2)
        self._freq.setSuffix(' Hz')

        self._duty = QDoubleSpinBox()
        self._duty.setRange(0.1, 99.9)
        self._duty.setDecimals(1)
        self._duty.setValue(90.0)
        self._duty.setSuffix(' %')

        self._auto_offset = QCheckBox('offset = −amp/2')
        self._auto_offset.setChecked(True)
        self._offset = QDoubleSpinBox()
        self._offset.setRange(-5.0, 5.0)
        self._offset.setDecimals(3)
        self._offset.setValue(-0.525)
        self._offset.setSuffix(' V')

        self._sent_label = QLabel('')
        self._sent_label.setStyleSheet('color: gray;')
        self._sent_label.setWordWrap(True)

        self._apply_btn = QPushButton('Apply settings')
        self._apply_btn.clicked.connect(lambda: self.apply_settings())

        self._output_btn = QPushButton('⚡  HV ON')
        self._output_btn.clicked.connect(self.toggle_output)

        self._state_label = QLabel('output off')
        self._state_label.setStyleSheet('color: gray;')

        for spin in (self._volt, self._freq, self._duty, self._offset):
            spin.valueChanged.connect(self._on_settings_changed)
        self._auto_offset.stateChanged.connect(self._on_settings_changed)

        layout.addWidget(QLabel('Channel:'),   0, 0)
        layout.addWidget(self._channel,        0, 1)
        layout.addWidget(QLabel('Voltage:'),   1, 0)
        layout.addWidget(self._volt,           1, 1, 1, 2)
        layout.addWidget(QLabel('Pulse rate:'), 2, 0)
        layout.addWidget(self._freq,           2, 1, 1, 2)
        layout.addWidget(QLabel('Duty cycle:'), 3, 0)
        layout.addWidget(self._duty,           3, 1, 1, 2)
        layout.addWidget(QLabel('Offset:'),    4, 0)
        layout.addWidget(self._offset,         4, 1)
        layout.addWidget(self._auto_offset,    4, 2)
        layout.addWidget(self._sent_label,     5, 0, 1, 3)
        layout.addWidget(self._apply_btn,      6, 0)
        layout.addWidget(self._output_btn,     6, 1)
        layout.addWidget(self._state_label,    6, 2)

        self._on_settings_changed()
        return group

    # ── Values ──────────────────────────────────────────────────────────────────

    def offset(self) -> float:
        if self._auto_offset.isChecked():
            return -self._volt.value() / 2
        return self._offset.value()

    def _on_settings_changed(self) -> None:
        self._offset.setEnabled(not self._auto_offset.isChecked())
        if self._auto_offset.isChecked():
            self._offset.blockSignals(True)
            self._offset.setValue(-self._volt.value() / 2)
            self._offset.blockSignals(False)
        self._sent_label.setText(
            f'sent to generator: ch{self._channel.value()} pulse '
            f'{self._volt.value():g} Vpp, {self._freq.value():g} Hz, '
            f'{self._duty.value():g} % duty, offset {self.offset():g} V')
        self._update_widgets()

    # ── Connection ──────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._gen is not None

    def _connect(self, address: str) -> None:
        if self._gen is not None:
            return
        try:
            gen = rig.FuncGen(address)
        except Exception as exc:
            self.log(f'could not connect to {address}: {exc}')
            self._connection.set_connected(False, f'connection failed: {exc}')
            return
        self._gen = gen
        description = f'{gen._maker} {gen._model}, serial {gen._serial}'
        self._connection.set_connected(True, description)
        self.log(f'connected: {description}')
        self._update_widgets()
        self.state_changed.emit()

    def _disconnect(self) -> None:
        if self._gen is None:
            return
        self.stop()
        self.safe(self._gen._inst.close, 'close')
        self._gen = None
        self._connection.set_connected(False)
        self.log('disconnected')
        self._update_widgets()
        self.state_changed.emit()

    # ── Actions ─────────────────────────────────────────────────────────────────

    def apply_settings(self) -> bool:
        if self._gen is None:
            self.log('not connected')
            return False
        try:
            self._gen.pulse(channel=self._channel.value(),
                            amp=self._volt.value(),
                            off=self.offset(),
                            freq=self._freq.value(),
                            duty=self._duty.value())
        except Exception as exc:
            self.log(f'apply failed: {exc}')
            return False
        self.log(f'applied: {self._volt.value():g} kV control, '
                 f'{self._freq.value():g} Hz, {self._duty.value():g} % duty')
        return True

    def toggle_output(self) -> None:
        if self._gen is None:
            return
        if self._output_on:
            self.stop()
        else:
            if not self.apply_settings():
                return
            self.safe(lambda: self._gen.turn_on(self._channel.value()), 'turn on')
            self._output_on = True
            self._auto_off.start()
            self.log('HV triggered')
            self._update_widgets()
            self.state_changed.emit()

    def _on_auto_off(self) -> None:
        if self._output_on:
            self.log('auto off reached')
            self.stop()

    # ── TaskPanel contract ──────────────────────────────────────────────────────

    def is_active(self) -> bool:
        return self._output_on

    def status_text(self) -> str:
        if not self._output_on:
            return ''
        return (f'HV ON — {self._volt.value():g} kV control at '
                f'{self._freq.value():g} Hz')

    def stop(self) -> None:
        self._auto_off.cancel()
        if self._gen is not None and self._output_on:
            self.safe(lambda: self._gen.turn_off(self._channel.value()), 'turn off')
            self.log('HV switched off')
        self._output_on = False
        self._update_widgets()
        self.state_changed.emit()

    def shutdown(self) -> None:
        self.stop()
        if self._gen is not None:
            self.safe(self._gen._inst.close, 'close')
            self._gen = None

    def _update_widgets(self) -> None:
        connected = self.is_connected
        self._apply_btn.setEnabled(connected)
        self._output_btn.setEnabled(connected)
        self._output_btn.setText('⚡  HV OFF' if self._output_on else '⚡  HV ON')
        self._output_btn.setStyleSheet(
            'font-weight: bold; color: #b00;' if self._output_on else '')
        if not connected:
            self._state_label.setText('not connected')
            self._state_label.setStyleSheet('color: gray;')
        elif self._output_on:
            self._state_label.setText('HV TRIGGERING')
            self._state_label.setStyleSheet('color: #b00; font-weight: bold;')
        else:
            self._state_label.setText('output off')
            self._state_label.setStyleSheet('color: gray;')

    # ── Presets ─────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = {
            'address':     self._connection.address(),
            'channel':     self._channel.value(),
            'voltage_kv':  self._volt.value(),
            'frequency':   self._freq.value(),
            'duty':        self._duty.value(),
            'auto_offset': self._auto_offset.isChecked(),
            'offset':      self._offset.value(),
        }
        d.update(self._auto_off.to_dict())
        return d

    def from_dict(self, d: dict) -> None:
        if 'address' in d and not self.is_connected:
            self._connection.set_address(d['address'])
        setters = [
            ('channel',     self._channel.setValue,      int),
            ('voltage_kv',  self._volt.setValue,         float),
            ('frequency',   self._freq.setValue,         float),
            ('duty',        self._duty.setValue,         float),
            ('auto_offset', self._auto_offset.setChecked, bool),
            ('offset',      self._offset.setValue,       float),
        ]
        for key, setter, cast in setters:
            if key in d:
                setter(cast(d[key]))
        self._auto_off.from_dict(d)
        self._on_settings_changed()
