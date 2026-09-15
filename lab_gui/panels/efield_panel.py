"""
E-field drive — the Tektronix AFG1022 half of ``control/charge_measurement.py``.

The script sets a sine on channel 1 and a DC level on channel 2, turns both on,
then sits in a ``while True`` until Ctrl-C:

    tek.sine_wave(addr, amplitude=AMP, frequency=FREQ, offset=OFFSET1, channel=1)
    tek.dc_offset(addr, offset=OFFSET2, channel=2)
    tek.turn_on(addr, channel=1); tek.turn_on(addr, channel=2)

This panel is that, with the numbers editable while it runs and an auto-off
timer instead of Ctrl-C.

Amplifier convention (kept exactly as in the script): the values you type are
the ones you want *at the electrodes*.  With "Amplified" ticked the amplitude is
divided by the gain and the two offsets by **minus** the gain, because the HV
amplifier inverts.  The "sent to generator" line always shows what actually goes
out over VISA, so there is no guessing which side of the divider a number is on.

The panel holds the VISA session open rather than reopening it per call the way
``AFG1022_control``'s module-level helpers do — the same connection is reused
for every knob turn, so changing the amplitude mid-run is immediate.

Transient-free changes
----------------------
Turning the knob on the front panel changes one digit at a time and touches
nothing else; the naive "write everything" apply does neither, and both
differences kick the sphere:

1. *Redundant writes.*  Re-sending ``FUNCtion:SHAPe SIN``, ``FREQuency`` and
   ``BURSt:STATe OFF`` restarts the generator's waveform — the output glitches
   even when the numbers did not change.  ``_live`` mirrors what is actually on
   the instrument so ``_write_state`` only sends parameters that moved, and the
   shape/burst commands are sent at most once per connection, and only if the
   generator is not already in that state.

2. *Steps.*  A single write takes the drive from 5 to 10 Vpp in one sample; the
   sphere sees a step in force.  With "Ramp" ticked and the output on, a
   ``QTimer`` walks amplitude, both offsets and the frequency to their new
   values over the ramp time in ~50 ms steps.  Switching the output on and off
   ramps from and to zero the same way.

What a ramp cannot smooth
-------------------------
A ramp is still a staircase: every write is a step in the force on the sphere,
and the step is what rings it, not the time between steps.  So the size of the
step, not the length of the ramp, is the knob that matters — hence "at most N
volts per write", which lengthens a big change rather than stepping harder.
The envelope is a raised cosine so the steps also taper to nothing at both ends
instead of starting and stopping abruptly.

Two things the generator does by itself are outside the ramp entirely:

* The **output relay** clicks when ``OUTPut:STATe`` opens or closes it, and that
  click reaches the electrodes at full speed no matter how gently the drive was
  ramped.  "Leave the output enabled" avoids ever operating it in normal use —
  OFF ramps to zero drive and stops there.
* The **output attenuator** switches ranges at fixed amplitudes, and each switch
  is a discontinuity of its own.  Nothing here can suppress that; what it can do
  is not cross ranges needlessly, which is why a change between two working
  amplitudes no longer goes anywhere near zero.

Why the ramp writes SCPI directly
---------------------------------
``FuncGenChannel.set_amplitude`` queries the offset (and ``set_offset`` the
amplitude) to range-check them, so every ramp write was two queries and a write.
The AFG1022 intermittently answers a query with an empty string — the driver's
own ``open()`` works around it by asking twice — and ``float('')`` then aborted
the ramp mid-way, which on a ramp *down* left the drive stuck at whatever it had
reached.  ``_write_state`` therefore does its own range check against the
limits the driver already holds locally (no I/O) and writes the SCPI itself.
Nothing is queried during a ramp, which removes the failure mode and halves the
traffic.  A write that does fail is retried on the next tick, and a ramp down
that cannot be finished falls back to switching the output off rather than
leaving a field on the sphere.

``stop()`` — the window's "All outputs OFF", Escape, and the auto-off timer —
never ramps and always opens the relay: making the trap safe must not be
something that takes seconds, and a relay click is cheaper than a lost sphere.
"""

import math
from typing import Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QVBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton, QCheckBox,
    QDoubleSpinBox, QSpinBox,
)

from control.src.Tektronix_control.AFG1022.AFG1022_control import FuncGen

from lab_gui.task_panel import TaskPanel, ConnectionBar, AutoOffBox, FAILED


DEFAULT_ADDRESS = 'USB0::0x0699::0x0353::2238362::INSTR'

# HV amplifier gain assumed by charge_measurement.py (inverting, hence the sign
# flip applied to the DC offsets).
DEFAULT_GAIN = 20.0

# Generator resolution: 0.1 mV on the voltages, and there is no point rewriting
# a frequency that moved by less than a µHz.  Anything smaller than these is not
# a change, it is float noise, and writing it would glitch the output.
V_EPS  = 1e-4
HZ_EPS = 1e-6

# Consecutive failed writes before a ramp gives up rather than retrying.
RAMP_RETRIES = 3


class EFieldPanel(TaskPanel):
    """Sine drive on one channel, DC bias on the other."""

    title = 'E field'

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gen: Optional[FuncGen] = None
        self._output_on = False       # the output relay is closed
        self._at_zero   = False       # relay closed but ramped down to no drive
        # Mirror of what is actually on the generator, so a change writes only
        # the parameters that moved.  Seeded by _seed_live() on connect.
        self._live: dict = {}
        self._amp_min = 0.002          # highZ minimum, refreshed on connect
        self._ramp: Optional[dict] = None
        self._ramp_timer = QTimer(self)
        self._ramp_timer.timeout.connect(self._ramp_tick)

        layout = QVBoxLayout(self)
        self._connection = ConnectionBar('Function generator (Tektronix AFG1022)',
                                         DEFAULT_ADDRESS)
        self._connection.connect_requested.connect(self._connect)
        self._connection.disconnect_requested.connect(self._disconnect)
        layout.addWidget(self._connection)
        # Built first because the drive group's widgets refer to it, added
        # second because it belongs under the numbers it smooths.
        ramp_group = self._build_ramp_group()
        layout.addWidget(self._build_drive_group())
        layout.addWidget(ramp_group)

        self._auto_off = AutoOffBox(minutes=10.0, checked=False)
        self._auto_off.expired.connect(self._on_auto_off)
        layout.addWidget(self._auto_off)
        layout.addStretch()

        self._update_widgets()

    # ── Construction ────────────────────────────────────────────────────────────

    def _build_drive_group(self) -> QGroupBox:
        group  = QGroupBox('Drive')
        layout = QGridLayout(group)

        self._amplified = QCheckBox('Amplified')
        self._amplified.setChecked(True)
        self._amplified.setToolTip(
            'Values below are what you want at the electrodes;\n'
            'the generator gets amplitude/gain and offset/(-gain).')

        self._gain = QDoubleSpinBox()
        self._gain.setRange(1.0, 1000.0)
        self._gain.setDecimals(1)
        self._gain.setValue(DEFAULT_GAIN)
        self._gain.setPrefix('× ')

        self._amp = QDoubleSpinBox()
        self._amp.setRange(0.0, 400.0)
        self._amp.setDecimals(3)
        self._amp.setSingleStep(0.5)
        self._amp.setValue(5.0)
        self._amp.setSuffix(' Vpp')

        self._freq = QDoubleSpinBox()
        self._freq.setRange(1e-3, 25e6)
        self._freq.setDecimals(1)
        self._freq.setSingleStep(1000.0)
        self._freq.setValue(87000.0)
        self._freq.setSuffix(' Hz')

        self._offset1 = QDoubleSpinBox()
        self._offset2 = QDoubleSpinBox()
        for spin in (self._offset1, self._offset2):
            spin.setRange(-400.0, 400.0)
            spin.setDecimals(2)
            spin.setSingleStep(5.0)
            spin.setValue(0.0)
            spin.setSuffix(' V')

        self._sine_channel = QSpinBox()
        self._sine_channel.setRange(1, 2)
        self._sine_channel.setValue(1)
        self._dc_channel = QSpinBox()
        self._dc_channel.setRange(1, 2)
        self._dc_channel.setValue(2)

        self._use_dc = QCheckBox('Drive the DC channel too')
        self._use_dc.setChecked(True)
        self._use_dc.setToolTip('Untick to leave the second channel alone '
                                '(sine only, as when both offsets are 0).')

        self._sent_label = QLabel('')
        self._sent_label.setStyleSheet('color: gray;')
        self._sent_label.setWordWrap(True)

        self._apply_btn = QPushButton('Apply settings')
        self._apply_btn.setToolTip('Push the values to the generator without '
                                   'changing the output state')
        self._apply_btn.clicked.connect(lambda: self.apply_settings())

        self._output_btn = QPushButton('⏻  E field ON')
        self._output_btn.clicked.connect(self.toggle_output)

        self._state_label = QLabel('output off')
        self._state_label.setStyleSheet('color: gray;')

        for spin in (self._amp, self._freq, self._offset1, self._offset2, self._gain):
            spin.valueChanged.connect(self._on_settings_changed)
        self._amplified.stateChanged.connect(self._on_settings_changed)
        self._use_dc.stateChanged.connect(self._on_channels_changed)
        self._sine_channel.valueChanged.connect(self._on_channels_changed)
        self._dc_channel.valueChanged.connect(self._on_channels_changed)

        layout.addWidget(self._amplified,             0, 0)
        layout.addWidget(QLabel('gain:'),             0, 1)
        layout.addWidget(self._gain,                  0, 2)
        layout.addWidget(QLabel('Amplitude:'),        1, 0)
        layout.addWidget(self._amp,                   1, 1, 1, 2)
        layout.addWidget(QLabel('Frequency:'),        2, 0)
        layout.addWidget(self._freq,                  2, 1, 1, 2)
        layout.addWidget(QLabel('Sine channel:'),     3, 0)
        layout.addWidget(self._sine_channel,          3, 1)
        layout.addWidget(QLabel('DC bias (sine ch):'), 4, 0)
        layout.addWidget(self._offset1,               4, 1, 1, 2)
        layout.addWidget(QLabel('DC channel:'),       5, 0)
        layout.addWidget(self._dc_channel,            5, 1)
        layout.addWidget(self._use_dc,                5, 2)
        layout.addWidget(QLabel('DC bias (DC ch):'),  6, 0)
        layout.addWidget(self._offset2,               6, 1, 1, 2)
        layout.addWidget(self._sent_label,            7, 0, 1, 3)
        layout.addWidget(self._apply_btn,             8, 0)
        layout.addWidget(self._output_btn,            8, 1)
        layout.addWidget(self._state_label,           8, 2)

        self._on_settings_changed()
        return group

    def _build_ramp_group(self) -> QGroupBox:
        group  = QGroupBox('Smooth changes')
        layout = QGridLayout(group)

        self._ramp_on = QCheckBox('Ramp instead of stepping')
        self._ramp_on.setChecked(True)
        self._ramp_on.setToolTip(
            'While the output is on, walk amplitude, offsets and frequency to\n'
            'the new values in small steps instead of one jump.  The jump is an\n'
            'impulse on the sphere; the ramp is what turning the front-panel\n'
            'knob does.')

        self._ramp_time = QDoubleSpinBox()
        self._ramp_time.setRange(0.05, 120.0)
        self._ramp_time.setDecimals(2)
        self._ramp_time.setSingleStep(0.5)
        self._ramp_time.setValue(2.0)
        self._ramp_time.setSuffix(' s')
        self._ramp_time.setToolTip('Time to get from the old values to the new ones')

        self._ramp_step_ms = QSpinBox()
        self._ramp_step_ms.setRange(10, 1000)
        self._ramp_step_ms.setSingleStep(10)
        self._ramp_step_ms.setValue(50)
        self._ramp_step_ms.setSuffix(' ms')
        self._ramp_step_ms.setToolTip(
            'Interval between writes.  Below ~20 ms the VISA round trips, not\n'
            'the timer, set the pace and the ramp just takes longer.')

        self._ramp_max_step = QDoubleSpinBox()
        self._ramp_max_step.setRange(0.005, 100.0)
        self._ramp_max_step.setDecimals(3)
        self._ramp_max_step.setSingleStep(0.05)
        self._ramp_max_step.setValue(0.25)
        self._ramp_max_step.setSuffix(' V/write')
        self._ramp_max_step.setToolTip(
            'Largest change any single write may make, in the same volts you\n'
            'type above.  Every write is a step in force on the sphere, so this\n'
            'is what sets how hard a ramp kicks — the ramp time only sets how\n'
            'long the smallest ramps take.  A change too big to make in the\n'
            'ramp time at this step size takes longer instead of stepping harder.')

        self._ramp_keep_on = QCheckBox('Leave the output enabled at zero drive')
        self._ramp_keep_on.setChecked(True)
        self._ramp_keep_on.setToolTip(
            'The output relay clicks when it opens or closes, and that click\n'
            'reaches the sphere at full speed however gently the drive was\n'
            'ramped.  Ticked, OFF ramps the drive to zero and leaves the relay\n'
            'closed, so switching on and off never operates it.\n'
            '"All outputs OFF", Escape and the auto-off timer still open it.')

        self._ramp_retune = QCheckBox('Retune at zero drive')
        self._ramp_retune.setChecked(True)
        self._ramp_retune.setToolTip(
            'A frequency change while driving sweeps the drive across every\n'
            'frequency in between — including any resonance in that span — and\n'
            'each write restarts the phase.  Ticked, a frequency change ramps\n'
            'the drive down, retunes, and ramps back up.')

        self._ramp_onoff = QCheckBox('Ramp the output on and off as well')
        self._ramp_onoff.setChecked(True)
        self._ramp_onoff.setToolTip(
            'Switch on at zero and ramp up, ramp down before switching off.\n'
            '"All outputs OFF", Escape and the auto-off timer always cut\n'
            'immediately, whatever this is set to.')

        self._ramp_label = QLabel('')
        self._ramp_label.setStyleSheet('color: gray;')

        self._ramp_on.stateChanged.connect(self._update_widgets)

        layout.addWidget(self._ramp_on,          0, 0, 1, 2)
        layout.addWidget(QLabel('over:'),        0, 2)
        layout.addWidget(self._ramp_time,        0, 3)
        layout.addWidget(self._ramp_onoff,       1, 0, 1, 2)
        layout.addWidget(QLabel('in steps of:'), 1, 2)
        layout.addWidget(self._ramp_step_ms,     1, 3)
        layout.addWidget(self._ramp_keep_on,     2, 0, 1, 2)
        layout.addWidget(QLabel('at most:'),     2, 2)
        layout.addWidget(self._ramp_max_step,    2, 3)
        layout.addWidget(self._ramp_retune,      3, 0, 1, 2)
        layout.addWidget(self._ramp_label,       4, 0, 1, 4)
        return group

    # ── Values ──────────────────────────────────────────────────────────────────

    def hardware_values(self):
        """(amplitude, offset on the sine channel, offset on the DC channel) in
        volts as they will be written to the generator."""
        amp  = self._amp.value()
        off1 = self._offset1.value()
        off2 = self._offset2.value()
        if self._amplified.isChecked():
            gain = self._gain.value()
            # + 0.0 so a zero offset reads '0', not '-0', after the sign flip.
            return amp / gain, off1 / -gain + 0.0, off2 / -gain + 0.0
        return amp, off1, off2

    def target_state(self) -> dict:
        """The generator-side numbers the panel currently asks for."""
        amp, off1, off2 = self.hardware_values()
        return {'amp': amp, 'off1': off1, 'off2': off2, 'freq': self._freq.value()}

    def _zero_state(self) -> dict:
        """
        No drive: where a ramp up starts and a ramp down ends.

        The amplitude floor is the generator's minimum, not 0 — ``set_amplitude``
        refuses anything below it — but at /gain into the amplifier that is a
        fraction of a mV at the electrodes.

        The frequency is whatever is *already* being driven, so ramping down
        does not retune on the way; only when the output is off (nothing to
        disturb) does it jump straight to the frequency being asked for.
        """
        freq = (self._live['freq'] if self._output_on and 'freq' in self._live
                else self.target_state()['freq'])
        return {'amp': self._amp_min, 'off1': 0.0, 'off2': 0.0, 'freq': freq}

    def _on_settings_changed(self) -> None:
        amp, off1, off2 = self.hardware_values()
        text = (f'sent to generator: ch{self._sine_channel.value()} sine '
                f'{amp:.4g} Vpp, {self._freq.value():g} Hz, offset {off1:.4g} V')
        if self._use_dc.isChecked():
            text += f' · ch{self._dc_channel.value()} DC {off2:.4g} V'
        self._sent_label.setText(text)
        self._update_widgets()

    # ── Connection ──────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._gen is not None

    def _connect(self, address: str) -> None:
        if self._gen is not None:
            return
        try:
            gen = FuncGen(address, verbose=False)
        except Exception as exc:
            self.log(f'could not connect to {address}: {exc}')
            self._connection.set_connected(False, f'connection failed: {exc}')
            return
        self._gen = gen
        description = f'{gen._maker} {gen._model}, serial {gen._serial}'
        self._connection.set_connected(True, description)
        self.log(f'connected: {description}')

        # Reflect whatever the generator is already doing, so a panel opened
        # onto a live drive does not claim the output is off.
        state = self.safe(
            lambda: gen.channels[self._sine_channel.value() - 1].get_output_state(),
            'read output state')
        if state is not FAILED:
            self._output_on = bool(state)
            self.log(f'sine channel output is currently {"ON" if state else "OFF"}')
        self._seed_live()
        self._update_widgets()
        self.state_changed.emit()

    def _read(self, read, what: str):
        """
        Query the generator, once more if the first attempt throws.

        The AFG1022 sometimes answers a query with an empty string, which comes
        back as ``float('')``; ``FuncGen.open`` has the same workaround.
        """
        try:
            return read()
        except Exception:
            return self.safe(read, what)

    def _seed_live(self) -> None:
        """
        Read back what the generator is set to, so the first apply only writes
        the parameters that actually differ.

        Without this the first apply after connecting rewrites shape, frequency
        and burst state — which restarts the waveform and glitches the output —
        even when the panel is asking for exactly what is already running.
        """
        self._live = {}
        if self._gen is None:
            return
        sine = self._gen.channels[self._sine_channel.value() - 1]
        readings = {
            'amp':  lambda: sine.get_amplitude(),
            'off1': lambda: sine.get_offset(),
            'freq': lambda: sine.get_frequency(),
            'func': lambda: sine.get_function(),
        }
        if self._use_dc.isChecked():
            dc = self._gen.channels[self._dc_channel.value() - 1]
            readings['off2']     = lambda: dc.get_offset()
            readings['dc_amp']   = lambda: dc.get_amplitude()
            readings['dc_func']  = lambda: dc.get_function()
        for key, read in readings.items():
            value = self._read(read, f'read {key}')
            if value is not FAILED:
                self._live[key] = value
        # No getter for burst in the driver, and a blind BURSt:STATe OFF is one
        # of the writes that glitches, so ask the instrument directly.
        burst = self._read(
            lambda: self._gen.query(f'SOURce{self._sine_channel.value()}:BURSt:STATe?'),
            'read burst state')
        if burst is not FAILED:
            self._live['burst'] = burst.strip() not in ('0', 'OFF')
        limits = self.safe(sine.get_amplitude_lims, 'read amplitude limits')
        if limits is not FAILED:
            self._amp_min = limits[0]

    def _on_channels_changed(self) -> None:
        """Channel numbers moved: the mirror describes the wrong channels now."""
        self._cancel_ramp()
        if self.is_connected:
            self._seed_live()
        self._on_settings_changed()

    def _disconnect(self) -> None:
        if self._gen is None:
            return
        self.stop()
        self.safe(self._gen.close, 'close')
        self._gen = None
        self._live = {}
        self._connection.set_connected(False)
        self.log('disconnected')
        self._update_widgets()
        self.state_changed.emit()

    # ── Actions ─────────────────────────────────────────────────────────────────

    def _shape_writes(self) -> None:
        """
        Put the channels in the right mode — sine (continuous) and DC.

        These are the writes that restart the waveform, so each one is sent only
        when the mirror says the generator is not already in that state: on a
        generator that is already driving a continuous sine, this does nothing.
        """
        sine = self._gen.channels[self._sine_channel.value() - 1]
        if str(self._live.get('func', '')).upper()[:3] != 'SIN':
            sine.set_function('SIN')
            self._live['func'] = 'SIN'
        if self._live.get('burst', True):
            sine.set_burst_off()
            self._live['burst'] = False
        if self._use_dc.isChecked():
            dc = self._gen.channels[self._dc_channel.value() - 1]
            if str(self._live.get('dc_func', '')).upper() != 'DC':
                dc.set_function('DC')
                self._live['dc_func'] = 'DC'

    def _check_limits(self, channel, amplitude: float, offset: float) -> None:
        """
        The range check ``set_amplitude``/``set_offset`` do, without their
        queries: both limit lookups are local to the driver's channel object.
        """
        min_amp, max_amp = channel.get_amplitude_lims()
        min_v,   max_v   = channel.get_voltage_lims()
        if not min_amp <= amplitude <= max_amp:
            raise ValueError(f'amplitude {amplitude} Vpp is outside '
                             f'[{min_amp}, {max_amp}] Vpp')
        if amplitude / 2 - offset < min_v or amplitude / 2 + offset > max_v:
            raise ValueError(f'amplitude {amplitude} Vpp with offset {offset} V '
                             f'is outside [{min_v}, {max_v}] V')

    def _write_state(self, state: dict) -> None:
        """
        Write the parts of ``state`` that differ from what is on the generator.

        Raises on a refused or failed write; callers decide what to do about it.

        Amplitude and offset are range-checked against *each other* (|offset| +
        amplitude/2 must stay inside the output range), so a shrinking amplitude
        is written before the offset and a growing one after it — otherwise a
        ramp that trades amplitude for offset can be refused half way through
        even though both its end points are legal.
        """
        n    = self._sine_channel.value()
        sine = self._gen.channels[n - 1]

        def moved(key, value, eps=V_EPS):
            return key not in self._live or abs(value - self._live[key]) >= eps

        freq = round(state['freq'], 6)
        amp  = round(state['amp'],  4)
        off1 = round(state['off1'], 4)
        self._check_limits(sine, amp, off1)

        if moved('freq', freq, HZ_EPS):
            self._gen.write(f'SOURce{n}:FREQuency:FIXed {freq}Hz')
            self._live['freq'] = freq

        def write_amp():
            if moved('amp', amp):
                self._gen.write(f'SOURce{n}:VOLTage:LEVel {amp}Vpp')
                self._live['amp'] = amp

        def write_off1():
            if moved('off1', off1):
                self._gen.write(f'SOURce{n}:VOLTage:LEVel:OFFSet {off1}V')
                self._live['off1'] = off1

        shrinking_amp = amp <= self._live.get('amp', float('inf'))
        for write in ((write_amp, write_off1) if shrinking_amp
                      else (write_off1, write_amp)):
            write()

        if self._use_dc.isChecked():
            m    = self._dc_channel.value()
            dc   = self._gen.channels[m - 1]
            off2 = round(state['off2'], 4)
            if moved('off2', off2):
                self._check_limits(dc, self._live.get('dc_amp', 0.0), off2)
                self._gen.write(f'SOURce{m}:VOLTage:LEVel:OFFSet {off2}V')
                self._live['off2'] = off2

    def apply_settings(self) -> bool:
        """
        Bring the generator to the values in the panel.

        With the output off, or ramping unticked, that is a direct write.  While
        driving it starts a ramp from wherever the generator is now and returns
        immediately — ``is_ramping`` stays True until it arrives.
        """
        if self._gen is None:
            self.log('not connected')
            return False
        if self._at_zero:
            # The relay is closed but the drive was deliberately taken to zero.
            # Writing the new values now would bring the field back up without
            # anyone asking for it.
            self.log('drive is at zero — the new values go out when you press ON')
            return True
        try:
            self._shape_writes()
        except Exception as exc:
            self.log(f'apply failed: {exc}')
            return False

        target = self.target_state()
        if self._output_on and self._ramp_on.isChecked():
            retune = (self._ramp_retune.isChecked()
                      and 'freq' in self._live
                      and abs(target['freq'] - self._live['freq']) >= HZ_EPS)
            if retune:
                # Down to zero at the old frequency, retune there, back up: a
                # sweep at full drive would drag the drive across every
                # resonance between the two frequencies.
                self.log(f"retuning {self._live['freq']:g} → {target['freq']:g} Hz "
                         f"at zero drive")
                self._start_ramp(self._zero_state(), label='ramping down to retune',
                                 after=lambda: self._retune_and_resume(target))
            else:
                self._start_ramp(target)
            return True

        self._cancel_ramp()
        try:
            self._write_state(target)
        except Exception as exc:
            self.log(f'apply failed: {exc}')
            return False
        self.log(f"applied: {target['amp']:.4g} Vpp @ {target['freq']:g} Hz, "
                 f"offsets {target['off1']:.4g} / {target['off2']:.4g} V")
        return True

    def _retune_and_resume(self, target: dict) -> None:
        """Second half of a retune: change frequency at zero drive, ramp back."""
        state = self._zero_state()
        state['freq'] = target['freq']
        try:
            self._write_state(state)
        except Exception as exc:
            self.log(f'retune failed: {exc}')
            return
        self._start_ramp(target, label='ramping up')

    # ── Ramping ─────────────────────────────────────────────────────────────────

    @property
    def is_ramping(self) -> bool:
        return self._ramp is not None

    def _ramp_writes(self, start: dict, target: dict) -> int:
        """
        How many writes to get from ``start`` to ``target``.

        Enough to fill the ramp time, and never fewer than the step limit
        allows: a change too big to make in the ramp time takes longer rather
        than stepping harder, because it is the size of the step — not the time
        between steps — that the sphere feels.

        The span is measured in the volts the user typed (electrode volts), and
        the raised-cosine envelope is steepest in the middle at pi/2 times the
        average slope, so that factor comes off the budget.
        """
        by_time = max(1, int(round(self._ramp_time.value() * 1000.0
                                   / self._ramp_step_ms.value())))
        span = max(abs(target[key] - start[key]) for key in ('amp', 'off1', 'off2'))
        span *= self._display_gain()
        by_size = math.ceil(span * (math.pi / 2) / max(self._ramp_max_step.value(), 1e-9))
        return max(1, by_time, by_size)

    @staticmethod
    def _ease(frac: float) -> float:
        """
        Raised cosine, so the ramp starts and stops with zero slope.

        A linear ramp puts its full step size on the very first and very last
        write, which is exactly where the drive is being connected or
        disconnected and where a kick is least welcome.
        """
        return 0.5 - 0.5 * math.cos(math.pi * frac)

    def _start_ramp(self, target: dict, after=None, label: str = 'ramping') -> None:
        """
        Walk from what is on the generator now to ``target``, then call ``after``.

        The start point is the mirror rather than the previous target, so a ramp
        started while another one is still running carries on from where that
        one had got to instead of jumping back.
        """
        self._cancel_ramp()
        start = {key: self._live.get(key, target[key])
                 for key in ('amp', 'off1', 'off2', 'freq')}
        step_ms = self._ramp_step_ms.value()
        writes  = self._ramp_writes(start, target)
        # A ramp towards zero drive is one whose failure must not leave the
        # field where it stopped.
        heading_for_zero = target['amp'] <= self._amp_min + V_EPS
        self._ramp = {'start': start, 'target': target, 'steps': writes, 'i': 0,
                      'after': after, 'label': label, 'fails': 0,
                      'fail_safe': heading_for_zero}
        self._ramp_timer.start(step_ms)
        span = max(abs(target[key] - start[key])
                   for key in ('amp', 'off1', 'off2')) * self._display_gain()
        self.log(f'{label}: {writes} writes of ≤{span * (math.pi / 2) / writes:.3g} V '
                 f'over {writes * step_ms / 1000.0:.1f} s')
        self._update_widgets()
        self.state_changed.emit()

    def _ramp_tick(self) -> None:
        ramp = self._ramp
        if ramp is None or self._gen is None:
            self._cancel_ramp()
            return
        step  = ramp['i'] + 1
        frac  = self._ease(min(1.0, step / ramp['steps']))
        state = {key: ramp['start'][key]
                      + (ramp['target'][key] - ramp['start'][key]) * frac
                 for key in ('amp', 'off1', 'off2', 'freq')}
        try:
            self._write_state(state)
        except Exception as exc:
            self._ramp_step_failed(exc)
            return

        # Only count the step once it is actually on the instrument, so a
        # retried write repeats the same step instead of skipping past it.
        ramp['i'], ramp['fails'] = step, 0
        if step < ramp['steps']:
            self._ramp_label.setText(
                f"{ramp['label']}… {100.0 * step / ramp['steps']:.0f}%  "
                f"(now {state['amp'] * self._display_gain():.3g} Vpp)")
            return

        after = ramp['after']
        self._cancel_ramp()
        if after is None:
            self.log(f"ramped to {state['amp']:.4g} Vpp @ {state['freq']:g} Hz, "
                     f"offsets {state['off1']:.4g} / {state['off2']:.4g} V")
        else:
            after()
        self._update_widgets()
        self.state_changed.emit()

    def _ramp_step_failed(self, exc: Exception) -> None:
        """
        A write did not go through.

        Nearly always transient (a dropped USB reply), so the same step is
        retried on the next tick.  Once it is clearly not transient, a ramp on
        its way down switches the output off rather than leaving the drive
        parked at whatever fraction it reached — a field nobody asked for is
        worse than a relay click.
        """
        ramp = self._ramp
        ramp['fails'] += 1
        percent = 100.0 * ramp['i'] / ramp['steps']
        if ramp['fails'] < RAMP_RETRIES:
            self.log(f"{ramp['label']} at {percent:.0f}%: {exc} — retrying")
            return
        fail_safe = ramp['fail_safe']
        self.log(f"{ramp['label']} gave up at {percent:.0f}%: {exc}")
        self._cancel_ramp()
        if fail_safe:
            self.log('could not finish ramping down — switching the output off')
            self.stop()
        else:
            self._update_widgets()
            self.state_changed.emit()

    def _cancel_ramp(self) -> None:
        self._ramp_timer.stop()
        self._ramp = None
        self._ramp_label.setText('')

    def _display_gain(self) -> float:
        """Factor from generator volts back to electrode volts, for the log."""
        return self._gain.value() if self._amplified.isChecked() else 1.0

    def is_driving(self) -> bool:
        """
        Is there actually a field on the sphere?

        Not the same as the relay being closed: parked at zero drive with the
        output enabled, the relay is closed and nothing is being driven, and the
        button has to read ON.
        """
        return self._output_on and not self._at_zero

    def toggle_output(self) -> None:
        if self._gen is None:
            return
        if self.is_driving():
            self._switch_off()
        else:
            self._switch_on()

    def _switch_on(self) -> None:
        """
        Bring the drive up.

        If the relay is already closed and the drive is merely parked at zero,
        this is a ramp and nothing else — no relay, no shape writes, no step.
        Otherwise the generator is set to zero drive *while it is still off*, so
        closing the relay cannot step the field, and the drive comes up after.
        """
        ramped = self._ramp_on.isChecked() and self._ramp_onoff.isChecked()

        if self._output_on and self._at_zero:
            self._at_zero = False
            self._auto_off.start()
            if ramped:
                self._start_ramp(self.target_state(), label='ramping up')
            else:
                try:
                    self._write_state(self.target_state())
                except Exception as exc:
                    self.log(f'could not bring the drive up: {exc}')
                    self._at_zero = True
                    return
                self.log('E field back up')
            self._update_widgets()
            self.state_changed.emit()
            return

        try:
            self._shape_writes()
            self._write_state(self._zero_state() if ramped else self.target_state())
        except Exception as exc:
            self.log(f'could not switch on: {exc}')
            return
        self._set_outputs('ON')
        self._output_on = True
        self._at_zero   = False
        self._auto_off.start()
        if ramped:
            self._start_ramp(self.target_state(), label='ramping up')
        else:
            self.log('E field switched on')
        self._update_widgets()
        self.state_changed.emit()

    def _switch_off(self) -> None:
        """
        Take the drive down.

        Ramped, that means down to zero first; whether the relay then opens is
        the "leave the output enabled" tick box, because opening it clicks.
        """
        if not (self._ramp_on.isChecked() and self._ramp_onoff.isChecked()):
            self.stop()
            return
        self._auto_off.cancel()
        keep_on = self._ramp_keep_on.isChecked()
        self._start_ramp(self._zero_state(),
                         after=self._finish_idle if keep_on else self._finish_off,
                         label='ramping down')

    def _finish_off(self) -> None:
        """End of a ramp down that opens the relay."""
        self._outputs_off()
        self._output_on = False
        self._at_zero   = False
        self.log('E field ramped down and switched off')

    def _finish_idle(self) -> None:
        """End of a ramp down that leaves the relay closed at zero drive."""
        self._at_zero = True
        self.log('drive at zero — output left enabled, no relay click')

    def _set_outputs(self, state: str) -> None:
        channels = [self._sine_channel.value()]
        if self._use_dc.isChecked() and self._dc_channel.value() not in channels:
            channels.append(self._dc_channel.value())
        for channel in channels:
            self.safe(lambda c=channel: self._gen.channels[c - 1].set_output(state),
                      f'set channel {channel} output {state}')

    def _outputs_off(self) -> None:
        # Both channels regardless of the DC tick box: if it was unticked while
        # the output was on, channel 2 would otherwise stay driving.
        for channel in (1, 2):
            self.safe(lambda c=channel: self._gen.channels[c - 1].set_output('OFF'),
                      f'set channel {channel} output OFF')

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
        if self._at_zero and not self.is_ramping:
            return 'E field output enabled, drive at zero'
        amp, _, _ = self.hardware_values()
        text = (f'E field ON — {self._amp.value():g} Vpp requested '
                f'({amp:.4g} Vpp out) @ {self._freq.value():g} Hz')
        if self.is_ramping:
            done = 100.0 * self._ramp['i'] / self._ramp['steps']
            text += f" · {self._ramp['label']} {done:.0f}%"
        return text

    def stop(self) -> None:
        """Immediate off — never ramps: "All outputs OFF", Escape, auto-off."""
        self._auto_off.cancel()
        self._cancel_ramp()
        if self._gen is not None and self._output_on:
            self._outputs_off()
            self.log('E field switched off')
        self._at_zero = False
        self._output_on = False
        self._update_widgets()
        self.state_changed.emit()

    def shutdown(self) -> None:
        self.stop()
        self._cancel_ramp()
        if self._gen is not None:
            self.safe(self._gen.close, 'close')
            self._gen = None

    def _update_widgets(self) -> None:
        connected = self.is_connected
        self._apply_btn.setEnabled(connected)
        self._output_btn.setEnabled(connected)
        for widget in (self._ramp_time, self._ramp_step_ms, self._ramp_onoff,
                       self._ramp_max_step, self._ramp_keep_on, self._ramp_retune):
            widget.setEnabled(self._ramp_on.isChecked())
        driving = self.is_driving()
        self._output_btn.setText('⏻  E field OFF' if driving else '⏻  E field ON')
        self._output_btn.setStyleSheet(
            'font-weight: bold; color: #b00;' if driving else '')
        if not connected:
            self._state_label.setText('not connected')
            self._state_label.setStyleSheet('color: gray;')
        elif self.is_ramping:
            self._state_label.setText(self._ramp['label'].upper())
            self._state_label.setStyleSheet('color: #b60; font-weight: bold;')
        elif self._at_zero:
            self._state_label.setText('enabled, at zero')
            self._state_label.setStyleSheet('color: #b60;')
        elif self._output_on:
            self._state_label.setText('DRIVING')
            self._state_label.setStyleSheet('color: #b00; font-weight: bold;')
        else:
            self._state_label.setText('output off')
            self._state_label.setStyleSheet('color: gray;')

    # ── Presets ─────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = {
            'address':      self._connection.address(),
            'amplified':    self._amplified.isChecked(),
            'gain':         self._gain.value(),
            'amplitude':    self._amp.value(),
            'frequency':    self._freq.value(),
            'offset1':      self._offset1.value(),
            'offset2':      self._offset2.value(),
            'sine_channel': self._sine_channel.value(),
            'dc_channel':   self._dc_channel.value(),
            'use_dc':       self._use_dc.isChecked(),
            'ramp':         self._ramp_on.isChecked(),
            'ramp_time':    self._ramp_time.value(),
            'ramp_step_ms': self._ramp_step_ms.value(),
            'ramp_onoff':   self._ramp_onoff.isChecked(),
            'ramp_max_step': self._ramp_max_step.value(),
            'ramp_keep_on':  self._ramp_keep_on.isChecked(),
            'ramp_retune':   self._ramp_retune.isChecked(),
        }
        d.update(self._auto_off.to_dict())
        return d

    def from_dict(self, d: dict) -> None:
        if 'address' in d and not self.is_connected:
            self._connection.set_address(d['address'])
        setters = [
            ('amplified',    self._amplified.setChecked,    bool),
            ('gain',         self._gain.setValue,           float),
            ('amplitude',    self._amp.setValue,            float),
            ('frequency',    self._freq.setValue,           float),
            ('offset1',      self._offset1.setValue,        float),
            ('offset2',      self._offset2.setValue,        float),
            ('sine_channel', self._sine_channel.setValue,   int),
            ('dc_channel',   self._dc_channel.setValue,     int),
            ('use_dc',       self._use_dc.setChecked,       bool),
            ('ramp',         self._ramp_on.setChecked,      bool),
            ('ramp_time',    self._ramp_time.setValue,      float),
            ('ramp_step_ms', self._ramp_step_ms.setValue,   int),
            ('ramp_onoff',   self._ramp_onoff.setChecked,   bool),
            ('ramp_max_step', self._ramp_max_step.setValue,  float),
            ('ramp_keep_on',  self._ramp_keep_on.setChecked, bool),
            ('ramp_retune',   self._ramp_retune.setChecked,  bool),
        ]
        for key, setter, cast in setters:
            if key in d:
                setter(cast(d[key]))
        self._auto_off.from_dict(d)
        self._on_settings_changed()
