"""
Calibration pulses — the Quantum Composers 9614, as a section.

This is a thin adapter around ``pulse_gen/pulse_panel.py``'s
``PulseControlPanel``: connection, pulse shape and the manual output gate, which
is the "turn pulses on for a moment" job.  It deliberately leaves out the
amplitude sweep — for a stepped calibration run use the sweep in
``pulse_gen/pulse_gui.py``, or the DAQ GUI's Pulses tab, which steps per file
and records the settings into the HDF5.

It also shows what wrapping an existing panel costs: implement the five
TaskPanel methods and forward two signals.
"""

from PyQt5.QtWidgets import QVBoxLayout

from pulse_gen.pulse_panel import PulseControlPanel

from lab_gui.task_panel import TaskPanel


class PulseTaskPanel(TaskPanel):

    title = 'Calibration pulses'

    def __init__(self, parent=None):
        super().__init__(parent)
        self._panel = PulseControlPanel(show_schematic=True, show_log=False)
        self._panel.log_message.connect(self.log_message.emit)
        self._panel.state_changed.connect(self.state_changed.emit)

        layout = QVBoxLayout(self)
        layout.addWidget(self._panel)
        layout.addStretch()

    def is_active(self) -> bool:
        return self._panel.output_is_on

    def status_text(self) -> str:
        if not self._panel.output_is_on:
            return ''
        return f'pulses ON — {self._panel.amplitude():.2f} V'

    def stop(self) -> None:
        try:
            self._panel.output_off()
        except Exception as exc:
            self.log(f'pulse output off failed: {exc}')

    def shutdown(self) -> None:
        try:
            self._panel.shutdown()
        except Exception as exc:
            self.log(f'pulse shutdown failed: {exc}')

    def to_dict(self) -> dict:
        return self._panel.to_dict()

    def from_dict(self, d: dict) -> None:
        self._panel.from_dict(d)
