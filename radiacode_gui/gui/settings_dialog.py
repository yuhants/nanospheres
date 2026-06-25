"""Dialog to configure logging: file format and which data streams to save."""

from PyQt5.QtWidgets import (
    QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QGroupBox, QLabel,
    QRadioButton, QVBoxLayout,
)

from ..data_logger import FORMAT_CSV, FORMAT_HDF5, LogConfig


class LoggingSettingsDialog(QDialog):
    """Edit a :class:`LogConfig`; read the result from :meth:`config`."""

    def __init__(self, config: LogConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Logging Settings')
        self.setMinimumWidth(340)
        self._build_ui(config)

    def _build_ui(self, config: LogConfig):
        layout = QVBoxLayout(self)

        # --- File format ---
        fmt_box = QGroupBox('File format')
        fmt_layout = QVBoxLayout(fmt_box)
        self._rb_hdf5 = QRadioButton('HDF5  (single .h5 file per device)')
        self._rb_csv = QRadioButton('CSV  (one file per stream)')
        self._fmt_group = QButtonGroup(self)
        self._fmt_group.addButton(self._rb_hdf5)
        self._fmt_group.addButton(self._rb_csv)
        self._rb_csv.setChecked(config.fmt == FORMAT_CSV)
        self._rb_hdf5.setChecked(config.fmt != FORMAT_CSV)
        fmt_layout.addWidget(self._rb_hdf5)
        fmt_layout.addWidget(self._rb_csv)
        layout.addWidget(fmt_box)

        # --- Data streams ---
        data_box = QGroupBox('Data to save')
        data_layout = QVBoxLayout(data_box)
        self._cb_realtime = QCheckBox('Real-time data (count rate, dose rate)')
        self._cb_status = QCheckBox('Status (temperature, battery)')
        self._cb_spectrum = QCheckBox('Spectrum snapshots')
        self._cb_realtime.setChecked(config.save_realtime)
        self._cb_status.setChecked(config.save_status)
        self._cb_spectrum.setChecked(config.save_spectrum)
        for cb in (self._cb_realtime, self._cb_status, self._cb_spectrum):
            data_layout.addWidget(cb)
        layout.addWidget(data_box)

        note = QLabel(
            'Changes apply to newly recorded files. If recording is already on, '
            'loggers are reopened (a new file is started).'
        )
        note.setWordWrap(True)
        note.setStyleSheet('color: #888; font-size: 9pt;')
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def config(self) -> LogConfig:
        return LogConfig(
            fmt=FORMAT_CSV if self._rb_csv.isChecked() else FORMAT_HDF5,
            save_realtime=self._cb_realtime.isChecked(),
            save_status=self._cb_status.isChecked(),
            save_spectrum=self._cb_spectrum.isChecked(),
        )
