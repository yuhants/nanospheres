"""Configurable data logging — HDF5 or CSV, with selectable data streams.

Use :func:`create_logger` to build a logger from a :class:`LogConfig`. Both
backends expose the same interface::

    log_realtime(item)   # RealTimeData  (count rate / dose rate, ~1 Hz)
    log_status(item)     # RareData      (temperature / battery, slow)
    log_spectrum(spec)   # Spectrum      (energy histogram snapshot)
    close()

Disabled streams are silently ignored, so callers don't need to branch.
"""

import csv
import datetime
import logging
import os
from dataclasses import dataclass
from typing import Optional, TextIO

import h5py
import numpy as np

from radiacode.types import RareData, RealTimeData, Spectrum

log = logging.getLogger(__name__)

# Supported file formats
FORMAT_HDF5 = 'hdf5'
FORMAT_CSV = 'csv'


@dataclass
class LogConfig:
    """What to record and in which file format."""

    fmt: str = FORMAT_HDF5            # FORMAT_HDF5 or FORMAT_CSV
    save_realtime: bool = True        # count rate + dose rate (~1 Hz)
    save_status: bool = True          # temperature + battery (RareData)
    save_spectrum: bool = True        # energy spectrum snapshots

    def copy(self) -> 'LogConfig':
        return LogConfig(
            self.fmt, self.save_realtime, self.save_status, self.save_spectrum
        )


def _base_filename(serial: str) -> str:
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    safe = serial.replace('/', '_').replace(':', '_').strip()
    return f'RC_{safe}_{ts}'


def create_logger(
    config: LogConfig,
    output_dir: str,
    address: str,
    serial: str,
    firmware: str,
):
    """Factory: return an HDF5 or CSV logger per ``config.fmt``."""
    os.makedirs(output_dir, exist_ok=True)
    if config.fmt == FORMAT_CSV:
        return CsvLogger(config, output_dir, address, serial, firmware)
    return Hdf5Logger(config, output_dir, address, serial, firmware)


# ---------------------------------------------------------------------------
#  HDF5 backend
# ---------------------------------------------------------------------------

class Hdf5Logger:
    """Appends selected streams to a single HDF5 file.

    File layout (groups created only for enabled streams)::

        /metadata          (attrs: ble_address, serial, firmware, start_time)
        /realtime/         timestamp, count_rate, dose_rate, count_rate_err, dose_rate_err
        /status/           timestamp, temperature, charge_level, duration, dose
        /spectrum_snapshots/snapshot_NNNN/  counts  (attrs: timestamp, duration_s, a0..a2)
    """

    def __init__(self, config, output_dir, address, serial, firmware):
        self._cfg = config
        filename = os.path.join(output_dir, _base_filename(serial) + '.h5')
        self._file = h5py.File(filename, 'w')
        self._file.attrs['ble_address'] = address
        self._file.attrs['serial'] = serial
        self._file.attrs['firmware'] = firmware
        self._file.attrs['start_time'] = datetime.datetime.now().isoformat()
        self.path = filename

        if config.save_realtime:
            rt = self._file.create_group('realtime')
            self._rt = {
                'timestamp':      rt.create_dataset('timestamp',      (0,), maxshape=(None,), dtype='f8'),
                'count_rate':     rt.create_dataset('count_rate',     (0,), maxshape=(None,), dtype='f4'),
                'dose_rate':      rt.create_dataset('dose_rate',      (0,), maxshape=(None,), dtype='f4'),
                'count_rate_err': rt.create_dataset('count_rate_err', (0,), maxshape=(None,), dtype='f4'),
                'dose_rate_err':  rt.create_dataset('dose_rate_err',  (0,), maxshape=(None,), dtype='f4'),
            }
            self._n_rt = 0

        if config.save_status:
            st = self._file.create_group('status')
            self._st = {
                'timestamp':    st.create_dataset('timestamp',    (0,), maxshape=(None,), dtype='f8'),
                'temperature':  st.create_dataset('temperature',  (0,), maxshape=(None,), dtype='f4'),
                'charge_level': st.create_dataset('charge_level', (0,), maxshape=(None,), dtype='f4'),
                'duration':     st.create_dataset('duration',     (0,), maxshape=(None,), dtype='f8'),
                'dose':         st.create_dataset('dose',         (0,), maxshape=(None,), dtype='f4'),
            }
            self._n_st = 0

        if config.save_spectrum:
            self._spec_grp = self._file.create_group('spectrum_snapshots')
            self._n_spec = 0

        log.info('Logging %s → %s', serial, filename)

    @staticmethod
    def _append(ds_map, n, values):
        for key, ds in ds_map.items():
            ds.resize((n + 1,))
            ds[n] = values[key]

    def log_realtime(self, item: RealTimeData) -> None:
        if not self._cfg.save_realtime:
            return
        ts = item.dt.timestamp() if item.dt else datetime.datetime.now().timestamp()
        self._append(self._rt, self._n_rt, {
            'timestamp': ts,
            'count_rate': item.count_rate,
            'dose_rate': item.dose_rate,
            'count_rate_err': getattr(item, 'count_rate_err', 0.0),
            'dose_rate_err': getattr(item, 'dose_rate_err', 0.0),
        })
        self._n_rt += 1

    def log_status(self, item: RareData) -> None:
        if not self._cfg.save_status:
            return
        ts = item.dt.timestamp() if item.dt else datetime.datetime.now().timestamp()
        self._append(self._st, self._n_st, {
            'timestamp': ts,
            'temperature': getattr(item, 'temperature', float('nan')),
            'charge_level': getattr(item, 'charge_level', float('nan')),
            'duration': getattr(item, 'duration', 0),
            'dose': getattr(item, 'dose', float('nan')),
        })
        self._n_st += 1

    def log_spectrum(self, spectrum: Spectrum) -> None:
        if not self._cfg.save_spectrum:
            return
        grp = self._spec_grp.create_group(f'snapshot_{self._n_spec:04d}')
        grp.attrs['timestamp']  = datetime.datetime.now().isoformat()
        grp.attrs['duration_s'] = spectrum.duration.total_seconds()
        grp.attrs['a0'] = spectrum.a0
        grp.attrs['a1'] = spectrum.a1
        grp.attrs['a2'] = spectrum.a2
        grp.create_dataset('counts', data=np.array(spectrum.counts, dtype=np.int32))
        self._n_spec += 1
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
            log.info('Logger closed: %s', self.path)
        except Exception as e:
            log.error('Error closing %s: %s', self.path, e)


# ---------------------------------------------------------------------------
#  CSV backend
# ---------------------------------------------------------------------------

class CsvLogger:
    """Writes selected streams to one CSV file per stream.

    Files (only for enabled streams), sharing a common ``RC_<serial>_<ts>`` prefix::

        ..._realtime.csv   timestamp_iso, count_rate, dose_rate, count_rate_err, dose_rate_err
        ..._status.csv     timestamp_iso, temperature_C, charge_level_pct, duration_s, dose
        ..._spectrum.csv   timestamp_iso, duration_s, a0, a1, a2, ch0 … chN
    """

    def __init__(self, config, output_dir, address, serial, firmware):
        self._cfg = config
        self._prefix = os.path.join(output_dir, _base_filename(serial))
        self.path = self._prefix + '_*.csv'

        self._rt_f: Optional[TextIO] = None
        self._st_f: Optional[TextIO] = None
        self._spec_f: Optional[TextIO] = None
        self._spec_header_written = False

        if config.save_realtime:
            self._rt_f = open(self._prefix + '_realtime.csv', 'w', newline='')
            self._rt_w = csv.writer(self._rt_f)
            self._rt_w.writerow(
                ['timestamp_iso', 'count_rate', 'dose_rate',
                 'count_rate_err', 'dose_rate_err']
            )

        if config.save_status:
            self._st_f = open(self._prefix + '_status.csv', 'w', newline='')
            self._st_w = csv.writer(self._st_f)
            self._st_w.writerow(
                ['timestamp_iso', 'temperature_C', 'charge_level_pct',
                 'duration_s', 'dose']
            )

        if config.save_spectrum:
            self._spec_f = open(self._prefix + '_spectrum.csv', 'w', newline='')
            self._spec_w = csv.writer(self._spec_f)

        log.info('Logging %s → %s_*.csv', serial, self._prefix)

    @staticmethod
    def _iso(item) -> str:
        dt = getattr(item, 'dt', None)
        return (dt or datetime.datetime.now()).isoformat()

    def log_realtime(self, item: RealTimeData) -> None:
        if not self._rt_f:
            return
        self._rt_w.writerow([
            self._iso(item), item.count_rate, item.dose_rate,
            getattr(item, 'count_rate_err', ''), getattr(item, 'dose_rate_err', ''),
        ])
        self._rt_f.flush()

    def log_status(self, item: RareData) -> None:
        if not self._st_f:
            return
        self._st_w.writerow([
            self._iso(item),
            getattr(item, 'temperature', ''),
            getattr(item, 'charge_level', ''),
            getattr(item, 'duration', ''),
            getattr(item, 'dose', ''),
        ])
        self._st_f.flush()

    def log_spectrum(self, spectrum: Spectrum) -> None:
        if not self._spec_f:
            return
        if not self._spec_header_written:
            n = len(spectrum.counts)
            self._spec_w.writerow(
                ['timestamp_iso', 'duration_s', 'a0', 'a1', 'a2']
                + [f'ch{i}' for i in range(n)]
            )
            self._spec_header_written = True
        self._spec_w.writerow(
            [datetime.datetime.now().isoformat(),
             spectrum.duration.total_seconds(),
             spectrum.a0, spectrum.a1, spectrum.a2]
            + list(spectrum.counts)
        )
        self._spec_f.flush()

    def close(self) -> None:
        for f in (self._rt_f, self._st_f, self._spec_f):
            if f:
                try:
                    f.close()
                except Exception as e:
                    log.error('Error closing %s: %s', getattr(f, 'name', '?'), e)
        log.info('Logger closed: %s_*.csv', self._prefix)
