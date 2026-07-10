"""
backscatter_reader.py — short-snippet RMS reader for the trap monitor.

The backscatter photodetector (usually PicoScope channel D) gives a quick,
hardware-independent handle on whether something is trapped and roughly how
much it is moving.  A flat / low-RMS trace means an empty trap; a single
trapped sphere shows a characteristic motional RMS; a cluster typically reads
larger / noisier.  We grab a short streaming snippet, compute its RMS (about
the mean), and hand it back together with the raw waveform so it can be stored
alongside the camera capture.

`BackscatterReader` wraps `daq.picoscope_daq.PicoscopeDAQ` for a single short
acquisition of one channel.  `MockBackscatterReader` provides a synthetic
signal so the trap-monitor GUI runs with `--mock` and no PicoScope attached
(mirrors `MockCamera` in thorlabs_camera.py).

The PicoScope SDK (`picosdk`) is imported lazily so importing this module never
fails on a machine without the drivers — the GUI can still run in mock mode.
"""

import os
import sys
import time
import math
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PicoScope serials, mirrored from daq.picoscope_daq so the GUI can offer a
# dropdown without importing picosdk (which would fail without drivers).
PICO_SERIALS = {
    'Cloud  (JO279/0118)': b'JO279/0118',
    'Local  (JY140/0294)': b'JY140/0294',
}

# Range labels / mV values, indexed the same way as picoscope_daq.
RANGE_LABELS = ['10 mV', '20 mV', '50 mV', '100 mV', '200 mV', '500 mV',
                '1 V',   '2 V',   '5 V',   '10 V',   '20 V',   '50 V']
RANGE_MV = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]


def _summarize(waveform_mv: np.ndarray, dt: float) -> dict:
    """Compute the standard backscatter statistics from a mV waveform."""
    wf   = np.asarray(waveform_mv, dtype=np.float64)
    mean = float(wf.mean()) if wf.size else 0.0
    rms  = float(np.sqrt(np.mean((wf - mean) ** 2))) if wf.size else 0.0
    pk2pk = float(wf.max() - wf.min()) if wf.size else 0.0
    return {
        'rms_mv':   rms,        # RMS about the mean — the motional amplitude
        'mean_mv':  mean,       # DC level
        'pk2pk_mv': pk2pk,
        'dt':       float(dt),
        'fs':       (1.0 / dt) if dt else 0.0,
        'n':        int(wf.size),
    }


class BackscatterReader:
    """
    Single-channel short-snippet reader built on PicoscopeDAQ.

    Typical use::

        rdr = BackscatterReader(serial=b'JO279/0118', channel='D', range_idx=8)
        rdr.connect()
        res = rdr.read()          # {'rms_mv', 'mean_mv', 'waveform', 'adc2mv', ...}
        rdr.disconnect()

    The connection is held open so repeated `read()` calls (e.g. for a live RMS
    display) only pay the streaming cost, not the open-unit cost.
    """

    def __init__(self, serial: bytes = b'JO279/0118', channel: str = 'D',
                 range_idx: int = 8, coupling: str = 'AC',
                 sample_interval: int = 1, sample_units: str = 'PS4000A_US',
                 n_samples: int = 100_000):
        self.serial          = serial
        self.channel         = channel.upper()
        self.range_idx       = range_idx
        self.coupling        = coupling
        self.sample_interval = sample_interval
        self.sample_units    = sample_units
        self.n_samples       = n_samples
        self._daq            = None
        self._connected      = False

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        # Lazy import: only touch picosdk when real hardware is requested.
        from daq.picoscope_daq import (
            PicoscopeDAQ, AcquisitionConfig, ChannelConfig)
        cfg = AcquisitionConfig(
            channels=[ChannelConfig(self.channel, self.range_idx,
                                    coupling=self.coupling)],
            sample_interval=self.sample_interval,
            sample_units=self.sample_units,
            buffer_size=self.n_samples,
        )
        self._daq = PicoscopeDAQ(cfg)
        self._daq.connect(self.serial)
        self._connected = True

    def read(self) -> dict:
        """
        Acquire one snippet and return a dict with:
            rms_mv, mean_mv, pk2pk_mv, dt, fs, n   (stats)
            waveform  int16 ndarray   (raw counts, channel D)
            adc2mv    float            (counts → mV scale factor)
            channel, range_mv, coupling
        """
        if not self._connected or self._daq is None:
            raise RuntimeError('BackscatterReader not connected')
        _ts, dt, adc2mvs, data = self._daq.acquire_one()
        counts = data[0]
        adc2mv = float(adc2mvs[0])
        stats  = _summarize(counts.astype(np.float64) * adc2mv, dt)
        stats.update({
            'waveform': counts.copy(),
            'adc2mv':   adc2mv,
            'channel':  self.channel,
            'range_mv': RANGE_MV[self.range_idx],
            'coupling': self.coupling,
        })
        return stats

    def disconnect(self) -> None:
        if self._daq is not None:
            try:
                self._daq.disconnect()
            except Exception:
                pass
            self._daq = None
        self._connected = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.disconnect()


class MockBackscatterReader:
    """
    Synthetic backscatter source for GUI development without a PicoScope.

    Generates a damped sinusoid (single-particle-like motion) plus noise whose
    RMS slowly drifts, so the live readout and capture path can be exercised.
    Matches the BackscatterReader interface used by the GUI.
    """

    def __init__(self, serial: bytes = b'MOCK', channel: str = 'D',
                 range_idx: int = 8, coupling: str = 'AC',
                 sample_interval: int = 1, sample_units: str = 'PS4000A_US',
                 n_samples: int = 100_000):
        self.serial          = serial
        self.channel         = channel.upper()
        self.range_idx       = range_idx
        self.coupling        = coupling
        self.sample_interval = sample_interval
        self.sample_units    = sample_units
        self.n_samples       = min(n_samples, 50_000)   # keep mock cheap
        self._connected      = False
        self._t0             = 0.0

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True
        self._t0 = time.time()

    def read(self) -> dict:
        if not self._connected:
            raise RuntimeError('MockBackscatterReader not connected')
        t  = time.time() - self._t0
        dt = self.sample_interval * 1e-6   # treat units as microseconds
        n  = self.n_samples
        ts = np.arange(n) * dt
        rng = np.random.default_rng(int(t * 5))
        # Motional tone ~ 80 kHz whose amplitude breathes over ~8 s, plus noise.
        amp = 40.0 * (0.6 + 0.4 * math.sin(t * 2 * math.pi / 8.0))
        wf_mv = (amp * np.sin(2 * math.pi * 80_000 * ts)
                 + rng.normal(0, 6, n))
        adc2mv = RANGE_MV[self.range_idx] / 32767.0
        counts = np.clip(wf_mv / adc2mv, -32767, 32767).astype(np.int16)
        # Light pause so the live-RMS thread doesn't spin at full CPU.
        time.sleep(0.1)
        stats = _summarize(counts.astype(np.float64) * adc2mv, dt)
        stats.update({
            'waveform': counts,
            'adc2mv':   adc2mv,
            'channel':  self.channel,
            'range_mv': RANGE_MV[self.range_idx],
            'coupling': self.coupling,
        })
        return stats

    def disconnect(self) -> None:
        self._connected = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.disconnect()
