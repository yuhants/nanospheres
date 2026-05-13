"""
PicoscopeDAQ — reusable acquisition class for PicoScope 4000a.

Solves the global-variable callback problem documented in daq_dev_notes.txt:
the streaming callback is created as a closure inside acquire_one(), so all
mutable state (next_sample, total_buffer, etc.) is captured through the closure
rather than module-level globals.  Each call to acquire_one() gets its own
fresh state dict and its own C function pointer.
"""

import ctypes
import time
import h5py
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from picosdk.ps4000a import ps4000a as ps
from picosdk.functions import assert_pico_ok
from picosdk.constants import PICO_STATUS_LOOKUP


# ── Constants ──────────────────────────────────────────────────────────────────

CHANNEL_DICT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}

# Indexed by range_idx (0–11); values in mV
CHANNEL_INPUT_RANGES_MV = np.array([10, 20, 50, 100, 200, 500,
                                     1000, 2000, 5000, 10000, 20000, 50000])

RANGE_LABELS = ['10 mV', '20 mV', '50 mV', '100 mV', '200 mV', '500 mV',
                '1 V',   '2 V',   '5 V',   '10 V',   '20 V',   '50 V']

TIME_UNITS_SEC = {
    'PS4000A_NS': 1e-9,
    'PS4000A_US': 1e-6,
    'PS4000A_MS': 1e-3,
}

PICO_SERIALS = {
    'Cloud  (JO279/0118)': b'JO279/0118',
    'Local  (JY140/0294)': b'JY140/0294',
}


# ── Config dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ChannelConfig:
    name: str                   # 'A' through 'H'
    range_idx: int              # index into CHANNEL_INPUT_RANGES_MV (0–11)
    coupling: str = 'DC'        # 'DC' or 'AC'
    analog_offset: float = 0.0
    save_mean_only: bool = False # if True, write channel mean (mV) instead of full waveform


@dataclass
class AcquisitionConfig:
    channels: List[ChannelConfig]
    sample_interval: int        # numeric value, interpreted in sample_units
    sample_units: str           # 'PS4000A_NS', 'PS4000A_US', or 'PS4000A_MS'
    buffer_size: int            # samples per streaming buffer
    n_buffer: int = 1           # buffers per file (>1 not tested, kept for compatibility)

    @property
    def dt(self) -> float:
        """Nominal sample period in seconds."""
        return self.sample_interval * TIME_UNITS_SEC[self.sample_units]

    @property
    def fs(self) -> float:
        """Nominal sample rate in Hz."""
        return 1.0 / self.dt

    @property
    def duration(self) -> float:
        """Duration of one file capture in seconds."""
        return self.buffer_size * self.n_buffer * self.dt


# ── Main class ─────────────────────────────────────────────────────────────────

class PicoscopeDAQ:
    """
    Manages a single PicoScope 4000a session.

    Typical use (headless):

        config = AcquisitionConfig(
            channels=[ChannelConfig('C', range_idx=9), ChannelConfig('D', range_idx=6)],
            sample_interval=200, sample_units='PS4000A_NS', buffer_size=2**25,
        )
        with PicoscopeDAQ(config) as daq:
            daq.connect(b'JO279/0118')
            for i in range(n_files):
                timestamp, dt, adc2mvs, data = daq.acquire_one()
                daq.write_hdf5(f'run_{i}.hdf5', timestamp, dt, adc2mvs, data)

    The GUI (daq_gui.py) drives this class from a QThread.
    """

    def __init__(self, config: AcquisitionConfig):
        self.config = config
        self._chandle = ctypes.c_int16()
        self._status: dict = {}
        self._one_buffer: Optional[np.ndarray] = None
        self._connected = False

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self, serial: bytes) -> None:
        """Open the scope and configure all channels."""
        serial_buf = ctypes.create_string_buffer(serial)
        status = ps.ps4000aOpenUnit(ctypes.byref(self._chandle), serial_buf)
        self._status['openunit'] = status

        # Power-source fixups: 282 (USB3 on USB2 port) and 286 (no aux power)
        # are both resolvable by ChangePowerSource.
        if status in (282, 286):
            self._status['changePowerSource'] = ps.ps4000aChangePowerSource(
                self._chandle, status)
            assert_pico_ok(self._status['changePowerSource'])
        elif status != 0:
            name = PICO_STATUS_LOOKUP.get(status, f'status={status}')
            raise RuntimeError(
                f'ps4000aOpenUnit failed: {name} ({status}). '
                f'Serial={serial!r}. '
                f'If this is PICO_HARDWARE_VERSION_NOT_SUPPORTED, update '
                f'PicoScope drivers on the PC or power-cycle the scope USB.')

        self._configure_channels()
        self._allocate_buffers()
        self._connected = True

    def disconnect(self) -> None:
        """Stop streaming and close the scope."""
        if not self._connected:
            return
        self._status['stop'] = ps.ps4000aStop(self._chandle)
        assert_pico_ok(self._status['stop'])
        self._status['close'] = ps.ps4000aCloseUnit(self._chandle)
        assert_pico_ok(self._status['close'])
        self._connected = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ── Setup helpers ───────────────────────────────────────────────────────────

    def _configure_channels(self) -> None:
        for ch in self.config.channels:
            coupling = 'PS4000A_AC' if ch.coupling == 'AC' else 'PS4000A_DC'
            key = f'setCh{ch.name}'
            self._status[key] = ps.ps4000aSetChannel(
                self._chandle,
                CHANNEL_DICT[ch.name],
                1,  # enabled
                ps.PS4000A_COUPLING[coupling],
                ch.range_idx,
                ch.analog_offset,
            )
            assert_pico_ok(self._status[key])

    def _allocate_buffers(self) -> None:
        n_ch = len(self.config.channels)
        self._one_buffer = np.zeros((n_ch, self.config.buffer_size), dtype=np.int16)
        for i, ch in enumerate(self.config.channels):
            key = f'setDataBuffers{ch.name}'
            self._status[key] = ps.ps4000aSetDataBuffers(
                self._chandle,
                CHANNEL_DICT[ch.name],
                self._one_buffer[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                self.config.buffer_size,
                0,   # memory_segment
                ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'],
            )
            assert_pico_ok(self._status[key])

    # ── Acquisition ─────────────────────────────────────────────────────────────

    def acquire_one(self, on_chunk=None, chunk_period: float = 0.5,
                    window_duration: float = 2.0):
        """
        Stream one capture.

        Parameters
        ----------
        on_chunk : callable, optional
            Called periodically during acquisition with a rolling window of
            the most recent data.  Signature::

                on_chunk(window: np.ndarray, adc2mvs: np.ndarray, dt: float)

            ``window`` has shape (n_channels, window_samples), dtype int16.
            ``adc2mvs`` has shape (n_channels,) — multiply to convert to mV.
            ``dt`` is the nominal sample period in seconds.
            The callable is invoked from the worker thread; emit Qt signals
            from it directly (PyQt queued connections are thread-safe).
        chunk_period : float
            How often to call on_chunk, in seconds of acquired data (default 0.5).
        window_duration : float
            How many seconds of history to include in each on_chunk call (default 2.0).

        Returns
        -------
        timestamp : float     Unix time at the start of acquisition.
        dt        : float     Actual sample period in seconds (may differ from nominal).
        adc2mvs   : ndarray   Shape (n_channels,). Multiply int16 counts by this to get mV.
        data      : ndarray   Shape (n_channels, total_samples), dtype int16.
        """
        cfg = self.config
        n_ch = len(cfg.channels)
        total_samples = cfg.buffer_size * cfg.n_buffer

        # ps4000aMaximumValue is a device constant — safe to query before streaming.
        # Doing it here lets on_chunk use adc2mvs without waiting for acquisition to end.
        max_adc = ctypes.c_int16()
        self._status['maximumValue'] = ps.ps4000aMaximumValue(
            self._chandle, ctypes.byref(max_adc))
        assert_pico_ok(self._status['maximumValue'])
        ranges  = np.array([ch.range_idx for ch in cfg.channels])
        adc2mvs = CHANNEL_INPUT_RANGES_MV[ranges] / max_adc.value

        nominal_dt     = cfg.sample_interval * TIME_UNITS_SEC[cfg.sample_units]
        chunk_every    = max(1, int(chunk_period    / nominal_dt))
        window_samples = max(1, int(window_duration / nominal_dt))

        # ── Closure-based callback ────────────────────────────────────────────
        # All mutable streaming state lives in this dict, captured by the closure.
        # This is the fix for the issue in daq_dev_notes.txt: defining the callback
        # as a closure here means the C function pointer is bound to *this call's*
        # state, not to module globals that could conflict across imports or calls.
        state = {
            'total_buffer':   np.zeros((n_ch, total_samples), dtype=np.int16),
            'next_sample':    0,
            'auto_stop':      False,
            'was_called_back': False,
            'last_chunk':     0,
        }
        one_buffer = self._one_buffer  # local ref so the closure doesn't hold self

        def _streaming_callback(handle, noOfSamples, startIndex, overflow,
                                 triggerAt, triggered, autoStop, param):
            state['was_called_back'] = True
            dst = state['next_sample']
            state['total_buffer'][:, dst:dst + noOfSamples] = (
                one_buffer[:, startIndex:startIndex + noOfSamples])
            state['next_sample'] += noOfSamples
            if autoStop:
                state['auto_stop'] = True

            # ── Semi-realtime preview hook ────────────────────────────────────
            if on_chunk is not None:
                n = state['next_sample']
                if n - state['last_chunk'] >= chunk_every:
                    window_start = max(0, n - window_samples)
                    on_chunk(
                        state['total_buffer'][:, window_start:n].copy(),
                        adc2mvs,
                        nominal_dt,
                    )
                    state['last_chunk'] = n
            # ─────────────────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────

        c_interval = ctypes.c_int32(cfg.sample_interval)
        timestamp = time.time()

        self._status['runStreaming'] = ps.ps4000aRunStreaming(
            self._chandle,
            ctypes.byref(c_interval),
            ps.PS4000A_TIME_UNITS[cfg.sample_units],
            0,              # maxPreTriggerSamples
            total_samples,
            1,              # autoStopOn
            1,              # downsampleRatio
            ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'],
            cfg.buffer_size,
        )
        assert_pico_ok(self._status['runStreaming'])

        c_func_ptr = ps.StreamingReadyType(_streaming_callback)
        while state['next_sample'] < total_samples and not state['auto_stop']:
            state['was_called_back'] = False
            self._status['getStreamingLatestValues'] = ps.ps4000aGetStreamingLatestValues(
                self._chandle, c_func_ptr, None)
            if not state['was_called_back']:
                time.sleep(0.01)

        actual_dt = c_interval.value * TIME_UNITS_SEC[cfg.sample_units]
        return timestamp, actual_dt, adc2mvs, state['total_buffer']

    # ── HDF5 writer ─────────────────────────────────────────────────────────────

    def write_hdf5(self, filepath: str, timestamp: float, dt: float,
                   adc2mvs: np.ndarray, data: np.ndarray,
                   pressure_mbar: Optional[float] = None) -> None:
        """
        Write one capture to HDF5 using the standard lab format.

        Layout (mirrors existing data files):
            /data                    group
            /data.attrs['timestamp'] Unix time
            /data.attrs['delta_t']   sample period in seconds
            /data.attrs['pressure_mbar']  (optional)
            /data/channel_c          int16 dataset  (if save_mean_only is False)
            /data/channel_c.attrs['adc2mv']
            /data.attrs['channel_c_mean_mv']  (if save_mean_only is True)
        """
        with h5py.File(filepath, 'w') as f:
            g = f.create_group('data')
            g.attrs['timestamp'] = timestamp
            g.attrs['delta_t'] = dt
            if pressure_mbar is not None:
                g.attrs['pressure_mbar'] = pressure_mbar
            for i, ch in enumerate(self.config.channels):
                if ch.save_mean_only:
                    g.attrs[f'channel_{ch.name.lower()}_mean_mv'] = (
                        float(np.mean(data[i])) * adc2mvs[i])
                else:
                    ds = g.create_dataset(
                        f'channel_{ch.name.lower()}', data=data[i], dtype=np.int16)
                    ds.attrs['adc2mv'] = float(adc2mvs[i])
