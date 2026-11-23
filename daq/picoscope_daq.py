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
    slow: bool = False           # inherently slow input: kept as a slow series (slow_data.py)


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

    For a free-running look at the signal without writing files, use
    monitor_stream() instead of acquire_one() — same scope session, same channel
    setup, and it leaves the driver ready for acquire_one() when it returns.

    The GUI (daq_gui.py) drives this class from a QThread.
    """

    def __init__(self, config: AcquisitionConfig):
        self.config = config
        self._chandle = ctypes.c_int16()
        self._status: dict = {}
        self._one_buffer: Optional[np.ndarray] = None
        self._connected = False
        # Diagnostics.  ps4000aGetStreamingLatestValues is the only driver call
        # whose return code used to be stored and never looked at, and it is the
        # one that reports the scope having run out of room to put samples — so
        # data loss during a long monitor was silent by construction.
        self.connect_status: str = ''
        self.stream_events: dict = {}
        self.last_capture_events: dict = {}

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def stream_health(self) -> str:
        """Everything the driver complained about while streaming, most frequent first."""
        if not self.stream_events:
            return ''
        return ' · '.join(f'{name} ×{n}' for name, n in
                          sorted(self.stream_events.items(), key=lambda kv: -kv[1]))

    def reset_stream_events(self) -> None:
        self.stream_events.clear()

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self, serial: bytes) -> None:
        """Open the scope and configure all channels."""
        serial_buf = ctypes.create_string_buffer(serial)
        status = ps.ps4000aOpenUnit(ctypes.byref(self._chandle), serial_buf)
        self._status['openunit'] = status

        # Power-source fixups.  ChangePowerSource lets the unit open either way,
        # but neither condition is harmless and the caller should know which one
        # it got:
        #   282 PICO_POWER_SUPPLY_NOT_CONNECTED — running on USB power alone
        #   286 PICO_USB3_0_DEVICE_NON_USB3_0_PORT — USB 3.0 scope on a USB 2.0
        #       port, which cuts the sustainable streaming rate several-fold and
        #       will overrun on a fast multi-channel monitor.
        self.connect_status = PICO_STATUS_LOOKUP.get(status, '') if status else ''
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
            self._set_channel(ch.name, True, ch.range_idx, coupling, ch.analog_offset)

    def _set_channel(self, name: str, enabled: bool, range_idx: int = 7,
                     coupling: str = 'PS4000A_DC', analog_offset: float = 0.0) -> None:
        key = f'setCh{name}'
        self._status[key] = ps.ps4000aSetChannel(
            self._chandle,
            CHANNEL_DICT[name],
            1 if enabled else 0,
            ps.PS4000A_COUPLING[coupling],
            range_idx,
            analog_offset,
        )
        assert_pico_ok(self._status[key])

    def _allocate_buffers(self) -> None:
        """Allocate and register the acquisition buffer (one file's worth of streaming)."""
        n_ch = len(self.config.channels)
        self._one_buffer = np.zeros((n_ch, self.config.buffer_size), dtype=np.int16)
        self._register_buffers(self._one_buffer)

    def _register_buffers(self, buffer: np.ndarray) -> None:
        """Point the driver at ``buffer`` (shape (n_channels, n_samples), int16)."""
        for i, ch in enumerate(self.config.channels):
            key = f'setDataBuffers{ch.name}'
            self._status[key] = ps.ps4000aSetDataBuffers(
                self._chandle,
                CHANNEL_DICT[ch.name],
                buffer[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                buffer.shape[1],
                0,   # memory_segment
                ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'],
            )
            assert_pico_ok(self._status[key])

    def _safe_stop(self) -> None:
        """Stop streaming, ignoring errors (the scope may already be idle)."""
        try:
            ps.ps4000aStop(self._chandle)
        except Exception:
            pass

    def reconfigure(self, config: AcquisitionConfig) -> None:
        """
        Apply a new configuration to an already-open scope.

        Channels that were enabled but are absent from the new config are
        explicitly disabled, then the remaining channels are re-set and the
        acquisition buffers re-allocated.  Any streaming in progress is stopped
        first, so callers must not be inside acquire_one()/monitor_stream().
        """
        if not self._connected:
            self.config = config
            return
        self._safe_stop()
        previous = {ch.name for ch in self.config.channels}
        self.config = config
        current = {ch.name for ch in config.channels}
        for name in sorted(previous - current):
            self._set_channel(name, False)
        self._configure_channels()
        self._allocate_buffers()

    def _adc2mv_factors(self) -> np.ndarray:
        """Per-channel mV-per-count scale factors (device constant; safe when idle)."""
        max_adc = ctypes.c_int16()
        self._status['maximumValue'] = ps.ps4000aMaximumValue(
            self._chandle, ctypes.byref(max_adc))
        assert_pico_ok(self._status['maximumValue'])
        ranges = np.array([ch.range_idx for ch in self.config.channels])
        return CHANNEL_INPUT_RANGES_MV[ranges] / max_adc.value

    # ── Acquisition ─────────────────────────────────────────────────────────────

    def acquire_one(self, on_chunk=None, chunk_period: float = 0.5,
                    window_duration: float = 2.0, on_samples=None):
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
        on_samples : callable, optional
            ``on_samples(block, stable)`` — called from the streaming callback
            with every new run of samples, in order and without gaps: an int16
            view of shape (n_channels, n).  ``stable`` is True here: the view is
            into this capture's own buffer and stays valid.  It must return
            immediately (enqueue, do not compute) — this is the tap the slow-data
            reduction hangs off (see slow_data.SlowPipeline.push).

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

        # Re-assert ownership of the driver buffers: monitor_stream() temporarily
        # registers its own smaller buffer, and this is cheap and idempotent.
        if self._one_buffer is None or self._one_buffer.shape != (n_ch, cfg.buffer_size):
            self._allocate_buffers()
        else:
            self._register_buffers(self._one_buffer)

        # ps4000aMaximumValue is a device constant — safe to query before streaming.
        # Doing it here lets on_chunk use adc2mvs without waiting for acquisition to end.
        adc2mvs = self._adc2mv_factors()

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
            if on_samples is not None and noOfSamples > 0:
                on_samples(state['total_buffer'][:, dst:dst + noOfSamples], True)

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

        capture_events: dict = {}
        c_func_ptr = ps.StreamingReadyType(_streaming_callback)
        while state['next_sample'] < total_samples and not state['auto_stop']:
            state['was_called_back'] = False
            self._status['getStreamingLatestValues'] = st = (
                ps.ps4000aGetStreamingLatestValues(self._chandle, c_func_ptr, None))
            if st != 0:
                # Recorded files are exposed to exactly the same data loss as the
                # live monitor, and a file with a hole in it looks entirely normal
                # on disk — so the count travels with the data (see write_hdf5).
                name = PICO_STATUS_LOOKUP.get(st, f'status={st}')
                capture_events[name] = capture_events.get(name, 0) + 1
                self.stream_events[name] = self.stream_events.get(name, 0) + 1
            if not state['was_called_back']:
                time.sleep(0.01)
        self.last_capture_events = capture_events

        actual_dt = c_interval.value * TIME_UNITS_SEC[cfg.sample_units]
        return timestamp, actual_dt, adc2mvs, state['total_buffer']

    # ── Free-running monitor (no data saved) ───────────────────────────────────

    def monitor_stream(self, on_window, ring_duration: float = 1.0,
                       update_period: float = 0.2, chunk_duration: float = 0.05,
                       should_stop=None, max_ring_samples: int = 2 ** 24,
                       stall_timeout: float = 5.0, on_samples=None,
                       burst: bool = False, on_burst=None) -> None:
        """
        Stream continuously without autostop and hand rolling windows to a callback.

        Nothing is written to disk and no full-length buffer is kept: the driver
        writes into a small chunk buffer (``chunk_duration`` seconds) and the
        callback copies each chunk into a circular ring holding the most recent
        ``ring_duration`` seconds.  The polling loop — not the callback — invokes
        ``on_window`` every ``update_period`` seconds, so slow consumers (PSD,
        plotting) can never stall the driver callback.

        Parameters
        ----------
        on_window : callable
            ``on_window(get_window, adc2mvs, dt, overflow)``.

            ``get_window(duration=None)`` returns a fresh int16 array of shape
            (n_channels, n) ordered oldest → newest holding the most recent
            ``duration`` seconds (everything retained, if omitted).  It is a
            getter rather than a ready-made array so the consumer can pull a
            short slice at a high rate and a long one only occasionally —
            copying a multi-second ring 30 times a second would be pure waste.
            Valid only for the duration of the call.

            ``dt`` is the nominal sample period and ``overflow`` the OR of the
            driver's over-range bitmask since the previous call.
        ring_duration : float
            Seconds of history retained (clamped to ``max_ring_samples``).
        update_period : float
            Wall-clock seconds between ``on_window`` calls.
        chunk_duration : float
            Driver callback granularity in seconds.  Smaller = lower latency,
            more callback overhead.
        should_stop : callable, optional
            Polled every loop iteration; monitoring returns as soon as it is true.
            This makes stopping responsive (unlike acquire_one, which blocks for a
            whole file), which is what lets the GUI switch modes promptly.
        max_ring_samples : int
            Per-channel cap on the ring, so a long window at a high sample rate
            cannot exhaust memory.
        on_samples : callable, optional
            ``on_samples(block, stable)`` — as in acquire_one(), every sample in
            order, but ``stable`` is False: the view is into the driver's chunk
            buffer and is overwritten by the next callback, so the receiver must
            copy it before returning.

        burst : bool
            Instead of one endless stream, take a bounded capture of
            ``ring_duration`` every ``update_period`` and stop in between.

            An endless stream has to get every sample across USB as fast as it
            is taken.  Several channels at MS/s exceed what a USB 2.0 link
            carries; the scope's memory hides the deficit for some tens of
            seconds and then the driver returns torn data (a comb in the PSD,
            a thickened trace).  A bounded capture is buffered in the scope and
            drained afterwards — the way acquire_one() works — so each burst is
            a clean, full-rate window at any channel count, at the price of
            dead time between them.
        on_burst : callable, optional
            ``on_burst('start', t_wall)`` before each burst's first sample and
            ``on_burst('end', t_wall)`` after its last, so a consumer of
            ``on_samples`` knows where the stream is not contiguous.

        Returns when ``should_stop()`` becomes true, or early if the driver stops
        delivering callbacks for ``stall_timeout`` seconds (the caller is expected
        to simply call again — MonitorWorker's outer loop does).  Either way the
        scope is left stopped and the acquisition buffers re-registered, ready for
        acquire_one().
        """
        cfg  = self.config
        n_ch = len(cfg.channels)
        if n_ch == 0:
            raise RuntimeError('No channels configured.')

        nominal_dt     = cfg.dt
        chunk_samples  = int(np.clip(chunk_duration / nominal_dt, 1024, max_ring_samples))
        window_samples = int(np.clip(ring_duration  / nominal_dt, chunk_samples,
                                     max_ring_samples))

        adc2mvs = self._adc2mv_factors()

        # Same closure pattern as acquire_one: all mutable streaming state is
        # captured here rather than in module globals (see daq_dev_notes.txt).
        mon_buffer = np.zeros((n_ch, chunk_samples), dtype=np.int16)
        ring       = np.zeros((n_ch, window_samples), dtype=np.int16)
        state = {
            'ptr':             0,      # next write position in the ring
            'filled':          0,      # valid samples in the ring
            'overflow':        0,
            'was_called_back': False,
            'received':        0,      # samples in the current burst
            'auto_stop':       False,
        }

        def _monitor_callback(handle, noOfSamples, startIndex, overflow,
                              triggerAt, triggered, autoStop, param):
            state['was_called_back'] = True
            if autoStop:
                state['auto_stop'] = True
            if noOfSamples <= 0:
                return
            state['overflow'] |= overflow
            state['received'] += noOfSamples
            src = mon_buffer[:, startIndex:startIndex + noOfSamples]
            n   = src.shape[1]
            if on_samples is not None:
                on_samples(src, False)
            if n >= window_samples:
                ring[:, :] = src[:, -window_samples:]
                state['ptr']    = 0
                state['filled'] = window_samples
                return
            p   = state['ptr']
            end = p + n
            if end <= window_samples:
                ring[:, p:end] = src
            else:
                split = window_samples - p
                ring[:, p:]           = src[:, :split]
                ring[:, :n - split]   = src[:, split:]
            state['ptr']    = end % window_samples
            state['filled'] = min(window_samples, state['filled'] + n)

        def _snapshot(duration: Optional[float] = None) -> np.ndarray:
            """
            Copy out the most recent ``duration`` seconds, oldest → newest.

            Only the requested tail is copied.  When the ring has not wrapped
            past it the tail is contiguous and this is a single memcpy;
            otherwise the two pieces are stitched in order.
            """
            filled = state['filled']
            n = filled if duration is None else min(
                filled, max(1, int(duration / nominal_dt)))
            end = state['ptr'] if filled >= window_samples else filled
            start = end - n
            if start >= 0:
                return ring[:, start:end].copy()
            out = np.empty((n_ch, n), dtype=np.int16)
            head = -start                       # samples living at the ring's tail
            out[:, :head] = ring[:, window_samples + start:]
            out[:, head:] = ring[:, :end]
            return out

        self._register_buffers(mon_buffer)
        c_interval = ctypes.c_int32(cfg.sample_interval)
        c_func_ptr = ps.StreamingReadyType(_monitor_callback)

        def _run_streaming(auto_stop: bool) -> float:
            # autoStop=0 → the driver recycles the chunk buffer indefinitely;
            # autoStop=1 → it stops by itself after window_samples (one burst).
            self._status['runStreamingMonitor'] = ps.ps4000aRunStreaming(
                self._chandle,
                ctypes.byref(c_interval),
                ps.PS4000A_TIME_UNITS[cfg.sample_units],
                0,               # maxPreTriggerSamples
                window_samples if auto_stop else chunk_samples,
                1 if auto_stop else 0,
                1,               # downsampleRatio
                ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'],
                chunk_samples,
            )
            assert_pico_ok(self._status['runStreamingMonitor'])
            return c_interval.value * TIME_UNITS_SEC[cfg.sample_units]

        def _poll() -> None:
            state['was_called_back'] = False
            self._status['getStreamingLatestValues'] = st = (
                ps.ps4000aGetStreamingLatestValues(self._chandle, c_func_ptr, None))
            if st != 0:
                # PICO_BUFFER_STALL (28) here means the driver had nowhere to
                # put samples and threw some away; whatever arrives next is
                # discontinuous.  Counting these by name is what turns "the
                # trace looks wrong" into a number.
                name = PICO_STATUS_LOOKUP.get(st, f'status={st}')
                self.stream_events[name] = self.stream_events.get(name, 0) + 1
            if not state['was_called_back']:
                time.sleep(0.005)

        def _stopping() -> bool:
            return bool(should_stop and should_stop())

        min_samples = max(chunk_samples, int(0.01 / nominal_dt))
        try:
            if burst:
                while not _stopping():
                    state.update(ptr=0, filled=0, received=0, auto_stop=False)
                    burst_t0 = time.perf_counter()
                    if on_burst is not None:
                        on_burst('start', time.time())
                    actual_dt = _run_streaming(True)
                    last_data = time.perf_counter()
                    while (state['received'] < window_samples
                           and not state['auto_stop'] and not _stopping()):
                        _poll()
                        now = time.perf_counter()
                        if state['was_called_back']:
                            last_data = now
                        elif now - last_data > stall_timeout:
                            return
                    self._safe_stop()
                    if on_burst is not None:
                        on_burst('end', time.time())
                    if _stopping():
                        break
                    if state['filled'] > 0:
                        overflow = state['overflow']
                        state['overflow'] = 0
                        on_window(_snapshot, adc2mvs, actual_dt, overflow)
                    # Idle out the rest of the period in small steps, so a stop
                    # request is still answered within milliseconds.
                    while (time.perf_counter() - burst_t0 < update_period
                           and not _stopping()):
                        time.sleep(0.005)
                return

            actual_dt = _run_streaming(False)
            last_emit = 0.0
            last_data = time.perf_counter()
            while not _stopping():
                _poll()
                now = time.perf_counter()
                if state['was_called_back']:
                    last_data = now
                elif now - last_data > stall_timeout:
                    # Driver went quiet (stopped stream, USB hiccup) — bail out and
                    # let the caller start a fresh streaming run.
                    return
                if state['filled'] >= min_samples and now - last_emit >= update_period:
                    overflow = state['overflow']
                    state['overflow'] = 0
                    on_window(_snapshot, adc2mvs, actual_dt, overflow)
                    last_emit = time.perf_counter()
        finally:
            self._safe_stop()
            # Hand the driver back its acquisition-sized buffers so a mode switch
            # into acquire_one() needs no extra setup.
            if self._one_buffer is not None:
                self._register_buffers(self._one_buffer)

    # ── HDF5 writer ─────────────────────────────────────────────────────────────

    def write_hdf5(self, filepath: str, timestamp: float, dt: float,
                   adc2mvs: np.ndarray, data: np.ndarray,
                   pressure_mbar: Optional[float] = None,
                   extra_attrs: Optional[dict] = None,
                   slow: Optional[dict] = None) -> None:
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

        ``extra_attrs`` adds further attributes to /data — used to record the
        calibration-pulse settings in force for the file, so the amplitude does
        not have to be recovered from the filename.

        ``slow`` is the result of SlowPipeline.end() for this capture: the slow
        quantities reduced from these very samples, written as the group /slow
        (layout in slow_data.py) so they travel with the waveform.
        """
        with h5py.File(filepath, 'w') as f:
            g = f.create_group('data')
            g.attrs['timestamp'] = timestamp
            g.attrs['delta_t'] = dt
            if pressure_mbar is not None:
                g.attrs['pressure_mbar'] = pressure_mbar
            # Anything the driver reported during the capture this call follows.
            # A stream the USB link could not keep up with yields a file that is
            # the right length and looks perfectly normal, with samples missing
            # in the middle — so the warning has to be recorded at write time or
            # it cannot be recovered afterwards.
            if self.last_capture_events:
                g.attrs['driver_events'] = ', '.join(
                    f'{name}x{n}' for name, n in sorted(self.last_capture_events.items()))
            for key, value in (extra_attrs or {}).items():
                g.attrs[key] = value
            for i, ch in enumerate(self.config.channels):
                if ch.save_mean_only:
                    g.attrs[f'channel_{ch.name.lower()}_mean_mv'] = (
                        float(np.mean(data[i])) * adc2mvs[i])
                else:
                    ds = g.create_dataset(
                        f'channel_{ch.name.lower()}', data=data[i], dtype=np.int16)
                    ds.attrs['adc2mv'] = float(adc2mvs[i])
            if slow is not None:
                try:
                    from .slow_data import write_slow_group
                except ImportError:         # picoscope_daq imported as a script
                    from slow_data import write_slow_group
                write_slow_group(f, slow)
