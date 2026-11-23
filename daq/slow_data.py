"""
Slow data — slow signals taken on the same scope, at the same time, as the fast
ones.

The PicoScope samples every enabled channel at one rate, so an input that is
inherently slow — the output of a lock-in amplifier with a 100 ms time constant,
a temperature or humidity sensor's voltage — arrives sampled at MS/s like the
sphere's motion next to it.  Storing that as a waveform is pure waste; storing
it as one mean per file (the old "Mean" save mode) throws the signal away.
Here the stream is averaged, block by block, to one row per 1/rate seconds:

  mean_<ch>, rms_<ch>      per slow channel (or every channel), in mV; the
                           average over the block, which also buys resolution
                           well below one ADC count
  <name>_x, <name>_y       lock-in output: the component of the signal channel
                           in phase / in quadrature with the reference channel
                           (peak amplitude, mV x scale)
  <name>_r, <name>_phase   its magnitude and phase (degrees)
  <name>_ref               reference amplitude (peak, mV)
  <name>_freq              demodulation frequency in use (Hz)

One block is 1/rate seconds, so every sample contributes to exactly one row —
the rows are an average over the whole block, not a snapshot at the tick.  (The
lock-in columns are a software demodulator for when there is no hardware one.)

Nothing here knows about Qt or the scope:

  SlowReducer    the arithmetic; feed() it consecutive int16 blocks
  SlowPipeline   a thread and a queue around a reducer, so the driver callback
                 only ever enqueues
  SlowHistory    bounded store behind the strip chart
  SlowLogWriter  appendable HDF5 log
  write_slow_group / read_slow   the on-disk layout, shared by the continuous
                 log and by the /slow group embedded in each fast file

On-disk layout (group ``slow``)::

    slow/time            unix seconds, centre of each block
    slow/<column>        float64, attrs['unit']
    slow.attrs: rate_hz, block_samples, delta_t, config (JSON)

    >>> from daq.slow_data import read_slow
    >>> s = read_slow(r'E:\\...\\20261123_slow_....hdf5')
    >>> plt.plot(s['time'] - s['time'][0], s['charge_x'])
"""

import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class LockinConfig:
    """
    One demodulator.

    ``reference`` names the channel carrying the drive (function-generator
    monitor or sync).  With one, x/y are measured against the drive's own phase
    and the demodulation frequency follows the drive, so the clocks of the scope
    and the generator need not agree.  Without one, ``freq`` is used as typed:
    r is still meaningful if ``freq`` is right to much better than the slow
    rate, x/y/phase are relative to the PC and drift.

    ``phase_deg`` rotates the output so the response can be put wholly into x
    (the sign of x is then the sign of the charge).  ``scale`` and ``unit``
    turn mV into something calibrated, e.g. elementary charges.
    """
    name: str
    channel: str
    freq: float
    reference: str = ''
    phase_deg: float = 0.0
    scale: float = 1.0
    unit: str = 'mV'


@dataclass
class SlowConfig:
    enabled: bool = False
    rate_hz: float = 10.0
    stats: bool = False                     # mean and rms of every channel...
    # ...otherwise only of these: the channels whose input is itself slow (a
    # lock-in amplifier's output, a temperature sensor) and which are therefore
    # kept as a slow series instead of a waveform.
    slow_channels: List[str] = field(default_factory=list)
    lockins: List[LockinConfig] = field(default_factory=list)

    def stat_channels(self, channel_names: Sequence[str]) -> List[str]:
        if self.stats:
            return list(channel_names)
        return [ch for ch in channel_names if ch in self.slow_channels]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'SlowConfig':
        lockins = []
        for item in d.get('lockins', []):
            try:
                lockins.append(LockinConfig(**item))
            except TypeError:
                continue
        return cls(enabled=bool(d.get('enabled', False)),
                   rate_hz=float(d.get('rate_hz', 10.0)),
                   stats=bool(d.get('stats', False)),
                   slow_channels=list(d.get('slow_channels', [])),
                   lockins=lockins)


def safe_name(name: str) -> str:
    """A lock-in name as it appears in column / dataset names."""
    return re.sub(r'[^0-9A-Za-z_]+', '_', name.strip()) or 'lockin'


def slow_columns(config: SlowConfig, channel_names: Sequence[str]
                 ) -> Tuple[List[str], Dict[str, str]]:
    """
    Column names (in row order) and their units for a configuration.

    Lock-ins whose signal or reference channel is not enabled are left out — the
    same rule the reducer applies, so the two always agree.
    """
    columns: List[str] = []
    units: Dict[str, str] = {}
    for ch in config.stat_channels(channel_names):
        for kind in ('mean', 'rms'):
            col = f'{kind}_{ch.lower()}'
            columns.append(col)
            units[col] = 'mV'
    for lk in usable_lockins(config, channel_names):
        base = safe_name(lk.name)
        for suffix, unit in (('x', lk.unit), ('y', lk.unit), ('r', lk.unit),
                             ('phase', 'deg'), ('ref', 'mV'), ('freq', 'Hz')):
            col = f'{base}_{suffix}'
            columns.append(col)
            units[col] = unit
    return columns, units


def usable_lockins(config: SlowConfig, channel_names: Sequence[str]
                   ) -> List[LockinConfig]:
    out, seen = [], set()
    for lk in config.lockins:
        base = safe_name(lk.name)
        if base in seen or lk.channel not in channel_names or lk.freq <= 0:
            continue
        if lk.reference and lk.reference not in channel_names:
            continue
        seen.add(base)
        out.append(lk)
    return out


# ── Reduction ──────────────────────────────────────────────────────────────────

class _Lockin:
    """Accumulators for one demodulator over one slow block."""

    def __init__(self, cfg: LockinConfig, sig_idx: int, ref_idx: Optional[int],
                 dt: float, sub: int):
        self.cfg     = cfg
        self.sig_idx = sig_idx
        self.ref_idx = ref_idx
        self.dt      = dt
        self.sub     = sub
        self.freq    = float(cfg.freq)
        self.phase   = 0.0          # LO phase (cycles) at the next sample
        self._make_tables()
        self.reset_block()

    def _make_tables(self) -> None:
        arg = 2 * np.pi * self.freq * self.dt * np.arange(self.sub)
        self._cos = np.cos(arg)
        self._sin = np.sin(arg)

    def reset_block(self) -> None:
        self.z_sig = 0j
        self.z_ref = 0j
        self._sub_sig = 0j          # the sub-block being filled
        self._sub_ref = 0j
        self._prev_ref: Optional[complex] = None
        self._adv = 0j              # sum of ref[j] * conj(ref[j-1]) over sub-blocks

    def _demod(self, x: np.ndarray) -> complex:
        m = len(x)
        return complex(x @ self._cos[:m], -(x @ self._sin[:m]))

    def feed(self, floats: Dict[int, np.ndarray], m: int) -> None:
        """Add ``m`` samples that lie within one sub-block."""
        rot = np.exp(-2j * np.pi * self.phase)
        self._sub_sig += self._demod(floats[self.sig_idx]) * rot
        if self.ref_idx is not None:
            self._sub_ref += self._demod(floats[self.ref_idx]) * rot
        self.phase = (self.phase + self.freq * self.dt * m) % 1.0

    def end_sub(self, full: bool) -> None:
        """
        Close the current sub-block.

        The phase the reference advances from one full sub-block to the next is
        the frequency error of the LO, measured many times per slow block — that
        is what lets the LO be pulled onto the drive (see finish()).
        """
        self.z_sig += self._sub_sig
        self.z_ref += self._sub_ref
        if self.ref_idx is not None and full:
            if self._prev_ref is not None:
                self._adv += self._sub_ref * np.conj(self._prev_ref)
            self._prev_ref = self._sub_ref
        self._sub_sig = 0j
        self._sub_ref = 0j

    def skip(self, n: int) -> None:
        """Samples lost upstream: keep the LO phase continuous across the hole."""
        self.phase = (self.phase + self.freq * self.dt * n) % 1.0

    def finish(self, n: int, adc2mvs: np.ndarray) -> List[float]:
        """Row values for a completed block of ``n`` samples; retunes the LO."""
        cfg  = self.cfg
        a_sig = 2.0 * self.z_sig / n * adc2mvs[self.sig_idx]     # peak, mV
        used_freq = self.freq
        nan = complex(np.nan, np.nan)
        if self.ref_idx is not None:
            a_ref = 2.0 * self.z_ref / n * adc2mvs[self.ref_idx]
            ref_mv = abs(a_ref)
            rel = a_sig * np.conj(a_ref) / ref_mv if ref_mv > 0 else nan
            slip = abs(self._retune(n)) * n * self.dt       # cycles over the block
            if slip > 0.05:
                # The LO was still being pulled onto the drive during this block
                # (the first one after a start, or a retuned generator), so both
                # amplitudes are low by an unknown factor.  A gap is honest; a
                # dip that looks like a lost charge is not.
                rel, ref_mv = nan, np.nan
        else:
            ref_mv = np.nan
            rel = a_sig
        rel = rel * np.exp(-1j * np.deg2rad(cfg.phase_deg)) * cfg.scale
        self.reset_block()
        return [rel.real, rel.imag, abs(rel),
                float(np.degrees(np.angle(rel))) if np.isfinite(rel.real) else np.nan,
                ref_mv, used_freq]

    def _retune(self, n: int) -> float:
        """Pull the LO onto the reference; returns the error found, in Hz."""
        # A reference too small to be a drive (a few counts) carries no phase.
        if abs(self._adv) == 0 or abs(self.z_ref) * 2.0 / n < 4.0:
            return 0.0
        err_hz = float(np.angle(self._adv) / (2 * np.pi * self.sub * self.dt))
        # Leave a well-tuned LO alone: below a milli-cycle of slip per block the
        # amplitude error is parts per million, and constantly rebuilding the
        # tables on measurement noise would only add jitter.
        if abs(err_hz) * n * self.dt < 1e-3:
            return err_hz
        self.freq += err_hz
        self._make_tables()
        return err_hz


class SlowReducer:
    """
    Reduce consecutive int16 blocks, shape (n_channels, n), to slow rows.

    ``feed()`` takes blocks of any length and returns the rows completed by
    them, as (times, values) with values of shape (n_rows, n_columns); a slow
    block may span many calls or one call may complete several.
    """

    def __init__(self, config: SlowConfig, channel_names: Sequence[str],
                 adc2mvs: np.ndarray, dt: float, t0: float):
        self.config   = config
        self.channels = list(channel_names)
        self.adc2mvs  = np.asarray(adc2mvs, dtype=float)
        self.dt       = float(dt)
        self.t0       = float(t0)
        self.columns, self.units = slow_columns(config, self.channels)

        self.block = max(1, int(round(1.0 / (config.rate_hz * self.dt))))
        # Sub-blocks of a few ms: short enough that a reference tens of Hz off
        # the typed frequency is still tracked, long enough that NumPy call
        # overhead is negligible.
        self.sub = int(np.clip(round(0.005 / self.dt), 256, 65536))
        self.sub = max(1, min(self.sub, max(1, self.block // 4)))

        self._lockins: List[_Lockin] = []
        for lk in usable_lockins(config, self.channels):
            ref = self.channels.index(lk.reference) if lk.reference else None
            self._lockins.append(
                _Lockin(lk, self.channels.index(lk.channel), ref, self.dt, self.sub))
        self._stat_idx = [self.channels.index(ch)
                          for ch in config.stat_channels(self.channels)]
        needed = set(self._stat_idx)
        for lk in self._lockins:
            needed.add(lk.sig_idx)
            if lk.ref_idx is not None:
                needed.add(lk.ref_idx)
        self._needed = sorted(needed)

        n_ch = len(self.channels)
        self._sum  = np.zeros(n_ch)
        self._sum2 = np.zeros(n_ch)
        self._pos        = 0        # samples into the current slow block
        self._block_start = 0       # stream index of the current block's start
        self._index      = 0        # stream index of the next sample
        self._rows_since_gap = 0

    @property
    def active(self) -> bool:
        return bool(self.columns)

    def tracked_freqs(self) -> Dict[str, Tuple[float, float]]:
        """{lock-in name: (typed frequency, frequency the LO has settled on)}."""
        return {lk.cfg.name: (lk.cfg.freq, lk.freq) for lk in self._lockins}

    def seed_freqs(self, tracked: Dict[str, Tuple[float, float]]) -> None:
        """
        Start from where a previous stream's LOs ended up, so the first block of
        every file does not have to be spent re-acquiring the drive.
        """
        for lk in self._lockins:
            typed, found = tracked.get(lk.cfg.name, (None, None))
            if typed == lk.cfg.freq and lk.ref_idx is not None:
                lk.freq = found
                lk._make_tables()

    def skip(self, n: int, force: bool = False) -> None:
        """
        ``n`` samples were lost before the next feed().

        The block they fell in cannot be completed honestly, so it is abandoned
        and a new one starts at the next sample; timestamps stay right because
        they come from the sample index.
        """
        if n <= 0 and not force:
            return
        for lk in self._lockins:
            lk.end_sub(full=False)
            lk.skip(n)
            lk.reset_block()
        self._sum[:]  = 0
        self._sum2[:] = 0
        self._pos     = 0
        self._index  += n
        self._block_start = self._index

    def feed(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        times: List[float] = []
        rows:  List[List[float]] = []
        n_total = data.shape[1]
        i = 0
        while i < n_total:
            # Never cross a sub-block (hence never a block) boundary in one go.
            m = min(n_total - i, self.sub - self._pos % self.sub,
                    self.block - self._pos)
            floats = {c: data[c, i:i + m].astype(np.float64) for c in self._needed}
            for c in self._stat_idx:
                x = floats[c]
                self._sum[c]  += x.sum()
                self._sum2[c] += x @ x
            for lk in self._lockins:
                lk.feed(floats, m)
            i           += m
            self._pos   += m
            self._index += m
            at_block = self._pos == self.block
            if at_block or self._pos % self.sub == 0:
                full = self._pos % self.sub == 0
                for lk in self._lockins:
                    lk.end_sub(full)
            if at_block:
                times.append(self.t0 + (self._block_start + 0.5 * self.block) * self.dt)
                rows.append(self._finish_block(self.block))
                self._rows_since_gap += 1
        return self._as_arrays(times, rows)

    def _as_arrays(self, times, rows) -> Tuple[np.ndarray, np.ndarray]:
        if not rows:
            return np.empty(0), np.empty((0, len(self.columns)))
        return np.asarray(times), np.asarray(rows, dtype=float)

    def flush(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Close a stream that stops short of a block boundary (the end of a
        monitor burst): make a row out of the part-filled block, so a burst
        shorter than 1/rate still yields its one value.  A short remainder
        after full blocks is dropped instead — it would only be a noisier
        duplicate of the row just produced.
        """
        times, rows = [], []
        n = self._pos
        if n > 0 and (self._rows_since_gap == 0 or 2 * n >= self.block):
            if n % self.sub:
                for lk in self._lockins:
                    lk.end_sub(full=False)
            times.append(self.t0 + (self._block_start + 0.5 * n) * self.dt)
            rows.append(self._finish_block(n))
        self.skip(0, force=True)
        return self._as_arrays(times, rows)

    def resync(self, t_wall: float) -> None:
        """
        The next sample was taken at ``t_wall``: account for the dead time since
        the last one (between monitor bursts) so timestamps follow the clock.
        """
        gap = int(round((t_wall - self.t0) / self.dt)) - self._index
        self.skip(max(gap, 0), force=True)
        self._rows_since_gap = 0

    def _finish_block(self, n: int) -> List[float]:
        row: List[float] = []
        if self._stat_idx:
            mean = self._sum / n
            var  = np.maximum(self._sum2 / n - mean ** 2, 0.0)
            for c in self._stat_idx:
                row.append(mean[c] * self.adc2mvs[c])
                row.append(np.sqrt(var[c]) * self.adc2mvs[c])
        for lk in self._lockins:
            row.extend(lk.finish(n, self.adc2mvs))
        self._sum[:]  = 0
        self._sum2[:] = 0
        self._pos     = 0
        self._block_start = self._index
        return row


# ── Threaded pipeline ──────────────────────────────────────────────────────────

class SlowPipeline:
    """
    Runs a SlowReducer on its own thread.

    The streaming callback calls push() and returns at once; the arithmetic
    happens here.  That separation is not a nicety: anything slow inside the
    driver callback keeps ps4000aGetStreamingLatestValues from being called and
    the scope then has nowhere to put samples.

    A stream is bracketed by begin() and end().  end() waits for the queue to
    drain and returns every row of that stream, which is what gets embedded in a
    recorded file; rows are also handed to ``on_rows(columns, times, values)``
    as they are produced (on this thread) for live display and logging.

    If the reducer cannot keep up, copied blocks beyond ``max_pending_bytes``
    are dropped and the slow block they belonged to is discarded — the series
    gets a gap, never a wrong value, and ``dropped_samples`` counts it.
    """

    def __init__(self, config: SlowConfig,
                 on_rows: Optional[Callable] = None,
                 max_pending_bytes: int = 256 * 2 ** 20,
                 tracked: Optional[dict] = None):
        self.config   = config
        self._on_rows = on_rows
        self._max_pending = max_pending_bytes
        self._queue: 'queue.Queue' = queue.Queue()
        self._lock    = threading.Lock()
        self._pending_bytes = 0
        self._lost    = 0            # samples dropped since the last accepted push
        self.dropped_samples = 0
        self._reducer: Optional[SlowReducer] = None
        # Where each lock-in's LO has settled.  Passing the same dict to the
        # next pipeline (monitor -> recording) carries the lock across.
        self._tracked: Dict[str, Tuple[float, float]] = (
            tracked if tracked is not None else {})
        self._segment_times:  List[np.ndarray] = []
        self._segment_values: List[np.ndarray] = []
        self.error: Optional[str] = None
        self._thread = threading.Thread(target=self._run, name='slow-reducer',
                                        daemon=True)
        self._thread.start()

    # ── called from the acquisition thread ────────────────────────────────────

    def begin(self, t0: float, dt: float, channel_names: Sequence[str],
              adc2mvs: np.ndarray) -> None:
        reducer = SlowReducer(self.config, channel_names, adc2mvs, dt, t0)
        self._queue.put(('begin', reducer))

    def push(self, block: np.ndarray, stable: bool) -> None:
        """
        Queue samples.  ``stable`` says the array will not be overwritten (a
        slice of a capture buffer); otherwise it is copied here.
        """
        if not stable:
            nbytes = block.nbytes
            with self._lock:
                if self._pending_bytes + nbytes > self._max_pending:
                    self._lost += block.shape[1]
                    self.dropped_samples += block.shape[1]
                    return
                self._pending_bytes += nbytes
                lost, self._lost = self._lost, 0
            self._queue.put(('data', block.copy(), nbytes, lost))
        else:
            with self._lock:
                lost, self._lost = self._lost, 0
            self._queue.put(('data', block, 0, lost))

    def mark_burst(self, event: str, t_wall: float) -> None:
        """
        Monitor bursts (PicoscopeDAQ.monitor_stream, burst=True): 'start' with
        the wall-clock time of the burst's first sample, 'end' after its last.
        """
        self._queue.put(('burst', event, t_wall))

    def end(self, timeout: float = 30.0) -> Optional[dict]:
        """
        Finish the stream; returns {'columns', 'units', 'time', 'values',
        'block_samples', 'rate_hz', 'delta_t'} or None if nothing was produced.
        """
        done = threading.Event()
        box: dict = {}
        self._queue.put(('end', done, box))
        done.wait(timeout)
        return box.get('result')

    def close(self) -> None:
        self._queue.put(('quit',))
        self._thread.join(timeout=5.0)

    # ── reducer thread ────────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            kind = item[0]
            try:
                if kind == 'quit':
                    return
                if kind == 'begin':
                    self._reducer = item[1]
                    self._reducer.seed_freqs(self._tracked)
                    self._segment_times, self._segment_values = [], []
                elif kind == 'data':
                    _, block, nbytes, lost = item
                    try:
                        self._process(block, lost)
                    finally:
                        if nbytes:
                            with self._lock:
                                self._pending_bytes -= nbytes
                elif kind == 'burst':
                    reducer = self._reducer
                    if reducer is not None and reducer.active:
                        if item[1] == 'start':
                            reducer.resync(item[2])
                        else:
                            self._deliver(*reducer.flush())
                elif kind == 'end':
                    _, done, box = item
                    box['result'] = self._segment_result()
                    if self._reducer is not None:
                        self._tracked.update(self._reducer.tracked_freqs())
                    self._reducer = None
                    done.set()
            except Exception as exc:          # never let the thread die silently
                self.error = f'{type(exc).__name__}: {exc}'
                if kind == 'end':
                    item[1].set()

    def _process(self, block: np.ndarray, lost: int) -> None:
        reducer = self._reducer
        if reducer is None or not reducer.active:
            return
        if lost:
            reducer.skip(lost)
        self._deliver(*reducer.feed(block))

    def _deliver(self, times: np.ndarray, values: np.ndarray) -> None:
        reducer = self._reducer
        if len(times):
            self._segment_times.append(times)
            self._segment_values.append(values)
            if self._on_rows is not None:
                self._on_rows(reducer.columns, reducer.units, times, values)

    def _segment_result(self) -> Optional[dict]:
        reducer = self._reducer
        if reducer is None or not self._segment_times:
            return None
        return {
            'columns':       list(reducer.columns),
            'units':         dict(reducer.units),
            'time':          np.concatenate(self._segment_times),
            'values':        np.concatenate(self._segment_values),
            'block_samples': reducer.block,
            'rate_hz':       1.0 / (reducer.block * reducer.dt),
            'delta_t':       reducer.dt,
            'config':        self.config.to_dict(),
        }


# ── History for the strip chart ────────────────────────────────────────────────

class SlowHistory:
    """The most recent ``capacity`` rows, contiguous in memory for plotting."""

    def __init__(self, capacity: int = 500_000):
        self.capacity = max(100, int(capacity))
        self.columns: List[str] = []
        self.units: Dict[str, str] = {}
        self._time = np.empty(0)
        self._data = np.empty((0, 0))
        self._n = 0

    def reset(self, columns: Sequence[str], units: Dict[str, str]) -> None:
        self.columns = list(columns)
        self.units   = dict(units)
        size = min(self.capacity, 4096)
        self._time = np.empty(size)
        self._data = np.empty((size, len(self.columns)))
        self._n = 0

    def clear(self) -> None:
        self._n = 0

    def set_capacity(self, capacity: int) -> None:
        self.capacity = max(100, int(capacity))
        if self._n > self.capacity:
            keep = self.capacity
            self._time[:keep] = self._time[self._n - keep:self._n]
            self._data[:keep] = self._data[self._n - keep:self._n]
            self._n = keep

    def append(self, times: np.ndarray, values: np.ndarray) -> None:
        k = len(times)
        if k == 0 or values.shape[1] != len(self.columns):
            return
        if k >= self.capacity:
            times, values, k = times[-self.capacity:], values[-self.capacity:], self.capacity
            self._n = 0
        if self._n + k > len(self._time):
            # Grow up to twice the capacity, then slide: appends stay amortised
            # O(1) and the stored rows are always one contiguous slice.
            keep = min(self._n, self.capacity - k)
            size = min(2 * self.capacity, max(2 * (keep + k), 4096))
            new_t = np.empty(size)
            new_d = np.empty((size, len(self.columns)))
            new_t[:keep] = self._time[self._n - keep:self._n]
            new_d[:keep] = self._data[self._n - keep:self._n]
            self._time, self._data, self._n = new_t, new_d, keep
        self._time[self._n:self._n + k] = times
        self._data[self._n:self._n + k] = values
        self._n += k

    def __len__(self) -> int:
        return min(self._n, self.capacity)

    def series(self, column: str, since: Optional[float] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """(time, values) of one column, optionally only rows at or after ``since``."""
        if column not in self.columns or self._n == 0:
            return np.empty(0), np.empty(0)
        lo = max(0, self._n - self.capacity)
        t = self._time[lo:self._n]
        y = self._data[lo:self._n, self.columns.index(column)]
        if since is not None:
            start = int(np.searchsorted(t, since))
            t, y = t[start:], y[start:]
        return t, y

    def latest(self, column: str) -> Optional[float]:
        if column not in self.columns or self._n == 0:
            return None
        return float(self._data[self._n - 1, self.columns.index(column)])


# ── Files ──────────────────────────────────────────────────────────────────────

def write_slow_group(f: h5py.File, slow: dict) -> None:
    """Write a finished stream's rows (SlowPipeline.end()) as the group ``slow``."""
    g = f.create_group('slow')
    g.attrs['rate_hz']       = slow['rate_hz']
    g.attrs['block_samples'] = slow['block_samples']
    g.attrs['delta_t']       = slow['delta_t']
    g.attrs['config']        = json.dumps(slow.get('config', {}))
    g.create_dataset('time', data=slow['time'])
    for j, col in enumerate(slow['columns']):
        ds = g.create_dataset(col, data=slow['values'][:, j])
        ds.attrs['unit'] = slow['units'].get(col, '')


class SlowLogWriter:
    """
    Continuous slow log: one HDF5 file that grows for as long as slow data is
    being produced, across monitoring, recording and the gaps between files.
    Same layout as the embedded group, with resizable datasets.
    """

    def __init__(self, path: str, columns: Sequence[str], units: Dict[str, str],
                 config: SlowConfig, flush_period: float = 5.0):
        self.path    = path
        self.columns = list(columns)
        self._flush_period = flush_period
        self._last_flush   = time.time()
        self._f = h5py.File(path, 'w')
        g = self._f.create_group('slow')
        g.attrs['rate_hz'] = config.rate_hz
        g.attrs['config']  = json.dumps(config.to_dict())
        g.attrs['created'] = time.time()
        self._time = g.create_dataset('time', shape=(0,), maxshape=(None,),
                                      dtype='f8', chunks=(4096,))
        self._sets = []
        for col in self.columns:
            ds = g.create_dataset(col, shape=(0,), maxshape=(None,),
                                  dtype='f8', chunks=(4096,))
            ds.attrs['unit'] = units.get(col, '')
            self._sets.append(ds)
        self._f.flush()
        self.n_rows = 0

    def append(self, times: np.ndarray, values: np.ndarray) -> None:
        if self._f is None or len(times) == 0:
            return
        n0, n1 = self.n_rows, self.n_rows + len(times)
        self._time.resize((n1,))
        self._time[n0:n1] = times
        for j, ds in enumerate(self._sets):
            ds.resize((n1,))
            ds[n0:n1] = values[:, j]
        self.n_rows = n1
        now = time.time()
        if now - self._last_flush >= self._flush_period:
            # Flushed regularly so a crash or power cut costs seconds, not the run.
            self._f.flush()
            self._last_flush = now

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None


def read_slow(path: str) -> Dict[str, np.ndarray]:
    """
    Read the slow group of a file — a continuous log or a recorded fast file.

    Returns {'time': ..., '<column>': ..., '_units': {...}, '_attrs': {...}}.
    """
    with h5py.File(path, 'r') as f:
        g = f['slow']
        out: Dict[str, object] = {name: ds[()] for name, ds in g.items()}
        out['_units'] = {name: ds.attrs.get('unit', '') for name, ds in g.items()
                         if name != 'time'}
        out['_attrs'] = dict(g.attrs)
    return out
