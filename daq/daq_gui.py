"""
Picoscope DAQ GUI — PyQt5 interface for PicoscopeDAQ.

Two modes share one scope session, one channel setup and one pair of plots:

  Monitor mode    free-running view of the signal, nothing written to disk.
                  Used for aligning, tuning feedback and checking ranges.
  Recording mode  the original file-by-file acquisition, with the same live
                  plots updating between files.

Switching between them is a single click: Record while monitoring stops the
monitor stream and starts acquisition without touching the scope setup, and
(optionally) drops back into monitoring when the run finishes.

Scales are set the way a scope sets them: sample rate in MS/s (kept in sync with
the integer ns/µs/ms period the driver actually takes), horizontal window as
s/div × divisions, and vertical range as autoscale / input range / explicit
min-max / mV per division.  The PSD panel takes explicit frequency and V²/Hz
limits.  All of it is editable while monitoring or recording.

The time panel rolls: history sits at negative time with the newest sample at 0,
on a fixed axis, so the trace slides leftwards instead of restarting each frame.
Trace and spectrum run on separate clocks, since a Welch transform costs far more
than a trace, and the drawing path is built around blitting so the trace can
update tens of times a second (see DaqPlotCanvas).  Both panels are zoomable by
wheel or toolbar, which suspends the live rescaling until "Reset view".

Recent frames are kept in a bounded ring (FrameHistory) so a transient that has
already scrolled off can be stepped back to, without interrupting acquisition.

The "Pulses" tab drives a Quantum Composers 9614 (pulse_gen/pulse_panel.py) and
can step the calibration-pulse amplitude with the run: N files at each amplitude
in a list, the amplitude changed between files, written into the filename and
into each file's HDF5 attributes.  That is dm_search/impulse_calibration.py with
the DAQ, not a clock, driving the loop.  Without the pulse generator connected
the tab is inert and everything else behaves exactly as before.

Slow inputs (a lock-in amplifier's output, a temperature sensor) share the scope
with the fast ones: a channel whose Save mode is "Slow" is averaged down to a
few rows a second in both modes, shown on a strip chart under the fast plots,
written into each recorded file (/slow) and optionally to a continuous log that
spans monitoring and recording (slow_data.py, slow_panel.py).

The monitor can stream endlessly or take bursts (see "Monitor stream"): several
channels at MS/s outrun a USB 2.0 link, and an endless stream then breaks up
after some tens of seconds, where bursts stay clean.

The "Inspect" tab reads files back off disk: browse a directory, plot a capture
in time and frequency, read its metadata.  It draws through the same
prepare_display/DaqPlotCanvas path as the live view, so a file looks the way it
looked while it was being taken, and reads are sliced so a long record can be
walked a window at a time.  Nothing there touches the scope, so it can be used
while a run is in progress.

Layout:
  Tabs        — Live and Inspect; the control bar sits below them and stays
                visible in both, so a run can be watched and stopped from either
  Live tab
    Left panel  — tabs: Acquisition (hardware selector, channel table,
                  acquisition settings, output path, live-view settings, display
                  settings) and Pulses (generator connection, pulse shape,
                  amplitude list, sequencing)
    Right panel — display channel selector + toolbar + dual plot (time series
                  top, PSD bottom) + per-channel readouts + history navigator
  Inspect tab — file browser, segment / analysis / display settings, the same
                dual plot, and a metadata tree
  Bottom bar  — Monitor / Record / Stop, mode, file counter, pressure, elapsed time

Both worker threads compute small display arrays (decimated trace + PSD) in the
background and emit them; the main thread only draws.  The last result is cached
so toggling display channels or view options re-renders immediately without
waiting for the next frame.

Run directly:
    python daq/daq_gui.py
or from the repo root:
    python take_data_gui.py
"""

import sys
import os
import re
import json
import time
import threading
from collections import deque
import h5py
import numpy as np
from scipy.signal import welch
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox, QSplitter,
    QScrollArea, QFrame, QSlider, QTabWidget, QListWidget, QListWidgetItem,
    QTreeWidget, QTreeWidgetItem,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daq.picoscope_daq import (
    PicoscopeDAQ, AcquisitionConfig, ChannelConfig,
    CHANNEL_INPUT_RANGES_MV, RANGE_LABELS, TIME_UNITS_SEC, PICO_SERIALS,
)

from daq.slow_data import (
    SlowConfig, SlowPipeline, SlowHistory, SlowLogWriter, slow_columns)
from daq.slow_panel import SlowSettingsPanel, SlowPlotCanvas

try:
    from control.src.agilent_twisstorr_84fsag_control import read_pressure as _hw_read_pressure
    _HAS_PRESSURE = True
except Exception:
    _HAS_PRESSURE = False

# Calibration pulses are optional: without pyserial (or without the pulse_gen
# package) the DAQ GUI still runs, just without the Pulses tab.
try:
    from pulse_gen.pulse_panel import PulseControlPanel, PulseSequencer
    _HAS_PULSES = True
except Exception:
    _HAS_PULSES = False

ALL_CHANNELS       = list('ABCDEFGH')
TIME_UNIT_OPTIONS  = ['PS4000A_NS', 'PS4000A_US', 'PS4000A_MS']
PRESETS_DIR        = os.path.join(os.path.dirname(__file__), 'presets')
NPERSEG_OPTIONS    = ['auto', '1024', '4096', '16384', '65536', '262144']
YLIM_OPTIONS       = ['auto', 'full range', 'manual', 'mV/div']
# Matplotlib's default cycle, for tinting the readout strip to match the traces.
_MPL_COLOURS       = {f'C{i}': c for i, c in enumerate(
    matplotlib.rcParams['axes.prop_cycle'].by_key().get('color', []))}
PSD_YLIM_OPTIONS   = ['auto', 'manual']
# Sustained streaming the link can carry, in bytes per second (2 per sample per
# channel).  The USB 2.0 figure is measured on this setup (~25 MB/s, with
# margin); the USB 3.0 one is deliberately conservative.
USB2_STREAM_LIMIT  = 20e6
USB3_STREAM_LIMIT  = 120e6


def _fmt_range(range_mv: int) -> str:
    """Format a range value (in mV) as a tidy label, e.g. '±1 V' or '±500 mV'."""
    if range_mv >= 1000:
        return f'±{range_mv // 1000} V'
    return f'±{range_mv} mV'


def _fmt_rate(fs: float) -> str:
    fs = float(f'{fs:.12g}')      # keep 1e6 from formatting as 1000 kS/s
    if fs >= 1e6:
        return f'{fs / 1e6:.4g} MS/s'
    if fs >= 1e3:
        return f'{fs / 1e3:.4g} kS/s'
    return f'{fs:.4g} S/s'


RATE_UNITS = {'MS/s': 1e6, 'kS/s': 1e3, 'S/s': 1.0}

# Samples per second per unit interval — dividing by these keeps exact rates exact
# (1e9 / 1000 is precisely 1 MS/s, whereas 1 / (1000 * 1e-9) is not).
_UNIT_HZ = {'PS4000A_NS': 1e9, 'PS4000A_US': 1e6, 'PS4000A_MS': 1e3}


def interval_to_rate(interval: int, units: str) -> float:
    return _UNIT_HZ[units] / interval


def rate_to_interval(fs: float) -> Tuple[int, str]:
    """
    Convert a desired sample rate to the (interval, time-units) pair the scope takes.

    ps4000aRunStreaming only accepts an integer interval in ns / µs / ms, so the
    finest unit that yields an in-range integer is chosen.  The rate actually
    achieved is therefore quantized — 1 MS/s is exact (1000 ns), but an arbitrary
    rate may land on a neighbouring period.
    """
    period = 1.0 / fs
    for units, scale in (('PS4000A_NS', 1e-9), ('PS4000A_US', 1e-6),
                         ('PS4000A_MS', 1e-3)):
        n = int(round(period / scale))
        if 1 <= n <= 999999:
            return n, units
    return (1, 'PS4000A_NS') if period < 1e-9 else (999999, 'PS4000A_MS')


def _fmt_hz(hz: float) -> str:
    if hz >= 1e6:
        return f'{hz / 1e6:.4g} MHz'
    if hz >= 1e3:
        return f'{hz / 1e3:.4g} kHz'
    return f'{hz:.4g} Hz'


def _fmt_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f'{seconds / 3600:.3g} h'
    if seconds >= 120:
        return f'{seconds / 60:.3g} min'
    if seconds >= 1.0:
        return f'{seconds:.3g} s'
    if seconds >= 1e-3:
        return f'{seconds * 1e3:.3g} ms'
    return f'{seconds * 1e6:.3g} µs'


# ── Display preparation (runs in worker threads) ────────────────────────────────

@dataclass
class DisplayOptions:
    """
    Everything that controls how raw counts become the small arrays we plot.

    Held by both workers and refreshed from the GUI on every frame, so the user
    can retune the view while a monitor or a run is in progress.
    """
    n_points:  int   = 4000     # target points in the time panel (before enveloping)
    psd_points: int  = 3000     # target points in the PSD panel (0 = full resolution)
    envelope:  bool  = True     # min/max decimation — keeps true amplitude visible
    nperseg:   int   = 0        # 0 = auto (≈ fs/10, capped)
    detrend:   bool  = True     # remove the DC level before the PSD
    time_span: float = 0.0      # seconds shown in the time panel (0 = whole record)
    psd_span:  float = 0.0      # seconds fed to Welch (0 = whole record)
    want_psd:  bool  = True
    roll:      bool  = True     # newest sample at t=0, history at negative t


def _decimate_trace(signal_mv: np.ndarray, dt: float, target_points: int,
                    envelope: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stride-decimate a trace to ~target_points for plotting.

    ``envelope`` is accepted for symmetry but is handled by _envelope_bins, which
    returns a min/max band instead of a single-valued trace.
    """
    n = len(signal_mv)
    if n <= target_points or target_points < 2:
        return np.arange(n) * dt, signal_mv
    stride = int(np.ceil(n / target_points))
    return np.arange(0, n, stride) * dt, signal_mv[::stride]


def _envelope_bins(signal: np.ndarray, dt: float, target_points: int, scale: float
                   ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Per-bin min/max of a raw int16 trace, as (bin centres, low, high) in mV.

    Two reasons this beats stride decimation:

    * Correctness — taking every Nth sample of a 100 kHz oscillation sampled at
      5 MS/s aliases into a squiggle whose amplitude is simply wrong.  The
      min/max band always spans the real excursion.
    * Speed — the reduction runs on the int16 samples and only the ~2000 results
      are converted to mV, instead of converting the whole half-million-sample
      window.  The band is then drawn as one filled polygon, which is orders of
      magnitude cheaper than stroking a 4000-vertex zigzag.

    Returns None when the window is too short to be worth binning.
    """
    n = len(signal)
    if target_points < 2:
        return None
    stride = int(np.ceil(n / target_points))
    if stride < 4:
        return None
    n_bins  = n // stride
    trimmed = signal[:n_bins * stride].reshape(n_bins, stride)
    lo = trimmed.min(axis=1).astype(np.float32) * scale
    hi = trimmed.max(axis=1).astype(np.float32) * scale
    tt = (np.arange(n_bins) + 0.5) * stride * dt
    return tt, lo, hi


def _psd(signal_mv: np.ndarray, fs: float, opts: DisplayOptions
         ) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD in V²/Hz from a trace in mV."""
    if opts.nperseg > 0:
        nperseg = opts.nperseg
    else:
        nperseg = max(256, min(int(fs / 10), 65536))
    nperseg = int(min(nperseg, len(signal_mv)))
    ff, pp_mv2hz = welch(signal_mv, fs=fs, nperseg=nperseg,
                         detrend='constant' if opts.detrend else False)
    return ff, pp_mv2hz * 1e-6


def _decimate_psd(ff: np.ndarray, pp: np.ndarray, target_points: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin a spectrum for display, keeping the largest value in each bin.

    A 65536-point Welch gives ~33000 bins; drawing them all costs ~8 ms per
    channel per frame for detail no screen can show.  Keeping the maximum (rather
    than the mean) preserves narrow peaks, which is what these spectra are read
    for — at the cost of showing the broadband floor a few dB high, since the
    maximum of a handful of noisy bins sits above their mean.  Set psd_points to
    0 to plot every bin, and note the peak readout always comes from the
    full-resolution spectrum regardless.

    Frequencies are the bin centres, not the position of each maximum, so the
    grid is identical from frame to frame — the running average depends on that.
    """
    n = len(ff)
    if target_points <= 0 or n < 2 * target_points:
        return ff, pp
    stride = int(np.ceil(n / target_points))
    n_bins = n // stride
    pp_b = pp[:n_bins * stride].reshape(n_bins, stride)
    ff_b = ff[:n_bins * stride].reshape(n_bins, stride)
    return ff_b[:, stride // 2], pp_b.max(axis=1)


def prepare_display(data: np.ndarray, dt: float, adc2mvs: np.ndarray,
                    channels: List[ChannelConfig], opts: DisplayOptions) -> Dict:
    """
    Turn an int16 capture into the per-channel display dict the canvas draws.

    Shared by both workers so a monitor frame and a recorded file are rendered
    through exactly the same path — the plots do not change appearance when the
    mode changes, only their update cadence does.

    Besides one entry per channel, the result carries a ``'_meta'`` entry.  Its
    ``graticule`` flag says whether this frame is a fixed-length live window (so
    the canvas may pin the x-axis to the s/div graticule) or a whole-file record
    of some other length (autoscale instead).
    """
    fs      = 1.0 / dt
    n_total = data.shape[1]
    n_time  = n_total if opts.time_span <= 0 else min(
        n_total, max(2, int(opts.time_span / dt)))
    n_psd   = n_total if opts.psd_span <= 0 else min(
        n_total, max(256, int(opts.psd_span / dt)))

    # Roll mode puts the newest sample at t=0 with history running negative, so
    # the x-axis is fixed and the trace slides leftwards like a scope in roll —
    # instead of every frame restarting at t=0 and appearing to flash in place.
    roll = opts.roll and opts.time_span > 0

    display: Dict[str, dict] = {}
    for i, ch in enumerate(channels):
        scale   = float(adc2mvs[i])
        raw_seg = data[i, n_total - n_time:]

        band = _envelope_bins(raw_seg, dt, opts.n_points, scale) if opts.envelope \
            else None
        if band is not None:
            tt, lo, hi = band
            trace = None
        else:
            lo = hi = None
            tt, trace = _decimate_trace(
                raw_seg.astype(np.float32) * scale, dt, opts.n_points, False)
        if roll and len(tt):
            tt = tt - n_time * dt

        # Statistics come off the int16 samples with a float32 accumulator: the
        # full window never has to be materialised as floats.
        entry = {
            'tt':        tt,
            'trace_mv':  trace,
            'lo_mv':     lo,
            'hi_mv':     hi,
            'range_mv':  int(CHANNEL_INPUT_RANGES_MV[ch.range_idx]),
            'mean_mv':   float(np.mean(raw_seg, dtype=np.float32)) * scale,
            'rms_mv':    float(np.std(raw_seg, dtype=np.float32)) * scale,
            'ff':        None,
            'pp':        None,
            'peak_hz':   None,
        }
        if opts.want_psd:
            psd_seg = data[i, n_total - n_psd:].astype(np.float32) * scale
            ff, pp = _psd(psd_seg, fs, opts)
            # Peak from the full-resolution spectrum, before any thinning.
            if len(ff) > 2:
                entry['peak_hz'] = float(ff[1 + int(np.argmax(pp[1:]))])
            ff, pp = _decimate_psd(ff, pp, opts.psd_points)
            entry['ff'] = ff
            entry['pp'] = pp
        display[ch.name] = entry

    display['_meta'] = {
        'time_span': n_time * dt,
        'graticule': opts.time_span > 0,
        'roll':      roll,
        'fs':        fs,
    }
    return display


# ── Frame history ──────────────────────────────────────────────────────────────

class FrameHistory:
    """
    Bounded ring of recent display frames — the equivalent of a scope's display
    memory, so a transient that has already scrolled off can be looked at again.

    What is kept is the *display* frame: the decimated trace or min/max band plus
    whatever spectrum that frame carried, not the raw samples.  Retaining raw
    samples would cost megabytes per frame at a high sample rate (0.5 s at 5 MS/s
    is 5 MB per channel), so, exactly as on a scope, stepping back shows the trace
    at the resolution it was drawn — zooming into an old frame reveals no detail
    that was not already on screen.  A frame is a few tens of kilobytes, so
    hundreds of them are affordable; ``nbytes`` reports the real figure.

    Frames are addressed by an ever-increasing sequence number rather than a
    position, so a selection stays pointed at the same waveform as new frames
    arrive and old ones fall off the back.
    """

    def __init__(self, depth: int = 200):
        self._depth  = max(0, depth)
        self._frames: deque = deque(maxlen=self._depth or 1)
        self._next_seq = 0

    @property
    def enabled(self) -> bool:
        return self._depth > 0

    @property
    def depth(self) -> int:
        return self._depth

    def set_depth(self, depth: int) -> None:
        """Resize, keeping the newest frames that still fit."""
        depth = max(0, depth)
        if depth == self._depth:
            return
        kept = list(self._frames)[-depth:] if depth else []
        self._depth  = depth
        self._frames = deque(kept, maxlen=depth or 1)
        if not depth:
            self._frames.clear()

    def clear(self) -> None:
        self._frames.clear()

    def append(self, frame: Dict) -> Optional[int]:
        if not self.enabled:
            return None
        seq = self._next_seq
        self._next_seq += 1
        self._frames.append((seq, frame))
        return seq

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def oldest_seq(self) -> Optional[int]:
        return self._frames[0][0] if self._frames else None

    @property
    def newest_seq(self) -> Optional[int]:
        return self._frames[-1][0] if self._frames else None

    def get(self, seq: int) -> Optional[Dict]:
        """Frame with this sequence number, or None if it has aged out."""
        if not self._frames:
            return None
        index = seq - self._frames[0][0]
        if 0 <= index < len(self._frames):
            found_seq, frame = self._frames[index]
            if found_seq == seq:
                return frame
        for found_seq, frame in self._frames:      # deque was resized under us
            if found_seq == seq:
                return frame
        return None

    def wall_time(self, seq: int) -> Optional[float]:
        frame = self.get(seq)
        if frame is None:
            return None
        return (frame.get('_meta') or {}).get('wall_time')

    def time_span(self) -> float:
        """Wall-clock seconds between the oldest and newest stored frame."""
        if len(self._frames) < 2:
            return 0.0
        first = (self._frames[0][1].get('_meta') or {}).get('wall_time')
        last  = (self._frames[-1][1].get('_meta') or {}).get('wall_time')
        if first is None or last is None:
            return 0.0
        return max(0.0, last - first)

    def nbytes(self) -> int:
        """
        Actual memory held by the stored arrays.

        Consecutive frames share a carried-forward spectrum, so identical arrays
        are counted once — otherwise this would overstate usage several-fold.
        """
        seen: set = set()
        total = 0
        for _seq, frame in self._frames:
            for ch, d in frame.items():
                if ch.startswith('_'):
                    continue
                for key in ('tt', 'trace_mv', 'lo_mv', 'hi_mv', 'ff', 'pp'):
                    arr = d.get(key)
                    if arr is not None and id(arr) not in seen:
                        seen.add(id(arr))
                        total += arr.nbytes
        return total


# ── Monitor worker ─────────────────────────────────────────────────────────────

class MonitorWorker(QThread):
    """
    Free-running monitor: stream, compute time trace + PSD, emit, repeat.

    The time trace and the spectrum run on separate clocks.  The trace is cheap
    (pull the last ``time_span`` seconds, decimate) so it goes out at the full
    refresh rate to give a smoothly rolling display; Welch over a long span is
    expensive, so the PSD is recomputed at most every ``psd_refresh`` seconds and
    frames in between are flagged ``psd_updated=False`` — the canvas then only
    redraws the top panel.

    Settings are mutable while the thread runs.  Cadence and display settings are
    picked up on the next frame; changing the ring length (time window or PSD
    span) needs a new streaming run, so those set a restart flag — the outer loop
    re-enters monitor_stream() with the new geometry, which takes well under a
    second and never drops the scope connection.

    Three threads are involved, and the split between them is the whole point:

    * the QThread body polls the driver and must never be made to wait;
    * a render thread turns a window of samples into a display frame (the Welch
      is by far the most expensive thing here);
    * the GUI thread draws.

    Each hand-off keeps at most one item and drops the rest.  Neither queue may
    grow: a backlog in front of the driver corrupts the data (see _on_window) and
    a backlog in front of the GUI exhausts memory (see _render_frame).  Dropping
    a live frame costs nothing — the next one is already more current.
    """
    frame_ready = pyqtSignal(dict, bool)   # display dict, psd_updated
    status      = pyqtSignal(str)          # over-range / rate line
    slow_rows   = pyqtSignal(object)       # (columns, units, times, values)
    done        = pyqtSignal()
    error       = pyqtSignal(str)

    def __init__(self, daq: PicoscopeDAQ, refresh: float, time_span: float,
                 psd_span: float, chunk: float, display_opts: DisplayOptions,
                 psd_average: int = 1, psd_refresh: float = 0.3,
                 slow_config: Optional[SlowConfig] = None,
                 slow_tracked: Optional[dict] = None,
                 burst: bool = False):
        super().__init__()
        self._daq   = daq
        self._burst = burst
        # Slow data is reduced from every sample on a thread of its own (see
        # slow_data.SlowPipeline) — independent of the display path below, which
        # only ever looks at the newest window and drops the rest.
        self._slow: Optional[SlowPipeline] = None
        if slow_config is not None and slow_config.enabled:
            self._slow = SlowPipeline(
                slow_config, on_rows=lambda *rows: self.slow_rows.emit(rows),
                tracked=slow_tracked)
        self._lock  = threading.Lock()
        self._stop_requested = False
        self._restart        = False
        self._refresh     = refresh
        self._time_span   = time_span
        self._psd_span    = psd_span
        self._chunk       = chunk
        self._opts        = display_opts
        self._psd_average = max(1, psd_average)
        self._psd_refresh = psd_refresh
        self._psd_acc: Dict[str, np.ndarray] = {}
        self._last_psd    = 0.0
        self._frames      = 0
        self._fps         = 0.0
        self._fps_t0      = 0.0
        # Hand-off to the render thread: one slot, newest wins.
        self._pending: Optional[tuple] = None
        self._pending_cv   = threading.Condition()
        self._dump_path: Optional[str] = None
        self._render_thread: Optional[threading.Thread] = None
        self._in_flight    = 0      # frames emitted but not yet drawn
        self._dropped      = 0

    # ── control from the GUI thread ────────────────────────────────────────────

    def request_stop(self) -> None:
        self._stop_requested = True
        with self._pending_cv:
            self._pending_cv.notify_all()

    def request_raw_dump(self, path: str) -> None:
        """
        Write the next monitor window to ``path`` as raw int16, exactly as it
        came out of the ring.

        Monitoring keeps nothing, so an artefact that appears only here cannot be
        looked at offline — which is precisely the case where staring at the
        rendered picture stops being enough.  This is the escape hatch: no
        decimation, no Welch, no plotting, just the samples.
        """
        with self._pending_cv:
            self._dump_path = path

    def frame_consumed(self) -> None:
        """
        Called from the GUI thread once a frame has been drawn.

        This is the acknowledgement half of the backpressure in _render_frame;
        without it the worker would emit one frame and then stall for good.
        """
        with self._pending_cv:
            self._in_flight = max(0, self._in_flight - 1)

    def apply_settings(self, refresh: float, time_span: float, psd_span: float,
                       chunk: float, display_opts: DisplayOptions,
                       psd_average: int, psd_refresh: float,
                       burst: Optional[bool] = None) -> None:
        """Push new settings in; restart the stream only if the ring must resize."""
        with self._lock:
            if burst is None:
                burst = self._burst
            geometry_changed = (time_span != self._time_span
                                or burst    != self._burst
                                or psd_span != self._psd_span
                                or chunk    != self._chunk
                                or refresh  != self._refresh)
            average_changed  = psd_average != self._psd_average
            psd_shape_changed = display_opts.nperseg != self._opts.nperseg
            self._refresh     = refresh
            self._burst       = burst
            self._time_span   = time_span
            self._psd_span    = psd_span
            self._chunk       = chunk
            self._opts        = display_opts
            self._psd_average = max(1, psd_average)
            self._psd_refresh = psd_refresh
            if average_changed or psd_shape_changed or geometry_changed:
                self._psd_acc = {}
            if geometry_changed:
                self._restart = True

    def reset_average(self) -> None:
        with self._lock:
            self._psd_acc = {}

    # ── thread body ────────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            self._render_thread = threading.Thread(
                target=self._render_loop, name='monitor-render', daemon=True)
            self._render_thread.start()
            while not self._stop_requested:
                with self._lock:
                    self._restart = False
                    ring     = max(self._time_span, self._psd_span, 0.01)
                    refresh  = self._refresh
                    # A driver chunk longer than the refresh interval would cap
                    # the frame rate at the chunk rate, so keep it under.
                    chunk    = min(self._chunk, refresh)
                    burst    = self._burst
                if burst:
                    # A burst is the whole ring, so the ring is exactly what the
                    # display uses; the chunk is only the transfer granularity.
                    chunk = min(self._chunk, ring)
                self._last_psd = 0.0
                self._fps_t0   = time.perf_counter()
                self._frames   = 0
                slow = self._slow
                if slow is not None:
                    cfg = self._daq.config
                    slow.begin(time.time(), cfg.dt,
                               [ch.name for ch in cfg.channels],
                               self._daq._adc2mv_factors())
                try:
                    self._daq.monitor_stream(
                        on_window=self._on_window,
                        ring_duration=ring,
                        update_period=refresh,
                        chunk_duration=chunk,
                        should_stop=lambda: self._stop_requested or self._restart,
                        on_samples=slow.push if slow is not None else None,
                        burst=burst,
                        on_burst=slow.mark_burst if slow is not None else None,
                    )
                finally:
                    if slow is not None:
                        slow.end(timeout=5.0)
            self.done.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self._stop_requested = True
            if self._slow is not None:
                self._slow.close()
            with self._pending_cv:
                self._pending_cv.notify_all()
            if self._render_thread is not None:
                self._render_thread.join(timeout=2.0)
                self._render_thread = None

    def _on_window(self, get_window, adc2mvs: np.ndarray,
                   dt: float, overflow: int) -> None:
        """
        Copy the ring and hand it to the render thread.  Nothing else.

        This runs on the thread that polls the driver, and while we are inside it
        ps4000aGetStreamingLatestValues is not being called, so the scope has
        nowhere to put new samples.  Building the display frame here instead —
        ~80 ms for three channels at 5 MS/s, almost all of it Welch — starves the
        poll loop enough that the driver's capture buffer slowly falls behind and
        eventually laps.  What comes back after that is torn at the driver's
        transfer-block granularity: a comb of lines at multiples of fs/512 right
        across the spectrum, identical on every channel, appearing after tens of
        seconds of monitoring rather than straight away, because the backlog has
        to build first.  The copy below is ~3 ms and leaves the loop free.
        """
        with self._lock:
            opts        = self._opts
            psd_refresh = self._psd_refresh

        now = time.perf_counter()
        want_psd = opts.want_psd and (now - self._last_psd >= psd_refresh)

        # Pull only what this frame needs: the trace span at every frame, the
        # (possibly much longer) PSD span only when the spectrum is due.
        span = max(opts.time_span, opts.psd_span if want_psd else 0.0)
        window = get_window(span if span > 0 else None)
        if want_psd:
            self._last_psd = now

        with self._pending_cv:
            if self._pending is not None:
                # Render thread still busy — the newer window wins, but inherit
                # the dropped frame's want_psd so a run of trace-only frames
                # cannot starve the spectrum.
                self._dropped += 1
                want_psd = want_psd or self._pending[-1]
            self._pending = (window, adc2mvs, dt, overflow, opts, want_psd)
            self._pending_cv.notify()

    def _render_loop(self) -> None:
        """Turn handed-off windows into display frames, newest first."""
        while True:
            with self._pending_cv:
                while self._pending is None and not self._stop_requested:
                    self._pending_cv.wait(0.1)
                if self._pending is None:
                    if self._stop_requested:
                        return
                    continue
                job, self._pending = self._pending, None
            try:
                self._render_frame(*job)
            except Exception as exc:
                self.error.emit(str(exc))
                return

    def _render_frame(self, window: np.ndarray, adc2mvs: np.ndarray, dt: float,
                      overflow: int, opts: DisplayOptions, want_psd: bool) -> None:
        with self._lock:
            psd_average = self._psd_average

        channels   = self._daq.config.channels

        with self._pending_cv:
            dump_path, self._dump_path = self._dump_path, None
        if dump_path:
            try:
                with h5py.File(dump_path, 'w') as f:
                    g = f.create_group('data')
                    g.attrs['timestamp'] = time.time()
                    g.attrs['delta_t']   = dt
                    g.attrs['source']    = 'monitor_stream raw window'
                    for i, ch in enumerate(channels):
                        ds = g.create_dataset(f'channel_{ch.name.lower()}',
                                              data=window[i], dtype=np.int16)
                        ds.attrs['adc2mv'] = float(adc2mvs[i])
                self.status.emit(f'Raw window written to {dump_path}')
            except Exception as exc:
                self.error.emit(f'Raw dump failed: {exc}')

        frame_opts = opts if want_psd else replace(opts, want_psd=False)
        display    = prepare_display(window, dt, adc2mvs, channels, frame_opts)
        if want_psd and psd_average > 1:
            display = self._accumulate_psd(display, psd_average)

        # Backpressure.  frame_ready crosses threads, so Qt queues it, and that
        # queue has no bound: if the canvas draws slower than we produce, frames
        # of a megabyte apiece pile up in the event queue until the machine is
        # swapping — which then slows this thread and the poll loop with it.
        # Keep exactly one frame in flight and drop the rest.
        with self._pending_cv:
            busy = self._in_flight > 0
            if not busy:
                self._in_flight += 1
        if busy:
            self._dropped += 1
            if want_psd:
                self._last_psd = 0.0    # recompute the spectrum on the next frame
        else:
            self.frame_ready.emit(display, want_psd)

        self._frames += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 1.0:
            self._fps    = self._frames / (now - self._fps_t0)
            self._frames = 0
            self._fps_t0 = now

        over = [ch.name for i, ch in enumerate(channels) if overflow & (1 << i)]
        msg  = (f'Monitoring{" (sweeps)" if self._burst else ""} — '
                f'{_fmt_duration(window.shape[1] * dt)} @ '
                f'{_fmt_rate(1 / dt)} · {self._fps:.3g} fps')
        if self._dropped:
            msg += f' · {self._dropped} dropped'
        if self._slow is not None:
            if self._slow.error:
                msg += f'   ⚠ SLOW DATA: {self._slow.error}'
            elif self._slow.dropped_samples:
                msg += (f'   ⚠ slow data behind — '
                        f'{self._slow.dropped_samples * dt:.1f} s skipped')
        health = self._daq.stream_health
        if health:
            # The driver losing samples is the one failure that looks like a
            # signal rather than an error, so it goes on the status line.
            msg += f'   ⚠ DRIVER: {health}'
        if over:
            msg += f'   ⚠ OVER-RANGE: {", ".join(over)}'
        self.status.emit(msg)

    def _accumulate_psd(self, display: Dict, n_average: int) -> Dict:
        """
        Exponential running average of the PSD across frames.

        A single monitor frame is short, so its PSD is noisy; averaging makes weak
        peaks (e.g. a drive line or a low-amplitude mode) stand out without
        lengthening the frame and slowing the refresh.
        """
        alpha = 1.0 / n_average
        for ch, d in display.items():
            if ch.startswith('_'):          # '_meta', not a channel
                continue
            pp = d['pp']
            if pp is None:
                continue
            acc = self._psd_acc.get(ch)
            if acc is None or acc.shape != pp.shape:
                acc = pp.copy()
            else:
                acc = (1 - alpha) * acc + alpha * pp
            self._psd_acc[ch] = acc
            d['pp'] = acc
            ff = d['ff']
            if ff is not None and len(ff) > 2:
                d['peak_hz'] = float(ff[1 + int(np.argmax(acc[1:]))])
        return display


# ── Acquisition worker ─────────────────────────────────────────────────────────

class AcquisitionWorker(QThread):
    """
    Acquire → save → compute display data, in a background thread.

    Signals
    -------
    preview_ready : emitted during acquisition every ~refresh seconds with a
                    rolling window — both panels update, exactly as in monitor mode.
    data_ready    : emitted after each complete file, with the PSD computed from
                    the whole file (finer frequency resolution than the preview).
    progress      : file count + pressure update.
    pulse_changed : (file index, amplitude) when a pulse sequence steps.
    message       : log line from the pulse sequencer.

    Only small display arrays cross the thread boundary, not the raw int16 buffer.
    Stop is cooperative (only checked between files; acquire_one is blocking).

    Calibration pulses
    ------------------
    With a ``sequencer``, the amplitude is stepped from this thread between
    files — never during one — so every file belongs to exactly one amplitude.
    The amplitude is recorded in the filename and in the file's attributes, and
    the output is turned off on every exit path (finish() in a finally block).
    """
    preview_ready = pyqtSignal(dict)
    data_ready    = pyqtSignal(int, dict)
    progress      = pyqtSignal(int, int, float)
    pulse_changed = pyqtSignal(int, float)
    message       = pyqtSignal(str)
    slow_rows     = pyqtSignal(object)        # (columns, units, times, values)
    done          = pyqtSignal()              # avoid shadowing QThread.finished
    error         = pyqtSignal(str)

    def __init__(self, daq: PicoscopeDAQ, file_directory: str, file_prefix: str,
                 idx_start: int, n_files: int, read_pressure: bool, pressure_port: str,
                 display_opts: DisplayOptions, chunk_period: float = 0.5,
                 window_duration: float = 2.0, sequencer=None,
                 tag_filenames: bool = True, restart_index: bool = True,
                 pulse_attrs: Optional[dict] = None,
                 slow_config: Optional[SlowConfig] = None,
                 slow_embed: bool = True,
                 slow_tracked: Optional[dict] = None):
        super().__init__()
        # Same reduction as in monitor mode, fed from the capture buffer.  The
        # rows of each file are also written into that file (slow_embed).
        self._slow: Optional[SlowPipeline] = None
        self._slow_embed = slow_embed
        if slow_config is not None and slow_config.enabled:
            self._slow = SlowPipeline(
                slow_config, on_rows=lambda *rows: self.slow_rows.emit(rows),
                tracked=slow_tracked)
        self._daq              = daq
        self._file_directory   = file_directory
        self._file_prefix      = file_prefix
        self._idx_start        = idx_start
        self._n_files          = n_files
        self._read_pressure    = read_pressure
        self._pressure_port    = pressure_port
        self._chunk_period     = chunk_period
        self._window_duration  = window_duration
        self._sequencer        = sequencer
        self._tag_filenames    = tag_filenames
        self._restart_index    = restart_index
        self._pulse_attrs      = pulse_attrs or {}
        self._lock             = threading.Lock()
        self._opts             = display_opts
        self._stop_requested   = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def apply_display_options(self, opts: DisplayOptions) -> None:
        """Retune the live view mid-run; picked up on the next preview chunk."""
        with self._lock:
            self._opts = opts

    def _current_opts(self) -> DisplayOptions:
        with self._lock:
            return self._opts

    def run(self) -> None:
        try:
            os.makedirs(self._file_directory, exist_ok=True)
            for i in range(self._n_files):
                if self._stop_requested:
                    break

                # Pulses first: the scope is idle here, so the amplitude change
                # and its settling land between files rather than inside one.
                attrs = dict(self._pulse_attrs)
                tag   = ''
                index = i + self._idx_start
                if self._sequencer is not None:
                    previous  = self._sequencer.step_index(i - 1) if i else None
                    amplitude = self._sequencer.before_file(i)
                    if self._sequencer.step_index(i) != previous:
                        self.pulse_changed.emit(i, amplitude)
                    attrs['pulse_amplitude_v'] = amplitude
                    if self._tag_filenames:
                        tag = self._sequencer.file_tag(i)
                    if self._restart_index:
                        # Number files within their amplitude, as the script did:
                        # ..._2.5v_0, ..._2.5v_1, ..._5v_0, …
                        index = self._sequencer.index_in_step + self._idx_start

                pressure = None
                if self._read_pressure and _HAS_PRESSURE:
                    try:
                        pressure = _hw_read_pressure(port=self._pressure_port)
                    except Exception:
                        pressure = -1.0

                slow = self._slow
                if slow is not None:
                    cfg = self._daq.config
                    slow.begin(time.time(), cfg.dt,
                               [ch.name for ch in cfg.channels],
                               self._daq._adc2mv_factors())
                slow_result = None
                try:
                    timestamp, dt, adc2mvs, data = self._daq.acquire_one(
                        on_chunk=self._on_preview_chunk,
                        chunk_period=self._chunk_period,
                        window_duration=self._window_duration,
                        on_samples=slow.push if slow is not None else None,
                    )
                finally:
                    if slow is not None:
                        # Waits for the reducer to finish this file's samples;
                        # it runs alongside the capture, so this is short.
                        slow_result = slow.end(timeout=60.0)
                        if slow.error:
                            self.message.emit(f'slow data: {slow.error}')

                fname = f'{self._file_prefix}{tag}{index}.hdf5'
                self._daq.write_hdf5(
                    os.path.join(self._file_directory, fname),
                    timestamp, dt, adc2mvs, data, pressure,
                    extra_attrs=attrs or None,
                    slow=slow_result if self._slow_embed else None,
                )

                # Whole-file PSD: same code path as the preview, but with the
                # spans opened up so the full record is used.
                opts = self._current_opts()
                full = replace(opts, time_span=0.0, psd_span=0.0,
                               want_psd=True, roll=False)
                display = prepare_display(
                    data, dt, adc2mvs, self._daq.config.channels, full)
                self.data_ready.emit(i, display)
                self.progress.emit(i + 1, self._n_files,
                                   pressure if pressure is not None else -1.0)

            self.done.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            # Whatever happened — finished, stopped, or blown up — the pulses
            # must not be left running.
            if self._sequencer is not None:
                try:
                    self._sequencer.finish()
                except Exception as exc:
                    self.message.emit(f'pulse shutdown failed: {exc}')
            if self._slow is not None:
                self._slow.close()

    def _on_preview_chunk(self, window: np.ndarray,
                          adc2mvs: np.ndarray, dt: float) -> None:
        """
        Called from inside the streaming callback (still in this worker thread).
        Qt queued connections handle thread-safe delivery to the main thread.
        """
        display = prepare_display(window, dt, adc2mvs,
                                  self._daq.config.channels, self._current_opts())
        self.preview_ready.emit(display)


# ── Dual-panel plot canvas ─────────────────────────────────────────────────────

@dataclass
class ViewOptions:
    """Pure drawing choices — applied on the GUI thread, no reacquisition needed."""
    psd_logx:  bool  = False
    f_min:     float = 0.0        # 0 = auto
    f_max:     float = 0.0        # 0 = auto
    psd_ymin:  float = 0.0        # 0 = auto (log axis, so must be > 0 to apply)
    psd_ymax:  float = 0.0        # 0 = auto
    ylim_mode: str   = 'auto'     # see YLIM_OPTIONS
    y_min:     float = -1000.0    # 'manual' mode, mV
    y_max:     float = 1000.0     # 'manual' mode, mV
    mv_per_div: float = 500.0     # 'mV/div' mode
    n_div_y:   int   = 8
    time_per_div: float = 0.0     # 0 = do not pin the x-axis to a graticule
    n_div_x:   int   = 10
    show_grid_divs: bool = True   # ticks on division boundaries, scope style
    show_peak: bool  = False
    show_rms:  bool  = True


class DaqPlotCanvas(FigureCanvasQTAgg):
    """
    Two stacked subplots:
      top    — time series (mV), one trace per selected channel
      bottom — PSD (V²/Hz, log y), one trace per selected channel

    Each channel keeps a consistent colour from the default cycle so the legend
    matches between the two panels.  The legend entries include the digitization
    range so scale differences between channels are immediately visible
    (e.g. "Ch D  ±1 V", "Ch C  ±5 V").

    Fast path
    ---------
    Clearing the axes and re-plotting costs tens of milliseconds, which caps the
    frame rate far below what a rolling display needs.  So the artists persist
    and only their data is replaced: the static furniture (frame, grid, ticks,
    labels, legends) is cached as a bitmap, and every frame restores that bitmap
    and blits the traces, spectra and peak markers on top of it.

    A full redraw is needed only when the structure changes — channel set, axis
    scale, or limits that no longer fit the data.  A new frame on settled axes,
    spectrum included, never triggers one.

    Two details matter for speed, both measured rather than guessed:
      * The envelope is drawn as a filled band, not a min/max zigzag line.
        Stroking 4000 alternating vertices costs ~35 ms; filling the same band
        costs ~0.2 ms.
      * The figure uses fixed margins instead of tight_layout, which otherwise
        re-solves the layout on every full redraw (~25 ms each).

    Zooming
    -------
    Wheel over a panel zooms it about the cursor (x only in the time panel,
    both axes in the PSD panel; hold Shift for the y-axis alone).  Toolbar pan
    and box-zoom work as usual.  Any of these hands control of the limits to the
    user: ``user_zoomed`` goes true, the live limit logic stands down so the view
    stops jumping back, and it stays that way until reset_view().
    """
    zoom_changed = pyqtSignal(bool)      # True once the user owns the limits

    def __init__(self):
        self.fig = Figure()
        super().__init__(self.fig)
        self._apply_margins()
        self._ax_time = None
        self._ax_psd  = None
        self._color_map: Dict[str, str] = {}
        self._view = ViewOptions()

        self._lines_time: Dict[str, object] = {}   # plain traces
        self._bands:      Dict[str, object] = {}   # min/max envelope bands
        self._lines_psd:  Dict[str, object] = {}
        self._peak_lines: Dict[str, object] = {}   # peak markers on the spectrum
        self._layout_key: Optional[tuple]   = None
        self._background                    = None
        self._in_own_draw                   = False
        self._setting_limits                = False
        self.user_zoomed                    = False

        self.setMinimumSize(480, 400)
        self._init_axes()
        self.mpl_connect('scroll_event', self._on_scroll)
        self.mpl_connect('draw_event', self._on_draw_event)

    # ── setup ──────────────────────────────────────────────────────────────────

    def _apply_margins(self) -> None:
        """
        Reserve a fixed number of pixels for the axis furniture.

        tight_layout would do this automatically but re-solves on every full
        redraw (~25 ms).  Fractional margins cannot replace it directly, since a
        fraction that leaves room for the tick labels on a wide canvas clips them
        on a narrow one — so the pixel budget is converted to fractions at the
        current size, and refreshed on resize.
        """
        scale = self.fig.dpi / 100.0
        w = max(float(self.fig.bbox.width),  120.0)
        h = max(float(self.fig.bbox.height), 120.0)
        # The right pad also has to cover half of the last x tick label, which is
        # centred on the axes edge and would otherwise spill off the canvas.
        left   = min(0.40, 84 * scale / w)
        right  = max(left + 0.2, 1 - 26 * scale / w)
        bottom = min(0.40, 52 * scale / h)
        top    = max(bottom + 0.2, 1 - 16 * scale / h)
        self.fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                                 hspace=min(0.6, 120 * scale / h))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_margins()
        self._background = None      # geometry changed; cache is void

    def reset_axes(self) -> None:
        """Discard both panels — used when the active channel set changes."""
        self._init_axes()

    def _init_axes(self) -> None:
        self.fig.clf()
        self._ax_time = self.fig.add_subplot(2, 1, 1)
        self._ax_psd  = self.fig.add_subplot(2, 1, 2)
        self._lines_time.clear()
        self._bands.clear()
        self._lines_psd.clear()
        self._peak_lines.clear()
        self._layout_key  = None
        self._background  = None
        self._style_time_axis()
        self._style_psd_axis()
        for ax in (self._ax_time, self._ax_psd):
            ax.callbacks.connect('xlim_changed', self._on_limits_changed)
            ax.callbacks.connect('ylim_changed', self._on_limits_changed)
        self.draw()

    def set_view_options(self, view: ViewOptions) -> None:
        """
        Install new drawing options.

        Any change invalidates the cached layout: the fast path deliberately does
        not re-examine limits or styling, so without this a new vertical range or
        frequency window would not appear until something else forced a rebuild.
        """
        if view != self._view:
            self._layout_key = None
        self._view = view

    def _style_time_axis(self) -> None:
        self._ax_time.set_ylabel('Signal (mV)')
        self._ax_time.set_xlabel('Time (s)')
        self._ax_time.grid(True, alpha=0.3)

    def _style_psd_axis(self) -> None:
        self._ax_psd.set_ylabel('PSD (V²/Hz)')
        self._ax_psd.set_xlabel('Frequency (Hz)')
        self._ax_psd.set_yscale('log')
        self._ax_psd.set_xscale('log' if self._view.psd_logx else 'linear')
        self._ax_psd.grid(True, which='both', alpha=0.3)

    def assign_colors(self, channel_names: List[str]) -> None:
        """Pre-assign consistent colours to channels from the default cycle."""
        colours = [f'C{i}' for i in range(len(channel_names))]
        self._color_map = {ch: colours[i] for i, ch in enumerate(channel_names)}

    def _label(self, ch: str, d: dict) -> str:
        return f'Ch {ch}  {_fmt_range(d["range_mv"])}'

    # ── user interaction ───────────────────────────────────────────────────────

    def _on_limits_changed(self, ax) -> None:
        """
        Axis limits moved.  If it was not us, the user is driving (toolbar pan,
        box zoom, wheel) and the live limit logic must stop fighting them.
        """
        if self._setting_limits or self.user_zoomed:
            return
        self.user_zoomed = True
        self.zoom_changed.emit(True)

    def reset_view(self) -> None:
        """Give the limits back to the live settings."""
        self.user_zoomed = False
        self._background = None
        self._layout_key = None          # force a clean rebuild
        self.zoom_changed.emit(False)

    def _on_scroll(self, event) -> None:
        """Wheel zoom about the cursor."""
        ax = event.inaxes
        if ax is None:
            return
        factor = 1 / 1.25 if event.button == 'up' else 1.25
        shift  = bool(event.key and 'shift' in event.key)
        zoom_x = (ax is self._ax_psd or not shift)
        zoom_y = (ax is self._ax_psd or shift)

        if zoom_x and event.xdata is not None:
            self._zoom_axis(ax, 'x', event.xdata, factor)
        if zoom_y and event.ydata is not None:
            self._zoom_axis(ax, 'y', event.ydata, factor)

        if not self.user_zoomed:         # _on_limits_changed did not catch a no-op
            self.user_zoomed = True
            self.zoom_changed.emit(True)
        self._background = None
        self.draw_idle()

    def _zoom_axis(self, ax, which: str, center: float, factor: float) -> None:
        """Scale one axis about ``center``, honouring a log scale."""
        get, set_, scale = ((ax.get_xlim, ax.set_xlim, ax.get_xscale())
                            if which == 'x' else
                            (ax.get_ylim, ax.set_ylim, ax.get_yscale()))
        lo, hi = get()
        if scale == 'log':
            if lo <= 0 or hi <= 0 or center <= 0:
                return
            lo, hi, center = np.log10(lo), np.log10(hi), np.log10(center)
        new = (center - (center - lo) * factor, center + (hi - center) * factor)
        if scale == 'log':
            new = (10 ** new[0], 10 ** new[1])
        if new[1] > new[0]:
            set_(*new)

    def _on_draw_event(self, event) -> None:
        """
        A draw we did not initiate (resize, toolbar, matplotlib internals)
        invalidates the cached background, so the next frame rebuilds it.
        """
        if not self._in_own_draw:
            self._background = None

    # ── rendering ──────────────────────────────────────────────────────────────

    def render(self, display_data: Optional[Dict], selected: List[str],
               psd_updated: bool = True) -> None:
        """
        Show a frame.

        ``psd_updated`` False means this frame carries no new spectrum: the PSD
        panel keeps what it has and only the trace is redrawn — the fast path.
        """
        if self._ax_time is None or not display_data:
            return

        has_psd = any(display_data.get(ch, {}).get('ff') is not None
                      for ch in selected)
        enveloped = any(display_data.get(ch, {}).get('lo_mv') is not None
                        for ch in selected)
        # has_psd deliberately stays out of the key: frames alternate between
        # carrying a spectrum and not, and keying on it would force a rebuild on
        # every alternation — exactly the cost blitting is here to avoid.
        key = (tuple(selected), enveloped, self._view.psd_logx,
               self._view.show_peak, self._view.show_grid_divs)
        rebuild = key != self._layout_key or self._background is None

        if not rebuild:
            self._set_trace_data(display_data, selected)
            if psd_updated and has_psd:
                rebuild = self._set_psd_data(display_data, selected)
            rebuild = rebuild or self._limits_need_update(display_data, selected)

        if rebuild:
            self._rebuild(display_data, selected, has_psd, key)
            return

        self.restore_region(self._background)
        self._draw_dynamic()
        self.blit(self.fig.bbox)
        # No flush_events() here: blit() already schedules the repaint, and
        # pumping the event loop mid-render could deliver the next queued frame
        # and re-enter this method.

    def _set_trace_data(self, display_data: Dict, selected: List[str]) -> None:
        for ch in selected:
            d = display_data.get(ch)
            if d is None:
                continue
            band = self._bands.get(ch)
            if band is not None and d.get('lo_mv') is not None:
                band.set_verts([self._band_verts(d)])
            line = self._lines_time.get(ch)
            if line is not None and d.get('trace_mv') is not None:
                line.set_data(d['tt'], d['trace_mv'])

    def _set_psd_data(self, display_data: Dict, selected: List[str]) -> bool:
        """
        Push a new spectrum into the existing artists.

        Returns True if a full rebuild is unavoidable — either an artist does not
        exist yet (first spectrum for a channel) or the new data no longer fits
        the autoscaled y-axis.
        """
        view = self._view
        lo = hi = None
        for ch in selected:
            d = display_data.get(ch)
            if d is None or d.get('ff') is None:
                continue
            line = self._lines_psd.get(ch)
            if line is None:
                return True
            ff, pp = d['ff'], d['pp']
            mask = ff > 0 if view.psd_logx else slice(None)
            line.set_data(ff[mask], pp[mask])
            positive = pp[pp > 0]
            if len(positive):
                a, b = float(positive.min()), float(positive.max())
                lo = a if lo is None else min(lo, a)
                hi = b if hi is None else max(hi, b)
            if view.show_peak and d.get('peak_hz'):
                marker = self._peak_lines.get(ch)
                if marker is None:
                    return True
                marker.set_xdata([d['peak_hz'], d['peak_hz']])

        # Only an autoscaled end can force a rescale; a pinned one never does.
        if self.user_zoomed or lo is None:
            return False
        cur = self._ax_psd.get_ylim()
        if view.psd_ymin <= 0 and (lo < cur[0] or hi > cur[1]):
            return True
        if view.psd_ymax <= 0 and hi > cur[1]:
            return True
        # Decades of headroom left unused — worth one redraw to zoom back in.
        return (view.psd_ymax <= 0 and cur[1] > 0 and hi > 0
                and np.log10(cur[1] / hi) > 2.0)

    def _draw_dynamic(self) -> None:
        """Draw every artist that is not baked into the cached background."""
        for artist in list(self._bands.values()) + list(self._lines_time.values()):
            self._ax_time.draw_artist(artist)
        for artist in list(self._lines_psd.values()) + list(self._peak_lines.values()):
            self._ax_psd.draw_artist(artist)

    def _band_verts(self, d: dict) -> np.ndarray:
        """
        Closed polygon around the min/max band.

        A band from a quiet channel can be thinner than a pixel and would vanish,
        so it is inflated to a fraction of the visible range — enough to read as a
        line, small enough not to misstate the amplitude.
        """
        tt, lo, hi = d['tt'], d['lo_mv'], d['hi_mv']
        span = abs(self._ax_time.get_ylim()[1] - self._ax_time.get_ylim()[0])
        pad  = 0.002 * span
        if pad > 0:
            mid = 0.5 * (lo + hi)
            thin = (hi - lo) < 2 * pad
            lo = np.where(thin, mid - pad, lo)
            hi = np.where(thin, mid + pad, hi)
        return np.concatenate([np.column_stack([tt, hi]),
                               np.column_stack([tt[::-1], lo[::-1]])])

    def _rebuild(self, display_data: Dict, selected: List[str],
                 has_psd: bool, key: tuple) -> None:
        """Full redraw: recreate the artists, set the limits, cache the background."""
        # cla() resets the limits, so a rebuild would silently throw away a zoom
        # the user is holding.  Capture it first and put it back at the end.
        held = None
        if self.user_zoomed:
            held = (self._ax_time.get_xlim(), self._ax_time.get_ylim(),
                    self._ax_psd.get_xlim(),  self._ax_psd.get_ylim())

        self._ax_time.cla()
        self._lines_time.clear()
        self._bands.clear()
        max_range = 0
        drawn = False
        for ch in selected:
            d = display_data.get(ch)
            if d is None:
                continue
            colour = self._color_map.get(ch)
            label  = self._label(ch, d)
            if d.get('lo_mv') is not None:
                # Filled band, no edge: stroking the outline of a 2000-bin
                # envelope costs ~30x more than filling it.
                self._bands[ch] = self._ax_time.fill_between(
                    d['tt'], d['lo_mv'], d['hi_mv'], lw=0,
                    facecolor=colour, label=label)
            else:
                (line,) = self._ax_time.plot(d['tt'], d['trace_mv'], label=label,
                                             color=colour, lw=0.8)
                self._lines_time[ch] = line
            max_range = max(max_range, d['range_mv'])
            drawn = True
        self._style_time_axis()
        self._apply_time_limits(display_data, selected, max_range)
        # The pad in _band_verts depends on the y-range, so refresh the vertices
        # now that the limits for this frame are settled.
        self._set_trace_data(display_data, selected)
        if drawn:
            self._ax_time.legend(fontsize=8, loc='upper right', framealpha=0.7)

        if has_psd:
            self._render_psd(display_data, selected)
        else:
            self._ax_psd.set_xscale('log' if self._view.psd_logx else 'linear')

        if held is not None:
            self._setting_limits = True
            try:
                self._ax_time.set_xlim(held[0]); self._ax_time.set_ylim(held[1])
                self._ax_psd.set_xlim(held[2]);  self._ax_psd.set_ylim(held[3])
            finally:
                self._setting_limits = False

        # Capture the background with every data artist hidden, so restoring it
        # later cannot leave a stale trace or spectrum behind.
        dynamic = (list(self._bands.values()) + list(self._lines_time.values())
                   + list(self._lines_psd.values()) + list(self._peak_lines.values()))
        for artist in dynamic:
            artist.set_visible(False)
        self._in_own_draw = True
        try:
            self.draw()
            self._background = self.copy_from_bbox(self.fig.bbox)
        finally:
            self._in_own_draw = False
        for artist in dynamic:
            artist.set_visible(True)
        self._draw_dynamic()
        self.blit(self.fig.bbox)
        self._layout_key = key

    def _limits_need_update(self, display_data: Dict, selected: List[str]) -> bool:
        """
        True when the current axes can no longer show the data honestly.

        Only the autoscaling modes can need this; fixed modes never move.  A
        margin of hysteresis keeps a noisy signal from triggering a full redraw
        every single frame.
        """
        view = self._view
        if self.user_zoomed:
            return False

        meta = display_data.get('_meta') or {}
        if view.time_per_div > 0 and meta.get('graticule'):
            want = self._graticule_xlim(meta)
            cur  = self._ax_time.get_xlim()
            if abs(cur[0] - want[0]) > 1e-12 or abs(cur[1] - want[1]) > 1e-12:
                return True
        else:
            lo, hi = self._data_range(display_data, selected, 'tt')
            cur = self._ax_time.get_xlim()
            if lo is not None and (lo < cur[0] or hi > cur[1]):
                return True

        if view.ylim_mode != 'auto':
            return False
        lo, hi = self._data_range(display_data, selected, 'trace_mv')
        if lo is None:
            return False
        cur  = self._ax_time.get_ylim()
        span = cur[1] - cur[0]
        if lo < cur[0] or hi > cur[1]:
            return True
        return span > 0 and (hi - lo) < 0.4 * span      # zoomed way out

    @staticmethod
    def _data_range(display_data: Dict, selected: List[str], field: str):
        """
        Extent of the plotted data across channels.

        For ``trace_mv`` this also covers frames that carry a min/max band
        instead of a single-valued trace.
        """
        lo = hi = None
        for ch in selected:
            d = display_data.get(ch)
            if d is None:
                continue
            if field == 'trace_mv' and d.get('lo_mv') is not None:
                arrays = (d['lo_mv'], d['hi_mv'])
            elif d.get(field) is not None:
                arrays = (d[field],)
            else:
                continue
            if not len(arrays[0]):
                continue
            a = float(min(np.min(x) for x in arrays))
            b = float(max(np.max(x) for x in arrays))
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
        return lo, hi

    def _graticule_xlim(self, meta: Dict) -> Tuple[float, float]:
        """
        Time-axis limits for a graticule frame.

        Normally the requested window (n_div × s/div) is used, which lands the
        divisions on exact values — the frame's own span is a hair shorter, being
        rounded to whole samples.  But a frame replayed from history may have been
        captured at a different timebase, and pinning that to the current window
        would crop it, so its actual span wins when the two genuinely differ.
        """
        span  = self._view.time_per_div * self._view.n_div_x
        frame = float(meta.get('time_span') or 0.0)
        if frame > 0 and abs(frame - span) > 0.02 * max(span, 1e-12):
            span = frame
        return (-span, 0.0) if meta.get('roll') else (0.0, span)

    def _apply_time_limits(self, display_data: Dict, selected: List[str],
                           max_range_mv: int) -> None:
        """
        Vertical range and the scope graticule for the time panel.

        The x-axis is pinned to n_div_x × time_per_div only for live windows of
        exactly that length; a whole-file frame (which is far longer than the
        division setting) is left to autoscale so nothing gets clipped.
        """
        view = self._view
        if self.user_zoomed:
            return
        self._setting_limits = True
        try:
            v = view.ylim_mode
            if v == 'full range' and max_range_mv:
                self._ax_time.set_ylim(-max_range_mv, max_range_mv)
            elif v == 'manual' and view.y_max > view.y_min:
                self._ax_time.set_ylim(view.y_min, view.y_max)
            elif v == 'mV/div' and view.mv_per_div > 0:
                half = view.mv_per_div * view.n_div_y / 2.0
                self._ax_time.set_ylim(-half, half)
                if view.show_grid_divs:
                    self._ax_time.yaxis.set_major_locator(
                        MultipleLocator(view.mv_per_div))
            elif v == 'auto':
                lo, hi = self._data_range(display_data, selected, 'trace_mv')
                if lo is not None:
                    pad = 0.15 * (hi - lo) or max(abs(hi), 1.0) * 0.1
                    self._ax_time.set_ylim(lo - pad, hi + pad)

            meta = display_data.get('_meta') or {}
            if view.time_per_div > 0 and meta.get('graticule'):
                self._ax_time.set_xlim(*self._graticule_xlim(meta))
                if view.show_grid_divs:
                    self._ax_time.xaxis.set_major_locator(
                        MultipleLocator(view.time_per_div))
        finally:
            self._setting_limits = False

    def _render_psd(self, display_data: Dict, selected: List[str]) -> None:
        self._ax_psd.cla()
        self._lines_psd.clear()
        self._peak_lines.clear()
        drawn = False
        for ch in selected:
            d = display_data.get(ch)
            if d is None or d.get('ff') is None:
                continue
            ff, pp = d['ff'], d['pp']
            mask = ff > 0 if self._view.psd_logx else slice(None)
            # The label stays free of live numbers so the legend can live in the
            # cached background; the peak frequency is in the readout strip.
            (line,) = self._ax_psd.plot(ff[mask], pp[mask], label=self._label(ch, d),
                                        color=self._color_map.get(ch), lw=0.8)
            self._lines_psd[ch] = line
            if self._view.show_peak and d.get('peak_hz'):
                self._peak_lines[ch] = self._ax_psd.axvline(
                    d['peak_hz'], color=self._color_map.get(ch),
                    ls=':', lw=0.7, alpha=0.7)
            drawn = True
        self._style_psd_axis()
        if drawn and not self.user_zoomed:
            view = self._view
            self._setting_limits = True
            try:
                left  = view.f_min if view.f_min > 0 else None
                right = view.f_max if view.f_max > 0 else None
                if left is not None or right is not None:
                    self._ax_psd.set_xlim(left=left, right=right)
                # The PSD y-axis is logarithmic, so only strictly positive limits
                # can be applied; 0 means "leave this end on autoscale".
                bottom = view.psd_ymin if view.psd_ymin > 0 else None
                top    = view.psd_ymax if view.psd_ymax > 0 else None
                if bottom is not None or top is not None:
                    self._ax_psd.set_ylim(bottom=bottom, top=top)
            finally:
                self._setting_limits = False
        if drawn:
            self._ax_psd.legend(fontsize=8, loc='upper right', framealpha=0.7)


# ── Reading files back ─────────────────────────────────────────────────────────

DATA_SUFFIXES = ('.h5', '.hdf5')

# ps4000a full-scale count.  The files record only mV-per-count, so the input
# range a channel was digitized on has to be recovered from that.
_MAX_ADC = 32767.0


def _range_idx_from_adc2mv(adc2mv: float) -> int:
    """Nearest input-range index to a stored mV-per-count factor."""
    if not np.isfinite(adc2mv) or adc2mv <= 0:
        return len(CHANNEL_INPUT_RANGES_MV) - 1
    return int(np.argmin(np.abs(CHANNEL_INPUT_RANGES_MV - adc2mv * _MAX_ADC)))


def _attr_value(v):
    """Make an HDF5 attribute printable: bytes → str, numpy scalars → Python."""
    if isinstance(v, bytes):
        return v.decode('utf-8', 'replace')
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.item() if v.ndim == 0 else v.tolist()
    return v


def _natural_key(name: str) -> list:
    """Sort key that orders ..._2.h5 before ..._10.h5."""
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r'(\d+)', name)]


@dataclass
class Capture:
    """
    One recorded file, or one slice of it, read back off disk.

    ``data`` holds only the requested segment — the sample offset it starts at is
    ``i_start`` — while ``n_total`` is the length of the file.  Keeping those
    separate is what lets the inspector page through a multi-gigabyte record
    without ever materialising the whole thing.
    """
    path:       str
    dt:         float
    channels:   List[ChannelConfig]
    adc2mvs:    np.ndarray
    data:       np.ndarray            # (n_channels, n_segment) int16
    attrs:      Dict[str, object]     # /data attributes, printable values
    means_mv:   Dict[str, float]      # channels written as a mean only
    n_total:    int                   # samples in the file
    i_start:    int                   # first sample of the segment
    size_bytes: int

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    @property
    def t_start(self) -> float:
        return self.i_start * self.dt

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1]) if self.data.ndim == 2 else 0

    @property
    def segment_duration(self) -> float:
        return self.n_samples * self.dt

    @property
    def duration(self) -> float:
        return self.n_total * self.dt


def read_capture(path: str, t_start: float = 0.0, t_length: float = 0.0) -> Capture:
    """
    Read a DAQ file written by :meth:`PicoscopeDAQ.write_hdf5`.

    ``t_length`` of 0 means the rest of the file.  The segment is sliced in h5py,
    so asking for a second out of a ten-second record reads a tenth of the bytes
    rather than loading it all and throwing most of it away.

    Files written by the older scripts put the same datasets at the root instead
    of under /data, so both layouts are accepted.
    """
    with h5py.File(path, 'r') as f:
        g = f['data'] if 'data' in f else f
        attrs = {k: _attr_value(v) for k, v in g.attrs.items()}

        dt = float(attrs.get('delta_t') or 0.0)
        if dt <= 0:
            raise ValueError(
                "no usable 'delta_t' attribute — this does not look like a "
                "DAQ capture")

        names, dsets = [], []
        for key in sorted(g.keys()):
            obj = g[key]
            if key.startswith('channel_') and isinstance(obj, h5py.Dataset):
                names.append(key.split('_', 1)[1].upper())
                dsets.append(obj)

        n_total = min((int(d.shape[0]) for d in dsets), default=0)
        i0 = min(max(int(round(t_start / dt)), 0), max(n_total - 1, 0))
        n  = (n_total - i0 if t_length <= 0
              else min(n_total - i0, max(1, int(round(t_length / dt)))))

        data     = np.empty((len(dsets), n), dtype=np.int16)
        adc2mvs  = np.ones(len(dsets), dtype=float)
        channels: List[ChannelConfig] = []
        for i, (name, ds) in enumerate(zip(names, dsets)):
            data[i] = ds[i0:i0 + n]
            scale = float(_attr_value(ds.attrs.get('adc2mv', 1.0)) or 1.0)
            adc2mvs[i] = scale
            channels.append(ChannelConfig(
                name=name, range_idx=_range_idx_from_adc2mv(scale)))

    means = {k[len('channel_'):-len('_mean_mv')].upper(): float(v)
             for k, v in attrs.items()
             if k.startswith('channel_') and k.endswith('_mean_mv')}

    return Capture(path=path, dt=dt, channels=channels, adc2mvs=adc2mvs,
                   data=data, attrs=attrs, means_mv=means, n_total=n_total,
                   i_start=i0, size_bytes=os.path.getsize(path))


class CaptureWorker(QThread):
    """
    Reads a file and prepares its display frame off the GUI thread.

    Both halves are slow enough to matter — a full-length record is tens of
    megabytes per channel to read and a Welch transform over it costs about a
    second — and neither should freeze the window while the user is arrowing
    through a directory.

    ``capture`` skips the read: re-analysing a file already in memory (new
    nperseg, envelope off, more points) reuses the samples and only recomputes.
    Each request carries an id, so a result that arrives after the user has moved
    on can be recognised as stale and dropped.
    """
    ready  = pyqtSignal(int, object, object)   # req_id, Capture, display dict
    failed = pyqtSignal(int, str)

    def __init__(self, req_id: int, path: str, capture: Optional[Capture],
                 t_start: float, t_length: float, opts: DisplayOptions):
        super().__init__()
        self._req_id  = req_id
        self._path    = path
        self._capture = capture
        self._t_start = t_start
        self._length  = t_length
        self._opts    = opts

    def run(self) -> None:
        try:
            cap = self._capture or read_capture(self._path, self._t_start,
                                                self._length)
            display: Dict = {}
            if cap.channels and cap.n_samples:
                display = prepare_display(cap.data, cap.dt, cap.adc2mvs,
                                          cap.channels, self._opts)
                # prepare_display times a frame from its own first sample; here
                # the segment may start anywhere in the file, and the axis should
                # read as time from the start of the recording.
                for ch, d in display.items():
                    if not ch.startswith('_') and len(d['tt']):
                        d['tt'] = d['tt'] + cap.t_start
                display['_meta']['graticule'] = False
                display['_meta']['roll']      = False
                display['_meta']['t_start']   = cap.t_start
            self.ready.emit(self._req_id, cap, display)
        except Exception as exc:
            self.failed.emit(self._req_id, f'{type(exc).__name__}: {exc}')


class InspectorPanel(QWidget):
    """
    Quick look at files already on disk — browse a directory, plot a capture in
    time and frequency, read its metadata.

    It is deliberately the same plot code as the live view (prepare_display +
    DaqPlotCanvas), so a file looks exactly the way it looked while it was being
    taken: same decimation, same envelope, same spectrum, same colours.  What is
    different is that nothing here is live, so the whole record is available and
    the axes never move underneath you.

    Three things make it usable on real files:

    * Reads are sliced.  A segment is fetched with h5py slicing, so a 30-second
      record can be walked a second at a time without loading it whole.
    * Reads and analysis run on a worker (CaptureWorker), one at a time, with the
      newest request winning — holding an arrow key down through a directory
      queues at most one pending load rather than a thread per file.
    * Re-analysis reuses the samples in memory.  Changing nperseg or the trace
      point count does not go back to disk; changing the segment does.

    The metadata tree shows the file's own attributes verbatim (timestamp,
    pressure, and whatever the pulse sequencer wrote) alongside what can be
    derived from them — sample rate, duration, per-channel input range — and the
    statistics of the segment on screen.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._capture:  Optional[Capture] = None
        self._display:  Optional[Dict]    = None
        self._worker:   Optional[CaptureWorker] = None
        self._pending:  Optional[tuple]   = None    # (path, capture)
        self._req_id                      = 0
        self._present:  List[str]         = []      # channels in the loaded file
        self._seeded                      = False   # directory chosen yet?
        self._loading_path: Optional[str] = None

        # Spin boxes fire on every step; a short debounce keeps a click-and-hold
        # on the segment arrows from queueing a read per tick.
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(400)
        self._reload_timer.timeout.connect(lambda: self._request(reload=True))

        self._rescan_timer = QTimer(self)
        self._rescan_timer.setInterval(3000)
        self._rescan_timer.timeout.connect(lambda: self._rescan(quiet=True))

        self._build_ui()

    # ── construction ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self._build_files_group())
        side_layout.addWidget(self._build_segment_group())
        side_layout.addWidget(self._build_analysis_group())
        side_layout.addWidget(self._build_display_group())
        side_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(370)
        scroll.setMaximumWidth(470)

        splitter.addWidget(scroll)
        splitter.addWidget(self._build_plot_side())
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter)

    def _build_files_group(self) -> QGroupBox:
        group  = QGroupBox('Files')
        layout = QGridLayout(group)

        self._dir_edit = QLineEdit('')
        self._dir_edit.setPlaceholderText('directory of .h5 files')
        self._dir_edit.editingFinished.connect(self._on_dir_edited)
        browse = QPushButton('…')
        browse.setMaximumWidth(28)
        browse.clicked.connect(self._on_browse)

        self._filter_edit = QLineEdit('')
        self._filter_edit.setPlaceholderText('name contains… (blank = all)')
        self._filter_edit.textChanged.connect(lambda _t: self._rescan(quiet=True))

        self._file_list = QListWidget()
        self._file_list.setMinimumHeight(180)
        self._file_list.setToolTip(
            'Click or arrow through the list — the selected file is loaded and '
            'plotted.\nSorting is natural, so _2 comes before _10.')
        self._file_list.currentItemChanged.connect(self._on_file_selected)

        refresh_btn = QPushButton('Refresh')
        refresh_btn.clicked.connect(lambda: self._rescan(quiet=False))
        newest_btn  = QPushButton('Newest')
        newest_btn.setToolTip('Select the most recently written file')
        newest_btn.clicked.connect(self._on_newest)

        self._auto_check = QCheckBox('Auto-refresh')
        self._auto_check.setToolTip(
            'Rescan the directory every few seconds — useful while a run is '
            'writing into it.')
        self._auto_check.toggled.connect(self._on_auto_toggled)

        self._count_label = QLabel('')
        self._count_label.setStyleSheet('color: gray;')

        layout.addWidget(QLabel('Directory:'), 0, 0)
        layout.addWidget(self._dir_edit,       0, 1, 1, 2)
        layout.addWidget(browse,               0, 3)
        layout.addWidget(QLabel('Filter:'),    1, 0)
        layout.addWidget(self._filter_edit,    1, 1, 1, 3)
        layout.addWidget(self._file_list,      2, 0, 1, 4)
        layout.addWidget(refresh_btn,          3, 0)
        layout.addWidget(newest_btn,           3, 1)
        layout.addWidget(self._auto_check,     3, 2, 1, 2)
        layout.addWidget(self._count_label,    4, 0, 1, 4)
        return group

    def _build_segment_group(self) -> QGroupBox:
        group  = QGroupBox('Segment')
        layout = QGridLayout(group)

        self._whole_check = QCheckBox('Whole file')
        self._whole_check.setChecked(True)
        self._whole_check.setToolTip(
            'Untick to read a window out of the file instead — the read is '
            'sliced,\nso stepping through a long record stays quick.')
        self._whole_check.toggled.connect(self._on_whole_toggled)

        self._seg_start = QDoubleSpinBox()
        self._seg_start.setDecimals(4)
        self._seg_start.setRange(0.0, 1e7)
        self._seg_start.setSuffix(' s')
        self._seg_start.setToolTip('Start of the window, seconds from the '
                                   'beginning of the file')

        self._seg_length = QDoubleSpinBox()
        self._seg_length.setDecimals(4)
        self._seg_length.setRange(1e-4, 1e7)
        self._seg_length.setValue(1.0)
        self._seg_length.setSuffix(' s')
        self._seg_length.setToolTip('Length of the window')

        for spin in (self._seg_start, self._seg_length):
            spin.setEnabled(False)
            spin.valueChanged.connect(lambda _v: self._reload_timer.start())

        step_back = QPushButton('◀ prev')
        step_fwd  = QPushButton('next ▶')
        step_back.setToolTip('Move the window back by its own length')
        step_fwd.setToolTip('Move the window forward by its own length')
        step_back.clicked.connect(lambda: self._step_segment(-1))
        step_fwd.clicked.connect(lambda: self._step_segment(+1))
        self._seg_buttons = (step_back, step_fwd)
        for b in self._seg_buttons:
            b.setEnabled(False)

        self._segment_label = QLabel('no file loaded')
        self._segment_label.setWordWrap(True)
        self._segment_label.setStyleSheet('color: gray;')

        layout.addWidget(self._whole_check,     0, 0, 1, 2)
        layout.addWidget(QLabel('Start:'),      1, 0)
        layout.addWidget(self._seg_start,       1, 1)
        layout.addWidget(QLabel('Length:'),     2, 0)
        layout.addWidget(self._seg_length,      2, 1)
        layout.addWidget(step_back,             3, 0)
        layout.addWidget(step_fwd,              3, 1)
        layout.addWidget(self._segment_label,   4, 0, 1, 2)
        return group

    def _build_analysis_group(self) -> QGroupBox:
        """Options that change the numbers — each one re-runs the analysis."""
        group  = QGroupBox('Analysis')
        layout = QGridLayout(group)

        self._i_points = QSpinBox()
        self._i_points.setRange(200, 200000)
        self._i_points.setSingleStep(500)
        self._i_points.setValue(4000)
        self._i_points.setToolTip('Target points in the time panel')

        self._i_envelope = QCheckBox('Min/max envelope')
        self._i_envelope.setChecked(True)
        self._i_envelope.setToolTip(
            'Decimate by min/max per bin, so a fast oscillation keeps its true '
            'amplitude')

        self._i_psd_check = QCheckBox('Compute PSD')
        self._i_psd_check.setChecked(True)

        self._i_nperseg = QComboBox()
        self._i_nperseg.addItems(NPERSEG_OPTIONS)
        self._i_nperseg.setToolTip(
            'Welch segment length — longer resolves narrower peaks, shorter '
            'averages more')

        self._i_detrend = QCheckBox('Detrend PSD')
        self._i_detrend.setChecked(True)

        self._i_psd_points = QSpinBox()
        self._i_psd_points.setRange(0, 200000)
        self._i_psd_points.setSingleStep(500)
        self._i_psd_points.setValue(3000)
        self._i_psd_points.setSpecialValueText('all')
        self._i_psd_points.setToolTip(
            'Points drawn in the PSD panel (0 = every bin).  Thinning keeps the '
            'maximum\nof each group, so peaks survive but the floor reads a few '
            'dB high.')

        for w in (self._i_points, self._i_psd_points):
            w.valueChanged.connect(lambda _v: self._request())
        for w in (self._i_envelope, self._i_psd_check, self._i_detrend):
            w.toggled.connect(lambda _c: self._request())
        self._i_nperseg.currentIndexChanged.connect(lambda _i: self._request())

        layout.addWidget(QLabel('Trace points:'), 0, 0)
        layout.addWidget(self._i_points,          0, 1)
        layout.addWidget(self._i_envelope,        1, 0, 1, 2)
        layout.addWidget(self._i_psd_check,       2, 0, 1, 2)
        layout.addWidget(QLabel('nperseg:'),      3, 0)
        layout.addWidget(self._i_nperseg,         3, 1)
        layout.addWidget(QLabel('PSD points:'),   4, 0)
        layout.addWidget(self._i_psd_points,      4, 1)
        layout.addWidget(self._i_detrend,         5, 0, 1, 2)
        return group

    def _build_display_group(self) -> QGroupBox:
        """Drawing-only options — applied to the frame already in hand."""
        group  = QGroupBox('Display')
        layout = QGridLayout(group)

        self._i_ylim = QComboBox()
        self._i_ylim.addItems(YLIM_OPTIONS)
        self._i_ylim.setToolTip(
            'Time-panel vertical range:\n'
            '  auto        — autoscale to the data\n'
            '  full range  — ±(channel input range)\n'
            '  manual      — the min/max below\n'
            '  mV/div      — symmetric about 0, scope style')
        self._i_ylim.currentIndexChanged.connect(self._on_ylim_mode_changed)

        self._i_ymin = QLineEdit('-1000')
        self._i_ymax = QLineEdit('1000')
        self._i_mvdiv = QDoubleSpinBox()
        self._i_mvdiv.setDecimals(3)
        self._i_mvdiv.setRange(1e-3, 20000.0)
        self._i_mvdiv.setValue(500.0)
        self._i_mvdiv.setSuffix(' mV/div')
        self._i_mvdiv.valueChanged.connect(lambda _v: self._render())

        self._i_logx = QCheckBox('Log frequency axis')
        self._i_fmin = QLineEdit('')
        self._i_fmax = QLineEdit('')
        self._i_pmin = QLineEdit('')
        self._i_pmax = QLineEdit('')
        for w, tip in ((self._i_fmin, 'Lower frequency limit (blank = auto)'),
                       (self._i_fmax, 'Upper frequency limit (blank = auto)'),
                       (self._i_pmin, 'Lower PSD limit in V²/Hz (blank = auto)'),
                       (self._i_pmax, 'Upper PSD limit in V²/Hz (blank = auto)')):
            w.setPlaceholderText('auto')
            w.setToolTip(tip)
        self._i_peak = QCheckBox('Mark PSD peak')
        self._i_peak.setChecked(True)

        for w in (self._i_ymin, self._i_ymax, self._i_fmin, self._i_fmax,
                  self._i_pmin, self._i_pmax):
            w.editingFinished.connect(self._render)
        for w in (self._i_logx, self._i_peak):
            w.toggled.connect(lambda _c: self._render())

        layout.addWidget(QLabel('Vertical:'),          0, 0)
        layout.addWidget(self._i_ylim,                 0, 1, 1, 2)
        layout.addWidget(QLabel('min / max (mV):'),    1, 0)
        layout.addWidget(self._i_ymin,                 1, 1)
        layout.addWidget(self._i_ymax,                 1, 2)
        layout.addWidget(QLabel('Scale:'),             2, 0)
        layout.addWidget(self._i_mvdiv,                2, 1, 1, 2)
        layout.addWidget(QLabel('f range (Hz):'),      3, 0)
        layout.addWidget(self._i_fmin,                 3, 1)
        layout.addWidget(self._i_fmax,                 3, 2)
        layout.addWidget(QLabel('PSD range (V²/Hz):'), 4, 0)
        layout.addWidget(self._i_pmin,                 4, 1)
        layout.addWidget(self._i_pmax,                 4, 2)
        layout.addWidget(self._i_logx,                 5, 0, 1, 3)
        layout.addWidget(self._i_peak,                 6, 0, 1, 3)
        self._on_ylim_mode_changed()
        return group

    def _build_plot_side(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Channel selector row
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(4, 2, 4, 2)
        top_layout.addWidget(QLabel('Display:'))
        self._i_checks: Dict[str, QCheckBox] = {}
        for ch in ALL_CHANNELS:
            cb = QCheckBox(ch)
            cb.setVisible(False)
            cb.toggled.connect(lambda _c: self._render())
            self._i_checks[ch] = cb
            top_layout.addWidget(cb)
        top_layout.addStretch()

        self._i_reset_btn = QPushButton('Reset view')
        self._i_reset_btn.setEnabled(False)
        self._i_reset_btn.setToolTip(
            'Hand the axis limits back to the settings on the left.\n'
            'Wheel zooms about the cursor (Shift = vertical in the time panel).')
        self._i_reset_btn.clicked.connect(self._on_reset_view)
        top_layout.addWidget(self._i_reset_btn)

        export_btn = QPushButton('Export segment…')
        export_btn.setToolTip(
            'Write the segment on screen to a .npz — raw counts per channel plus '
            'adc2mv,\ndelta_t and the file attributes, ready to load in a '
            'notebook.')
        export_btn.clicked.connect(self._on_export)
        top_layout.addWidget(export_btn)

        self._i_canvas = DaqPlotCanvas()
        self._i_canvas.zoom_changed.connect(self._on_zoom_changed)
        toolbar = NavigationToolbar2QT(self._i_canvas, self)

        self._i_measure = QLabel('')
        self._i_measure.setTextFormat(Qt.RichText)
        self._i_measure.setStyleSheet('font-family: monospace;')

        plot_side = QWidget()
        plot_layout = QVBoxLayout(plot_side)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(top)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self._i_canvas, 1)
        plot_layout.addWidget(self._i_measure)

        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(plot_side)
        vsplit.addWidget(self._build_metadata_widget())
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([620, 280])      # metadata deep enough to read unaided
        layout.addWidget(vsplit)

        self._i_status = QLabel('')
        self._i_status.setStyleSheet('color: gray;')
        layout.addWidget(self._i_status)
        return widget

    def _build_metadata_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        header.addWidget(QLabel('<b>Metadata</b>'))
        header.addStretch()
        copy_btn = QPushButton('Copy')
        copy_btn.setToolTip('Copy the whole tree to the clipboard')
        copy_btn.clicked.connect(self._on_copy_metadata)
        header.addWidget(copy_btn)
        layout.addLayout(header)

        self._meta_tree = QTreeWidget()
        self._meta_tree.setColumnCount(2)
        self._meta_tree.setHeaderLabels(['Field', 'Value'])
        self._meta_tree.setAlternatingRowColors(True)
        self._meta_tree.setRootIsDecorated(True)
        self._meta_tree.setMinimumHeight(120)
        layout.addWidget(self._meta_tree)
        return widget

    # ── directory browsing ─────────────────────────────────────────────────────

    def note_output_dir(self, path: str) -> None:
        """
        Offer the recording directory to the browser.

        Used when the Inspect tab is first opened and when a run finishes: the
        files you want to look at are almost always the ones just written.  A
        directory the user has typed or browsed to is never overwritten.

        The offer only sticks if it exists — opening the tab before the output
        path has been set would otherwise pin the browser to a default drive that
        is not there, and the real directory would never take its place.
        """
        path = (path or '').strip()
        if not path:
            return
        current = self._dir_edit.text().strip()
        if not self._seeded:
            self._dir_edit.setText(path)
            self._seeded = os.path.isdir(path)
            self._rescan(quiet=True)
        elif current and os.path.normpath(current) == os.path.normpath(path):
            self._rescan(quiet=True)

    def _on_browse(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, 'Select data directory', self._dir_edit.text())
        if d:
            self._dir_edit.setText(d)
            self._seeded = True
            self._rescan(quiet=False)

    def _on_dir_edited(self) -> None:
        self._seeded = True
        self._rescan(quiet=False)

    def _on_auto_toggled(self, on: bool) -> None:
        if on:
            self._rescan_timer.start()
        else:
            self._rescan_timer.stop()

    def _rescan(self, quiet: bool = True) -> None:
        """
        Re-list the directory, keeping the selection where it is.

        Called on a timer while auto-refresh is on, so it must be cheap and must
        not disturb what the user is looking at — the list is only rebuilt when
        the set of names actually changed.
        """
        directory = self._dir_edit.text().strip()
        names: List[str] = []
        if directory and os.path.isdir(directory):
            needle = self._filter_edit.text().strip().lower()
            try:
                names = [n for n in os.listdir(directory)
                         if n.lower().endswith(DATA_SUFFIXES)
                         and (not needle or needle in n.lower())]
            except OSError as exc:
                self._set_status(f'cannot list directory: {exc}')
                return
            names.sort(key=_natural_key)
        elif directory and not quiet:
            self._set_status('not a directory')

        existing = [self._file_list.item(i).text()
                    for i in range(self._file_list.count())]
        if names != existing:
            current = self._file_list.currentItem()
            keep    = current.text() if current else None
            self._file_list.blockSignals(True)
            self._file_list.clear()
            for n in names:
                self._file_list.addItem(QListWidgetItem(n))
            # The selection is restored with signals still blocked: the file it
            # names is already loaded, and a rescan (auto-refresh fires one every
            # few seconds during a run) must not reload it.
            if keep in names:
                self._file_list.setCurrentRow(names.index(keep))
            self._file_list.blockSignals(False)

        self._count_label.setText(
            f'{len(names)} file(s)' if names else 'no .h5 files here')

    def _on_newest(self) -> None:
        """Select the most recently modified file in the list."""
        directory = self._dir_edit.text().strip()
        best_row, best_mtime = None, None
        for row in range(self._file_list.count()):
            path = os.path.join(directory, self._file_list.item(row).text())
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if best_mtime is None or mtime > best_mtime:
                best_row, best_mtime = row, mtime
        if best_row is not None:
            self._file_list.setCurrentRow(best_row)

    def _on_file_selected(self, current, _previous) -> None:
        if current is None:
            return
        path = os.path.join(self._dir_edit.text().strip(), current.text())
        self._request(path=path, reload=True)

    # ── segment ────────────────────────────────────────────────────────────────

    def _on_whole_toggled(self, whole: bool) -> None:
        for w in (self._seg_start, self._seg_length) + self._seg_buttons:
            w.setEnabled(not whole)
        self._request(reload=True)

    def _step_segment(self, direction: int) -> None:
        length = self._seg_length.value()
        start  = max(0.0, self._seg_start.value() + direction * length)
        if self._capture is not None:
            start = min(start, max(0.0, self._capture.duration - length))
        self._seg_start.setValue(start)      # triggers the debounced reload

    def _segment_request(self) -> Tuple[float, float]:
        if self._whole_check.isChecked():
            return 0.0, 0.0
        return self._seg_start.value(), self._seg_length.value()

    # ── analysis / rendering ───────────────────────────────────────────────────

    def _display_options(self) -> DisplayOptions:
        nperseg_text = self._i_nperseg.currentText()
        return DisplayOptions(
            n_points=self._i_points.value(),
            psd_points=self._i_psd_points.value(),
            envelope=self._i_envelope.isChecked(),
            nperseg=0 if nperseg_text == 'auto' else int(nperseg_text),
            detrend=self._i_detrend.isChecked(),
            time_span=0.0,          # the segment is already the window
            psd_span=0.0,
            want_psd=self._i_psd_check.isChecked(),
            roll=False,
        )

    def _view_options(self) -> ViewOptions:
        def parse(edit: QLineEdit, default: float = 0.0) -> float:
            try:
                return float(edit.text())
            except ValueError:
                return default
        return ViewOptions(
            psd_logx=self._i_logx.isChecked(),
            f_min=parse(self._i_fmin),
            f_max=parse(self._i_fmax),
            psd_ymin=parse(self._i_pmin),
            psd_ymax=parse(self._i_pmax),
            ylim_mode=self._i_ylim.currentText(),
            y_min=parse(self._i_ymin, -1000.0),
            y_max=parse(self._i_ymax, 1000.0),
            mv_per_div=self._i_mvdiv.value(),
            n_div_y=8,
            time_per_div=0.0,       # a file is not a graticule window
            show_grid_divs=False,
            show_peak=self._i_peak.isChecked(),
            show_rms=True,
        )

    def _on_ylim_mode_changed(self) -> None:
        mode = self._i_ylim.currentText()
        self._i_ymin.setEnabled(mode == 'manual')
        self._i_ymax.setEnabled(mode == 'manual')
        self._i_mvdiv.setEnabled(mode == 'mV/div')
        self._render()

    def _request(self, path: Optional[str] = None, reload: bool = False) -> None:
        """
        Ask for a new frame.

        ``reload`` goes back to disk; without it the samples already in memory
        are re-analysed, which is what every option on the Analysis group needs.
        Only one request is ever queued: while a read is in flight, a newer
        request replaces the pending one instead of starting a second thread.
        """
        self._reload_timer.stop()
        current = self._capture.path if self._capture else None
        target  = path or self._loading_path or current
        if not target:
            return
        # Samples can only be reused if they are the samples of the file being
        # asked for — including when a read of a different file is still in
        # flight, where what is in memory belongs to the file before it.
        if target != current:
            reload = True
        capture = None if reload else self._capture
        if capture is None:
            self._loading_path = target
            self._set_status(f'loading {os.path.basename(target)}…')

        self._req_id += 1
        self._pending = (target, capture)
        self._start_pending()

    def _start_pending(self) -> None:
        if self._pending is None:
            return
        if self._worker is not None and self._worker.isRunning():
            return                       # picked up again from finished()
        path, capture = self._pending
        self._pending = None
        t_start, t_length = self._segment_request()
        worker = CaptureWorker(self._req_id, path, capture, t_start, t_length,
                               self._display_options())
        worker.ready.connect(self._on_ready)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._start_pending)
        self._worker = worker
        worker.start()

    def _on_ready(self, req_id: int, capture: Capture, display: Dict) -> None:
        if req_id != self._req_id:
            return                       # the user has already moved on
        self._capture = capture
        self._display = display or None
        self._loading_path = None
        self._sync_channel_checks(capture)
        self._sync_segment_ui(capture)
        self._fill_metadata(capture)
        self._render()
        self._set_status(
            f'{os.path.basename(capture.path)} — '
            f'{capture.n_samples:,} samples of {capture.n_total:,} at '
            f'{_fmt_rate(capture.fs)}')

    def _on_failed(self, req_id: int, message: str) -> None:
        if req_id != self._req_id:
            return
        self._loading_path = None
        self._set_status(f'could not read the file: {message}')

    def _sync_channel_checks(self, capture: Capture) -> None:
        """
        Show a checkbox per channel in the file, ticked when it first appears.

        Which channels exist is tracked here rather than read back from the
        widgets: a panel sitting behind the Live tab is not "visible" in Qt's
        sense, and asking the checkboxes would then plot nothing.
        """
        names = [ch.name for ch in capture.channels]
        self._i_canvas.assign_colors(names)
        for ch, cb in self._i_checks.items():
            present = ch in names
            cb.setVisible(present)
            if present and ch not in self._present:
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
        self._present = names

    def _sync_segment_ui(self, capture: Capture) -> None:
        """
        Refresh the segment controls for the file just loaded.

        In windowed mode the spin boxes are the authority and are left alone:
        writing the loaded segment back into them would undo an edit the user
        made while the previous read was still in flight (the read is debounced,
        so that overlap is the normal case, not a rare one).  Only whole-file
        mode fills them in, which also gives a sensible window to start from when
        the mode is switched off.
        """
        for w in (self._seg_start, self._seg_length):
            w.blockSignals(True)
        self._seg_start.setMaximum(max(capture.duration, 1e-4))
        self._seg_length.setMaximum(max(capture.duration, 1e-4))
        if self._whole_check.isChecked():
            self._seg_start.setValue(0.0)
            self._seg_length.setValue(max(capture.duration, 1e-4))
        for w in (self._seg_start, self._seg_length):
            w.blockSignals(False)

        self._segment_label.setText(
            f'file: {_fmt_duration(capture.duration)}, {capture.n_total:,} '
            f'samples, {capture.size_bytes / 1e6:.1f} MB\n'
            f'shown: {_fmt_duration(capture.t_start)} – '
            f'{_fmt_duration(capture.t_start + capture.segment_duration)}')

    def _render(self) -> None:
        if not self._display:
            return
        selected = [ch for ch in self._present if self._i_checks[ch].isChecked()]
        self._i_canvas.set_view_options(self._view_options())
        self._i_canvas.render(self._display, selected, psd_updated=True)
        self._update_measurements(selected)

    def _update_measurements(self, selected: List[str]) -> None:
        parts = []
        for ch in selected:
            d = (self._display or {}).get(ch)
            if d is None:
                continue
            colour = _MPL_COLOURS.get(self._i_canvas._color_map.get(ch, ''), '#888')
            text = (f'<b>{ch}</b> rms {d["rms_mv"]:>8.3g} mV'
                    f'  mean {d["mean_mv"]:>8.3g} mV')
            if d.get('peak_hz'):
                text += f'  pk {_fmt_hz(d["peak_hz"])}'
            parts.append(f'<span style="color:{colour}">{text}</span>')
        self._i_measure.setText('&nbsp;&nbsp;|&nbsp;&nbsp;'.join(parts))

    def _on_zoom_changed(self, zoomed: bool) -> None:
        self._i_reset_btn.setEnabled(zoomed)
        self._i_reset_btn.setText('Reset view ⚠' if zoomed else 'Reset view')

    def _on_reset_view(self) -> None:
        self._i_canvas.reset_view()
        self._render()

    def _set_status(self, text: str) -> None:
        self._i_status.setText(text)

    # ── metadata ───────────────────────────────────────────────────────────────

    def _fill_metadata(self, capture: Capture) -> None:
        """
        Rebuild the tree: what the file says, what follows from it, and what the
        segment on screen measures.
        """
        self._meta_tree.clear()

        try:
            modified = time.strftime('%Y-%m-%d %H:%M:%S',
                                     time.localtime(os.path.getmtime(capture.path)))
        except OSError:
            modified = '—'
        file_rows = [
            ('name',     os.path.basename(capture.path)),
            ('folder',   os.path.dirname(capture.path)),
            ('size',     f'{capture.size_bytes / 1e6:.2f} MB'),
            ('modified', modified),
        ]

        stamp = capture.attrs.get('timestamp')
        capture_rows = [
            ('sample rate',   _fmt_rate(capture.fs)),
            ('delta_t',       f'{capture.dt:.6g} s'),
            ('samples',       f'{capture.n_total:,}'),
            ('duration',      _fmt_duration(capture.duration)),
            ('segment',       f'{capture.n_samples:,} samples, '
                              f'{_fmt_duration(capture.segment_duration)} from '
                              f'{_fmt_duration(capture.t_start)}'),
            ('channels',      ', '.join(ch.name for ch in capture.channels) or '—'),
        ]
        if isinstance(stamp, (int, float)):
            capture_rows.insert(0, (
                'timestamp',
                f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stamp))}'
                f'  ({stamp:.3f})'))

        attr_rows = [(k, self._format_attr(v))
                     for k, v in sorted(capture.attrs.items())]

        channel_items = []
        for i, ch in enumerate(capture.channels):
            d = (self._display or {}).get(ch.name) or {}
            rows = [
                ('input range', _fmt_range(int(CHANNEL_INPUT_RANGES_MV[ch.range_idx]))),
                ('adc2mv',      f'{capture.adc2mvs[i]:.6g} mV/count'),
            ]
            if d:
                rows += [('mean (segment)', f'{d["mean_mv"]:.4g} mV'),
                         ('rms (segment)',  f'{d["rms_mv"]:.4g} mV')]
                if d.get('peak_hz'):
                    rows.append(('PSD peak', _fmt_hz(d['peak_hz'])))
            channel_items.append((f'Channel {ch.name}', rows))
        for name, mean_mv in sorted(capture.means_mv.items()):
            channel_items.append((f'Channel {name} (mean only)',
                                  [('mean', f'{mean_mv:.4g} mV')]))

        for title, rows in [('File', file_rows), ('Capture', capture_rows),
                            ('/data attributes', attr_rows)]:
            self._add_meta_group(title, rows)
        for title, rows in channel_items:
            self._add_meta_group(title, rows)

        self._meta_tree.expandAll()
        self._meta_tree.resizeColumnToContents(0)

    def _add_meta_group(self, title: str, rows: List[Tuple[str, str]]) -> None:
        parent = QTreeWidgetItem([title, ''])
        font = parent.font(0)
        font.setBold(True)
        parent.setFont(0, font)
        for key, value in rows:
            parent.addChild(QTreeWidgetItem([str(key), str(value)]))
        self._meta_tree.addTopLevelItem(parent)

    @staticmethod
    def _format_attr(value) -> str:
        if isinstance(value, float):
            # Enough digits to keep a Unix timestamp's sub-second part.
            return f'{value:.15g}'
        if isinstance(value, (list, tuple)) and len(value) > 8:
            return f'{list(value[:8])}… ({len(value)} values)'
        return str(value)

    def _on_copy_metadata(self) -> None:
        lines = []
        for i in range(self._meta_tree.topLevelItemCount()):
            top = self._meta_tree.topLevelItem(i)
            lines.append(top.text(0))
            for j in range(top.childCount()):
                child = top.child(j)
                lines.append(f'    {child.text(0)}: {child.text(1)}')
        QApplication.clipboard().setText('\n'.join(lines))
        self._set_status('metadata copied to the clipboard')

    # ── export ─────────────────────────────────────────────────────────────────

    def _on_export(self) -> None:
        """
        Write the loaded segment to a .npz — raw counts, not display data.

        The point is to hand a slice straight to a notebook: the arrays are the
        int16 samples as recorded, with the scale factors and attributes needed
        to interpret them, so nothing about the analysis is baked in.
        """
        capture = self._capture
        if capture is None or not capture.channels:
            return
        stem = os.path.splitext(os.path.basename(capture.path))[0]
        default = os.path.join(os.path.dirname(capture.path),
                               f'{stem}_segment.npz')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export segment', default, 'NumPy archive (*.npz)')
        if not path:
            return
        arrays = {f'channel_{ch.name.lower()}': capture.data[i]
                  for i, ch in enumerate(capture.channels)}
        meta = {
            'source':      capture.path,
            'delta_t':     capture.dt,
            't_start':     capture.t_start,
            'i_start':     capture.i_start,
            'n_total':     capture.n_total,
            'adc2mv':      {ch.name.lower(): float(capture.adc2mvs[i])
                            for i, ch in enumerate(capture.channels)},
            'attrs':       capture.attrs,
            'note':        'int16 counts as recorded; multiply by adc2mv for mV',
        }
        try:
            np.savez(path, meta=json.dumps(meta, default=str), **arrays)
        except Exception as exc:
            QMessageBox.critical(self, 'Export failed', str(exc))
            return
        self._set_status(f'exported {os.path.basename(path)}')

    # ── settings / lifecycle ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            'directory':    self._dir_edit.text(),
            'filter':       self._filter_edit.text(),
            'auto_refresh': self._auto_check.isChecked(),
            'whole_file':   self._whole_check.isChecked(),
            'seg_length':   self._seg_length.value(),
            'trace_points': self._i_points.value(),
            'psd_points':   self._i_psd_points.value(),
            'envelope':     self._i_envelope.isChecked(),
            'want_psd':     self._i_psd_check.isChecked(),
            'nperseg':      self._i_nperseg.currentText(),
            'detrend':      self._i_detrend.isChecked(),
            'ylim_mode':    self._i_ylim.currentText(),
            'y_min':        self._i_ymin.text(),
            'y_max':        self._i_ymax.text(),
            'mv_per_div':   self._i_mvdiv.value(),
            'psd_logx':     self._i_logx.isChecked(),
            'f_min':        self._i_fmin.text(),
            'f_max':        self._i_fmax.text(),
            'psd_ymin':     self._i_pmin.text(),
            'psd_ymax':     self._i_pmax.text(),
            'show_peak':    self._i_peak.isChecked(),
        }

    def from_dict(self, d: dict) -> None:
        for key, widget in [('directory', self._dir_edit),
                            ('filter',    self._filter_edit),
                            ('y_min',     self._i_ymin),
                            ('y_max',     self._i_ymax),
                            ('f_min',     self._i_fmin),
                            ('f_max',     self._i_fmax),
                            ('psd_ymin',  self._i_pmin),
                            ('psd_ymax',  self._i_pmax)]:
            if key in d:
                widget.blockSignals(True)
                widget.setText(str(d[key]))
                widget.blockSignals(False)
        for key, widget in [('seg_length',   self._seg_length),
                            ('trace_points', self._i_points),
                            ('psd_points',   self._i_psd_points),
                            ('mv_per_div',   self._i_mvdiv)]:
            if key in d:
                widget.blockSignals(True)
                widget.setValue(d[key])
                widget.blockSignals(False)
        for key, widget in [('nperseg',   self._i_nperseg),
                            ('ylim_mode', self._i_ylim)]:
            if key in d:
                idx = widget.findText(str(d[key]))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
        for key, widget in [('auto_refresh', self._auto_check),
                            ('whole_file',   self._whole_check),
                            ('envelope',     self._i_envelope),
                            ('want_psd',     self._i_psd_check),
                            ('detrend',      self._i_detrend),
                            ('psd_logx',     self._i_logx),
                            ('show_peak',    self._i_peak)]:
            if key in d:
                widget.setChecked(bool(d[key]))
        if d.get('directory'):
            self._seeded = True
        self._on_ylim_mode_changed()
        self._rescan(quiet=True)

    def shutdown(self) -> None:
        self._rescan_timer.stop()
        self._reload_timer.stop()
        self._pending = None
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(5000)


# ── Main window ────────────────────────────────────────────────────────────────

class DaqMainWindow(QMainWindow):
    """
    Mode state machine
    ------------------
    'idle'       connected (or not), nothing streaming
    'monitor'    MonitorWorker running, no files written
    'recording'  AcquisitionWorker running

    Only one worker exists at a time.  Transitions that need the scope idle
    (start recording, apply new channel settings) stop the monitor first and
    queue themselves in ``self._pending``, which is dispatched from the monitor's
    done signal — so the scope is never reconfigured underneath a live stream.

    Display state
    -------------
    ``_live_frame`` is the newest frame received and ``_shown_frame`` is what is
    on screen; they differ while the display is held or scrolled back through
    ``_history``.  Every frame is stored regardless, so going Live lands on the
    present rather than on wherever the display was left.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Picoscope DAQ')
        self.setMinimumSize(1200, 760)

        self._daq:               Optional[PicoscopeDAQ]      = None
        self._monitor:           Optional[MonitorWorker]     = None
        self._worker:            Optional[AcquisitionWorker] = None
        self._live_frame:        Optional[Dict]              = None   # newest received
        self._shown_frame:       Optional[Dict]              = None   # what is on screen
        self._history                                        = FrameHistory(200)
        self._follow_live                                    = True
        self._selected_seq:      Optional[int]               = None
        self._updating_slider                                = False
        self._start_time:        Optional[float]             = None
        self._mode                                           = 'idle'
        self._pending:           Optional[str]               = None
        self._files_written                                  = 0
        self._index_restarted                                = False
        self._syncing_rate                                   = False
        self._config_dirty                                   = False

        self._slow_history                                   = SlowHistory()
        self._slow_log:          Optional[SlowLogWriter]     = None
        self._slow_config:       Optional[SlowConfig]        = None
        self._slow_dirty                                     = False
        self._slow_tracked:      dict                        = {}

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        # The strip chart is redrawn on a timer rather than per row: rows can
        # arrive in bursts, and a full redraw each time would steal the GUI
        # thread from the fast canvas.
        self._slow_timer = QTimer(self)
        self._slow_timer.timeout.connect(self._draw_slow)
        self._slow_timer.start(400)

        self._build_ui()
        self._on_interval_edited()       # seed the rate spin from the default interval
        self._on_timebase_changed()
        self._on_ylim_mode_changed()
        self._update_duration_label()
        self._update_history_ui()
        self._update_buttons()
        if _HAS_PULSES:
            self._update_pulse_summary()

    # ── UI construction ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # Left: config panel (scrollable — there are a lot of knobs)
        config_widget = QWidget()
        cfg_layout = QVBoxLayout(config_widget)
        cfg_layout.addWidget(self._build_hardware_group())
        cfg_layout.addWidget(self._build_channel_group())
        cfg_layout.addWidget(self._build_acquisition_group())
        cfg_layout.addWidget(self._build_output_group())
        cfg_layout.addWidget(self._build_live_group())
        cfg_layout.addWidget(self._build_view_group())

        btn_row = QHBoxLayout()
        for label, slot in [('Save config', self._save_config),
                            ('Load config', self._load_config)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        cfg_layout.addLayout(btn_row)
        cfg_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(config_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        # The pulse controls get their own tab rather than a seventh group: they
        # are a different instrument, and most runs do not use them.
        left = QTabWidget()
        left.addTab(scroll, 'Acquisition')
        self._slow_panel = SlowSettingsPanel()
        self._slow_panel.apply_requested.connect(self._on_slow_apply)
        self._slow_panel.view_changed.connect(self._on_slow_view_changed)
        self._slow_panel.clear_requested.connect(self._on_slow_clear)
        self._slow_panel.new_log_requested.connect(self._on_slow_new_log)
        self._slow_panel.log_toggled.connect(self._sync_slow_log)
        left.addTab(self._slow_panel, 'Slow data')
        if _HAS_PULSES:
            left.addTab(self._build_pulse_tab(), 'Pulses')
        left.setMinimumWidth(430)
        left.setMaximumWidth(500)

        # Right: display selector + dual plot + control bar
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self._build_display_selector())
        self._canvas = DaqPlotCanvas()
        self._canvas.zoom_changed.connect(self._on_zoom_changed)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        right_layout.addWidget(self._toolbar)
        # Fast plots above, slow strip chart below; the chart only takes room
        # once slow data is switched on.
        self._slow_canvas = SlowPlotCanvas()
        self._slow_canvas.setVisible(False)
        self._plot_splitter = QSplitter(Qt.Vertical)
        self._plot_splitter.addWidget(self._canvas)
        self._plot_splitter.addWidget(self._slow_canvas)
        self._plot_splitter.setStretchFactor(0, 3)
        self._plot_splitter.setStretchFactor(1, 2)
        self._plot_splitter.setChildrenCollapsible(False)
        right_layout.addWidget(self._plot_splitter, 1)
        right_layout.addWidget(self._build_measurement_strip())
        right_layout.addWidget(self._build_history_bar())

        splitter.addWidget(left)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)

        # Live acquisition and file inspection are two whole workspaces, not two
        # panels, so they get the window in turn.  The control bar stays outside
        # the tabs: a run in progress must remain visible (and stoppable) while
        # files from it are being looked at.
        self._inspector = InspectorPanel()
        self._tabs = QTabWidget()
        self._tabs.addTab(splitter, 'Live')
        self._tabs.addTab(self._inspector, 'Inspect')
        self._tabs.currentChanged.connect(self._on_tab_changed)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(self._tabs, 1)
        central_layout.addWidget(self._build_control_bar())
        self.setCentralWidget(central)

    def _on_tab_changed(self, index: int) -> None:
        """Point the inspector at the output directory the first time it opens."""
        if self._tabs.widget(index) is self._inspector:
            self._inspector.note_output_dir(self._dir_edit.text().strip())

    def _build_hardware_group(self) -> QGroupBox:
        group  = QGroupBox('Hardware')
        layout = QGridLayout(group)

        self._pico_combo = QComboBox()
        for label in PICO_SERIALS:
            self._pico_combo.addItem(label)

        self._connect_btn    = QPushButton('Connect')
        self._disconnect_btn = QPushButton('Disconnect')
        self._apply_btn      = QPushButton('Apply settings')
        self._apply_btn.setToolTip(
            'Push the channel table and sample rate to the connected scope.\n'
            'Monitoring restarts automatically; not available while recording.')
        self._connect_btn.clicked.connect(self._on_connect)
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._apply_btn.clicked.connect(self._on_apply_settings)

        layout.addWidget(QLabel('Scope:'),     0, 0)
        layout.addWidget(self._pico_combo,     0, 1, 1, 2)
        layout.addWidget(self._connect_btn,    1, 0)
        layout.addWidget(self._disconnect_btn, 1, 1)
        layout.addWidget(self._apply_btn,      1, 2)
        return group

    def _build_channel_group(self) -> QGroupBox:
        group  = QGroupBox('Channels')
        layout = QGridLayout(group)

        for col, header in enumerate(['', 'Ch', 'Range', 'Coupling', 'Save']):
            layout.addWidget(QLabel(f'<b>{header}</b>'), 0, col)

        self._ch_rows: Dict[str, dict] = {}
        for row, ch in enumerate(ALL_CHANNELS, start=1):
            enable      = QCheckBox()
            range_cb    = QComboBox()
            for lbl in RANGE_LABELS:
                range_cb.addItem(lbl)
            range_cb.setCurrentIndex(7)   # 2 V default
            coupling_cb = QComboBox()
            coupling_cb.addItems(['DC', 'AC'])
            save_cb     = QComboBox()
            save_cb.addItems(['Full', 'Mean', 'Slow'])
            save_cb.setToolTip(
                'How this channel is kept.\n'
                'Full — the waveform, in every recorded file.\n'
                'Mean — one number per file.\n'
                'Slow — for an input that is itself slow (lock-in output,\n'
                '  temperature…): averaged to the slow-data rate, plotted on the\n'
                '  strip chart while monitoring and recording, and written as a\n'
                '  slow series instead of a waveform (see the Slow data tab).')
            save_cb.currentIndexChanged.connect(self._mark_config_dirty)

            enable.stateChanged.connect(self._mark_config_dirty)
            range_cb.currentIndexChanged.connect(self._mark_config_dirty)
            coupling_cb.currentIndexChanged.connect(self._mark_config_dirty)

            layout.addWidget(enable,      row, 0)
            layout.addWidget(QLabel(ch),  row, 1)
            layout.addWidget(range_cb,    row, 2)
            layout.addWidget(coupling_cb, row, 3)
            layout.addWidget(save_cb,     row, 4)

            self._ch_rows[ch] = {
                'enable':   enable,
                'range':    range_cb,
                'coupling': coupling_cb,
                'save':     save_cb,
            }
        return group

    def _build_acquisition_group(self) -> QGroupBox:
        group  = QGroupBox('Acquisition')
        layout = QGridLayout(group)

        # Sample rate and interval are two views of the same setting and stay in
        # sync; _syncing_rate guards the mutual updates from recursing.
        self._rate_spin = QDoubleSpinBox()
        self._rate_spin.setDecimals(6)
        self._rate_spin.setToolTip(
            'Desired sample rate.  The scope only accepts integer periods in '
            'ns/µs/ms,\nso the nearest achievable rate is used — check the summary line.')
        self._rate_unit = QComboBox()
        self._rate_unit.addItems(list(RATE_UNITS))
        self._apply_rate_range()          # range depends on the selected unit
        self._rate_spin.setValue(5.0)
        self._rate_spin.valueChanged.connect(self._on_rate_edited)
        self._rate_unit.currentIndexChanged.connect(self._on_rate_unit_changed)

        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 999999)
        self._interval_spin.setValue(200)
        self._interval_spin.setToolTip(
            'Sample period as the scope takes it — kept in sync with the rate above.\n'
            'Applies to monitoring and recording alike.')

        self._units_combo = QComboBox()
        self._units_combo.addItems(TIME_UNIT_OPTIONS)
        self._interval_spin.valueChanged.connect(self._on_interval_edited)
        self._units_combo.currentIndexChanged.connect(self._on_interval_edited)

        self._buffer_edit = QLineEdit('33554432')   # 2**25
        self._buffer_edit.setToolTip('Samples per file (recording mode only)')

        self._nfiles_spin = QSpinBox()
        self._nfiles_spin.setRange(1, 999999)
        self._nfiles_spin.setValue(100)

        self._idxstart_spin = QSpinBox()
        self._idxstart_spin.setRange(0, 999999)

        self._duration_label = QLabel('')
        self._duration_label.setWordWrap(True)
        self._duration_label.setStyleSheet('color: gray;')
        self._nfiles_spin.valueChanged.connect(self._update_duration_label)
        self._buffer_edit.textChanged.connect(self._update_duration_label)

        self._pressure_check     = QCheckBox('Read pressure')
        self._pressure_port_edit = QLineEdit('COM7')
        self._pressure_port_edit.setMaximumWidth(60)
        self._pressure_check.setChecked(True)
        self._pressure_check.setEnabled(_HAS_PRESSURE)
        if not _HAS_PRESSURE:
            self._pressure_check.setToolTip('Pressure driver not importable')

        layout.addWidget(QLabel('Sample rate:'),   0, 0)
        layout.addWidget(self._rate_spin,          0, 1)
        layout.addWidget(self._rate_unit,          0, 2)
        layout.addWidget(QLabel('Interval:'),      1, 0)
        layout.addWidget(self._interval_spin,      1, 1)
        layout.addWidget(self._units_combo,        1, 2)
        layout.addWidget(QLabel('Buffer size:'),   2, 0)
        layout.addWidget(self._buffer_edit,        2, 1, 1, 2)
        layout.addWidget(QLabel('N files:'),       3, 0)
        layout.addWidget(self._nfiles_spin,        3, 1)
        layout.addWidget(QLabel('Start index:'),   4, 0)
        layout.addWidget(self._idxstart_spin,      4, 1)
        layout.addWidget(self._duration_label,     5, 0, 1, 3)
        layout.addWidget(self._pressure_check,     6, 0, 1, 2)
        layout.addWidget(self._pressure_port_edit, 6, 2)
        return group

    def _build_output_group(self) -> QGroupBox:
        group  = QGroupBox('Output (recording mode)')
        layout = QGridLayout(group)

        self._dir_edit    = QLineEdit(r'E:\data')
        browse_btn        = QPushButton('…')
        browse_btn.setMaximumWidth(28)
        browse_btn.clicked.connect(self._browse_dir)
        self._prefix_edit = QLineEdit('YYYYMMDD_abc_')

        layout.addWidget(QLabel('Directory:'), 0, 0)
        layout.addWidget(self._dir_edit,       0, 1)
        layout.addWidget(browse_btn,           0, 2)
        layout.addWidget(QLabel('Prefix:'),    1, 0)
        layout.addWidget(self._prefix_edit,    1, 1, 1, 2)
        return group

    def _build_pulse_tab(self) -> QWidget:
        """
        Calibration-pulse controls: the shared panel plus the sequencing rules
        that only make sense next to a recording.

        The panel owns the serial session; this window only asks it for the
        generator when a run starts.  Nothing here affects monitor mode — the
        output gate on the panel works there too, it just is not stepped.
        """
        self._pulse_panel = PulseControlPanel()
        self._pulse_panel.settings_changed.connect(self._update_pulse_summary)
        self._pulse_panel.state_changed.connect(self._update_buttons)

        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.addWidget(self._pulse_panel)
        layout.addWidget(self._build_pulse_sequence_group())
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    def _build_pulse_sequence_group(self) -> QGroupBox:
        group  = QGroupBox('Sequence with recording')
        layout = QGridLayout(group)

        self._pulse_enable_check = QCheckBox('Step the amplitude while recording')
        self._pulse_enable_check.setToolTip(
            'Take N files at each amplitude in the list, changing the amplitude\n'
            'between files — the calibration script\'s loop, driven by the DAQ.')

        self._pulse_files_spin = QSpinBox()
        self._pulse_files_spin.setRange(1, 99999)
        self._pulse_files_spin.setValue(5)

        self._pulse_off_check = QCheckBox('Output off between amplitudes')
        self._pulse_off_check.setChecked(True)
        self._pulse_off_check.setToolTip(
            'Matches the script.  Unticked, the train runs continuously and only\n'
            'the amplitude steps.')

        self._pulse_cycle_check = QCheckBox('Cycle the list if the run is longer')
        self._pulse_cycle_check.setToolTip(
            'Unticked, a run longer than the list holds the last amplitude.')

        self._pulse_tag_check = QCheckBox('Put the amplitude in the filename')
        self._pulse_tag_check.setChecked(True)
        self._pulse_tag_check.setToolTip(
            'prefix + "2.5v_" + index, as in the calibration script.\n'
            'The amplitude is written into every file\'s attributes either way.')

        self._pulse_restart_check = QCheckBox('Restart the file index at each amplitude')
        self._pulse_restart_check.setChecked(True)
        self._pulse_restart_check.setToolTip(
            'Number files within their amplitude — 2.5v_0, 2.5v_1, 5v_0, … — as the\n'
            'calibration script did.  Unticked, the index runs through the whole run.\n'
            'With it ticked the start index is not advanced after a run, so a second\n'
            'run into the same directory with the same prefix overwrites the first.')

        match_btn = QPushButton('Set N files to match')
        match_btn.setToolTip('Set the run length to amplitudes × files per amplitude')
        match_btn.clicked.connect(self._on_match_files)

        self._pulse_summary = QLabel('')
        self._pulse_summary.setWordWrap(True)
        self._pulse_summary.setStyleSheet('color: gray;')

        self._pulse_enable_check.stateChanged.connect(self._update_pulse_summary)
        self._pulse_files_spin.valueChanged.connect(self._update_pulse_summary)
        self._pulse_cycle_check.stateChanged.connect(self._update_pulse_summary)
        self._nfiles_spin.valueChanged.connect(self._update_pulse_summary)

        layout.addWidget(self._pulse_enable_check,      0, 0, 1, 3)
        layout.addWidget(QLabel('Files per amplitude:'), 1, 0)
        layout.addWidget(self._pulse_files_spin,        1, 1)
        layout.addWidget(match_btn,                     1, 2)
        layout.addWidget(self._pulse_off_check,         2, 0, 1, 3)
        layout.addWidget(self._pulse_cycle_check,       3, 0, 1, 3)
        layout.addWidget(self._pulse_tag_check,         4, 0, 1, 3)
        layout.addWidget(self._pulse_restart_check,     5, 0, 1, 3)
        layout.addWidget(self._pulse_summary,           6, 0, 1, 3)
        return group

    def _update_pulse_summary(self) -> None:
        if not _HAS_PULSES:
            return
        try:
            amps = self._pulse_panel.amplitudes()
        except ValueError as exc:
            self._pulse_summary.setText(f'amplitude list: {exc}')
            return
        if not amps:
            self._pulse_summary.setText('amplitude list is empty')
            return

        per_amp = self._pulse_files_spin.value()
        needed  = len(amps) * per_amp
        n_files = self._nfiles_spin.value()
        file_s  = self._file_duration()
        text = (f'{len(amps)} amplitudes × {per_amp} files = {needed} files'
                f' ({_fmt_duration(needed * file_s)} at {_fmt_duration(file_s)} per file)')
        if n_files < needed:
            text += (f'\nrun is {n_files} files — the last '
                     f'{len(amps) - (n_files + per_amp - 1) // per_amp} '
                     f'amplitude(s) will not be reached')
        elif n_files > needed:
            text += (f'\nrun is {n_files} files — '
                     + ('the list cycles' if self._pulse_cycle_check.isChecked()
                        else 'the last amplitude is held for the remainder'))
        if self._pulse_enable_check.isChecked() and not self._pulse_panel.is_connected:
            text += '\nnot connected — Record will refuse to start'
        self._pulse_summary.setText(text)

    def _file_duration(self) -> float:
        """Seconds per file at the current rate and buffer size (0 if unset)."""
        try:
            buffer_size = int(self._buffer_edit.text())
        except ValueError:
            return 0.0
        return buffer_size / max(self._sample_rate(), 1e-9)

    def _on_match_files(self) -> None:
        try:
            amps = self._pulse_panel.amplitudes()
        except ValueError as exc:
            QMessageBox.warning(self, 'Amplitude list', str(exc))
            return
        if not amps:
            return
        self._nfiles_spin.setValue(len(amps) * self._pulse_files_spin.value())

    def _build_live_group(self) -> QGroupBox:
        """
        Live-view settings — shared by monitor mode and the in-run preview, so the
        two modes look and update the same way.  Every field is editable while a
        monitor or a run is in progress.
        """
        group  = QGroupBox('Live view (monitor + recording preview)')
        layout = QGridLayout(group)

        def spin(lo, hi, val, step, suffix, decimals=3, tip=''):
            s = QDoubleSpinBox()
            s.setDecimals(decimals)
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setValue(val)
            s.setSuffix(suffix)
            if tip:
                s.setToolTip(tip)
            s.valueChanged.connect(self._push_live_settings)
            return s

        self._refresh_spin = spin(
            0.02, 30.0, 0.05, 0.01, ' s',
            tip='Time-trace refresh interval — 0.05 s is 20 frames per second.\n'
                'For a smooth rolling display keep the time window several times '
                'longer than this.')
        self._psd_refresh_spin = spin(
            0.05, 60.0, 0.3, 0.1, ' s',
            tip='How often the spectrum is recomputed.  Welch over a long span is\n'
                'expensive, so this is kept slower than the trace refresh; frames\n'
                'in between redraw the trace alone.')

        # Horizontal scale, scope style: the window is n_div × s/div.
        self._tdiv_spin = QDoubleSpinBox()
        self._tdiv_spin.setDecimals(6)
        self._tdiv_spin.setRange(1e-6, 12.0)
        self._tdiv_spin.setSingleStep(0.001)
        self._tdiv_spin.setValue(0.05)
        self._tdiv_spin.setSuffix(' s/div')
        self._tdiv_spin.setToolTip('Horizontal scale of the time panel')
        self._tdiv_spin.valueChanged.connect(self._on_timebase_changed)

        self._xdiv_spin = QSpinBox()
        self._xdiv_spin.setRange(1, 20)
        self._xdiv_spin.setValue(10)
        self._xdiv_spin.setSuffix(' div')
        self._xdiv_spin.setToolTip('Number of horizontal divisions')
        self._xdiv_spin.valueChanged.connect(self._on_timebase_changed)

        self._window_label = QLabel('')
        self._window_label.setStyleSheet('color: gray;')

        self._psdspan_spin = spin(
            0.01, 120.0, 0.2, 0.1, ' s',
            tip='Seconds of signal fed to the PSD.\n'
                'Longer = finer frequency resolution, slower refresh.')
        self._chunk_spin = spin(
            0.005, 5.0, 0.05, 0.01, ' s',
            tip='Driver callback granularity (monitor mode).\n'
                'Lower = lower latency, more overhead.')

        self._stream_combo = QComboBox()
        self._stream_combo.addItems(['Auto', 'Roll (continuous)', 'Sweeps (scope)'])
        self._stream_combo.setToolTip(
            'How the monitor gets its data.\n'
            'Roll — one endless stream; smooth rolling trace, but every\n'
            '  sample must cross USB in real time.  Channels × rate beyond what\n'
            '  the link carries (about 20 MB/s on a USB 2.0 port, i.e. 2 channels\n'
            '  at 5 MS/s) looks fine for some tens of seconds and then breaks up:\n'
            '  thick traces and a comb of lines in the PSD.\n'
            'Sweeps — what a bench scope (and the PicoScope application) does\n'
            '  at fast timebases: capture one screenful at full rate into the\n'
            '  scope\'s own memory, transfer it, draw it, re-arm.  Clean at any\n'
            '  channel count and rate; the signal between sweeps is not seen.\n'
            'Auto — roll while the link can carry it, sweeps beyond that.')
        self._stream_combo.currentIndexChanged.connect(self._push_live_settings)

        self._nperseg_combo = QComboBox()
        self._nperseg_combo.addItems(NPERSEG_OPTIONS)
        self._nperseg_combo.setToolTip('Welch segment length (auto ≈ fs/10, capped at 65536)')
        self._nperseg_combo.currentIndexChanged.connect(self._push_live_settings)

        self._psdavg_spin = QSpinBox()
        self._psdavg_spin.setRange(1, 200)
        self._psdavg_spin.setValue(4)
        self._psdavg_spin.setToolTip('Running average of the PSD over N monitor frames '
                                     '(1 = no averaging)')
        self._psdavg_spin.valueChanged.connect(self._push_live_settings)

        self._psd_check = QCheckBox('Compute PSD live (monitor)')
        self._psd_check.setChecked(True)
        self._psd_check.setToolTip('Uncheck for the fastest possible time-domain refresh')
        self._psd_check.stateChanged.connect(self._push_live_settings)

        self._record_psd_check = QCheckBox('…also during a recording run')
        self._record_psd_check.setToolTip(
            'While recording, the preview is computed inside the streaming callback,\n'
            'so a live PSD delays the driver and risks dropped samples in the saved\n'
            'files.  Leave off: the spectrum still updates from the whole file after\n'
            'every file, at better frequency resolution.')
        self._record_psd_check.stateChanged.connect(self._push_live_settings)

        self._detrend_check = QCheckBox('Remove DC for PSD')
        self._detrend_check.setChecked(True)
        self._detrend_check.stateChanged.connect(self._push_live_settings)

        self._depth_spin = QSpinBox()
        self._depth_spin.setRange(0, 5000)
        self._depth_spin.setValue(200)
        self._depth_spin.setSpecialValueText('off')
        self._depth_spin.setSuffix(' frames')
        self._depth_spin.setToolTip(
            'How many past frames to keep for review (0 disables history).\n'
            'Frames hold the decimated trace and spectrum, not raw samples, so\n'
            'hundreds are cheap — the memory in use is shown below.')
        self._depth_spin.valueChanged.connect(self._on_depth_changed)

        self._hist_mem_label = QLabel('')
        self._hist_mem_label.setStyleSheet('color: gray;')

        self._roll_check = QCheckBox('Roll mode (newest at right)')
        self._roll_check.setChecked(True)
        self._roll_check.setToolTip(
            'Plot history at negative time with the newest sample at 0, so the\n'
            'trace slides leftwards on a fixed axis like a scope in roll mode.\n'
            'Unchecked, each frame starts at t=0 instead.')
        self._roll_check.stateChanged.connect(self._push_live_settings)

        self._resume_check = QCheckBox('Resume monitoring after a run')
        self._resume_check.setChecked(True)
        self._resume_check.setToolTip(
            'When a recording run finishes or is stopped, go straight back to monitoring')

        layout.addWidget(QLabel('Time scale:'), 0, 0)
        layout.addWidget(self._tdiv_spin,       0, 1)
        layout.addWidget(self._xdiv_spin,       0, 2)
        layout.addWidget(self._window_label,    1, 0, 1, 3)

        rows = [
            ('Refresh:',      self._refresh_spin),
            ('PSD refresh:',  self._psd_refresh_spin),
            ('PSD span:',     self._psdspan_spin),
            ('PSD nperseg:',  self._nperseg_combo),
            ('PSD average:',  self._psdavg_spin),
            ('Stream chunk:', self._chunk_spin),
            ('Monitor stream:', self._stream_combo),
            ('History:',      self._depth_spin),
        ]
        for i, (label, widget) in enumerate(rows):
            layout.addWidget(QLabel(label), i + 2, 0)
            layout.addWidget(widget,        i + 2, 1, 1, 2)
        base = len(rows) + 2
        layout.addWidget(self._hist_mem_label,   base,     0, 1, 3)
        layout.addWidget(self._psd_check,        base + 1, 0, 1, 3)
        layout.addWidget(self._record_psd_check, base + 2, 0, 1, 3)
        layout.addWidget(self._detrend_check,    base + 3, 0, 1, 3)
        layout.addWidget(self._roll_check,       base + 4, 0, 1, 3)
        layout.addWidget(self._resume_check,     base + 5, 0, 1, 3)
        return group

    def _build_view_group(self) -> QGroupBox:
        """Drawing-only options — take effect on the cached frame immediately."""
        group  = QGroupBox('Display')
        layout = QGridLayout(group)

        self._points_spin = QSpinBox()
        self._points_spin.setRange(200, 200000)
        self._points_spin.setSingleStep(500)
        self._points_spin.setValue(4000)
        self._points_spin.setToolTip('Target number of points drawn in the time panel')
        self._points_spin.valueChanged.connect(self._push_live_settings)

        self._psd_points_spin = QSpinBox()
        self._psd_points_spin.setRange(0, 200000)
        self._psd_points_spin.setSingleStep(500)
        self._psd_points_spin.setValue(3000)
        self._psd_points_spin.setSpecialValueText('all')
        self._psd_points_spin.setToolTip(
            'Points drawn in the PSD panel (0 = every bin).\n'
            'Thinning keeps the maximum of each group, so narrow peaks survive but\n'
            'the broadband floor reads a few dB high.  The peak readout always uses\n'
            'the full-resolution spectrum.')
        self._psd_points_spin.valueChanged.connect(self._push_live_settings)

        self._envelope_check = QCheckBox('Min/max envelope')
        self._envelope_check.setChecked(True)
        self._envelope_check.setToolTip(
            'Decimate by min/max per bin instead of taking every Nth sample,\n'
            'so a fast oscillation keeps its true amplitude on screen.')
        self._envelope_check.stateChanged.connect(self._push_live_settings)

        self._ylim_combo = QComboBox()
        self._ylim_combo.addItems(YLIM_OPTIONS)
        self._ylim_combo.setToolTip(
            'Time-panel vertical range:\n'
            '  auto        — autoscale to the data\n'
            '  full range  — ±(channel input range), shows headroom\n'
            '  manual      — the min/max below\n'
            '  mV/div      — symmetric about 0, scope style')
        self._ylim_combo.currentIndexChanged.connect(self._on_ylim_mode_changed)

        self._ymin_edit = QLineEdit('-1000')
        self._ymax_edit = QLineEdit('1000')
        for w, tip in ((self._ymin_edit, 'Lower limit of the time panel, mV'),
                       (self._ymax_edit, 'Upper limit of the time panel, mV')):
            w.setToolTip(tip)
            w.editingFinished.connect(self._on_view_changed)

        self._mvdiv_spin = QDoubleSpinBox()
        self._mvdiv_spin.setDecimals(3)
        self._mvdiv_spin.setRange(1e-3, 20000.0)
        self._mvdiv_spin.setValue(500.0)
        self._mvdiv_spin.setSuffix(' mV/div')
        self._mvdiv_spin.valueChanged.connect(self._on_view_changed)

        self._ydiv_spin = QSpinBox()
        self._ydiv_spin.setRange(2, 20)
        self._ydiv_spin.setValue(8)
        self._ydiv_spin.setSuffix(' div')
        self._ydiv_spin.valueChanged.connect(self._on_view_changed)

        self._griddiv_check = QCheckBox('Ticks on divisions')
        self._griddiv_check.setChecked(True)
        self._griddiv_check.setToolTip(
            'Put gridlines exactly on the division boundaries instead of letting '
            'matplotlib choose')
        self._griddiv_check.stateChanged.connect(self._on_view_changed)

        self._logx_check = QCheckBox('Log frequency axis')
        self._logx_check.stateChanged.connect(self._on_view_changed)

        self._fmin_edit = QLineEdit('')
        self._fmax_edit = QLineEdit('')
        self._psd_ymin_edit = QLineEdit('')
        self._psd_ymax_edit = QLineEdit('')
        for w, tip in ((self._fmin_edit, 'Lower frequency limit (blank = auto)'),
                       (self._fmax_edit, 'Upper frequency limit (blank = auto)'),
                       (self._psd_ymin_edit,
                        'Lower PSD limit in V²/Hz, e.g. 1e-14 (blank = auto)'),
                       (self._psd_ymax_edit,
                        'Upper PSD limit in V²/Hz, e.g. 1e-6 (blank = auto)')):
            w.setPlaceholderText('auto')
            w.setToolTip(tip)
            w.editingFinished.connect(self._on_view_changed)

        self._peak_check = QCheckBox('Mark PSD peak')
        self._peak_check.setToolTip('Annotate the largest peak of each spectrum')
        self._peak_check.stateChanged.connect(self._on_view_changed)

        self._rms_check = QCheckBox('Show RMS in legend')
        self._rms_check.setChecked(True)
        self._rms_check.stateChanged.connect(self._on_view_changed)

        layout.addWidget(QLabel('<b>Time panel</b>'),  0, 0, 1, 3)
        layout.addWidget(QLabel('Vertical:'),          1, 0)
        layout.addWidget(self._ylim_combo,             1, 1, 1, 2)
        layout.addWidget(QLabel('min / max (mV):'),    2, 0)
        layout.addWidget(self._ymin_edit,              2, 1)
        layout.addWidget(self._ymax_edit,              2, 2)
        layout.addWidget(QLabel('Scale:'),             3, 0)
        layout.addWidget(self._mvdiv_spin,             3, 1)
        layout.addWidget(self._ydiv_spin,              3, 2)
        layout.addWidget(QLabel('Trace points:'),      4, 0)
        layout.addWidget(self._points_spin,            4, 1, 1, 2)
        layout.addWidget(self._envelope_check,         5, 0, 1, 3)
        layout.addWidget(self._griddiv_check,          6, 0, 1, 3)

        layout.addWidget(QLabel('<b>PSD panel</b>'),   7, 0, 1, 3)
        layout.addWidget(QLabel('f range (Hz):'),      8, 0)
        layout.addWidget(self._fmin_edit,              8, 1)
        layout.addWidget(self._fmax_edit,              8, 2)
        layout.addWidget(QLabel('PSD range (V²/Hz):'), 9, 0)
        layout.addWidget(self._psd_ymin_edit,          9, 1)
        layout.addWidget(self._psd_ymax_edit,          9, 2)
        layout.addWidget(QLabel('PSD points:'),       10, 0)
        layout.addWidget(self._psd_points_spin,       10, 1, 1, 2)
        layout.addWidget(self._logx_check,            11, 0, 1, 3)
        layout.addWidget(self._peak_check,            12, 0, 1, 3)
        layout.addWidget(self._rms_check,             13, 0, 1, 3)
        return group   # field enabling is seeded from __init__, once the canvas exists

    def _build_display_selector(self) -> QWidget:
        """
        A row of checkboxes (one per recorded channel) plus a freeze toggle.
        Hidden until connected; toggling re-renders from the cached last frame.
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(QLabel('Display:'))

        self._display_checks: Dict[str, QCheckBox] = {}
        for ch in ALL_CHANNELS:
            cb = QCheckBox(ch)
            cb.setVisible(False)
            cb.stateChanged.connect(self._on_display_selection_changed)
            self._display_checks[ch] = cb
            layout.addWidget(cb)

        layout.addStretch()

        self._freeze_check = QCheckBox('Hold')
        self._freeze_check.setToolTip(
            'Hold the current frame on screen.  Streaming continues underneath\n'
            'and history keeps filling, so Live returns you to the present.')
        self._freeze_check.toggled.connect(self._on_freeze_toggled)
        layout.addWidget(self._freeze_check)

        reset_btn = QPushButton('Reset average')
        reset_btn.setToolTip('Restart the PSD running average')
        reset_btn.clicked.connect(self._on_reset_average)
        layout.addWidget(reset_btn)

        self._reset_view_btn = QPushButton('Reset view')
        self._reset_view_btn.setToolTip(
            'Hand the axis limits back to the settings on the left.\n'
            'Wheel over a panel zooms about the cursor (Shift = vertical in the '
            'time panel);\nthe toolbar has pan and box zoom.  While you are '
            'zoomed the live rescaling stands down.')
        self._reset_view_btn.setEnabled(False)
        self._reset_view_btn.clicked.connect(self._on_reset_view)
        layout.addWidget(self._reset_view_btn)
        return widget

    def _build_measurement_strip(self) -> QWidget:
        """
        Per-channel live readouts under the plots.

        These change every frame; keeping them out of the legend means the legend
        stays static and the trace can be blitted instead of re-rendered.
        """
        self._measure_label = QLabel('')
        self._measure_label.setTextFormat(Qt.RichText)
        self._measure_label.setStyleSheet('font-family: monospace;')
        return self._measure_label

    def _build_history_bar(self) -> QWidget:
        """
        Navigator across stored frames.

        Stepping back does not interrupt the scope: frames keep arriving and being
        stored, only the display stops following them, until Live is pressed.
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 0, 4, 0)

        layout.addWidget(QLabel('History:'))
        self._hist_prev_btn = QPushButton('◀')
        self._hist_next_btn = QPushButton('▶')
        for b, step, tip in ((self._hist_prev_btn, -1, 'Step back one frame'),
                             (self._hist_next_btn, +1, 'Step forward one frame')):
            b.setMaximumWidth(32)
            b.setToolTip(tip)
            b.clicked.connect(lambda _checked, s=step: self._on_history_step(s))

        self._hist_slider = QSlider(Qt.Horizontal)
        self._hist_slider.setToolTip('Scroll back through stored frames')
        self._hist_slider.valueChanged.connect(self._on_history_slider)

        self._live_btn = QPushButton('▶ Live')
        self._live_btn.setToolTip('Resume following the newest frame')

        self._live_btn.clicked.connect(self._on_go_live)
        self._hist_label = QLabel('empty')
        self._hist_label.setStyleSheet('color: gray;')
        self._hist_label.setMinimumWidth(210)

        self._save_frame_btn = QPushButton('Save frame…')
        self._save_frame_btn.setToolTip(
            'Write the frame on screen to a .npz — trace/band, spectrum and the '
            'readouts,\nat display resolution (this is not the raw waveform).')
        self._save_frame_btn.clicked.connect(self._on_save_frame)

        layout.addWidget(self._hist_prev_btn)
        layout.addWidget(self._hist_slider, 1)
        layout.addWidget(self._hist_next_btn)
        layout.addWidget(self._live_btn)
        layout.addWidget(self._hist_label)
        layout.addWidget(self._save_frame_btn)
        return widget

    def _build_control_bar(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)

        self._monitor_btn = QPushButton('◉  Monitor')
        self._record_btn  = QPushButton('▶  Record')
        self._stop_btn    = QPushButton('■  Stop')
        self._monitor_btn.setToolTip('Free-running view — no files written')
        self._record_btn.setToolTip('Start writing files (stops monitoring first)')
        self._dump_btn = QPushButton('Dump raw')
        self._dump_btn.setToolTip(
            'Write the next monitor window to an HDF5 file, raw int16, straight\n'
            'from the ring — no decimation, no PSD.  For diagnosing something\n'
            'that only shows up live.')
        self._monitor_btn.clicked.connect(self._on_monitor_clicked)
        self._record_btn.clicked.connect(self._on_record_clicked)
        self._stop_btn.clicked.connect(self._on_stop)
        self._dump_btn.clicked.connect(self._on_dump_raw)

        self._mode_label     = QLabel('Not connected')
        self._mode_label.setStyleSheet('font-weight: bold;')
        self._progress_label = QLabel('')
        self._pressure_label = QLabel('')
        self._pulse_label    = QLabel('')
        self._pulse_label.setStyleSheet('color: #b00;')
        self._elapsed_label  = QLabel('00:00')

        layout.addWidget(self._monitor_btn)
        layout.addWidget(self._record_btn)
        layout.addWidget(self._stop_btn)
        layout.addWidget(self._dump_btn)
        layout.addWidget(self._mode_label)
        layout.addWidget(self._progress_label)
        layout.addStretch()
        layout.addWidget(self._pulse_label)
        layout.addWidget(self._pressure_label)
        layout.addWidget(self._elapsed_label)
        return widget

    # ── Settings gathering ──────────────────────────────────────────────────────

    def _display_options(self, want_psd: Optional[bool] = None) -> DisplayOptions:
        nperseg_text = self._nperseg_combo.currentText()
        return DisplayOptions(
            n_points=self._points_spin.value(),
            psd_points=self._psd_points_spin.value(),
            envelope=self._envelope_check.isChecked(),
            nperseg=0 if nperseg_text == 'auto' else int(nperseg_text),
            detrend=self._detrend_check.isChecked(),
            time_span=self._time_span(),
            psd_span=self._psdspan_spin.value(),
            want_psd=self._psd_check.isChecked() if want_psd is None else want_psd,
            roll=self._roll_check.isChecked(),
        )

    def _record_display_options(self) -> DisplayOptions:
        """
        Preview options for a recording run.

        The preview is prepared inside the streaming callback, so a live PSD there
        stalls the driver — it is opt-in, and when it is off the preview window is
        shrunk to just what the time panel needs.
        """
        live_psd = self._psd_check.isChecked() and self._record_psd_check.isChecked()
        opts = self._display_options(want_psd=live_psd)
        if not live_psd:
            opts.psd_span = 0.0
        return opts

    def _record_window_duration(self) -> float:
        live_psd = self._psd_check.isChecked() and self._record_psd_check.isChecked()
        return max(self._time_span(),
                   self._psdspan_spin.value() if live_psd else 0.0)

    def _view_options(self) -> ViewOptions:
        def parse(edit: QLineEdit, default: float = 0.0) -> float:
            try:
                return float(edit.text())
            except ValueError:
                return default
        return ViewOptions(
            psd_logx=self._logx_check.isChecked(),
            f_min=parse(self._fmin_edit),
            f_max=parse(self._fmax_edit),
            psd_ymin=parse(self._psd_ymin_edit),
            psd_ymax=parse(self._psd_ymax_edit),
            ylim_mode=self._ylim_combo.currentText(),
            y_min=parse(self._ymin_edit, -1000.0),
            y_max=parse(self._ymax_edit, 1000.0),
            mv_per_div=self._mvdiv_spin.value(),
            n_div_y=self._ydiv_spin.value(),
            time_per_div=self._tdiv_spin.value(),
            n_div_x=self._xdiv_spin.value(),
            show_grid_divs=self._griddiv_check.isChecked(),
            show_peak=self._peak_check.isChecked(),
            show_rms=self._rms_check.isChecked(),
        )

    # ── Timebase / rate ─────────────────────────────────────────────────────────

    def _time_span(self) -> float:
        """Length of the live time window: n divisions × s/div."""
        return self._tdiv_spin.value() * self._xdiv_spin.value()

    def _on_timebase_changed(self) -> None:
        """s/div or division count changed — resize the live window and redraw."""
        self._window_label.setText(f'window = {_fmt_duration(self._time_span())}')
        self._push_live_settings()
        self._on_view_changed()

    def _sample_rate(self) -> float:
        return interval_to_rate(self._interval_spin.value(),
                                self._units_combo.currentText())

    def _apply_rate_range(self) -> None:
        """
        Re-range the rate spin for the selected unit.

        Without this the maximum (80 MS/s, the scope's ceiling) would clamp a
        perfectly valid 250 kS/s down to 80.
        """
        scale = RATE_UNITS[self._rate_unit.currentText()]
        self._rate_spin.setRange(1e-6, 80e6 / scale)

    def _on_rate_unit_changed(self) -> None:
        """Unit switched — keep the typed number and reinterpret it in the new unit."""
        if self._syncing_rate:
            return
        self._apply_rate_range()
        self._on_rate_edited()

    def _on_rate_edited(self) -> None:
        """Rate spin edited — convert to the scope's integer interval + units."""
        if self._syncing_rate:
            return
        fs = self._rate_spin.value() * RATE_UNITS[self._rate_unit.currentText()]
        if fs <= 0:
            return
        interval, units = rate_to_interval(fs)
        self._syncing_rate = True
        self._interval_spin.setValue(interval)
        self._units_combo.setCurrentText(units)
        self._syncing_rate = False
        self._after_rate_change()

    def _on_interval_edited(self) -> None:
        """Interval/units edited directly — show the resulting rate."""
        if self._syncing_rate:
            return
        fs = self._sample_rate()
        for unit, scale in RATE_UNITS.items():
            if fs >= scale or scale == 1.0:
                self._syncing_rate = True
                self._rate_unit.setCurrentText(unit)
                self._apply_rate_range()      # before setValue, or it would clamp
                self._rate_spin.setValue(fs / scale)
                self._syncing_rate = False
                break
        self._after_rate_change()

    def _after_rate_change(self) -> None:
        self._mark_config_dirty()

    def _mark_config_dirty(self) -> None:
        """
        Note that the panel no longer matches the scope.

        Sample rate, ranges and coupling only reach the hardware through
        reconfigure(), so a change made after connecting does nothing until
        "Apply settings" is pressed — say so rather than letting the user wonder.
        """
        self._config_dirty = True
        self._update_duration_label()

    def _clear_config_dirty(self) -> None:
        self._config_dirty = False
        self._update_duration_label()

    def _push_live_settings(self) -> None:
        """Send current live-view settings to whichever worker is running."""
        if self._monitor is not None and self._monitor.isRunning():
            self._monitor.apply_settings(
                refresh=self._refresh_spin.value(),
                time_span=self._time_span(),
                psd_span=self._psdspan_spin.value(),
                chunk=self._chunk_spin.value(),
                display_opts=self._display_options(),
                psd_average=self._psdavg_spin.value(),
                psd_refresh=self._psd_refresh_spin.value(),
                burst=self._monitor_burst(),
            )
        if self._worker is not None and self._worker.isRunning():
            self._worker.apply_display_options(self._record_display_options())

    def _monitor_burst(self) -> bool:
        """
        Whether the monitor should take bursts rather than stream endlessly.

        In Auto this is decided by arithmetic, not by watching for damage: the
        failure of an over-driven continuous stream is silent for the first tens
        of seconds and looks like signal afterwards.
        """
        mode = self._stream_combo.currentText()
        if mode != 'Auto' or self._daq is None:
            return mode.startswith('Sweeps')
        cfg    = self._daq.config
        demand = len(cfg.channels) * cfg.fs * 2            # bytes per second
        usb2   = self._daq.connect_status == 'PICO_USB3_0_DEVICE_NON_USB3_0_PORT'
        return demand > (USB2_STREAM_LIMIT if usb2 else USB3_STREAM_LIMIT)

    def _on_view_changed(self) -> None:
        self._canvas.set_view_options(self._view_options())
        self._redraw_plots()

    def _on_ylim_mode_changed(self) -> None:
        """Grey out the range fields that do not apply to the selected mode."""
        mode = self._ylim_combo.currentText()
        for w in (self._ymin_edit, self._ymax_edit):
            w.setEnabled(mode == 'manual')
        for w in (self._mvdiv_spin, self._ydiv_spin):
            w.setEnabled(mode == 'mV/div')
        self._on_view_changed()

    def _on_reset_average(self) -> None:
        if self._monitor is not None and self._monitor.isRunning():
            self._monitor.reset_average()

    def _update_duration_label(self) -> None:
        try:
            buffer_size = int(self._buffer_edit.text())
        except ValueError:
            self._duration_label.setText('buffer size must be an integer')
            return
        actual   = self._sample_rate()
        per_file = buffer_size / actual
        total    = per_file * self._nfiles_spin.value()
        mb       = buffer_size * 2 / 1e6
        # Show what the scope will actually sample at: the requested rate is
        # quantized to an integer ns/µs/ms period, so the two can differ.
        wanted = self._rate_spin.value() * RATE_UNITS[self._rate_unit.currentText()]
        rate   = _fmt_rate(actual)
        if abs(actual - wanted) / max(wanted, 1e-12) > 1e-6:
            rate += f' (requested {_fmt_rate(wanted)})'
        text = (f'{rate} · {_fmt_duration(per_file)}/file · '
                f'{mb:.0f} MB/ch · total {_fmt_duration(total)}')
        if self._config_dirty and self._daq is not None:
            text += '\n⚠ differs from the scope — press "Apply settings"'
        self._duration_label.setText(text)

    # ── Hardware actions ────────────────────────────────────────────────────────

    def _on_connect(self) -> None:
        config = self._build_daq_config()
        if config is None:
            return
        serial = PICO_SERIALS[self._pico_combo.currentText()]
        try:
            self._daq = PicoscopeDAQ(config)
            self._daq.connect(serial)
        except Exception as exc:
            QMessageBox.critical(self, 'Connection error', str(exc))
            self._daq = None
            return

        self._refresh_channel_display(config)
        self._canvas.set_view_options(self._view_options())
        self._clear_config_dirty()
        self._mode = 'idle'
        self._mode_label.setText('Connected — idle')
        self._update_buttons()

        # The unit opens either way, but a USB 3.0 scope on a USB 2.0 port
        # sustains only a fraction of the streaming rate and will drop samples
        # part-way into a fast monitor rather than failing outright.
        if self._daq.connect_status == 'PICO_USB3_0_DEVICE_NON_USB3_0_PORT':
            QMessageBox.warning(
                self, 'USB 2.0 port',
                'The scope opened, but it is a USB 3.0 device on a USB 2.0 port.\n\n'
                'Sustained streaming bandwidth is several times lower than on a '
                'USB 3.0 port. A fast multi-channel monitor may run cleanly for '
                'tens of seconds and then start losing samples.\n\n'
                'Move it to a USB 3.0 port (blue connector) and use the cable '
                'that came with the scope.')
        elif self._daq.connect_status:
            self._mode_label.setText(
                f'Connected — idle  ({self._daq.connect_status})')

    def _on_dump_raw(self) -> None:
        if self._monitor is None or not self._monitor.isRunning():
            QMessageBox.information(self, 'Not monitoring',
                                    'Start the monitor first, then dump while the '
                                    'artefact is on screen.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Dump raw monitor window',
            os.path.join(os.path.expanduser('~'),
                         time.strftime('monitor_raw_%Y%m%d_%H%M%S.hdf5')),
            'HDF5 (*.hdf5)')
        if path:
            self._monitor.request_raw_dump(path)

    def _on_disconnect(self) -> None:
        if self._mode != 'idle':
            QMessageBox.warning(self, 'Busy', 'Stop the current run first.')
            return
        if self._daq:
            try:
                self._daq.disconnect()
            except Exception:
                pass
            self._daq = None
        self._close_slow_log()
        self._hide_display_checkboxes()
        self._mode = 'idle'
        self._mode_label.setText('Disconnected')
        self._progress_label.setText('')
        self._update_buttons()

    def _on_apply_settings(self) -> None:
        """
        Push the channel table / sample rate to the open scope.

        While monitoring this stops the stream, reconfigures, and restarts —
        so ranges and coupling can be tuned with the signal on screen.
        """
        if self._daq is None:
            return
        if self._mode == 'recording':
            QMessageBox.warning(self, 'Recording',
                                'Cannot change scope settings during a run.')
            return
        if self._mode == 'monitor':
            self._stop_monitor_then('apply_and_monitor')
            return
        self._apply_config_now()

    def _apply_config_now(self) -> bool:
        config = self._build_daq_config()
        if config is None:
            return False
        try:
            self._daq.reconfigure(config)
        except Exception as exc:
            QMessageBox.critical(self, 'Configuration error', str(exc))
            return False
        self._refresh_channel_display(config)
        self._clear_config_dirty()
        self._mode_label.setText('Settings applied')
        return True

    def _refresh_channel_display(self, config: AcquisitionConfig) -> None:
        """Re-sync colours and the display checkbox row with the active channels."""
        names = [ch.name for ch in config.channels]
        self._canvas.assign_colors(names)
        self._canvas.reset_axes()
        self._canvas.set_view_options(self._view_options())
        # Stored frames describe the old channel set, so they cannot be replayed
        # against the new one.
        self._live_frame = self._shown_frame = None
        self._selected_seq = None
        self._history.clear()
        self._on_go_live()
        for ch, cb in self._display_checks.items():
            cb.blockSignals(True)
            cb.setVisible(ch in names)
            cb.setChecked(ch in names)
            cb.blockSignals(False)

    def _hide_display_checkboxes(self) -> None:
        for cb in self._display_checks.values():
            cb.setVisible(False)

    # ── Mode transitions ────────────────────────────────────────────────────────

    def _update_buttons(self) -> None:
        connected = self._daq is not None
        idle      = self._mode == 'idle'
        monitoring = self._mode == 'monitor'
        recording  = self._mode == 'recording'

        self._connect_btn.setEnabled(not connected)
        self._disconnect_btn.setEnabled(connected and idle)
        self._apply_btn.setEnabled(connected and not recording)
        self._monitor_btn.setEnabled(connected and (idle or monitoring))
        self._monitor_btn.setText('■  Stop monitor' if monitoring else '◉  Monitor')
        self._record_btn.setEnabled(connected and (idle or monitoring))
        self._stop_btn.setEnabled(recording)

    def _on_monitor_clicked(self) -> None:
        if self._mode == 'monitor':
            self._stop_monitor_then(None)
        elif self._mode == 'idle':
            self._start_monitor()

    def _start_monitor(self) -> None:
        if self._daq is None:
            return
        self._monitor = MonitorWorker(
            daq=self._daq,
            refresh=self._refresh_spin.value(),
            time_span=self._time_span(),
            psd_span=self._psdspan_spin.value(),
            chunk=self._chunk_spin.value(),
            display_opts=self._display_options(),
            psd_average=self._psdavg_spin.value(),
            psd_refresh=self._psd_refresh_spin.value(),
            slow_config=self._prepare_slow(),
            slow_tracked=self._slow_tracked,
            burst=self._monitor_burst(),
        )
        self._monitor.slow_rows.connect(self._on_slow_rows)
        self._monitor.frame_ready.connect(self._on_monitor_frame)
        self._monitor.status.connect(self._mode_label.setText)
        self._monitor.error.connect(self._on_monitor_error)
        self._monitor.done.connect(self._on_monitor_done)

        self._mode = 'monitor'
        self._progress_label.setText('not saving')
        self._start_time = time.time()
        self._elapsed_timer.start(1000)
        self._mode_label.setText('Monitoring…')
        self._update_buttons()
        self._monitor.start()

    def _stop_monitor_then(self, pending: Optional[str]) -> None:
        """
        Ask the monitor to stop and remember what to do once it has.

        monitor_stream() polls its stop flag every loop iteration, so this
        returns to idle within a few milliseconds — the queued action then runs
        from _on_monitor_done with the scope guaranteed idle.
        """
        self._pending = pending
        if self._monitor is None or not self._monitor.isRunning():
            self._mode = 'idle'
            self._dispatch_pending()
            return
        self._mode_label.setText('Stopping monitor…')
        for btn in (self._monitor_btn, self._record_btn, self._apply_btn):
            btn.setEnabled(False)     # no second queued action while stopping
        self._monitor.request_stop()

    def _on_monitor_done(self) -> None:
        self._monitor = None
        self._mode = 'idle'
        self._elapsed_timer.stop()
        self._progress_label.setText('')
        self._mode_label.setText('Connected — idle')
        self._update_buttons()
        self._dispatch_pending()

    def _dispatch_pending(self) -> None:
        pending, self._pending = self._pending, None
        if pending == 'record':
            self._start_recording()
        elif pending == 'restart_monitor':
            self._start_monitor()
        elif pending == 'apply_and_monitor':
            if self._apply_config_now():
                self._start_monitor()
            else:
                self._update_buttons()

    def _on_monitor_error(self, msg: str) -> None:
        self._monitor = None
        self._pending = None
        self._mode = 'idle'
        self._elapsed_timer.stop()
        self._mode_label.setText('Monitor error')
        self._update_buttons()
        QMessageBox.critical(self, 'Monitor error', msg)

    # ── Recording ───────────────────────────────────────────────────────────────

    def _on_record_clicked(self) -> None:
        if self._mode == 'monitor':
            self._stop_monitor_then('record')     # seamless hand-off
        elif self._mode == 'idle':
            self._start_recording()

    def _start_recording(self) -> None:
        if self._daq is None:
            return
        directory = self._dir_edit.text().strip()
        if not directory:
            QMessageBox.warning(self, 'No directory', 'Set an output directory.')
            self._update_buttons()
            return

        sequencer, pulse_attrs = self._build_sequencer()
        if sequencer is False:            # requested but unusable — do not record
            self._update_buttons()
            return

        self._worker = AcquisitionWorker(
            daq=self._daq,
            file_directory=directory,
            file_prefix=self._prefix_edit.text(),
            idx_start=self._idxstart_spin.value(),
            n_files=self._nfiles_spin.value(),
            read_pressure=self._pressure_check.isChecked(),
            pressure_port=self._pressure_port_edit.text(),
            display_opts=self._record_display_options(),
            chunk_period=self._refresh_spin.value(),
            window_duration=self._record_window_duration(),
            sequencer=sequencer,
            tag_filenames=(_HAS_PULSES and self._pulse_tag_check.isChecked()),
            restart_index=(_HAS_PULSES and self._pulse_restart_check.isChecked()),
            pulse_attrs=pulse_attrs,
            slow_config=self._prepare_slow(),
            slow_embed=self._slow_panel.embed_in_files(),
            slow_tracked=self._slow_tracked,
        )
        self._worker.slow_rows.connect(self._on_slow_rows)
        self._worker.message.connect(self._on_worker_message)
        self._slow_panel.set_busy(True)
        # Remembered for _on_finished: per-amplitude numbering means the start
        # index must not drift, since each amplitude counts from it afresh.
        self._index_restarted = (sequencer is not None and _HAS_PULSES
                                 and self._pulse_restart_check.isChecked())
        self._worker.preview_ready.connect(self._on_preview_ready)
        self._worker.data_ready.connect(self._on_data_ready)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_error)
        self._worker.done.connect(self._on_finished)
        if sequencer is not None:
            self._worker.pulse_changed.connect(self._on_pulse_changed)
            self._worker.message.connect(self._pulse_panel.log_message)
            # The sequencer logs from the acquisition thread; routing it through
            # the worker's signal gets it onto the GUI thread.
            sequencer.set_log(self._worker.message.emit)
            self._pulse_panel.set_busy(True)
            self._pulse_panel.note_output_state(True, sequencer.amplitudes[0])

        self._mode = 'recording'
        self._files_written = 0
        self._start_time = time.time()
        self._elapsed_timer.start(1000)
        self._mode_label.setText('Recording…')
        self._progress_label.setText(f'Files: 0 / {self._nfiles_spin.value()}')
        self._update_buttons()
        self._worker.start()

    def _build_sequencer(self):
        """
        Build the pulse sequencer for a run.

        Returns (sequencer, extra HDF5 attributes), where the sequencer is None
        when pulses are not in use, or False when they were asked for but cannot
        be set up — in which case the run must not start, since a calibration
        run without pulses would silently produce useless files.
        """
        if not _HAS_PULSES or not self._pulse_enable_check.isChecked():
            return None, {}
        if not self._pulse_panel.is_connected:
            QMessageBox.warning(
                self, 'Calibration pulses',
                'Connect to the pulse generator on the Pulses tab, or untick '
                '"Step the amplitude while recording".')
            return False, {}
        amps = self._pulse_panel.validated_amplitudes(self)
        if amps is None:
            return False, {}
        # Shape goes out once, with the first amplitude already in place; from
        # here the sequencer only touches the amplitude and the output gate.
        if not self._pulse_panel.apply_shape(amps[0], parent=self):
            return False, {}
        try:
            sequencer = PulseSequencer(
                self._pulse_panel.generator,
                channel=self._pulse_panel.channel(),
                amplitudes=amps,
                files_per_amplitude=self._pulse_files_spin.value(),
                output_off_between=self._pulse_off_check.isChecked(),
                cycle=self._pulse_cycle_check.isChecked(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, 'Calibration pulses', str(exc))
            return False, {}
        return sequencer, self._pulse_panel.pulse_attrs()

    def _on_pulse_changed(self, file_index: int, amplitude: float) -> None:
        self._pulse_label.setText(f'pulse {amplitude:g} V')
        self._pulse_panel.note_output_state(True, amplitude)

    def _release_pulses(self) -> None:
        """Hand the pulse panel back to the user after a run (output already off)."""
        if not _HAS_PULSES:
            return
        self._pulse_label.setText('')
        self._pulse_panel.set_busy(False)
        self._pulse_panel.note_output_state(False)

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.request_stop()
        self._stop_btn.setEnabled(False)
        self._mode_label.setText('Stopping after current file…')

    def _on_finished(self) -> None:
        self._worker = None
        self._mode = 'idle'
        self._elapsed_timer.stop()
        self._slow_panel.set_busy(False)
        self._release_pulses()
        self._mode_label.setText(f'Run finished — {self._files_written} files')
        self._update_buttons()
        # The files just written are the ones most likely to be inspected next.
        self._inspector.note_output_dir(self._dir_edit.text().strip())
        # Advance the start index by what was actually written, so the next run
        # cannot overwrite this one (matters when a run is stopped early).  Not
        # when each amplitude numbered its own files: there the index is a base
        # every amplitude counts from, and advancing it would just offset the
        # next run's names.
        if not self._index_restarted:
            self._idxstart_spin.setValue(
                self._idxstart_spin.value() + self._files_written)
        if self._resume_check.isChecked() and self._daq is not None:
            self._start_monitor()

    def _on_error(self, msg: str) -> None:
        self._worker = None
        self._mode = 'idle'
        self._elapsed_timer.stop()
        self._slow_panel.set_busy(False)
        self._release_pulses()
        self._mode_label.setText('Acquisition error')
        self._update_buttons()
        QMessageBox.critical(self, 'Acquisition error', msg)

    # ── Slow data ───────────────────────────────────────────────────────────────

    def _prepare_slow(self) -> Optional[SlowConfig]:
        """
        Get the chart, the history and the log ready for a stream that is about
        to start, and return the configuration its worker should reduce with
        (None when slow data is off).

        The columns are fixed for a stream.  If they differ from what the
        history holds, the history starts over and so does the log file — rows
        of two different shapes cannot share either.
        """
        config = self._slow_panel.config()
        if self._daq is not None:
            config.slow_channels = [ch.name for ch in self._daq.config.channels
                                    if ch.slow]
            # A channel marked Slow is a request for slow data in itself.
            config.enabled = config.enabled or bool(config.slow_channels)
        self._show_slow_canvas(config.enabled)
        if not config.enabled or self._daq is None:
            self._slow_panel.set_status('')
            return None
        names = [ch.name for ch in self._daq.config.channels]
        self._slow_panel.set_sample_period(self._daq.config.dt)
        columns, units = slow_columns(config, names)

        skipped = [lk.name for lk in config.lockins
                   if f'{lk.name}_x' not in columns]
        if not columns:
            self._slow_panel.set_status(
                '⚠ nothing to compute — set a channel\'s Save mode to "Slow" '
                '(then Apply settings), or tick the statistics')
            return None
        self._slow_panel.set_status(
            f'⚠ skipped (channel not enabled, or duplicate name): '
            f'{", ".join(skipped)}' if skipped else
            f'{len(columns)} series at {config.rate_hz:g} Hz')

        if columns != self._slow_history.columns or (
                self._slow_config is not None
                and config.to_dict() != self._slow_config.to_dict()):
            # A changed rate, frequency or scale keeps the column names but not
            # their meaning, so the log starts a new file for it too.
            self._close_slow_log()
        if columns != self._slow_history.columns:
            self._slow_history.reset(columns, units)
            self._slow_panel.set_columns(columns, units)
        self._slow_history.units = dict(units)
        self._slow_history.set_capacity(self._slow_panel.keep_points())
        self._slow_config = config
        self._sync_slow_log(starting=True)
        return config

    def _sync_slow_log(self, starting: bool = False) -> None:
        """
        Open or close the continuous log to match the checkbox.

        A file is only opened when rows are about to arrive (a stream starting,
        or the box ticked mid-stream), so ticking it while idle does not litter
        the output directory with empty files.
        """
        if not self._slow_panel.log_enabled():
            self._close_slow_log()
            return
        if self._slow_log is not None or not self._slow_history.columns:
            return
        if not starting and self._mode == 'idle':
            self._slow_panel.set_log_text('logging starts with the next stream')
            return
        if self._slow_config is None:
            return
        directory = self._dir_edit.text().strip()
        try:
            os.makedirs(directory, exist_ok=True)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            path = os.path.join(
                directory, f'{self._prefix_edit.text()}slow_{stamp}.hdf5')
            self._slow_log = SlowLogWriter(
                path, self._slow_history.columns, self._slow_history.units,
                self._slow_config)
            self._slow_panel.set_log_text(f'logging to {path}')
        except Exception as exc:
            self._slow_log = None
            self._slow_panel.set_log_text(f'⚠ cannot open a log file: {exc}')

    def _close_slow_log(self) -> None:
        if self._slow_log is not None:
            path, rows = self._slow_log.path, self._slow_log.n_rows
            try:
                self._slow_log.close()
            except Exception:
                pass
            self._slow_log = None
            self._slow_panel.set_log_text(f'closed {path} ({rows} rows)')

    def _on_slow_new_log(self) -> None:
        self._close_slow_log()
        self._sync_slow_log()

    def _on_slow_apply(self) -> None:
        """New reduction settings: restart the monitor so they take hold."""
        if self._mode == 'monitor':
            self._stop_monitor_then('restart_monitor')
        elif self._mode == 'idle':
            self._show_slow_canvas(self._slow_panel.config().enabled)

    def _show_slow_canvas(self, show: bool) -> None:
        if show and not self._slow_canvas.isVisible():
            self._slow_canvas.setVisible(True)
            total = max(sum(self._plot_splitter.sizes()), 300)
            self._plot_splitter.setSizes([int(total * 0.62), int(total * 0.38)])
        elif not show:
            self._slow_canvas.setVisible(False)

    def _on_slow_rows(self, payload) -> None:
        columns, _units, times, values = payload
        if list(columns) != self._slow_history.columns:
            return                      # left over from the previous configuration
        self._slow_history.append(times, values)
        if self._slow_log is not None:
            try:
                self._slow_log.append(times, values)
            except Exception as exc:
                self._close_slow_log()
                self._slow_panel.set_log_text(f'⚠ log write failed: {exc}')
        self._slow_dirty = True

    def _on_slow_view_changed(self) -> None:
        self._slow_history.set_capacity(self._slow_panel.keep_points())
        self._slow_dirty = True

    def _on_slow_clear(self) -> None:
        self._slow_history.clear()
        self._slow_dirty = True

    def _draw_slow(self) -> None:
        if not self._slow_dirty or not self._slow_canvas.isVisible():
            return
        self._slow_dirty = False
        self._slow_canvas.render(self._slow_history,
                                 self._slow_panel.selected_series(),
                                 self._slow_panel.span_seconds())

    def _on_worker_message(self, text: str) -> None:
        if text.startswith('slow data:'):
            self._slow_panel.set_status(f'⚠ {text}')

    # ── Plot updates ────────────────────────────────────────────────────────────

    def _on_monitor_frame(self, display_data: Dict, psd_updated: bool) -> None:
        try:
            self._ingest_frame(display_data, psd_updated)
        finally:
            # Always acknowledge, even if drawing raised: the worker holds off
            # sending the next frame until we do, so missing this would freeze
            # the monitor rather than just skipping one frame.
            if self._monitor is not None:
                self._monitor.frame_consumed()

    def _ingest_frame(self, frame: Dict, psd_updated: bool,
                      wall_time: Optional[float] = None) -> None:
        """
        Take a new frame: complete it, store it, and show it if we are following.

        Frames are stored even while the display is held or scrolled back, so
        history keeps filling in and going Live again lands on the present.
        """
        if not psd_updated:
            self._carry_psd_forward(frame)
        meta = frame.setdefault('_meta', {})
        meta['wall_time'] = time.time() if wall_time is None else wall_time

        self._live_frame = frame
        self._history.append(frame)

        if self._follow_live and not self._freeze_check.isChecked():
            self._shown_frame  = frame
            self._selected_seq = self._history.newest_seq
            self._redraw_plots(psd_updated=psd_updated)
        self._update_history_ui()

    # ── History navigation ──────────────────────────────────────────────────────

    def _on_depth_changed(self) -> None:
        self._history.set_depth(self._depth_spin.value())
        if not self._history.enabled:
            self._on_go_live()
        self._update_history_ui()

    def _on_history_slider(self, value: int) -> None:
        if self._updating_slider:
            return
        newest = self._history.newest_seq
        self._follow_live = newest is not None and value >= newest
        self._select_seq(value)

    def _on_history_step(self, step: int) -> None:
        if self._selected_seq is None:
            return
        self._follow_live = False
        self._select_seq(self._selected_seq + step)

    def _select_seq(self, seq: int) -> None:
        """Display the stored frame with this sequence number, clamped in range."""
        oldest, newest = self._history.oldest_seq, self._history.newest_seq
        if oldest is None:
            return
        seq = int(np.clip(seq, oldest, newest))
        frame = self._history.get(seq)
        if frame is None:
            return
        self._selected_seq = seq
        self._shown_frame  = frame
        if seq >= newest:
            self._follow_live = True
        if not self._follow_live:
            # Browsing implies holding, so the two controls agree.
            self._freeze_check.blockSignals(True)
            self._freeze_check.setChecked(True)
            self._freeze_check.blockSignals(False)
        self._redraw_plots(psd_updated=True)
        self._update_history_ui()

    def _on_go_live(self) -> None:
        self._follow_live = True
        self._freeze_check.blockSignals(True)
        self._freeze_check.setChecked(False)
        self._freeze_check.blockSignals(False)
        if self._live_frame is not None:
            self._shown_frame  = self._live_frame
            self._selected_seq = self._history.newest_seq
            self._redraw_plots(psd_updated=True)
        self._update_history_ui()

    def _on_freeze_toggled(self, checked: bool) -> None:
        """Freeze holds the display; unfreezing resumes following the newest frame."""
        if checked:
            self._follow_live = False
            self._update_history_ui()
        else:
            self._on_go_live()

    def _update_history_ui(self) -> None:
        oldest, newest = self._history.oldest_seq, self._history.newest_seq
        have = oldest is not None
        for w in (self._hist_slider, self._hist_prev_btn, self._hist_next_btn,
                  self._save_frame_btn):
            w.setEnabled(have)
        self._live_btn.setEnabled(have and not self._follow_live)

        if not have:
            self._hist_label.setText('history off' if not self._history.enabled
                                     else 'empty')
            self._hist_mem_label.setText('')
            return

        self._updating_slider = True
        try:
            self._hist_slider.setRange(oldest, newest)
            if self._selected_seq is None:
                self._selected_seq = newest
            self._hist_slider.setValue(int(np.clip(self._selected_seq,
                                                   oldest, newest)))
        finally:
            self._updating_slider = False

        n_back = newest - (self._selected_seq if self._selected_seq is not None
                           else newest)
        stamp  = self._history.wall_time(self._selected_seq)
        newest_t = self._history.wall_time(newest)
        if self._follow_live and n_back == 0:
            text = f'live · {len(self._history)} frames'
        else:
            age = (newest_t - stamp) if (stamp and newest_t) else 0.0
            text = (f'−{n_back} of {len(self._history)} · '
                    f'{_fmt_duration(age)} back')
        if stamp:
            text += time.strftime('  %H:%M:%S', time.localtime(stamp))
        self._hist_label.setText(text)

        n     = len(self._history)
        mb    = self._history.nbytes() / 1e6
        span  = self._history.time_span()
        text  = f'{n} frames · {mb:.1f} MB · {_fmt_duration(span)} deep'
        if 0 < n < self._history.depth:
            # Project to a full buffer so the depth can be judged before it fills.
            text += (f'  (≈{mb / n * self._history.depth:.0f} MB, '
                     f'{_fmt_duration(span / n * self._history.depth)} when full)')
        self._hist_mem_label.setText(text)

    def _carry_psd_forward(self, display_data: Dict, previous: Optional[Dict] = None
                           ) -> None:
        """
        Copy the most recent spectrum into a trace-only frame.

        The cached frame is what re-renders when the user changes a PSD view
        option or toggles a channel; without this, a frame that arrived between
        two spectra would leave those controls with nothing to act on.  Only
        references are copied, so this is free.
        """
        previous = previous if previous is not None else (self._live_frame or {})
        for ch, d in display_data.items():
            if ch.startswith('_') or d.get('ff') is not None:
                continue
            old = previous.get(ch)
            if old and old.get('ff') is not None:
                d['ff']       = old['ff']
                d['pp']       = old['pp']
                d['peak_hz']  = old['peak_hz']

    def _on_preview_ready(self, display_data: Dict) -> None:
        """Mid-file update while recording — same rendering path as monitor mode."""
        has_psd = any(d.get('ff') is not None for ch, d in display_data.items()
                      if not ch.startswith('_'))
        self._ingest_frame(display_data, psd_updated=has_psd)

    def _on_data_ready(self, _idx: int, display_data: Dict) -> None:
        """End-of-file update: whole-file PSD, so a finer spectrum than the preview."""
        self._ingest_frame(display_data, psd_updated=True)

    def _on_display_selection_changed(self) -> None:
        """Re-render immediately when the user toggles a display checkbox."""
        self._redraw_plots()

    def _redraw_plots(self, psd_updated: bool = True) -> None:
        if not self._shown_frame:
            return
        selected = [ch for ch, cb in self._display_checks.items() if cb.isChecked()]
        self._canvas.render(self._shown_frame, selected, psd_updated)
        self._update_measurements(selected)

    def _update_measurements(self, selected: List[str]) -> None:
        """Refresh the per-channel readout strip under the plots."""
        if not self._rms_check.isChecked():
            self._measure_label.setText('')
            return
        parts = []
        for ch in selected:
            d = (self._shown_frame or {}).get(ch)
            if d is None:
                continue
            colour = _MPL_COLOURS.get(self._canvas._color_map.get(ch, ''), '#888')
            text = (f'<b>{ch}</b> rms {d["rms_mv"]:>8.3g} mV'
                    f'  mean {d["mean_mv"]:>8.3g} mV')
            if d.get('peak_hz'):
                text += f'  pk {_fmt_hz(d["peak_hz"])}'
            parts.append(f'<span style="color:{colour}">{text}</span>')
        self._measure_label.setText('&nbsp;&nbsp;|&nbsp;&nbsp;'.join(parts))

    def _on_save_frame(self) -> None:
        """
        Write the displayed frame to a .npz for offline inspection.

        This is display-resolution data — the decimated trace or min/max band and
        the plotted spectrum — not the raw waveform, which history does not keep.
        """
        frame = self._shown_frame
        if not frame:
            return
        meta  = frame.get('_meta') or {}
        stamp = time.strftime('%Y%m%d_%H%M%S',
                              time.localtime(meta.get('wall_time') or time.time()))
        default = os.path.join(self._dir_edit.text().strip() or os.getcwd(),
                              f'daq_frame_{stamp}.npz')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save displayed frame', default, 'NumPy archive (*.npz)')
        if not path:
            return

        arrays: Dict[str, np.ndarray] = {}
        info:   Dict[str, dict]       = {}
        for ch, d in frame.items():
            if ch.startswith('_'):
                continue
            for key in ('tt', 'trace_mv', 'lo_mv', 'hi_mv', 'ff', 'pp'):
                if d.get(key) is not None:
                    arrays[f'{ch}_{key}'] = np.asarray(d[key])
            info[ch] = {k: d.get(k) for k in
                        ('range_mv', 'rms_mv', 'mean_mv', 'peak_hz')}
        try:
            np.savez_compressed(
                path, meta=json.dumps({'frame': meta, 'channels': info,
                                       'note': 'display-resolution data'}),
                **arrays)
        except Exception as exc:
            QMessageBox.critical(self, 'Save failed', str(exc))
            return
        self._mode_label.setText(f'Saved {os.path.basename(path)}')

    def _on_zoom_changed(self, zoomed: bool) -> None:
        self._reset_view_btn.setEnabled(zoomed)
        self._reset_view_btn.setText('Reset view ⚠' if zoomed else 'Reset view')

    def _on_reset_view(self) -> None:
        self._canvas.reset_view()
        self._redraw_plots()

    def _on_progress(self, n_done: int, n_total: int, pressure: float) -> None:
        self._files_written = n_done
        self._progress_label.setText(f'Files: {n_done} / {n_total}')
        if pressure > 0:
            self._pressure_label.setText(f'{pressure:.2e} mbar')

    def _tick_elapsed(self) -> None:
        if self._start_time:
            elapsed = int(time.time() - self._start_time)
            h, rem = divmod(elapsed, 3600)
            m, s   = divmod(rem, 60)
            self._elapsed_label.setText(
                f'{h:d}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}')

    # ── Config helpers ──────────────────────────────────────────────────────────

    def _build_daq_config(self) -> Optional[AcquisitionConfig]:
        channels = []
        for ch_name in ALL_CHANNELS:
            row = self._ch_rows[ch_name]
            if not row['enable'].isChecked():
                continue
            channels.append(ChannelConfig(
                name=ch_name,
                range_idx=row['range'].currentIndex(),
                coupling=row['coupling'].currentText(),
                save_mean_only=(row['save'].currentText() in ('Mean', 'Slow')),
                slow=(row['save'].currentText() == 'Slow'),
            ))
        if not channels:
            QMessageBox.warning(self, 'No channels', 'Enable at least one channel.')
            return None
        try:
            buffer_size = int(self._buffer_edit.text())
        except ValueError:
            QMessageBox.warning(self, 'Invalid buffer size', 'Must be an integer.')
            return None
        return AcquisitionConfig(
            channels=channels,
            sample_interval=self._interval_spin.value(),
            sample_units=self._units_combo.currentText(),
            buffer_size=buffer_size,
        )

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, 'Select output directory', self._dir_edit.text())
        if d:
            self._dir_edit.setText(d)

    def _to_dict(self) -> dict:
        channels = {}
        for ch, row in self._ch_rows.items():
            channels[ch] = {
                'enabled':   row['enable'].isChecked(),
                'range_idx': row['range'].currentIndex(),
                'coupling':  row['coupling'].currentText(),
                'save_mode': row['save'].currentText(),
            }
        d = {
            'picoscope':       self._pico_combo.currentText(),
            'channels':        channels,
            'sample_interval': self._interval_spin.value(),
            'sample_units':    self._units_combo.currentText(),
            'buffer_size':     self._buffer_edit.text(),
            'n_files':         self._nfiles_spin.value(),
            'idx_start':       self._idxstart_spin.value(),
            'file_directory':  self._dir_edit.text(),
            'file_prefix':     self._prefix_edit.text(),
            'read_pressure':   self._pressure_check.isChecked(),
            'pressure_port':   self._pressure_port_edit.text(),
            'rate_unit':       self._rate_unit.currentText(),
            # live view
            'refresh':         self._refresh_spin.value(),
            'time_per_div':    self._tdiv_spin.value(),
            'n_div_x':         self._xdiv_spin.value(),
            'psd_span':        self._psdspan_spin.value(),
            'stream_chunk':    self._chunk_spin.value(),
            'monitor_stream':  self._stream_combo.currentText(),
            'nperseg':         self._nperseg_combo.currentText(),
            'psd_average':     self._psdavg_spin.value(),
            'psd_live':        self._psd_check.isChecked(),
            'psd_live_record': self._record_psd_check.isChecked(),
            'psd_refresh':     self._psd_refresh_spin.value(),
            'roll_mode':       self._roll_check.isChecked(),
            'history_depth':   self._depth_spin.value(),
            'psd_detrend':     self._detrend_check.isChecked(),
            'resume_monitor':  self._resume_check.isChecked(),
            # display
            'trace_points':    self._points_spin.value(),
            'psd_points':      self._psd_points_spin.value(),
            'envelope':        self._envelope_check.isChecked(),
            'ylim_mode':       self._ylim_combo.currentText(),
            'y_min':           self._ymin_edit.text(),
            'y_max':           self._ymax_edit.text(),
            'mv_per_div':      self._mvdiv_spin.value(),
            'n_div_y':         self._ydiv_spin.value(),
            'grid_divs':       self._griddiv_check.isChecked(),
            'psd_logx':        self._logx_check.isChecked(),
            'f_min':           self._fmin_edit.text(),
            'f_max':           self._fmax_edit.text(),
            'psd_ymin':        self._psd_ymin_edit.text(),
            'psd_ymax':        self._psd_ymax_edit.text(),
            'show_peak':       self._peak_check.isChecked(),
            'show_rms':        self._rms_check.isChecked(),
            # inspector
            'inspect':         self._inspector.to_dict(),
            'slow':            self._slow_panel.to_dict(),
        }
        if _HAS_PULSES:
            d['pulse'] = dict(
                self._pulse_panel.to_dict(),
                sequence_enabled=self._pulse_enable_check.isChecked(),
                files_per_amplitude=self._pulse_files_spin.value(),
                output_off_between=self._pulse_off_check.isChecked(),
                cycle=self._pulse_cycle_check.isChecked(),
                tag_filenames=self._pulse_tag_check.isChecked(),
                restart_index=self._pulse_restart_check.isChecked(),
            )
        return d

    def _from_dict(self, d: dict) -> None:
        if 'picoscope' in d:
            idx = self._pico_combo.findText(d['picoscope'])
            if idx >= 0:
                self._pico_combo.setCurrentIndex(idx)
        for ch, cfg in d.get('channels', {}).items():
            if ch not in self._ch_rows:
                continue
            row = self._ch_rows[ch]
            row['enable'].setChecked(cfg.get('enabled', False))
            row['range'].setCurrentIndex(cfg.get('range_idx', 7))
            for widget, key, default in [
                (row['coupling'], 'coupling',  'DC'),
                (row['save'],     'save_mode', 'Full'),
            ]:
                idx = widget.findText(cfg.get(key, default))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
        for attr, widget in [
            ('sample_interval', self._interval_spin),
            ('n_files',         self._nfiles_spin),
            ('idx_start',       self._idxstart_spin),
            ('refresh',         self._refresh_spin),
            ('psd_refresh',     self._psd_refresh_spin),
            ('psd_span',        self._psdspan_spin),
            ('stream_chunk',    self._chunk_spin),
            ('psd_average',     self._psdavg_spin),
            ('trace_points',    self._points_spin),
            ('psd_points',      self._psd_points_spin),
            ('history_depth',   self._depth_spin),
            ('n_div_x',         self._xdiv_spin),
            ('n_div_y',         self._ydiv_spin),
            ('mv_per_div',      self._mvdiv_spin),
        ]:
            if attr in d:
                widget.setValue(d[attr])
        # s/div is authoritative, but presets written before the scope-style
        # timebase only carry a total window length — convert it.
        if 'time_per_div' in d:
            self._tdiv_spin.setValue(d['time_per_div'])
        elif 'time_span' in d:
            self._tdiv_spin.setValue(float(d['time_span']) / self._xdiv_spin.value())
        for attr, widget in [
            ('sample_units', self._units_combo),
            ('nperseg',      self._nperseg_combo),
            ('ylim_mode',    self._ylim_combo),
            ('rate_unit',    self._rate_unit),
            ('monitor_stream', self._stream_combo),
        ]:
            if attr in d:
                idx = widget.findText(str(d[attr]))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
        for attr, widget in [
            ('buffer_size',    self._buffer_edit),
            ('file_directory', self._dir_edit),
            ('file_prefix',    self._prefix_edit),
            ('pressure_port',  self._pressure_port_edit),
            ('f_min',          self._fmin_edit),
            ('f_max',          self._fmax_edit),
            ('psd_ymin',       self._psd_ymin_edit),
            ('psd_ymax',       self._psd_ymax_edit),
            ('y_min',          self._ymin_edit),
            ('y_max',          self._ymax_edit),
        ]:
            if attr in d:
                widget.setText(str(d[attr]))
        for attr, widget in [
            ('read_pressure',  self._pressure_check),
            ('psd_live',       self._psd_check),
            ('psd_live_record', self._record_psd_check),
            ('psd_detrend',    self._detrend_check),
            ('resume_monitor', self._resume_check),
            ('envelope',       self._envelope_check),
            ('psd_logx',       self._logx_check),
            ('show_peak',      self._peak_check),
            ('show_rms',       self._rms_check),
            ('grid_divs',      self._griddiv_check),
            ('roll_mode',      self._roll_check),
        ]:
            if attr in d:
                widget.setChecked(bool(d[attr]))
        if isinstance(d.get('inspect'), dict):
            self._inspector.from_dict(d['inspect'])
        if isinstance(d.get('slow'), dict):
            self._slow_panel.from_dict(d['slow'])
        if _HAS_PULSES and isinstance(d.get('pulse'), dict):
            pulse = d['pulse']
            self._pulse_panel.from_dict(pulse)
            for key, widget in [
                ('sequence_enabled',   self._pulse_enable_check),
                ('output_off_between', self._pulse_off_check),
                ('cycle',              self._pulse_cycle_check),
                ('tag_filenames',      self._pulse_tag_check),
                ('restart_index',      self._pulse_restart_check),
            ]:
                if key in pulse:
                    widget.setChecked(bool(pulse[key]))
            if 'files_per_amplitude' in pulse:
                self._pulse_files_spin.setValue(int(pulse['files_per_amplitude']))
        self._on_interval_edited()      # resync the rate display with the interval
        self._on_timebase_changed()
        self._on_ylim_mode_changed()
        if _HAS_PULSES:
            self._update_pulse_summary()

    def _save_config(self) -> None:
        os.makedirs(PRESETS_DIR, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save config', PRESETS_DIR, 'JSON (*.json)')
        if path:
            with open(path, 'w') as f:
                json.dump(self._to_dict(), f, indent=2)

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load config', PRESETS_DIR, 'JSON (*.json)')
        if path:
            with open(path) as f:
                self._from_dict(json.load(f))

    def closeEvent(self, event) -> None:
        for worker in (self._monitor, self._worker):
            if worker is not None and worker.isRunning():
                worker.request_stop()
                worker.wait(5000)
        self._inspector.shutdown()
        self._close_slow_log()
        if _HAS_PULSES:
            self._pulse_panel.shutdown()      # output off, port closed
        if self._daq:
            try:
                self._daq.disconnect()
            except Exception:
                pass
        super().closeEvent(event)


# ── Entry points ───────────────────────────────────────────────────────────────

def launch_gui() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = DaqMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    launch_gui()
