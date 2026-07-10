"""
trap_monitor_gui.py — Automated trap-detection and capture monitor.

Continuously acquires camera frames, runs blob analysis, and saves a
capture whenever a particle has been stably trapped for a configurable
duration (default 10 s).  Each capture is stored in an HDF5 file and
can be reviewed and manually classified (single particle / cluster /
uncertain / reject).

Captures are organised into loading sessions.  Pressing Start Monitoring
waits a short delay (default 3 s, ≈ when spheres arrive after a loading)
and then opens a session timestamped at that moment; every capture taken
until monitoring stops carries that session's key as its `loading_id`,
and the pressure / manually-entered solution concentration are recorded
on the session.  This makes the capture rate per loading recoverable —
"Export all…" writes a loadings_index.csv with per-session capture
counts broken down by class.

Classification uses the same 2-D Gaussian fitting as camera_gui:
fitted σ is compared against a PSF σ reference to auto-classify each
capture.  The user can override the auto-classification per capture.

Run directly:
    python camera/trap_monitor_gui.py
    python camera/trap_monitor_gui.py --mock
"""

import sys
import os
import csv
import json
import time
import queue
import math
import threading
import datetime
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
    QSplitter, QSlider, QSizePolicy, QShortcut, QFrame, QListWidget,
    QListWidgetItem, QProgressBar, QScrollArea, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QFont, QKeySequence,
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.thorlabs_camera import ThorlabsCamera, MockCamera, analyze_frame
from camera.camera_gui import (
    CameraThread, AnalysisThread, LiveViewLabel,
    _CONTRAST_LUT, _HOT_LUT, _RESULT_COLORS, _DISPLAY_FPS,
)
from camera.backscatter_reader import (
    BackscatterReader, MockBackscatterReader, PICO_SERIALS, RANGE_LABELS,
)

# Pressure gauge is optional — its serial driver fails to import without
# pyserial / the gauge attached.  Manual entry always remains available.
try:
    from control.src.agilent_twisstorr_84fsag_control import (
        read_pressure as _hw_read_pressure)
    _HAS_PRESSURE = True
except Exception:
    _HAS_PRESSURE = False

_USER_CLASSES = ['single', 'cluster', 'uncertain', 'reject']

_CLS_COLORS = {
    'single':    '#00cc44',
    'cluster':   '#ff3333',
    'uncertain': '#ffaa00',
    'reject':    '#888888',
    '':          '#444444',
}

_CLS_LABELS = {
    'single':    'Single particle',
    'cluster':   'Cluster',
    'uncertain': 'Uncertain',
    'reject':    'Reject',
}


# ── Backscatter live-RMS thread ────────────────────────────────────────────────

class BackscatterThread(QThread):
    """
    Continuously reads short backscatter snippets from a connected reader and
    emits each result.  Owning the reader from a single thread keeps the
    PicoScope from being polled by two callers at once: the capture path reuses
    the latest emitted result rather than touching the scope directly.
    """
    rms_ready = pyqtSignal(dict)
    error     = pyqtSignal(str)

    def __init__(self, reader):
        super().__init__()
        self._reader = reader
        self._stop   = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        while not self._stop:
            try:
                res = self._reader.read()
            except Exception as exc:           # scope dropped / read failed
                self.error.emit(str(exc))
                break
            if self._stop:
                break
            self.rms_ready.emit(res)


# ── Pressure-gauge polling thread ───────────────────────────────────────────────

class PressureThread(QThread):
    """
    Polls the vacuum gauge off the GUI thread and caches the latest reading.

    The gauge driver opens a serial port and blocks on `readline()` for up to
    its 1 s timeout.  Doing that on the GUI thread (as the capture path used to)
    stalls the live-view QTimer and freezes the camera image for the duration.
    Here the read happens in the background; the window keeps the most recent
    value and uses it when saving a capture, so a capture never blocks.

    Polls every `interval` seconds while enabled; `trigger()` forces an
    immediate read (e.g. the "Read now" button) and `set_enabled(False)`
    pauses polling when auto-read is off.
    """
    reading = pyqtSignal(float, bool)   # (mbar, ok)

    def __init__(self, read_fn, port: str, interval: float = 3.0):
        super().__init__()
        self._read_fn  = read_fn
        self._port     = port
        self._interval = interval
        self._enabled  = True
        self._stop     = False
        self._wake     = threading.Event()

    def set_port(self, port: str) -> None:
        self._port = port
        self._wake.set()

    def set_enabled(self, on: bool) -> None:
        self._enabled = on
        if on:
            self._wake.set()

    def trigger(self) -> None:
        self._wake.set()

    def request_stop(self) -> None:
        self._stop = True
        self._wake.set()

    def run(self) -> None:
        while not self._stop:
            if self._enabled:
                try:
                    p = self._read_fn(port=self._port)
                except Exception:
                    p = -1.0
                ok = (p is not None) and (p >= 0)
                self.reading.emit(float(p) if ok else float('nan'), ok)
            self._wake.wait(self._interval)
            self._wake.clear()


# ── Trap tracker ──────────────────────────────────────────────────────────────

class TrapTracker:
    """
    Fires once per trap event when a blob has been continuously present
    for at least `min_duration` seconds.

    Gaps shorter than `gap_tolerance` seconds are bridged — a brief
    detection failure does not reset the clock.  After firing, stays
    silent until the trap is lost and a new one is acquired.
    """

    def __init__(self, min_duration: float = 10.0, gap_tolerance: float = 1.0):
        self.min_duration  = min_duration
        self.gap_tolerance = gap_tolerance
        self._first_seen:  float | None = None
        self._last_seen:   float | None = None
        self._triggered:   bool         = False

    def update(self, has_blob: bool, t: float | None = None) -> bool:
        """Returns True exactly once per trap event at the capture moment."""
        if t is None:
            t = time.time()

        if has_blob:
            if self._first_seen is None:
                self._first_seen = t
            self._last_seen = t
            if (t - self._first_seen >= self.min_duration
                    and not self._triggered):
                self._triggered = True
                return True
        else:
            if (self._last_seen is not None
                    and t - self._last_seen > self.gap_tolerance):
                self._first_seen = None
                self._last_seen  = None
                self._triggered  = False
        return False

    @property
    def duration(self) -> float:
        if self._first_seen is None:
            return 0.0
        return time.time() - self._first_seen

    @property
    def triggered(self) -> bool:
        return self._triggered


# ── Capture storage ───────────────────────────────────────────────────────────

class CaptureStore:
    """
    HDF5-backed storage for trap captures.

    Each capture is a group named by its timestamp key containing:
      datasets : frame (full uint16), crop (blob region uint16)
      attrs    : all analysis results, camera settings, trap timing,
                 auto_classification, user_classification (updatable)
    """

    def __init__(self, path: str):
        self.path = path

    def save(self, frame: np.ndarray, crop: np.ndarray,
             cx_in_crop: float, cy_in_crop: float, meta: dict,
             datasets: dict | None = None) -> str:
        """
        Store one capture.

        `datasets` (optional) maps name → {'data': ndarray, 'attrs': {...}} for
        auxiliary arrays such as the backscatter waveform.
        """
        import h5py
        key = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with h5py.File(self.path, 'a') as f:
            grp = f.require_group('captures').create_group(key)
            grp.create_dataset('frame', data=frame,
                               compression='gzip', compression_opts=1)
            grp.create_dataset('crop',  data=crop,
                               compression='gzip', compression_opts=1)
            grp.attrs['cx_in_crop']          = cx_in_crop
            grp.attrs['cy_in_crop']          = cy_in_crop
            grp.attrs['user_classification'] = ''
            for k, v in meta.items():
                grp.attrs[k] = str(v) if isinstance(v, str) else v
            for name, spec in (datasets or {}).items():
                ds = grp.create_dataset(
                    name, data=spec['data'],
                    compression='gzip', compression_opts=1)
                for ak, av in spec.get('attrs', {}).items():
                    ds.attrs[ak] = av
        return key

    def load_index(self) -> list:
        """
        Return [{key, meta}] for every capture, sorted oldest first, reading
        ONLY the lightweight attributes — never the frame/crop image arrays.

        The capture list and its grouping only need metadata, so this keeps a
        list refresh O(N attrs) instead of O(N decompressed frames); the heavy
        arrays are pulled lazily by `load_one` when a capture is selected.
        """
        import h5py
        if not os.path.exists(self.path):
            return []
        results = []
        with h5py.File(self.path, 'r') as f:
            g = f.get('captures')
            if g is None:
                return []
            for key in sorted(g.keys()):
                meta = {k: v for k, v in g[key].attrs.items()
                        if k not in ('cx_in_crop', 'cy_in_crop')}
                results.append({'key': key, 'meta': meta})
        return results

    def load_one(self, key: str) -> dict | None:
        """Return the full capture (frame, crop, centre, meta) for one key."""
        import h5py
        if not os.path.exists(self.path):
            return None
        with h5py.File(self.path, 'r') as f:
            g = f.get('captures')
            if g is None or key not in g:
                return None
            grp   = g[key]
            frame = grp['frame'][:]
            crop  = grp['crop'][:]
            cx    = float(grp.attrs.get('cx_in_crop', crop.shape[1] / 2))
            cy    = float(grp.attrs.get('cy_in_crop', crop.shape[0] / 2))
            meta  = {k: v for k, v in grp.attrs.items()
                     if k not in ('cx_in_crop', 'cy_in_crop')}
            return {'key': key, 'frame': frame, 'crop': crop,
                    'cx': cx, 'cy': cy, 'meta': meta}

    def load_all(self) -> list:
        """Returns list of full capture dicts sorted by key (oldest first)."""
        return [self.load_one(e['key']) for e in self.load_index()]

    def set_attr(self, key: str, name: str, value) -> None:
        """Update or add a single attribute on an existing capture."""
        import h5py
        with h5py.File(self.path, 'a') as f:
            g = f.get('captures')
            if g is not None and key in g:
                g[key].attrs[name] = value

    def set_user_classification(self, key: str, cls: str) -> None:
        self.set_attr(key, 'user_classification', cls)

    def delete(self, key: str) -> None:
        import h5py
        with h5py.File(self.path, 'a') as f:
            g = f.get('captures')
            if g is not None and key in g:
                del g[key]

    def load_aux(self, key: str, name: str):
        """Return an auxiliary dataset (e.g. 'backscatter') as (data, attrs)
        or (None, {}) if absent."""
        import h5py
        if not os.path.exists(self.path):
            return None, {}
        with h5py.File(self.path, 'r') as f:
            g = f.get('captures')
            if g is None or key not in g or name not in g[key]:
                return None, {}
            ds = g[key][name]
            return ds[:], {k: v for k, v in ds.attrs.items()}

    def count(self) -> int:
        import h5py
        if not os.path.exists(self.path):
            return 0
        with h5py.File(self.path, 'r') as f:
            g = f.get('captures')
            return len(g) if g else 0

    # ── Loading sessions ────────────────────────────────────────────────────
    # A "loading" is one press of Start Monitoring: the operator introduces
    # spheres and a fresh detection session begins a few seconds later.  Every
    # capture taken during that session carries the session key as its
    # `loading_id`, so the capture rate per loading is recoverable offline.

    def start_loading(self, meta: dict) -> str:
        """Create a loading-session group and return its key."""
        import h5py
        key = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with h5py.File(self.path, 'a') as f:
            grp = f.require_group('loadings').create_group(key)
            grp.attrs['n_captures'] = 0
            for k, v in meta.items():
                grp.attrs[k] = str(v) if isinstance(v, str) else v
        return key

    def increment_loading_count(self, key: str) -> None:
        """Bump the cached capture count on a loading session."""
        import h5py
        with h5py.File(self.path, 'a') as f:
            g = f.get('loadings')
            if g is not None and key in g:
                g[key].attrs['n_captures'] = \
                    int(g[key].attrs.get('n_captures', 0)) + 1

    def load_loadings(self) -> dict:
        """Return {loading_key: {attrs…}} for every recorded loading session."""
        import h5py
        if not os.path.exists(self.path):
            return {}
        out: dict = {}
        with h5py.File(self.path, 'r') as f:
            g = f.get('loadings')
            if g is None:
                return {}
            for key in g.keys():
                out[key] = {k: v for k, v in g[key].attrs.items()}
        return out

    def set_loading_attr(self, key: str, name: str, value) -> None:
        """Update or add a single attribute on a loading session."""
        import h5py
        with h5py.File(self.path, 'a') as f:
            g = f.get('loadings')
            if g is not None and key in g:
                g[key].attrs[name] = value

    def set_capture_attr_for_loading(self, loading_id: str, name: str,
                                     value) -> int:
        """Set an attr on every capture in a loading session; return the count."""
        import h5py
        n = 0
        with h5py.File(self.path, 'a') as f:
            g = f.get('captures')
            if g is None:
                return 0
            for k in g.keys():
                if str(g[k].attrs.get('loading_id', '')).strip() == loading_id:
                    g[k].attrs[name] = value
                    n += 1
        return n


# ── Capture viewer canvas ─────────────────────────────────────────────────────

class CaptureCanvas(FigureCanvasQTAgg):
    """
    Displays a saved capture: 2-D crop (hot colourmap) with 1σ/2σ
    Gaussian ellipses, plus Δx and Δy marginal projections.
    """

    def __init__(self):
        self.fig = Figure(facecolor='#1a1a1a')
        super().__init__(self.fig)
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        gs = GridSpec(2, 2, figure=self.fig,
                      width_ratios=[4, 1], height_ratios=[1, 4],
                      hspace=0.04, wspace=0.04,
                      left=0.10, right=0.97, top=0.91, bottom=0.08)
        self._ax_2d     = self.fig.add_subplot(gs[1, 0])
        self._ax_proj_x = self.fig.add_subplot(gs[0, 0], sharex=self._ax_2d)
        self._ax_proj_y = self.fig.add_subplot(gs[1, 1], sharey=self._ax_2d)
        self._style_axes()
        self.draw()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return w

    def _style_axes(self):
        for ax in (self._ax_2d, self._ax_proj_x, self._ax_proj_y):
            ax.set_facecolor('#1a1a1a')
            for sp in ax.spines.values():
                sp.set_color('#444')
            ax.tick_params(colors='#aaa', labelsize=7)

    def show_capture(self, crop: np.ndarray, cx: float, cy: float,
                     meta: dict) -> None:
        from matplotlib.patches import Ellipse
        for ax in (self._ax_2d, self._ax_proj_x, self._ax_proj_y):
            ax.cla()
        self._style_axes()

        ch, cw = crop.shape
        self._ax_2d.imshow(crop, cmap='hot', origin='upper',
                           aspect='equal', interpolation='nearest')

        sx = float(meta.get('sigma_x', 0) or 0)
        sy = float(meta.get('sigma_y', 0) or 0)
        fit_ok = str(meta.get('fit_ok', 'False')).lower() in ('true', '1')
        if fit_ok and sx > 0 and sy > 0:
            for n, ls, alpha in [(1, '-', 0.9), (2, '--', 0.55)]:
                self._ax_2d.add_patch(
                    Ellipse((cx, cy), width=2*n*sx, height=2*n*sy,
                            fill=False, edgecolor='cyan',
                            linestyle=ls, linewidth=1, alpha=alpha))

        self._ax_2d.axhline(cy, color='white', lw=0.5, alpha=0.35)
        self._ax_2d.axvline(cx, color='white', lw=0.5, alpha=0.35)
        self._ax_2d.set_xticks([0, cx, cw - 1])
        self._ax_2d.set_xticklabels(
            [f'{-cx:.0f}', '0', f'{cw-1-cx:.0f}'], fontsize=7)
        self._ax_2d.set_yticks([0, cy, ch - 1])
        self._ax_2d.set_yticklabels(
            [f'{-cy:.0f}', '0', f'{ch-1-cy:.0f}'], fontsize=7)
        self._ax_2d.set_xlabel('Δx (px)', fontsize=8, color='#ccc')
        self._ax_2d.set_ylabel('Δy (px)', fontsize=8, color='#ccc')

        # Δx projection
        proj_x = crop.sum(axis=0).astype(float)
        xs = np.arange(cw)
        self._ax_proj_x.plot(xs, proj_x, color='#ff6b35', lw=1.2)
        self._ax_proj_x.fill_between(xs, proj_x, alpha=0.25, color='#ff6b35')
        self._ax_proj_x.axvline(cx, color='white', lw=0.5, alpha=0.35)
        self._ax_proj_x.tick_params(labelbottom=False, labelsize=6, colors='#aaa')
        self._ax_proj_x.set_ylabel('ΣΔy', fontsize=7, color='#aaa')

        sigma = meta.get('sigma', float('nan'))
        try:
            sigma_str = f'  σ={float(sigma):.2f}px'
        except (TypeError, ValueError):
            sigma_str = ''
        auto_cls = str(meta.get('auto_classification', ''))
        self._ax_proj_x.set_title(f'{auto_cls}{sigma_str}',
                                  fontsize=9, color='white', pad=3)

        # Δy projection
        proj_y = crop.sum(axis=1).astype(float)
        ys = np.arange(ch)
        self._ax_proj_y.plot(proj_y, ys, color='#4ecdc4', lw=1.2)
        self._ax_proj_y.fill_betweenx(ys, proj_y, alpha=0.25, color='#4ecdc4')
        self._ax_proj_y.axhline(cy, color='white', lw=0.5, alpha=0.35)
        self._ax_proj_y.tick_params(labelleft=False, labelsize=6, colors='#aaa')
        self._ax_proj_y.set_xlabel('ΣΔx', fontsize=7, color='#aaa')
        self._ax_proj_y.invert_xaxis()

        self.draw_idle()

    def clear(self) -> None:
        for ax in (self._ax_2d, self._ax_proj_x, self._ax_proj_y):
            ax.cla()
        self._style_axes()
        self._ax_proj_x.set_title('No capture selected',
                                  fontsize=9, color='#888')
        self.draw_idle()


# ── Main window ───────────────────────────────────────────────────────────────

class TrapMonitorWindow(QMainWindow):

    def __init__(self, use_mock: bool = False):
        super().__init__()
        self.setWindowTitle('Trap Monitor — Auto-capture & Classification')
        self.setMinimumSize(1400, 780)

        self._use_mock = use_mock
        self._camera   = None
        self._cam_thread:      CameraThread | None   = None
        self._analysis_thread: AnalysisThread | None = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)

        self._last_frame:  np.ndarray | None = None
        self._background:  np.ndarray | None = None
        self._last_blobs:  list  = []
        self._last_cls:    str   = 'none'
        self._rotation_k:  int   = 0
        self._fps:         float = 0.0
        self._fps_counter: int   = 0
        self._fps_t0:      float = time.time()
        self._frame_count: int   = 0

        self._last_contrast:   tuple             = (-1, -1)
        self._disp_buf:        np.ndarray | None = None
        self._false_color_buf: np.ndarray | None = None

        self._trap_tracker  = TrapTracker()
        self._capture_store: CaptureStore | None = None
        self._captures:      list = []
        self._selected_key:  str | None = None
        self._selected_kind: str | None = None   # 'capture' | 'loading'
        self._selected_loading_id: str | None = None
        self._monitoring:    bool = False

        # Loading sessions: each Start press begins a delayed session whose
        # start timestamp groups the captures that follow it.
        self._starting:           bool      = False
        self._current_loading_id: str | None = None
        self._countdown_until:    float     = 0.0
        self._row_map:            list      = []   # list-row → capture index
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(100)
        self._countdown_timer.timeout.connect(self._tick_countdown)

        # Backscatter (PicoScope channel D) live RMS
        self._bs_reader = None
        self._bs_thread: BackscatterThread | None = None
        self._last_bs:   dict | None = None

        # Background pressure polling — keeps the latest gauge value without
        # ever blocking the GUI thread (see PressureThread).
        self._pressure_thread: PressureThread | None = None
        self._last_pressure:    float | None = None

        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._poll_frame)

        self._build_ui()
        self._setup_shortcuts()
        self._refresh_store()
        self._start_pressure_thread()
        QTimer.singleShot(0, self._on_scan)

    def _start_pressure_thread(self) -> None:
        """Launch background gauge polling (no-op in mock / without the driver)."""
        if self._use_mock or not _HAS_PRESSURE:
            return
        self._pressure_thread = PressureThread(
            _hw_read_pressure, self._pressure_port_edit.text().strip())
        self._pressure_thread.set_enabled(self._pressure_auto.isChecked())
        self._pressure_thread.reading.connect(self._on_pressure_reading)
        self._pressure_thread.start()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # Left: camera + settings (scrollable — many groups stacked)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(2, 2, 2, 2)
        lv.addWidget(self._build_camera_group())
        lv.addWidget(self._build_settings_group())
        lv.addWidget(self._build_analysis_group())
        lv.addWidget(self._build_conditions_group())
        lv.addWidget(self._build_backscatter_group())
        lv.addWidget(self._build_trap_group())
        lv.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(310)
        left_scroll.setMaximumWidth(330)

        # Centre: live view
        centre = QWidget()
        cv = QVBoxLayout(centre)
        cv.setContentsMargins(4, 4, 4, 4)
        self._live = LiveViewLabel()
        self._live.zoom_changed.connect(
            lambda z: self._zoom_label.setText(f'{z:.1f}×'))
        self._live.roi_changed.connect(self._on_roi_changed)
        cv.addWidget(self._live)
        cv.addWidget(self._build_contrast_bar())
        cv.addWidget(self._build_trap_status_bar())

        # Right: captures
        right = QWidget()
        right.setMinimumWidth(380)
        rv = QVBoxLayout(right)
        rv.addWidget(self._build_capture_panel())

        splitter.addWidget(left_scroll)
        splitter.addWidget(centre)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        container = QWidget()
        vl = QVBoxLayout(container)
        vl.addWidget(splitter)
        vl.addWidget(self._build_status_bar())
        self.setCentralWidget(container)

    def _build_camera_group(self) -> QGroupBox:
        g = QGroupBox('Camera')
        layout = QGridLayout(g)

        self._serial_combo = QComboBox()
        self._serial_combo.setEditable(True)
        self._serial_combo.setPlaceholderText('Serial or auto-detect')
        if self._use_mock:
            self._serial_combo.addItem('MOCK-001')
            self._serial_combo.setEnabled(False)

        self._scan_btn       = QPushButton('Scan')
        self._connect_btn    = QPushButton('Connect')
        self._disconnect_btn = QPushButton('Disconnect')
        self._disconnect_btn.setEnabled(False)

        self._scan_btn.clicked.connect(self._on_scan)
        self._connect_btn.clicked.connect(self._on_connect)
        self._disconnect_btn.clicked.connect(self._on_disconnect)

        layout.addWidget(QLabel('Serial:'),     0, 0)
        layout.addWidget(self._serial_combo,    0, 1, 1, 2)
        layout.addWidget(self._scan_btn,        1, 0)
        layout.addWidget(self._connect_btn,     1, 1)
        layout.addWidget(self._disconnect_btn,  1, 2)
        return g

    def _build_settings_group(self) -> QGroupBox:
        g = QGroupBox('Camera Settings')
        layout = QGridLayout(g)

        self._exposure_spin = QSpinBox()
        self._exposure_spin.setRange(100, 10_000_000)
        self._exposure_spin.setValue(10_000)
        self._exposure_spin.setSuffix(' μs')
        self._exposure_spin.setSingleStep(1000)
        self._exposure_spin.setEnabled(False)
        self._exposure_spin.valueChanged.connect(self._on_exposure_changed)

        self._gain_spin = QDoubleSpinBox()
        self._gain_spin.setRange(0.0, 480.0)
        self._gain_spin.setValue(0.0)
        self._gain_spin.setSingleStep(10.0)
        self._gain_spin.setEnabled(False)
        self._gain_spin.valueChanged.connect(self._on_gain_changed)

        self._rotation_combo = QComboBox()
        self._rotation_combo.addItems(['0°', '90° CW', '180°', '90° CCW'])
        self._rotation_combo.currentIndexChanged.connect(self._on_rotation_changed)

        self._bg_check = QCheckBox('Subtract background')
        self._bg_check.setEnabled(False)
        self._capture_bg_btn = QPushButton('Capture background (B)')
        self._capture_bg_btn.setEnabled(False)
        self._capture_bg_btn.clicked.connect(self._on_capture_background)

        layout.addWidget(QLabel('Exposure:'), 0, 0)
        layout.addWidget(self._exposure_spin, 0, 1)
        layout.addWidget(QLabel('Gain:'),     1, 0)
        layout.addWidget(self._gain_spin,     1, 1)
        layout.addWidget(QLabel('Rotation:'), 2, 0)
        layout.addWidget(self._rotation_combo, 2, 1)
        layout.addWidget(self._bg_check,      3, 0, 1, 2)
        layout.addWidget(self._capture_bg_btn, 4, 0, 1, 2)
        return g

    def _build_analysis_group(self) -> QGroupBox:
        g = QGroupBox('Analysis')
        layout = QGridLayout(g)

        self._thresh_spin = QSpinBox()
        self._thresh_spin.setRange(1, 4095)
        self._thresh_spin.setValue(200)

        self._minarea_spin = QSpinBox()
        self._minarea_spin.setRange(1, 10000)
        self._minarea_spin.setValue(1)

        self._psf_sigma_spin = QDoubleSpinBox()
        self._psf_sigma_spin.setRange(0.0, 50.0)
        self._psf_sigma_spin.setValue(0.0)
        self._psf_sigma_spin.setDecimals(2)
        self._psf_sigma_spin.setSingleStep(0.1)
        self._psf_sigma_spin.setSpecialValueText('not set')
        self._psf_sigma_spin.setToolTip(
            'Expected PSF σ for a single particle. 0 = not calibrated.')

        self._sigma_tol_spin = QDoubleSpinBox()
        self._sigma_tol_spin.setRange(0.1, 2.0)
        self._sigma_tol_spin.setValue(0.5)
        self._sigma_tol_spin.setSingleStep(0.1)
        self._sigma_tol_spin.setDecimals(1)

        self._overlay_check = QCheckBox('Show overlay')
        self._overlay_check.setChecked(True)

        layout.addWidget(QLabel('Threshold:'),     0, 0)
        layout.addWidget(self._thresh_spin,        0, 1)
        layout.addWidget(QLabel('Min area (px²):'),1, 0)
        layout.addWidget(self._minarea_spin,       1, 1)
        layout.addWidget(QLabel('PSF σ (px):'),    2, 0)
        layout.addWidget(self._psf_sigma_spin,     2, 1)
        layout.addWidget(QLabel('σ tolerance:'),   3, 0)
        layout.addWidget(self._sigma_tol_spin,     3, 1)
        layout.addWidget(self._overlay_check,      4, 0, 1, 2)
        return g

    def _build_conditions_group(self) -> QGroupBox:
        g = QGroupBox('Trap Conditions')
        layout = QGridLayout(g)

        self._pressure_edit = QLineEdit('1e-6')
        self._pressure_edit.setToolTip(
            'Chamber pressure in mbar, stored with each capture.\n'
            'Filled automatically when "Auto-read" is on and the gauge '
            'responds; otherwise type it in manually.')

        self._pressure_auto = QCheckBox('Auto-read from gauge')
        self._pressure_auto.setChecked(_HAS_PRESSURE)
        self._pressure_auto.setEnabled(_HAS_PRESSURE)
        self._pressure_auto.toggled.connect(self._on_pressure_auto_toggled)
        if not _HAS_PRESSURE:
            self._pressure_auto.setToolTip(
                'Gauge driver unavailable — manual entry only.')

        self._pressure_port_edit = QLineEdit('COM6')
        self._pressure_port_edit.setMaximumWidth(70)
        self._pressure_port_edit.editingFinished.connect(
            self._on_pressure_port_changed)

        self._pressure_read_btn = QPushButton('Read now')
        self._pressure_read_btn.setEnabled(_HAS_PRESSURE)
        self._pressure_read_btn.clicked.connect(self._on_read_pressure_now)

        self._pressure_src_lbl = QLabel('source: manual')
        self._pressure_src_lbl.setStyleSheet('color: #888; font-size: 8pt;')

        # Manually-entered solution concentration / dilution.  Stored on each
        # loading session and capture so the library is groupable by it.
        self._concentration_edit = QLineEdit('')
        self._concentration_edit.setPlaceholderText(
            'e.g. 1:1e5 in IPA, 100 nm silica')
        self._concentration_edit.setToolTip(
            'Nebuliser solution concentration / dilution.\n'
            'Recorded with each loading session and each capture; enter it '
            'before pressing Start Monitoring.')

        layout.addWidget(QLabel('Pressure (mbar):'), 0, 0)
        layout.addWidget(self._pressure_edit,        0, 1, 1, 2)
        layout.addWidget(self._pressure_auto,        1, 0, 1, 3)
        layout.addWidget(QLabel('Gauge port:'),      2, 0)
        layout.addWidget(self._pressure_port_edit,   2, 1)
        layout.addWidget(self._pressure_read_btn,    2, 2)
        layout.addWidget(self._pressure_src_lbl,     3, 0, 1, 3)
        layout.addWidget(QLabel('Solution conc.:'),  4, 0)
        layout.addWidget(self._concentration_edit,   4, 1, 1, 2)
        return g

    def _build_backscatter_group(self) -> QGroupBox:
        g = QGroupBox('Backscatter (PicoScope)')
        layout = QGridLayout(g)

        self._bs_serial_combo = QComboBox()
        for label in PICO_SERIALS:
            self._bs_serial_combo.addItem(label)

        self._bs_channel_combo = QComboBox()
        self._bs_channel_combo.addItems(list('ABCDEFGH'))
        self._bs_channel_combo.setCurrentText('D')

        self._bs_range_combo = QComboBox()
        self._bs_range_combo.addItems(RANGE_LABELS)
        self._bs_range_combo.setCurrentIndex(8)   # ±5 V

        self._bs_coupling_combo = QComboBox()
        self._bs_coupling_combo.addItems(['AC', 'DC'])

        self._bs_connect_btn    = QPushButton('Connect')
        self._bs_disconnect_btn = QPushButton('Disconnect')
        self._bs_disconnect_btn.setEnabled(False)
        self._bs_connect_btn.clicked.connect(self._on_bs_connect)
        self._bs_disconnect_btn.clicked.connect(self._on_bs_disconnect)

        self._bs_rms_lbl = QLabel('RMS: —')
        self._bs_rms_lbl.setFont(QFont('Arial', 11, QFont.Bold))
        self._bs_rms_lbl.setStyleSheet('color: #4ecdc4;')

        layout.addWidget(QLabel('Scope:'),    0, 0)
        layout.addWidget(self._bs_serial_combo, 0, 1, 1, 2)
        layout.addWidget(QLabel('Channel:'),  1, 0)
        layout.addWidget(self._bs_channel_combo, 1, 1)
        layout.addWidget(self._bs_coupling_combo, 1, 2)
        layout.addWidget(QLabel('Range:'),    2, 0)
        layout.addWidget(self._bs_range_combo, 2, 1, 1, 2)
        layout.addWidget(self._bs_connect_btn,    3, 0, 1, 2)
        layout.addWidget(self._bs_disconnect_btn, 3, 2)
        layout.addWidget(self._bs_rms_lbl,    4, 0, 1, 3)
        return g

    def _build_trap_group(self) -> QGroupBox:
        g = QGroupBox('Trap Detection')
        layout = QGridLayout(g)

        # Start / Stop monitoring (prominent toggle)
        self._monitor_btn = QPushButton('Start Monitoring')
        self._monitor_btn.setCheckable(True)
        self._monitor_btn.setEnabled(False)
        self._monitor_btn.setMinimumHeight(34)
        self._monitor_btn.setFont(QFont('Arial', 10, QFont.Bold))
        self._monitor_btn.setStyleSheet(
            'QPushButton { background: #2a5c2a; color: #ccc; border-radius: 4px; }'
            'QPushButton:checked { background: #8b0000; color: white; }'
            'QPushButton:disabled { background: #333; color: #666; }')
        self._monitor_btn.clicked.connect(self._on_toggle_monitoring)

        # Delay between pressing Start and the session actually beginning.
        # Spheres typically arrive ~3 s after loading, so the recorded loading
        # timestamp is taken when monitoring begins, not when the button is hit.
        self._start_delay_spin = QDoubleSpinBox()
        self._start_delay_spin.setRange(0.0, 60.0)
        self._start_delay_spin.setValue(3.0)
        self._start_delay_spin.setSuffix(' s')
        self._start_delay_spin.setSingleStep(0.5)
        self._start_delay_spin.setDecimals(1)
        self._start_delay_spin.setToolTip(
            'Delay between pressing Start and monitoring beginning.\n'
            'The loading timestamp is recorded when monitoring begins '
            '(≈ when spheres arrive).')

        # ROI controls
        self._roi_btn = QPushButton('Draw ROI')
        self._roi_btn.setCheckable(True)
        self._roi_btn.setEnabled(False)
        self._roi_btn.setToolTip(
            'Click and drag on the live view to set the analysis region.\n'
            'Only pixels inside the ROI are used for trap detection.')
        self._roi_btn.clicked.connect(self._on_roi_btn_clicked)

        self._roi_clear_btn = QPushButton('Clear ROI')
        self._roi_clear_btn.setEnabled(False)
        self._roi_clear_btn.clicked.connect(self._on_roi_clear)

        self._roi_label = QLabel('Full frame')
        self._roi_label.setAlignment(Qt.AlignCenter)
        self._roi_label.setStyleSheet('color: #00e5ff; font-size: 9pt;')

        # Timing
        self._min_dur_spin = QDoubleSpinBox()
        self._min_dur_spin.setRange(1.0, 300.0)
        self._min_dur_spin.setValue(10.0)
        self._min_dur_spin.setSuffix(' s')
        self._min_dur_spin.setSingleStep(1.0)
        self._min_dur_spin.setDecimals(1)
        self._min_dur_spin.setToolTip(
            'Capture triggered when a particle is detected continuously '
            'for this many seconds.')
        self._min_dur_spin.valueChanged.connect(
            lambda v: setattr(self._trap_tracker, 'min_duration', v))

        self._gap_tol_spin = QDoubleSpinBox()
        self._gap_tol_spin.setRange(0.1, 10.0)
        self._gap_tol_spin.setValue(1.0)
        self._gap_tol_spin.setSuffix(' s')
        self._gap_tol_spin.setSingleStep(0.5)
        self._gap_tol_spin.setDecimals(1)
        self._gap_tol_spin.setToolTip(
            'Detection gaps shorter than this are bridged '
            '(do not reset the timer).')
        self._gap_tol_spin.valueChanged.connect(
            lambda v: setattr(self._trap_tracker, 'gap_tolerance', v))

        self._capture_now_btn = QPushButton('Capture now')
        self._capture_now_btn.setEnabled(False)
        self._capture_now_btn.setToolTip('Manually trigger a capture')
        self._capture_now_btn.clicked.connect(self._do_capture)

        self._capture_path_edit = QLineEdit(r'E:\camera_data\trap_captures.hdf5')
        self._capture_path_edit.editingFinished.connect(self._refresh_store)
        browse_btn = QPushButton('…')
        browse_btn.setMaximumWidth(28)
        browse_btn.clicked.connect(self._browse_capture_file)

        self._capture_count_lbl = QLabel('0 captures')

        row = 0
        layout.addWidget(self._monitor_btn,        row, 0, 1, 3); row += 1
        layout.addWidget(QLabel('Start delay:'),   row, 0)
        layout.addWidget(self._start_delay_spin,   row, 1, 1, 2); row += 1
        layout.addWidget(self._roi_btn,            row, 0)
        layout.addWidget(self._roi_clear_btn,      row, 1, 1, 2); row += 1
        layout.addWidget(self._roi_label,          row, 0, 1, 3); row += 1
        layout.addWidget(QLabel('Min duration:'),  row, 0)
        layout.addWidget(self._min_dur_spin,       row, 1, 1, 2); row += 1
        layout.addWidget(QLabel('Gap tolerance:'), row, 0)
        layout.addWidget(self._gap_tol_spin,       row, 1, 1, 2); row += 1
        layout.addWidget(self._capture_now_btn,    row, 0, 1, 3); row += 1
        layout.addWidget(QLabel('File:'),          row, 0)
        layout.addWidget(self._capture_path_edit,  row, 1)
        layout.addWidget(browse_btn,               row, 2); row += 1
        layout.addWidget(self._capture_count_lbl,  row, 0, 1, 3)
        return g

    def _build_contrast_bar(self) -> QWidget:
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 2, 0, 2)

        self._contrast_auto = QCheckBox('Auto contrast')
        self._contrast_auto.setChecked(True)
        self._contrast_auto.stateChanged.connect(self._on_contrast_mode_changed)

        self._cmin_slider = QSlider(Qt.Horizontal)
        self._cmin_slider.setRange(0, 4095)
        self._cmin_slider.setValue(0)
        self._cmin_slider.setEnabled(False)
        self._cmax_slider = QSlider(Qt.Horizontal)
        self._cmax_slider.setRange(0, 4095)
        self._cmax_slider.setValue(4095)
        self._cmax_slider.setEnabled(False)

        self._cmin_label = QLabel('Min: 0')
        self._cmax_label = QLabel('Max: 4095')
        self._cmin_slider.valueChanged.connect(
            lambda v: self._cmin_label.setText(f'Min: {v}'))
        self._cmax_slider.valueChanged.connect(
            lambda v: self._cmax_label.setText(f'Max: {v}'))

        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        zoom_out = QPushButton('−'); zoom_out.setMaximumWidth(26)
        zoom_out.clicked.connect(lambda: self._live.zoom_by(1 / 1.25))
        self._zoom_label = QLabel('1.0×')
        self._zoom_label.setMinimumWidth(38)
        self._zoom_label.setAlignment(Qt.AlignCenter)
        zoom_in = QPushButton('+'); zoom_in.setMaximumWidth(26)
        zoom_in.clicked.connect(lambda: self._live.zoom_by(1.25))
        zoom_fit = QPushButton('Fit'); zoom_fit.setMaximumWidth(34)
        zoom_fit.clicked.connect(self._live.reset_zoom)

        self._false_color_check = QCheckBox('False colour')
        self._false_color_check.setChecked(False)

        hl.addWidget(self._contrast_auto)
        hl.addWidget(self._cmin_label)
        hl.addWidget(self._cmin_slider)
        hl.addWidget(self._cmax_label)
        hl.addWidget(self._cmax_slider)
        hl.addWidget(sep)
        hl.addWidget(zoom_out)
        hl.addWidget(self._zoom_label)
        hl.addWidget(zoom_in)
        hl.addWidget(zoom_fit)
        hl.addWidget(self._false_color_check)
        return w

    def _build_trap_status_bar(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(36)
        hl = QHBoxLayout(w)
        hl.setContentsMargins(6, 2, 6, 2)

        self._trap_label = QLabel('No particle')
        self._trap_label.setMinimumWidth(180)
        self._trap_label.setFont(QFont('Arial', 10))

        self._trap_progress = QProgressBar()
        self._trap_progress.setRange(0, 1000)
        self._trap_progress.setValue(0)
        self._trap_progress.setTextVisible(False)
        self._trap_progress.setFixedHeight(14)

        hl.addWidget(self._trap_label)
        hl.addWidget(self._trap_progress, 1)
        return w

    def _build_capture_panel(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)

        # ── Capture list ─────────────────────────────────────────────────
        list_grp = QGroupBox('Captures')
        list_layout = QVBoxLayout(list_grp)
        self._capture_list = QListWidget()
        self._capture_list.setMaximumHeight(160)
        self._capture_list.currentRowChanged.connect(self._on_capture_selected)
        list_layout.addWidget(self._capture_list)

        # Library management: open another library file, delete or export.
        lib_btns = QHBoxLayout()
        open_btn       = QPushButton('Open library…')
        self._delete_btn      = QPushButton('Delete')
        self._export_sel_btn  = QPushButton('Export selected…')
        self._export_all_btn  = QPushButton('Export all…')
        open_btn.clicked.connect(self._on_open_library)
        self._delete_btn.clicked.connect(self._on_delete_capture)
        self._export_sel_btn.clicked.connect(self._on_export_selected)
        self._export_all_btn.clicked.connect(self._on_export_all)
        for b in (self._delete_btn, self._export_sel_btn):
            b.setEnabled(False)
        for b in (open_btn, self._delete_btn,
                  self._export_sel_btn, self._export_all_btn):
            lib_btns.addWidget(b)
        list_layout.addLayout(lib_btns)

        # ── Capture viewer (2D + projections) ────────────────────────────
        self._capture_canvas = CaptureCanvas()

        # ── Classification buttons ────────────────────────────────────────
        cls_grp = QGroupBox('Classification')
        cls_layout = QHBoxLayout(cls_grp)
        self._cls_buttons: dict[str, QPushButton] = {}
        for cls in _USER_CLASSES:
            btn = QPushButton(_CLS_LABELS[cls])
            btn.setCheckable(True)
            btn.setEnabled(False)
            color = _CLS_COLORS[cls]
            btn.setStyleSheet(
                f'QPushButton:checked {{ background: {color}; color: white; '
                f'border: 2px solid {color}; }}')
            btn.clicked.connect(lambda checked, c=cls: self._on_classify(c))
            cls_layout.addWidget(btn)
            self._cls_buttons[cls] = btn

        # ── Label / notes (editable, saved into the capture) ───────────────
        notes_grp = QGroupBox('Label / Notes')
        notes_layout = QHBoxLayout(notes_grp)
        self._notes_edit = QLineEdit()
        self._notes_edit.setPlaceholderText(
            'e.g. sphere-A, 150 nm silica, run 3')
        self._notes_edit.setEnabled(False)
        self._notes_edit.returnPressed.connect(self._on_save_notes)
        self._notes_save_btn = QPushButton('Save')
        self._notes_save_btn.setEnabled(False)
        self._notes_save_btn.clicked.connect(self._on_save_notes)
        notes_layout.addWidget(self._notes_edit, 1)
        notes_layout.addWidget(self._notes_save_btn)

        # ── Conditions (editable — correct mistakes after the fact) ────────
        cond_grp = QGroupBox('Conditions (editable)')
        cond_layout = QGridLayout(cond_grp)
        self._cond_target_lbl = QLabel('No selection')
        self._cond_target_lbl.setStyleSheet('color: #888; font-size: 8pt;')
        self._cond_pressure_edit = QLineEdit()
        self._cond_pressure_edit.setPlaceholderText('mbar, e.g. 1e-6')
        self._cond_conc_edit = QLineEdit()
        self._cond_conc_edit.setPlaceholderText('e.g. 1:1e5 in IPA')
        self._cond_apply_loading = QCheckBox('Apply to whole loading')
        self._cond_apply_loading.setToolTip(
            'Also write these values to every capture in the same loading '
            'session (and to the session record).')
        self._cond_save_btn = QPushButton('Save conditions')
        self._cond_save_btn.clicked.connect(self._on_save_conditions)
        self._cond_pressure_edit.returnPressed.connect(self._on_save_conditions)
        self._cond_conc_edit.returnPressed.connect(self._on_save_conditions)
        cond_layout.addWidget(self._cond_target_lbl,    0, 0, 1, 2)
        cond_layout.addWidget(QLabel('Pressure (mbar):'), 1, 0)
        cond_layout.addWidget(self._cond_pressure_edit, 1, 1)
        cond_layout.addWidget(QLabel('Solution conc.:'), 2, 0)
        cond_layout.addWidget(self._cond_conc_edit,     2, 1)
        cond_layout.addWidget(self._cond_apply_loading, 3, 0, 1, 2)
        cond_layout.addWidget(self._cond_save_btn,      4, 0, 1, 2)
        for wdg in (self._cond_pressure_edit, self._cond_conc_edit,
                    self._cond_apply_loading, self._cond_save_btn):
            wdg.setEnabled(False)

        # ── Metadata ──────────────────────────────────────────────────────
        meta_grp = QGroupBox('Metadata')
        meta_layout = QVBoxLayout(meta_grp)
        self._meta_label = QLabel('—')
        self._meta_label.setFont(QFont('Courier', 8))
        self._meta_label.setWordWrap(True)
        self._meta_label.setAlignment(Qt.AlignTop)
        meta_scroll = QScrollArea()
        meta_scroll.setWidget(self._meta_label)
        meta_scroll.setWidgetResizable(True)
        meta_scroll.setMaximumHeight(120)
        meta_layout.addWidget(meta_scroll)

        vl.addWidget(list_grp)
        vl.addWidget(self._capture_canvas, 1)
        vl.addWidget(cls_grp)
        vl.addWidget(notes_grp)
        vl.addWidget(cond_grp)
        vl.addWidget(meta_grp)
        return w

    def _build_status_bar(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(28)
        hl = QHBoxLayout(w)
        hl.setContentsMargins(6, 2, 6, 2)
        self._fps_label    = QLabel('FPS: —')
        self._frame_label  = QLabel('Frame: 0')
        self._size_label   = QLabel('—')
        self._status_label = QLabel('Not connected')
        for lbl in [self._fps_label, self._frame_label, self._size_label]:
            hl.addWidget(lbl)
            sep = QFrame(); sep.setFrameShape(QFrame.VLine)
            hl.addWidget(sep)
        hl.addStretch()
        hl.addWidget(self._status_label)
        return w

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence('B'), self).activated.connect(
            self._on_capture_background)
        QShortcut(QKeySequence('+'), self).activated.connect(
            lambda: self._live.zoom_by(1.25))
        QShortcut(QKeySequence('-'), self).activated.connect(
            lambda: self._live.zoom_by(1 / 1.25))
        QShortcut(QKeySequence('0'), self).activated.connect(
            self._live.reset_zoom)

    # ── Camera actions ────────────────────────────────────────────────────────

    def _on_scan(self) -> None:
        cls = MockCamera if self._use_mock else ThorlabsCamera
        try:
            serials = cls.list_serials()
        except Exception as e:
            self._status_label.setText(f'Scan error: {e}')
            return
        self._serial_combo.clear()
        for s in serials:
            self._serial_combo.addItem(s)
        self._status_label.setText(
            f'{len(serials)} camera(s) found' if serials else 'No cameras found')

    def _on_connect(self) -> None:
        serial  = self._serial_combo.currentText().strip() or None
        cam_cls = MockCamera if self._use_mock else ThorlabsCamera
        try:
            cam = cam_cls()
            cam.connect(serial)
        except Exception as exc:
            QMessageBox.critical(self, 'Connection error', str(exc))
            return

        self._camera = cam
        self._frame_queue  = queue.Queue(maxsize=2)
        self._cam_thread   = CameraThread(cam, self._frame_queue)
        self._cam_thread.error.connect(self._on_camera_error)
        self._cam_thread.start()
        self._analysis_thread = AnalysisThread()
        self._analysis_thread.result_ready.connect(self._on_analysis_result)
        self._analysis_thread.start()

        W, H = cam.width, cam.height
        self._size_label.setText(f'{W}×{H}  {cam.bit_depth}-bit')
        self._exposure_spin.blockSignals(True)
        lo, hi = cam.exposure_range_us()
        self._exposure_spin.setRange(lo, hi)
        self._exposure_spin.setValue(cam.exposure_us)
        self._exposure_spin.blockSignals(False)
        self._gain_spin.blockSignals(True)
        glo, ghi = cam.gain_range()
        self._gain_spin.setRange(glo, ghi)
        self._gain_spin.setValue(cam.gain)
        self._gain_spin.blockSignals(False)
        self._cmax_slider.setRange(0, (1 << cam.bit_depth) - 1)
        self._cmax_slider.setValue((1 << cam.bit_depth) - 1)
        self._thresh_spin.setRange(1, (1 << cam.bit_depth) - 1)

        for w in [self._exposure_spin, self._gain_spin,
                  self._capture_bg_btn, self._capture_now_btn,
                  self._monitor_btn, self._roi_btn]:
            w.setEnabled(True)
        self._disconnect_btn.setEnabled(True)
        self._connect_btn.setEnabled(False)

        self._trap_tracker = TrapTracker(
            self._min_dur_spin.value(), self._gap_tol_spin.value())
        self._display_timer.start(1000 // _DISPLAY_FPS)
        self._fps_t0 = time.time(); self._fps_counter = 0
        self._status_label.setText(
            'Streaming (mock)…' if self._use_mock else 'Streaming…')

    def _on_disconnect(self) -> None:
        self._display_timer.stop()
        if self._analysis_thread:
            self._analysis_thread.request_stop()
            self._analysis_thread.wait(1000)
            self._analysis_thread = None
        if self._cam_thread:
            self._cam_thread.request_stop()
            self._cam_thread.wait(2000)
            self._cam_thread = None
        if self._camera:
            self._camera.disconnect()
            self._camera = None
        self._countdown_timer.stop()
        self._starting = False
        self._monitoring = False
        self._current_loading_id = None
        for w in [self._exposure_spin, self._gain_spin,
                  self._capture_bg_btn, self._capture_now_btn,
                  self._monitor_btn, self._roi_btn, self._roi_clear_btn]:
            w.setEnabled(False)
        self._monitor_btn.setChecked(False)
        self._monitor_btn.setText('Start Monitoring')
        self._live.clear_roi()
        self._roi_label.setText('Full frame')
        self._disconnect_btn.setEnabled(False)
        self._connect_btn.setEnabled(True)
        self._live.setText('No camera connected')
        self._live.setPixmap(QPixmap())
        self._trap_label.setText('No particle')
        self._trap_progress.setValue(0)
        self._status_label.setText('Disconnected')
        self._fps_label.setText('FPS: —')

    def _on_camera_error(self, msg: str) -> None:
        self._display_timer.stop()
        QMessageBox.critical(self, 'Camera error', msg)

    def _on_exposure_changed(self, v: int) -> None:
        if self._camera:
            self._camera.exposure_us = v

    def _on_gain_changed(self, v: float) -> None:
        if self._camera:
            self._camera.gain = v

    def _on_rotation_changed(self, idx: int) -> None:
        self._rotation_k = [0, 3, 2, 1][idx]
        if self._background is not None:
            self._background = None
            self._bg_check.setChecked(False)
            self._bg_check.setEnabled(False)
            self._status_label.setText('Rotation changed — recapture background')
        self._disp_buf = self._false_color_buf = None
        self._live.reset_zoom()

    def _on_capture_background(self) -> None:
        if self._last_frame is None:
            return
        self._background = self._last_frame.copy()
        self._bg_check.setEnabled(True)
        self._bg_check.setChecked(True)
        self._status_label.setText('Background captured')

    def _on_contrast_mode_changed(self) -> None:
        manual = not self._contrast_auto.isChecked()
        self._cmin_slider.setEnabled(manual)
        self._cmax_slider.setEnabled(manual)

    # ── Pressure ──────────────────────────────────────────────────────────────
    # All gauge I/O happens on PressureThread; the GUI only consumes the cached
    # value, so neither a capture nor "Read now" blocks the live view.

    def _on_pressure_reading(self, value: float, ok: bool) -> None:
        """Receive a background gauge reading and cache / display it."""
        self._last_pressure = value if ok else None
        if not self._pressure_auto.isChecked():
            return
        if ok:
            self._pressure_edit.setText(f'{value:.3e}')
            self._pressure_src_lbl.setText('source: gauge')
        else:
            self._pressure_src_lbl.setText('source: gauge read FAILED')

    def _on_pressure_auto_toggled(self, on: bool) -> None:
        if self._pressure_thread is not None:
            self._pressure_thread.set_enabled(on)
        self._pressure_src_lbl.setText(
            'source: gauge' if on else 'source: manual')

    def _on_pressure_port_changed(self) -> None:
        if self._pressure_thread is not None:
            self._pressure_thread.set_port(self._pressure_port_edit.text().strip())

    def _on_read_pressure_now(self) -> None:
        if self._pressure_thread is not None:
            self._pressure_thread.trigger()       # non-blocking; result via signal
            self._status_label.setText('Reading pressure…')

    def _current_pressure(self) -> tuple[float, str]:
        """
        Resolve the pressure to store with a capture — without any blocking I/O.

        Returns (value_mbar, source).  When auto-read is on and the background
        poll has a fresh value, that is used ('gauge'); otherwise we fall back to
        the manual field so a capture is never blocked or dropped.
        """
        if self._pressure_auto.isChecked() and self._last_pressure is not None:
            return self._last_pressure, 'gauge'
        try:
            return float(self._pressure_edit.text().strip()), 'manual'
        except ValueError:
            return float('nan'), 'manual'

    # ── Backscatter (PicoScope) ─────────────────────────────────────────────────

    def _on_bs_connect(self) -> None:
        if self._bs_thread is not None:
            return
        label  = self._bs_serial_combo.currentText()
        serial = PICO_SERIALS.get(label, b'MOCK')
        cls = MockBackscatterReader if self._use_mock else BackscatterReader
        reader = cls(
            serial=serial,
            channel=self._bs_channel_combo.currentText(),
            range_idx=self._bs_range_combo.currentIndex(),
            coupling=self._bs_coupling_combo.currentText(),
        )
        try:
            reader.connect()
        except Exception as exc:
            QMessageBox.critical(self, 'PicoScope error', str(exc))
            return
        self._bs_reader = reader
        self._bs_thread = BackscatterThread(reader)
        self._bs_thread.rms_ready.connect(self._on_bs_rms)
        self._bs_thread.error.connect(self._on_bs_error)
        self._bs_thread.start()

        self._bs_connect_btn.setEnabled(False)
        self._bs_disconnect_btn.setEnabled(True)
        for w in (self._bs_serial_combo, self._bs_channel_combo,
                  self._bs_range_combo, self._bs_coupling_combo):
            w.setEnabled(False)
        self._status_label.setText('Backscatter connected')

    def _on_bs_disconnect(self) -> None:
        if self._bs_thread is not None:
            self._bs_thread.request_stop()
            self._bs_thread.wait(3000)
            self._bs_thread = None
        if self._bs_reader is not None:
            self._bs_reader.disconnect()
            self._bs_reader = None
        self._last_bs = None
        self._bs_rms_lbl.setText('RMS: —')
        self._bs_connect_btn.setEnabled(True)
        self._bs_disconnect_btn.setEnabled(False)
        for w in (self._bs_serial_combo, self._bs_channel_combo,
                  self._bs_range_combo, self._bs_coupling_combo):
            w.setEnabled(True)

    def _on_bs_rms(self, res: dict) -> None:
        self._last_bs = res
        self._bs_rms_lbl.setText(
            f'RMS: {res["rms_mv"]:.3g} mV   (pk-pk {res["pk2pk_mv"]:.3g})')

    def _on_bs_error(self, msg: str) -> None:
        self._on_bs_disconnect()
        self._status_label.setText(f'Backscatter error: {msg}')

    # ── Frame loop ────────────────────────────────────────────────────────────

    def _poll_frame(self) -> None:
        try:
            frame = self._frame_queue.get_nowait()
        except queue.Empty:
            return

        if self._rotation_k:
            frame = np.ascontiguousarray(np.rot90(frame, self._rotation_k))
        self._last_frame = frame
        self._frame_count += 1
        self._fps_counter += 1

        now = time.time()
        if now - self._fps_t0 >= 1.0:
            self._fps = self._fps_counter / (now - self._fps_t0)
            self._fps_counter = 0
            self._fps_t0 = now
            self._fps_label.setText(f'FPS: {self._fps:.1f}')
        self._frame_label.setText(f'Frame: {self._frame_count}')

        display_image = frame
        if self._bg_check.isChecked() and self._background is not None:
            display_image = np.clip(
                frame.astype(np.int32) - self._background.astype(np.int32),
                0, None).astype(np.uint16)

        if self._analysis_thread is not None:
            analysis_image = display_image
            roi_offset     = (0, 0)
            roi = self._live.roi
            if roi is not None:
                rx0, ry0, rx1, ry1 = roi
                rx0 = max(0, rx0);  ry0 = max(0, ry0)
                rx1 = min(frame.shape[1], rx1)
                ry1 = min(frame.shape[0], ry1)
                if rx1 > rx0 and ry1 > ry0:
                    analysis_image = display_image[ry0:ry1, rx0:rx1]
                    roi_offset     = (rx0, ry0)
            self._analysis_thread.submit(
                analysis_image,
                self._thresh_spin.value(),
                self._minarea_spin.value(),
                self._psf_sigma_spin.value(),
                self._sigma_tol_spin.value(),
                roi_offset,
            )

        qimg = self._frame_to_qimage(display_image)
        self._live.update_frame(
            qimg, self._last_blobs, self._last_cls,
            self._overlay_check.isChecked(),
            frame.shape[1], frame.shape[0],
        )

    def _on_analysis_result(self, result: dict) -> None:
        self._last_blobs = result['blobs']
        self._last_cls   = result['classification']

        # During the pre-start delay the countdown timer owns the status bar;
        # don't let the trap clock accumulate before the loading begins.
        if self._starting:
            self._trap_progress.setValue(0)
            return

        has_blob = len(result['blobs']) > 0
        now      = time.time()
        trigger  = self._trap_tracker.update(has_blob, now)

        # Update trap status bar
        dur  = self._trap_tracker.duration
        tmax = self._trap_tracker.min_duration
        if not self._monitoring:
            self._trap_progress.setValue(0)
            self._trap_label.setText('Monitoring stopped')
        elif has_blob or self._trap_tracker.triggered:
            pct = min(dur / tmax, 1.0) if tmax > 0 else 1.0
            self._trap_progress.setValue(int(pct * 1000))
            status = ('Captured ✓' if self._trap_tracker.triggered
                      else f'{dur:.1f} s / {tmax:.0f} s')
            self._trap_label.setText(f'Trap: {status}')
        else:
            self._trap_progress.setValue(0)
            self._trap_label.setText('Monitoring — no particle')

        if trigger and self._monitoring:
            self._do_capture()

    def _on_toggle_monitoring(self, checked: bool) -> None:
        if checked:
            self._start_countdown()
        else:
            self._stop_monitoring()

    def _start_countdown(self) -> None:
        """Begin the pre-start delay, then activate monitoring."""
        self._monitoring = False
        delay = self._start_delay_spin.value()
        if delay <= 0:
            self._activate_monitoring()
            return
        self._starting = True
        self._countdown_until = time.time() + delay
        self._trap_progress.setValue(0)
        self._monitor_btn.setText(f'Starting in {delay:.1f}s — cancel')
        self._trap_label.setText(f'Loading… monitoring in {delay:.1f} s')
        self._status_label.setText(
            f'Loading — monitoring starts in {delay:.1f} s')
        self._countdown_timer.start()

    def _tick_countdown(self) -> None:
        remaining = self._countdown_until - time.time()
        if remaining <= 0:
            self._countdown_timer.stop()
            self._activate_monitoring()
        else:
            self._monitor_btn.setText(f'Starting in {remaining:.1f}s — cancel')
            self._trap_label.setText(
                f'Loading… monitoring in {remaining:.1f} s')

    def _activate_monitoring(self) -> None:
        """Monitoring goes live: record the loading session and start the clock."""
        self._starting   = False
        self._monitoring = True
        # Fresh tracker so the trap clock starts from now.
        self._trap_tracker = TrapTracker(
            self._min_dur_spin.value(), self._gap_tol_spin.value())
        self._start_loading_session()
        self._monitor_btn.setChecked(True)
        self._monitor_btn.setText('Stop Monitoring')
        self._status_label.setText(
            'Monitoring started (loading recorded)')

    def _stop_monitoring(self) -> None:
        """Stop monitoring, or cancel a pending delayed start."""
        if self._starting:
            self._countdown_timer.stop()
            self._starting = False
            self._monitor_btn.setChecked(False)
            self._monitor_btn.setText('Start Monitoring')
            self._trap_progress.setValue(0)
            self._trap_label.setText('Monitoring stopped')
            self._status_label.setText('Start cancelled')
            return
        self._monitoring = False
        self._current_loading_id = None
        self._monitor_btn.setChecked(False)
        self._monitor_btn.setText('Start Monitoring')
        self._trap_progress.setValue(0)
        self._trap_label.setText('Monitoring stopped')
        self._status_label.setText('Monitoring stopped')

    def _start_loading_session(self) -> None:
        """Record a new loading session keyed by the moment monitoring began."""
        self._current_loading_id = None
        if self._capture_store is None:
            return
        pressure, p_source = self._current_pressure()
        now = datetime.datetime.now()
        try:
            self._current_loading_id = self._capture_store.start_loading({
                'start_timestamp':        now.isoformat(timespec='seconds'),
                'start_time_unix':        now.timestamp(),
                'pressure_mbar':          pressure,
                'pressure_source':        p_source,
                'solution_concentration': self._concentration_edit.text().strip(),
                'start_delay_s':          self._start_delay_spin.value(),
            })
            self._load_captures()   # show the new (empty) loading header
        except Exception as exc:
            self._status_label.setText(f'Loading record failed: {exc}')

    def _on_roi_btn_clicked(self, checked: bool) -> None:
        self._live.set_roi_mode(checked)
        self._roi_btn.setText('Cancel ROI' if checked else 'Draw ROI')

    def _on_roi_clear(self) -> None:
        self._live.clear_roi()

    def _on_roi_changed(self, roi) -> None:
        self._roi_btn.setChecked(False)
        self._roi_btn.setText('Draw ROI')
        if roi is None:
            self._roi_label.setText('Full frame')
            self._roi_clear_btn.setEnabled(False)
        else:
            x0, y0, x1, y1 = roi
            self._roi_label.setText(f'ROI: {x1-x0}×{y1-y0} px  ({x0},{y0})')
            self._roi_clear_btn.setEnabled(True)

    def _do_capture(self) -> None:
        """Save the current frame as a trap capture."""
        if self._last_frame is None or self._capture_store is None:
            return

        frame = self._last_frame
        H, W  = frame.shape
        blobs = self._last_blobs

        # Crop around brightest blob if present, otherwise full frame
        if blobs:
            b  = max(blobs, key=lambda b: b['peak'])
            cx, cy = b['cx'], b['cy']
            sigma_est = b.get('sigma', float('nan'))
            if math.isnan(sigma_est):
                sigma_est = b.get('radius', 5)
            half = max(30, int(sigma_est * 6))
            x0 = max(0, int(cx) - half);  x1 = min(W, int(cx) + half + 1)
            y0 = max(0, int(cy) - half);  y1 = min(H, int(cy) + half + 1)
            crop       = frame[y0:y1, x0:x1]
            cx_in_crop = cx - x0
            cy_in_crop = cy - y0
        else:
            b          = {}
            crop       = frame
            cx_in_crop = W / 2.0
            cy_in_crop = H / 2.0

        pressure, p_source = self._current_pressure()

        meta = {
            'timestamp':          datetime.datetime.now().isoformat(timespec='seconds'),
            'loading_id':         self._current_loading_id or '',
            'solution_concentration': self._concentration_edit.text().strip(),
            'trap_duration_s':    self._trap_tracker.duration,
            'auto_classification': self._last_cls,
            'pressure_mbar':      pressure,
            'pressure_source':    p_source,
            'notes':              '',
            'exposure_us':        self._camera.exposure_us if self._camera else 0,
            'gain':               self._camera.gain        if self._camera else 0.0,
            'bit_depth':          self._camera.bit_depth   if self._camera else 0,
            'camera_width':       W,
            'camera_height':      H,
            'sigma':              b.get('sigma',       float('nan')),
            'sigma_x':            b.get('sigma_x',     float('nan')),
            'sigma_y':            b.get('sigma_y',     float('nan')),
            'fit_quality':        b.get('fit_quality', float('nan')),
            'fit_ok':             b.get('fit_ok',      False),
            'peak':               b.get('peak',        0),
            'cx':                 b.get('cx',          cx_in_crop),
            'cy':                 b.get('cy',          cy_in_crop),
            'psf_sigma_ref':      self._psf_sigma_spin.value(),
        }

        # Attach the most recent backscatter snippet, if the scope is connected.
        datasets = None
        bs = self._last_bs
        if bs is not None:
            meta['backscatter_rms_mv']   = bs['rms_mv']
            meta['backscatter_mean_mv']  = bs['mean_mv']
            meta['backscatter_pk2pk_mv'] = bs['pk2pk_mv']
            meta['backscatter_channel']  = bs['channel']
            meta['backscatter_range_mv'] = bs['range_mv']
            meta['backscatter_coupling'] = bs['coupling']
            datasets = {
                'backscatter': {
                    'data':  bs['waveform'],
                    'attrs': {
                        'adc2mv':   bs['adc2mv'],
                        'delta_t':  bs['dt'],
                        'fs':       bs['fs'],
                        'rms_mv':   bs['rms_mv'],
                        'channel':  bs['channel'],
                        'range_mv': bs['range_mv'],
                        'coupling': bs['coupling'],
                    },
                }
            }

        try:
            key = self._capture_store.save(
                frame, crop, cx_in_crop, cy_in_crop, meta, datasets)
            if self._current_loading_id:
                self._capture_store.increment_loading_count(
                    self._current_loading_id)
            self._load_captures()
            self._status_label.setText(
                f'Captured: {key}  ({self._last_cls})')
            self._select_key(key)
        except Exception as exc:
            QMessageBox.critical(self, 'Capture failed', str(exc))

    # ── Capture list & viewer ─────────────────────────────────────────────────

    def _refresh_store(self) -> None:
        path = self._capture_path_edit.text().strip()
        self._capture_store = CaptureStore(path)
        self._load_captures()

    def _browse_capture_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, 'Capture file', self._capture_path_edit.text(),
            'HDF5 files (*.hdf5 *.h5)')
        if path:
            self._capture_path_edit.setText(path)
            self._refresh_store()

    @staticmethod
    def _capture_line(cap: dict) -> str:
        """One list entry's text for a single capture."""
        key      = cap['key']
        meta     = cap['meta']
        auto_cls = str(meta.get('auto_classification', '?'))
        user_cls = str(meta.get('user_classification', ''))
        sigma    = meta.get('sigma', float('nan'))
        try:
            sigma_str = f'σ={float(sigma):.2f}px'
        except (TypeError, ValueError):
            sigma_str = 'σ=?'
        line = f'   {key}\n   {auto_cls}  {sigma_str}'
        if 'backscatter_rms_mv' in meta:
            try:
                line += f'  bs={float(meta["backscatter_rms_mv"]):.2g}mV'
            except (TypeError, ValueError):
                pass
        if user_cls:
            line += f'  → {user_cls}'
        notes = str(meta.get('notes', '')).strip()
        if notes:
            line += f'\n   “{notes}”'
        return line

    @staticmethod
    def _loading_header_text(lid: str, attrs: dict, n: int) -> str:
        """Header line summarising one loading session and its capture count."""
        cnt = f'{n} capture{"s" if n != 1 else ""}'
        if not lid:
            return f'── Unassigned · {cnt} ──'
        ts    = str(attrs.get('start_timestamp', lid))
        parts = [f'▼ Loading {ts}']
        try:
            parts.append(f'{float(attrs.get("pressure_mbar")):.1e} mbar')
        except (TypeError, ValueError):
            pass
        conc = str(attrs.get('solution_concentration', '')).strip()
        if conc:
            parts.append(f'conc={conc}')
        parts.append(cnt)
        return '  ·  '.join(parts)

    def _load_captures(self) -> None:
        if self._capture_store is None:
            return
        self._captures = self._capture_store.load_index()
        loadings = self._capture_store.load_loadings()

        # Group capture indices by their loading session.
        groups: dict = {}
        for i, cap in enumerate(self._captures):
            lid = str(cap['meta'].get('loading_id', '')).strip()
            groups.setdefault(lid, []).append(i)
        # Include loadings that have no captures yet, so empty sessions show.
        for lid in loadings:
            groups.setdefault(lid, [])

        # Sessions in chronological order; unassigned captures (no loading) last.
        ordered_ids = sorted(k for k in groups if k)
        if '' in groups:
            ordered_ids.append('')

        self._capture_list.blockSignals(True)
        self._capture_list.clear()
        self._row_map = []
        for lid in ordered_ids:
            header = QListWidgetItem(
                self._loading_header_text(lid, loadings.get(lid, {}),
                                          len(groups[lid])))
            header.setForeground(QColor('#00e5ff'))
            if lid:
                # Real loading session: selectable so its conditions are editable.
                self._row_map.append(lid)
            else:
                # Unassigned bucket: not a session, nothing to edit.
                header.setFlags(header.flags() & ~Qt.ItemIsSelectable)
                self._row_map.append(None)
            self._capture_list.addItem(header)
            for ci in groups[lid]:
                self._capture_list.addItem(self._capture_line(self._captures[ci]))
                self._row_map.append(ci)
        self._capture_list.blockSignals(False)

        n = len(self._captures)
        nl = len([k for k in groups if k])
        self._capture_count_lbl.setText(
            f'{n} capture{"s" if n != 1 else ""} · '
            f'{nl} loading{"s" if nl != 1 else ""}')

    def _select_key(self, key: str, block: bool = False) -> None:
        """Select the list row for a capture key, accounting for header rows."""
        for ci, cap in enumerate(self._captures):
            if cap['key'] == key:
                try:
                    row = self._row_map.index(ci)
                except ValueError:
                    return
                if block:
                    self._capture_list.blockSignals(True)
                self._capture_list.setCurrentRow(row)
                if block:
                    self._capture_list.blockSignals(False)
                return

    def _set_capture_controls_enabled(self, on: bool) -> None:
        """Enable/disable the per-capture editing controls together."""
        for btn in self._cls_buttons.values():
            btn.setEnabled(on)
            if not on:
                btn.setChecked(False)
        for w in (self._notes_edit, self._notes_save_btn,
                  self._delete_btn, self._export_sel_btn):
            w.setEnabled(on)

    def _set_conditions_enabled(self, on: bool) -> None:
        for w in (self._cond_pressure_edit, self._cond_conc_edit,
                  self._cond_save_btn):
            w.setEnabled(on)

    @staticmethod
    def _fmt_pressure(v) -> str:
        try:
            return f'{float(v):.3e}'
        except (TypeError, ValueError):
            return '' if v is None else str(v)

    def _clear_detail(self) -> None:
        self._capture_canvas.clear()
        self._meta_label.setText('—')
        self._selected_key  = None
        self._selected_kind = None
        self._selected_loading_id = None
        self._notes_edit.setText('')
        self._cond_target_lbl.setText('No selection')
        self._cond_pressure_edit.clear()
        self._cond_conc_edit.clear()
        self._set_capture_controls_enabled(False)
        self._set_conditions_enabled(False)
        self._cond_apply_loading.setEnabled(False)
        self._cond_apply_loading.setChecked(False)

    def _on_capture_selected(self, row: int) -> None:
        entry = self._row_map[row] if 0 <= row < len(self._row_map) else None
        if entry is None:
            self._clear_detail()
        elif isinstance(entry, str):
            self._show_loading_detail(entry)
        else:
            self._show_capture_detail(entry)

    def _show_capture_detail(self, ci: int) -> None:
        cap  = self._captures[ci]
        meta = cap['meta']
        self._selected_kind = 'capture'
        self._selected_key  = cap['key']
        self._selected_loading_id = str(meta.get('loading_id', '')).strip() or None

        # Pull the heavy frame/crop arrays only now, for the one selected capture.
        full = self._capture_store.load_one(cap['key'])
        if full is not None:
            self._capture_canvas.show_capture(
                full['crop'], full['cx'], full['cy'], meta)
        else:
            self._capture_canvas.clear()
        self._set_capture_controls_enabled(True)

        # Classification buttons
        user_cls = str(meta.get('user_classification', ''))
        for cls, btn in self._cls_buttons.items():
            btn.setChecked(cls == user_cls)

        # Editable notes field
        self._notes_edit.blockSignals(True)
        self._notes_edit.setText(str(meta.get('notes', '')))
        self._notes_edit.blockSignals(False)

        # Editable conditions
        self._cond_target_lbl.setText(f'Editing capture {cap["key"]}')
        self._cond_pressure_edit.setText(self._fmt_pressure(meta.get('pressure_mbar')))
        self._cond_conc_edit.setText(str(meta.get('solution_concentration', '')))
        self._set_conditions_enabled(True)
        self._cond_apply_loading.setEnabled(self._selected_loading_id is not None)
        self._cond_apply_loading.setChecked(False)

        self._show_capture_meta(cap)

    def _show_loading_detail(self, lid: str) -> None:
        self._selected_kind = 'loading'
        self._selected_key  = None
        self._selected_loading_id = lid

        loadings = (self._capture_store.load_loadings()
                    if self._capture_store else {})
        attrs = loadings.get(lid, {})

        self._capture_canvas.clear()
        self._set_capture_controls_enabled(False)

        self._cond_target_lbl.setText(
            f'Editing loading {attrs.get("start_timestamp", lid)} '
            f'(applies to every capture in it)')
        self._cond_pressure_edit.setText(self._fmt_pressure(attrs.get('pressure_mbar')))
        self._cond_conc_edit.setText(str(attrs.get('solution_concentration', '')))
        self._set_conditions_enabled(True)
        self._cond_apply_loading.setChecked(True)   # a session edit is always all
        self._cond_apply_loading.setEnabled(False)

        self._show_loading_meta(lid, attrs)

    def _show_capture_meta(self, cap: dict) -> None:
        meta = cap['meta']
        _ORDER = ['timestamp', 'loading_id', 'solution_concentration',
                  'trap_duration_s', 'auto_classification',
                  'user_classification', 'notes',
                  'pressure_mbar', 'pressure_source',
                  'backscatter_rms_mv', 'backscatter_pk2pk_mv',
                  'backscatter_mean_mv', 'backscatter_channel',
                  'backscatter_range_mv', 'backscatter_coupling',
                  'exposure_us', 'gain',
                  'sigma', 'sigma_x', 'sigma_y', 'fit_quality', 'peak',
                  'psf_sigma_ref', 'camera_width', 'camera_height']
        lines = [f'Key: {cap["key"]}']
        shown: set = set()
        for k in _ORDER:
            if k in meta:
                v = meta[k]
                try:
                    fv = float(v)
                    v  = f'{fv:.4g}'
                except (TypeError, ValueError):
                    v = str(v)
                lines.append(f'{k}: {v}')
                shown.add(k)
        for k, v in meta.items():
            if k not in shown:
                lines.append(f'{k}: {v}')
        self._meta_label.setText('\n'.join(lines))

    def _show_loading_meta(self, lid: str, attrs: dict) -> None:
        caps = [c for c in self._captures
                if str(c['meta'].get('loading_id', '')).strip() == lid]
        classes = [self._library_class(c['meta']) for c in caps]
        lines = [f'Loading: {lid}']
        for k in ('start_timestamp', 'pressure_mbar', 'pressure_source',
                  'solution_concentration', 'start_delay_s'):
            if k in attrs:
                v = attrs[k]
                try:
                    v = f'{float(v):.4g}'
                except (TypeError, ValueError):
                    v = str(v)
                lines.append(f'{k}: {v}')
        lines.append(f'captures: {len(caps)}')
        lines.append(
            f'  single={classes.count("single")} '
            f'cluster={classes.count("cluster")} '
            f'uncertain={classes.count("uncertain")} '
            f'reject={classes.count("reject")}')
        self._meta_label.setText('\n'.join(lines))

    def _select_loading(self, lid: str) -> None:
        """Select the list row for a loading header."""
        try:
            row = self._row_map.index(lid)
        except ValueError:
            return
        self._capture_list.setCurrentRow(row)

    def _on_save_conditions(self) -> None:
        """Apply edited pressure / concentration to the current selection."""
        if self._capture_store is None or self._selected_kind is None:
            return
        store = self._capture_store

        p_text = self._cond_pressure_edit.text().strip()
        if p_text:
            try:
                pressure = float(p_text)
            except ValueError:
                QMessageBox.warning(
                    self, 'Invalid pressure',
                    f'“{p_text}” is not a number — use e.g. 1e-6.\n'
                    'Leave the field blank to keep the existing value.')
                return
        else:
            pressure = None   # blank → leave pressure unchanged
        conc = self._cond_conc_edit.text().strip()

        def _apply_to_loading(lid: str) -> None:
            if pressure is not None:
                store.set_loading_attr(lid, 'pressure_mbar', pressure)
                store.set_capture_attr_for_loading(lid, 'pressure_mbar', pressure)
                store.set_capture_attr_for_loading(
                    lid, 'pressure_source', 'manual (edited)')
            store.set_loading_attr(lid, 'solution_concentration', conc)
            store.set_capture_attr_for_loading(
                lid, 'solution_concentration', conc)

        if self._selected_kind == 'loading':
            lid = self._selected_loading_id
            _apply_to_loading(lid)
            self._load_captures()
            self._select_loading(lid)
            self._status_label.setText(f'Updated loading {lid}')
            return

        # Single capture
        key = self._selected_key
        lid = self._selected_loading_id
        apply_all = self._cond_apply_loading.isChecked() and lid is not None
        if pressure is not None:
            store.set_attr(key, 'pressure_mbar', pressure)
            store.set_attr(key, 'pressure_source', 'manual (edited)')
        store.set_attr(key, 'solution_concentration', conc)
        if apply_all:
            _apply_to_loading(lid)
        self._load_captures()
        self._select_key(key)
        scope = ' (whole loading)' if apply_all else ''
        self._status_label.setText(f'Updated conditions for {key}{scope}')

    def _on_classify(self, cls: str) -> None:
        if self._selected_key is None or self._capture_store is None:
            return
        self._capture_store.set_user_classification(self._selected_key, cls)
        # Update button states
        for c, btn in self._cls_buttons.items():
            btn.setChecked(c == cls)
        # Refresh list item text, then reselect the same key
        self._load_captures()
        self._select_key(self._selected_key, block=True)
        self._status_label.setText(
            f'{self._selected_key} → {_CLS_LABELS[cls]}')

    def _reselect(self, key: str) -> None:
        """Reselect a capture by key after the list was rebuilt."""
        self._select_key(key)

    def _on_save_notes(self) -> None:
        if self._selected_key is None or self._capture_store is None:
            return
        text = self._notes_edit.text()
        self._capture_store.set_attr(self._selected_key, 'notes', text)
        key = self._selected_key
        self._load_captures()
        self._reselect(key)
        self._status_label.setText(f'Notes saved for {key}')

    # ── Library: open / delete / export ─────────────────────────────────────────

    def _on_open_library(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open capture library', self._capture_path_edit.text(),
            'HDF5 files (*.hdf5 *.h5)')
        if path:
            self._capture_path_edit.setText(path)
            self._refresh_store()
            self._status_label.setText(f'Opened {os.path.basename(path)}')

    def _on_delete_capture(self) -> None:
        if self._selected_key is None or self._capture_store is None:
            return
        key = self._selected_key
        if QMessageBox.question(
                self, 'Delete capture',
                f'Delete capture {key} from the library?\n'
                'This cannot be undone.') != QMessageBox.Yes:
            return
        self._capture_store.delete(key)
        self._load_captures()
        self._capture_list.setCurrentRow(-1)
        self._on_capture_selected(-1)
        self._status_label.setText(f'Deleted {key}')

    def _on_export_selected(self) -> None:
        if self._selected_key is None:
            return
        cap = next((c for c in self._captures
                    if c['key'] == self._selected_key), None)
        if cap is None:
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, 'Export selected capture to folder')
        if not out_dir:
            return
        try:
            base = self._export_one(cap, out_dir)
            self._status_label.setText(f'Exported {base}')
        except Exception as exc:
            QMessageBox.critical(self, 'Export failed', str(exc))

    def _on_export_all(self) -> None:
        if not self._captures:
            QMessageBox.information(self, 'Export all', 'No captures to export.')
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, 'Export entire library to folder')
        if not out_dir:
            return
        rows = []
        try:
            for cap in self._captures:
                base = self._export_one(cap, out_dir)
                m = cap['meta']
                rows.append({
                    'base':       base,
                    'key':        cap['key'],
                    'loading_id': m.get('loading_id', ''),
                    'timestamp':  m.get('timestamp', ''),
                    'class':      self._library_class(m),
                    'pressure_mbar':          m.get('pressure_mbar', ''),
                    'pressure_source':        m.get('pressure_source', ''),
                    'solution_concentration': m.get('solution_concentration', ''),
                    'backscatter_rms_mv':     m.get('backscatter_rms_mv', ''),
                    'sigma':      m.get('sigma', ''),
                    'notes':      m.get('notes', ''),
                })
            index_path = os.path.join(out_dir, 'library_index.csv')
            with open(index_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            n_load = self._export_loadings_index(out_dir)
            self._status_label.setText(
                f'Exported {len(rows)} captures + index + '
                f'{n_load} loadings to {out_dir}')
        except Exception as exc:
            QMessageBox.critical(self, 'Export all failed', str(exc))

    def _export_loadings_index(self, out_dir: str) -> int:
        """
        Write loadings_index.csv: one row per loading session with its
        conditions and per-class capture counts — the table for computing
        capture rate per loading.
        """
        loadings = (self._capture_store.load_loadings()
                    if self._capture_store else {})

        by_loading: dict = {}
        for cap in self._captures:
            lid = str(cap['meta'].get('loading_id', '')).strip()
            by_loading.setdefault(lid, []).append(cap)

        all_ids = sorted(k for k in set(loadings) | set(by_loading) if k)
        rows = []
        for lid in all_ids:
            attrs   = loadings.get(lid, {})
            caps    = by_loading.get(lid, [])
            classes = [self._library_class(c['meta']) for c in caps]
            rows.append({
                'loading_id':             lid,
                'start_timestamp':        attrs.get('start_timestamp', ''),
                'pressure_mbar':          attrs.get('pressure_mbar', ''),
                'pressure_source':        attrs.get('pressure_source', ''),
                'solution_concentration': attrs.get('solution_concentration', ''),
                'start_delay_s':          attrs.get('start_delay_s', ''),
                'n_captures':             len(caps),
                'n_single':               classes.count('single'),
                'n_cluster':              classes.count('cluster'),
                'n_uncertain':            classes.count('uncertain'),
                'n_reject':               classes.count('reject'),
            })
        if not rows:
            return 0
        path = os.path.join(out_dir, 'loadings_index.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return len(rows)

    @staticmethod
    def _library_class(meta: dict) -> str:
        """User classification if set, else the auto classification."""
        return (str(meta.get('user_classification', '')).strip()
                or str(meta.get('auto_classification', '')).strip()
                or 'unclassified')

    def _export_one(self, cap: dict, out_dir: str) -> str:
        """
        Write one capture to standalone, clearly-labelled files:
            <base>.json            all stats / trapping conditions
            <base>_crop.png        cropped object (hot colourmap)
            <base>_frame.tif/.png  full frame (16-bit TIFF if tifffile present)
            <base>_backscatter.npy backscatter waveform in mV (if recorded)
        The base name encodes class, pressure and timestamp so the folder is
        browsable as a labelled library.
        """
        import matplotlib.image as mpimg

        key   = cap['key']
        # Index entries are metadata-only; pull the image arrays for this key.
        full  = (cap if 'frame' in cap
                 else self._capture_store.load_one(key))
        if full is None:
            raise RuntimeError(f'capture {key} not found in library')
        meta  = dict(cap['meta'])
        cls   = self._library_class(meta)

        def _sanitize(s: str) -> str:
            return ''.join(ch if ch.isalnum() else '_'
                           for ch in str(s)).strip('_')

        try:
            p_tag = f'{float(meta.get("pressure_mbar")):.1e}mbar'
        except (TypeError, ValueError):
            p_tag = 'pNA'
        notes_tag = _sanitize(meta.get('notes', ''))[:32]
        base = f'{key}__{_sanitize(cls)}__{_sanitize(p_tag)}'
        if notes_tag:
            base += f'__{notes_tag}'

        def _norm8(arr: np.ndarray) -> np.ndarray:
            a = arr.astype(np.float64)
            lo = float(np.percentile(a, 1))
            hi = float(np.percentile(a, 99.5))
            if hi <= lo:
                hi = lo + 1.0
            return np.clip((a - lo) / (hi - lo), 0.0, 1.0)

        # Crop visualization
        mpimg.imsave(os.path.join(out_dir, f'{base}_crop.png'),
                     _norm8(full['crop']), cmap='hot', vmin=0.0, vmax=1.0)

        # Full frame — prefer lossless 16-bit TIFF
        frame = full['frame']
        try:
            import tifffile
            tifffile.imwrite(os.path.join(out_dir, f'{base}_frame.tif'),
                             frame.astype(np.uint16))
        except Exception:
            mpimg.imsave(os.path.join(out_dir, f'{base}_frame.png'),
                         _norm8(frame), cmap='gray', vmin=0.0, vmax=1.0)

        # Backscatter waveform (mV), if present
        if self._capture_store is not None:
            wf, attrs = self._capture_store.load_aux(key, 'backscatter')
            if wf is not None:
                adc2mv = float(attrs.get('adc2mv', 1.0))
                np.save(os.path.join(out_dir, f'{base}_backscatter.npy'),
                        wf.astype(np.float64) * adc2mv)
                meta['backscatter_npy'] = f'{base}_backscatter.npy'

        # Stats / conditions sidecar
        record = {'key': key, 'class': cls}
        for k, v in meta.items():
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            record[k] = v
        with open(os.path.join(out_dir, f'{base}.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(record, f, indent=2, default=str)
        return base

    # ── Frame → QImage ────────────────────────────────────────────────────────

    def _frame_to_qimage(self, frame: np.ndarray) -> QImage:
        if self._contrast_auto.isChecked():
            sub = frame[::4, ::4]
            lo  = int(np.percentile(sub, 1))
            hi  = int(np.percentile(sub, 99))
        else:
            lo = self._cmin_slider.value()
            hi = self._cmax_slider.value()
        hi = max(hi, lo + 1)

        if (lo, hi) != self._last_contrast:
            self._last_contrast = (lo, hi)
            span = hi - lo
            idx  = np.arange(65536, dtype=np.int32)
            _CONTRAST_LUT[:] = np.clip(
                (idx - lo) * 255 // span, 0, 255).astype(np.uint8)

        h, w = frame.shape
        if self._disp_buf is None or self._disp_buf.shape != frame.shape:
            self._disp_buf = np.empty((h, w), dtype=np.uint8)
        np.take(_CONTRAST_LUT, frame, out=self._disp_buf)

        if self._false_color_check.isChecked():
            if (self._false_color_buf is None
                    or self._false_color_buf.shape[:2] != (h, w)):
                self._false_color_buf = np.empty((h, w, 3), dtype=np.uint8)
            self._false_color_buf[:] = _HOT_LUT[self._disp_buf]
            return QImage(self._false_color_buf.data, w, h,
                          w * 3, QImage.Format_RGB888)
        else:
            return QImage(self._disp_buf.data, w, h, w, QImage.Format_Grayscale8)

    def closeEvent(self, event) -> None:
        if self._pressure_thread is not None:
            self._pressure_thread.request_stop()
            self._pressure_thread.wait(2000)
            self._pressure_thread = None
        self._on_bs_disconnect()
        self._on_disconnect()
        super().closeEvent(event)


# ── Entry point ───────────────────────────────────────────────────────────────

def launch_gui(use_mock: bool = False) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    win = TrapMonitorWindow(use_mock=use_mock)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    launch_gui(use_mock='--mock' in sys.argv)
