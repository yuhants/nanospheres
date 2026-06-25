"""
trap_monitor_gui.py — Automated trap-detection and capture monitor.

Continuously acquires camera frames, runs blob analysis, and saves a
capture whenever a particle has been stably trapped for a configurable
duration (default 10 s).  Each capture is stored in an HDF5 file and
can be reviewed and manually classified (single particle / cluster /
uncertain / reject).

Classification uses the same 2-D Gaussian fitting as camera_gui:
fitted σ is compared against a PSF σ reference to auto-classify each
capture.  The user can override the auto-classification per capture.

Run directly:
    python camera/trap_monitor_gui.py
    python camera/trap_monitor_gui.py --mock
"""

import sys
import os
import time
import queue
import math
import datetime
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
    QSplitter, QSlider, QSizePolicy, QShortcut, QFrame, QListWidget,
    QProgressBar, QScrollArea, QAbstractItemView,
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
             cx_in_crop: float, cy_in_crop: float, meta: dict) -> str:
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
        return key

    def load_all(self) -> list:
        """Returns list of capture dicts sorted by key (oldest first)."""
        import h5py
        if not os.path.exists(self.path):
            return []
        results = []
        with h5py.File(self.path, 'r') as f:
            g = f.get('captures')
            if g is None:
                return []
            for key in sorted(g.keys()):
                grp   = g[key]
                frame = grp['frame'][:]
                crop  = grp['crop'][:]
                cx    = float(grp.attrs.get('cx_in_crop', crop.shape[1] / 2))
                cy    = float(grp.attrs.get('cy_in_crop', crop.shape[0] / 2))
                meta  = {k: v for k, v in grp.attrs.items()
                         if k not in ('cx_in_crop', 'cy_in_crop')}
                results.append({
                    'key': key, 'frame': frame, 'crop': crop,
                    'cx': cx, 'cy': cy, 'meta': meta,
                })
        return results

    def set_user_classification(self, key: str, cls: str) -> None:
        import h5py
        with h5py.File(self.path, 'a') as f:
            g = f.get('captures')
            if g is not None and key in g:
                g[key].attrs['user_classification'] = cls

    def count(self) -> int:
        import h5py
        if not os.path.exists(self.path):
            return 0
        with h5py.File(self.path, 'r') as f:
            g = f.get('captures')
            return len(g) if g else 0


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
        self._monitoring:    bool = False

        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._poll_frame)

        self._build_ui()
        self._setup_shortcuts()
        self._refresh_store()
        QTimer.singleShot(0, self._on_scan)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # Left: camera + settings
        left = QWidget()
        left.setMaximumWidth(290)
        lv = QVBoxLayout(left)
        lv.addWidget(self._build_camera_group())
        lv.addWidget(self._build_settings_group())
        lv.addWidget(self._build_analysis_group())
        lv.addWidget(self._build_trap_group())
        lv.addStretch()

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

        splitter.addWidget(left)
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
        self._monitoring = False
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
        self._monitoring = checked
        if checked:
            self._monitor_btn.setText('Stop Monitoring')
            # Fresh tracker so the clock starts from now
            self._trap_tracker = TrapTracker(
                self._min_dur_spin.value(), self._gap_tol_spin.value())
            self._status_label.setText('Monitoring started')
        else:
            self._monitor_btn.setText('Start Monitoring')
            self._trap_progress.setValue(0)
            self._trap_label.setText('Monitoring stopped')
            self._status_label.setText('Monitoring stopped')

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

        meta = {
            'timestamp':          datetime.datetime.now().isoformat(timespec='seconds'),
            'trap_duration_s':    self._trap_tracker.duration,
            'auto_classification': self._last_cls,
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

        try:
            key = self._capture_store.save(
                frame, crop, cx_in_crop, cy_in_crop, meta)
            self._load_captures()
            self._status_label.setText(
                f'Captured: {key}  ({self._last_cls})')
            # Select the new capture in the list
            for i, cap in enumerate(self._captures):
                if cap['key'] == key:
                    self._capture_list.setCurrentRow(i)
                    break
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

    def _load_captures(self) -> None:
        if self._capture_store is None:
            return
        self._captures = self._capture_store.load_all()
        self._capture_list.blockSignals(True)
        self._capture_list.clear()
        for cap in self._captures:
            key      = cap['key']
            meta     = cap['meta']
            auto_cls = str(meta.get('auto_classification', '?'))
            user_cls = str(meta.get('user_classification', ''))
            sigma    = meta.get('sigma', float('nan'))
            try:
                sigma_str = f'σ={float(sigma):.2f}px'
            except (TypeError, ValueError):
                sigma_str = 'σ=?'
            line  = f'{key}\n{auto_cls}  {sigma_str}'
            if user_cls:
                line += f'  → {user_cls}'
            self._capture_list.addItem(line)
        self._capture_list.blockSignals(False)
        n = len(self._captures)
        self._capture_count_lbl.setText(f'{n} capture{"s" if n != 1 else ""}')

    def _on_capture_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._captures):
            self._capture_canvas.clear()
            self._meta_label.setText('—')
            self._selected_key = None
            for btn in self._cls_buttons.values():
                btn.setEnabled(False)
                btn.setChecked(False)
            return

        cap  = self._captures[row]
        self._selected_key = cap['key']
        self._capture_canvas.show_capture(
            cap['crop'], cap['cx'], cap['cy'], cap['meta'])

        # Update classification buttons
        user_cls = str(cap['meta'].get('user_classification', ''))
        for cls, btn in self._cls_buttons.items():
            btn.setEnabled(True)
            btn.setChecked(cls == user_cls)

        # Show metadata
        meta = cap['meta']
        _ORDER = ['timestamp', 'trap_duration_s', 'auto_classification',
                  'user_classification', 'exposure_us', 'gain',
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

    def _on_classify(self, cls: str) -> None:
        if self._selected_key is None or self._capture_store is None:
            return
        self._capture_store.set_user_classification(self._selected_key, cls)
        # Update button states
        for c, btn in self._cls_buttons.items():
            btn.setChecked(c == cls)
        # Refresh list item text
        self._load_captures()
        # Reselect the same key
        for i, cap in enumerate(self._captures):
            if cap['key'] == self._selected_key:
                self._capture_list.blockSignals(True)
                self._capture_list.setCurrentRow(i)
                self._capture_list.blockSignals(False)
                break
        self._status_label.setText(
            f'{self._selected_key} → {_CLS_LABELS[cls]}')

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
