"""
Thorlabs CS165MU1 Camera GUI — PyQt5 interface for live view, capture, and
particle analysis.

Layout:
  Left panel   — connect/disconnect, camera settings (exposure, gain, etc.)
  Centre       — live view (QLabel/QPixmap, fast rendering)
  Right panel  — analysis controls + result + radial-profile plot

The acquisition loop runs in a background QThread; a 30 Hz QTimer in the
main thread dequeues the latest frame and updates the display.  Analysis
(blob detection) runs on every displayed frame but can be disabled.

Keyboard shortcuts:
  Space  — snapshot
  R      — toggle recording
  B      — capture background frame

Run directly:
    python camera/camera_gui.py
or:
    python camera/camera_gui.py --mock      (simulated camera, no hardware)
"""

import sys
import os
import time
import queue
import threading
import math
import datetime
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
    QSplitter, QSlider, QSizePolicy, QShortcut, QFrame,
    QDialog, QListWidget, QScrollArea,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QFont, QKeySequence,
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.thorlabs_camera import ThorlabsCamera, MockCamera, analyze_frame

_DISPLAY_FPS    = 30        # Hz — QTimer rate
_MAX_REC_FRAMES = 3000      # hard cap on buffered recording frames (~100 s at 30 fps)
_PLOT_MIN_INTERVAL = 2.0    # seconds between matplotlib redraws

# Pre-computed lookup tables (built once at import, reused every frame)
# _CONTRAST_LUT: uint16 → uint8 for the current lo/hi; rebuilt only when they change
_CONTRAST_LUT = np.zeros(65536, dtype=np.uint8)
# _HOT_LUT: uint8 → RGB (hot colormap: black→red→yellow→white)
_HOT_LUT = np.zeros((256, 3), dtype=np.uint8)
_i = np.arange(256)
_HOT_LUT[:, 0] = np.clip(_i * 3,       0, 255)
_HOT_LUT[:, 1] = np.clip(_i * 3 - 255, 0, 255)
_HOT_LUT[:, 2] = np.clip(_i * 3 - 510, 0, 255)

_RESULT_COLORS  = {
    'single':    '#00cc44',
    'cluster':   '#ff3333',
    'uncertain': '#ffaa00',
    'none':      '#888888',
}


# ── Acquisition thread ────────────────────────────────────────────────────────

class CameraThread(QThread):
    """
    Continuously grabs frames from the camera and puts them in a queue.
    The main thread's QTimer dequeues them at display rate.
    """
    error = pyqtSignal(str)

    def __init__(self, camera, frame_queue: queue.Queue):
        super().__init__()
        self._camera = camera
        self._queue  = frame_queue
        self._stop   = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            while not self._stop:
                frame = self._camera.get_frame()
                if frame is not None:
                    # Keep only the latest; discard stale frames to avoid lag
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._queue.put(frame)
                else:
                    self.msleep(2)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Analysis thread ───────────────────────────────────────────────────────────

class AnalysisThread(QThread):
    """
    Runs blob detection in a background thread so it never blocks the display.

    The main thread calls submit() with the latest frame; if a previous frame
    is still being analysed the new one replaces it (we only care about the
    most recent result, not every frame).
    """
    result_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._queue = queue.Queue(maxsize=1)
        self._stop  = False

    def submit(self, frame: np.ndarray, threshold: int, min_area: int,
               psf_sigma: float, sigma_tol: float,
               roi_offset: tuple = (0, 0)) -> None:
        try:
            self._queue.get_nowait()       # drop stale pending frame
        except queue.Empty:
            pass
        self._queue.put((frame, threshold, min_area, psf_sigma, sigma_tol, roi_offset))

    def request_stop(self) -> None:
        self._stop = True
        try:
            self._queue.put_nowait(None)   # unblock get()
        except queue.Full:
            pass

    def run(self) -> None:
        while not self._stop:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None or self._stop:
                break
            frame, threshold, min_area, psf_sigma, sigma_tol, roi_offset = item
            try:
                result = analyze_frame(frame, threshold, min_area, psf_sigma, sigma_tol)
                ox, oy = roi_offset
                if ox or oy:
                    for blob in result['blobs']:
                        blob['cx'] += ox
                        blob['cy'] += oy
                self.result_ready.emit(result)
            except Exception:
                pass


# ── Live view canvas ──────────────────────────────────────────────────────────

class LiveViewLabel(QLabel):
    """
    Camera live-view widget with zoom and pan.

    Mouse wheel  — zoom in/out, keeping the pixel under the cursor fixed.
    Left drag    — pan when zoomed in.
    Double-click — reset to fit view.
    """

    zoom_changed = pyqtSignal(float)   # emits zoom factor (1.0 = fit)
    roi_changed  = pyqtSignal(object)  # emits (x0,y0,x1,y1) tuple or None

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet('QLabel { background: #111; color: #555; }')
        self.setText('No camera connected')
        self.setFont(QFont('Arial', 12))

        self._blobs          = []
        self._show_overlay   = True
        self._classification = 'none'

        # Image dimensions (set on first frame)
        self._img_w: int   = 1
        self._img_h: int   = 1
        self._last_pm: QPixmap | None = None

        # Zoom/pan state
        self._zoom:   float = 1.0   # 1.0 = full image fits the widget
        self._pan_cx: float = 0.5   # centre of view in image pixels
        self._pan_cy: float = 0.5

        # Display transform (image px → widget px), updated by _render()
        self._crop_x0:  float = 0.0
        self._crop_y0:  float = 0.0
        self._scale_x:  float = 1.0
        self._scale_y:  float = 1.0
        self._offset_x: int   = 0
        self._offset_y: int   = 0
        self._disp_w:   int   = 1
        self._disp_h:   int   = 1

        self._drag_start:     tuple | None = None
        self._drag_pan_start: tuple | None = None

        # ROI selection
        self._roi:            tuple | None = None  # (x0,y0,x1,y1) in image coords
        self._roi_mode:       bool         = False
        self._roi_drag_start: tuple | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update_frame(self, qimg: QImage, blobs: list,
                     classification: str, show_overlay: bool,
                     img_w: int, img_h: int) -> None:
        self._blobs          = blobs
        self._show_overlay   = show_overlay
        self._classification = classification

        first = self._last_pm is None
        self._img_w   = img_w
        self._img_h   = img_h
        self._last_pm = QPixmap.fromImage(qimg)

        if first:
            self._pan_cx = img_w / 2.0
            self._pan_cy = img_h / 2.0

        self._render()

    def reset_zoom(self) -> None:
        self._zoom   = 1.0
        self._pan_cx = self._img_w / 2.0
        self._pan_cy = self._img_h / 2.0
        self._render()
        self.zoom_changed.emit(self._zoom)

    # ── ROI ──────────────────────────────────────────────────────────────────

    @property
    def roi(self) -> tuple | None:
        return self._roi

    def set_roi_mode(self, active: bool) -> None:
        self._roi_mode     = active
        self._drag_start   = None   # cancel any pan in progress
        self.setCursor(Qt.CrossCursor if active else
                       (Qt.OpenHandCursor if self._zoom > 1.0 else Qt.ArrowCursor))

    def clear_roi(self) -> None:
        self._roi            = None
        self._roi_drag_start = None
        self.roi_changed.emit(None)
        self._render()

    # ── Zoom ─────────────────────────────────────────────────────────────────

    def zoom_by(self, factor: float) -> None:
        """Zoom in/out by factor, centred on the image centre."""
        new_zoom = max(1.0, min(20.0, self._zoom * factor))
        if new_zoom != self._zoom:
            self._zoom = new_zoom
            self._clamp_pan()
            self._render()
            self.zoom_changed.emit(self._zoom)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self) -> None:
        if self._last_pm is None:
            return

        iw, ih = self._img_w, self._img_h
        zoom    = self._zoom

        # Visible crop rectangle in image coords
        half_w = iw / (2.0 * zoom)
        half_h = ih / (2.0 * zoom)
        x0 = max(0, int(self._pan_cx - half_w))
        y0 = max(0, int(self._pan_cy - half_h))
        x1 = min(iw, int(self._pan_cx + half_w) + 1)
        y1 = min(ih, int(self._pan_cy + half_h) + 1)
        cw, ch = x1 - x0, y1 - y0

        # Preserve aspect ratio: compute display rect manually
        ww, wh  = self.width(), self.height()
        scale   = min(ww / cw, wh / ch) if cw > 0 and ch > 0 else 1.0
        dw, dh  = int(cw * scale), int(ch * scale)
        ox      = (ww - dw) // 2
        oy      = (wh - dh) // 2

        # Save transform for blob overlay and mouse hit-testing
        self._crop_x0  = float(x0)
        self._crop_y0  = float(y0)
        self._scale_x  = dw / cw if cw > 0 else 1.0
        self._scale_y  = dh / ch if ch > 0 else 1.0
        self._offset_x = ox
        self._offset_y = oy
        self._disp_w   = dw
        self._disp_h   = dh

        # Single QPainter call: crop + scale in one step, no intermediate QPixmap
        out = QPixmap(ww, wh)
        out.fill(QColor('#111'))
        painter = QPainter(out)
        painter.drawPixmap(QRect(ox, oy, dw, dh),
                           self._last_pm,
                           QRect(x0, y0, cw, ch))
        if self._show_overlay and self._blobs:
            self._draw_blobs(painter)
        if self._roi is not None:
            self._draw_roi(painter)
        painter.end()
        self.setPixmap(out)

    def _draw_roi(self, painter: QPainter) -> None:
        x0, y0, x1, y1 = self._roi
        sx0 = int((x0 - self._crop_x0) * self._scale_x) + self._offset_x
        sy0 = int((y0 - self._crop_y0) * self._scale_y) + self._offset_y
        sx1 = int((x1 - self._crop_x0) * self._scale_x) + self._offset_x
        sy1 = int((y1 - self._crop_y0) * self._scale_y) + self._offset_y
        pen = QPen(QColor('#00e5ff'), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(sx0, sy0, sx1 - sx0, sy1 - sy0)
        painter.setPen(QPen(QColor('#00e5ff'), 1))
        painter.setFont(QFont('Arial', 8))
        painter.drawText(sx0 + 3, sy0 - 4, f'{x1-x0}×{y1-y0} px')

    def _draw_blobs(self, painter: QPainter) -> None:
        color = QColor(_RESULT_COLORS.get(self._classification, '#ffffff'))
        pen   = QPen(color, 2)
        painter.setPen(pen)
        painter.setFont(QFont('Arial', 9))

        for i, b in enumerate(self._blobs):
            x = int((b['cx'] - self._crop_x0) * self._scale_x) + self._offset_x
            y = int((b['cy'] - self._crop_y0) * self._scale_y) + self._offset_y
            r = max(6, int(b['radius'] * max(self._scale_x, self._scale_y) * 2.5))
            painter.drawEllipse(x - r, y - r, 2 * r, 2 * r)
            painter.setPen(QPen(color, 1))
            painter.drawText(x + r + 3, y + 4, f'#{i+1}')
            painter.setPen(pen)

    # ── Zoom / pan helpers ────────────────────────────────────────────────────

    def _clamp_pan(self) -> None:
        half_w = self._img_w / (2.0 * self._zoom)
        half_h = self._img_h / (2.0 * self._zoom)
        self._pan_cx = max(half_w,               min(self._img_w - half_w, self._pan_cx))
        self._pan_cy = max(half_h,               min(self._img_h - half_h, self._pan_cy))

    def _widget_to_image(self, sx: int, sy: int):
        """Convert widget pixel coords to image pixel coords."""
        ix = self._crop_x0 + (sx - self._offset_x) / self._scale_x
        iy = self._crop_y0 + (sy - self._offset_y) / self._scale_y
        return (max(0.0, min(float(self._img_w), ix)),
                max(0.0, min(float(self._img_h), iy)))

    # ── Qt events ────────────────────────────────────────────────────────────

    def wheelEvent(self, event) -> None:
        if self._last_pm is None:
            return
        sx, sy  = event.x(), event.y()
        img_x, img_y = self._widget_to_image(sx, sy)

        factor   = 1.25 if event.angleDelta().y() > 0 else 1.0 / 1.25
        new_zoom = max(1.0, min(20.0, self._zoom * factor))
        if new_zoom == self._zoom:
            return

        # Adjust pan so the image pixel under the cursor stays fixed.
        # After zoom: img_x = new_pan_cx + img_w/new_zoom * (0.5 - (sx-ox)/dw)
        # Solving for new_pan_cx:
        dw = self._disp_w if self._disp_w > 0 else self.width()
        dh = self._disp_h if self._disp_h > 0 else self.height()
        self._pan_cx = img_x + self._img_w / new_zoom * (0.5 - (sx - self._offset_x) / dw)
        self._pan_cy = img_y + self._img_h / new_zoom * (0.5 - (sy - self._offset_y) / dh)
        self._zoom   = new_zoom
        self._clamp_pan()
        self._render()
        self.zoom_changed.emit(self._zoom)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        if self._roi_mode:
            ix, iy = self._widget_to_image(event.x(), event.y())
            self._roi_drag_start = (ix, iy)
            self._roi = (int(ix), int(iy), int(ix), int(iy))
        elif self._zoom > 1.0:
            self._drag_start     = (event.x(), event.y())
            self._drag_pan_start = (self._pan_cx, self._pan_cy)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._roi_mode and self._roi_drag_start is not None:
            ix, iy = self._widget_to_image(event.x(), event.y())
            x0 = int(min(self._roi_drag_start[0], ix))
            y0 = int(min(self._roi_drag_start[1], iy))
            x1 = int(max(self._roi_drag_start[0], ix))
            y1 = int(max(self._roi_drag_start[1], iy))
            self._roi = (x0, y0, x1, y1)
            self._render()
        elif self._drag_start is not None:
            dx = event.x() - self._drag_start[0]
            dy = event.y() - self._drag_start[1]
            self._pan_cx = self._drag_pan_start[0] - dx / self._scale_x
            self._pan_cy = self._drag_pan_start[1] - dy / self._scale_y
            self._clamp_pan()
            self._render()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        if self._roi_mode and self._roi_drag_start is not None:
            self._roi_drag_start = None
            # Minimum 5×5 px ROI — discard accidental clicks
            if self._roi and (self._roi[2]-self._roi[0] < 5
                              or self._roi[3]-self._roi[1] < 5):
                self._roi = None
            self.roi_changed.emit(self._roi)
            # Auto-exit ROI draw mode after one rectangle
            self._roi_mode = False
            self.setCursor(Qt.ArrowCursor if self._zoom <= 1.0 else Qt.OpenHandCursor)
        elif self._drag_start is not None:
            self._drag_start = None
            self.setCursor(Qt.ArrowCursor if self._zoom <= 1.0 else Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_zoom()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()


# ── Analysis plot ─────────────────────────────────────────────────────────────

class AnalysisCanvas(FigureCanvasQTAgg):
    """2-D brightness distribution + 1-D Δx / Δy projections."""

    def __init__(self):
        from matplotlib.gridspec import GridSpec
        self.fig = Figure(facecolor='#1a1a1a')
        super().__init__(self.fig)
        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        gs = GridSpec(2, 2, figure=self.fig,
                      width_ratios=[4, 1], height_ratios=[1, 4],
                      hspace=0.04, wspace=0.04,
                      left=0.10, right=0.97, top=0.93, bottom=0.08)

        self._ax_2d     = self.fig.add_subplot(gs[1, 0])
        self._ax_proj_x = self.fig.add_subplot(gs[0, 0], sharex=self._ax_2d)
        self._ax_proj_y = self.fig.add_subplot(gs[1, 1], sharey=self._ax_2d)

        for ax in (self._ax_2d, self._ax_proj_x, self._ax_proj_y):
            ax.set_facecolor('#1a1a1a')
            for sp in ax.spines.values():
                sp.set_color('#444')
            ax.tick_params(colors='#aaa', labelsize=7)

        self.draw()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return width

    def update_plots(self, image: np.ndarray, blobs: list, classification: str) -> None:
        from matplotlib.patches import Ellipse

        for ax in (self._ax_2d, self._ax_proj_x, self._ax_proj_y):
            ax.cla()
            ax.set_facecolor('#1a1a1a')
            for sp in ax.spines.values():
                sp.set_color('#444')

        if blobs and image is not None:
            b    = max(blobs, key=lambda b: b['peak'])
            H, W = image.shape
            cx, cy = b['cx'], b['cy']

            sigma_est = b.get('sigma', float('nan'))
            if math.isnan(sigma_est):
                sigma_est = b['radius']
            half = max(15, int(sigma_est * 5))
            x0 = max(0, int(cx) - half);  x1 = min(W, int(cx) + half + 1)
            y0 = max(0, int(cy) - half);  y1 = min(H, int(cy) + half + 1)
            crop = image[y0:y1, x0:x1]
            lcx  = cx - x0
            lcy  = cy - y0
            ch, cw = crop.shape

            # ── 2-D image ─────────────────────────────────────────────────
            self._ax_2d.imshow(crop, cmap='hot', origin='upper',
                               aspect='equal', interpolation='nearest')

            if b.get('fit_ok') and not math.isnan(b.get('sigma', float('nan'))):
                sx, sy = b['sigma_x'], b['sigma_y']
                for n, ls, alpha in [(1, '-', 0.9), (2, '--', 0.55)]:
                    self._ax_2d.add_patch(
                        Ellipse((lcx, lcy), width=2*n*sx, height=2*n*sy,
                                fill=False, edgecolor='cyan',
                                linestyle=ls, linewidth=1, alpha=alpha))

            self._ax_2d.axhline(lcy, color='white', lw=0.5, alpha=0.35)
            self._ax_2d.axvline(lcx, color='white', lw=0.5, alpha=0.35)

            self._ax_2d.set_xticks([0, lcx, cw - 1])
            self._ax_2d.set_xticklabels(
                [f'{-lcx:.0f}', '0', f'{cw-1-lcx:.0f}'], fontsize=7)
            self._ax_2d.set_yticks([0, lcy, ch - 1])
            self._ax_2d.set_yticklabels(
                [f'{-lcy:.0f}', '0', f'{ch-1-lcy:.0f}'], fontsize=7)
            self._ax_2d.set_xlabel('Δx (px)', fontsize=8, color='#ccc')
            self._ax_2d.set_ylabel('Δy (px)', fontsize=8, color='#ccc')

            # ── Δx projection (sum along y, plotted vs column) ────────────
            proj_x = crop.sum(axis=0).astype(float)
            xs = np.arange(cw)
            self._ax_proj_x.plot(xs, proj_x, color='#ff6b35', lw=1.2)
            self._ax_proj_x.fill_between(xs, proj_x, alpha=0.25, color='#ff6b35')
            self._ax_proj_x.axvline(lcx, color='white', lw=0.5, alpha=0.35)
            self._ax_proj_x.tick_params(labelbottom=False, labelsize=6, colors='#aaa')
            self._ax_proj_x.set_ylabel('ΣΔy', fontsize=7, color='#aaa')

            sigma_str = f'  σ={b["sigma"]:.2f}px' if b.get('fit_ok') else ''
            self._ax_proj_x.set_title(f'Blob{sigma_str}',
                                      fontsize=9, color='white', pad=3)

            # ── Δy projection (sum along x, plotted vs row) ───────────────
            proj_y = crop.sum(axis=1).astype(float)
            ys = np.arange(ch)
            self._ax_proj_y.plot(proj_y, ys, color='#4ecdc4', lw=1.2)
            self._ax_proj_y.fill_betweenx(ys, proj_y, alpha=0.25, color='#4ecdc4')
            self._ax_proj_y.axhline(lcy, color='white', lw=0.5, alpha=0.35)
            self._ax_proj_y.tick_params(labelleft=False, labelsize=6, colors='#aaa')
            self._ax_proj_y.set_xlabel('ΣΔx', fontsize=7, color='#aaa')
            self._ax_proj_y.invert_xaxis()   # keep large values pointing outward

        else:
            self._ax_proj_x.set_title('No blob', fontsize=9, color='white')
            for ax in (self._ax_2d, self._ax_proj_x, self._ax_proj_y):
                ax.axis('off')

        for ax in (self._ax_2d, self._ax_proj_x, self._ax_proj_y):
            ax.tick_params(colors='#aaa', labelsize=7)

        self.draw_idle()


# ── Template storage ──────────────────────────────────────────────────────────

class TemplateStore:
    """Saves / loads single-particle reference images in one HDF5 file."""

    def __init__(self, path: str):
        self.path = path

    def save(self, crop: np.ndarray, cx_in_crop: float, cy_in_crop: float,
             meta: dict) -> str:
        import h5py
        key = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        with h5py.File(self.path, 'a') as f:
            g = f.require_group('templates')
            ds = g.create_dataset(key, data=crop,
                                  compression='gzip', compression_opts=1)
            ds.attrs['cx_in_crop'] = cx_in_crop
            ds.attrs['cy_in_crop'] = cy_in_crop
            for k, v in meta.items():
                ds.attrs[k] = str(v) if isinstance(v, str) else v
        return key

    def load_all(self) -> list:
        """Returns [(key, crop, cx_in_crop, cy_in_crop, meta), ...] sorted by key."""
        import h5py
        if not os.path.exists(self.path):
            return []
        results = []
        with h5py.File(self.path, 'r') as f:
            g = f.get('templates')
            if g is None:
                return []
            for key in sorted(g.keys()):
                ds   = g[key]
                crop = ds[:]
                cx   = float(ds.attrs.get('cx_in_crop', crop.shape[1] / 2))
                cy   = float(ds.attrs.get('cy_in_crop', crop.shape[0] / 2))
                meta = {k: v for k, v in ds.attrs.items()
                        if k not in ('cx_in_crop', 'cy_in_crop')}
                results.append((key, crop, cx, cy, meta))
        return results

    def delete(self, key: str) -> None:
        import h5py
        with h5py.File(self.path, 'a') as f:
            g = f.get('templates')
            if g is not None and key in g:
                del g[key]

    def count(self) -> int:
        import h5py
        if not os.path.exists(self.path):
            return 0
        with h5py.File(self.path, 'r') as f:
            g = f.get('templates')
            return len(g) if g else 0


# ── Template gallery dialog ────────────────────────────────────────────────────

class TemplateGalleryDialog(QDialog):
    """Browse, inspect, and delete saved single-particle reference images."""

    def __init__(self, store: TemplateStore, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Single-particle templates')
        self.setMinimumSize(860, 560)
        self._store     = store
        self._templates = store.load_all()
        self._cur_idx   = -1

        # ── Left: scrollable list ────────────────────────────────────────
        self._list = QListWidget()
        self._list.setMinimumWidth(200)
        self._list.setMaximumWidth(240)
        self._list.currentRowChanged.connect(self._on_select)
        self._populate_list()

        # ── Centre: image ─────────────────────────────────────────────────
        self._fig = Figure(tight_layout=True)
        self._fig.patch.set_facecolor('#1a1a1a')
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setMinimumSize(320, 320)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._ax.set_facecolor('#1a1a1a')

        # ── Right: metadata ───────────────────────────────────────────────
        self._meta_label = QLabel('Select a template')
        self._meta_label.setWordWrap(True)
        self._meta_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._meta_label.setFont(QFont('Courier', 9))
        self._meta_label.setMinimumWidth(190)

        meta_scroll = QWidget()
        meta_vl = QVBoxLayout(meta_scroll)
        meta_vl.addWidget(self._meta_label)
        meta_vl.addStretch()

        # ── Buttons ───────────────────────────────────────────────────────
        self._del_btn = QPushButton('Delete selected')
        self._del_btn.setEnabled(False)
        self._del_btn.clicked.connect(self._on_delete)
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self._del_btn)
        btn_row.addWidget(close_btn)

        # ── Layout ────────────────────────────────────────────────────────
        content = QHBoxLayout()
        content.addWidget(self._list)
        content.addWidget(self._canvas, 1)
        content.addWidget(meta_scroll)

        root = QVBoxLayout(self)
        root.addLayout(content, 1)
        root.addLayout(btn_row)

        if self._templates:
            self._list.setCurrentRow(0)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _populate_list(self) -> None:
        self._list.clear()
        for key, crop, cx, cy, meta in self._templates:
            note  = str(meta.get('note', ''))
            sigma = meta.get('sigma', float('nan'))
            exp   = meta.get('exposure_us', '?')
            try:
                sigma_str = f'σ={float(sigma):.2f}px'
            except (TypeError, ValueError):
                sigma_str = 'σ=?'
            try:
                exp_str = f'{int(float(exp))}μs'
            except (TypeError, ValueError):
                exp_str = str(exp)
            text = f'{key}\n{sigma_str}  {exp_str}'
            if note:
                text += f'\n"{note}"'
            self._list.addItem(text)

    def _on_select(self, row: int) -> None:
        if row < 0 or row >= len(self._templates):
            return
        self._cur_idx = row
        self._del_btn.setEnabled(True)
        key, crop, cx, cy, meta = self._templates[row]
        self._draw(crop, cx, cy, meta)
        self._show_meta(key, meta)

    def _draw(self, crop: np.ndarray, cx: float, cy: float, meta: dict) -> None:
        from matplotlib.patches import Ellipse
        self._ax.cla()
        self._ax.set_facecolor('#1a1a1a')
        self._ax.imshow(crop, cmap='hot', origin='upper',
                        aspect='equal', interpolation='nearest')

        sx = float(meta.get('sigma_x', 0) or 0)
        sy = float(meta.get('sigma_y', 0) or 0)
        if sx > 0 and sy > 0 and meta.get('fit_ok', False):
            for n, ls, alpha in [(1, '-', 0.9), (2, '--', 0.55)]:
                self._ax.add_patch(
                    Ellipse((cx, cy), width=2*n*sx, height=2*n*sy,
                            fill=False, edgecolor='cyan',
                            linestyle=ls, linewidth=1, alpha=alpha))

        self._ax.axhline(cy, color='white', lw=0.5, alpha=0.35)
        self._ax.axvline(cx, color='white', lw=0.5, alpha=0.35)

        sigma = meta.get('sigma', float('nan'))
        try:
            title = f'σ={float(sigma):.2f} px'
        except (TypeError, ValueError):
            title = ''
        self._ax.set_title(title, fontsize=9, color='white')
        self._ax.tick_params(colors='#aaa', labelsize=7)
        for sp in self._ax.spines.values():
            sp.set_color('#444')
        self._canvas.draw_idle()

    def _show_meta(self, key: str, meta: dict) -> None:
        _ORDER = ['note', 'timestamp', 'exposure_us', 'gain',
                  'sigma', 'sigma_x', 'sigma_y', 'fit_quality',
                  'peak', 'cx', 'cy', 'psf_sigma_ref',
                  'camera_width', 'camera_height', 'bit_depth']
        lines = [f'Key: {key}', '']
        shown: set = set()
        for k in _ORDER:
            if k in meta:
                v = meta[k]
                try:
                    fv = float(v)
                    v  = f'{fv:.4g}'
                except (TypeError, ValueError):
                    v = str(v)
                lines.append(f'{k}:\n  {v}')
                shown.add(k)
        for k, v in meta.items():
            if k not in shown:
                lines.append(f'{k}:\n  {v}')
        self._meta_label.setText('\n'.join(lines))

    def _on_delete(self) -> None:
        if self._cur_idx < 0 or self._cur_idx >= len(self._templates):
            return
        key = self._templates[self._cur_idx][0]
        if QMessageBox.question(
                self, 'Delete template', f'Delete  {key}?',
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self._store.delete(key)
        self._templates.pop(self._cur_idx)
        self._populate_list()
        self._ax.cla()
        self._canvas.draw_idle()
        self._meta_label.setText('Select a template')
        self._del_btn.setEnabled(False)
        self._cur_idx = -1
        if self._templates:
            self._list.setCurrentRow(
                min(self._cur_idx, len(self._templates) - 1))


# ── Main window ───────────────────────────────────────────────────────────────

class CameraMainWindow(QMainWindow):

    def __init__(self, use_mock: bool = False):
        super().__init__()
        self.setWindowTitle('Thorlabs CS165MU1 — Nanoparticle Camera')
        self.setMinimumSize(1500, 800)

        self._use_mock  = use_mock
        self._camera    = None
        self._cam_thread:      CameraThread | None    = None
        self._analysis_thread: AnalysisThread | None  = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)

        self._last_frame:  np.ndarray | None = None
        self._background:  np.ndarray | None = None
        self._last_blobs:  list = []
        self._last_cls:    str  = 'none'
        self._last_result: dict = {}
        self._frame_count: int  = 0
        self._fps_counter: int  = 0
        self._fps_t0:      float = time.time()
        self._fps:         float = 0.0

        self._rotation_k: int = 0   # np.rot90 k: 0=none, 1=90CCW, 2=180, 3=90CW

        self._is_recording:  bool = False
        self._rec_frames:    list = []
        self._rec_start:     float = 0.0

        # LUT contrast cache: rebuilt only when lo/hi changes
        self._last_contrast: tuple = (-1, -1)
        # Reusable uint8 display buffers (avoids per-frame allocation)
        self._disp_buf:       np.ndarray | None = None
        self._false_color_buf: np.ndarray | None = None
        # Throttle matplotlib redraws
        self._plot_last_t: float = 0.0

        self._template_store: TemplateStore | None = None  # set after UI built

        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._poll_frame)

        self._build_ui()
        self._setup_shortcuts()
        self._refresh_template_store()
        QTimer.singleShot(0, self._on_scan)  # scan after event loop starts

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # ── Left: camera settings ─────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(280)
        lv = QVBoxLayout(left)
        lv.addWidget(self._build_connect_group())
        lv.addWidget(self._build_settings_group())
        lv.addWidget(self._build_display_group())
        lv.addWidget(self._build_capture_group())
        lv.addStretch()

        # ── Centre: live view ─────────────────────────────────────────────
        centre = QWidget()
        cv = QVBoxLayout(centre)
        cv.setContentsMargins(4, 4, 4, 4)
        self._live = LiveViewLabel()
        self._live.zoom_changed.connect(self._on_zoom_changed)
        self._live.roi_changed.connect(self._on_roi_changed)
        cv.addWidget(self._live)
        cv.addWidget(self._build_contrast_bar())

        # ── Right: analysis ───────────────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(360)
        rv = QVBoxLayout(right)
        rv.addWidget(self._build_analysis_group())
        rv.addWidget(self._build_result_group())
        rv.addWidget(self._build_templates_group())
        self._analysis_canvas = AnalysisCanvas()
        rv.addWidget(self._analysis_canvas)
        rv.addStretch()

        splitter.addWidget(left)
        splitter.addWidget(centre)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        container = QWidget()
        vl = QVBoxLayout(container)
        vl.addWidget(splitter)
        vl.addWidget(self._build_status_bar())
        self.setCentralWidget(container)

    def _build_connect_group(self) -> QGroupBox:
        g = QGroupBox('Camera')
        layout = QGridLayout(g)

        self._serial_combo = QComboBox()
        self._serial_combo.setEditable(True)
        self._serial_combo.setPlaceholderText('Serial or auto-detect')

        self._scan_btn       = QPushButton('Scan')
        self._connect_btn    = QPushButton('Connect')
        self._disconnect_btn = QPushButton('Disconnect')
        self._disconnect_btn.setEnabled(False)

        if self._use_mock:
            self._serial_combo.addItem('MOCK-001')
            self._serial_combo.setEnabled(False)

        self._scan_btn.clicked.connect(self._on_scan)
        self._connect_btn.clicked.connect(self._on_connect)
        self._disconnect_btn.clicked.connect(self._on_disconnect)

        layout.addWidget(QLabel('Serial:'),          0, 0)
        layout.addWidget(self._serial_combo,         0, 1, 1, 2)
        layout.addWidget(self._scan_btn,             1, 0)
        layout.addWidget(self._connect_btn,          1, 1)
        layout.addWidget(self._disconnect_btn,       1, 2)
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
        self._gain_spin.setSuffix(' ')
        self._gain_spin.setSingleStep(10.0)
        self._gain_spin.setEnabled(False)
        self._gain_spin.valueChanged.connect(self._on_gain_changed)

        self._black_spin = QSpinBox()
        self._black_spin.setRange(0, 255)
        self._black_spin.setValue(0)
        self._black_spin.setEnabled(False)
        self._black_spin.valueChanged.connect(self._on_black_changed)

        layout.addWidget(QLabel('Exposure:'),   0, 0)
        layout.addWidget(self._exposure_spin,   0, 1)
        layout.addWidget(QLabel('Gain:'),       1, 0)
        layout.addWidget(self._gain_spin,       1, 1)
        layout.addWidget(QLabel('Black level:'),2, 0)
        layout.addWidget(self._black_spin,      2, 1)
        return g

    def _build_display_group(self) -> QGroupBox:
        g = QGroupBox('Display')
        layout = QGridLayout(g)

        self._overlay_check = QCheckBox('Show blob overlay')
        self._overlay_check.setChecked(True)

        self._false_color_check = QCheckBox('False colour (hot)')
        self._false_color_check.setChecked(False)

        self._rotation_combo = QComboBox()
        self._rotation_combo.addItems(['0°', '90° CW', '180°', '90° CCW'])
        self._rotation_combo.currentIndexChanged.connect(self._on_rotation_changed)

        self._bg_check = QCheckBox('Subtract background')
        self._bg_check.setChecked(False)
        self._bg_check.setEnabled(False)

        self._show_original_check = QCheckBox('Show original (view only)')
        self._show_original_check.setChecked(False)
        self._show_original_check.setEnabled(False)
        self._show_original_check.setToolTip(
            'Display the raw (non-subtracted) frame; analysis still uses the BG-subtracted image')

        self._capture_bg_btn = QPushButton('Capture background (B)')
        self._capture_bg_btn.setEnabled(False)
        self._capture_bg_btn.clicked.connect(self._on_capture_background)

        layout.addWidget(self._overlay_check,        0, 0, 1, 2)
        layout.addWidget(self._false_color_check,    1, 0, 1, 2)
        layout.addWidget(QLabel('Rotation:'),        2, 0)
        layout.addWidget(self._rotation_combo,       2, 1)
        layout.addWidget(self._bg_check,             3, 0, 1, 2)
        layout.addWidget(self._show_original_check,  4, 0, 1, 2)
        layout.addWidget(self._capture_bg_btn,       5, 0, 1, 2)
        return g

    def _build_capture_group(self) -> QGroupBox:
        g = QGroupBox('Capture')
        layout = QGridLayout(g)

        self._snapshot_btn = QPushButton('Snapshot (Space)')
        self._snapshot_btn.setEnabled(False)
        self._snapshot_btn.clicked.connect(self._on_snapshot)

        self._record_btn = QPushButton('Start Recording (R)')
        self._record_btn.setEnabled(False)
        self._record_btn.setCheckable(True)
        self._record_btn.clicked.connect(self._on_record_toggle)

        self._rec_label = QLabel('—')

        self._dir_edit = QLineEdit(r'E:\camera_data')
        browse_btn = QPushButton('…')
        browse_btn.setMaximumWidth(28)
        browse_btn.clicked.connect(self._browse_dir)

        self._prefix_edit = QLineEdit('YYYYMMDD_cam_')

        layout.addWidget(self._snapshot_btn,   0, 0, 1, 2)
        layout.addWidget(self._record_btn,     1, 0, 1, 2)
        layout.addWidget(self._rec_label,      2, 0, 1, 2)
        layout.addWidget(QLabel('Dir:'),       3, 0)
        layout.addWidget(self._dir_edit,       3, 1)
        layout.addWidget(browse_btn,           3, 2)
        layout.addWidget(QLabel('Prefix:'),    4, 0)
        layout.addWidget(self._prefix_edit,    4, 1, 1, 2)
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
        self._cmin_slider.setToolTip('Display min')

        self._cmax_slider = QSlider(Qt.Horizontal)
        self._cmax_slider.setRange(0, 4095)
        self._cmax_slider.setValue(4095)
        self._cmax_slider.setEnabled(False)
        self._cmax_slider.setToolTip('Display max')

        self._cmin_label = QLabel('Min: 0')
        self._cmax_label = QLabel('Max: 4095')

        self._cmin_slider.valueChanged.connect(
            lambda v: self._cmin_label.setText(f'Min: {v}'))
        self._cmax_slider.valueChanged.connect(
            lambda v: self._cmax_label.setText(f'Max: {v}'))

        # Zoom controls
        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        zoom_out_btn = QPushButton('−')
        zoom_out_btn.setMaximumWidth(26)
        zoom_out_btn.setToolTip('Zoom out  (−)')
        zoom_out_btn.clicked.connect(lambda: self._live.zoom_by(1 / 1.25))

        self._zoom_label = QLabel('1.0×')
        self._zoom_label.setMinimumWidth(40)
        self._zoom_label.setAlignment(Qt.AlignCenter)

        zoom_in_btn = QPushButton('+')
        zoom_in_btn.setMaximumWidth(26)
        zoom_in_btn.setToolTip('Zoom in  (+)')
        zoom_in_btn.clicked.connect(lambda: self._live.zoom_by(1.25))

        zoom_fit_btn = QPushButton('Fit')
        zoom_fit_btn.setMaximumWidth(34)
        zoom_fit_btn.setToolTip('Reset to fit  (double-click image or 0)')
        zoom_fit_btn.clicked.connect(self._live.reset_zoom)

        hl.addWidget(self._contrast_auto)
        hl.addWidget(self._cmin_label)
        hl.addWidget(self._cmin_slider)
        hl.addWidget(self._cmax_label)
        hl.addWidget(self._cmax_slider)
        hl.addWidget(sep)
        hl.addWidget(zoom_out_btn)
        hl.addWidget(self._zoom_label)
        hl.addWidget(zoom_in_btn)
        hl.addWidget(zoom_fit_btn)
        return w

    def _build_analysis_group(self) -> QGroupBox:
        g = QGroupBox('Analysis Settings')
        layout = QGridLayout(g)

        self._analyze_check = QCheckBox('Enable analysis')
        self._analyze_check.setChecked(False)
        self._analyze_check.setEnabled(False)
        self._analyze_check.setToolTip(
            'Analyze the current display image (raw frame if no background subtraction, '
            'background-subtracted frame if "Subtract background" is on)')

        self._thresh_spin = QSpinBox()
        self._thresh_spin.setRange(1, 4095)
        self._thresh_spin.setValue(200)
        self._thresh_spin.setToolTip('Pixel threshold for blob detection')

        self._minarea_spin = QSpinBox()
        self._minarea_spin.setRange(1, 10000)
        self._minarea_spin.setValue(1)
        self._minarea_spin.setToolTip('Minimum blob area (pixels)')

        self._psf_sigma_spin = QDoubleSpinBox()
        self._psf_sigma_spin.setRange(0.0, 50.0)
        self._psf_sigma_spin.setValue(0.0)
        self._psf_sigma_spin.setSingleStep(0.1)
        self._psf_sigma_spin.setDecimals(2)
        self._psf_sigma_spin.setSpecialValueText('not set')
        self._psf_sigma_spin.setToolTip(
            'Expected PSF σ in pixels for a single particle.\n'
            'Set to 0 to disable σ-based cluster detection.\n'
            'Use "Set PSF σ" to calibrate from a known single-particle frame.')

        self._set_psf_btn = QPushButton('Set PSF σ')
        self._set_psf_btn.setToolTip(
            'While a single particle is trapped, click to set the PSF σ\n'
            'from the current Gaussian fit result.')
        self._set_psf_btn.clicked.connect(self._on_set_psf_sigma)

        self._sigma_tol_spin = QDoubleSpinBox()
        self._sigma_tol_spin.setRange(0.1, 2.0)
        self._sigma_tol_spin.setValue(0.5)
        self._sigma_tol_spin.setSingleStep(0.1)
        self._sigma_tol_spin.setDecimals(1)
        self._sigma_tol_spin.setToolTip(
            'Cluster threshold: blob called "cluster" when\n'
            'fitted σ > PSF σ × (1 + tolerance).\n'
            'Default 0.5 → cluster when σ > 1.5 × PSF σ.')

        self._update_plot_check = QCheckBox('Update plots')
        self._update_plot_check.setChecked(True)

        # ROI selection
        self._roi_btn = QPushButton('Draw ROI')
        self._roi_btn.setCheckable(True)
        self._roi_btn.setEnabled(False)
        self._roi_btn.setToolTip(
            'Click and drag on the live view to set the analysis region of interest')
        self._roi_btn.clicked.connect(self._on_roi_btn_clicked)

        self._roi_clear_btn = QPushButton('Clear ROI')
        self._roi_clear_btn.setEnabled(False)
        self._roi_clear_btn.clicked.connect(self._on_roi_clear)

        self._roi_label = QLabel('Full frame')
        self._roi_label.setAlignment(Qt.AlignCenter)
        self._roi_label.setStyleSheet('color: #00e5ff; font-size: 9pt;')

        layout.addWidget(self._analyze_check,         0, 0, 1, 2)
        layout.addWidget(QLabel('Threshold:'),        1, 0)
        layout.addWidget(self._thresh_spin,           1, 1)
        layout.addWidget(QLabel('Min area (px²):'),   2, 0)
        layout.addWidget(self._minarea_spin,          2, 1)
        layout.addWidget(QLabel('PSF σ (px):'),       3, 0)
        layout.addWidget(self._psf_sigma_spin,        3, 1)
        layout.addWidget(self._set_psf_btn,           4, 0, 1, 2)
        layout.addWidget(QLabel('σ tolerance:'),      5, 0)
        layout.addWidget(self._sigma_tol_spin,        5, 1)
        layout.addWidget(self._update_plot_check,     6, 0, 1, 2)
        layout.addWidget(self._roi_btn,               7, 0)
        layout.addWidget(self._roi_clear_btn,         7, 1)
        layout.addWidget(self._roi_label,             8, 0, 1, 2)
        return g

    def _build_result_group(self) -> QGroupBox:
        g = QGroupBox('Result')
        layout = QVBoxLayout(g)

        self._result_label = QLabel('—')
        self._result_label.setAlignment(Qt.AlignCenter)
        self._result_label.setFont(QFont('Arial', 15, QFont.Bold))
        self._result_label.setMinimumHeight(60)
        self._result_label.setFrameShape(QFrame.Box)
        self._result_label.setStyleSheet(
            'QLabel { border: 2px solid #555; border-radius: 4px; padding: 4px; }')

        self._detail_label = QLabel('')
        self._detail_label.setAlignment(Qt.AlignCenter)
        self._detail_label.setWordWrap(True)
        self._detail_label.setFont(QFont('Arial', 9))

        layout.addWidget(self._result_label)
        layout.addWidget(self._detail_label)
        return g

    def _build_status_bar(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(28)
        hl = QHBoxLayout(w)
        hl.setContentsMargins(6, 2, 6, 2)

        self._fps_label      = QLabel('FPS: —')
        self._frame_label    = QLabel('Frame: 0')
        self._size_label     = QLabel('—')
        self._status_label   = QLabel('Not connected')
        self._rec_status_lbl = QLabel('')

        for lbl in [self._fps_label, self._frame_label, self._size_label,
                    self._rec_status_lbl]:
            hl.addWidget(lbl)
            sep = QFrame()
            sep.setFrameShape(QFrame.VLine)
            hl.addWidget(sep)

        hl.addStretch()
        hl.addWidget(self._status_label)
        return w

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence('Space'), self).activated.connect(self._on_snapshot)
        QShortcut(QKeySequence('R'),     self).activated.connect(self._on_record_toggle)
        QShortcut(QKeySequence('B'),     self).activated.connect(self._on_capture_background)
        QShortcut(QKeySequence('+'),     self).activated.connect(lambda: self._live.zoom_by(1.25))
        QShortcut(QKeySequence('='),     self).activated.connect(lambda: self._live.zoom_by(1.25))
        QShortcut(QKeySequence('-'),     self).activated.connect(lambda: self._live.zoom_by(1/1.25))
        QShortcut(QKeySequence('0'),     self).activated.connect(self._live.reset_zoom)

    def _on_zoom_changed(self, zoom: float) -> None:
        self._zoom_label.setText(f'{zoom:.1f}×')

    # ── Hardware actions ───────────────────────────────────────────────────────

    def _on_scan(self) -> None:
        cls = MockCamera if self._use_mock else ThorlabsCamera
        try:
            serials = cls.list_serials()
        except Exception as e:
            QMessageBox.warning(self, 'Scan failed', str(e))
            return
        self._serial_combo.clear()
        for s in serials:
            self._serial_combo.addItem(s)
        if not serials:
            self._status_label.setText('No cameras found')
        else:
            self._status_label.setText(f'{len(serials)} camera(s) found')

    def _on_connect(self) -> None:
        serial = self._serial_combo.currentText().strip() or None
        cam_cls = MockCamera if self._use_mock else ThorlabsCamera
        try:
            cam = cam_cls()
            cam.connect(serial)
        except Exception as exc:
            QMessageBox.critical(self, 'Connection error', str(exc))
            return

        self._camera = cam
        self._frame_queue = queue.Queue(maxsize=2)
        self._cam_thread = CameraThread(cam, self._frame_queue)
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
        self._black_spin.setValue(cam.black_level)

        for w in [self._exposure_spin, self._gain_spin, self._black_spin,
                  self._snapshot_btn, self._record_btn, self._capture_bg_btn,
                  self._analyze_check, self._roi_btn, self._tmpl_save_btn]:
            w.setEnabled(True)
        self._analyze_check.setChecked(True)
        self._disconnect_btn.setEnabled(True)
        self._connect_btn.setEnabled(False)
        self._cmax_slider.setRange(0, (1 << cam.bit_depth) - 1)
        self._cmax_slider.setValue((1 << cam.bit_depth) - 1)
        self._thresh_spin.setRange(1, (1 << cam.bit_depth) - 1)

        self._display_timer.start(1000 // _DISPLAY_FPS)
        self._fps_t0 = time.time()
        self._fps_counter = 0
        self._status_label.setText('Streaming…')
        if self._use_mock:
            self._status_label.setText('Streaming (mock camera)')

    def _on_disconnect(self) -> None:
        self._display_timer.stop()
        if self._is_recording:
            self._stop_recording()

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

        for w in [self._exposure_spin, self._gain_spin, self._black_spin,
                  self._snapshot_btn, self._record_btn, self._capture_bg_btn,
                  self._tmpl_save_btn]:
            w.setEnabled(False)
        self._bg_check.setChecked(False)
        self._bg_check.setEnabled(False)
        self._show_original_check.setChecked(False)
        self._show_original_check.setEnabled(False)
        self._analyze_check.setChecked(False)
        self._analyze_check.setEnabled(False)
        self._roi_btn.setEnabled(False)
        self._roi_btn.setChecked(False)
        self._roi_btn.setText('Draw ROI')
        self._roi_clear_btn.setEnabled(False)
        self._roi_label.setText('Full frame')
        self._live.clear_roi()
        self._background = None
        self._disconnect_btn.setEnabled(False)
        self._connect_btn.setEnabled(True)
        self._live.setText('No camera connected')
        self._live.setPixmap(QPixmap())
        self._result_label.setText('—')
        self._detail_label.setText('')
        self._status_label.setText('Disconnected')
        self._fps_label.setText('FPS: —')

    def _on_camera_error(self, msg: str) -> None:
        self._display_timer.stop()
        QMessageBox.critical(self, 'Camera error', msg)
        self._status_label.setText('Error')

    # ── Settings callbacks ─────────────────────────────────────────────────────

    def _on_exposure_changed(self, value: int) -> None:
        if self._camera:
            self._camera.exposure_us = value

    def _on_gain_changed(self, value: float) -> None:
        if self._camera:
            self._camera.gain = value

    def _on_black_changed(self, value: int) -> None:
        if self._camera:
            self._camera.black_level = value

    def _on_contrast_mode_changed(self) -> None:
        manual = not self._contrast_auto.isChecked()
        self._cmin_slider.setEnabled(manual)
        self._cmax_slider.setEnabled(manual)

    # ── Frame display loop ─────────────────────────────────────────────────────

    def _poll_frame(self) -> None:
        try:
            frame = self._frame_queue.get_nowait()
        except queue.Empty:
            return

        # Apply rotation first so everything downstream sees the rotated frame
        if self._rotation_k:
            frame = np.ascontiguousarray(np.rot90(frame, self._rotation_k))

        self._last_frame = frame
        self._frame_count += 1
        self._fps_counter += 1

        # FPS estimate
        now = time.time()
        if now - self._fps_t0 >= 1.0:
            self._fps = self._fps_counter / (now - self._fps_t0)
            self._fps_counter = 0
            self._fps_t0 = now
            self._fps_label.setText(f'FPS: {self._fps:.1f}')
        self._frame_label.setText(f'Frame: {self._frame_count}')

        # Recording
        if self._is_recording:
            if len(self._rec_frames) < _MAX_REC_FRAMES:
                self._rec_frames.append(frame.copy())
                elapsed = now - self._rec_start
                n = len(self._rec_frames)
                self._rec_label.setText(f'Recording… {n} frames  {elapsed:.1f}s')
                self._rec_status_lbl.setText(f'REC {n}fr')
            else:
                self._stop_recording()

        # Background subtraction
        display_image = frame
        if self._bg_check.isChecked() and self._background is not None:
            display_image = np.clip(
                frame.astype(np.int32) - self._background.astype(np.int32),
                0, None).astype(np.uint16)

        # Submit to background analysis thread, cropped to ROI if set
        if self._analyze_check.isChecked() and self._analysis_thread is not None:
            analysis_image = display_image
            roi_offset = (0, 0)
            roi = self._live.roi
            if roi is not None:
                rx0, ry0, rx1, ry1 = roi
                rx0 = max(0, rx0);  ry0 = max(0, ry0)
                rx1 = min(frame.shape[1], rx1);  ry1 = min(frame.shape[0], ry1)
                if rx1 > rx0 and ry1 > ry0:
                    analysis_image = display_image[ry0:ry1, rx0:rx1]
                    roi_offset = (rx0, ry0)
            self._analysis_thread.submit(
                analysis_image,
                self._thresh_spin.value(),
                self._minarea_spin.value(),
                self._psf_sigma_spin.value(),
                self._sigma_tol_spin.value(),
                roi_offset,
            )

        # What to render: raw frame or background-subtracted
        # "Show original" lets the user see the raw image while analysis runs on BG-sub
        view_image = (frame
                      if (self._show_original_check.isChecked()
                          and self._show_original_check.isEnabled())
                      else display_image)

        # Convert frame to QImage for display (fast LUT path)
        qimg = self._frame_to_qimage(view_image)
        self._live.update_frame(
            qimg, self._last_blobs, self._last_cls,
            self._overlay_check.isChecked(),
            frame.shape[1], frame.shape[0],
        )

    def _on_analysis_result(self, result: dict) -> None:
        """Called from the main thread when AnalysisThread emits result_ready."""
        self._last_blobs  = result['blobs']
        self._last_cls    = result['classification']
        self._last_result = result
        self._update_result_display(result)

        # Throttle matplotlib: redraw at most every _PLOT_MIN_INTERVAL seconds
        now = time.time()
        if (self._update_plot_check.isChecked()
                and self._last_frame is not None
                and now - self._plot_last_t >= _PLOT_MIN_INTERVAL):
            self._plot_last_t = now
            self._analysis_canvas.update_plots(
                self._last_frame, result['blobs'], result['classification'])

    def _frame_to_qimage(self, frame: np.ndarray) -> QImage:
        if self._contrast_auto.isChecked():
            # Subsample 4× in each axis (16× fewer pixels) for fast percentile
            sub = frame[::4, ::4]
            lo  = int(np.percentile(sub, 1))
            hi  = int(np.percentile(sub, 99))
        else:
            lo = self._cmin_slider.value()
            hi = self._cmax_slider.value()
        hi = max(hi, lo + 1)

        # Rebuild LUT only when lo/hi changes (not every frame)
        if (lo, hi) != self._last_contrast:
            self._last_contrast = (lo, hi)
            span = hi - lo
            idx  = np.arange(65536, dtype=np.int32)
            _CONTRAST_LUT[:] = np.clip((idx - lo) * 255 // span, 0, 255).astype(np.uint8)

        h, w = frame.shape

        # Apply LUT in-place into a persistent buffer (no per-frame allocation)
        if self._disp_buf is None or self._disp_buf.shape != frame.shape:
            self._disp_buf = np.empty((h, w), dtype=np.uint8)
        np.take(_CONTRAST_LUT, frame, out=self._disp_buf)

        if self._false_color_check.isChecked():
            if self._false_color_buf is None or self._false_color_buf.shape[:2] != (h, w):
                self._false_color_buf = np.empty((h, w, 3), dtype=np.uint8)
            # Apply hot LUT via fancy indexing into persistent buffer
            self._false_color_buf[:] = _HOT_LUT[self._disp_buf]
            return QImage(self._false_color_buf.data, w, h, w * 3, QImage.Format_RGB888)
        else:
            return QImage(self._disp_buf.data, w, h, w, QImage.Format_Grayscale8)

    # ── Analysis display ───────────────────────────────────────────────────────

    def _update_result_display(self, result: dict) -> None:
        cls = result['classification']
        msg = result['message']
        color = _RESULT_COLORS.get(cls, '#888888')

        labels = {
            'single':    'SINGLE\nPARTICLE',
            'cluster':   'CLUSTER',
            'uncertain': 'UNCERTAIN',
            'none':      'NO PARTICLE',
        }
        self._result_label.setText(labels.get(cls, '?'))
        self._result_label.setStyleSheet(
            f'QLabel {{ border: 2px solid {color}; border-radius: 4px; '
            f'padding: 4px; color: {color}; }}')

        lines = [msg]
        for i, b in enumerate(result.get('blobs', []), 1):
            sigma_str = (f'σ={b["sigma"]:.2f}px'
                         if b.get('fit_ok') and not math.isnan(b['sigma'])
                         else 'σ=fit failed')
            lines.append(
                f'  #{i}: {sigma_str}  '
                f'peak={b["peak"]}  area={b["area"]}px²')
        self._detail_label.setText('\n'.join(lines))

    # ── Capture ────────────────────────────────────────────────────────────────

    def _on_snapshot(self) -> None:
        if self._last_frame is None:
            return
        os.makedirs(self._dir_edit.text(), exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        fname = os.path.join(self._dir_edit.text(),
                             f'{self._prefix_edit.text()}{ts}.tiff')
        try:
            import tifffile
            tifffile.imwrite(fname, self._last_frame)
        except ImportError:
            # Fallback: save via numpy as 16-bit raw (still loadable in ImageJ)
            np.save(fname.replace('.tiff', '.npy'), self._last_frame)
            fname = fname.replace('.tiff', '.npy')
        self._status_label.setText(f'Saved {os.path.basename(fname)}')

    def _on_record_toggle(self) -> None:
        if not self._is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        if self._last_frame is None:
            return
        os.makedirs(self._dir_edit.text(), exist_ok=True)
        self._rec_frames   = []
        self._rec_start    = time.time()
        self._is_recording = True
        self._record_btn.setText('Stop Recording (R)')
        self._record_btn.setChecked(True)
        self._status_label.setText('Recording…')

    def _stop_recording(self) -> None:
        self._is_recording = False
        self._record_btn.setText('Start Recording (R)')
        self._record_btn.setChecked(False)
        frames = self._rec_frames
        self._rec_frames = []
        self._rec_label.setText('—')
        self._rec_status_lbl.setText('')

        if not frames:
            return
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(self._dir_edit.text(),
                             f'{self._prefix_edit.text()}{ts}.hdf5')
        try:
            import h5py
            stack = np.stack(frames, axis=0)  # (N, H, W) uint16
            with h5py.File(fname, 'w') as f:
                g = f.create_group('data')
                g.attrs['timestamp'] = self._rec_start
                g.attrs['n_frames']  = len(frames)
                g.attrs['fps']       = self._fps
                g.attrs['exposure_us'] = (self._camera.exposure_us
                                          if self._camera else 0)
                g.create_dataset('frames', data=stack, dtype=np.uint16,
                                 compression='gzip', compression_opts=1)
            self._status_label.setText(
                f'Saved {len(frames)} frames → {os.path.basename(fname)}')
        except Exception as exc:
            QMessageBox.critical(self, 'Save error', str(exc))

    def _on_rotation_changed(self, idx: int) -> None:
        # np.rot90 k: combo index 0→k=0, 1→k=3(90CW), 2→k=2(180), 3→k=1(90CCW)
        self._rotation_k = [0, 3, 2, 1][idx]
        # Invalidate background — it was captured in the old orientation
        if self._background is not None:
            self._background = None
            self._bg_check.setChecked(False)
            self._bg_check.setEnabled(False)
            self._show_original_check.setChecked(False)
            self._show_original_check.setEnabled(False)
            self._status_label.setText('Rotation changed — recapture background')
        # Clear ROI since image dimensions may have changed
        self._live.clear_roi()
        # Reset display buffers since frame shape may have changed
        self._disp_buf        = None
        self._false_color_buf = None
        self._live.reset_zoom()

    def _on_capture_background(self) -> None:
        if self._last_frame is None:
            return
        self._background = self._last_frame.copy()
        self._bg_check.setEnabled(True)
        self._bg_check.setChecked(True)
        self._show_original_check.setEnabled(True)
        # Enable analysis and ROI now that we have a background
        self._analyze_check.setEnabled(True)
        self._analyze_check.setChecked(True)
        self._roi_btn.setEnabled(True)
        self._status_label.setText('Background captured — analysis enabled')

    # ── Templates UI ──────────────────────────────────────────────────────────

    def _build_templates_group(self) -> QGroupBox:
        g = QGroupBox('Templates')
        layout = QGridLayout(g)

        self._tmpl_note_edit = QLineEdit()
        self._tmpl_note_edit.setPlaceholderText('Optional note…')

        self._tmpl_save_btn = QPushButton('Save as template')
        self._tmpl_save_btn.setEnabled(False)
        self._tmpl_save_btn.setToolTip(
            'Save a crop of the current blob as a single-particle reference image')
        self._tmpl_save_btn.clicked.connect(self._on_save_template)

        self._tmpl_view_btn = QPushButton('View templates…')
        self._tmpl_view_btn.clicked.connect(self._on_view_templates)

        self._tmpl_load_psf_btn = QPushButton('Load PSF σ from templates')
        self._tmpl_load_psf_btn.setToolTip(
            'Compute the mean fitted σ across all saved templates and set it '
            'as the PSF σ reference used for single/cluster classification.')
        self._tmpl_load_psf_btn.clicked.connect(self._on_load_psf_from_templates)

        self._tmpl_path_edit = QLineEdit(r'E:\camera_data\particle_templates.hdf5')
        self._tmpl_path_edit.setToolTip('HDF5 file where templates are stored')
        self._tmpl_path_edit.editingFinished.connect(self._refresh_template_store)
        tmpl_browse_btn = QPushButton('…')
        tmpl_browse_btn.setMaximumWidth(28)
        tmpl_browse_btn.clicked.connect(self._browse_template_file)

        self._tmpl_count_label = QLabel('0 saved')
        self._tmpl_count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(QLabel('Note:'),           0, 0)
        layout.addWidget(self._tmpl_note_edit,      0, 1, 1, 2)
        layout.addWidget(self._tmpl_save_btn,         1, 0, 1, 2)
        layout.addWidget(self._tmpl_count_label,     1, 2)
        layout.addWidget(self._tmpl_view_btn,        2, 0, 1, 3)
        layout.addWidget(self._tmpl_load_psf_btn,    3, 0, 1, 3)
        layout.addWidget(QLabel('File:'),            4, 0)
        layout.addWidget(self._tmpl_path_edit,       4, 1)
        layout.addWidget(tmpl_browse_btn,            4, 2)
        return g

    # ── PSF calibration ────────────────────────────────────────────────────────

    def _on_set_psf_sigma(self) -> None:
        """Set the reference PSF σ from the current single-blob fit result."""
        blobs = self._last_blobs
        if not blobs:
            self._status_label.setText('No blob detected — cannot calibrate PSF σ')
            return
        if len(blobs) > 1:
            self._status_label.setText(
                'Multiple blobs detected — calibrate with a single particle')
            return
        b = blobs[0]
        if not b.get('fit_ok') or math.isnan(b.get('sigma', float('nan'))):
            self._status_label.setText('Gaussian fit failed — cannot calibrate PSF σ')
            return
        sigma = b['sigma']
        self._psf_sigma_spin.setValue(sigma)
        self._status_label.setText(f'PSF σ set to {sigma:.2f} px from current fit')

    # ── ROI ────────────────────────────────────────────────────────────────────

    def _on_roi_btn_clicked(self, checked: bool) -> None:
        self._live.set_roi_mode(checked)
        self._roi_btn.setText('Cancel ROI' if checked else 'Draw ROI')

    def _on_roi_clear(self) -> None:
        self._live.clear_roi()

    def _on_roi_changed(self, roi) -> None:
        # Reset draw-mode button when user finishes drawing (or clears)
        self._roi_btn.setChecked(False)
        self._roi_btn.setText('Draw ROI')
        if roi is None:
            self._roi_label.setText('Full frame')
            self._roi_clear_btn.setEnabled(False)
        else:
            x0, y0, x1, y1 = roi
            self._roi_label.setText(f'ROI: {x1-x0}×{y1-y0} px  ({x0},{y0})')
            self._roi_clear_btn.setEnabled(True)

    # ── Template handlers ──────────────────────────────────────────────────────

    def _refresh_template_store(self) -> None:
        path = self._tmpl_path_edit.text().strip()
        self._template_store = TemplateStore(path)
        n = self._template_store.count()
        self._tmpl_count_label.setText(f'{n} saved')

    def _browse_template_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, 'Template file', self._tmpl_path_edit.text(),
            'HDF5 files (*.hdf5 *.h5)')
        if path:
            self._tmpl_path_edit.setText(path)
            self._refresh_template_store()

    def _on_save_template(self) -> None:
        if not self._last_blobs or self._last_frame is None:
            self._status_label.setText('No blob detected — cannot save template')
            return
        if len(self._last_blobs) > 1:
            self._status_label.setText(
                'Multiple blobs — save a template only for a single particle')
            return

        b     = self._last_blobs[0]
        frame = self._last_frame
        H, W  = frame.shape
        cx, cy = b['cx'], b['cy']

        # Generous crop: 6× fitted σ, minimum 30 px each side
        half = max(30, int(b.get('sigma', 5) * 6))
        x0 = max(0, int(cx) - half);  x1 = min(W, int(cx) + half + 1)
        y0 = max(0, int(cy) - half);  y1 = min(H, int(cy) + half + 1)
        crop       = frame[y0:y1, x0:x1]
        cx_in_crop = cx - x0
        cy_in_crop = cy - y0

        meta = {
            'timestamp':    datetime.datetime.now().isoformat(timespec='seconds'),
            'note':         self._tmpl_note_edit.text().strip(),
            'exposure_us':  self._camera.exposure_us  if self._camera else 0,
            'gain':         self._camera.gain          if self._camera else 0.0,
            'camera_width': W,
            'camera_height': H,
            'bit_depth':    self._camera.bit_depth     if self._camera else 0,
            'sigma':        b.get('sigma',       float('nan')),
            'sigma_x':      b.get('sigma_x',     float('nan')),
            'sigma_y':      b.get('sigma_y',     float('nan')),
            'fit_quality':  b.get('fit_quality', float('nan')),
            'fit_ok':       b.get('fit_ok',      False),
            'peak':         b['peak'],
            'cx':           cx,
            'cy':           cy,
            'psf_sigma_ref': self._psf_sigma_spin.value(),
        }

        try:
            key = self._template_store.save(crop, cx_in_crop, cy_in_crop, meta)
            n   = self._template_store.count()
            self._tmpl_count_label.setText(f'{n} saved')
            self._tmpl_note_edit.clear()
            self._status_label.setText(f'Template saved: {key}')
        except Exception as exc:
            QMessageBox.critical(self, 'Save failed', str(exc))

    def _on_load_psf_from_templates(self) -> None:
        templates = self._template_store.load_all()
        if not templates:
            QMessageBox.information(self, 'No templates',
                                    'No templates found in the current file.')
            return

        sigmas = []
        skipped = 0
        for key, crop, cx, cy, meta in templates:
            fit_ok = meta.get('fit_ok', False)
            # h5py stores bools as numpy bool_ — coerce to Python bool
            try:
                fit_ok = bool(fit_ok)
            except Exception:
                fit_ok = False
            sigma = meta.get('sigma', float('nan'))
            try:
                sigma = float(sigma)
            except (TypeError, ValueError):
                sigma = float('nan')
            if fit_ok and not math.isnan(sigma) and sigma > 0:
                sigmas.append(sigma)
            else:
                skipped += 1

        if not sigmas:
            QMessageBox.warning(self, 'No usable templates',
                                'None of the saved templates have a valid Gaussian fit.')
            return

        mean_sigma = float(np.mean(sigmas))
        std_sigma  = float(np.std(sigmas))
        self._psf_sigma_spin.setValue(mean_sigma)

        detail = '\n'.join(f'  {s:.3f} px' for s in sigmas)
        msg = (f'PSF σ set to {mean_sigma:.3f} px\n'
               f'(mean of {len(sigmas)} templates, std={std_sigma:.3f} px)\n\n'
               f'Individual values:\n{detail}')
        if skipped:
            msg += f'\n\n{skipped} template(s) skipped (fit failed or no σ stored).'
        QMessageBox.information(self, 'PSF σ loaded', msg)
        self._status_label.setText(
            f'PSF σ = {mean_sigma:.3f} px  (from {len(sigmas)} templates)')

    def _on_view_templates(self) -> None:
        if self._template_store is None:
            return
        dlg = TemplateGalleryDialog(self._template_store, parent=self)
        dlg.exec_()
        # Refresh count in case user deleted entries
        n = self._template_store.count()
        self._tmpl_count_label.setText(f'{n} saved')

    # ── Misc ───────────────────────────────────────────────────────────────────

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, 'Select output directory', self._dir_edit.text())
        if d:
            self._dir_edit.setText(d)

    def closeEvent(self, event) -> None:
        self._on_disconnect()
        super().closeEvent(event)


# ── Entry point ───────────────────────────────────────────────────────────────

def launch_gui(use_mock: bool = False) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    win = CameraMainWindow(use_mock=use_mock)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    mock = '--mock' in sys.argv
    launch_gui(use_mock=mock)
