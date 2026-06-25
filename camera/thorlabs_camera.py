"""
Thorlabs CS165MU1 camera driver.

Wraps thorlabs_tsi_sdk for continuous live-view acquisition.
Falls back to MockCamera if the SDK is not installed — useful for
developing / testing the GUI away from the lab PC.

The SDK is NOT on PyPI.  Install it from the wheel bundled with the
"Thorlabs Scientific Camera Support" software (free download from thorlabs.com):

    1. Install "Scientific Camera Support" from thorlabs.com → Software
    2. pip install "<install_dir>\\Scientific Camera Interfaces\\SDK\\Python Toolkit\\thorlabs_tsi_sdk-*.whl"

Typical install dir:
    C:\\Program Files\\Thorlabs\\Scientific Imaging\\Scientific Camera Support

This module tries to locate and install the wheel automatically on first import.
"""

import os
import glob
import subprocess
import sys
import time
import math
import numpy as np
from typing import Optional, Tuple, List

# ── Locate Thorlabs SDK DLL directory ────────────────────────────────────────
# Search order: repo-local SDK download, then standard ThorCam install paths.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DLL_SEARCH_ROOTS = [
    # Repo-local SDK download (scientific_camera_interfaces_windows-*/)
    os.path.join(_REPO_ROOT, 'scientific_camera_interfaces_windows-2.1',
                 'Scientific Camera Interfaces', 'SDK',
                 'Python Toolkit', 'dlls', '64_lib'),
    # Standard ThorCam installation
    r'C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support'
    r'\Scientific Camera Interfaces\SDK\Python Toolkit\dlls\64_lib',
    r'C:\Program Files (x86)\Thorlabs\Scientific Imaging\Scientific Camera Support'
    r'\Scientific Camera Interfaces\SDK\Python Toolkit\dlls\64_lib',
]

_SDK_INSTALL_DIR: Optional[str] = None
for _root in _DLL_SEARCH_ROOTS:
    if os.path.isfile(os.path.join(_root, 'thorlabs_tsi_camera_sdk.dll')):
        _SDK_INSTALL_DIR = _root
        break

# Add DLL directory so the SDK can load its native libraries
if _SDK_INSTALL_DIR:
    os.environ['PATH'] = _SDK_INSTALL_DIR + os.pathsep + os.environ.get('PATH', '')
    try:
        os.add_dll_directory(_SDK_INSTALL_DIR)
    except AttributeError:
        pass  # Python < 3.8 — rely on PATH

def _try_install_sdk_wheel() -> bool:
    """Auto-install the Thorlabs SDK wheel if it exists but isn't pip-installed."""
    if not _SDK_INSTALL_DIR:
        return False
    wheels = glob.glob(os.path.join(_SDK_INSTALL_DIR, 'thorlabs_tsi_sdk*.whl'))
    if not wheels:
        return False
    wheel = sorted(wheels)[-1]  # newest version
    print(f'[camera] Installing Thorlabs SDK from {wheel}')
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', wheel],
        capture_output=True, text=True,
    )
    return result.returncode == 0

try:
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE
    _HAS_SDK = True
except ImportError:
    if _try_install_sdk_wheel():
        try:
            from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE
            _HAS_SDK = True
        except ImportError:
            _HAS_SDK = False
    else:
        _HAS_SDK = False

_SDK_INSTALL_HINT = (
    'thorlabs_tsi_sdk not found.\n\n'
    '1. Download and install "Scientific Camera Support" from thorlabs.com → Software\n'
    '2. Then run:\n'
    '   pip install "C:\\Program Files\\Thorlabs\\Scientific Imaging\\'
    'Scientific Camera Support\\Scientific Camera Interfaces\\'
    'SDK\\Python Toolkit\\thorlabs_tsi_sdk-*.whl"\n\n'
    'Use --mock flag to run the GUI without hardware.'
)


# ── Blob analysis ─────────────────────────────────────────────────────────────

def _fit_gaussian_2d(image: np.ndarray, cx: float, cy: float,
                     est_sigma: float = 4.0) -> dict:
    """
    Fit a 2D elliptical Gaussian to a window around (cx, cy).

    Model: f(x,y) = offset + amp * exp(-0.5*((x-x0)²/σx² + (y-y0)²/σy²))

    Returns dict with keys: fit_ok, sigma (geometric mean √(σx·σy)),
    sigma_x, sigma_y, amplitude, fit_quality (residual RMS / amplitude).
    """
    from scipy.optimize import curve_fit

    H, W = image.shape
    half = max(6, int(est_sigma * 4))
    x0w = max(0, int(cx) - half);  x1w = min(W, int(cx) + half + 1)
    y0w = max(0, int(cy) - half);  y1w = min(H, int(cy) + half + 1)

    crop = image[y0w:y1w, x0w:x1w].astype(np.float64)
    if crop.size < 16:
        return {'fit_ok': False, 'sigma': float('nan'),
                'sigma_x': float('nan'), 'sigma_y': float('nan'),
                'amplitude': 0.0, 'fit_quality': float('nan')}

    yy, xx = np.mgrid[y0w:y1w, x0w:x1w].astype(np.float64)

    offset0 = float(np.percentile(crop, 20))
    amp0    = float(crop.max()) - offset0

    def model(xy, amp, x0, y0, sx, sy, offset):
        x, y = xy
        return offset + amp * np.exp(
            -0.5 * ((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2))

    try:
        popt, _ = curve_fit(
            model,
            (xx.ravel(), yy.ravel()), crop.ravel(),
            p0=[amp0, float(cx), float(cy), est_sigma, est_sigma, offset0],
            bounds=(
                [0,      x0w, y0w, 0.3,  0.3,  -np.inf],
                [np.inf, x1w, y1w, half, half,   np.inf],
            ),
            maxfev=600,
        )
        amp, fx, fy, sx, sy, offset = popt
        sx, sy = abs(sx), abs(sy)
        sigma  = math.sqrt(sx * sy)
        resid  = crop.ravel() - model((xx.ravel(), yy.ravel()), *popt)
        rms    = float(np.sqrt(np.mean(resid ** 2)))
        return {
            'fit_ok':      True,
            'sigma':       sigma,
            'sigma_x':     sx,
            'sigma_y':     sy,
            'amplitude':   float(amp),
            'fit_quality': rms / max(amp, 1.0),
        }
    except Exception:
        return {'fit_ok': False, 'sigma': float('nan'),
                'sigma_x': float('nan'), 'sigma_y': float('nan'),
                'amplitude': 0.0, 'fit_quality': float('nan')}


def analyze_frame(image: np.ndarray, threshold: int, min_area: int,
                  psf_sigma: float = 0.0, sigma_tol: float = 0.5) -> dict:
    """
    Detect bright blobs and classify as single particle / cluster / none.

    Each blob is fit with a 2D elliptical Gaussian.  For sub-wavelength
    particles the PSF width is the only reliable discriminant:

      n_blobs > 1                                  → 'cluster'
      n_blobs == 1, fit failed                     → 'uncertain'
      n_blobs == 1, psf_sigma > 0,
          fitted σ > psf_sigma * (1 + sigma_tol)  → 'cluster'
      n_blobs == 1, otherwise                      → 'single'

    Parameters
    ----------
    image      : 2-D uint16 array (raw or background-subtracted)
    threshold  : pixel value above which a pixel is part of a blob
    min_area   : blobs smaller than this (pixels) are discarded as noise
    psf_sigma  : expected PSF σ in pixels for a single particle.
                 0 (default) = not calibrated; σ is reported but not used
                 for cluster/single discrimination.
    sigma_tol  : fraction above psf_sigma that triggers 'cluster' label.
                 Default 0.5 → cluster when fitted σ > 1.5 × psf_sigma.

    Returns dict with keys:
        blobs          list of {cx, cy, area, peak, radius,
                                sigma, sigma_x, sigma_y,
                                fit_quality, fit_ok}
        n_blobs        int
        classification 'single' | 'cluster' | 'uncertain' | 'none'
        message        human-readable string
    """
    from scipy.ndimage import label as nd_label

    binary = image > threshold
    labeled, n_comp = nd_label(binary)

    est_sigma = psf_sigma if psf_sigma > 0 else 4.0

    blobs = []
    for idx in range(1, n_comp + 1):
        mask = labeled == idx
        area = int(mask.sum())
        if area < min_area:
            continue

        ys, xs = np.nonzero(mask)
        cx     = float(xs.mean())
        cy     = float(ys.mean())
        peak   = int(image[mask].max())
        radius = math.sqrt(area / math.pi)

        fit = _fit_gaussian_2d(image, cx, cy, est_sigma)
        blobs.append({
            'cx': cx, 'cy': cy,
            'area': area, 'radius': radius, 'peak': peak,
            'sigma':       fit['sigma'],
            'sigma_x':     fit['sigma_x'],
            'sigma_y':     fit['sigma_y'],
            'fit_quality': fit['fit_quality'],
            'fit_ok':      fit['fit_ok'],
        })

    n = len(blobs)
    if n == 0:
        cls, msg = 'none', 'No particle detected'
    elif n > 1:
        cls = 'cluster'
        msg = f'Cluster — {n} distinct spots'
    else:
        b = blobs[0]
        if not b['fit_ok']:
            cls = 'uncertain'
            msg = 'Gaussian fit failed — result uncertain'
        elif psf_sigma > 0:
            ratio = b['sigma'] / psf_sigma
            if ratio > 1.0 + sigma_tol:
                cls = 'cluster'
                msg = (f'Cluster — σ={b["sigma"]:.2f} px  '
                       f'({ratio:.2f}× PSF σ={psf_sigma:.2f} px)')
            else:
                cls = 'single'
                msg = (f'Single particle — σ={b["sigma"]:.2f} px  '
                       f'({ratio:.2f}× PSF σ={psf_sigma:.2f} px)')
        else:
            cls = 'single'
            msg = (f'Single particle — σ={b["sigma"]:.2f} px  '
                   f'(calibrate PSF σ to detect clusters)')

    return {'blobs': blobs, 'n_blobs': n, 'classification': cls, 'message': msg}


# ── Real camera ───────────────────────────────────────────────────────────────

class ThorlabsCamera:
    """
    Continuous-acquisition wrapper for Thorlabs TSI cameras (CS165MU1).

    Typical use::

        cam = ThorlabsCamera()
        cam.connect()
        while True:
            frame = cam.get_frame()     # returns uint16 ndarray or None
            if frame is not None:
                process(frame)
        cam.disconnect()
    """

    def __init__(self):
        self._sdk    = None
        self._cam    = None
        self._connected = False
        self.width:     int = 0
        self.height:    int = 0
        self.bit_depth: int = 12

    # ── Connection ─────────────────────────────────────────────────────────

    @staticmethod
    def sdk_available() -> bool:
        return _HAS_SDK

    @staticmethod
    def list_serials() -> List[str]:
        if not _HAS_SDK:
            return []
        sdk = TLCameraSDK()
        serials = sdk.discover_available_cameras()
        sdk.dispose()
        return serials

    def connect(self, serial: Optional[str] = None) -> None:
        if not _HAS_SDK:
            raise RuntimeError(_SDK_INSTALL_HINT)
        self._sdk = TLCameraSDK()
        serials = self._sdk.discover_available_cameras()
        if not serials:
            self._sdk.dispose()
            self._sdk = None
            raise RuntimeError('No Thorlabs cameras detected.')
        target = serial or serials[0]
        if target not in serials:
            self._sdk.dispose()
            self._sdk = None
            raise RuntimeError(
                f'Camera {target!r} not found.\nAvailable: {serials}')

        cam = self._sdk.open_camera(target)
        cam.image_poll_timeout_ms = 0          # non-blocking
        cam.frames_per_trigger_zero_for_unlimited = 0   # continuous
        cam.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
        cam.arm(2)
        cam.issue_software_trigger()

        self._cam = cam
        self.width     = cam.image_width_pixels
        self.height    = cam.image_height_pixels
        self.bit_depth = cam.bit_depth
        self._connected = True

    def disconnect(self) -> None:
        if self._cam:
            try:
                self._cam.disarm()
            except Exception:
                pass
            try:
                self._cam.dispose()
            except Exception:
                pass
            self._cam = None
        if self._sdk:
            try:
                self._sdk.dispose()
            except Exception:
                pass
            self._sdk = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Settings ────────────────────────────────────────────────────────────

    @property
    def exposure_us(self) -> int:
        return self._cam.exposure_time_us if self._cam else 0

    @exposure_us.setter
    def exposure_us(self, value: int) -> None:
        if not self._cam:
            return
        r = self._cam.exposure_time_range_us
        self._cam.exposure_time_us = int(max(r.min, min(r.max, value)))

    def exposure_range_us(self) -> Tuple[int, int]:
        if self._cam:
            r = self._cam.exposure_time_range_us
            return r.min, r.max
        return 100, 10_000_000

    @property
    def gain(self) -> float:
        return float(self._cam.gain) if self._cam else 0.0

    @gain.setter
    def gain(self, value: float) -> None:
        if not self._cam:
            return
        r = self._cam.gain_range
        self._cam.gain = int(max(r.min, min(r.max, value)))

    def gain_range(self) -> Tuple[float, float]:
        if self._cam:
            r = self._cam.gain_range
            return float(r.min), float(r.max)
        return 0.0, 480.0

    @property
    def black_level(self) -> int:
        return self._cam.black_level if self._cam else 0

    @black_level.setter
    def black_level(self, value: int) -> None:
        if self._cam:
            self._cam.black_level = int(value)

    # ── Acquisition ─────────────────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """Non-blocking poll. Returns uint16 (H, W) array or None."""
        if not self._cam:
            return None
        frame = self._cam.get_pending_frame_or_null()
        if frame is None:
            return None
        img = np.array(frame.image_buffer, dtype=np.uint16).reshape(
            self.height, self.width)
        self._cam.issue_software_trigger()
        return img

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.disconnect()


# ── Mock camera (for GUI development without hardware) ────────────────────────

class MockCamera:
    """
    Simulated camera that generates synthetic nanoparticle images.
    Cycles through: single particle → cluster (2 spots) → no particle.
    """

    width     = 640
    height    = 480
    bit_depth = 12
    _MAX_ADU  = 4095

    def __init__(self):
        self._connected  = False
        self._t0         = 0.0
        self._exposure   = 10_000
        self._gain       = 0.0
        self._black      = 0
        self._phase      = 0.0

    # ── Same interface as ThorlabsCamera ───────────────────────────────────

    @staticmethod
    def sdk_available() -> bool:
        return True

    @staticmethod
    def list_serials() -> List[str]:
        return ['MOCK-001']

    def connect(self, serial: Optional[str] = None) -> None:
        self._connected = True
        self._t0 = time.time()

    def disconnect(self) -> None:
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def exposure_us(self) -> int:
        return self._exposure

    @exposure_us.setter
    def exposure_us(self, v: int) -> None:
        self._exposure = max(100, min(10_000_000, v))

    def exposure_range_us(self) -> Tuple[int, int]:
        return 100, 10_000_000

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, v: float) -> None:
        self._gain = max(0.0, min(480.0, v))

    def gain_range(self) -> Tuple[float, float]:
        return 0.0, 480.0

    @property
    def black_level(self) -> int:
        return self._black

    @black_level.setter
    def black_level(self, v: int) -> None:
        self._black = v

    def get_frame(self) -> Optional[np.ndarray]:
        if not self._connected:
            return None
        t = time.time() - self._t0
        # Cycle every 8 s: 0-3 s single, 3-6 s cluster, 6-8 s no particle
        phase = t % 8.0
        rng = np.random.default_rng(int(t * 30))
        img = self._background(rng)
        if phase < 3.0:
            self._add_spot(img, rng, 320.0 + 4 * math.sin(t), 240.0, 4.0, 3600)
        elif phase < 6.0:
            self._add_spot(img, rng, 300.0, 235.0, 3.5, 3000)
            self._add_spot(img, rng, 345.0, 248.0, 3.5, 2800)
        # else: no particle
        return img.astype(np.uint16)

    def _background(self, rng) -> np.ndarray:
        img = rng.integers(self._black, self._black + 30,
                           size=(self.height, self.width), dtype=np.int32)
        return np.clip(img, 0, self._MAX_ADU)

    def _add_spot(self, img, rng, cx, cy, sigma, peak):
        H, W = img.shape
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys)
        spot = peak * np.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * sigma**2))
        noise = rng.normal(0, 8, spot.shape)
        img += (spot + noise).astype(np.int32)
        np.clip(img, 0, self._MAX_ADU, out=img)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.disconnect()
