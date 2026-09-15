"""Entry point for the RadiaCode multi-device GUI.

Usage::

    python -m radiacode_gui.main
    # or
    python radiacode_gui/main.py
"""

import argparse
import logging
import logging.handlers
import os
import sys
import traceback

# Allow ``python radiacode_gui/main.py`` (no package context) as well as
# ``python -m radiacode_gui.main``.
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 'radiacode_gui'   # noqa: A001

from PyQt5.QtWidgets import QApplication, QMessageBox

log = logging.getLogger('radiacode_gui')

_LOG_FILE = os.path.join(os.getcwd(), 'radiacode_gui.log')


def _check_deps():
    missing = []
    for pkg, import_name in [
        ('bleak',     'bleak'),
        ('radiacode', 'radiacode'),
        ('h5py',      'h5py'),
        ('numpy',     'numpy'),
        ('matplotlib','matplotlib'),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    return missing


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog='radiacode_gui', description='RadiaCode multi-device monitor')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='DEBUG logging (includes bleak/WinRT BLE internals)')
    p.add_argument('--poll-period', type=float, default=1.0, metavar='S',
                   help='seconds between poll cycles (default 1.0)')
    p.add_argument('--spectrum-interval', type=int, default=None, metavar='N',
                   help='poll cycles between spectrum reads; 0 disables the '
                        'spectrum poll (it is the heaviest BLE transfer, so '
                        'raising or disabling it is the first thing to try if '
                        'the link keeps dropping)')
    p.add_argument('--temperature-register', action='store_true',
                   help='poll the TEMP_degC register every cycle instead of '
                        'taking temperature from the RareData packet. KNOWN TO '
                        'HANG RC-102 firmware 4.14 over BLE (the device stops '
                        'responding and powers off) — diagnostic use only')
    return p.parse_args(argv)


def _setup_logging(verbose=False):
    fmt = logging.Formatter(
        '%(asctime)s  %(levelname)-8s  %(name)s: %(message)s', datefmt='%H:%M:%S'
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    try:
        fh = logging.handlers.RotatingFileHandler(
            _LOG_FILE, maxBytes=2_000_000, backupCount=2, encoding='utf-8'
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError as e:
        log.warning('Could not open log file %s: %s', _LOG_FILE, e)
    # bleak is chatty at DEBUG; keep INFO unless we are chasing a link drop
    logging.getLogger('bleak').setLevel(logging.DEBUG if verbose else logging.INFO)


def _install_excepthook():
    """Keep the GUI alive when a Python exception escapes a Qt slot.

    PyQt5 >= 5.5 calls qFatal() (process abort, Windows exception code
    0xc0000409 in Qt5Core.dll) whenever an unhandled Python exception
    propagates out of a slot or reimplemented virtual. That turns any small
    bug in a plot update or a signal handler into "the window just closed".
    Installing our own hook logs the traceback instead and lets Qt continue.
    """
    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        msg = ''.join(traceback.format_exception(exc_type, exc, tb))
        log.error('Unhandled exception in Qt callback (GUI kept running):\n%s', msg)

    sys.excepthook = hook


def main():
    args = _parse_args()
    _setup_logging(args.verbose)
    _install_excepthook()
    log.info('Starting RadiaCode Monitor (log file: %s)', _LOG_FILE)

    # Qt should not see our own CLI flags.
    app = QApplication(sys.argv[:1])
    app.setApplicationName('RadiaCode Monitor')
    app.setStyle('Fusion')

    # Dependency check before importing anything BLE-related
    missing = _check_deps()
    if missing:
        QMessageBox.critical(
            None,
            'Missing Dependencies',
            'Please install the following packages and try again:\n\n'
            + '\n'.join(f'  pip install {p}' for p in missing),
        )
        sys.exit(1)

    from .device_manager import DeviceManager
    from .gui.main_window import MainWindow

    mgr = DeviceManager()
    mgr.poll_period = max(0.2, args.poll_period)
    mgr.poll_temperature = args.temperature_register
    if args.spectrum_interval is not None:
        mgr.spectrum_interval = args.spectrum_interval
    log.info('poll profile: period=%.1fs spectrum_interval=%s temperature=%s',
             mgr.poll_period, mgr.spectrum_interval or 'off', mgr.poll_temperature)
    win = MainWindow(mgr)
    win.show()

    rc = app.exec_()
    log.info('GUI closed (exit code %d)', rc)
    sys.exit(rc)


if __name__ == '__main__':
    main()
