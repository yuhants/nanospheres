"""Entry point for the RadiaCode multi-device GUI.

Usage::

    python -m radiacode_gui.main
    # or
    python radiacode_gui/main.py
"""

import logging
import sys

from PyQt5.QtWidgets import QApplication, QMessageBox


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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    app = QApplication(sys.argv)
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
    win = MainWindow(mgr)
    win.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
