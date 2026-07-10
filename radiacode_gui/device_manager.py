"""Manages multiple RadiaCode BLE connections from a single asyncio event loop.

The asyncio loop runs in a daemon thread so it can coexist with PyQt5's
event loop. Qt signals carry data from the async thread to the GUI thread
safely via QueuedConnection (Qt's default for cross-thread signals).
"""

import asyncio
import datetime
import logging
import threading
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal

from radiacode.types import RareData, RealTimeData, Spectrum

from .ble_transport import BleakTransport, scan_devices
from .data_logger import LogConfig, create_logger
from .device import RadiaCodeBLE

log = logging.getLogger(__name__)

# Seconds between spectrum polls (data_buf is polled every 1 s)
_SPECTRUM_INTERVAL = 5

# The device reports dose_rate as a raw float in R/h (roentgen/hour); the GUI
# displays µSv/h. With 1 R ≈ 0.01 Sv: µSv/h = raw[R/h] * 1e6 * 0.01 = raw * 1e4.
_DOSE_RATE_TO_USV_H = 1e4

# BLE resilience: a single dropped notification or read timeout is transient and
# must not tear down the whole session. Tolerate this many consecutive poll
# errors (while the link is still up) before giving up.
_MAX_CONSECUTIVE_ERRORS = 5


class DeviceManager(QObject):
    """Thread-safe manager for multiple RadiaCode BLE devices.

    Methods that touch BLE (start_scan, connect_device, disconnect_device)
    submit coroutines to the asyncio loop and return immediately — results
    arrive via the signals below.
    """

    # Scanning
    scan_started    = pyqtSignal()
    scan_finished   = pyqtSignal(list)   # [(address, name, rssi), ...]
    scan_error      = pyqtSignal(str)

    # Connection lifecycle
    device_connected      = pyqtSignal(str, str, str, str)  # addr, name, serial, firmware
    device_disconnected   = pyqtSignal(str)                 # addr
    device_connect_failed = pyqtSignal(str, str)            # addr, error message

    # Live data (emitted from asyncio thread, delivered to Qt thread)
    realtime_data  = pyqtSignal(str, object)   # addr, RealTimeData
    spectrum_data  = pyqtSignal(str, object)   # addr, Spectrum
    rare_data      = pyqtSignal(str, object)   # addr, RareData

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        # ProactorEventLoop is required on Windows for WinRT BLE (bleak)
        self._loop = asyncio.ProactorEventLoop()
        self._thread = threading.Thread(
            target=self._run_loop, name='RadiaCodeBLE', daemon=True
        )
        self._thread.start()

        self._devices:     Dict[str, RadiaCodeBLE]        = {}
        self._poll_tasks:  Dict[str, asyncio.Task]        = {}
        self._loggers:     Dict[str, object]              = {}
        self._output_dir:  Optional[str]                  = None
        self._logging_on:  bool                           = False
        self._log_config:  LogConfig                      = LogConfig()

        # Serialises all BLE adapter access. On Windows, running a scan while a
        # device is connected (or while the poll loop is mid-read) drops the
        # active link — after which the RadiaCode auto-powers off. Holding this
        # lock around scanning, connecting, and each poll read guarantees the
        # scanner is never active at the same time as a live connection.
        self._ble_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    #  Public API (called from Qt thread)                                 #
    # ------------------------------------------------------------------ #

    def set_output_dir(self, path: str) -> None:
        self._output_dir = path

    def set_log_config(self, config: LogConfig) -> None:
        """Update what/how to record. Reopens loggers if already recording.

        No-op (keeps the current file open) if the config is unchanged, so
        e.g. dismissing the settings dialog without edits doesn't split the
        active log file — which would otherwise reset the wait for the next
        slow-arriving RareData (temperature/battery) sample.
        """
        if config == self._log_config:
            return
        self._log_config = config.copy()
        if self._logging_on:
            # Apply the new config by cycling loggers off then on.
            self._submit(self._apply_logging(False))
            self._submit(self._apply_logging(True))

    def get_log_config(self) -> LogConfig:
        return self._log_config.copy()

    def set_logging_enabled(self, enabled: bool) -> None:
        self._logging_on = enabled
        # Open/close loggers for already-connected devices on the asyncio thread.
        self._submit(self._apply_logging(enabled))

    async def _apply_logging(self, enabled: bool) -> None:
        if enabled:
            for addr, device in list(self._devices.items()):
                if not self._loggers.get(addr):
                    self._loggers[addr] = self._open_logger(addr, device)
        else:
            for addr in list(self._loggers.keys()):
                lgr = self._loggers.get(addr)
                if lgr:
                    lgr.close()
                self._loggers[addr] = None

    def start_scan(self, timeout: float = 5.0) -> None:
        self.scan_started.emit()
        self._submit(self._do_scan(timeout))

    def connect_device(self, address: str, name: str = '') -> None:
        self._submit(self._do_connect(address, name))

    def disconnect_device(self, address: str) -> None:
        self._submit(self._do_disconnect(address))

    def request_dose_reset(self, address: str) -> None:
        self._submit(self._safe_call(address, lambda d: d.dose_reset()))

    def request_spectrum_reset(self, address: str) -> None:
        self._submit(self._safe_call(address, lambda d: d.spectrum_reset()))

    def connected_addresses(self) -> List[str]:
        return list(self._devices.keys())

    def stop(self) -> None:
        async def _shutdown():
            for addr in list(self._devices.keys()):
                await self._do_disconnect(addr)
            self._loop.stop()

        self._submit(_shutdown())
        self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------ #
    #  Internal asyncio helpers                                            #
    # ------------------------------------------------------------------ #

    def _run_loop(self) -> None:
        # Initialize COM as MTA via ctypes *before* any WinRT/bleak call.
        # We use ctypes rather than pythoncom because bleak 0.22 errors if it
        # sees pythoncom imported in an STA thread — ctypes init avoids that.
        import ctypes
        _COINIT_MULTITHREADED = 0x0
        ctypes.windll.ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)

        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            ctypes.windll.ole32.CoUninitialize()

    def _submit(self, coro) -> asyncio.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _safe_call(self, address: str, fn) -> None:
        device = self._devices.get(address)
        if device:
            try:
                await fn(device)
            except Exception as e:
                log.error('[%s] command error: %s', address, e)

    # ------------------------------------------------------------------ #
    #  Scan                                                                #
    # ------------------------------------------------------------------ #

    async def _do_scan(self, timeout: float) -> None:
        try:
            # Exclusive adapter access: pauses all poll-loop reads and blocks
            # connects for the duration of the scan, so scanning can't knock a
            # connected device off the air.
            async with self._ble_lock:
                results = await scan_devices(timeout)
            self.scan_finished.emit(results)
        except Exception as e:
            log.error('Scan error: %s', e)
            self.scan_error.emit(str(e))

    # ------------------------------------------------------------------ #
    #  Connect / disconnect                                                #
    # ------------------------------------------------------------------ #

    async def _do_connect(self, address: str, name: str) -> None:
        if address in self._devices:
            log.warning('[%s] already connected', address)
            return

        transport = BleakTransport(address, self._loop)
        try:
            # Don't connect while a scan is in flight — wait for the scanner to
            # fully stop first (see _ble_lock).
            async with self._ble_lock:
                await transport.connect()
                device = RadiaCodeBLE(transport, address)
                await device.initialize()
        except Exception as e:
            log.error('[%s] connect failed: %s', address, e)
            self.device_connect_failed.emit(address, str(e))
            try:
                await transport.disconnect()
            except Exception:
                pass
            return

        device.name = name or f'RC [{address[-8:]}]'
        self._devices[address] = device
        self._loggers[address] = self._open_logger(address, device)

        transport.set_disconnect_callback(lambda: self._on_ble_disconnect(address))

        task = self._loop.create_task(
            self._poll_loop(address), name=f'poll-{address}'
        )
        self._poll_tasks[address] = task

        self.device_connected.emit(
            address, device.name, device.serial, device.firmware
        )

    async def _do_disconnect(self, address: str) -> None:
        task = self._poll_tasks.get(address)
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        device = self._devices.get(address)
        if device:
            try:
                await device._transport.disconnect()
            except Exception:
                pass

        # Cleanup is done in _poll_loop's finally block; emit signal here
        # only if the task was already done (e.g. already disconnected).
        if address not in self._devices:
            self.device_disconnected.emit(address)

    def _on_ble_disconnect(self, address: str) -> None:
        """Called from bleak's disconnect callback (asyncio thread).

        End the session cleanly by cancelling the poll task. We do NOT try to
        auto-reconnect: on Windows, a BLE scan re-initialises the radio and
        drops active links, so churning connect/disconnect here would just
        repeatedly disconnect the device — which makes the RadiaCode auto-power
        off. The user reconnects explicitly instead.
        """
        log.warning('[%s] BLE link dropped', address)
        task = self._poll_tasks.get(address)
        if task and not task.done():
            task.cancel()

    # ------------------------------------------------------------------ #
    #  Polling loop (runs as asyncio Task per device)                      #
    # ------------------------------------------------------------------ #

    async def _poll_loop(self, address: str) -> None:
        device = self._devices[address]
        tick = 0
        consecutive_errors = 0

        # Battery / accumulated dose only arrive in the device's infrequent
        # RareData packet. We remember the latest values and carry them forward
        # so every 1 Hz status sample has a battery reading, instead of the
        # sparse, mostly-empty status log we had before.
        last_charge   = float('nan')
        last_dose     = float('nan')
        last_duration = 0

        try:
            while True:
                # Re-read the logger each pass so toggling Record / changing the
                # log config mid-session takes effect immediately.
                logger = self._loggers.get(address)

                try:
                    # Hold the adapter for this read cycle so a scan can't run
                    # concurrently and drop the link. If a scan is in progress,
                    # this simply waits until it finishes.
                    async with self._ble_lock:
                        # --- data_buf (1 Hz): count/dose rate + battery snapshots ---
                        items = await device.data_buf()
                        for item in items:
                            if isinstance(item, RealTimeData):
                                # Convert raw R/h → µSv/h once, so every downstream
                                # consumer (panel, sidebar, time series, logger) is
                                # consistent.
                                item.dose_rate *= _DOSE_RATE_TO_USV_H
                                self.realtime_data.emit(address, item)
                                if logger:
                                    logger.log_realtime(item)
                            elif isinstance(item, RareData):
                                # Refresh the values we carry forward. Don't log the
                                # raw packet here — the dense status sample below
                                # captures it at a consistent cadence.
                                last_charge   = item.charge_level
                                last_dose     = item.dose
                                last_duration = item.duration

                        # --- live temperature (register read, 1 Hz) ---
                        # Poll the TEMP_degC register directly rather than waiting
                        # for it in the rare buffer packet, so temperature is logged
                        # at the same rate as the count rate.
                        try:
                            temperature = await device.read_temperature()
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            log.debug('[%s] temperature read failed: %s', address, e)
                            temperature = float('nan')

                        # --- spectrum (every _SPECTRUM_INTERVAL s) ---
                        tick += 1
                        spec = None
                        if tick % _SPECTRUM_INTERVAL == 0:
                            try:
                                spec = await device.spectrum()
                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                log.error('[%s] spectrum error: %s', address, e)

                    # --- dense status sample (temperature + battery, 1 Hz) ---
                    status = RareData(
                        dt=datetime.datetime.now(),
                        duration=last_duration,
                        dose=last_dose,
                        temperature=temperature,
                        charge_level=last_charge,
                        flags=0,
                    )
                    self.rare_data.emit(address, status)
                    if logger:
                        logger.log_status(status)

                    if spec is not None:
                        self.spectrum_data.emit(address, spec)
                        if logger:
                            logger.log_spectrum(spec)

                    consecutive_errors = 0

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    consecutive_errors += 1
                    log.warning('[%s] poll error (%d/%d): %s',
                                address, consecutive_errors,
                                _MAX_CONSECUTIVE_ERRORS, e)
                    # Tolerate transient blips, but give up once the link is
                    # gone or errors persist. Ending the loop lets the user
                    # reconnect explicitly (no churn that powers the device off).
                    if (not device._transport.is_connected
                            or consecutive_errors >= _MAX_CONSECUTIVE_ERRORS):
                        log.error('[%s] ending poll loop', address)
                        raise

                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error('[%s] poll loop ended: %s', address, e)
        finally:
            self._poll_tasks.pop(address, None)
            self._devices.pop(address, None)
            lgr = self._loggers.pop(address, None)
            if lgr:
                lgr.close()
            self.device_disconnected.emit(address)
            log.info('[%s] poll loop exited', address)

    # ------------------------------------------------------------------ #
    #  Logger helpers                                                      #
    # ------------------------------------------------------------------ #

    def _open_logger(self, address: str, device: RadiaCodeBLE):
        if not self._logging_on or not self._output_dir:
            return None
        try:
            return create_logger(
                self._log_config, self._output_dir,
                address, device.serial, device.firmware,
            )
        except Exception as e:
            log.error('[%s] logger init failed: %s', address, e)
            return None
