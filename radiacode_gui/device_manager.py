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

from radiacode.types import Event, RareData, RealTimeData, Spectrum

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
        self._connecting:  set                            = set()
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

        # Polling profile. Tunable from the command line (see main.py) so a
        # link that the device cannot sustain can be thinned out without an
        # edit: the spectrum read is a multi-kilobyte notification burst and
        # is by far the heaviest thing we ask of the BLE link.
        self.poll_period       = 1.0
        self.spectrum_interval = _SPECTRUM_INTERVAL
        # OFF by default, and it should stay off: RD_VIRT_SFR_BATCH(TEMP_degC)
        # wedges RC-102 firmware 4.14 over BLE. The device never answers it,
        # stops responding entirely, and drops the link ~9 s later — after which
        # it powers itself off. Temperature comes from the RareData packet
        # instead (slower, but it is what the device volunteers).
        self.poll_temperature  = False
        # Pause between the individual requests of one poll cycle. Gives the
        # device's BLE stack a moment between bursts; 0 restores the old
        # back-to-back behaviour.
        self.request_gap       = 0.05

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
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(self._do_disconnect(a) for a in list(self._devices)),
                        return_exceptions=True,
                    ),
                    timeout=6.0,
                )
            except Exception as e:
                log.warning('shutdown: %s', e)
            finally:
                self._loop.stop()

        if not self._loop.is_running():
            return
        self._submit(_shutdown())
        self._thread.join(timeout=8.0)
        if self._thread.is_alive():
            log.warning('BLE thread did not stop in time')

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
        self._loop.set_exception_handler(self._loop_exception_handler)
        try:
            self._loop.run_forever()
        finally:
            ctypes.windll.ole32.CoUninitialize()

    @staticmethod
    def _loop_exception_handler(loop, context) -> None:
        # Never let a stray exception in the BLE thread go unlogged; the GUI
        # thread keeps running regardless.
        exc = context.get('exception')
        log.error('asyncio error: %s', context.get('message'),
                  exc_info=(type(exc), exc, exc.__traceback__) if exc else None)

    def _submit(self, coro) -> asyncio.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _safe_call(self, address: str, fn) -> None:
        device = self._devices.get(address.strip().upper())
        if device:
            try:
                # Serialise with the poll loop so a command never interleaves
                # with an in-flight read (responses would be mismatched).
                async with self._ble_lock:
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
        address = address.strip().upper()
        if address in self._devices:
            log.warning('[%s] already connected', address)
            return
        if address in self._connecting:
            log.warning('[%s] connect already in progress', address)
            return
        self._connecting.add(address)

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
        finally:
            self._connecting.discard(address)

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
        address = address.strip().upper()
        # Grab the device *before* cancelling the poll task: the task's
        # finally-block removes it from self._devices, so looking it up
        # afterwards found nothing and the BLE link was never actually closed
        # (the device stayed connected, and Reconnect then failed).
        device = self._devices.get(address)
        task = self._poll_tasks.get(address)

        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            # The poll loop's finally block has now cleaned up and emitted
            # device_disconnected.

        if device is not None:
            try:
                async with self._ble_lock:
                    await device._transport.disconnect()
            except Exception as e:
                log.warning('[%s] disconnect error: %s', address, e)

        # Only reached if no poll task did the cleanup (should not happen).
        if self._devices.pop(address, None) is not None:
            self._poll_tasks.pop(address, None)
            lgr = self._loggers.pop(address, None)
            if lgr:
                lgr.close()
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
        last_temp     = float('nan')
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
                            elif isinstance(item, Event):
                                # The device records its own power events. If
                                # the unit is switching itself off, it says so
                                # here — the single most useful clue when the
                                # link dies moments later.
                                log.warning('[%s] device event: %s', address, item)
                            elif isinstance(item, RareData):
                                # Refresh the values we carry forward. Don't log the
                                # raw packet here — the dense status sample below
                                # captures it at a consistent cadence.
                                last_charge   = item.charge_level
                                last_dose     = item.dose
                                last_temp     = item.temperature
                                last_duration = item.duration

                        # --- live temperature (register read, 1 Hz) ---
                        # Opt-in only. Reading TEMP_degC directly would log
                        # temperature at the count-rate cadence instead of the
                        # RareData cadence, but it kills RC-102 fw 4.14 over BLE
                        # (see poll_temperature).
                        # Carried forward from the last RareData packet unless
                        # the (device-wedging) register read is enabled.
                        temperature = last_temp
                        try:
                            if self.poll_temperature:
                                await asyncio.sleep(self.request_gap)
                                temperature = await device.read_temperature()
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            log.debug('[%s] temperature read failed: %s', address, e)

                        # --- spectrum (every _SPECTRUM_INTERVAL s) ---
                        tick += 1
                        spec = None
                        if (self.spectrum_interval > 0
                                and tick % self.spectrum_interval == 0):
                            try:
                                await asyncio.sleep(self.request_gap)
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

                await asyncio.sleep(self.poll_period)

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
