"""Standalone BLE diagnostic for the RadiaCode link drop / auto power-off.

The GUI connects fine, initialises, then the link dies ~10 s later and the
device powers itself off. This script reproduces the session *without* Qt, one
poll command at a time, so we can tell which request the device (or the Windows
BLE stack) chokes on.

Run, in this order, until one of them drops:

    python radiacode_gui/ble_diag.py --scan                 # find the MAC
    python radiacode_gui/ble_diag.py <MAC> --mode idle      # no polling at all
    python radiacode_gui/ble_diag.py <MAC> --mode buf       # data_buf only, 1 Hz
    python radiacode_gui/ble_diag.py <MAC> --mode full      # + spectrum every 5 s
    python radiacode_gui/ble_diag.py <MAC> --mode buf+temp  # + TEMP_degC register

WARNING: ``buf+temp`` is the known-bad probe. RD_VIRT_SFR_BATCH(TEMP_degC) hangs
RC-102 firmware 4.14 over BLE — the device stops answering and powers itself off
about 9 s later. It is kept so the fault can be re-confirmed on other firmware.

Each line is timestamped with seconds since connect, so the drop can be lined
up with the request that preceded it.  Device Events (POWER_OFF / POWER_ON …)
found in the data buffer are printed in full: if the unit is powering itself
off, it usually says so there before the link goes.
"""

import argparse
import asyncio
import logging
import sys
import time

sys.path.insert(0, __file__.rsplit('radiacode_gui', 1)[0])

from radiacode.types import Event, RareData, RawData, RealTimeData, DoseRateDB  # noqa: E402

from radiacode_gui.ble_transport import BleakTransport, scan_devices  # noqa: E402
from radiacode_gui.device import RadiaCodeBLE  # noqa: E402

log = logging.getLogger('diag')
_t0 = time.monotonic()


class _Elapsed(logging.Formatter):
    def format(self, record):
        record.elapsed = time.monotonic() - _t0
        return super().format(record)


def _setup_logging(verbose: bool) -> None:
    h = logging.StreamHandler()
    h.setFormatter(_Elapsed('%(elapsed)7.2fs  %(levelname)-7s %(name)s: %(message)s'))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(h)
    logging.getLogger('bleak').setLevel(logging.DEBUG if verbose else logging.WARNING)


async def _scan(timeout: float) -> None:
    log.info('scanning %.0f s …', timeout)
    for addr, name, rssi in await scan_devices(timeout):
        log.info('  %-20s  %-24s  rssi=%s', addr, name, rssi)


async def _run(address: str, mode: str, duration: float,
               spectrum_interval: int, period: float) -> None:
    loop = asyncio.get_running_loop()
    transport = BleakTransport(address, loop)

    dropped = loop.create_future()

    def _on_drop():
        if not dropped.done():
            dropped.set_result(True)

    log.info('connecting to %s …', address)
    await transport.connect()
    transport.set_disconnect_callback(_on_drop)

    device = RadiaCodeBLE(transport, address)
    await device.initialize()
    log.info('initialised: serial=%s fw=%s  (mode=%s)',
             device.serial, device.firmware, mode)

    async def _poll():
        tick = 0
        while True:
            tick += 1
            if mode != 'idle':
                t = time.monotonic()
                items = await device.data_buf()
                log.info('#%03d data_buf ok (%.0f ms, %d items)',
                         tick, 1e3 * (time.monotonic() - t), len(items))
                for item in items:
                    if isinstance(item, Event):
                        log.warning('#%03d DEVICE EVENT: %s', tick, item)
                    elif isinstance(item, RareData):
                        log.info('#%03d rare: batt=%.0f%% temp=%.1fC dose=%.4g '
                                 'flags=0x%x', tick, item.charge_level,
                                 item.temperature, item.dose, item.flags)
                    elif isinstance(item, RealTimeData):
                        log.debug('#%03d rt: cps=%.1f flags=0x%x rt_flags=0x%x',
                                  tick, item.count_rate, item.flags,
                                  item.real_time_flags)
                    elif isinstance(item, (RawData, DoseRateDB)):
                        log.debug('#%03d %s', tick, type(item).__name__)

            if mode == 'buf+temp':
                t = time.monotonic()
                temp = await device.read_temperature()
                log.info('#%03d temperature ok (%.0f ms): %.1f C',
                         tick, 1e3 * (time.monotonic() - t), temp)

            if mode == 'full' and tick % spectrum_interval == 0:
                t = time.monotonic()
                spec = await device.spectrum()
                log.info('#%03d spectrum ok (%.0f ms, %d channels, %s)',
                         tick, 1e3 * (time.monotonic() - t),
                         len(spec.counts), spec.duration)

            await asyncio.sleep(period)

    poll = loop.create_task(_poll())
    try:
        done, _ = await asyncio.wait(
            {poll, dropped}, timeout=duration,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if dropped in done:
            log.error('LINK DROPPED after %.1f s (mode=%s)',
                      time.monotonic() - _t0, mode)
        elif poll in done:
            log.error('poll loop stopped: %r', poll.exception())
        else:
            log.info('SURVIVED %.0f s in mode=%s — this mode is safe', duration, mode)
    finally:
        poll.cancel()
        try:
            await transport.disconnect()
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('address', nargs='?', help='device MAC (from --scan)')
    p.add_argument('--scan', action='store_true', help='scan and exit')
    p.add_argument('--mode', default='full',
                   choices=['idle', 'buf', 'buf+temp', 'full'])
    p.add_argument('--duration', type=float, default=120.0,
                   help='seconds to stay connected (default 120)')
    p.add_argument('--period', type=float, default=1.0,
                   help='seconds between poll cycles (default 1)')
    p.add_argument('--spectrum-interval', type=int, default=5,
                   help='poll cycles between spectrum reads (default 5)')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='DEBUG logging, including bleak/WinRT internals')
    args = p.parse_args()

    _setup_logging(args.verbose)

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    if args.scan or not args.address:
        asyncio.run(_scan(6.0))
        return
    asyncio.run(_run(args.address, args.mode, args.duration,
                     args.spectrum_interval, args.period))


if __name__ == '__main__':
    main()
