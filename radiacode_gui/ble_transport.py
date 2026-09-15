"""BLE transport for RadiaCode devices using bleak (Windows/macOS/Linux)."""

import asyncio
import struct
import logging
from typing import Dict, Optional, Callable, List, Tuple

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from radiacode.bytes_buffer import BytesBuffer

log = logging.getLogger(__name__)

_SERVICE_UUID = 'e63215e5-7003-49d8-96b0-b024798fb901'
_CHUNK_SIZE = 18

# Connecting on Windows/WinRT is flaky on the first attempt (the stack often
# reports "Could not get GATT services" or a WinError right after the device
# was seen advertising). One quiet retry fixes the vast majority of cases.
_CONNECT_ATTEMPTS = 3
_CONNECT_TIMEOUT = 15.0
_RESPONSE_TIMEOUT = 10.0
# Pause between the 18-byte chunks of one request (see execute_async).
_WRITE_GAP = 0.02

# BLEDevice objects from the most recent scan, keyed by upper-case address.
# Handing bleak a BLEDevice (instead of a bare address string) lets it connect
# directly. With a plain string, BleakClient.connect() first runs its *own*
# BLE scan (BleakScanner.find_device_by_address) to look the device up — on
# Windows that hidden scan collides with any already-connected RadiaCode and
# can knock it offline (after which the unit auto-powers off).
_device_cache: Dict[str, BLEDevice] = {}


def _norm(address: str) -> str:
    return address.strip().upper()


async def scan_devices(timeout: float = 5.0) -> List[Tuple[str, str, int]]:
    """Scan for BLE devices and return all found, with RadiaCode devices first.

    Returns list of (address, name, rssi) tuples.

    Note: service-UUID filtering is unreliable on Windows WinRT (devices may
    only include their service UUID in scan-response packets, not the primary
    advertisement).  We therefore scan all BLE devices and mark RC-* ones.
    """
    found: dict = {}   # address -> (name, rssi)

    def _callback(device, adv_data):
        name = (device.name
                or getattr(adv_data, 'local_name', None)
                or device.address)
        rssi = getattr(adv_data, 'rssi', None) or -999
        addr = _norm(device.address)
        # Keep a name if we ever saw one (some adverts carry no local name).
        prev = found.get(addr)
        if prev and prev[0] != addr and name == addr:
            name = prev[0]
        found[addr] = (name, rssi)
        _device_cache[addr] = device

    scanner = BleakScanner(detection_callback=_callback)
    await scanner.start()
    try:
        await asyncio.sleep(timeout)
    finally:
        await scanner.stop()

    def _is_radiacode(name: str) -> bool:
        nl = (name or '').lower()
        return 'rc-' in nl or 'radiacode' in nl

    # Sort: RadiaCode devices first, then by RSSI (strongest signal first)
    items = list(found.items())
    items.sort(key=lambda x: (not _is_radiacode(x[1][0]), -(x[1][1] or -999)))
    return [(addr, name, rssi) for addr, (name, rssi) in items]


class BleakTransport:
    """Async BLE transport wrapping a bleak BleakClient.

    Protocol: requests are split into 18-byte chunks and written without
    response to the write characteristic. Responses arrive as BLE
    notifications, length-prefixed with a 4-byte little-endian uint32.
    """

    def __init__(self, address: str, loop: asyncio.AbstractEventLoop):
        self._address = _norm(address)
        self._loop = loop
        self._client: Optional[BleakClient] = None
        self._write_uuid: Optional[str] = None
        self._notify_uuid: Optional[str] = None
        self._recv_buf = bytearray()
        self._notify_count = 0
        self._response_future: Optional[asyncio.Future] = None
        self._disconnect_cb: Optional[Callable] = None
        self._closing = False

    def set_disconnect_callback(self, cb: Callable) -> None:
        self._disconnect_cb = cb

    # --- internal BLE callbacks (bleak invokes these on the asyncio thread) ---

    def _on_disconnect(self, client: BleakClient) -> None:
        if client is not self._client:
            # Late callback from a client we already discarded (e.g. a failed
            # connect attempt that we retried) — nothing to do.
            return
        if self._closing:
            log.info('BLE disconnected (requested): %s', self._address)
        else:
            log.warning('BLE disconnect: %s', self._address)
        fut = self._response_future
        if fut and not fut.done():
            fut.set_exception(
                ConnectionError(f'BLE disconnected from {self._address}')
            )
        if self._disconnect_cb and not self._closing:
            try:
                self._disconnect_cb()
            except Exception:
                log.exception('disconnect callback failed')

    def _on_notify(self, _sender, data: bytearray) -> None:
        self._recv_buf.extend(data)
        self._notify_count += 1
        log.debug('[%s] notify #%d: +%d bytes (buffer %d)', self._address,
                  self._notify_count, len(data), len(self._recv_buf))
        if len(self._recv_buf) < 4:
            return
        expected = struct.unpack_from('<I', self._recv_buf, 0)[0] + 4
        if len(self._recv_buf) < expected:
            return
        payload = bytes(self._recv_buf[4:expected])
        self._recv_buf = bytearray(self._recv_buf[expected:])
        if self._response_future and not self._response_future.done():
            self._response_future.set_result(payload)

    # --- public async interface ---

    async def _resolve_target(self):
        """Return what to hand BleakClient: a cached BLEDevice if we have one."""
        dev = _device_cache.get(self._address)
        if dev is not None:
            return dev
        # Not seen in a scan yet (e.g. MAC typed by hand). Look it up
        # explicitly so the error message is meaningful; on Windows bleak
        # cannot connect to an address it has never seen advertise.
        log.info('%s not in scan cache, looking it up…', self._address)
        dev = await BleakScanner.find_device_by_address(
            self._address, timeout=_CONNECT_TIMEOUT
        )
        if dev is None:
            raise ConnectionError(
                f'{self._address} is not advertising. Make sure the RadiaCode '
                'is switched on and NOT connected to the phone app (it stops '
                'advertising while the app is connected), then try again.'
            )
        _device_cache[self._address] = dev
        return dev

    async def _connect_once(self) -> None:
        target = await self._resolve_target()
        self._client = BleakClient(
            target, disconnected_callback=self._on_disconnect
        )
        self._closing = False
        await self._client.connect(timeout=_CONNECT_TIMEOUT)

        service = self._client.services.get_service(_SERVICE_UUID)
        if service is None:
            raise RuntimeError(
                f'RadiaCode BLE service not found on {self._address}. '
                'Make sure the device is on and not connected to another host.'
            )

        self._write_uuid = self._notify_uuid = None
        for char in service.characteristics:
            props = char.properties
            if 'write-without-response' in props:
                self._write_uuid = char.uuid
            if 'notify' in props:
                self._notify_uuid = char.uuid

        if not self._write_uuid or not self._notify_uuid:
            raise RuntimeError(
                f'Required BLE characteristics not found on {self._address}'
            )

        await self._client.start_notify(self._notify_uuid, self._on_notify)
        log.info('Connected to %s', self._address)

    async def connect(self) -> None:
        self._closing = False
        last_exc: Optional[BaseException] = None
        for attempt in range(1, _CONNECT_ATTEMPTS + 1):
            try:
                await self._connect_once()
                return
            except asyncio.CancelledError:
                await self._drop_client()
                raise
            except ConnectionError:
                # Device not advertising — retrying immediately won't help.
                await self._drop_client()
                raise
            except Exception as e:
                last_exc = e
                log.warning('[%s] connect attempt %d/%d failed: %s',
                            self._address, attempt, _CONNECT_ATTEMPTS, e)
                await self._drop_client()
                if attempt < _CONNECT_ATTEMPTS:
                    await asyncio.sleep(1.5)
        raise ConnectionError(
            f'Could not connect to {self._address} after {_CONNECT_ATTEMPTS} '
            f'attempts: {last_exc}'
        )

    async def _drop_client(self) -> None:
        """Best-effort teardown of a half-open client (no callbacks fired)."""
        client, self._client = self._client, None
        if client is None:
            return
        self._closing = True
        try:
            if client.is_connected:
                await client.disconnect()
        except Exception:
            pass

    async def disconnect(self) -> None:
        self._closing = True
        client = self._client
        if client is None:
            return
        if client.is_connected:
            try:
                if self._notify_uuid:
                    await client.stop_notify(self._notify_uuid)
            except Exception:
                pass
            try:
                await client.disconnect()
            except Exception as e:
                log.warning('[%s] disconnect error: %s', self._address, e)
        self._client = None

    async def execute_async(self, data: bytes) -> BytesBuffer:
        """Send a length-prefixed request and await the response."""
        if not self.is_connected:
            raise ConnectionError(f'Not connected to {self._address}')

        self._recv_buf = bytearray()
        self._notify_count = 0
        self._response_future = self._loop.create_future()

        # Writes go out without response, so nothing throttles them. Back-to-back
        # write-commands are a known way to knock a BLE link over on Windows
        # (bleak discussion #1156); pace multi-chunk requests slightly.
        for offset in range(0, len(data), _CHUNK_SIZE):
            chunk = data[offset:offset + _CHUNK_SIZE]
            await self._client.write_gatt_char(
                self._write_uuid, chunk, response=False
            )
            if offset + _CHUNK_SIZE < len(data):
                await asyncio.sleep(_WRITE_GAP)

        try:
            payload = await asyncio.wait_for(
                self._response_future, timeout=_RESPONSE_TIMEOUT
            )
        except asyncio.TimeoutError:
            # Say *what* arrived, not just that nothing completed: a silent
            # device (0 bytes) and a response cut short mid-stream (some bytes,
            # short of the length prefix) are completely different faults.
            got = len(self._recv_buf)
            want = (struct.unpack_from('<I', self._recv_buf, 0)[0] + 4
                    if got >= 4 else None)
            detail = (f'nothing received' if got == 0 else
                      f'{got} of {want} bytes in {self._notify_count} '
                      f'notifications' if want else
                      f'{got} bytes (no length prefix)')
            raise TimeoutError(
                f'No response within {_RESPONSE_TIMEOUT:.0f} s from '
                f'{self._address} — {detail}'
            )
        finally:
            self._response_future = None

        return BytesBuffer(payload)

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected
