"""BLE transport for RadiaCode devices using bleak (Windows/macOS/Linux)."""

import asyncio
import struct
import logging
from typing import Optional, Callable, List, Tuple

from bleak import BleakClient, BleakScanner
from radiacode.bytes_buffer import BytesBuffer

log = logging.getLogger(__name__)

_SERVICE_UUID = 'e63215e5-7003-49d8-96b0-b024798fb901'
_CHUNK_SIZE = 18


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
        found[device.address] = (name, rssi)

    scanner = BleakScanner(detection_callback=_callback)
    await scanner.start()
    import asyncio as _asyncio
    await _asyncio.sleep(timeout)
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
        self._address = address
        self._loop = loop
        self._client: Optional[BleakClient] = None
        self._write_uuid: Optional[str] = None
        self._notify_uuid: Optional[str] = None
        self._recv_buf = bytearray()
        self._response_future: Optional[asyncio.Future] = None
        self._disconnect_cb: Optional[Callable] = None

    def set_disconnect_callback(self, cb: Callable) -> None:
        self._disconnect_cb = cb

    # --- internal BLE callbacks ---

    def _on_disconnect(self, _client: BleakClient) -> None:
        log.warning('BLE disconnect: %s', self._address)
        if self._response_future and not self._response_future.done():
            self._response_future.set_exception(
                ConnectionError(f'BLE disconnected from {self._address}')
            )
        if self._disconnect_cb:
            self._disconnect_cb()

    def _on_notify(self, _sender, data: bytearray) -> None:
        self._recv_buf.extend(data)
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

    async def connect(self) -> None:
        self._client = BleakClient(
            self._address, disconnected_callback=self._on_disconnect
        )
        await self._client.connect(timeout=15.0)

        service = self._client.services.get_service(_SERVICE_UUID)
        if service is None:
            raise RuntimeError(
                f'RadiaCode BLE service not found on {self._address}. '
                'Make sure the device is on and not connected to another host.'
            )

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

    async def disconnect(self) -> None:
        if self._client and self._client.is_connected:
            try:
                if self._notify_uuid:
                    await self._client.stop_notify(self._notify_uuid)
            except Exception:
                pass
            try:
                await self._client.disconnect()
            except Exception:
                pass

    async def execute_async(self, data: bytes) -> BytesBuffer:
        """Send a length-prefixed request and await the response."""
        self._recv_buf = bytearray()
        self._response_future = self._loop.create_future()

        for offset in range(0, len(data), _CHUNK_SIZE):
            chunk = data[offset:offset + _CHUNK_SIZE]
            await self._client.write_gatt_char(
                self._write_uuid, chunk, response=False
            )

        try:
            payload = await asyncio.wait_for(self._response_future, timeout=10.0)
        except asyncio.TimeoutError:
            raise TimeoutError(f'No response within 10 s from {self._address}')

        return BytesBuffer(payload)

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected
