"""RadiaCode device protocol layer — uses BleakTransport instead of bluepy."""

import asyncio
import datetime
import struct
import logging
from typing import List, Optional, Union

from radiacode.bytes_buffer import BytesBuffer
from radiacode.decoders.databuf import decode_VS_DATA_BUF
from radiacode.decoders.spectrum import decode_RC_VS_SPECTRUM
from radiacode.types import (
    COMMAND, VS, VSFR,
    DoseRateDB, Event, RareData, RawData, RealTimeData, Spectrum,
)

from .ble_transport import BleakTransport

log = logging.getLogger(__name__)

DataBufItem = Union[RealTimeData, DoseRateDB, RareData, RawData, Event]


class RadiaCodeBLE:
    """Async interface to a single RadiaCode device over BLE.

    Call ``await device.initialize()`` immediately after the transport
    has been connected.
    """

    def __init__(self, transport: BleakTransport, address: str):
        self._transport = transport
        self.address = address
        self._seq = 0
        self._base_time: Optional[datetime.datetime] = None
        self._spectrum_format_version = 0

        # Populated by initialize()
        self.serial: str = ''
        self.firmware: str = ''
        self.name: str = ''

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Run the same startup sequence as the upstream RadiaCode.__init__."""
        await self._execute(COMMAND.SET_EXCHANGE, b'\x01\xff\x12\xff')

        dt = datetime.datetime.now()
        time_bytes = struct.pack(
            '<BBBBBBBB',
            dt.day, dt.month, dt.year - 2000, 0,
            dt.second, dt.minute, dt.hour, 0,
        )
        await self._execute(COMMAND.SET_TIME, time_bytes)
        await self._write_reg(VSFR.DEVICE_TIME, struct.pack('<I', 0))
        self._base_time = datetime.datetime.now() + datetime.timedelta(seconds=128)

        # Firmware version
        r = await self._execute(COMMAND.GET_VERSION)
        _bmin, _bmaj = r.unpack('<HH')
        r.unpack_string()                    # boot date (discard)
        tmin, tmaj = r.unpack('<HH')
        r.unpack_string()                    # target date (discard)
        self.firmware = f'{tmaj}.{tmin}'

        # Serial number
        r = await self._read_vs(VS.SERIAL_NUMBER)
        self.serial = r.data().decode('ascii').strip()

        # Spectrum format version (needed for decoding)
        r = await self._read_vs(VS.CONFIGURATION)
        config = r.data().decode('cp1251')
        for line in config.split('\n'):
            if line.startswith('SpecFormatVersion'):
                try:
                    self._spectrum_format_version = int(line.split('=')[1].strip())
                except ValueError:
                    pass
                break

        log.info('[%s] initialized — serial=%s fw=%s fmt=%d',
                 self.address, self.serial, self.firmware,
                 self._spectrum_format_version)

    # ------------------------------------------------------------------ #
    #  Low-level protocol helpers                                          #
    # ------------------------------------------------------------------ #

    async def _execute(
        self,
        reqtype: COMMAND,
        args: Optional[bytes] = None,
    ) -> BytesBuffer:
        req_seq_no = 0x80 + self._seq
        self._seq = (self._seq + 1) % 32
        req_header = struct.pack('<HBB', int(reqtype), 0, req_seq_no)
        request = req_header + (args or b'')
        full_request = struct.pack('<I', len(request)) + request

        response = await self._transport.execute_async(full_request)

        resp_header = response.unpack('<4s')[0]
        if resp_header != req_header:
            raise RuntimeError(
                f'[{self.address}] Header mismatch: '
                f'got {resp_header.hex()}, expected {req_header.hex()}'
            )
        return response

    async def _read_vs(self, command_id: Union[VS, int]) -> BytesBuffer:
        r = await self._execute(COMMAND.RD_VIRT_STRING, struct.pack('<I', int(command_id)))
        retcode, flen = r.unpack('<II')
        if retcode != 1:
            raise RuntimeError(f'[{self.address}] {command_id}: retcode={retcode}')
        # Upstream firmware quirk: occasional extra null byte
        if r.size() == flen + 1 and r._data[-1] == 0x00:
            r._data = r._data[:-1]
        if r.size() != flen:
            raise RuntimeError(
                f'[{self.address}] {command_id}: size {r.size()} != {flen}'
            )
        return r

    async def _write_reg(
        self,
        command_id: Union[VSFR, int],
        data: Optional[bytes] = None,
    ) -> None:
        r = await self._execute(
            COMMAND.WR_VIRT_SFR,
            struct.pack('<I', int(command_id)) + (data or b''),
        )
        retcode = r.unpack('<I')[0]
        if retcode != 1:
            raise RuntimeError(f'[{self.address}] write_reg retcode={retcode}')
        assert r.size() == 0

    # ------------------------------------------------------------------ #
    #  Public data-acquisition methods                                     #
    # ------------------------------------------------------------------ #

    async def data_buf(self) -> List[DataBufItem]:
        """Poll the device's data buffer (call ~1 Hz)."""
        r = await self._read_vs(VS.DATA_BUF)
        return decode_VS_DATA_BUF(r, self._base_time)

    async def spectrum(self) -> Spectrum:
        """Retrieve the current energy spectrum."""
        r = await self._read_vs(VS.SPECTRUM)
        return decode_RC_VS_SPECTRUM(r, self._spectrum_format_version)

    async def dose_reset(self) -> None:
        await self._write_reg(VSFR.DOSE_RESET)

    async def spectrum_reset(self) -> None:
        r = await self._execute(
            COMMAND.WR_VIRT_STRING,
            struct.pack('<II', int(VS.SPECTRUM), 0),
        )
        retcode = r.unpack('<I')[0]
        if retcode != 1:
            raise RuntimeError(f'[{self.address}] spectrum_reset retcode={retcode}')
        assert r.size() == 0

    async def energy_calib(self) -> List[float]:
        r = await self._read_vs(VS.ENERGY_CALIB)
        return list(r.unpack('<fff'))
