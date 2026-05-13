"""
Set bias voltage and current limit on CAEN DT8031P channels.

Connects, enables remote mode, applies VSET/ISET to all channels,
turns them on, then prints the monitored V and I.

Edit the parameters section before each run.
"""

import serial
import time
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Parameters — edit before each run
# ──────────────────────────────────────────────────────────────────────────────

PORT  = 'COM5'
BAUD  = 115200
BOARD = 0

CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7]

VSET = 50    # V   — bias voltage applied to all channels
ISET = 100.0    # µA  — current limit applied to all channels

# Per-channel overrides: {ch: (VSET, ISET)}
# Channels not listed use the defaults above.
CHANNEL_CONFIG: dict[int, tuple[float, float]] = {
    # 2: (40.0, 15.0),
}

# ──────────────────────────────────────────────────────────────────────────────

TIMEOUT_S = 2.0


def _open(port: str, baud: int) -> serial.Serial:
    ser = serial.Serial(port, baud, timeout=TIMEOUT_S)
    time.sleep(0.1)
    ser.reset_input_buffer()
    return ser


def _send(ser: serial.Serial, msg: str) -> str:
    ser.write(msg.encode())
    resp = ser.readline().decode().strip()
    if not resp:
        raise TimeoutError(f"No response to: {msg.strip()!r}")
    return resp


def _board_cmd(ser, board, cmd_type, par, val=None):
    msg = (f'$BD:{board:02d},CMD:{cmd_type},PAR:{par},VAL:{val}\r\n' if val is not None
           else f'$BD:{board:02d},CMD:{cmd_type},PAR:{par}\r\n')
    return _send(ser, msg)


def _ch_cmd(ser, board, cmd_type, ch, par, val=None):
    msg = (f'$BD:{board:02d},CMD:{cmd_type},CH:{ch},PAR:{par},VAL:{val:.4f}\r\n' if val is not None
           else f'$BD:{board:02d},CMD:{cmd_type},CH:{ch},PAR:{par}\r\n')
    resp = _send(ser, msg)
    if 'LOC:ERR' in resp:
        raise RuntimeError(
            "Device is in local mode.\n"
            "Press the LOC/REM button on the front panel and re-run."
        )
    if 'ERR' in resp:
        raise RuntimeError(f"Device error for {msg.strip()!r}: {resp!r}")
    return resp


def _parse_val(resp: str) -> float:
    for part in resp.split(','):
        if part.startswith('VAL:'):
            return float(part[4:])
    raise ValueError(f"Cannot parse value from: {resp!r}")


def enable_remote(ser, board=0):
    resp = _board_cmd(ser, board, 'SET', 'REM')
    if 'LOC:ERR' in resp:
        raise RuntimeError(
            "Device is locked in local mode.\n"
            "Press the LOC/REM button on the front panel and re-run."
        )
    print(f"  Remote mode enabled: {resp}")


def _ch_cfg(ch: int) -> tuple[float, float]:
    override = CHANNEL_CONFIG.get(ch, ())
    defaults = (VSET, ISET)
    return (override[0] if len(override) > 0 else defaults[0],
            override[1] if len(override) > 1 else defaults[1])


def apply_hv(ser, channels: list[int], board: int = 0, turn_on: bool = True) -> None:
    for ch in channels:
        vset, iset = _ch_cfg(ch)
        _ch_cmd(ser, board, 'SET', ch, 'VSET', vset)
        _ch_cmd(ser, board, 'SET', ch, 'ISET', iset)
        if turn_on:
            _ch_cmd(ser, board, 'SET', ch, 'ON')
        print(f"  ch{ch}: VSET={vset:.2f} V   ISET={iset:.2f} uA   {'ON' if turn_on else 'set only'}")


def read_all(ser, channels: list[int], board: int = 0) -> None:
    print("\nMonitor readings:")
    for ch in channels:
        v = _parse_val(_ch_cmd(ser, board, 'MON', ch, 'VMON'))
        i = _parse_val(_ch_cmd(ser, board, 'MON', ch, 'IMON'))
        print(f"  ch{ch}:  Vmon={v:.4f} V   Imon={i:.4f} uA")


def turn_off_all(ser, channels: list[int], board: int = 0) -> None:
    for ch in channels:
        try:
            _ch_cmd(ser, board, 'SET', ch, 'VSET', 0.0)
            _ch_cmd(ser, board, 'SET', ch, 'OFF')
        except Exception:
            pass


def main() -> None:
    off_mode = '--off' in sys.argv

    print(f"Connecting to DT8031P on {PORT} at {BAUD} baud...")
    ser = _open(PORT, BAUD)
    enable_remote(ser, BOARD)
    print("Connected.\n")

    try:
        if off_mode:
            print("Turning off all channels...")
            turn_off_all(ser, CHANNELS, BOARD)
            print("Done.")
        else:
            print(f"Applying HV to channels {CHANNELS}:")
            apply_hv(ser, CHANNELS, BOARD, turn_on=True)
            print("\nChannels on. Press Ctrl+C to turn off and exit.\n")
            try:
                while True:
                    read_all(ser, CHANNELS, BOARD)
                    time.sleep(5.0)
            except KeyboardInterrupt:
                print("\nInterrupted — turning off channels...")
                turn_off_all(ser, CHANNELS, BOARD)
    finally:
        ser.close()
        print("Device disconnected.")


if __name__ == '__main__':
    main()
