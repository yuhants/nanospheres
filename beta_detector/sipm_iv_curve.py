"""
I-V curve measurement for SiPM detectors using CAEN DT8031P.

The DT8031P uses an ASCII serial protocol over USB-CDC (VID=21E1).
No CAEN SDK or third-party packages needed beyond pyserial and numpy.

Edit the parameters section before each run.
Output: one CSV per channel in OUTPUT_DIR.
"""

import serial
import threading
import time
import sys
import numpy as np
import csv
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Parameters — edit before each run
# ──────────────────────────────────────────────────────────────────────────────

PORT   = 'COM5'     # DT8031P serial port (VID=21E1, PID=0014)
BAUD   = 115200
BOARD  = 0          # board index (0 for single-board units)

# Channels to sweep (order determines sweep order)
# CHANNELS = [0]
CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7]

# Default sweep settings — apply to all channels unless overridden below
V_START = 35.0      # V — sweep start voltage
V_STOP  = 50.0      # V — sweep end voltage (inclusive)
V_STEP  = 0.5       # V — voltage step size
WAIT_S  = 10.0      # s — stabilisation wait at each step
COMPLIANCE_uA = 20.0  # µA — abort channel sweep if |I| exceeds this

# Per-channel overrides: {ch: (V_START, V_STOP, V_STEP, WAIT_S, COMPLIANCE_uA)}
# Any channel not listed here uses the defaults above.
# Omit trailing fields to keep their defaults, e.g. (35.0, 43.0) uses global V_STEP etc.
CHANNEL_CONFIG: dict[int, tuple] = {
    # 0: (35.0, 45.0, 1.0, 10.0, 20.0),
    # 1: (33.0, 43.0, 1.0, 10.0, 20.0),
}

OUTPUT_DIR = r'D:\beta_detector\iv_curves\all_33ohm_laseroff'
RUN_LABEL  = 'sipm'   # prefix for output filenames (e.g. 'sipm_deviceA')

# ──────────────────────────────────────────────────────────────────────────────

TIMEOUT_S = 2.0     # serial read timeout

# Single lock protecting the shared serial port across all channel threads
_serial_lock = threading.Lock()


def _open(port: str, baud: int) -> serial.Serial:
    ser = serial.Serial(port, baud, timeout=TIMEOUT_S)
    time.sleep(0.1)
    ser.reset_input_buffer()
    return ser


def _ch_cfg(ch: int) -> tuple[float, float, float, float, float]:
    """Return (V_START, V_STOP, V_STEP, WAIT_S, COMPLIANCE_uA) for a channel."""
    override = CHANNEL_CONFIG.get(ch, ())
    defaults = (V_START, V_STOP, V_STEP, WAIT_S, COMPLIANCE_uA)
    return tuple(override[i] if i < len(override) else defaults[i] for i in range(5))


def _send(ser: serial.Serial, msg: str) -> str:
    with _serial_lock:
        ser.write(msg.encode())
        resp = ser.readline().decode().strip()
    if not resp:
        raise TimeoutError(f"No response to: {msg.strip()!r}")
    return resp


def _board_cmd(ser: serial.Serial, board: int, cmd_type: str, par: str, val: str | None = None) -> str:
    """Board-level command (no CH field): $BD:bb,CMD:type,PAR:name[,VAL:v]"""
    if val is not None:
        msg = f'$BD:{board:02d},CMD:{cmd_type},PAR:{par},VAL:{val}\r\n'
    else:
        msg = f'$BD:{board:02d},CMD:{cmd_type},PAR:{par}\r\n'
    return _send(ser, msg)


def enable_remote(ser: serial.Serial, board: int = 0) -> None:
    """
    Switch the board from local (front-panel) to remote (serial) control.
    If this raises, toggle the LOC/REM button on the device front panel first.
    """
    resp = _board_cmd(ser, board, 'SET', 'REM')
    if 'LOC:ERR' in resp:
        raise RuntimeError(
            "Device is locked in local mode — all SET commands will fail.\n"
            "Press the LOC/REM (or LOCAL/REMOTE) button on the DT8031P front panel\n"
            "to switch to remote mode, then re-run the script."
        )
    print(f"  Remote mode enabled: {resp}")


def _cmd(ser: serial.Serial, board: int, cmd_type: str, ch: int, par: str, val: float | None = None) -> str:
    """
    Send one CAEN ASCII channel command and return the response string.
    Command format:  $BD:bb,CMD:type,CH:c,PAR:name[,VAL:v]\r\n
    Response format: #CMD:OK,VAL:value\r\n  or  #CMD:ERR\r\n
    """
    if val is not None:
        msg = f'$BD:{board:02d},CMD:{cmd_type},CH:{ch},PAR:{par},VAL:{val:.4f}\r\n'
    else:
        msg = f'$BD:{board:02d},CMD:{cmd_type},CH:{ch},PAR:{par}\r\n'

    resp = _send(ser, msg)

    if 'LOC:ERR' in resp:
        raise RuntimeError(
            f"Device is in local mode — call enable_remote() after connecting.\n"
            f"Command: {msg.strip()!r}  Response: {resp!r}"
        )
    if 'ERR' in resp:
        raise RuntimeError(f"Device error for command {msg.strip()!r}: {resp!r}")
    return resp


def _parse_val(resp: str) -> float:
    """Extract the numeric value from a '#CMD:OK,VAL:x.xxxx' response."""
    for part in resp.split(','):
        if part.startswith('VAL:'):
            return float(part[4:])
    raise ValueError(f"Cannot parse value from response: {resp!r}")


def channel_on(ser: serial.Serial, board: int, ch: int) -> None:
    _cmd(ser, board, 'SET', ch, 'ON')


def channel_off(ser: serial.Serial, board: int, ch: int) -> None:
    _cmd(ser, board, 'SET', ch, 'OFF')


def set_voltage(ser: serial.Serial, board: int, ch: int, v: float) -> None:
    _cmd(ser, board, 'SET', ch, 'VSET', v)


def read_vmon(ser: serial.Serial, board: int, ch: int) -> float:
    return _parse_val(_cmd(ser, board, 'MON', ch, 'VMON'))


def read_imon(ser: serial.Serial, board: int, ch: int) -> float:
    return _parse_val(_cmd(ser, board, 'MON', ch, 'IMON'))


def sweep_iv(
    ser: serial.Serial,
    ch: int,
    v_start: float,
    v_stop: float,
    v_step: float,
    wait_s: float,
    compliance_ua: float,
    board: int = 0,
) -> tuple[list[float], list[float]]:
    voltages = np.arange(v_start, v_stop + v_step / 2, v_step)
    v_out, i_out = [], []

    WARMUP_S = 60
    print(f"  ch{ch}: {v_start:.1f} → {v_stop:.1f} V ({len(voltages)} steps)")
    channel_on(ser, board, ch)
    set_voltage(ser, board, ch, float(voltages[0]))
    print(f"  Warming up at {voltages[0]:.1f} V for {WARMUP_S} s...")
    time.sleep(WARMUP_S)

    try:
        for v_set in voltages:
            set_voltage(ser, board, ch, float(v_set))
            time.sleep(wait_s)
            v_mon = read_vmon(ser, board, ch)
            i_mon = read_imon(ser, board, ch)
            v_out.append(v_mon)
            i_out.append(i_mon)
            print(f"    Vset={v_set:6.2f} V  Vmon={v_mon:7.3f} V  I={i_mon:8.3f} uA")

            if abs(i_mon) > compliance_ua:
                print(f"  !! Compliance ({i_mon:.1f} uA > {compliance_ua:.0f} uA) -- stopping ch{ch}")
                break
    finally:
        set_voltage(ser, board, ch, 0.0)
        time.sleep(0.3)
        channel_off(ser, board, ch)

    return v_out, i_out


def save_csv(path: Path, v_data: list[float], i_data: list[float]) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['voltage_V', 'current_uA'])
        for v, i in zip(v_data, i_data):
            writer.writerow([f'{v:.4f}', f'{i:.4f}'])


def plot_iv(results: dict[int, tuple[list[float], list[float]]], save_path: Path | None = None) -> None:
    """
    Plot I-V curves for all channels.
    results: {ch: (v_data, i_data)}
    If save_path is given, saves a PNG alongside the CSV files.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for ch, (v_data, i_data) in results.items():
        ax.plot(v_data, i_data, marker='.', markersize=4, label=f'ch{ch}')

    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current (uA)')
    ax.set_title(f'SiPM I-V — {RUN_LABEL}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  → {save_path}")

    plt.show()


def probe(port: str = PORT, baud: int = BAUD, board: int = BOARD) -> None:
    """
    Read VMON and IMON on all channels listed in CHANNELS and print them.
    Useful for confirming the connection before a full sweep.
    """
    ser = _open(port, baud)
    enable_remote(ser, board)
    print(f"Connected to {port} at {baud} baud.\n")
    for ch in CHANNELS:
        try:
            v = read_vmon(ser, board, ch)
            i = read_imon(ser, board, ch)
            print(f"  ch{ch}:  Vmon={v:.4f} V   Imon={i:.4f} uA")
        except Exception as e:
            print(f"  ch{ch}: ERROR — {e}")
    ser.close()


def main() -> None:
    if '--probe' in sys.argv:
        probe()
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to DT8031P on {PORT} at {BAUD} baud...")
    ser = _open(PORT, BAUD)
    enable_remote(ser, BOARD)
    print("Connected.\n")

    results = {}

    def _run_channel(ch: int) -> tuple[int, list[float], list[float]]:
        vs, ve, vstep, wait, comp = _ch_cfg(ch)
        print(f"ch{ch}: starting ({vs:.1f}→{ve:.1f} V, step={vstep:.2f} V, wait={wait:.1f} s)")
        v_data, i_data = sweep_iv(ser, ch, vs, ve, vstep, wait, comp, BOARD)
        fname = out_dir / f'{RUN_LABEL}_{timestamp}_ch{ch}.csv'
        save_csv(fname, v_data, i_data)
        print(f"ch{ch}: saved → {fname}")
        return ch, v_data, i_data

    try:
        with ThreadPoolExecutor(max_workers=len(CHANNELS)) as pool:
            futures = {pool.submit(_run_channel, ch): ch for ch in CHANNELS}
            for fut in as_completed(futures):
                ch = futures[fut]
                try:
                    ch, v_data, i_data = fut.result()
                    results[ch] = (v_data, i_data)
                except Exception as exc:
                    print(f"ch{ch}: ERROR — {exc}")
    finally:
        for ch in CHANNELS:
            try:
                set_voltage(ser, BOARD, ch, 0.0)
                channel_off(ser, BOARD, ch)
            except Exception:
                pass
        ser.close()
        print("Device disconnected.")

    if results:
        plot_path = out_dir / f'{RUN_LABEL}_{timestamp}_iv.png'
        plot_iv(results, save_path=plot_path)


if __name__ == '__main__':
    main()
