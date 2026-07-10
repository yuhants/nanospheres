"""Driver for the Red Pitaya dual-PLL parametric-cooling FPGA design.

This talks to the *custom* dual-PLL bitstream vendored in ``pll_fpga/`` (see
``pll_fpga/README.md`` for provenance). Unlike PyRPL / the SCPI server, this
design implements two independent phase-locked loops entirely in the FPGA:

    input -> phase detector (CORDIC + exp. smoothing LPF) -> PI loop filter
          -> NCO (tracks the input frequency) -> optional /2 -> output stage
             (variable amplitude & phase, out = a*cos(phi)*sin + a*sin(phi)*cos)

Each PLL tracks the mechanical mode of a levitated nanoparticle and emits a
phase-locked drive (typically at the *second harmonic*) to parametrically cool
it. The host only pokes FPGA registers over SSH using the board's ``monitor``
utility; all real-time control runs on the FPGA at 31.25 MHz.

This is an independent Python reimplementation of the original ETH host software
(``server/pll.py`` in github.com/tefelixhu/redpitayapll), rewritten for:

* **Speed** -- a local mirror of every register means a parameter change writes
  exactly one register (one SSH round-trip) instead of re-reading all 14;
  monitor/sync reads are *batched* into a single SSH command. All commands run
  through ONE persistent remote shell opened at connect time, so each register
  access is a single round-trip on an existing channel instead of a fresh
  ``exec_command`` (which opens a new SSH channel and forks a shell on the
  board -- hundreds of ms on the Zynq CPU).
* **Robustness** -- explicit connect/exec timeouts and TCP keepalive, so a slow
  or dropped link raises instead of hanging forever (the old GUI's freeze).
* **Accuracy** -- desired ``a``/``phi``/``f0``/``bw`` are cached in engineering
  units and the register words recomputed from them, so repeated edits don't
  accumulate the rounding error of the 8-bit weight quantisation, and toggling
  the 2nd-harmonic rescales the frequency exactly.

The class is *not* internally locked; drive it from a single thread (the GUI
runs all hardware access on one worker thread -- see ``pll_gui.py``).

Register map (per PLL, base 0x41200000 / 0x41300000; offsets below). The
authoritative source is ``pll_fpga/doc/regs.pdf``.

    offset 0x00000  bits  output_1[2:0] output_2[5:3] pid_en[6] 2nd_harm[7]
                          ext_pins_p[23:16] ext_pins_n[31:24]
                          (outputs & ext_pins are GLOBAL -> only on PLL0's reg)
    offset 0x00008        w_a[15:8] w_b[7:0]   (signed 8-bit output weights)
    offset 0x10000        kp[31:0]   (signed)
    offset 0x10008        ki[31:0]   (signed)
    offset 0x20000        f0[31:0]   (frequency setpoint word)
    offset 0x20008        bw[31:0]   (frequency drift limit word)
    offset 0x30000        alpha[26:10] order[2:0]   (phase-detector LPF)
    offset 0x40000        fNCO[31:0]    (read-only: current NCO frequency word)
    offset 0x50000        fNCOErr[31:0] (read-only: PI/phase-detector error)
"""

import os
import socket
import time

import numpy as np

try:
    import paramiko
except ImportError:  # allow --mock GUI / offline import without paramiko
    paramiko = None


# --------------------------------------------------------------------------- #
# Hardware / firmware constants
# --------------------------------------------------------------------------- #
CLK_HZ = 31.25e6                       # PLL subsystem clock
NCO_HZ_PER_LSB = CLK_HZ / 2 ** 32      # = 0.0072760 Hz; NCO/f0 word -> Hz
FILTER_CLK_HZ = 122.07e3               # exp-smoothing filter reference rate
ALPHA_FC_PER_UNIT = FILTER_CLK_HZ / (2 * np.pi)   # fc = alpha * this  (Hz)

# Phase-detector output scaling: 2^13/pi counts per radian (see repo README).
PHASEDET_COUNTS_PER_RAD = 2 ** 13 / np.pi

# Documented loop gains (per README "Working principle"), for the Bode preview:
#   P = kp * 0.3311 Hz/deg ,  I = ki * 40422 Hz^2/deg
P_HZ_PER_DEG_PER_KP = 0.3311
I_HZ2_PER_DEG_PER_KI = 40422.0

MONITOR = "/opt/redpitaya/bin/monitor"
PLL_BASE_ADDR = [0x41200000, 0x41300000]
GLOB_BASE_ADDR = 0x41200000

REMOTE_BITFILE = "/tmp/pll_project.bit"
_LOCAL_BITFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "pll_fpga", "bitfile", "pll_project.bit")

# Register-field map: name -> (offset, msb, lsb, scope, signed)
#   scope 'pll'    -> lives at each PLL's base
#   scope 'global' -> lives only at GLOB_BASE_ADDR (== PLL0 base, offset 0)
PARAM_MAP = {
    "output_1":   (0x00000,  2,  0, "global", False),
    "output_2":   (0x00000,  5,  3, "global", False),
    "pid_en":     (0x00000,  6,  6, "pll",    False),
    "2nd_harm":   (0x00000,  7,  7, "pll",    False),
    "ext_pins_p": (0x00000, 23, 16, "global", False),
    "ext_pins_n": (0x00000, 31, 24, "global", False),
    "w_a":        (0x00008, 15,  8, "pll",    True),
    "w_b":        (0x00008,  7,  0, "pll",    True),
    "kp":         (0x10000, 31,  0, "pll",    True),
    "ki":         (0x10008, 31,  0, "pll",    True),
    "f0":         (0x20000, 31,  0, "pll",    False),
    "bw":         (0x20008, 31,  0, "pll",    False),
    "alpha":      (0x30000, 26, 10, "pll",    False),
    "order":      (0x30000,  2,  0, "pll",    False),
}

# Read-only monitor signals (offsets, per PLL).
SIGNAL_OFFSETS = {"fNCO": 0x40000, "fNCOErr": 0x50000}

# Distinct register offsets that hold settable parameters (for batched sync).
_PARAM_OFFSETS = sorted({v[0] for v in PARAM_MAP.values()})

OUTPUT_OPTIONS = {
    "PLL1": 0b000, "PLL2": 0b001, "PLL1 + PLL2": 0b010, "PLL1 + IN2": 0b011,
    "IN1": 0b100, "IN2": 0b101, "LI1_X": 0b110, "LI2_Y": 0b111,
}
OUTPUT_OPTIONS_INV = {v: k for k, v in OUTPUT_OPTIONS.items()}

# Parameters the GUI exposes per PLL (engineering units; a/phi are virtual,
# derived from the w_a/w_b weights).
PLL_PARAM_KEYS = ["2nd_harm", "pid_en", "a", "phi", "kp", "ki",
                  "f0", "bw", "alpha", "order"]
GLOBAL_PARAM_KEYS = ["output_1", "output_2", "ext_pins_p", "ext_pins_n"]

# Tested parametric-cooling preset (per repo README). a depends on your amp;
# phi must be tuned to the total system phase.
COOLING_PRESET = {"2nd_harm": 1, "pid_en": 1, "a": 30.0, "phi": 0.0,
                  "kp": -40.0, "ki": -0.128, "f0": 130e3, "bw": 5e3,
                  "alpha": 0.39, "order": 4}


# --------------------------------------------------------------------------- #
# Bit / unit helpers
# --------------------------------------------------------------------------- #
def _width(msb, lsb):
    return msb - lsb + 1


def _get_field(regval, msb, lsb):
    return (regval >> lsb) & ((1 << _width(msb, lsb)) - 1)


def _set_field(regval, msb, lsb, fieldval):
    mask = ((1 << _width(msb, lsb)) - 1) << lsb
    return (regval & ~mask) | ((int(fieldval) << lsb) & mask)


def _to_signed(uval, nbits):
    if uval >= 1 << (nbits - 1):
        return uval - (1 << nbits)
    return uval


def _to_unsigned(val, nbits):
    """Clamp to the signed range of nbits, return two's-complement unsigned."""
    val = int(round(val))
    lo, hi = -(1 << (nbits - 1)), (1 << (nbits - 1)) - 1
    val = max(lo, min(hi, val))
    return val & ((1 << nbits) - 1)


def _clamp_unsigned(val, nbits):
    val = int(round(val))
    return max(0, min((1 << nbits) - 1, val))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
class RedPitayaPLL:
    """Host driver for the dual-PLL parametric-cooling bitstream."""

    def __init__(self, hostname, username="root", password="root", n_pll=2,
                 connect_timeout=8.0, exec_timeout=5.0, keepalive=15,
                 connect=True):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.n_pll = n_pll
        self.exec_timeout = exec_timeout
        self._connect_timeout = connect_timeout
        self._keepalive = keepalive
        self.client = None

        # Persistent remote shell (opened on connect): every command is one
        # round-trip on this channel instead of a fresh exec_command.
        self._sh_in = None
        self._sh_out = None
        self._cmd_seq = 0

        # Local mirror of the full 32-bit registers: {addr: value}.
        self._reg = {}
        # Cached desired engineering values (avoid quantisation drift / readback
        # round-trips). Per PLL.
        self._cache = [dict(a=0.0, phi=0.0, f0=0.0, bw=0.0)
                       for _ in range(n_pll)]

        if connect:
            self.connect()

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    def connect(self):
        if paramiko is None:
            raise RuntimeError("paramiko is not installed (pip install paramiko)")
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(self.hostname, username=self.username,
                            password=self.password,
                            timeout=self._connect_timeout,
                            banner_timeout=self._connect_timeout,
                            auth_timeout=self._connect_timeout)
        # Keepalive so a dead link is detected instead of blocking forever.
        tr = self.client.get_transport()
        if tr is not None:
            tr.set_keepalive(self._keepalive)
        self._open_shell()
        self.sync_all()
        print(f"Connected to Red Pitaya PLL at {self.hostname} "
              f"({self.n_pll} PLL(s))")

    def close(self):
        self._close_shell()
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None
                print("Red Pitaya PLL connection closed.")

    @property
    def connected(self):
        if self.client is None:
            return False
        tr = self.client.get_transport()
        return tr is not None and tr.is_active()

    # ------------------------------------------------------------------ #
    # Bitstream loading
    # ------------------------------------------------------------------ #
    def upload_bitstream(self, local_path=None, remote_path=REMOTE_BITFILE,
                         load=True):
        """SCP the PLL bitstream to the board and (optionally) load the FPGA.

        Loading replaces the stock/PyRPL FPGA image, so the SCPI server and
        ``pyrpl_control.py`` must not be used while this design is loaded.
        """
        local_path = local_path or _LOCAL_BITFILE
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Bitstream not found: {local_path}")
        sftp = self.client.open_sftp()
        try:
            sftp.put(local_path, remote_path)
        finally:
            sftp.close()
        print(f"Uploaded {os.path.basename(local_path)} -> {remote_path}")
        if load:
            self._exec(f"cat {remote_path} > /dev/xdevcfg")
            time.sleep(0.5)
            self.sync_all()
            print("FPGA loaded with PLL bitstream.")

    # ------------------------------------------------------------------ #
    # Low-level SSH / monitor
    # ------------------------------------------------------------------ #
    class _RemoteCommandError(RuntimeError):
        """Remote command exited non-zero (shell itself is fine)."""

    def _open_shell(self):
        """Start the long-lived remote shell all commands are piped through."""
        self._sh_in, self._sh_out, _ = self.client.exec_command(
            "/bin/sh -s 2>&1")
        self._sh_out.channel.settimeout(self.exec_timeout)

    def _close_shell(self):
        if self._sh_in is not None:
            try:
                self._sh_in.channel.close()
            except Exception:
                pass
        self._sh_in = self._sh_out = None

    def _shell_exec(self, command):
        """Run one command on the persistent shell; return its output."""
        self._cmd_seq += 1
        marker = f"__RC_{self._cmd_seq}__"
        self._sh_in.write(f"{command} ; echo {marker} $?\n")
        self._sh_in.flush()
        lines = []
        while True:
            line = self._sh_out.readline()
            if not line:
                raise RuntimeError("persistent shell closed")
            line = line.strip()
            if line.startswith(marker):
                status = int(line.split()[-1])
                break
            if line:
                lines.append(line)
        out = "\n".join(lines)
        if status != 0:
            raise self._RemoteCommandError(
                f"Remote error for '{command}' (exit {status}): {out}")
        return out

    def _exec(self, command):
        if self.client is None:
            raise RuntimeError("Not connected")
        for attempt in (0, 1):
            try:
                if self._sh_in is None or self._sh_in.channel.closed:
                    self._open_shell()
                return self._shell_exec(command)
            except self._RemoteCommandError:
                raise            # command failed; shell is still in sync
            except socket.timeout:
                # Shell is now out of sync (marker may arrive later) -> drop
                # it, but don't retry: a dead board would just double the wait.
                self._close_shell()
                raise RuntimeError(
                    f"Timeout ({self.exec_timeout}s) for '{command}'")
            except Exception:
                # Channel died (idle drop, board reboot, ...): reopen once.
                self._close_shell()
                if attempt:
                    raise

    @staticmethod
    def _parse_hex(token):
        token = token.strip()
        return int(token, 16) if token else None

    def _read_addrs(self, addrs):
        """Read several registers in ONE ssh round-trip. Returns {addr: int}."""
        addrs = list(addrs)
        if not addrs:
            return {}
        cmd = " ; ".join(f"{MONITOR} 0x{a:08X}" for a in addrs)
        out = self._exec(cmd)
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if len(lines) != len(addrs):
            raise RuntimeError(
                f"monitor returned {len(lines)} values for {len(addrs)} reads")
        return {a: self._parse_hex(ln) for a, ln in zip(addrs, lines)}

    def _write_addr(self, addr, value):
        self._exec(f"{MONITOR} 0x{addr:08X} 0x{value & 0xFFFFFFFF:08X}")

    def _addr(self, name, pll):
        offset, _, _, scope, _ = PARAM_MAP[name]
        base = GLOB_BASE_ADDR if scope == "global" else PLL_BASE_ADDR[pll]
        return base + offset

    # ------------------------------------------------------------------ #
    # Register sync (mirror)
    # ------------------------------------------------------------------ #
    def _all_param_addrs(self):
        addrs = {GLOB_BASE_ADDR + off for off in _PARAM_OFFSETS}
        for pll in range(self.n_pll):
            for off in _PARAM_OFFSETS:
                addrs.add(PLL_BASE_ADDR[pll] + off)
        return sorted(addrs)

    def sync_all(self):
        """Refresh the full register mirror from the board (one round-trip)."""
        self._reg = self._read_addrs(self._all_param_addrs())
        # Seed engineering caches from hardware so the GUI shows real values.
        for pll in range(self.n_pll):
            self._cache[pll]["f0"] = self.get_param("f0", pll)
            self._cache[pll]["bw"] = self.get_param("bw", pll)
            self._cache[pll]["a"] = self.get_param("a", pll)
            self._cache[pll]["phi"] = self.get_param("phi", pll)

    # ------------------------------------------------------------------ #
    # Raw field access (via mirror; read-modify-write)
    # ------------------------------------------------------------------ #
    def _get_raw(self, name, pll):
        offset, msb, lsb, _, signed = PARAM_MAP[name]
        addr = self._addr(name, pll)
        regval = self._reg.get(addr, 0)
        field = _get_field(regval, msb, lsb)
        return _to_signed(field, _width(msb, lsb)) if signed else field

    def _set_raw(self, name, pll, raw_value):
        offset, msb, lsb, _, signed = PARAM_MAP[name]
        addr = self._addr(name, pll)
        regval = self._reg.get(addr, 0)
        if signed:
            raw_value = _to_unsigned(raw_value, _width(msb, lsb))
        regval = _set_field(regval, msb, lsb, raw_value)
        self._reg[addr] = regval
        self._write_addr(addr, regval)

    def _set_weights(self, pll, a, phi_deg):
        """Set w_a and w_b (which share one register) in a single write."""
        phi = np.deg2rad(phi_deg)
        w_a = _to_unsigned(a * np.sin(phi), _width(*PARAM_MAP["w_a"][1:3]))
        w_b = _to_unsigned(a * np.cos(phi), _width(*PARAM_MAP["w_b"][1:3]))
        addr = self._addr("w_a", pll)
        regval = self._reg.get(addr, 0)
        regval = _set_field(regval, PARAM_MAP["w_a"][1], PARAM_MAP["w_a"][2], w_a)
        regval = _set_field(regval, PARAM_MAP["w_b"][1], PARAM_MAP["w_b"][2], w_b)
        self._reg[addr] = regval
        self._write_addr(addr, regval)

    # ------------------------------------------------------------------ #
    # Engineering-unit get/set
    # ------------------------------------------------------------------ #
    def get_param(self, name, pll=0):
        """Read a parameter in engineering units from the register mirror."""
        if name in ("output_1", "output_2"):
            return OUTPUT_OPTIONS_INV.get(self._get_raw(name, 0), "PLL1")
        if name in ("ext_pins_p", "ext_pins_n"):
            return int(self._get_raw(name, 0))
        if name in ("2nd_harm", "pid_en"):
            return int(self._get_raw(name, pll))
        if name == "a":
            w_a, w_b = self._get_raw("w_a", pll), self._get_raw("w_b", pll)
            return float(np.hypot(w_a, w_b))
        if name == "phi":
            w_a, w_b = self._get_raw("w_a", pll), self._get_raw("w_b", pll)
            return float(np.rad2deg(np.arctan2(w_a, w_b)))
        if name in ("kp", "ki"):
            return self._get_raw(name, pll) / 2 ** 16
        if name in ("f0", "bw"):
            word = self._get_raw(name, pll)
            harm = int(self._get_raw("2nd_harm", pll))
            return word * NCO_HZ_PER_LSB / (1 + harm)
        if name == "alpha":
            return self._get_raw(name, pll) / 2 ** 17
        if name == "order":
            return int(self._get_raw(name, pll)) + 1
        raise KeyError(name)

    def set_param(self, name, value, pll=0):
        """Set a parameter in engineering units (writes one register)."""
        if name in ("output_1", "output_2"):
            self._set_raw(name, 0, OUTPUT_OPTIONS[value])
            return
        if name in ("ext_pins_p", "ext_pins_n"):
            self._set_raw(name, 0, _clamp_unsigned(value, 8))
            return
        if name == "pid_en":
            self._set_raw(name, pll, 1 if int(value) else 0)
            return
        if name == "2nd_harm":
            harm = 1 if int(value) else 0
            self._set_raw(name, pll, harm)
            # Rescale f0/bw words so the *input* frequency stays put.
            self.set_param("f0", self._cache[pll]["f0"], pll)
            self.set_param("bw", self._cache[pll]["bw"], pll)
            return
        if name in ("a", "phi"):
            if name == "a":
                self._cache[pll]["a"] = float(value)
            else:
                self._cache[pll]["phi"] = float(value)
            self._set_weights(pll, self._cache[pll]["a"],
                              self._cache[pll]["phi"])
            return
        if name in ("kp", "ki"):
            self._set_raw(name, pll, int(round(float(value) * 2 ** 16)))
            return
        if name in ("f0", "bw"):
            self._cache[pll][name] = float(value)
            harm = int(self._get_raw("2nd_harm", pll))
            word = int(round(float(value) * (1 + harm) / NCO_HZ_PER_LSB))
            self._set_raw(name, pll, _clamp_unsigned(word, 32))
            return
        if name == "alpha":
            val = min(float(value), 0.999999)
            self._set_raw(name, pll, _clamp_unsigned(val * 2 ** 17, 17))
            return
        if name == "order":
            order = int(max(1, min(8, int(value))))
            self._set_raw(name, pll, order - 1)
            return
        raise KeyError(name)

    # ------------------------------------------------------------------ #
    # Live monitor signals
    # ------------------------------------------------------------------ #
    def read_monitors(self, plls=None):
        """Read the read-only loop signals for the given PLLs in one round-trip.

        Returns ``{pll: {'nco_freq_hz', 'nco_word', 'err_counts', 'err_deg'}}``.

        * ``nco_freq_hz`` -- current NCO output frequency (the frequency the
          loop has tracked to). Scaled like ``f0``; this is the carrier the
          output stage emits (input frequency = this / (2nd_harm+1)).
        * ``err_*`` -- phase-detector / PI error. ``err_deg`` uses the
          documented 2^13/pi counts-per-rad scaling; verify against your board.
        """
        plls = range(self.n_pll) if plls is None else plls
        addrs = []
        for pll in plls:
            for off in SIGNAL_OFFSETS.values():
                addrs.append(PLL_BASE_ADDR[pll] + off)
        vals = self._read_addrs(addrs)
        out = {}
        for pll in plls:
            nco = vals[PLL_BASE_ADDR[pll] + SIGNAL_OFFSETS["fNCO"]]
            err = vals[PLL_BASE_ADDR[pll] + SIGNAL_OFFSETS["fNCOErr"]]
            nco_s = _to_signed(nco, 32)
            err_s = _to_signed(err, 32)
            out[pll] = {
                "nco_word": nco_s,
                "nco_freq_hz": nco_s * NCO_HZ_PER_LSB,
                "err_counts": err_s,
                "err_deg": err_s / PHASEDET_COUNTS_PER_RAD * 180.0 / np.pi,
            }
        return out

    # ------------------------------------------------------------------ #
    # Designed transfer functions (for the GUI's Bode previews; hardware-free)
    # ------------------------------------------------------------------ #
    @staticmethod
    def phase_detector_lpf(alpha, order, frequencies):
        """Exp-smoothing low-pass response: 1/(1 + i f/fc)^order, fc=alpha*fref."""
        f = np.asarray(frequencies, dtype=float)
        fc = max(alpha * ALPHA_FC_PER_UNIT, 1e-9)
        return 1.0 / (1.0 + 1j * f / fc) ** int(order)

    @staticmethod
    def pi_loop_tf(kp, ki, frequencies):
        """PI controller response in Hz/deg: P + I/(i 2pi f)."""
        f = np.asarray(frequencies, dtype=float)
        P = kp * P_HZ_PER_DEG_PER_KP
        Ig = ki * I_HZ2_PER_DEG_PER_KI
        with np.errstate(divide="ignore"):
            return P + Ig / (1j * 2 * np.pi * f)

    # ------------------------------------------------------------------ #
    # Save / load parameter sets
    # ------------------------------------------------------------------ #
    def all_params(self):
        """Snapshot all engineering-unit parameters as a nested dict."""
        snap = {"global": {k: self.get_param(k) for k in GLOBAL_PARAM_KEYS},
                "pll": []}
        for pll in range(self.n_pll):
            snap["pll"].append({k: self.get_param(k, pll)
                                for k in PLL_PARAM_KEYS})
        return snap

    def save_params(self, filename):
        snap = self.all_params()
        with open(filename, "w") as f:
            for k, v in snap["global"].items():
                f.write(f"global:{k}:{v}\n")
            for pll, pdict in enumerate(snap["pll"]):
                for k, v in pdict.items():
                    f.write(f"{pll}:{k}:{v}\n")
        print(f"Saved PLL parameters to {filename}")

    def load_params(self, filename):
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                scope, key, value = line.split(":", 2)
                if scope == "global":
                    self.set_param(key, value if key.startswith("output")
                                   else float(value), 0)
                else:
                    pll = int(scope)
                    if key in ("output_1", "output_2"):
                        self.set_param(key, value, pll)
                    else:
                        self.set_param(key, float(value), pll)
        print(f"Loaded PLL parameters from {filename}")

    def apply_cooling_preset(self, pll=0, preset=None):
        """Apply the README's tested parametric-cooling values to one PLL."""
        preset = preset or COOLING_PRESET
        # order matters: set 2nd_harm before f0/bw so scaling is right.
        for key in ("alpha", "order", "kp", "ki", "2nd_harm",
                    "f0", "bw", "a", "phi", "pid_en"):
            self.set_param(key, preset[key], pll)
        print(f"Applied cooling preset to PLL{pll}")


if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.9"
    rp = RedPitayaPLL(host)
    try:
        for pll in range(rp.n_pll):
            print(f"PLL{pll}: " + ", ".join(
                f"{k}={rp.get_param(k, pll)}" for k in PLL_PARAM_KEYS))
        print("monitors:", rp.read_monitors())
    finally:
        rp.close()
