"""On-board feedback-gating agent -- RUNS ON THE RED PITAYA, not on the lab PC.

Uploaded to the board by ``pll_control.RedPitayaPLL.upload_reheat_agent()`` and
launched over SSH; see ``reheat_burst()`` / ``start_reheat()`` there. Nothing in
this file is imported by the host software.

Why it exists
-------------
A reheating measurement needs the parametric drive switched off for a few tens
of ms and back on, repeated thousands of times, with the off-window duration
reproducible. Toggling from the host is not good enough: every register access
is an SSH round-trip plus a ``monitor`` process fork on the Zynq CPU -- several
ms typically and with tens-of-ms tails that cannot be bounded. Running the
timing loop here reduces a register write to an mmap store (sub-us) and leaves
only the local scheduler as a jitter source.

What it does per cycle
----------------------
1. wait until the cooling interval ``t_on`` has elapsed
2. zero ``w_a``/``w_b`` (low 16 bits of each gate register) -> drive off, and
   raise the marker bit on the E1 digital header
3. wait ``t_off``
4. restore the original weights, lower the marker

Zeroing the weights kills the drive but leaves the phase detector and PI loop
running on the ADC input, so the PLL stays locked to the particle through the
dark window and resumes with the correct phase. (``pid_en`` would do the
opposite: freeze the loop while the output keeps running.)

The achieved edge times are logged per cycle so bad cycles -- ones where the
kernel preempted us -- can be cut in analysis rather than silently smearing the
statistics.

Timing
------
Edges are scheduled against absolute deadlines derived from ``t0``, so the
period does not drift over a long run. Each wait sleeps until ``spin`` seconds
before its deadline and then busy-waits, trading a little CPU for edges that do
not inherit the ~1 ms granularity of ``time.sleep``. With ``--realtime`` the
process asks for SCHED_FIFO, which removes most of the remaining jitter.

Written against the stdlib only and kept 3.5-compatible (``.format`` rather than
f-strings): the Red Pitaya image may carry an old Python 3.

Usage (as issued by the host):
    python3 rp_reheat.py --cycles 2000 --t-on 0.5 --t-off 0.02 \
        --gate 0x41200008 --gate 0x41300008 \
        --marker-addr 0x41200000 --marker-mask 0x00010000 \
        --log /tmp/rp_reheat.log --realtime
"""

from __future__ import print_function

import argparse
import ctypes
import mmap
import os
import signal
import sys
import time

WEIGHT_MASK = 0xFFFF          # w_a[15:8] | w_b[7:0] -- what we zero to gate
PAGE = mmap.PAGESIZE
LOG_FLUSH_EVERY = 200         # cycles between log writes (done while cooling)


# --------------------------------------------------------------------------- #
# /dev/mem access
# --------------------------------------------------------------------------- #
class Mem(object):
    """Page-cached mmap of /dev/mem, addressed by absolute physical address."""

    def __init__(self):
        self._fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self._pages = {}

    def page(self, addr):
        """Return (mmap of the containing page, offset of addr within it)."""
        base = addr & ~(PAGE - 1)
        page = self._pages.get(base)
        if page is None:
            page = mmap.mmap(self._fd, PAGE, flags=mmap.MAP_SHARED,
                             prot=mmap.PROT_READ | mmap.PROT_WRITE,
                             offset=base)
            self._pages[base] = page
        return page, addr - base

    def cell(self, addr):
        """A 32-bit view of one register.

        ctypes rather than mmap slicing: assigning to ``.value`` is a single
        aligned 32-bit store, whereas a 4-byte slice assignment goes through
        memcpy and may reach the AXI register as separate byte writes -- which
        would briefly expose a half-updated w_a/w_b pair on every edge.
        """
        page, off = self.page(addr)
        return ctypes.c_uint32.from_buffer(page, off)

    def close(self):
        # Any cell() views must be released first; mmap.close() refuses while
        # exported pointers exist.
        for page in self._pages.values():
            try:
                page.close()
            except Exception:
                pass
        self._pages = {}
        try:
            os.close(self._fd)
        except Exception:
            pass


class Gate(object):
    """One PLL's weight register, with its on/off words precomputed.

    Resolving the register and both words up front means an edge costs one
    store per PLL inside the timed loop.
    """

    def __init__(self, mem, addr):
        self.addr = addr
        self.cell = mem.cell(addr)
        self.on_word = self.cell.value
        self.off_word = self.on_word & ~WEIGHT_MASK

    @property
    def weights_zero(self):
        return (self.on_word & WEIGHT_MASK) == 0

    def set_on(self):
        self.cell.value = self.on_word

    def set_off(self):
        self.cell.value = self.off_word

    def release(self):
        self.cell = None


class Marker(object):
    """Digital marker bit (E1 header) pulsed in step with the gate.

    Read-modify-write, so pulsing the marker does not clobber the other fields
    sharing the register (pid_en / 2nd_harm / output routing).
    """

    def __init__(self, mem, addr, mask, active_high=True):
        self.cell = mem.cell(addr)
        self.mask = mask
        self.active_high = active_high

    def _set(self, asserted):
        if asserted == self.active_high:
            self.cell.value = self.cell.value | self.mask
        else:
            self.cell.value = self.cell.value & ~self.mask

    def assert_(self):
        self._set(True)

    def deassert(self):
        self._set(False)

    def release(self):
        self.cell = None


class NullMarker(object):
    def assert_(self):
        pass

    def deassert(self):
        pass

    def release(self):
        pass


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def wait_until(deadline, spin):
    """Sleep until ``spin`` before ``deadline``, then busy-wait to the edge."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > spin:
            time.sleep(remaining - spin)


def try_realtime(priority=50):
    """Request SCHED_FIFO. Returns a status string for the log header."""
    try:
        param = os.sched_param(priority)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        return "SCHED_FIFO({0})".format(priority)
    except Exception as exc:
        return "unavailable ({0})".format(exc)


# --------------------------------------------------------------------------- #
# Main burst loop
# --------------------------------------------------------------------------- #
class Aborted(Exception):
    pass


def run(args):
    mem = Mem()
    gates = [Gate(mem, addr) for addr in args.gate]

    if args.marker_mask and args.marker_addr is not None:
        marker = Marker(mem, args.marker_addr, args.marker_mask,
                        active_high=(args.marker_active == "high"))
    else:
        marker = NullMarker()

    sched = try_realtime(args.priority) if args.realtime else "disabled"

    log = open(args.log, "w")
    log.write("# rp_reheat burst log\n")
    log.write("# wall_start_epoch={0!r}\n".format(time.time()))
    log.write("# cycles={0} t_on={1!r} t_off={2!r} settle={3!r} spin={4!r}\n"
              .format(args.cycles, args.t_on, args.t_off, args.settle,
                      args.spin))
    log.write("# scheduling={0}\n".format(sched))
    for gate in gates:
        log.write("# gate addr=0x{0:08X} on=0x{1:08X} off=0x{2:08X}\n"
                  .format(gate.addr, gate.on_word, gate.off_word))
        if gate.weights_zero:
            log.write("# WARNING gate 0x{0:08X} already has zero weights -- "
                      "this PLL's drive is off, nothing to gate\n"
                      .format(gate.addr))
    log.write("# columns: cycle,t_off_start_s,t_on_start_s,off_duration_s\n")
    log.flush()

    def on_signal(signum, frame):
        raise Aborted("signal {0}".format(signum))

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    records = []
    completed = 0
    status = "done"
    period = args.t_on + args.t_off
    # Bind to locals: attribute lookups inside the timed loop cost jitter.
    perf_counter = time.perf_counter
    spin = args.spin

    try:
        t0 = perf_counter() + args.settle
        for cycle in range(args.cycles):
            off_deadline = t0 + cycle * period + args.t_on
            on_deadline = off_deadline + args.t_off

            wait_until(off_deadline, spin)
            for gate in gates:
                gate.set_off()
            marker.assert_()
            t_off = perf_counter()

            wait_until(on_deadline, spin)
            for gate in gates:
                gate.set_on()
            marker.deassert()
            t_on = perf_counter()

            records.append((cycle, t_off - t0, t_on - t0, t_on - t_off))
            completed = cycle + 1

            # Flush during the cooling interval, where jitter is harmless.
            if len(records) >= LOG_FLUSH_EVERY:
                _write_records(log, records)
                records = []
    except Aborted as exc:
        status = "aborted: {0}".format(exc)
    except Exception as exc:                     # noqa: BLE001 -- must restore
        status = "error: {0!r}".format(exc)
    finally:
        # Whatever happened, leave the drive ON: a sphere is on the line.
        for gate in gates:
            gate.set_on()
        marker.deassert()
        _write_records(log, records)
        _write_summary(log, args, completed, status)
        log.close()
        for gate in gates:
            gate.release()
        marker.release()
        mem.close()

    if status != "done":
        print(status, file=sys.stderr)
        return 1
    return 0


def _write_records(log, records):
    for cycle, t_off, t_on, dur in records:
        log.write("{0},{1:.6f},{2:.6f},{3:.6f}\n".format(cycle, t_off, t_on,
                                                         dur))
    log.flush()


def _write_summary(log, args, completed, status):
    log.write("# completed={0}/{1}\n".format(completed, args.cycles))
    log.write("# status={0}\n".format(status))
    log.write("# wall_end_epoch={0!r}\n".format(time.time()))
    log.flush()


# --------------------------------------------------------------------------- #
def _int0(text):
    """int() accepting 0x-prefixed hex (argparse type)."""
    return int(text, 0)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gate", type=_int0, action="append", required=True,
                   metavar="ADDR",
                   help="weight-register address to gate (repeat per PLL)")
    p.add_argument("--cycles", type=int, required=True)
    p.add_argument("--t-on", type=float, required=True,
                   help="cooling (feedback on) time per cycle, seconds")
    p.add_argument("--t-off", type=float, required=True,
                   help="dark (feedback off) time per cycle, seconds")
    p.add_argument("--settle", type=float, default=1.0,
                   help="delay before the first cycle, seconds")
    p.add_argument("--spin", type=float, default=1.5e-3,
                   help="busy-wait window before each edge, seconds")
    p.add_argument("--marker-addr", type=_int0, default=None)
    p.add_argument("--marker-mask", type=_int0, default=0)
    p.add_argument("--marker-active", choices=("high", "low"), default="high",
                   help="marker level while the feedback is OFF")
    p.add_argument("--realtime", action="store_true",
                   help="request SCHED_FIFO for the timing loop")
    p.add_argument("--priority", type=int, default=50)
    p.add_argument("--log", default="/tmp/rp_reheat.log")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.t_off <= 0 or args.t_on <= 0:
        print("t_on and t_off must be positive", file=sys.stderr)
        return 2
    if args.cycles <= 0:
        print("cycles must be positive", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
