# Parametric-cooling PLL (Red Pitaya)

Our own host-side software for the custom dual-PLL Red Pitaya FPGA design used to
**parametrically cool a levitated nanosphere**. It is a from-scratch Python
reimplementation of the ETH GUI (github.com/tefelixhu/redpitayapll, and the C++
fork github.com/vhock/Phase-locked-Loop), against the *same* FPGA bitstream — so
no Vivado/FPGA work is needed to use it.

## Files

| File | Purpose |
|------|---------|
| `pll_control.py` | `RedPitayaPLL` driver — SSH/`monitor` register access, all unit conversions, live signals, reheating bursts, save/load. Headless-usable. |
| `pll_gui.py`     | PyQt5 + pyqtgraph GUI: all parameters + live lock monitor + loop-design Bode + reheating bursts. `--mock` runs with no hardware. |
| `rp_reheat.py`   | **Runs on the Red Pitaya**, not the lab PC. Uploaded automatically; does the on-board off/on gating loop for reheating measurements. |
| `pll_fpga/`      | Vendored FPGA design (bitstream + VHDL/Tcl + register doc). Lets the GUI load the FPGA, and keeps the option of *modifying* it open (see its README). |

This is independent of `pyrpl_control.py` / `redpitaya_control.py`. The PLL
bitstream **replaces** the stock/PyRPL FPGA image, so don't run them at the same
time on the same board.

## Install

```
pip install paramiko PyQt5 pyqtgraph numpy
```

## Run the GUI

```
python parametric_cooling/pll_gui.py --host 192.168.1.9
python parametric_cooling/pll_gui.py --host 192.168.1.9 --host 192.168.1.10  # two boards
python parametric_cooling/pll_gui.py --mock        # offline demo
```

Enter host/user/pass (default `root`/`root`). Tick **Load bitstream on connect**
the first time after a board reboot (SCPs `pll_fpga/bitfile/pll_project.bit` and
`cat … > /dev/xdevcfg`). The driver reads back the current register state and
fills in all controls.

### Multiple Red Pitayas

The window holds **one tab per board**. Pass several `--host` flags to open them
at startup, or use the **➕ Add Red Pitaya** toolbar button to add one at any
time; close a board with its tab's × (it disconnects cleanly first). Each board
connects independently (its own host/user/pass and Connect button) and runs on
its **own SSH worker thread**, so one slow or stalled board never blocks the
others or the UI. The tab label shows a ● (connected) / ○ (disconnected) dot.

### What you get

- **Per-PLL controls:** `2nd_harm`, `pid_en` (engage lock), `a` (amplitude
  0–127), `phi` (phase °), `kp`, `ki` (both negative!), `f0`, `bw`, `alpha`,
  `order`, plus a one-click **cooling preset**.
- **Global:** RF output 1/2 routing, `ext_pins_p/n`.
- **Lock monitor:** rolling NCO-frequency and loop-error traces per PLL, live
  numeric readout, and an output-vector display of (a, φ) in the w_a/w_b plane.
- **Loop design:** instant Bode of the phase-detector low-pass (alpha/order) and
  the PI filter (kp/ki) so you can tune without poking hardware blindly.
- **Reheat burst:** gate the drive off for ~10s of ms and back on, repeated as
  many times as you like, for reheating measurements (below).

## Reheating measurements

The **Reheat burst** tab (or `start_reheat()` from a script) cycles the cooling
drive off and on: cool for `t_on`, go dark for `t_off`, repeat `cycles` times.
Set the two times in ms, tick which PLLs to gate, press **Start burst**.

**What is actually switched.** The gate is the output weight register
(`w_a`/`w_b`, one register per PLL). Zeroing it kills the drive but leaves the
phase detector and PI loop running on the ADC input, so the PLL **stays locked
to the particle through the dark window** and resumes with the correct phase —
which is what keeps the sphere from being lost over thousands of repeats.
(`pid_en=0` does the opposite: it freezes the loop while the output keeps
running, so it is *not* a way to turn the feedback off.)

**Why an on-board agent.** Toggling registers from the lab PC is far too
jittery for this: every access is an SSH round-trip plus a `monitor` process
fork on the Zynq — several ms typically, with tens-of-ms tails that cannot be
bounded. So the timing loop runs on the board (`rp_reheat.py`, uploaded to
`/tmp` on first use): a register write becomes an mmap store, edges are
scheduled against absolute deadlines so the period does not drift, each edge is
approached by sleeping to within `spin` (default 1.5 ms) and then busy-waiting,
and the loop asks for `SCHED_FIFO`. Measured off-window scatter is well under
0.1 ms.

**Marking the dark windows in your data.** Tick the marker box to pulse an
`ext_pins_p` bit on the E1 digital header high for the duration of every off
window (bit *i* → `DIO`*i*`_P`, 3.3 V). Record it on a spare DAQ channel and
every dark window is timestamped in the same file as the particle motion —
much better than trying to align against host clocks. (The E1 pins are
outputs-only in this bitstream, so the Red Pitaya cannot be triggered
*externally* without an FPGA rebuild — it has to be the one issuing the gate.)

**Verifying the timing.** The agent logs the achieved edge times for every
cycle. The GUI fetches the log automatically when a burst ends and plots the
realised off-window per cycle, with mean/std and a count of cycles more than
1 ms off target — cut those in analysis rather than letting them smear the
statistics.

**Safety.** The agent restores the weights on normal exit, on error and on
SIGTERM; `stop_reheat()` also rewrites them from the host afterwards, so the
drive is never left off. The burst runs under `nohup`, so it survives the GUI
closing or the network dropping — reconnect and the GUI picks the burst back up.

From a script, alongside your Picoscope acquisition:

```python
rp = RedPitayaPLL('192.168.1.9')
rp.start_reheat(t_off=20e-3, t_on=0.5, cycles=2000, plls=[0], marker_bit=0)
...                                      # take data while it cycles
print(rp.reheat_progress())              # {'running': True, 'cycles_done': 812, …}
log = rp.fetch_reheat_log('reheat_timing.log')
print(log['off_duration'].mean(), log['off_duration'].std())
rp.stop_reheat()                         # early abort; restores the drive
```

`reheat_burst(t_off, t_on, cycles)` is the blocking version (starts, waits,
returns the parsed log). Both `t_off` and `t_on` are in **seconds** in the API,
milliseconds in the GUI.

## Use the driver from a script / notebook

```python
import sys, os
sys.path.append(r'C:\Users\yuhan\nanospheres\parametric_cooling')
from pll_control import RedPitayaPLL

rp = RedPitayaPLL('192.168.1.9')        # connects + syncs registers
rp.upload_bitstream(load=True)          # only needed after a reboot
rp.apply_cooling_preset(pll=0)          # tested ETH cooling values
rp.set_param('phi', 37.0, pll=0)        # tune feedback phase
print(rp.read_monitors())               # {0: {'nco_freq_hz':…, 'err_deg':…}, …}
rp.save_params('cooling.txt')
rp.close()
```

## Tested cooling values (from the original repo)

`2nd_harm=1, pid_en=1, a≈30` (amp-dependent), `phi` tuned to system phase,
`kp=-40`, `ki=-0.128`, `f0` = mode freq (e.g. 45/130/150 kHz), `bw=5 kHz`,
`alpha=0.39` (fc≈7.6 kHz), `order=4`. These are in `COOLING_PRESET`.

## Notes on speed & robustness (vs. the old GUI)

- All SSH runs on a worker thread; the UI never blocks. Monitor polling is
  self-paced (next poll scheduled only after the previous result), so a slow
  link lowers the refresh rate instead of freezing.
- A local register mirror means one SSH write per edit (not a full re-read), and
  monitor/sync reads are batched into a single `monitor` command.
- Connect/exec use explicit timeouts + TCP keepalive, so a dead board errors out
  fast.
- `a`/`phi`/`f0`/`bw` are cached in engineering units and the register words
  recomputed from them, avoiding 8-bit-weight quantisation drift and keeping the
  input frequency fixed when the 2nd-harmonic is toggled.

## Scaling caveats to confirm on your board

The read-back **`fNCO`** (NCO frequency word) is scaled like `f0`
(`×31.25e6/2³²`). **`fNCOErr`** is scaled to degrees using the documented
phase-detector gain `2¹³/π` counts/rad. If your `doc/regs.pdf` differs, adjust
`NCO_HZ_PER_LSB` / `PHASEDET_COUNTS_PER_RAD` in `pll_control.py`.
