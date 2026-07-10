# Parametric-cooling PLL (Red Pitaya)

Our own host-side software for the custom dual-PLL Red Pitaya FPGA design used to
**parametrically cool a levitated nanosphere**. It is a from-scratch Python
reimplementation of the ETH GUI (github.com/tefelixhu/redpitayapll, and the C++
fork github.com/vhock/Phase-locked-Loop), against the *same* FPGA bitstream — so
no Vivado/FPGA work is needed to use it.

## Files

| File | Purpose |
|------|---------|
| `pll_control.py` | `RedPitayaPLL` driver — SSH/`monitor` register access, all unit conversions, live signals, save/load. Headless-usable. |
| `pll_gui.py`     | PyQt5 + pyqtgraph GUI: all parameters + live lock monitor + loop-design Bode. `--mock` runs with no hardware. |
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
