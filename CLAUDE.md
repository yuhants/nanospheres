# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Experiment control and analysis code for a levitated nanosphere physics experiment. The lab traps a charged silica nanosphere in an optical trap, drives it with electric/magnetic fields, and records its motion via photodetectors. There are no automated tests and no build system — scripts are run directly on the lab PC.

## Repository structure

```
control/src/          # Hardware instrument drivers (one subdir per instrument)
control/              # Setup/calibration scripts run directly on the experiment PC
dm_search/            # Dark matter search: data-taking and calibration scripts
lorentz_force/        # Lorentz force measurement scripts and notebooks
charging/             # Charge manipulation and measurement scripts
utils/                # Shared analysis utilities imported by notebooks
analysis_notebooks/   # Jupyter notebooks for offline data analysis (date-named)
calculation/          # Theory/modelling notebooks
lab_gui/              # Single-page PyQt5 GUI for routine tasks (E field, HV, pulses)
daq/                  # Picoscope DAQ GUI: monitor/record fast data + derived slow data
archived/             # Deprecated code
```

## Lab task GUI (`lab_gui/`)

`python lab_gui/lab_gui.py` — one window, one page, a foldable section per
routine task, replacing the edit-the-top-and-run scripts
(`control/charge_measurement.py`, `control/apply_hv.py`). Structure:

- `task_panel.py` — `TaskPanel` base class (`is_active` / `status_text` / `stop` /
  `shutdown` / `to_dict` / `from_dict`), plus the shared `ConnectionBar`,
  `AutoOffBox` and `SectionFrame` widgets
- `panels/` — one module per task
- `registry.py` — `PANEL_SPECS` (module, class, column, enabled, collapsed):
  adding a task is a new panel module and one line here. Panels import lazily;
  a missing driver becomes an "unavailable" section rather than a GUI that will
  not start.
- `layout.json` (optional, per machine, written by the Sections menu) — overrides
  column count and any section's column/enabled/collapsed

The window owns the shared log, the "All outputs OFF" button (also Escape), the
"●" active marker on each section heading, and presets covering every section in
one JSON.

## Picoscope DAQ GUI (`daq/`)

`python daq/daq_gui.py` — monitor / record with one scope session
(`picoscope_daq.py`: `PicoscopeDAQ.monitor_stream()`, `acquire_one()` +
`write_hdf5()` per file).

- **Monitor stream** (Live view group): `Roll` is one endless stream;
  `Sweeps` (`monitor_stream(burst=True)`, as a bench scope does) takes a bounded full-rate window per
  refresh, buffered in the scope. Channels x rate x 2 B above the link's
  sustained throughput (~20 MB/s on a USB 2.0 port) corrupts a continuous
  monitor after tens of seconds (thick traces, PSD comb at fs/512); `Auto`
  switches to sweeps above `USB2_STREAM_LIMIT` / `USB3_STREAM_LIMIT`.
- **Slow inputs** (lock-in amplifier output, temperature...) share the scope
  with fast ones — the scope has one sample rate, so they are sampled fast and
  averaged in software. Channel table Save = `Slow` (`ChannelConfig.slow`)
  keeps a channel as a slow series instead of a waveform.
  - `slow_data.py` (no Qt) — `SlowReducer`: one row per 1/rate s with
    `mean_<ch>`, `rms_<ch>` (plus an optional software lock-in,
    `<name>_x/_y/_r/_phase/_ref/_freq`). `SlowPipeline` runs it on its own
    thread, fed by the `on_samples` tap of both streaming callbacks — the driver
    callback must only enqueue. In burst monitoring each burst gives one row.
    `read_slow(path)` reads the result back.
  - `slow_panel.py` — "Slow data" tab and the strip chart under the fast plots.
  - On disk: group `/slow` (`time` in unix s + one dataset per column) inside
    each recorded file, and optionally a continuous
    `<prefix>slow_<date>_<time>.hdf5` log that also runs while monitoring.
  - Sources that are not scope channels (serial/USB sensors) are not handled.

## Trap monitor GUI (`camera/`)

`python camera/trap_monitor_gui.py [--mock]` — Thorlabs camera view with
auto-capture/classification of trapped spheres. Two tabs:

- **Trap monitor** — camera, detection ROI, backscatter RMS, pressure, capture
  library (one HDF5, `captures/` + `loadings/` groups).
- **Laser stabilization** and **Homodyne phase lock** (`camera/laser_stab.py`)
  — two PI locks on ONE Red Pitaya via PyRPL (`RedPitayaPyRPL`, FPGA `pidN`
  modules) sharing a single connection (`SharedLink`; a second PyRPL client
  to the same board breaks the first). `PiLock` holds the lock logic,
  `PiLockPanel` the generic tab (scrollable controls, live strip chart,
  calibration sweep → setpoint crossing / start voltage / gain sign).
  - Laser (`LaserStabPanel`, pid0: photodiode in1 → out1 → HV amp → EOM → PBS)
    adds **Drop particle**: freeze gains, hold output at the extinction
    voltage for a dwell, restore `ival` then gains. The drop controls
    (`LaserStabPanel.drop_group`, built with `embed_drop=False`) are placed
    under the live camera view on the Trap monitor tab, where the operator
    decides; drops are recorded on the current loading session (`n_drops`,
    `drops` JSON attrs).
  - Homodyne (`HomodynePanel`, pid1: homodyne in2 → out2 → fiber stretcher)
    adds rail auto-reset: near Vmin/Vmax the integrator jumps by one fringe
    (V_2π, measurable from the sweep) or to the centre, so the lock re-acquires.
  Settings persist in `camera/laser_stab_settings.json` (gitignored, keys
  `link`/`laser`/`homodyne`). `MockBackend` simulates both plants for `--mock`.
  Connecting with "Reload FPGA" replaces the board's running app
  (e.g. Oscilloscope+Lock-in+PID) and zeroes every PID output, dropping a
  trapped particle. `link.fpga_loaded` in the settings file remembers that
  the bitstream is on the board, so after a PC reboot the default is to
  reconnect without reloading, a reload asks for confirmation, and a lock
  found running is adopted (settings read from the registers, nothing
  written). All PyRPL access stays on the GUI thread.

## Hardware drivers (`control/src/`)

Each instrument has its own subdirectory:
- `RIGOL_control/DG822/` — function generator (sin waves, pulses, DC bias)
- `RIGOL_control/DP821/` — DC power supply
- `Tektronix_control/AFG1022/` — arbitrary function generator
- `Red_pitaya_control/` — Red Pitaya FPGA board
- `Raspberry_pi_pico_control/` — RP Pico microcontroller
- `Picoscope_control/` — (legacy; active DAQ code is inline in data-taking scripts)

Two instrument drivers live loose at `control/src/` root (not in subdirs):
- `agilent_twisstorr_84fsag_control.py` — vacuum pressure gauge (serial/RS232, `read_pressure()`)
- `quantum_composers_9614_control.py` — pulse generator (serial, `set_pulse()` / `turn_on()` / `turn_off()`)

## Import pattern for data-taking scripts

Scripts in `dm_search/`, `lorentz_force/`, and `charging/` cannot be run as part of a package — they use a hardcoded `sys.path` fix to locate `control/`:

```python
import sys, os
sys.path.append(os.path.dirname(r'C:\Users\yuhan\nanospheres\control'))

from control.src.agilent_twisstorr_84fsag_control import read_pressure
from control.src.RIGOL_control.DG822.DG822_control import FuncGen
```

Scripts that remain inside `control/` (e.g. `apply_hv.py`) use relative imports instead:
```python
import src.RIGOL_control.DG822.DG822_control as rig
```

When moving a script out of `control/`, add the `sys.path.append` block and switch to absolute `control.src.*` imports.

## Data acquisition pattern (Picoscope ps4000a)

All active data-taking scripts follow the same structure:
1. Configure channels, ranges, and buffer sizes at the top of the file (edited before each run)
2. `set_up_pico()` → `stream_data()` in a loop
3. Each iteration writes one HDF5 file via `h5py` with this layout:
   - `data/` group with attrs: `timestamp`, `pressure_mbar`, `delta_t`
   - Per-channel datasets: `channel_c` etc. (ADC int16 counts + `adc2mv` attr)
   - Low-signal channels stored as mean only (`channel_a_mean_mv` attr)

Two Picoscopes are in use: `JO279/0118` ("cloud") and `JY140/0294`.

Channel letter → physical signal mapping is experiment-specific and set per-script.

## Shared utilities

`utils/utils.py` — loads `.mat` files (legacy format) and computes PSDs  
`utils/analysis_utils.py` — peak fitting (Lorentzian), STFT, charge calibration chi-square  
`utils/impulse_ana_utils.py` — bandpass filtering for impulse/pulse analysis  
`utils/get_sphere_charge.py`, `get_calibration_factor.py` — charge and displacement calibration  
`utils/scan_cooling_phase.py` / `scan_cooling_gain.py` — PSD area vs. feedback parameter scans

`analysis_notebooks/analysis_utils.py` is a **separate, older** file used only by notebooks in that directory — it is not the same as `utils/analysis_utils.py`.

## Data storage

Live data is written to `E:\` on the lab PC (not in this repo). Paths are hardcoded at the top of each data-taking script and must be updated before each run along with `sphere`, `file_prefix`, `idx_start`, and `n_file`.
