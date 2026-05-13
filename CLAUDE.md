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
archived/             # Deprecated code
```

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
