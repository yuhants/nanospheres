"""
Launch the Picoscope DAQ GUI.

Typical workflow:
  1. Run this script (or `python daq/daq_gui.py` directly).
  2. Select scope, enable channels, set ranges and sample rate.  The rate can be
     typed directly (e.g. 1 MS/s) — the equivalent integer interval in ns/µs/ms
     is filled in, and the summary line shows the rate actually achieved, since
     the scope quantizes it.
  3. Click Connect, then Monitor — a free-running time trace and PSD, nothing
     written to disk.  Tune ranges/coupling and press "Apply settings" to push
     them to the scope without leaving monitor mode (the summary line warns when
     the panel and the scope disagree); adjust the timebase (s/div × divisions),
     vertical range, PSD span, nperseg and averaging while it runs.
  4. Set output directory and file prefix, then click Record.  Monitoring stops
     and acquisition begins with the same channel setup and the same plots; the
     spectrum then refreshes from each whole file.  With "Resume monitoring
     after a run" ticked, Stop (or the last file) drops straight back to
     monitoring.
  5. For a pulse calibration, open the Pulses tab: connect to the Quantum
     Composers 9614, set the pulse shape and amplitude list, tick "Step the
     amplitude while recording" and set the files per amplitude ("Set N files to
     match" sizes the run to the list).  Recording then takes those files at each
     amplitude in turn, changing the amplitude only between files; filenames get
     a "2.5v_" tag and every file records the pulse settings in its attributes.
     By default the index restarts at each amplitude (2.5v_0, 2.5v_1, 5v_0, …,
     as in the script) — untick "Restart the file index at each amplitude" for
     one counter across the run.  The output is turned off when the run ends, is
     stopped, or errors.
  6. Save your settings as a preset (e.g. daq/presets/dm_search.json)
     so you can reload them next session.  Presets carry the live-view, display
     and pulse settings too; older presets still load (new fields keep defaults).

Scales (all live-editable, all saved in presets):
  - Sample rate         MS/s / kS/s / S/s, or the raw interval + time units.
  - Horizontal          s/div × number of divisions; ticks land on the divisions
                        and the axis is pinned to that window.  End-of-file
                        frames cover the whole file, so they autoscale instead.
  - Vertical (time)     auto | full range (±input range) | manual min/max in mV |
                        mV/div × divisions, symmetric about zero.
  - PSD panel           frequency min/max, V²/Hz min/max (blank = autoscale),
                        linear or log frequency axis.
  - Cadence             trace refresh, PSD refresh, and the driver chunk size.
  - Resolution          trace points and PSD points drawn (PSD thinning keeps the
                        maximum of each group so peaks survive, which reads the
                        broadband floor a few dB high; set 0 for every bin).

Reading the display:
  - Roll mode (default) plots history at negative time with the newest sample at
    0, so the trace slides leftwards on a fixed axis the way a scope in roll mode
    does.  For it to look continuous the time window must be several times longer
    than the refresh interval — the default 0.5 s window at 0.05 s refresh moves
    the trace 10% of a screen per frame.  Uncheck it to have each frame start
    at t=0 instead.
  - Trace and spectrum update on separate clocks ("Refresh" and "PSD refresh"):
    the trace is cheap and runs at up to 50 fps, while a Welch transform over a
    long span is not and runs a few times a second.  The status line reports the
    frame rate actually achieved.
  - The time panel draws a min/max envelope band rather than every Nth sample,
    so a fast oscillation shows its true amplitude instead of aliasing into a
    wrong-amplitude squiggle.  The band is the real excursion within each pixel
    column, not a single sample from it.
  - The readout strip under the plots gives per-channel rms, mean and peak
    frequency, updated every frame.
  - Over-range channels are called out in the status line.

History (display memory, either mode):
  - The last N frames are kept — 200 by default, which at the default refresh is
    about 10 s of stepping room.  The bar under the plots scrolls back through
    them: ◀ ▶ step one frame, the slider seeks, and "Live" returns to the present.
  - Stepping back does not interrupt anything.  The scope keeps streaming and
    frames keep being stored, so Live always lands on the current waveform.
    "Hold" freezes the display in place with the same guarantee.
  - What is stored is the frame as drawn — the decimated trace or min/max band
    plus that frame's spectrum — not the raw samples.  Keeping raw samples would
    cost megabytes per frame (0.5 s at 5 MS/s is 5 MB per channel), so, as on a
    scope, zooming into a frame from history reveals no detail that was not
    already on screen.  For full-resolution data, record to file.
  - The Live view group shows the memory in use and what it will be at full
    depth; set the depth to 0 to switch history off.
  - "Save frame…" writes the frame on screen to a .npz (per-channel arrays plus
    the readouts, at display resolution) for offline inspection.
  - Changing the channel set clears history, since stored frames describe the
    channels that were active when they were captured.  A frame captured at a
    different timebase keeps its own span when replayed, rather than being
    cropped to the current s/div.

Zooming (either mode, live or stopped):
  - Wheel over a panel zooms about the cursor: horizontally in the time panel
    (Shift for vertical), both axes in the PSD panel.
  - The toolbar above the plots has pan, box zoom, home and save-image.
  - Any of those hands the axis limits to you, and the live rescaling stands down
    so the view stops jumping back; "Reset view" returns control to the settings
    on the left.  Streaming continues throughout — use "Freeze" to hold a frame
    on screen while the scope keeps running.

Monitor mode notes:
  - Nothing is saved, so dropped samples are harmless — only the refresh rate
    suffers if the PSD span is set very long.

Headless monitoring (no GUI):

    with PicoscopeDAQ(config) as daq:
        daq.connect(PICO_SERIALS['Cloud  (JO279/0118)'])
        daq.monitor_stream(
            on_window=lambda w, adc2mv, dt, ovf: print(w.std(axis=1) * adc2mv),
            ring_duration=0.5, update_period=1.0,
            should_stop=lambda: False,     # Ctrl-C to break out
        )

Headless usage (no GUI, e.g. for scripted or triggered acquisition):

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from daq.picoscope_daq import PicoscopeDAQ, AcquisitionConfig, ChannelConfig, PICO_SERIALS
    from control.src.agilent_twisstorr_84fsag_control import read_pressure

    config = AcquisitionConfig(
        channels=[
            ChannelConfig('C', range_idx=9),           # x/y detection, 5 V range
            ChannelConfig('D', range_idx=6),            # z detection, 1 V range
            ChannelConfig('G', range_idx=9),            # feedback monitor, 5 V range
        ],
        sample_interval=2,
        sample_units='PS4000A_US',   # 500 kHz
        buffer_size=2**25,           # ~67 s per file
    )

    file_directory = r'E:\\dm_data\\sphere_YYYYMMDD\\YYYYMMDD_run'
    file_prefix    = 'YYYYMMDD_cdg_'
    n_files        = 1440

    with PicoscopeDAQ(config) as daq:
        daq.connect(PICO_SERIALS['Cloud  (JO279/0118)'])
        for i in range(n_files):
            pressure = read_pressure(port='COM7')
            timestamp, dt, adc2mvs, data = daq.acquire_one()
            daq.write_hdf5(
                f'{file_directory}\\{file_prefix}{i}.hdf5',
                timestamp, dt, adc2mvs, data, pressure_mbar=pressure,
            )
            print(f'[{i+1}/{n_files}]  pressure={pressure:.2e} mbar')
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daq.daq_gui import launch_gui

if __name__ == '__main__':
    launch_gui()
