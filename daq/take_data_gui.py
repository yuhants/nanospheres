"""
Launch the Picoscope DAQ GUI.

Typical workflow:
  1. Run this script (or `python daq/daq_gui.py` directly).
  2. Select scope, enable channels, set ranges and sample rate.
  3. Set output directory and file prefix.
  4. Click Connect, then Start.
  5. Save your settings as a preset (e.g. daq/presets/dm_search.json)
     so you can reload them next session.

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
