"""
This code performs impulse calibration by applying
impulses of various different ampitudes
"""

import sys, os
import time
sys.path.append(os.path.dirname(r'C:\Users\yuhan\nanospheres\control'))

import numpy as np
from control.apply_impulse import impulse_on, turn_off
from control.src.quantum_composers_9614_control import set_pulse, turn_on, turn_off
import matplotlib.pyplot as plt

from picosdk.ps4000a import ps4000a as ps
from picosdk.functions import assert_pico_ok
import ctypes
import h5py

import matplotlib.pyplot as plt

# Impulse setting
_VISA_ADDRESS_tektronix = "USB0::0x0699::0x0353::2238362::INSTR"
amps = [1, 3, 5, 7, 9]
# amps = [1]
offset_1, offset_2 = 0.01, 0.01

# Data collection setting
file_directory = r'E:\pulse\20241113_12e'
file_prefix = r'20241113_dg_12e'

serial_0 = ctypes.create_string_buffer(b'JO279/0118')  # Picoscope on cloud
serial_1 = ctypes.create_string_buffer(b'JY140/0294')

channels = ['A', 'B']
# Digitization range (0-11): 10, 20, 50, 100, 200, 500 (mV), 1, 2, 5, 10, 20, 50 (V)
channel_ranges = np.array([9, 9])
channel_couplings = ['DC', 'DC']
analog_offsets = None

# Trigger setting
trigger_channel = 'A'
threshold = 1000
direction = 2  # Rising
delay = 0
auto_trigger = 500

timebase = 1  # 25 ns
pre_trigger_samples = 500
post_trigger_samples = 500

n_capture = 2

# Variables used by Picoscope DAQ
enabled = 1
disabled = 0
channelInputRanges = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
channel_dict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7}
time_dict = {'PS4000A_NS':1e-9, 'PS4000A_US':1e-6, 'PS4000A_MS':1e-3}

def main():
    # if not os.path.isdir(file_directory):
    #     os.mkdir(file_directory)

    chandle, status, max_samples, t_interval_ns = set_up_pico_trigger(serial_1, channels, channel_ranges, channel_couplings, analog_offsets,
                                                  trigger_channel, threshold, direction, delay, auto_trigger,
                                                  timebase, pre_trigger_samples, post_trigger_samples, n_capture)
    
    adc2mvs, data = run_rapid_block(chandle, status, pre_trigger_samples, post_trigger_samples, timebase, n_capture, channel_ranges)
    print(t_interval_ns, adc2mvs)

    for i in range(1):
        for j in range(n_capture):
            plt.plot(data[i, j])
    plt.show()

    # for amp in amps:
    #     impulse_on(_VISA_ADDRESS_tektronix, amp, offset_1, offset_2)

    #     # Data taking
    #     for i in range(10):
    #         file_name = rf'{file_prefix}_{amp}v_{i}.hdf5'
    #         timestamp, dt, adc2mvs, data = stream_data(chandle, status, sample_interval, sample_units, channel_ranges, buffer_size, n_buffer)

    #         with h5py.File(os.path.join(file_directory, file_name), 'w') as f:
    #             print(f'Writing file {file_name}')

    #             g = f.create_group('data')
    #             g.attrs['timestamp'] = timestamp
    #             g.attrs['delta_t'] = dt * time_dict[sample_units]
    #             for i, channel in enumerate(channels):
    #                 dataset = g.create_dataset(f'channel_{channel.lower()}', data=data[i], dtype=np.int16)
    #                 dataset.attrs['adc2mv'] = adc2mvs[i]
    #             f.close()

    #     turn_off(_VISA_ADDRESS_tektronix)

    stop_and_disconnect(chandle, status)

# Functions
def set_up_pico_trigger(serial, channels, channel_ranges, channel_couplings, analog_offsets,
                        trigger_channel, threshold, direction, delay, auto_trigger,
                        timebase, pre_trigger_samples, post_trigger_samples, n_capture):
    # Create chandle and status
    chandle = ctypes.c_int16()
    status = {}

    # Initialize picoscope
    initialize_pico(chandle, status, serial)
    set_channels_pico(chandle, status, channels, channel_ranges, analog_offsets, channel_couplings)

    max_samples, t_interval_ns = set_trigger(chandle, status, trigger_channel, threshold, direction, delay, auto_trigger,
                                 timebase, pre_trigger_samples, post_trigger_samples, n_capture)

    # Create buffer that stores the data
    set_data_buffers(chandle, status, channels, buffer_size=max_samples, n_segment=n_capture)

    return chandle, status, max_samples, t_interval_ns

def initialize_pico(chandle, status, serial=None):
    # Returns handle to chandle for use in future API functions
    status["openunit"] = ps.ps4000aOpenUnit(ctypes.byref(chandle), serial)
    try:
        assert_pico_ok(status["openunit"])
    except:
        powerStatus = status["openunit"]
        if powerStatus == 286:
            status["changePowerSource"] = ps.ps4000aChangePowerSource(chandle, powerStatus)
        else:
            raise
        assert_pico_ok(status["changePowerSource"])

def set_channels_pico(chandle, status, channels, channel_ranges, analog_offsets, channel_couplings):
    if analog_offsets is None:
        analog_offsets = [0.0] * len(channels)
    for i, channel in enumerate(channels):
        set_channel(chandle, status, channel, channel_ranges[i], analog_offsets[i], channel_couplings[i])

def set_channel(chandle, status, channel, channel_range, analog_offset, coupling='DC'):
    status_prefix = 'setCh' + channel
    # There is a bug for Channel E in picoscope sdk so manually find the number
    channel_num = channel_dict[channel]

    pico_coupling = 'PS4000A_DC'
    if coupling == 'AC':
        pico_coupling = 'PS4000A_AC'
    status[status_prefix] = ps.ps4000aSetChannel(chandle,
                                                #  ps.PS4000A_CHANNEL[channel_prefix],
                                                 channel_num,
                                                 enabled,
                                                 ps.PS4000A_COUPLING[pico_coupling],
                                                 channel_range,
                                                 analog_offset)
    assert_pico_ok(status[status_prefix])

def set_trigger(chandle, status, channel, threshold, direction, delay, auto_trigger, 
                timebase, pre_trigger_samples, post_trigger_samples, n_capture):
    # Set up trigger
    channel_num = channel_dict[channel]
    status['set_trigger'] = ps.ps4000aSetSimpleTrigger(chandle,
                                                       enabled,
                                                       channel_num, 
                                                       threshold, 
                                                       direction, 
                                                       delay, 
                                                       auto_trigger)
    assert_pico_ok(status['set_trigger'])

    # Get timebase info, including max samples
    max_samples = pre_trigger_samples + post_trigger_samples
    timeIntervalns = ctypes.c_float()
    returnedMaxSamples = ctypes.c_int32()
    status["getTimebase2"] = ps.ps4000aGetTimebase2(chandle, timebase, max_samples, ctypes.byref(timeIntervalns), ctypes.byref(returnedMaxSamples), 0)
    assert_pico_ok(status["getTimebase2"])
    print(timeIntervalns)

    # Set up memory segments of the picoscope for rapid block
    nMaxSamples = ctypes.c_int32(0)
    status["setMemorySegments"] = ps.ps4000aMemorySegments(chandle, n_capture, ctypes.byref(nMaxSamples))
    assert_pico_ok(status["setMemorySegments"])

    # Set number of captures
    status["SetNoOfCaptures"] = ps.ps4000aSetNoOfCaptures(chandle, n_capture)
    assert_pico_ok(status["SetNoOfCaptures"])

    return max_samples, timeIntervalns.value

def set_data_buffers(chandle, status, channels, buffer_size, n_segment):
    global one_buffer

    sizeOfOneBuffer = buffer_size
    # Create buffers ready for assigning pointers for data collection
    one_buffer = np.zeros(shape=(len(channels), n_segment, buffer_size), dtype=np.int16)

    for i, channel in enumerate(channels):
        for j in range(n_segment):
            status_prefix_buff = f'setDataBuffers{channel}'
            channel_num = channel_dict[channel]

            status[status_prefix_buff] = ps.ps4000aSetDataBuffers(chandle,
                                                                  channel_num,
                                                                  one_buffer[i, j].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                                  None,
                                                                  sizeOfOneBuffer,
                                                                  j,
                                                                  ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'])
            assert_pico_ok(status[status_prefix_buff])

def run_rapid_block(chandle, status, pre_trigger_samples, post_trigger_samples, timebase, n_capture, channel_ranges):
    global one_buffer

    # The `segmentIndex` parameter is only used to determine the
    # allowed memeory depth fo just use `0` here
    status["runBlock"] = ps.ps4000aRunBlock(chandle, pre_trigger_samples, post_trigger_samples, timebase, None, 0, None, None)
    assert_pico_ok(status["runBlock"])

    # check for end of capture
    ready = ctypes.c_int16(0)
    check = ctypes.c_int16(0)
    while ready.value == check.value:
        status["isReady"] = ps.ps4000aIsReady(chandle, ctypes.byref(ready))

    max_samples = pre_trigger_samples + post_trigger_samples
    # Creates a overlow location for data
    overflow = (ctypes.c_int16 * 10)()
    # Creates converted types maxsamples
    cmaxSamples = ctypes.c_int32(max_samples)

    # Collect data 
    status["getValuesBulk"] = ps.ps4000aGetValuesBulk(chandle, ctypes.byref(cmaxSamples), 0, (n_capture-1), 1, 0, ctypes.byref(overflow))
    assert_pico_ok(status["getValuesBulk"])

    data = one_buffer

    # Find maximum ADC count value
    maxADC = ctypes.c_int16()
    status["maximumValue"] = ps.ps4000aMaximumValue(chandle, ctypes.byref(maxADC))
    assert_pico_ok(status["maximumValue"])

    # Convert ADC counts data to physical values
    # print(f'maxADC: {maxADC.value}')
    adc2mV_conversion_factors = channelInputRanges[channel_ranges] / maxADC.value

    return adc2mV_conversion_factors, data

def stop_and_disconnect(chandle, status):
    # Stop the scope
    status["stop"] = ps.ps4000aStop(chandle)
    assert_pico_ok(status["stop"])

    # Disconnect the scope
    status["close"] = ps.ps4000aCloseUnit(chandle)
    assert_pico_ok(status["close"])

if __name__=="__main__":
    main()