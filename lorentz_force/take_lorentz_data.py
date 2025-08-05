import sys, os
import time
sys.path.append(os.path.dirname(r'C:\Users\yuhan\nanospheres\control'))

import numpy as np
import matplotlib.pyplot as plt

from control.src.agilent_twisstorr_84fsag_control import read_pressure
from control.src.RIGOL_control.DP821.DP821_control import Supply
from picosdk.ps4000a import ps4000a as ps
from picosdk.functions import assert_pico_ok
import ctypes

import h5py

_VISA_ADDRESS_rigol_dp821 = 'USB0::0x1AB1::0x0E11::DP8G223300057::INSTR'

# Picoscope DAQ setting
serial_0 = ctypes.create_string_buffer(b'JO279/0118')  # Picoscope on cloud
serial_1 = ctypes.create_string_buffer(b'JY140/0294')

data_type = 'full'

# Digitization range (0-11): 10, 20, 50, 100, 200, 500 (mV), 1, 2, 5, 10, 20, 50 (V)
if data_type == 'full':
    channels = ['A', 'B', 'C', 'D', 'G']
    channel_ranges = np.array([9, 9, 9, 6, 8])
    channel_couplings = ['DC', 'DC', 'DC', 'DC', 'DC']

analog_offsets = None

n_buffer = 1  # Number of buffer to capture
# buffer_size = int(3e6)
buffer_size = int(2**22)

sample_interval = 2
sample_units = 'PS4000A_US'

sphere = 'sphere_20250708'
file_directory = rf"E:\lorentz_force\{sphere}\20250804_2e-8mbar_newsamp"
file_prefix = '20250804_m350e_79khz_100vpp_'

idx_start = 1915
n_file = 3000 - idx_start

# Variables used by Picoscope DAQ
enabled = 1
disabled = 0
channelInputRanges = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
channel_dict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7}
time_dict = {'PS4000A_NS':1e-9, 'PS4000A_US':1e-6, 'PS4000A_MS':1e-3}

def main():
    # Set up the power supply
    dp821 = Supply(_VISA_ADDRESS_rigol_dp821)
    dp821.apply(channel='CH1', voltage=12, current=0.5)
    dp821.apply(channel='CH2', voltage=8.4, current=10.5)

    dp821.turn_off(channel='CH1')
    dp821.turn_off(channel='CH2')

    if not os.path.isdir(file_directory):
        os.mkdir(file_directory)

    chandle, status = set_up_pico(serial_0, channels, channel_ranges, channel_couplings, analog_offsets,
                                  buffer_size)

    # Data taking
    for idx in range(n_file):            
        pressure = read_pressure(port=r'COM7', baudrate='9600')

        # With B field on
        dp821.turn_on(channel='CH2')
        time.sleep(1)   # Wait 1 sec for the transient to settle
        flip_info = dp821.get_info(channel='CH1')
        drive_info = dp821.get_info(channel='CH2')

        file_name = rf'{file_prefix}{'withb_'}{idx+idx_start}.hdf5'
        timestamp, dt, adc2mvs, data = stream_data(chandle, status, sample_interval, sample_units, channel_ranges, buffer_size, n_buffer)

        with h5py.File(os.path.join(file_directory, file_name), 'w') as f:
            print(f'Writing file {file_name}')

            g = f.create_group('data')
            g.attrs['timestamp'] = timestamp
            g.attrs['pressure_mbar'] = pressure
            g.attrs['flip_info'] = flip_info
            g.attrs['drive_info'] = drive_info
            g.attrs['delta_t'] = dt * time_dict[sample_units]
            for i, channel in enumerate(channels):
                if channel == 'D' or channel == 'G':
                    dataset = g.create_dataset(f'channel_{channel.lower()}', data=data[i], dtype=np.int16)
                    dataset.attrs['adc2mv'] = adc2mvs[i]
                else:
                    g.attrs[f'channel_{channel.lower()}_mean_mv'] = np.mean(data[i]) * adc2mvs[i]
            f.close()

        # With B field flipped
        dp821.turn_on(channel='CH1')
        time.sleep(1)   # Wait 1 sec for the transient to settle
        flip_info = dp821.get_info(channel='CH1')
        drive_info = dp821.get_info(channel='CH2')

        file_name = rf'{file_prefix}{'withb_flipped_'}{idx+idx_start}.hdf5'
        timestamp, dt, adc2mvs, data = stream_data(chandle, status, sample_interval, sample_units, channel_ranges, buffer_size, n_buffer)

        with h5py.File(os.path.join(file_directory, file_name), 'w') as f:
            print(f'Writing file {file_name}')

            g = f.create_group('data')
            g.attrs['timestamp'] = timestamp
            g.attrs['pressure_mbar'] = pressure
            g.attrs['flip_info'] = flip_info
            g.attrs['drive_info'] = drive_info
            g.attrs['delta_t'] = dt * time_dict[sample_units]
            for i, channel in enumerate(channels):
                if channel == 'D' or channel == 'G':
                    dataset = g.create_dataset(f'channel_{channel.lower()}', data=data[i], dtype=np.int16)
                    dataset.attrs['adc2mv'] = adc2mvs[i]
                else:
                    g.attrs[f'channel_{channel.lower()}_mean_mv'] = np.mean(data[i]) * adc2mvs[i]
            f.close()

        # No B field on
        dp821.turn_off(channel='CH1')
        dp821.turn_off(channel='CH2')
        time.sleep(1)   # Wait 1 sec for the transient to settle
        flip_info = dp821.get_info(channel='CH1')
        drive_info = dp821.get_info(channel='CH2')

        file_name = rf'{file_prefix}{'nob_'}{idx+idx_start}.hdf5'
        timestamp, dt, adc2mvs, data = stream_data(chandle, status, sample_interval, sample_units, channel_ranges, buffer_size, n_buffer)

        with h5py.File(os.path.join(file_directory, file_name), 'w') as f:
            print(f'Writing file {file_name}')

            g = f.create_group('data')
            g.attrs['timestamp'] = timestamp
            g.attrs['pressure_mbar'] = pressure
            g.attrs['flip_info'] = flip_info
            g.attrs['drive_info'] = drive_info
            g.attrs['delta_t'] = dt * time_dict[sample_units]
            for i, channel in enumerate(channels):
                if channel == 'D' or channel == 'G':
                    dataset = g.create_dataset(f'channel_{channel.lower()}', data=data[i], dtype=np.int16)
                    dataset.attrs['adc2mv'] = adc2mvs[i]
                else:
                    g.attrs[f'channel_{channel.lower()}_mean_mv'] = np.mean(data[i]) * adc2mvs[i]
            f.close()

        print(f'Channel A mean: {np.mean(data[0])*adc2mvs[0]}; Channel B mean: {np.mean(data[1])*adc2mvs[1]}')
        while (np.abs(np.mean(data[0])*adc2mvs[0]) > 4000 or np.abs(np.mean(data[1])*adc2mvs[1]) > 4000):
            print('Let cool off for a while')
            time.sleep(60 * 20)  # Rest for 20 min to dissipate some heat

    stop_and_disconnect(chandle, status)

# Functions
def set_up_pico(serial, channels, channel_ranges, channel_couplings, analog_offsets,
                buffer_size):
    # Create chandle and status
    chandle = ctypes.c_int16()
    status = {}

    # Initialize picoscope
    initialize_pico(chandle, status, serial)
    set_channels_pico(chandle, status, channels, channel_ranges, analog_offsets, channel_couplings)

    # Create buffer that stores the data
    set_data_buffers(chandle, status, channels, buffer_size)

    return chandle, status

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

def set_data_buffers(chandle, status, channels, buffer_size):
    global one_buffer

    sizeOfOneBuffer = buffer_size
    memory_segment = 0

    # Create buffers ready for assigning pointers for data collection
    one_buffer = np.zeros(shape=(len(channels), buffer_size), dtype=np.int16)

    for i, channel in enumerate(channels):
        status_prefix_buff = f'setDataBuffers{channel}'
        channel_num = channel_dict[channel]

        status[status_prefix_buff] = ps.ps4000aSetDataBuffers(chandle,
                                                              channel_num,
                                                              one_buffer[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                              None,
                                                              sizeOfOneBuffer,
                                                              memory_segment,
                                                              ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'])
        assert_pico_ok(status[status_prefix_buff])

def stream_data(chandle, status, sample_interval, sample_units, channel_ranges, buffer_size, n_buffer):
    global nextSample, total_buffer, one_buffer

    sizeOfOneBuffer = buffer_size
    numBuffersToCapture = n_buffer
    totalSamples = sizeOfOneBuffer * numBuffersToCapture
    total_buffer = np.zeros(shape=(len(channel_ranges), totalSamples), dtype=np.int16)

    sampleInterval = ctypes.c_int32(sample_interval)
    sampleUnits = ps.PS4000A_TIME_UNITS[sample_units]

    # We are not triggering:
    maxPreTriggerSamples = 0
    autoStopOn = 1

    # No downsampling:
    downsampleRatio = 1

    timestamp = time.time()
    status["runStreaming"] = ps.ps4000aRunStreaming(chandle,
                                                    ctypes.byref(sampleInterval),
                                                    sampleUnits,
                                                    maxPreTriggerSamples,
                                                    totalSamples,
                                                    autoStopOn,
                                                    downsampleRatio,
                                                    ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'],
                                                    sizeOfOneBuffer)
    
    assert_pico_ok(status["runStreaming"])

    actualSampleInterval = sampleInterval.value
    print(f"Capturing at sample interval {actualSampleInterval} {sample_units}")

    nextSample = 0
    autoStopOuter = False
    wasCalledBack = False

    # Convert the python function into a C function pointer.
    cFuncPtr = ps.StreamingReadyType(streaming_callback)

    # Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
    while nextSample < totalSamples and not autoStopOuter:
        wasCalledBack = False
        status["getStreamingLastestValues"] = ps.ps4000aGetStreamingLatestValues(chandle, cFuncPtr, None)
        if not wasCalledBack:
            # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
            # again.
            time.sleep(0.01)

    # Find maximum ADC count value
    maxADC = ctypes.c_int16()
    status["maximumValue"] = ps.ps4000aMaximumValue(chandle, ctypes.byref(maxADC))
    assert_pico_ok(status["maximumValue"])

    # Convert ADC counts data to physical values
    # print(f'maxADC: {maxADC.value}')
    adc2mV_conversion_factors = channelInputRanges[channel_ranges] / maxADC.value
    data = total_buffer

    return timestamp, actualSampleInterval, adc2mV_conversion_factors, data

def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
    global nextSample, autoStopOuter, wasCalledBack, total_buffer, one_buffer

    wasCalledBack = True
    destEnd = nextSample + noOfSamples
    sourceEnd = startIndex + noOfSamples

    for i, channel in enumerate(channels):
        total_buffer[i][nextSample:destEnd] = one_buffer[i][startIndex:sourceEnd]

    nextSample += noOfSamples
    if autoStop:
        autoStopOuter = True

def stop_and_disconnect(chandle, status):
    # Stop the scope
    status["stop"] = ps.ps4000aStop(chandle)
    assert_pico_ok(status["stop"])

    # Disconnect the scope
    status["close"] = ps.ps4000aCloseUnit(chandle)
    assert_pico_ok(status["close"])

if __name__=="__main__":
    main()