import numpy as np
import pyvisa
import time
import src.RIGOL_control.DG822.DG822_control as rig
import src.Tektronix_control.AFG1022.AFG1022_control as tek

# from PicoControl.take_data_pico import initialize_pico, stream_data

"""
Charge calibration by applying a sinuisoidal E field at a fixed
frequency through the arbitrary function generator.
Optionally send control signal to the HV amplifier to moidfy the 
charge state

!!!!!!!!!!

TODO: need to add the bit that collects and analyses the data. 

!!!!!!!!!!
"""

_VISA_ADDRESS_tektronix = "USB0::0x0699::0x0353::2238362::INSTR"

### Variables
# AMPLIFIED = True
AMPLIFIED = False

# DC bias in V

# OFFSET1 = 60
# OFFSET2 = 60
# OFFSET1 = -40
# OFFSET2 = -40
# OFFSET1 = 0
# OFFSET2 = 0

# AMP  = 60    # Peak-to-peak amplitude of the driving E field @ 1 mbar
# FREQ = 57000 # Driving frequency in Hz

## Values at low pressure
# OFFSET1 = 40
# OFFSET2 = 40
OFFSET1 = 0
OFFSET2 = 0
AMP  = 0.1
FREQ = 93000
# FREQ = 147000

if AMPLIFIED:
    AMP = AMP / 20
    OFFSET1 = OFFSET1 / -20
    OFFSET2 = OFFSET2 / -20
else:
    OFFSET1 = OFFSET1
    OFFSET2 = OFFSET2

# Connect to function generator and apply sine wave
tek.sine_wave(_VISA_ADDRESS_tektronix, amplitude=AMP, frequency=FREQ, offset=OFFSET1, channel=1)
tek.dc_offset(_VISA_ADDRESS_tektronix, offset=OFFSET2, channel=2)
tek.turn_on(_VISA_ADDRESS_tektronix, channel=1)
tek.turn_on(_VISA_ADDRESS_tektronix, channel=2)
print('E field switched on')

# Have the E field on for a while
i = 0
while True:
    try:
        time.sleep(1)
        i+=1
    except KeyboardInterrupt:
        break

tek.turn_off(_VISA_ADDRESS_tektronix, channel=1)
# if OFFSET2 != 0:
tek.turn_off(_VISA_ADDRESS_tektronix, channel=2)

print('E field switched off')
print('Program ends')