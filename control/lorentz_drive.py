import pyvisa
import time
import src.RIGOL_control.DG822.DG822_control as rig

DC_volt = 0.0
SIN_volt = 0.35
SIN_freq = 69000

# Connect to Rigol
_VISA_ADDRESS_rigol = 'USB0::0x1AB1::0x0643::DG8A261500550::INSTR'
DG822 = rig.FuncGen(_VISA_ADDRESS_rigol)

# DC_volt = -2
# DG822.DC(channel = 1, off = DC_volt)
# DG822.DC(channel = 2, off = DC_volt)

DG822.sin_wave(channel=1, amp = SIN_volt*1, off = DC_volt, freq = SIN_freq, phase = 200)
# DG822.sin_wave(channel=2, amp = SIN_volt*1.3, off = DC_volt, freq = SIN_freq, phase = 180.3)

DG822.turn_on(channel=1)
# DG822.turn_on(channel=2)

DG822.syncronise_waveforms()

i = 0
while True:
    try:
        time.sleep(1)
        i+=1
    except KeyboardInterrupt:
        break

DG822.turn_off(channel=1)
# DG822.turn_off(channel=2)