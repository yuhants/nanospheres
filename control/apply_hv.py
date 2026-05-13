import pyvisa
import time
import src.RIGOL_control.DG822.DG822_control as rig

VOLT = 1.05   # Voltage for triggering HV supply for needle. Value in kV.
              # There will be a minimum below which it will not ionise the air. 
              # I think this probably also maxes out around 1 kV as it can't supply more current.
FREQ_PULSE = 0.2  # Hz

# Connect Rigol channel 1 output to pin B on the Spellman J2 connector (indicated with x below)
#    o  x
#  o  o  o
#   o o o

_VISA_ADDRESS_rigol = "USB0::0x1AB1::0x0643::DG8A204201834::INSTR"
DG822 = rig.FuncGen(_VISA_ADDRESS_rigol)
DG822.pulse(amp=VOLT, duty=90, freq=FREQ_PULSE, off=-VOLT/2)

DG822.turn_on()
print('HV triggered')

# Hold in loop until cancel - have 5 minute timeout
i = 0
while i < 600:
    try:
        time.sleep(1)
        i+=1
    except KeyboardInterrupt:
        break
DG822.turn_off()
print('HV switched off')