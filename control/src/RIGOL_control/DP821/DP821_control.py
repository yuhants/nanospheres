import pyvisa
import time

class Supply:
    def __init__(self, visa_address):
        self._visa_address = visa_address
        self.open(visa_address)

    def open(self, visa_address):
        rm = pyvisa.ResourceManager()
        self._inst = rm.open_resource(visa_address)
        self.write("*CLS")
        self._id = self.query("*IDN?") 
        self._maker, self._model, self._serial = self._id.split(",")[:3]
        print(f"Connected to {self._maker} model {self._model}, serial {self._serial}")
    
    def write(self, command):
        self._inst.write(command)
    
    def query(self, command):
        response = self._inst.query(command)
        return response
    
    def apply(self, channel='CH2', voltage=8.4, current=10.5):
        cmd = f':APPL {channel},{voltage},{current}'
        self.write(cmd)

    def get_info(self, channel='CH2'):
        cmd = f':MEAS:ALL? {channel}'
        info = self.query(cmd)
        return info

    def turn_on(self, channel='CH2'):
        cmd = f':OUTP {channel},ON'
        self.write(cmd)
    
    def turn_off(self, channel='CH2'):
        cmd = f':OUTP {channel},OFF'
        self.write(cmd)

_VISA_ADDRESS_rigol_dp821 = 'USB0::0x1AB1::0x0E11::DP8G223300057::INSTR'

if __name__ == '__main__':
    supply = Supply(_VISA_ADDRESS_rigol_dp821)
    supply.apply(channel='CH1', voltage=12, current=0.5)
    supply.apply(channel='CH2', voltage=8.4, current=10.5)

    supply.turn_on(channel='CH2')
    time.sleep(2)
    supply.turn_on(channel='CH1')
    time.sleep(2)
    supply.turn_off(channel='CH1')
    supply.turn_off(channel='CH2')
    # print(supply.get_info(channel='CH2'))
