import serial
import time
from datetime import datetime
from pathlib import Path

SERIAL_PORT = r'COM6'

def start(port=SERIAL_PORT, baudrate='9600'):
    ser = serial.Serial(port, baudrate, timeout=1)
    ser.write(bytes.fromhex("02 80 30 30 30 31 31 03 42 33"))

    response = ser.readline()
    print(response)

def read_pressure(port=SERIAL_PORT, baudrate='9600'):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        ser.write(bytes.fromhex("02 80 32 32 34 30 03 38 37"))

        response = ser.readline()
        pressure_str = response[6:13].decode('UTF-8')

        ser.close()

        return float(pressure_str)
    except Exception as e:
        print(e)
        return float(-1)

def calculate_xor_checksum(data):
    """Calculates the XOR checksum for a given bytearray."""
    checksum = 0
    for byte in data:
        checksum ^= byte
    return checksum    

def print_pressure(save_path=None):
    # start()
    timestamp = time.time()

    utc_datetime = datetime.fromtimestamp(timestamp)
    print('Local time (US East Coast):', utc_datetime)

    pressure = read_pressure()
    print(f'Current pressure: {pressure} mbar')

    if save_path:
        with open(save_path, 'a') as f:
            f.write('{:.3f}, {:.3e}\n'.format(time.time(), pressure))

    # data = bytearray([0x80, 0x30, 0x30, 0x30, 0x31, 0x31, 0x03])
    # checksum = calculate_xor_checksum(data)
    # print(f"Checksum: 0x{checksum:02X}")

if __name__=="__main__":
    try:
        interval_second = 10 * 60
        save_path = None
        # today = datetime.strftime(datetime.now(), '%Y%m%d_%H-%M-%S')
        # save_path = 'C:/Users/yuhan/Desktop/CCG_data/pressures_' + today + '.txt'
        # interval_second = 5
        while True:
            print_pressure(save_path)
            time.sleep(interval_second)
    except KeyboardInterrupt:
        print('Finish reading pressure')


