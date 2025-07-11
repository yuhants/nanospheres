import sys, os
import time
sys.path.append(os.path.dirname(r'C:\Users\yuhan\nanospheres\control'))

import control.src.Raspberry_pi_pico_control.RP_pico_control as rpc


class StepperMotorMag:

    def __init__(self, port):
        self.port = port
        self.rpp = rpc.RPpico(port = port)
        command = 'NSLEEP = Pin(21, Pin.OUT)'
        self.rpp.send_command(command)
        command = 'STEP = Pin(20, Pin.OUT)'
        self.rpp.send_command(command)
        command = 'DIR = Pin(19, Pin.OUT)'
        self.rpp.send_command(command)

    def not_sleep(self):
        command = 'NSLEEP.value(1)'
        self.rpp.send_command(command)
        time.sleep(1)

    def sleep(self):
        command = 'NSLEEP.value(0)'
        self.rpp.send_command(command)
        time.sleep(1)

    def move_up(self, nsteps):
        command = 'DIR.value(1)'
        self.rpp.send_command(command)
        time.sleep(1)
        i = 0
        while i < nsteps:
            command = 'STEP.value(1)'
            self.rpp.send_command(command)
            time.sleep(0.01)
            command = 'STEP.value(0)'
            self.rpp.send_command(command)
            i += 1

    def move_down(self, nsteps):
        command = 'DIR.value(0)'
        self.rpp.send_command(command)
        time.sleep(1)
        i = 0
        while i < nsteps:
            command = 'STEP.value(1)'
            self.rpp.send_command(command)
            time.sleep(0.01)
            command = 'STEP.value(0)'
            self.rpp.send_command(command)
            i += 1

    def close(self):
        self.rpp.close()