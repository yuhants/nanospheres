"""
Quantum Composers 9614 pulse generator.

Two interfaces live here:

  * the original module-level functions (``set_pulse`` / ``turn_on`` /
    ``turn_off``), which open and close the serial port on every call.  Scripts
    such as ``dm_search/impulse_calibration.py`` use these.
  * ``PulseGenerator9614``, a session object that holds the port open, checks the
    controller's reply to every command and can read settings back.  Written for
    the pulse GUI (``pulse_gen/pulse_gui.py``), where reopening the port for each
    knob turn would be both slow and fragile.

Both speak the same wire protocol, so they can be mixed if needed — but not at
the same time, since the port is exclusive.
"""

import serial
import threading
import time

DEFAULT_PORT     = 'COM9'
DEFAULT_BAUDRATE = 38400

# Error replies from the 9600-series command parser
ERROR_CODES = {
    '?1': 'incorrect prefix (missing colon or *)',
    '?2': 'missing command keyword',
    '?3': 'invalid command keyword',
    '?4': 'missing parameter',
    '?5': 'invalid parameter',
    '?6': 'query only, command needs a question mark',
    '?7': 'invalid query, command does not have a query form',
    '?8': 'command unavailable in current system state',
}


def write_and_receive(ser, command):
    ser.write(bytes(command, encoding='UTF-8'))
    response = ser.readline()

    return(response.decode('UTF-8'))

def set_pulse(port=r'COM9', baudrate='38400', channel=1, amp=5, width='0.0000002', period='0.3'):
    ser = serial.Serial(r'COM9', baudrate='38400', timeout=1)

    pulse_channel = f'PULSE{str(channel)}'
    _ = write_and_receive(ser, f':{pulse_channel}:STATE ON \r\n')
    _ = write_and_receive(ser, f':{pulse_channel}:OUTP:AMPL {amp} \r\n')
    _ = write_and_receive(ser, f':PULSE1:WIDT {width} \r\n')
    _ = write_and_receive(ser, f':PULSE0:PER {period} \r\n')

    _ = write_and_receive(ser, f':{pulse_channel}:POL NORM \r\n')
    _ = write_and_receive(ser, f':{pulse_channel}:DELAY 0 \r\n')

    _ = write_and_receive(ser, ':PULSE0:MODE NORM \r\n')
    _ = write_and_receive(ser, ':PULSE0:EXT:MODE DIS \r\n')

    ser.close()

def turn_on(port=r'COM9', baudrate='38400'):
    ser = serial.Serial(r'COM9', baudrate='38400', timeout=1)
    ser.write(b':PULSE0:STATE ON \r\n')
    ser.close()

def turn_off(port=r'COM9', baudrate='38400'):
    ser = serial.Serial(r'COM9', baudrate='38400', timeout=1)
    ser.write(b':PULSE0:STATE OFF \r\n')
    ser.close()

class PulseGeneratorError(RuntimeError):
    """Raised when the controller rejects a command or the port is not open."""


def _fmt_seconds(value):
    """
    Format a time in seconds for the controller's parser.

    Trailing zeros are trimmed so 200 ns goes out as '0.0000002', exactly the
    string the calibration script sends, rather than padded past the 9614's
    timing resolution.
    """
    text = f'{float(value):.11f}'.rstrip('0')
    return text + '0' if text.endswith('.') else text


class PulseGenerator9614:
    """
    Persistent serial session with a 9614.

    The port stays open for the lifetime of the object, and every command is
    checked against the controller's reply ('ok' or '?n'), so a typo'd parameter
    surfaces immediately instead of silently leaving the pulse train wrong.

        gen = PulseGenerator9614(port='COM9')
        gen.connect()
        gen.configure(channel=1, amp=2.5, width=200e-9, period=0.15)
        gen.output_on()
        ...
        gen.output_off()
        gen.close()

    All access is serialized by an internal lock, so a GUI thread and a sweep
    thread can share one instance.
    """

    def __init__(self, port=DEFAULT_PORT, baudrate=DEFAULT_BAUDRATE, timeout=1.0):
        self.port     = port
        self.baudrate = int(baudrate)
        self.timeout  = timeout
        self._ser     = None
        self._lock    = threading.RLock()

    # ── Connection ─────────────────────────────────────────────────────────────

    @property
    def connected(self):
        return self._ser is not None and self._ser.is_open

    def connect(self):
        with self._lock:
            if self.connected:
                return
            self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

    def close(self):
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                finally:
                    self._ser = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Raw protocol ───────────────────────────────────────────────────────────

    def _transact(self, command):
        with self._lock:
            if not self.connected:
                raise PulseGeneratorError('pulse generator is not connected')
            self._ser.reset_input_buffer()
            self._ser.write(bytes(f'{command} \r\n', encoding='UTF-8'))
            return self._ser.readline().decode('UTF-8', errors='replace').strip()

    def command(self, command):
        """Send a command and raise if the controller rejects it."""
        response = self._transact(command)
        if response.startswith('?'):
            raise PulseGeneratorError(
                f'{command!r} rejected: {ERROR_CODES.get(response, response)}')
        return response

    def query(self, command):
        """Send a query and return the reply verbatim ('' if the read timed out)."""
        return self._transact(command)

    def idn(self):
        return self.query('*IDN?')

    # ── Channel settings ───────────────────────────────────────────────────────

    def set_channel_state(self, channel, on=True):
        self.command(f':PULSE{channel}:STATE {"ON" if on else "OFF"}')

    def set_amplitude(self, channel, amp):
        """Adjustable-mode output amplitude in volts (no effect in TTL mode)."""
        self.command(f':PULSE{channel}:OUTP:AMPL {float(amp):.2f}')

    def set_output_mode(self, channel, mode):
        """'TTL' for the fixed 5 V level, 'ADJ' to use the amplitude setting."""
        self.command(f':PULSE{channel}:OUTP:MODE {mode.upper()}')

    def set_width(self, channel, width):
        self.command(f':PULSE{channel}:WIDT {_fmt_seconds(width)}')

    def set_delay(self, channel, delay):
        self.command(f':PULSE{channel}:DELAY {_fmt_seconds(delay)}')

    def set_polarity(self, channel, polarity='NORM'):
        self.command(f':PULSE{channel}:POL {polarity.upper()}')

    # ── T0 (system) settings ───────────────────────────────────────────────────

    def set_period(self, period):
        self.command(f':PULSE0:PER {_fmt_seconds(period)}')

    def set_system_mode(self, mode='NORM', external=False):
        self.command(f':PULSE0:MODE {mode.upper()}')
        self.command(f':PULSE0:EXT:MODE {"TRIG" if external else "DIS"}')

    def output_on(self):
        self.command(':PULSE0:STATE ON')

    def output_off(self):
        self.command(':PULSE0:STATE OFF')

    def is_output_on(self):
        """True/False from the controller, or None if it did not answer."""
        response = self.query(':PULSE0:STATE?').strip().upper()
        if response in ('1', 'ON'):
            return True
        if response in ('0', 'OFF'):
            return False
        return None

    # ── Composite operations ───────────────────────────────────────────────────

    def configure(self, channel=1, amp=None, width=None, period=None, delay=0.0,
                  polarity='NORM', output_mode=None, system_mode='NORM'):
        """
        Set up one channel's pulse train, matching what ``set_pulse`` sends.

        ``output_mode`` of None leaves the TTL/adjustable setting alone, which is
        the behaviour of the original ``set_pulse``; pass 'ADJ' to make sure the
        amplitude setting is the one that reaches the output.
        """
        self.set_channel_state(channel, True)
        if output_mode is not None:
            self.set_output_mode(channel, output_mode)
        if amp is not None:
            self.set_amplitude(channel, amp)
        if width is not None:
            self.set_width(channel, width)
        if period is not None:
            self.set_period(period)
        if delay is not None:
            self.set_delay(channel, delay)
        if polarity is not None:
            self.set_polarity(channel, polarity)
        self.set_system_mode(system_mode)

    def read_settings(self, channel=1):
        """
        Read the current pulse setup back from the controller.

        Returns a dict of floats/strings; entries the controller did not answer
        are left as None rather than guessed.
        """
        def _float(cmd):
            response = self.query(cmd)
            try:
                return float(response)
            except ValueError:
                return None

        def _text(cmd):
            response = self.query(cmd).strip().upper()
            return response or None

        return {
            'amp':         _float(f':PULSE{channel}:OUTP:AMPL?'),
            'width':       _float(f':PULSE{channel}:WIDT?'),
            'delay':       _float(f':PULSE{channel}:DELAY?'),
            'period':      _float(':PULSE0:PER?'),
            'polarity':    _text(f':PULSE{channel}:POL?'),
            'output_mode': _text(f':PULSE{channel}:OUTP:MODE?'),
            'state':       _text(f':PULSE{channel}:STATE?'),
        }


def main():
    set_pulse(channel=1, amp=10, width='0.0000002', period='0.3')
    turn_on()

    i = 0
    while i < 1000:
        try:
            time.sleep(1)
            i+=1
        except KeyboardInterrupt:
            break

    turn_off()

if __name__ == '__main__':
    main()