"""High-level PyRPL wrapper for the lab Red Pitaya (STEMlab 125-14).

This driver exposes three FPGA instruments that all run *in real time on the
board*:

    - scope          : oscilloscope acquisition (both fast inputs)
    - PID  (pid0-2)  : hardware PI feedback controllers
    - lock-in (iq0-2): IQ demodulation / lock-in amplifiers

Unlike the SCPI driver (``redpitaya_control.py``), the PID and lock-in loops
execute inside the FPGA, so feedback runs at the full 125 MS/s sample rate
with no PC-in-the-loop latency -- suitable for real-time feedback (e.g. cooling
or locking). The PC only sets parameters and reads out monitor data.

Important notes
---------------
* PyRPL uploads its *own* FPGA bitstream over SSH when it connects, so the
  stock Red Pitaya SCPI server must NOT be used on the board at the same time.
  Use either this driver or ``redpitaya_control.py``, not both concurrently.
* The Red Pitaya FPGA "PID" is a PI controller (proportional + integral); there
  is no derivative term in hardware. ``p`` is the proportional gain and ``i``
  is the integrator unity-gain crossover frequency in Hz.
* Inputs are named ``'in1'`` / ``'in2'``; outputs ``'out1'`` / ``'out2'``.
  Any DSP module output (e.g. ``'iq0'``, ``'pid0'``) can also be used as the
  input of another module to chain them together entirely in the FPGA.

Example
-------
    rp = RedPitayaPyRPL(hostname='192.168.1.100')

    # 1) Scope
    t, ch1, ch2 = rp.acquire_scope(duration=0.01, input1='in1', input2='in2')

    # 2) Lock-in: demodulate in1 at 50 kHz, drive a 50 kHz reference on out2
    rp.setup_lockin('iq0', frequency=50e3, bandwidth=100, input='in1',
                    amplitude=0.1, output_direct='out2')
    R, theta = rp.lockin_amplitude_phase('iq0')

    # 3) PID: lock in1 to setpoint, actuate on out1
    rp.setup_pid('pid0', input='in1', output_direct='out1',
                 setpoint=0.0, p=-0.1, i=-100.0)

    # Real-time feedback chain (all in FPGA): demodulate -> PID -> output
    rp.setup_lockin('iq1', frequency=50e3, bandwidth=1e3, input='in1',
                    output_signal='quadrature', output_direct='off')
    rp.setup_pid('pid1', input='iq1', output_direct='out1', setpoint=0.0,
                 p=0.0, i=-50.0)

    rp.close()
"""

import numpy as np
from pyrpl import Pyrpl


# ---------------------------------------------------------------------------- #
# Analytic transfer functions (hardware-free; for Bode previews & notebooks).
# When connected, prefer the module's own ``transfer_function`` (it is the exact
# designed response PyRPL programs into the FPGA); these mirror that math for
# offline / mock use.
# ---------------------------------------------------------------------------- #

def inputfilter_tf(inputfilter, frequencies):
    """Complex response of a cascade of 1st-order stages (the module filter bank).

    ``inputfilter`` is a list of corner frequencies in Hz: positive = low-pass,
    negative = high-pass, 0 = bypassed. Matches PyRPL's ``inputfilter`` semantics.
    """
    f = np.asarray(frequencies, dtype=float)
    H = np.ones_like(f, dtype=complex)
    for fc in (inputfilter or []):
        if fc == 0:
            continue
        x = 1j * f / abs(fc)
        H *= (1.0 / (1.0 + x)) if fc > 0 else (x / (1.0 + x))
    return H


def pi_transfer_function(p, i, frequencies):
    """Complex response of the PI controller: ``p + i/(j f)`` (i = unity-gain Hz)."""
    f = np.asarray(frequencies, dtype=float)
    H = np.full_like(f, p, dtype=complex)
    if i:
        with np.errstate(divide="ignore"):
            H = H + i / (1j * f)
    return H


def iir_transfer_function(zeros, poles, gain, frequencies):
    """Approximate s-plane response from poles/zeros in Hz (preview only)."""
    s = 2j * np.pi * np.asarray(frequencies, dtype=float)
    num = np.full_like(s, gain, dtype=complex)
    for z in (zeros or []):
        num = num * (s - 2 * np.pi * z)
    den = np.ones_like(s, dtype=complex)
    for p in (poles or []):
        den = den * (s - 2 * np.pi * p)
    return num / den


class RedPitayaPyRPL:
    """Convenience wrapper around a PyRPL connection to one Red Pitaya."""

    def __init__(self, hostname, config="nanospheres", gui=False,
                 reloadfpga=True, **kwargs):
        """Connect to the Red Pitaya and load the PyRPL FPGA bitstream.

        Parameters
        ----------
        hostname : str
            IP address or hostname of the board, e.g. ``'192.168.1.100'``.
        config : str
            Name of the PyRPL config file (stored in the PyRPL user dir).
            Reusing the same name keeps settings between sessions.
        gui : bool
            If False (default), run headless -- no Qt window is opened, which
            is what you want from a script.
        reloadfpga : bool
            Re-upload the FPGA bitstream on connect. Leave True unless you know
            the correct bitstream is already running.
        """
        self.hostname = hostname
        self.p = Pyrpl(config=config, hostname=hostname, gui=gui,
                       reloadfpga=reloadfpga, **kwargs)
        self.rp = self.p.rp
        self.scope = self.rp.scope
        print("Connected to Red Pitaya (PyRPL). Host: " + str(hostname))

    # ------------------------------------------------------------------ #
    # Module accessors
    # ------------------------------------------------------------------ #
    def pid(self, module="pid0"):
        """Return a PID module by name ('pid0', 'pid1', 'pid2')."""
        return getattr(self.rp, module)

    def iq(self, module="iq0"):
        """Return an IQ / lock-in module by name ('iq0', 'iq1', 'iq2')."""
        return getattr(self.rp, module)

    def asg(self, module="asg0"):
        """Return an arbitrary signal generator ('asg0', 'asg1')."""
        return getattr(self.rp, module)

    # ------------------------------------------------------------------ #
    # 1) Scope
    # ------------------------------------------------------------------ #
    def acquire_scope(self, duration=0.01, input1="in1", input2="in2",
                      decimation=None, trigger_source="immediately",
                      trigger_delay=0.0, threshold=0.0, hysteresis=0.01,
                      average=True, timeout=5.0):
        """Acquire a single scope trace from both channels.

        Parameters
        ----------
        duration : float
            Total acquisition time in seconds. The scope always returns
            ``data_length`` (16384) samples, so this sets the sample rate.
            Ignored if ``decimation`` is given explicitly.
        input1, input2 : str
            Signal routed to each scope channel: 'in1', 'in2', or any DSP
            module output ('iq0', 'pid0', ...). Use this to monitor an internal
            signal such as a lock-in quadrature.
        decimation : int or None
            FPGA decimation factor (power of 2, 1..65536). If given it overrides
            ``duration``.
        trigger_source : str
            'immediately', 'ch1_positive_edge', 'ext_positive_edge', 'asg0', ...
        trigger_delay : float
            Trigger delay in seconds (0 centres the trigger in the trace).
        threshold : float
            Trigger threshold in volts (for edge triggers).
        average : bool
            Average decimated samples (reduces noise) rather than decimate-drop.
        timeout : float
            Seconds to wait for the acquisition to complete.

        Returns
        -------
        (times, ch1, ch2) : tuple of np.ndarray
            ``times`` in seconds; ``ch1`` / ``ch2`` in volts.
        """
        s = self.scope
        setup_kw = dict(input1=input1, input2=input2,
                        trigger_source=trigger_source,
                        trigger_delay=trigger_delay,
                        threshold=threshold, hysteresis=hysteresis,
                        average=average, ch1_active=True, ch2_active=True)
        if decimation is not None:
            setup_kw["decimation"] = decimation
        else:
            setup_kw["duration"] = duration
        s.setup(**setup_kw)

        data = np.asarray(s.curve(timeout=timeout))
        times = np.asarray(s.times)
        # curve() returns shape (2, N): [ch1, ch2]
        if data.ndim == 2:
            ch1, ch2 = data[0], data[1]
        else:  # single active channel fallback
            ch1, ch2 = data, None
        return times, ch1, ch2

    # ------------------------------------------------------------------ #
    # 2) PID (hardware PI controller)
    # ------------------------------------------------------------------ #
    def setup_pid(self, module="pid0", input="in1", output_direct="off",
                  setpoint=0.0, p=0.0, i=0.0, ival=0.0,
                  max_voltage=1.0, min_voltage=-1.0, inputfilter=None):
        """Configure a hardware PI controller.

        The controller computes ``output = p * err + i_integral(err)`` where
        ``err = input - setpoint`` (note PyRPL convention), clipped to
        [min_voltage, max_voltage], and routes it to ``output_direct``.

        Parameters
        ----------
        module : str
            'pid0', 'pid1' or 'pid2'.
        input : str
            Error signal source: 'in1'/'in2' or a DSP module ('iq0', ...).
        output_direct : str
            'out1', 'out2' or 'off'. Use 'off' to keep the loop open while you
            tune gains, or to route the PID output into another module instead.
        setpoint : float
            Lock setpoint in volts.
        p : float
            Proportional gain (dimensionless). Sign sets feedback polarity.
        i : float
            Integrator unity-gain crossover frequency in Hz. Sign must match p.
        ival : float
            Initial integrator value in volts (use to preset / reset the output).
        max_voltage, min_voltage : float
            Output saturation limits in volts.
        inputfilter : list of float or None
            Up to 4 first-order filter corner frequencies in Hz (positive = low
            pass, negative = high pass, 0 = bypass).

        Returns
        -------
        The configured PID module object.
        """
        pid = self.pid(module)
        kw = dict(input=input, output_direct=output_direct,
                  setpoint=setpoint, p=p, i=i, ival=ival,
                  max_voltage=max_voltage, min_voltage=min_voltage)
        if inputfilter is not None:
            kw["inputfilter"] = inputfilter
        pid.setup(**kw)
        print(f"{module}: PI lock  input={input} -> {output_direct} "
              f"(setpoint={setpoint} V, p={p}, i={i} Hz)")
        return pid

    def pause_pid(self, module="pid0"):
        """Open the loop: disconnect the PID output (keeps gains/integrator)."""
        self.pid(module).output_direct = "off"

    def reset_pid_integrator(self, module="pid0", ival=0.0):
        """Reset (or preset) the integrator state of a PID module."""
        self.pid(module).ival = ival

    # ------------------------------------------------------------------ #
    # 3) Lock-in (IQ demodulation)
    # ------------------------------------------------------------------ #
    def setup_lockin(self, module="iq0", frequency=1e3, bandwidth=100.0,
                     input="in1", phase=0.0, gain=0.0, amplitude=0.0,
                     quadrature_factor=1.0, acbandwidth=None,
                     output_direct="off", output_signal="quadrature"):
        """Configure an IQ module as a lock-in amplifier.

        The module multiplies ``input`` by an internal sine/cosine at
        ``frequency``, low-pass filters at ``bandwidth`` and exposes the
        demodulated quadrature(s). It can simultaneously emit a reference tone
        of the given ``amplitude`` on ``output_direct`` (set amplitude > 0).

        Typical uses
        ------------
        * Measurement: read amplitude/phase with ``lockin_amplitude_phase``.
        * Real-time feedback: leave ``output_signal='quadrature'`` and feed this
          module into a PID by setting that PID's ``input=module`` -- the error
          signal is then demodulated entirely in the FPGA.

        Parameters
        ----------
        module : str
            'iq0', 'iq1' or 'iq2'.
        frequency : float
            Demodulation (and reference output) frequency in Hz.
        bandwidth : float or list of float
            Low-pass corner of the demodulation filter in Hz. A list sets a
            higher-order filter (up to 4 stages).
        input : str
            Signal to demodulate ('in1'/'in2' or a DSP module).
        phase : float
            Demodulation phase in degrees.
        gain : float
            Gain applied to the demodulated signal before ``output_direct``
            (the closed-loop proportional gain when used as a controller).
        amplitude : float
            Reference output amplitude in volts (0 = no drive).
        quadrature_factor : float
            Scaling of the quadrature monitor signal (boosts the readout level).
        acbandwidth : float or None
            High-pass corner on the input before demodulation, in Hz.
        output_direct : str
            'out1', 'out2' or 'off' for the (gained) demodulated / reference out.
        output_signal : str
            What appears at ``output_direct`` and on the module monitor:
            'quadrature', 'output_direct', 'pfd', 'off', etc.

        Returns
        -------
        The configured IQ module object.
        """
        iq = self.iq(module)
        kw = dict(frequency=frequency, bandwidth=bandwidth, input=input,
                  phase=phase, gain=gain, amplitude=amplitude,
                  quadrature_factor=quadrature_factor,
                  output_direct=output_direct, output_signal=output_signal)
        if acbandwidth is not None:
            kw["acbandwidth"] = acbandwidth
        iq.setup(**kw)
        print(f"{module}: lock-in  f={frequency} Hz, bw={bandwidth} Hz, "
              f"input={input}, drive={amplitude} V -> {output_direct}")
        return iq

    def read_lockin(self, module="iq0", duration=0.01, timeout=5.0):
        """Read the demodulated quadrature time series via the scope.

        Routes the IQ module's monitor signal (its ``output_signal``, normally
        'quadrature') to a scope channel and captures one trace.

        Returns
        -------
        (times, signal, mean) : the trace, and its time average (the steady
        in-phase lock-in value at the module's current ``phase``).
        """
        times, ch1, _ = self.acquire_scope(duration=duration, input1=module,
                                            input2="off", timeout=timeout,
                                            trigger_source="immediately")
        return times, ch1, float(np.mean(ch1))

    def lockin_amplitude_phase(self, module="iq0", duration=0.01, timeout=5.0):
        """Measure lock-in amplitude R and phase theta of a steady signal.

        Reads the in-phase quadrature, then re-reads with the demodulation phase
        advanced by 90 deg to get the orthogonal quadrature, and combines them.
        The module's original phase is restored afterwards. This assumes the
        demodulated signal is stationary over the two acquisitions, so it is a
        monitoring tool, not a fast readout.

        Returns
        -------
        (R, theta) : amplitude in volts and phase in degrees.
        """
        iq = self.iq(module)
        phase0 = iq.phase
        try:
            iq.phase = phase0
            _, _, i_val = self.read_lockin(module, duration, timeout)
            iq.phase = phase0 + 90.0
            _, _, q_val = self.read_lockin(module, duration, timeout)
        finally:
            iq.phase = phase0
        R = float(np.hypot(i_val, q_val))
        theta = float(np.degrees(np.arctan2(q_val, i_val)))
        return R, theta

    # ------------------------------------------------------------------ #
    # 4) Signal generator (handy for driving / testing)
    # ------------------------------------------------------------------ #
    def setup_output(self, module="asg0", frequency=1e3, amplitude=0.1,
                     offset=0.0, waveform="sin", output_direct="out1",
                     trigger_source="immediately"):
        """Configure an arbitrary signal generator on an output channel."""
        asg = self.asg(module)
        asg.setup(frequency=frequency, amplitude=amplitude, offset=offset,
                  waveform=waveform, output_direct=output_direct,
                  trigger_source=trigger_source)
        print(f"{module}: {waveform} {frequency} Hz, {amplitude} V -> "
              f"{output_direct}")
        return asg

    # ------------------------------------------------------------------ #
    # 5) Generic routing / filtering (build arbitrary FPGA signal chains)
    # ------------------------------------------------------------------ #
    #
    # The Red Pitaya FPGA is a signal *bus*: every DSP module's output is a
    # named signal, and any module's ``input`` can be set to any of those
    # signals (or a physical input 'in1'/'in2'). You therefore build chains
    # such as  in1 -> pid0(filter+PI) -> pid1(filter) -> out1  simply by
    # pointing each module's input at the previous module. These helpers make
    # that wiring explicit and uniform across module types.

    #: DSP modules that can be wired into a chain (have input/output routing).
    DSP_MODULES = ["iq0", "iq1", "iq2", "pid0", "pid1", "pid2", "iir",
                   "asg0", "asg1"]

    def get_module(self, name):
        """Return any module by name ('iq0', 'pid1', 'iir', 'asg0', ...)."""
        return getattr(self.rp, name)

    def route(self, module, input=None, output_direct=None):
        """Wire a module's input and/or output without touching its other params.

        Parameters
        ----------
        module : str
            Any DSP module name.
        input : str or None
            New input signal ('in1', 'in2', or another module's name). The
            output of any module ``X`` is available as the signal ``'X'``.
        output_direct : str or None
            'out1', 'out2', or 'off'.
        """
        m = self.get_module(module)
        if input is not None and hasattr(m, "input"):
            m.input = input
        if output_direct is not None and hasattr(m, "output_direct"):
            m.output_direct = output_direct
        return m

    def set_filter(self, module, inputfilter):
        """Set the input filter bank of a module (pid0-2, iq0-2, iir).

        ``inputfilter`` is a list of up to 4 first-order corner frequencies in
        Hz: positive = low-pass, negative = high-pass, 0 = stage bypassed.
        e.g. ``[-100, 5e3]`` = 100 Hz high-pass then 5 kHz low-pass (a band-pass).
        """
        m = self.get_module(module)
        m.inputfilter = inputfilter
        return m

    def setup_iir(self, module="iir", zeros=None, poles=None, gain=1.0,
                  input="in1", output_direct="off", loops=None,
                  on=True, bypass=False):
        """Configure the high-order IIR filter for an arbitrary transfer function.

        Parameters
        ----------
        zeros, poles : list of complex or None
            Filter zeros / poles in Hz as complex numbers, e.g.
            ``poles=[-1e3 + 5e3j, -2e4]``. Real-axis values are real poles/zeros;
            complex values must come in conjugate pairs.
        gain : float
            Overall DC gain.
        input, output_direct : str
            Routing, as for any module.
        loops : int or None
            Number of FPGA biquad loops to allocate (sets the available order vs
            bandwidth trade-off). None lets PyRPL choose.
        on, bypass : bool
            Enable the filter / bypass it (pass-through) without unwiring it.
        """
        iir = self.get_module(module)
        kw = dict(gain=gain, input=input, output_direct=output_direct,
                  on=on, bypass=bypass)
        if zeros is not None:
            kw["zeros"] = zeros
        if poles is not None:
            kw["poles"] = poles
        if loops is not None:
            kw["loops"] = loops
        iir.setup(**kw)
        print(f"{module}: IIR  {len(zeros or [])} zeros / {len(poles or [])} "
              f"poles, gain={gain}, input={input} -> {output_direct}")
        return iir

    def routing_summary(self):
        """Return current wiring of every DSP module for inspection / display.

        Returns
        -------
        dict : ``{module_name: {'input': ..., 'output_direct': ...}}``
        """
        summary = {}
        for name in self.DSP_MODULES:
            try:
                m = self.get_module(name)
            except AttributeError:
                continue
            summary[name] = {
                "input": getattr(m, "input", None),
                "output_direct": getattr(m, "output_direct", None),
            }
        return summary

    # ------------------------------------------------------------------ #
    # 6) Transfer functions: designed (Bode) and measured (network analyzer)
    # ------------------------------------------------------------------ #
    def transfer_function(self, module, frequencies):
        """Designed complex transfer function of a module at ``frequencies`` (Hz).

        Uses the module's own ``transfer_function`` (pid/iq/iir), which already
        includes the input-filter bank and the controller/IIR response -- i.e.
        exactly what the FPGA implements. Returns a complex numpy array.
        """
        freqs = np.asarray(frequencies, dtype=float)
        return np.asarray(self.get_module(module).transfer_function(freqs))

    def network_analyzer(self, input="in1", output_direct="out1",
                         start_freq=1e3, stop_freq=1e6, points=201, rbw=100.0,
                         amplitude=0.1, acbandwidth=0.0, logscale=True,
                         avg_per_point=1):
        """Measure the complex transfer function from a drive output to any signal.

        Sweeps a sine of ``amplitude`` over [start_freq, stop_freq] on
        ``output_direct`` and demodulates ``input`` (any bus signal, including an
        internal module output) at each point. ``rbw`` is the resolution
        bandwidth (smaller = slower but lower noise).

        Returns
        -------
        (frequencies, H) : real frequency array (Hz) and complex response.

        Note
        ----
        ``input`` is the measured point; the reference is the drive. To get the
        transfer function of an arbitrary point A relative to another arbitrary
        point B, measure each against the same drive and divide the two results.
        """
        na = self.p.networkanalyzer
        na.setup(start_freq=start_freq, stop_freq=stop_freq, points=points,
                 rbw=rbw, amplitude=amplitude, input=input,
                 output_direct=output_direct, acbandwidth=acbandwidth,
                 logscale=logscale, avg_per_point=avg_per_point)
        H = np.asarray(na.curve())
        freqs = np.asarray(na.frequencies)
        return freqs, H

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def close(self):
        """Close the SSH/FPGA connection and free all modules."""
        try:
            self.rp.end_all()
        finally:
            print("Red Pitaya (PyRPL) connection closed.")


if __name__ == "__main__":
    # Minimal smoke test -- edit the hostname for your board.
    import sys

    host = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.100"
    rp = RedPitayaPyRPL(hostname=host)
    try:
        # Drive a 50 kHz tone on out1 and loop it back into in1 (cable in1<-out1)
        rp.setup_output("asg0", frequency=50e3, amplitude=0.2,
                        output_direct="out1")
        rp.setup_lockin("iq0", frequency=50e3, bandwidth=200, input="in1")
        R, theta = rp.lockin_amplitude_phase("iq0")
        print(f"Lock-in: R = {R:.4f} V, theta = {theta:.1f} deg")

        t, ch1, ch2 = rp.acquire_scope(duration=2e-3)
        print(f"Scope: {len(t)} samples, in1 mean = {np.mean(ch1):.4f} V")
    finally:
        rp.close()
