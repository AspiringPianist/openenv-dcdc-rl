import numpy as np
import matplotlib.pyplot as plt
from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from PySpice.Unit import *
from PySpice.Logging.Logging import setup_logging

setup_logging()

class BasicOperationalAmplifier(SubCircuitFactory):
    NAME = 'BasicOperationalAmplifier'
    NODES = ('non_inverting_input', 'inverting_input', 'output')

    def __init__(self):
        super().__init__()
        # Input impedance
        self.R('input', 'non_inverting_input', 'inverting_input', 10@u_MΩ)

        # Match LTspice UniversalOpAmp2: Avol=100, GBW=35Meg
        # pole1 = GBW / Avol = 350 kHz
        self.VCVS('gain', 1, self.gnd, 'non_inverting_input', 'inverting_input', voltage_gain=100)
        self.R('P1', 1, 2, 1@u_kΩ)
        # C = 1 / (2 * pi * 1k * 350kHz) = 454.7 pF
        self.C('P1', 2, self.gnd, 454.7@u_pF)

        # Output buffer and resistance
        self.VCVS('buffer', 3, self.gnd, 2, self.gnd, 1)
        self.R('out', 3, 'output', 10@u_Ω)


class BasicComparator(SubCircuitFactory):
    NAME = 'BasicComparator'
    NODES = ('non_inverting_input', 'inverting_input', 'voltage_plus', 'voltage_minus', 'output')

    def __init__(self):
        super().__init__()
        self.R(1, 'voltage_plus', 'voltage_minus', 1@u_MΩ)
        # Using a continuous behavioral source
        self.B(1, 'output', 'voltage_minus', 
               v='V(non_inverting_input) - V(inverting_input) < -0.001 ? 0 : (V(non_inverting_input) - V(inverting_input) > 0.001 ? 1.8 : 900 * (V(non_inverting_input) - V(inverting_input) + 0.001))')


# Construct the PySpice Circuit
circuit = Circuit('Buck Converter RL Environment')
circuit.subcircuit(BasicOperationalAmplifier())
circuit.subcircuit(BasicComparator())

# Power Supply
circuit.V('dd', 'vdd', circuit.gnd, 1.8@u_V)

# Reference Voltage
circuit.V('ref', 'n_ref', circuit.gnd, 0.6@u_V)

# PWM Sawtooth (Sweeps from 1V to 1.8V over 29.99ns, period 30ns)
circuit.PulseVoltageSource('sawtooth', 'n_saw', circuit.gnd,
                           initial_value=1@u_V, pulsed_value=1.8@u_V,
                           delay_time=0@u_ns, rise_time=29.99@u_ns, fall_time=0.01@u_ns,
                           pulse_width=0.01@u_ns, period=30@u_ns)

# Feedback Divider
circuit.R(2, 'vout', 'n_fb', 10@u_kΩ)
circuit.R(3, 'n_fb', circuit.gnd, 10@u_kΩ)

# Error Amplifier
circuit.X('amp', 'BasicOperationalAmplifier', 'n_ref', 'n_fb', 'n_err')

# Compensation Network (Type-II)
circuit.C(3, 'n_fb', 'n_err', 1@u_pF)
circuit.R(4, 'n_fb', 'n_comp_mid', 4@u_kΩ)
circuit.C(2, 'n_comp_mid', 'n_err', 4@u_nF)

# PWM Comparator
circuit.X('comp', 'BasicComparator', 'n_err', 'n_saw', 'vdd', circuit.gnd, 'n_pwm')

# Gate Drivers (Behavioral logic inversion)
circuit.B('gate_hs', 'n_hs_g', circuit.gnd, v='1.8 - V(n_pwm)')
# Fix: NMOS also needs to be logically inverted from PWM (so they form a CMOS inverter)!
# Otherwise both PMOS and NMOS turn on simultaneously (shoot-through) causing massive voltage drops
circuit.B('gate_ls', 'n_ls_g', circuit.gnd, v='1.8 - V(n_pwm)')

# Power MOSFET Models 
# Typical 0.18u CMOS KP is around 100uA/V^2 (100e-6), not 100mA/V^2
# At W=40000um L=0.18um, this gives an RDS(on) of ~10mOhm
circuit.model('PMOS_SW', 'PMOS', level=1, kp=100e-6, vto=-0.4)
circuit.model('NMOS_SW', 'NMOS', level=1, kp=100e-6, vto=0.4)

# Power Stage Switches
circuit.MOSFET(9, 'n_sw', 'n_hs_g', 'vdd', 'vdd', model='PMOS_SW', l=180@u_nm, w=40000@u_um)
circuit.MOSFET(10, 'n_sw', 'n_ls_g', circuit.gnd, circuit.gnd, model='NMOS_SW', l=180@u_nm, w=20000@u_um)

# LC Filter (with ESR from LTspice image)
# Inductor: 47nH, DCR = 0.093 Ohm
circuit.L(1, 'n_sw', 'n_ind_out', 47@u_nH)
circuit.R('L_ESR', 'n_ind_out', 'vout', 0.093@u_Ω)

# Capacitor: 0.068uF, ESR = 0.0171 Ohm
circuit.C(1, 'vout', 'n_cap_esr', 0.068@u_uF)
circuit.R('C_ESR', 'n_cap_esr', circuit.gnd, 0.0171@u_Ω)

# Load Resistor
circuit.R(1, 'vout', circuit.gnd, 4@u_Ω)

# Setting up simulator
print("Starting transient simulation...")
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
# Set options to help convergence
simulator.options(cshunt=1e-15)
# Add use_initial_condition=True to bypass DC operating point calculation which is failing to converge
# Let's give it slightly more time (200us) to fully settle the LC tank
analysis = simulator.transient(step_time=0.1@u_ns, end_time=200@u_us, use_initial_condition=True)
print("Simulation complete!")

# Extract results directly as NumPy arrays (Perfect for RL reward calculation)
time = np.array(analysis.time)
vout = np.array(analysis.vout)

# Analyze Steady State (T > 150us)
steady_state_mask = time > 150e-6
vout_steady = vout[steady_state_mask]
time_steady = time[steady_state_mask]

# RL Metrics extraction
v_avg = np.mean(vout_steady)
v_ripple_pk2pk = np.max(vout_steady) - np.min(vout_steady)

print("-" * 30)
print(f"Average Vout:     {v_avg:.4f} V")
print(f"Vout Ripple PkPk: {v_ripple_pk2pk*1000:.2f} mV")
print(f"Settling Target:  1.200 V")
print("-" * 30)

# Matplotlib visualization
plt.figure(figsize=(10, 6))
plt.plot(time * 1e6, vout, label='Vout')
plt.axhline(y=1.2, color='r', linestyle='--', label='Target (1.2V)')
plt.title('Buck Converter PySpice Output')
plt.xlabel('Time (us)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('pyspice_buck_output.png')
print("Plot saved to pyspice_buck_output.png")