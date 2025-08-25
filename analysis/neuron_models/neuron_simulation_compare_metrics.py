import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import posit_wrapper  # Your posit conversion module; should have a function "convert"
from skimage.metrics import structural_similarity as ssim_func

# -------------------------------
# Simulation function with forced rounding
# -------------------------------
def run_simulation_with_rounding(conversion_func, sim_time=100*ms, dt_override=0.02*ms):
    # Set the base time step to a smaller value so more integration steps occur
    defaultclock.dt = dt_override

    # Define neuron parameters
    tau = 10*ms             # Membrane time constant
    V_rest = -70*mV         # Resting potential
    V_reset = -65*mV        # Reset potential after a spike
    V_thresh = -50*mV       # Spike threshold
    g_leak = 0.005*nS       # Leak conductance

    # Define the model equations for a simple LIF neuron.
    # Note: The differential equation uses "I", the input current.
    #   I is defined as a parameter (with unit amp) and then used in the equation.
    eqs = '''
    dv/dt = (-(v - V_rest) + I / g_leak) / tau : volt
    I : amp
    '''

    # Create a neuron group with one neuron using the LIF model.
    neuron = NeuronGroup(1, eqs, threshold='v>V_thresh', reset='v=V_reset', method='euler')
    neuron.v = V_rest
    neuron.I = 2*nA  # <-- This is where the input current is set.

    # Calculate the number of time steps
    dt_steps = int(sim_time / dt_override)
    time_array = np.zeros(dt_steps)
    v_array = np.zeros(dt_steps)

    # Run the simulation in short chunks.
    # After each chunk, we force the membrane voltage to be rounded to the target precision.
        # Run the simulation in short chunks.
    for i in range(dt_steps):
        run(dt_override)  # Run one time chunk using full double precision internally.
        # Force rounding on the membrane voltage:
        new_v = conversion_func(neuron.v[0] / volt)
        neuron.v = new_v * volt
        # Also force rounding on the input current (in case it is dynamic)
        new_I = conversion_func(neuron.I[0] / amp)
        neuron.I = new_I * amp
        time_array[i] = i * float(dt_override/ms)
        v_array[i] = new_v

    return time_array, v_array

# -------------------------------
# Conversion functions
# -------------------------------
def to_posit(x):
    """
    Converts a float x to posit and back, using your posit_wrapper.
    """
    return posit_wrapper.convert(float(x))

def to_float32(x):
    """
    Converts a float x to np.float32.
    """
    return np.float32(x)

# -------------------------------
# Run simulations with both conversion functions
# -------------------------------
print("Running simulation with posit rounding...")
t_posit, v_posit = run_simulation_with_rounding(to_posit, sim_time=100*ms, dt_override=0.02*ms)

print("Running simulation with float32 rounding...")
t_float32, v_float32 = run_simulation_with_rounding(to_float32, sim_time=100*ms, dt_override=0.02*ms)

# -------------------------------
# Define metrics computation: PSNR, RMSE, and SSIM
# -------------------------------
def compute_metrics(ref_signal, test_signal):
    """
    Computes RMSE, PSNR, and SSIM between two 1D voltage signals.
    The PSNR is calculated using the maximum value from the reference signal.
    """
    rmse = np.sqrt(np.mean((ref_signal - test_signal) ** 2))
    psnr = 20 * np.log10(np.max(ref_signal) / rmse) if rmse != 0 else np.inf
    ssim_value = ssim_func(ref_signal, test_signal, data_range=ref_signal.max() - ref_signal.min())
    return psnr, rmse, ssim_value

# Use the posit simulation as the reference signal
psnr, rmse, ssim_value = compute_metrics(v_posit, v_float32)

print("\nComparing Float32 vs. Posit simulation (using posit as reference):")
print("PSNR: {:.2f} dB".format(psnr))
print("RMSE: {:.6f} V".format(rmse))
print("SSIM: {:.6f}".format(ssim_value))

# -------------------------------
# Plot the simulation traces for comparison
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_posit, v_posit, label="Posit Simulation")
plt.plot(t_float32, v_float32, label="Float32 Simulation", linestyle='--')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Voltage (V)")
plt.legend()
plt.title("Neuron Simulation: Posit vs. Float32 (Round-after-each-step)")
plt.show()
