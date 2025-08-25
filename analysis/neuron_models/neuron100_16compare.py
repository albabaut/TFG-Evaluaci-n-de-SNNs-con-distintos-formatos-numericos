import posit_wrapper
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from analyze_utils import analyze_voltage_spike_differences  # si la función está en otro archivo .py


# Cambiar precisión por defecto (opcional, para observar más diferencias)
prefs.core.default_float_dtype = np.float32

def run_precision_simulation(dtype, I_mean=1.0, noise_std=0.1, tau=10.0):
    """Ejecuta una simulación con precisión numérica controlada (float16 o posit16)"""
    duration = 200 * ms
    dt = 0.1 * ms
    defaultclock.dt = dt

    # Parámetros base
    V_rest_val = -65.0  # mV
    V_th_val   = -50.0  # mV
    V_reset_val = -65.0 # mV
    R_val = 100.0       # MOhm
    tau_val = tau       # ms

    # Conversión de tipos
    if dtype == 'posit16':
        convert = posit_wrapper.convert16
    else:
        convert = np.float16

    # Aplica conversión a parámetros
    V_rest = float(convert(V_rest_val))
    V_th   = float(convert(V_th_val))
    V_reset = float(convert(V_reset_val))
    R      = float(convert(R_val))
    tau    = float(convert(tau_val))

    # Generación de entrada ruidosa
    n_steps = int(duration / dt)
    I_vals = I_mean + noise_std * np.random.randn(n_steps)
    I_input = TimedArray(I_vals * nA, dt=dt)

    # Ecuaciones con unidades consistentes
    # Multiplicamos R por Mohm para que tenga unidad de resistencia (ohm)
    # El producto I_input(t)*R tendrá unidades de voltios como se espera
    eqs = '''
    dv/dt = (I*R*Mohm - (v - V_rest*mV)) / (tau*ms) : volt
    I : amp
    '''


    

    G = NeuronGroup(100, eqs,
                    threshold=f'v > {V_th}*mV',
                    reset=f'v = {V_reset}*mV',
                    method='euler')
    G.v = V_rest * mV
    G.I = (I_mean + noise_std * np.random.randn(n)) * nA

    # Monitores
    spikes = SpikeMonitor(G)
    M = StateMonitor(G, 'v', record=True)

    net = Network(G, spikes, M)
    net.run(duration)

    return spikes, M

# Ejecutar simulaciones
print("Running float16 simulation...")
spikes_f16, vm_f16 = run_precision_simulation('float16', I_mean=1.0)
print("Running posit16 simulation...")
spikes_pos, vm_pos = run_precision_simulation('posit16', I_mean=1.0)
spikes_f16, vm_f16 = run_precision_simulation('float16', I_mean=1.0)
spikes_pos, vm_pos = run_precision_simulation('posit16', I_mean=1.0)

df = analyze_voltage_spike_differences(vm_f16, vm_pos, spikes_f16, spikes_pos)
# Visualización
plt.figure(figsize=(12, 8))
gs = GridSpec(3, 1, height_ratios=[1, 1, 2])

# Raster plot
ax0 = plt.subplot(gs[0])
ax0.scatter(spikes_f16.t/ms, spikes_f16.i, c='red', s=15, label='Float16')
ax0.scatter(spikes_pos.t/ms, spikes_pos.i+0.2, c='blue', s=15, label='Posit16', alpha=0.7)
ax0.set_ylabel('Neuron ID')
ax0.set_title('Spike Timing Comparison')
ax0.legend()

# Voltage trace
ax1 = plt.subplot(gs[1])
ax1.plot(vm_f16.t/ms, vm_f16.v[0]/mV, 'r-', label='Float16', alpha=0.8)
ax1.plot(vm_pos.t/ms, vm_pos.v[0]/mV, 'b--', label='Posit16', alpha=0.8)
ax1.set_ylabel('Voltage (mV)')
ax1.legend()

# Diferencia de voltaje
ax2 = plt.subplot(gs[2])
time_points = vm_f16.t[::5]/ms
diff = [(float(vm_f16.v[0][i]/mV) - float(vm_pos.v[0][i]/mV)) for i in range(0, len(vm_f16.t), 5)]
ax2.plot(time_points, diff, 'k-', label='Float16 - Posit16')
ax2.axhline(0, color='gray', linestyle=':')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Voltage Difference (mV)')
ax2.legend()

plt.tight_layout()
plt.savefig('precision_comparison_10.png', dpi=300)
plt.show()

