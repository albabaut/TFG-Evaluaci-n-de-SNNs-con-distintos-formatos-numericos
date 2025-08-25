from brian2 import *
import numpy as np
import pandas as pd
from scipy.stats import entropy
from pathlib import Path

# Crear carpetas
Path("metrics_compare").mkdir(exist_ok=True)
Path("images_compare").mkdir(exist_ok=True)

def run_individual_noise_simulation(dtype_label, convert_func, n=100, I_mean=1.0, noise_std=0.5):
    duration = 200 * ms
    dt = 0.1 * ms
    defaultclock.dt = dt

    # Parámetros base
    V_rest = float(convert_func(-65.0))
    V_th = float(convert_func(-50.0))
    V_reset = float(convert_func(-65.0))
    R = float(convert_func(100.0))  # MOhm
    tau = float(convert_func(10.0))  # ms
    delta_v = float(convert_func(0.5))  # mV

    # Corrientes ruidosas independientes por neurona
    steps = int(duration / dt)
    noise_matrix = I_mean + noise_std * np.random.randn(n, steps)
    noise_matrix = np.clip(noise_matrix, 0, None)  # evitar valores negativos
    I_input = TimedArray(noise_matrix.T * nA, dt=dt)  # columnas: neuronas

    eqs = f'''
    dv/dt = (I_input(t, i)*{R}*Mohm - (v - {V_rest}*mV)) / ({tau}*ms) : volt
    '''

    G = NeuronGroup(n, eqs,
                    threshold=f'v > {V_th}*mV',
                    reset=f'v = {V_reset}*mV',
                    method='euler')
    G.v = V_rest * mV

    S = Synapses(G, G, on_pre=f'v_post += {delta_v}*mV')
    S.connect(p=0.05)

    spikes = SpikeMonitor(G)
    voltages = StateMonitor(G, 'v', record=True)

    net = Network(G, S, spikes, voltages)
    net.run(duration)

    return spikes, voltages

def summarize_individual_noise(spikes, voltages, label):
    results = []
    for i in range(len(spikes.count)):
        v_trace = voltages.v[i] / mV
        spike_count = spikes.count[i]

        # Histogram for entropy
        bins = np.histogram_bin_edges(v_trace, bins='auto')
        hist, _ = np.histogram(v_trace, bins=bins)
        ent = entropy(hist + 1)  # add 1 to avoid log(0)

        results.append({
            'neuron': i,
            'label': label,
            'total_spikes': spike_count,
            'mean_voltage': np.mean(v_trace),
            'voltage_entropy': ent
        })

    return pd.DataFrame(results)

# Simulación con float16
print("Simulating float16 with individual noisy inputs...")
f16_spikes, f16_vm = run_individual_noise_simulation("float16", lambda x: np.float16(x))
df_f16 = summarize_individual_noise(f16_spikes, f16_vm, "float16")

# Simulación con posit16
print("Simulating posit16 with individual noisy inputs...")
import posit_wrapper
p16_spikes, p16_vm = run_individual_noise_simulation("posit16", posit_wrapper.convert16)
df_p16 = summarize_individual_noise(p16_spikes, p16_vm, "posit16")

# Guardar resultados
df_all = pd.concat([df_f16, df_p16], ignore_index=True)
df_all.to_csv("metrics_compare/noise_individual_inputs.csv", index=False)
df_all.head()
