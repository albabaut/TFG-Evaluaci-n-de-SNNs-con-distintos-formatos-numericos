from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path

# Crear carpetas si no existen
Path("metrics_compare").mkdir(exist_ok=True)
Path("images_compare").mkdir(exist_ok=True)

def run_precision_simulation_connected(dtype, I_mean=1.0, noise_std=0.1, tau_val=10.0, n=100, w_val=0.5):
    duration = 200 * ms
    dt = 0.1 * ms
    defaultclock.dt = dt

    V_rest_val = -65.0  # mV
    V_th_val = -50.0    # mV
    V_reset_val = -65.0 # mV
    R_val = 100.0       # MOhm

    if dtype == 'posit16':
        import posit_wrapper
        convert = posit_wrapper.convert16
    else:
        convert = np.float16

    V_rest = float(convert(V_rest_val))
    V_th = float(convert(V_th_val))
    V_reset = float(convert(V_reset_val))
    R = float(convert(R_val))
    tau = float(convert(tau_val))
    delta_v = float(convert(w_val))

    eqs = '''
    dv/dt = (I*R*Mohm - (v - V_rest*mV)) / (tau*ms) : volt
    I : amp
    '''

    G = NeuronGroup(n, eqs,
                    threshold=f'v > {V_th}*mV',
                    reset=f'v = {V_reset}*mV',
                    method='euler')
    G.v = V_rest * mV
    G.I = (I_mean + noise_std * np.random.randn(n)) * nA

    S = Synapses(G, G, on_pre=f'v_post += {delta_v}*mV')
    S.connect(p=0.05)

    spikes = SpikeMonitor(G)
    M = StateMonitor(G, 'v', record=True)

    net = Network(G, S, spikes, M)
    net.run(duration)

    return spikes, M

def analyze_voltage_spike_differences(vm_f16, vm_pos, spikes_f16, spikes_pos):
    from brian2 import ms, mV
    n_neurons = len(vm_f16.v)
    times = vm_f16.t / ms
    results = []

    for i in range(n_neurons):
        v_f16 = vm_f16.v[i] / mV
        v_pos = vm_pos.v[i] / mV
        s_f16 = spikes_f16.t[spikes_f16.i == i] / ms
        s_pos = spikes_pos.t[spikes_pos.i == i] / ms

        rmse = np.sqrt(np.mean((v_f16 - v_pos) ** 2))
        acc_err = np.sum(np.abs(v_f16 - v_pos))
        delta_spikes = len(s_f16) - len(s_pos)

        tolerance = 1.0  # ms
        tp = 0
        fn = 0
        matched = set()
        for t_gt in s_f16:
            match = np.any(np.abs(s_pos - t_gt) <= tolerance)
            if match:
                tp += 1
                matched.add(t_gt)
            else:
                fn += 1
        fp = len(s_pos) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            'neuron': i,
            'delta_spikes': delta_spikes,
            'rmse': rmse,
            'accumulated_error': acc_err,
            'spikes_f16': len(s_f16),
            'spikes_pos': len(s_pos),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

    df = pd.DataFrame(results)

    # Visualizaciones
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df[['rmse', 'accumulated_error', 'delta_spikes']])
    plt.title("Distribución de errores por neurona")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images_compare/error_distributions_boxplot.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(df['rmse'], df['delta_spikes'], alpha=0.7)
    plt.xlabel("RMSE voltaje (mV)")
    plt.ylabel("Diferencia de spikes (float16 - posit16)")
    plt.title("Relación entre RMSE y diferencia de spikes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images_compare/rmse_vs_deltaspikes.png")
    plt.close()

    df.to_csv("metrics_compare/neuron_metrics.csv", index=False)
    return df

# Ejecución principal
print("Running float16 simulation...")
spikes_f16, vm_f16 = run_precision_simulation_connected('float16', I_mean=1.0)
print("Running posit16 simulation...")
spikes_pos, vm_pos = run_precision_simulation_connected('posit16', I_mean=1.0)

# Análisis
df = analyze_voltage_spike_differences(vm_f16, vm_pos, spikes_f16, spikes_pos)
