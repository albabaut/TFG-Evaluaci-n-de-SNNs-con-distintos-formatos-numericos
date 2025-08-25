from brian2 import *
import numpy as np
import pandas as pd
from pathlib import Path

# Crear carpetas de salida
Path("metrics_compare").mkdir(exist_ok=True)
Path("images_compare").mkdir(exist_ok=True)

def run_precision_simulation_connected(dtype_label, convert_func, I_mean=1.0, noise_std=0.1, tau_val=10.0, n=100, w_val=0.5):
    duration = 200 * ms
    dt = 0.1 * ms
    defaultclock.dt = dt

    V_rest_val = -65.0  # mV
    V_th_val = -50.0    # mV
    V_reset_val = -65.0 # mV
    R_val = 100.0       # MOhm

    # Convertir parámetros al formato deseado
    V_rest = float(convert_func(V_rest_val))
    V_th = float(convert_func(V_th_val))
    V_reset = float(convert_func(V_reset_val))
    R = float(convert_func(R_val))
    tau = float(convert_func(tau_val))
    delta_v = float(convert_func(w_val))

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

def compare_to_float64(reference_vm, reference_spikes, test_vm, test_spikes, label):
    from brian2 import ms, mV
    n_neurons = len(reference_vm.v)
    results = []

    for i in range(n_neurons):
        v_ref = reference_vm.v[i] / mV
        v_test = test_vm.v[i] / mV
        s_ref = reference_spikes.t[reference_spikes.i == i] / ms
        s_test = test_spikes.t[test_spikes.i == i] / ms

        rmse = np.sqrt(np.mean((v_ref - v_test) ** 2))
        acc_err = np.sum(np.abs(v_ref - v_test))
        delta_spikes = len(s_test) - len(s_ref)

        tolerance = 1.0  # ms
        tp = 0
        fn = 0
        for t_gt in s_ref:
            match = np.any(np.abs(s_test - t_gt) <= tolerance)
            if match:
                tp += 1
            else:
                fn += 1
        fp = len(s_test) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            'neuron': i,
            'label': label,
            'delta_spikes': delta_spikes,
            'rmse': rmse,
            'accumulated_error': acc_err,
            'spikes_ref': len(s_ref),
            'spikes_test': len(s_test),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

    return pd.DataFrame(results)

# Ejecutar simulaciones
print("Running float64 (ground truth)...")
ref_spikes, ref_vm = run_precision_simulation_connected('float64', lambda x: np.float64(x))

print("Running float16...")
f16_spikes, f16_vm = run_precision_simulation_connected('float16', lambda x: np.float16(x))

print("Running posit16...")
import posit_wrapper
posit_spikes, posit_vm = run_precision_simulation_connected('posit16', posit_wrapper.convert16)

# Comparar con float64
df_f16 = compare_to_float64(ref_vm, ref_spikes, f16_vm, f16_spikes, "float16")
df_pos = compare_to_float64(ref_vm, ref_spikes, posit_vm, posit_spikes, "posit16")

# Guardar resultados
df_all = pd.concat([df_f16, df_pos], ignore_index=True)
df_all.to_csv("metrics_compare/compare_to_float64.csv", index=False)
df_all.head()
