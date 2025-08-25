from brian2 import *
import numpy as np
import pandas as pd
from pathlib import Path

# Crear carpetas
Path("metrics_compare").mkdir(exist_ok=True)

def run_multilayer_with_ground_truth(dtype_label, convert_func, n_input=100, n_output=50, I_mean=1.0, noise_std=0.5):
    duration = 200 * ms
    dt = 0.1 * ms
    defaultclock.dt = dt

    V_rest = float(convert_func(-65.0))
    V_th = float(convert_func(-50.0))
    V_reset = float(convert_func(-65.0))
    R = float(convert_func(100.0))  # MOhm
    tau = float(convert_func(10.0))  # ms
    delta_v = float(convert_func(0.5))  # mV

    steps = int(duration / dt)
    noise_matrix = I_mean + noise_std * np.random.randn(n_input, steps)
    noise_matrix = np.clip(noise_matrix, 0, None)
    I_input = TimedArray(noise_matrix.T * nA, dt=dt)

    eqs_input = f'''
    dv/dt = (I_input(t, i)*{R}*Mohm - (v - {V_rest}*mV)) / ({tau}*ms) : volt
    '''
    G_input = NeuronGroup(n_input, eqs_input,
                          threshold=f'v > {V_th}*mV',
                          reset=f'v = {V_reset}*mV',
                          method='euler')
    G_input.v = V_rest * mV

    eqs_output = f'''
    dv/dt = (- (v - {V_rest}*mV)) / ({tau}*ms) : volt
    '''
    G_output = NeuronGroup(n_output, eqs_output,
                           threshold=f'v > {V_th}*mV',
                           reset=f'v = {V_reset}*mV',
                           method='euler')
    G_output.v = V_rest * mV

    S = Synapses(G_input, G_output, on_pre=f'v_post += {delta_v}*mV')
    S.connect(p=0.1)

    spikes_out = SpikeMonitor(G_output)
    net = Network(G_input, G_output, S, spikes_out)
    net.run(duration)

    return spikes_out

def collect_output_spikes(spikes_out, label):
    return pd.DataFrame({
        'neuron': np.arange(len(spikes_out.count)),
        'label': label,
        'output_spikes': spikes_out.count
    })

# Ejecutar simulaciones para comparación contra float64
print("Simulating float64 as ground truth...")
ref_spikes = run_multilayer_with_ground_truth("float64", lambda x: np.float64(x))
df_ref = collect_output_spikes(ref_spikes, "float64")

print("Simulating float16...")
f16_spikes = run_multilayer_with_ground_truth("float16", lambda x: np.float16(x))
df_f16 = collect_output_spikes(f16_spikes, "float16")

print("Simulating posit16...")
import posit_wrapper
p16_spikes = run_multilayer_with_ground_truth("posit16", posit_wrapper.convert16)
df_p16 = collect_output_spikes(p16_spikes, "posit16")

# Guardar resultados combinados
df_all = pd.concat([df_ref, df_f16, df_p16], ignore_index=True)
df_all.to_csv("metrics_compare/multilayer_compare_float64.csv", index=False)
df_all.head()
