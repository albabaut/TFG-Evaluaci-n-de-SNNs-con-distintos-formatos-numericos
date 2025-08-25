from brian2 import *
import numpy as np
import pandas as pd
from pathlib import Path

# Crear carpetas de salida
Path("metrics_compare").mkdir(exist_ok=True)
Path("images_compare").mkdir(exist_ok=True)

def run_pulse_detection_simulation(dtype_label, convert_func, n=50):
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

    # Generar estímulo pulsado (pico en [50, 55] ms)
    pulse = np.zeros(int(duration/dt))
    pulse[500:550] = 2.0  # 2 nA durante 5 ms
    I_input = TimedArray(pulse * nA, dt=dt)

    eqs = f'''
    dv/dt = (I_input(t)*{R}*Mohm - (v - {V_rest}*mV)) / ({tau}*ms) : volt
    '''

    G = NeuronGroup(n, eqs,
                    threshold=f'v > {V_th}*mV',
                    reset=f'v = {V_reset}*mV',
                    method='euler')
    G.v = V_rest * mV

    # Sinapsis recurrentes para amplificación de errores
    S = Synapses(G, G, on_pre=f'v_post += {delta_v}*mV')
    S.connect(p=0.1)

    spikes = SpikeMonitor(G)
    voltages = StateMonitor(G, 'v', record=True)

    net = Network(G, S, spikes, voltages)
    net.run(duration)

    return spikes, voltages

def extract_pulse_response_metrics(spikes, label, t_start=50, t_end=55):
    t_start *= ms
    t_end *= ms
    result = []

    for i in range(len(spikes.count)):
        neuron_spikes = spikes.t[spikes.i == i]
        spike_count = np.sum((neuron_spikes >= t_start) & (neuron_spikes <= t_end))
        result.append({
            'neuron': i,
            'label': label,
            'pulse_spikes': spike_count,
            'total_spikes': spikes.count[i]
        })

    return pd.DataFrame(result)

# Ejecutar simulaciones
print("Running float16 with pulse input...")
float16_spikes, _ = run_pulse_detection_simulation("float16", lambda x: np.float16(x))

print("Running posit16 with pulse input...")
import posit_wrapper
posit16_spikes, _ = run_pulse_detection_simulation("posit16", posit_wrapper.convert16)

# Extraer métricas
df_f16 = extract_pulse_response_metrics(float16_spikes, "float16")
df_pos = extract_pulse_response_metrics(posit16_spikes, "posit16")
df_pulse = pd.concat([df_f16, df_pos], ignore_index=True)

# Guardar
df_pulse.to_csv("metrics_compare/pulse_detection_compare.csv", index=False)
df_pulse.head()
