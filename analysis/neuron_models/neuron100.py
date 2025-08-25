import posit_wrapper
from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import entropy

# 1) Carpetas de salida
here = os.path.dirname(os.path.abspath(__file__))
metrics_dir = os.path.join(here, 'metrics')
images_dir  = os.path.join(here, 'images')
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(images_dir,  exist_ok=True)

# 2) Conversión y parámetros
conv  = lambda x: posit_wrapper.convert16(x)
n     = 100
duration = 100 * ms
dt       = 0.1 * ms
defaultclock.dt = dt

V_rest      = -70 * mV
V_reset     = -60 * mV
V_threshold = -50 * mV
g_leak      = conv(1.0) * nS
tau         = conv(10.0) * ms

# 3) Barrido de intensidades
I_vals = np.arange(0, 4.0 + 1e-9, 0.1)  # 0, 0.1, 0.2, … ,4.0
results = []

for I_val in I_vals:
    # TimedArray de corriente
    raw = conv(I_val) * np.ones(int(duration/dt))
    currents = raw * nA
    I_in = TimedArray(currents, dt=dt)

    # Ecuación
    eqs = '''
    dv/dt = (-(V_rest - v) + I_in(t)/g_leak) / tau : volt
    '''

    G = NeuronGroup(n, eqs,
                    threshold='v > V_threshold',
                    reset='v = V_reset',
                    method='euler')
    G.v = V_rest

    M = StateMonitor(G, 'v', record=True)
    spikes = SpikeMonitor(G)

    # Conexiones aleatorias
    delta_v = conv(0.2) * mV
    dv_num = float(delta_v / mV)
    S = Synapses(G, G, on_pre=f'v_post += {dv_num}*mV')
    S.connect(p=0.05)

    # Asegúrate de incluir S en el Network
    net = Network(G, M, spikes, S)
    net.run(duration)

    # Métricas
    volt_mV = (M.v / mV)
    mean_volt = np.mean(volt_mV, axis=1)
    rmse = np.sqrt(np.mean((mean_volt - I_val)**2))
    spike_counts = np.array([np.sum(spikes.i == i) for i in range(n)])
    ent = entropy(spike_counts + 1)
    acc_err = np.sum(np.abs(mean_volt - I_val))
    total_spikes = int(spike_counts.sum())

    results.append({
        'intensity_nA':    I_val,
        'rmse':            rmse,
        'entropy':         ent,
        'accumulated_err': acc_err,
        'total_spikes':    total_spikes
    })

# 4) Guarda CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(metrics_dir, 'metrics_posit16.csv'), index=False)

# 5) Guarda gráficos en ./images
for metric in ['rmse', 'entropy', 'accumulated_err', 'total_spikes']:
    plt.figure()
    plt.plot(df['intensity_nA'], df[metric], marker='o')
    plt.xlabel('Intensidad (nA)')
    plt.ylabel(metric)
    plt.title(f'{metric} vs intensidad – posit16')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{metric}_posit16.png'))
    plt.close()
