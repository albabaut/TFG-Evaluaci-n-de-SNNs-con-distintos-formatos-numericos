# posit16/1_neuron/sweep_2d_I_tau.py

import posit_wrapper
from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1) Rutas relativas
here = os.path.dirname(os.path.abspath(__file__))
base = here  # posit16/1_neuron
metrics_dir = os.path.join(base, 'sweep_metrics', '2d_I_tau')
images_dir  = os.path.join(base, 'sweep_plots',  '2d_I_tau')
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(images_dir,  exist_ok=True)

# 2) Conversión y parámetros fijos
conv = lambda x: posit_wrapper.convert16(x)
V_rest      = -70 * mV
V_reset     = -60 * mV
V_threshold = -50 * mV
g_leak      = conv(1.0) * nS
duration    = 200 * ms
defaultclock.dt = 0.1 * ms

# 3) Rangos para I y tau
I_vals   = np.linspace(0.5, 4.0, 8)   # nA
tau_vals = [5, 10, 20, 50]            # ms

# Prealoca DataFrame
records = []

# 4) Doble bucle: I vs tau
for I_val in I_vals:
    for tau_ms in tau_vals:
        # Construye la corriente y tau
        I_in = TimedArray((conv(I_val) * np.ones(int(duration/defaultclock.dt))) * nA,
                          dt=defaultclock.dt)
        tau   = conv(tau_ms) * ms

        eqs = '''
        dv/dt = (-(V_rest - v) + I_in(t)/g_leak) / tau : volt
        '''
        G = NeuronGroup(1, eqs,
                        threshold='v > V_threshold',
                        reset='v = V_reset',
                        method='euler')
        G.v = V_rest
        spikes = SpikeMonitor(G)
        run(duration)

        # Guarda número de spikes
        n_spikes = int(spikes.count[0])
        records.append({'I_nA': I_val, 'tau_ms': tau_ms, 'n_spikes': n_spikes})

# 5) DataFrame y CSV
df = pd.DataFrame(records)
csv_path = os.path.join(metrics_dir, 'metrics_2d_I_tau.csv')
df.to_csv(csv_path, index=False)

# 6) Heatmap de n_spikes (I vs tau)
pivot = df.pivot(index='tau_ms', columns='I_nA', values='n_spikes')
plt.figure(figsize=(6,4))
plt.imshow(pivot, origin='lower', aspect='auto',
           extent=[I_vals.min(), I_vals.max(), tau_vals[0], tau_vals[-1]])
plt.colorbar(label='n_spikes')
plt.xlabel('I (nA)')
plt.ylabel('tau (ms)')
plt.title('Heatmap n_spikes: I vs tau')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'heatmap_I_vs_tau.png'))
plt.close()
