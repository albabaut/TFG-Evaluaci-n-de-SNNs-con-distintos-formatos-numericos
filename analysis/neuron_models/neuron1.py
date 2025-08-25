# posit16/1_neuron/parameter_sweep_1_neuron.py

import posit_wrapper
from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# 1) Carpeta base junto al script
# --------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))
base = here  # ya estamos en posit16/1_neuron
metrics_dir = os.path.join(base, 'sweep_metrics')
images_dir  = os.path.join(base, 'sweep_plots')
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(images_dir,  exist_ok=True)

# --------------------------------------------------
# 2) Conversión específica
# --------------------------------------------------
conv = lambda x: posit_wrapper.convert16(x)

# --------------------------------------------------
# 3) Parámetros fijos de la neurona
# --------------------------------------------------
V_rest     = -70 * mV
V_reset    = -60 * mV
default_dt = 0.1 * ms
duration   = 200 * ms

# --------------------------------------------------
# 4) Rangos de barrido
# --------------------------------------------------
I_vals        = np.linspace(0.5, 4.0, 8)    # nA
tau_vals      = [5, 10, 20, 50]            # ms
V_thresh_vals = [-55, -50, -45, -40]       # mV
g_leak_vals   = [0.5, 1.0, 2.0, 5.0]        # nS

records = []

# --------------------------------------------------
# 5) Bucle de simulaciones
# --------------------------------------------------
for I_val in I_vals:
    for tau_ms in tau_vals:
        for V_th in V_thresh_vals:
            for g_leak_nS in g_leak_vals:
                # Unidades Brian2
                tau      = conv(tau_ms)    * ms
                V_th_u   = conv(V_th)      * mV
                g_leak   = conv(g_leak_nS) * nS
                I_input  = conv(I_val)     * nA

                eqs = '''
                dv/dt = (-(V_rest - v) + I_input/g_leak) / tau : volt
                '''

                net = Network()
                neuron = NeuronGroup(1, eqs,
                                     threshold='v > V_th_u',
                                     reset='v = V_reset',
                                     method='euler')
                neuron.v = V_rest
                net.add(neuron)

                spikes = SpikeMonitor(neuron)
                net.add(spikes)
                net.run(duration)

                n_spikes = int(spikes.count[0])
                t_first  = float(spikes.t[0]/ms) if spikes.count[0] > 0 else np.nan

                records.append({
                    'I_nA':       I_val,
                    'tau_ms':     tau_ms,
                    'V_th_mV':    V_th,
                    'g_leak_nS':  g_leak_nS,
                    'n_spikes':   n_spikes,
                    't_first_ms': t_first
                })

# --------------------------------------------------
# 6) Guardar CSV
# --------------------------------------------------
df = pd.DataFrame(records)
csv_path = os.path.join(metrics_dir, 'parameter_sweep_posit16.csv')
df.to_csv(csv_path, index=False)

# --------------------------------------------------
# 7) Graficar número de spikes vs parámetro
# --------------------------------------------------
# Valores intermedios para fijar
mid_tau = tau_vals[len(tau_vals)//2]
mid_Vth = V_thresh_vals[len(V_thresh_vals)//2]
mid_g   = g_leak_vals[len(g_leak_vals)//2]
mid_I   = I_vals[len(I_vals)//2]

def plot_vs(param, fixed):
    sub = df
    for k, v in fixed.items():
        sub = sub[sub[k] == v]
    plt.figure()
    plt.plot(sub[param], sub['n_spikes'], 'o-')
    plt.xlabel(param)
    plt.ylabel('n_spikes')
    plt.title(f'n_spikes vs {param} (fijos: {fixed})')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'n_spikes_vs_{param}.png'))
    plt.close()

plot_vs('I_nA',      {'tau_ms': mid_tau, 'V_th_mV': mid_Vth, 'g_leak_nS': mid_g})
plot_vs('tau_ms',    {'I_nA': mid_I,    'V_th_mV': mid_Vth, 'g_leak_nS': mid_g})
plot_vs('V_th_mV',   {'I_nA': mid_I,    'tau_ms': mid_tau,  'g_leak_nS': mid_g})
plot_vs('g_leak_nS', {'I_nA': mid_I,    'tau_ms': mid_tau,  'V_th_mV': mid_Vth})
