# posit16/1_neuron/sweep_noise_input.py

import posit_wrapper
from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1) Rutas relativas
here = os.path.dirname(os.path.abspath(__file__))
base = here
metrics_dir = os.path.join(base, 'sweep_metrics', 'noise_input')
images_dir  = os.path.join(base, 'sweep_plots',  'noise_input')
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(images_dir,  exist_ok=True)

# 2) Conversión y parámetros fijos
conv = lambda x: posit_wrapper.convert16(x)
V_rest      = -70 * mV
V_reset     = -60 * mV
V_threshold = -50 * mV
g_leak      = conv(1.0) * nS
tau         = 10 * ms
duration    = 200 * ms
dt          = 0.1 * ms
defaultclock.dt = dt

# 3) Definición de patrones de entrada
def white_noise(std_nA):
    # devuelve array en nA
    return np.random.normal(loc=1.5, scale=std_nA, size=int(duration/dt))
def sine_wave(freq, amp_nA):
    t = np.arange(0, float(duration/ms), float(dt/ms)) * ms
    return amp_nA * np.sin(2*np.pi*freq*(t/ms))

# 4) Rango de variables
noise_stds = [0.0, 0.2, 0.5, 1.0]   # nA
sine_amps  = [0.5, 1.0, 2.0]        # nA
sine_freqs = [5, 10, 20]           # Hz

records = []

# 5) Barrido ruido
for std in noise_stds:
    raw = white_noise(std)            # array de floats
    # vectorizamos convert16 elemento a elemento
    conv_vals = np.array([conv(float(x)) for x in raw]) * nA
    I_in = TimedArray(conv_vals, dt=dt)
    eqs = '''
    dv/dt = (-(V_rest - v) + I_in(t)/g_leak) / tau : volt
    '''
    G = NeuronGroup(1, eqs, threshold='v > V_threshold',
                    reset='v = V_reset', method='euler')
    G.v = V_rest
    spikes = SpikeMonitor(G)
    run(duration)
    records.append({'type':'noise', 'param':std, 'n_spikes':int(spikes.count[0])})

# 6) Barrido senoidal
for amp in sine_amps:
    for freq in sine_freqs:
        raw = sine_wave(freq, amp)    # array de floats en nA
        conv_vals = np.array([conv(float(x)) for x in raw]) * nA
        I_in = TimedArray(conv_vals, dt=dt)
        eqs = '''
        dv/dt = (-(V_rest - v) + I_in(t)/g_leak) / tau : volt
        '''
        G = NeuronGroup(1, eqs, threshold='v > V_threshold',
                        reset='v = V_reset', method='euler')
        G.v = V_rest
        spikes = SpikeMonitor(G)
        run(duration)
        records.append({'type':'sine', 'param':f'{amp}nA@{freq}Hz',
                        'n_spikes':int(spikes.count[0])})

# 7) Guardar CSV
df = pd.DataFrame(records)
csv_path = os.path.join(metrics_dir, 'metrics_noise_input.csv')
df.to_csv(csv_path, index=False)

# 8) Graficar resultados
# Ruido
noise_df = df[df.type=='noise']
plt.figure()
plt.plot(noise_df.param, noise_df.n_spikes, 'o-')
plt.xlabel('Std ruido (nA)')
plt.ylabel('n_spikes')
plt.title('Robustez vs ruido')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'noise_vs_spikes.png'))
plt.close()

# Senoidal (un plot por amplitud)
for amp in sine_amps:
    sub = df[(df.type=='sine') & (df.param.str.startswith(f'{amp}nA'))]
    freqs = [int(p.split('@')[1].replace('Hz','')) for p in sub.param]
    plt.figure()
    plt.plot(freqs, sub.n_spikes, 'o-')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('n_spikes')
    plt.title(f'Senoidal amp={amp}nA')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'sine_{amp}nA_vs_freq.png'))
    plt.close()
