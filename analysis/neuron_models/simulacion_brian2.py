# simulacion_brian2.py

import os
import sys

# 0) Para que Python encuentre posit_wrapper:
here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(here, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import posit_wrapper

# 1) Parámetros de simulación
dt = 0.1    # ms
T  = 200    # ms
time = np.arange(0, T + dt, dt)
nt   = len(time)

# 2) Parámetros de la neurona
V_rest      = -70.0    # mV
V_reset     = -60.0    # mV
V_threshold = -50.0    # mV
tau         = 10.0     # ms
g_leak      = 1.0      # nS
I_ext       = 1.01     # nA (ligeramente por encima del umbral)

# 3) Simulación en float64
def simulate_float64():
    v = np.full(nt, V_rest, dtype=np.float64)
    spikes = np.zeros(nt, dtype=bool)
    for t in range(nt - 1):
        dv = (-(V_rest - v[t]) + I_ext/g_leak)*(dt/tau)
        v_next = v[t] + dv
        if v_next > V_threshold:
            v[t+1]   = V_reset
            spikes[t+1] = True
        else:
            v[t+1] = v_next
    return v, spikes

# 4) Simulación en float16 puro
def simulate_float16():
    v = np.full(nt, V_rest, dtype=np.float16)
    spikes = np.zeros(nt, dtype=bool)
    for t in range(nt - 1):
        dv = np.float16((-(V_rest - v[t]) + np.float16(I_ext/g_leak)) * np.float16(dt/tau))
        v_next = np.float16(v[t] + dv)
        if v_next > V_threshold:
            v[t+1]     = V_reset
            spikes[t+1] = True
        else:
            v[t+1] = v_next
    # Convertimos a float64 para las métricas
    return v.astype(np.float64), spikes

# 5) Simulación en Posit16 puro (estado guardado como posit)
def simulate_posit16():
    v = [posit_wrapper.convert16(V_rest)]
    spikes = np.zeros(nt, dtype=bool)
    for t in range(nt - 1):
        dv_raw = (-(V_rest - float(v[t])) + I_ext/g_leak)*(dt/tau)
        dv_p   = posit_wrapper.convert16(dv_raw)
        v_next_raw = float(v[t]) + dv_p
        v_next  = posit_wrapper.convert16(v_next_raw)
        if v_next > V_threshold:
            v.append(posit_wrapper.convert16(V_reset))
            spikes[t+1] = True
        else:
            v.append(v_next)
    # Convertimos lista de posits a float64 para comparar
    v_float = np.array([float(x) for x in v], dtype=np.float64)
    return v_float, spikes

# 6) Ejecutar las tres simulaciones
v64, s64    = simulate_float64()
v16, s16    = simulate_float16()
vp16, sp16  = simulate_posit16()

# 7) Calcular métricas
metrics = []
for label, volt, spikes in [
    ('float64', v64, s64),
    ('float16', v16, s16),
    ('posit16', vp16, sp16)
]:
    total_spikes = int(spikes.sum())
    rmse = np.sqrt(np.mean((volt - v64)**2))
    metrics.append({'dtype':label,
                    'total_spikes':total_spikes,
                    'rmse_vs_float64':rmse})

df = pd.DataFrame(metrics)
print(df)

# 8) Guarda la gráfica comparativa
plt.figure(figsize=(10,5))
plt.plot(time, v64,   label='float64', linewidth=2)
plt.plot(time, v16,   label='float16', linestyle='--')
plt.plot(time, vp16,  label='posit16', linestyle=':')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('Voltage trace: float16 & posit16 vs float64')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(here, 'comparison_voltage_trace.png'))
