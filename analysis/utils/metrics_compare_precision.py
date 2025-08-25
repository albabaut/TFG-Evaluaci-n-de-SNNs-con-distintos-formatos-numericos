import posit_wrapper
from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# Modelo de neurona
tau = 10 * ms
V_rest = -70 * mV
V_reset = -60 * mV
V_threshold = -50 * mV
g_leak = 0.01 * nS
eqs = '''
dv/dt = (-(V_rest - v) + I) / tau : volt
I = (I_base + I_base * 0.000001 * sin(2 * pi * t / ms)) * amp : amp
I_base : 1
'''


# Valores de corriente para probar (muy cercanos entre sí)
I_values = [1.0 + i * 1e-7 for i in range(10)]  # 1.0, 1.0000001, ..., 1.0000009

# Resultados
resultados = {
    "I": [],
    "RMSE_Posit": [],
    "RMSE_Float32": [],
    "PSNR_Posit": [],
    "PSNR_Float32": [],
    "SSIM_Posit": [],
    "SSIM_Float32": []
}

def run_sim(tipo, valor):
    if tipo == "posit":
        conv = lambda x: posit_wrapper.convert(x)
    elif tipo == "float64":
        conv = lambda x: np.float64(x)
    else:
        conv = lambda x: np.float32(x)

    I_input = conv(valor) * amp
    neuron = NeuronGroup(1, eqs, threshold='v>V_threshold', reset='v = V_reset', method='euler')
    #neuron.I = I_input
    neuron.v = V_rest
    neuron.I_base = conv(I) 
    monitor = StateMonitor(neuron, 'v', record=True)
    net = Network(neuron, monitor)
    net.run(500 * ms)
    return monitor.v[0] / mV

def calcular_metricas(ref, test):
    rmse = np.sqrt(mean_squared_error(ref, test))
    psnr_val = psnr(ref, test, data_range=ref.max() - ref.min())
    ssim_val = ssim(ref, test, data_range=ref.max() - ref.min())
    return rmse, psnr_val, ssim_val

for I in I_values:
    v64 = run_sim("float64", I)
    v32 = run_sim("float32", I)
    vp = run_sim("posit", I)

    min_len = min(len(v64), len(v32), len(vp))
    v64, v32, vp = v64[:min_len], v32[:min_len], vp[:min_len]

    rmse_p, psnr_p, ssim_p = calcular_metricas(v64, vp)
    rmse_f, psnr_f, ssim_f = calcular_metricas(v64, v32)

    resultados["I"].append(I)
    resultados["RMSE_Posit"].append(rmse_p)
    resultados["RMSE_Float32"].append(rmse_f)
    resultados["PSNR_Posit"].append(psnr_p)
    resultados["PSNR_Float32"].append(psnr_f)
    resultados["SSIM_Posit"].append(ssim_p)
    resultados["SSIM_Float32"].append(ssim_f)

df = pd.DataFrame(resultados)

# Graficar cada métrica
for metrica in ["RMSE", "PSNR", "SSIM"]:
    plt.figure()
    plt.plot(df["I"], df[f"{metrica}_Posit"], label="Posit<32,2>", marker='o')
    plt.plot(df["I"], df[f"{metrica}_Float32"], label="Float32", marker='x')
    plt.xlabel("Corriente (A)")
    plt.ylabel(metrica)
    plt.title(f"{metrica} vs Corriente (respecto a Float64)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{metrica.lower()}_vs_corriente.png")
    print(f"Guardado: {metrica.lower()}_vs_corriente.png")
