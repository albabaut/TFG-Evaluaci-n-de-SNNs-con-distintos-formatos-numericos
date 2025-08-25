import posit_wrapper
from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# Parámetros del modelo
tau = 10 * ms
V_rest = -70 * mV
V_reset = -60 * mV
V_threshold = -50 * mV
g_leak = 0.01 * nS

# Ecuaciones con corriente compartida
eqs = '''
dv/dt = (-(V_rest - v) + I/ g_leak) / tau : volt
I = I_base + I_base * 1e-2 * sin(2 * pi * t / ms) : amp
I_base : amp (shared)
'''

def run_simulacion(tipo, I_val):
    if tipo == "posit":
        conv = lambda x: posit_wrapper.convert(x)
    elif tipo == "float64":
        conv = lambda x: np.float64(x)
    else:
        conv = lambda x: np.float16(x)

    I_input = conv(I_val) * amp

    neuron = NeuronGroup(1, eqs, threshold='v>V_threshold', reset='v = V_reset', method='euler')
    neuron.v = V_rest
    neuron.I_base = I_input


    monitor = StateMonitor(neuron, 'v', record=True)
    net = Network(neuron, monitor)
    net.run(500 * ms)

    return monitor.v[0] / mV

def calcular_metricas(ref, test):
    rmse = np.sqrt(mean_squared_error(ref, test))
    psnr_val = psnr(ref, test, data_range=ref.max() - ref.min())
    ssim_val = ssim(ref, test, data_range=ref.max() - ref.min())
    return rmse, psnr_val, ssim_val

# 🔁 Valores de corriente para escanear
# Valores de corriente pequeños, donde float32 empieza a degradar
I_values = [1e-4 + i * 1e-5 for i in range(10)]  # De 0.0001 a 0.00019 A





# Resultados
resultados = {
    "I": [],
    "RMSE_Posit": [],
    "RMSE_Float16": [],
    "PSNR_Posit": [],
    "PSNR_Float16": [],
    "SSIM_Posit": [],
    "SSIM_Float16": []
}

for I_val in I_values:
    v64 = run_simulacion("float64", I_val)
    v16 = run_simulacion("float16", I_val)
    vp = run_simulacion("posit", I_val)

    min_len = min(len(v64), len(v16), len(vp))
    v64, v16, vp = v64[:min_len], v16[:min_len], vp[:min_len]

    rmse_p, psnr_p, ssim_p = calcular_metricas(v64, vp)
    rmse_f, psnr_f, ssim_f = calcular_metricas(v64, v16)

    resultados["I"].append(I_val)
    resultados["RMSE_Posit"].append(rmse_p)
    resultados["RMSE_Float16"].append(rmse_f)
    resultados["PSNR_Posit"].append(psnr_p)
    resultados["PSNR_Float16"].append(psnr_f)
    resultados["SSIM_Posit"].append(ssim_p)
    resultados["SSIM_Float16"].append(ssim_f)

    print(f"I = {I_val:.8f} → RMSE_Posit: {rmse_p:.2e}, RMSE_Float16: {rmse_f:.2e}")

# 📊 Graficar resultados
df = pd.DataFrame(resultados)

for metrica in ["RMSE", "PSNR", "SSIM"]:
    plt.figure()
    plt.plot(df["I"], df[f"{metrica}_Posit"], label="Posit<16,1>", marker='o')
    plt.plot(df["I"], df[f"{metrica}_Float16"], label="Float16", marker='x')
    plt.xlabel("Corriente (A)")
    plt.ylabel(metrica)
    plt.title(f"{metrica} vs Corriente (respecto a Float64)")
    plt.legend()
    plt.grid(True)
    plt.xscale("log")
    plt.savefig(f"{metrica.lower()}_vs_corriente.png")
    print(f"📈 Guardado: {metrica.lower()}_vs_corriente.png")
