import posit_wrapper
from brian2 import *
import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd

# ---- Simulación para cualquier tipo de dato y corriente específica ----
def run_simulation(data_type, I_override=None):
    if data_type == "posit24":
        conv = lambda x: posit_wrapper.convert24(x)
    elif data_type == "posit20":
        conv = lambda x: posit_wrapper.convert20(x)
    elif data_type == "float32":
        conv = lambda x: np.float32(x)
    elif data_type == "float16":
        conv = lambda x: np.float16(x)
    else:
        conv = lambda x: np.float64(x)

    tau = 10 * ms
    V_rest = -70 * mV
    V_reset = -60 * mV
    V_threshold = -50 * mV
    g_leak = 1 * nS
    I_input = conv(I_override) * nA

    eqs = '''
    dv/dt = (-(V_rest - v) + I/g_leak) / tau : volt
    I : amp
    '''

    neuron = NeuronGroup(1, eqs, threshold='v > V_threshold', reset='v = V_reset', method='euler')
    neuron.v = V_rest
    neuron.I = I_input

    monitor = StateMonitor(neuron, 'v', record=True)
    spikemon = SpikeMonitor(neuron)

    net = Network(neuron, monitor, spikemon)
    net.run(100 * ms)

    return spikemon.num_spikes, monitor.v[0] / mV, spikemon.t[:] / ms

# ---- Barrido de corrientes (en nanoamperios) ----
print("\n🔍 Sweep de corrientes (en nA) para evaluar distintos formatos respecto a float64:")

currents = np.linspace(0.5, 1, 500)
results = []

for I_val in currents:
    # Ejecutar simulaciones
    spikes_ref, v_ref, _ = run_simulation("float64", I_override=I_val)
    spikes_p24, v_p24, _ = run_simulation("posit24", I_override=I_val)
    spikes_p20, v_p20, _ = run_simulation("posit20", I_override=I_val)
    spikes_f32, v_f32, _ = run_simulation("float32", I_override=I_val)

    min_len = min(len(v_ref), len(v_p24), len(v_p20), len(v_f32))
    v_ref_clip = v_ref[:min_len]
    v_p24_clip = v_p24[:min_len]
    v_p20_clip = v_p20[:min_len]
    v_f32_clip = v_f32[:min_len]

    # Métricas
    rmse_p24 = np.sqrt(mean_squared_error(v_ref_clip, v_p24_clip))
    rmse_p20 = np.sqrt(mean_squared_error(v_ref_clip, v_p20_clip))
    rmse_f32 = np.sqrt(mean_squared_error(v_ref_clip, v_f32_clip))
    rmse_f64 = 0.0

    range_v = v_ref_clip.max() - v_ref_clip.min() if v_ref_clip.max() != v_ref_clip.min() else 1.0
    v_norm = lambda x: (x - v_ref_clip.min()) / range_v

    results.append({
        'I': I_val,
        'rmse_f64': rmse_f64,
        'rmse_f32': rmse_f32,
        'rmse_posit24': rmse_p24,
        'rmse_posit20': rmse_p20,
        'psnr_f32': psnr(v_ref_clip, v_f32_clip, data_range=range_v),
        'psnr_posit24': psnr(v_ref_clip, v_p24_clip, data_range=range_v),
        'psnr_posit20': psnr(v_ref_clip, v_p20_clip, data_range=range_v),
        'ssim_f32': ssim(v_norm(v_ref_clip), v_norm(v_f32_clip), data_range=1.0),
        'ssim_posit24': ssim(v_norm(v_ref_clip), v_norm(v_p24_clip), data_range=1.0),
        'ssim_posit20': ssim(v_norm(v_ref_clip), v_norm(v_p20_clip), data_range=1.0),
    })

# Imprimir tabla resumen
print(f"\n{'I (nA)':>8} | {'RMSE f64':>26} | {'RMSE f32':>26} | {'RMSE posit24':>26} | {'RMSE posit20':>26}")
print("-" * 120)
for r in results:
    print(f"{r['I']:8.2f} | "
          f"{r['rmse_f64']:26.20e} | "
          f"{r['rmse_f32']:26.20e} | "
          f"{r['rmse_posit24']:26.20e} | "
          f"{r['rmse_posit20']:26.20e}")

# ---- Gráfico RMSE vs I ----
I_vals = [r['I'] for r in results]
rmse_f32_vals = [r['rmse_f32'] for r in results]
rmse_posit24_vals = [r['rmse_posit24'] for r in results]
rmse_posit20_vals = [r['rmse_posit20'] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(I_vals, rmse_f32_vals, label='float32', linewidth=2)
plt.plot(I_vals, rmse_posit24_vals, label='posit<24,2>', linewidth=2)
plt.plot(I_vals, rmse_posit20_vals, label='posit<20,2>', linewidth=2)

plt.xlabel("Corriente de entrada I (nA)")
plt.ylabel("RMSE respecto a float64 (voltaje)")
plt.title("RMSE vs Corriente para diferentes tipos de dato")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rmse_vs_I_p24_p20.png", dpi=300)
plt.show()