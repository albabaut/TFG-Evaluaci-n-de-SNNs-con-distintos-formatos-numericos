import posit_wrapper
from brian2 import *
import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim



# ---- Simulación para cualquier tipo de dato y corriente específica ----
def run_simulation(data_type, I_override=None):
    if data_type == "posit":
        conv = lambda x: posit_wrapper.convert16(x)
    elif data_type == "float16":
        conv = lambda x: np.float16(x)
    elif data_type == "float32":
        conv = lambda x: np.float32(x)
    else:
        conv = lambda x: np.float64(x)

    tau = 10 * ms
    V_rest = -70 *mV
    V_reset = -60 *mV
    V_threshold = -50 *mV 
    g_leak = 1*nS
    I_input = conv(I_override) *nA # <-- ahora usamos nA
  

    eqs = '''
    dv/dt = (-(V_rest - v) + I/g_leak ) / tau : volt
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
print("\n🔍 Sweep de corrientes (en nA) para evaluar Posit<16,2> vs float16 respecto a float64:")

# Valores de I en la golden zone, directamente sin unidades
currents = np.linspace(0.25, 2, 50)
results = []

results = []

for I_val in currents:
    # Ejecutar simulaciones
    spikes_ref, v_ref, _ = run_simulation("float64", I_override=I_val)
    spikes_p,   v_p,   _ = run_simulation("posit",   I_override=I_val)
    spikes_f16, v_f16, _ = run_simulation("float16", I_override=I_val)
    spikes_f32, v_f32, _ = run_simulation("float32", I_override=I_val)

    min_len = min(len(v_ref), len(v_p), len(v_f16), len(v_f32))
    v_ref_clip = v_ref[:min_len]
    v_p_clip   = v_p[:min_len]
    v_f16_clip = v_f16[:min_len]
    v_f32_clip = v_f32[:min_len]

    # Métricas RMSE
    rmse_p    = np.sqrt(mean_squared_error(v_ref_clip, v_p_clip))
    rmse_f16  = np.sqrt(mean_squared_error(v_ref_clip, v_f16_clip))
    rmse_f32  = np.sqrt(mean_squared_error(v_ref_clip, v_f32_clip))
    rmse_f64  = 0.0  # comparación consigo mismo

    # PSNR
    range_v = v_ref_clip.max() - v_ref_clip.min() if v_ref_clip.max() != v_ref_clip.min() else 1.0
    psnr_p    = psnr(v_ref_clip, v_p_clip, data_range=range_v)
    psnr_f16  = psnr(v_ref_clip, v_f16_clip, data_range=range_v)
    psnr_f32  = psnr(v_ref_clip, v_f32_clip, data_range=range_v)

    # SSIM
    v_norm = lambda x: (x - v_ref_clip.min()) / range_v
    ssim_p    = ssim(v_norm(v_ref_clip), v_norm(v_p_clip), data_range=1.0)
    ssim_f16  = ssim(v_norm(v_ref_clip), v_norm(v_f16_clip), data_range=1.0)
    ssim_f32  = ssim(v_norm(v_ref_clip), v_norm(v_f32_clip), data_range=1.0)

    results.append({
        'I': I_val,
        'rmse_p': rmse_p,
        'rmse_f16': rmse_f16,
        'rmse_f32': rmse_f32,
        'rmse_f64': rmse_f64,
        'psnr_p': psnr_p,
        'psnr_f16': psnr_f16,
        'psnr_f32': psnr_f32,
        'ssim_p': ssim_p,
        'ssim_f16': ssim_f16,
        'ssim_f32': ssim_f32
    })

print(f"\n{'I (nA)':>8} | {'RMSE f64':>26} | {'RMSE f32':>26} | {'RMSE f16':>26} | {'RMSE posit':>26} | {'Posit vs Float16'}")
print("-" * 140)
for r in results:
    closer_to_f64 = "Posit ✅" if r['rmse_p'] <= r['rmse_f16'] else "float16 ✅"

    print(f"{r['I']:8.2f} | "
          f"{r['rmse_f64']:26.20e} | "
          f"{r['rmse_f32']:26.20e} | "
          f"{r['rmse_f16']:26.20e} | "
          f"{r['rmse_p']:26.20e} | "
          f"{closer_to_f64}")


# --------------------------------------------
# 📈 Gráfico RMSE vs I
# --------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd

# Extraer arrays para graficar
I_vals = [r['I'] for r in results]
rmse_f16_vals = [r['rmse_f16'] for r in results]
rmse_f32_vals = [r['rmse_f32'] for r in results]
rmse_p_vals   = [r['rmse_p'] for r in results]

# Crear gráfico
plt.figure(figsize=(10, 6))
plt.plot(I_vals, rmse_f16_vals, label='float16', linewidth=2)
plt.plot(I_vals, rmse_f32_vals, label='float32', linewidth=2)
plt.plot(I_vals, rmse_p_vals,   label='posit<16,2>', linewidth=2)
plt.xlabel("Corriente de entrada I (nA)")
plt.ylabel("RMSE respecto a float64 (voltaje)")
plt.title("RMSE vs Corriente para diferentes tipos de dato")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar gráfico como imagen
plt.savefig("rmse_vs_I_golden.png", dpi=300)
