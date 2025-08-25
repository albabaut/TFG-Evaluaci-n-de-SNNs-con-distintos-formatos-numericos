import posit_wrapper
from brian2 import *  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error


def run_simulation(data_type, title, filename, csvfile):
    """
    Ejecuta la simulación con el tipo de dato especificado.
    :param data_type: 'posit', 'float32' o 'float64'
    :param title: Título del gráfico
    :param filename: Nombre del archivo de salida
    :param csvfile: Nombre del archivo CSV donde se guardarán los datos
    """
    if data_type == "posit":
        conversion_func = lambda x: posit_wrapper.convert(x)
    elif data_type == "float64":
        conversion_func = lambda x: np.float64(x)
    else:
        conversion_func = lambda x: np.float32(x)
    
    tau = 10 * ms  
    V_rest = -70 * mV  
    V_reset = -60 * mV  
    V_threshold = -50 * mV  
    g_leak = 0.01 * nS  # Conductancia de fuga en nanoSiemens
    
    I_value = conversion_func(1e-9)  # Convertir al tipo de dato correspondiente
    I_input = I_value * nA  # Convertir a nanoamperios (A)
    
    print(f"\nCorriente convertida a {data_type}: {I_input}")
    
    eqs = '''
    dv/dt = (-(V_rest - v) + I / g_leak) / tau : volt
    I : amp
    '''
    
    neuron = NeuronGroup(1, eqs, threshold='v>V_threshold', reset='v = V_reset', method='euler')
    neuron.v = V_rest
    neuron.I = I_input
    

    monitor = StateMonitor(neuron, 'v', record=True)
    spikemon = SpikeMonitor(neuron)

    net = Network(neuron, monitor, spikemon)
    net.run(500 * ms)

    print(f"Cantidad de spikes ({data_type}): {spikemon.num_spikes}")

    df = pd.DataFrame({
        "Tiempo (ms)": monitor.t / ms,
        "Voltaje (mV)": monitor.v[0] / mV
    })
    df.to_csv(csvfile, index=False)
    print(f"Datos guardados en {csvfile}")


"""
def calcular_metricas(ref_csv, test_csv, label):
   
    df_ref = pd.read_csv(ref_csv)
    df_test = pd.read_csv(test_csv)

    min_len = min(len(df_ref), len(df_test))
    v_ref = df_ref["Voltaje (mV)"][:min_len].values
    v_test = df_test["Voltaje (mV)"][:min_len].values

    rmse_val = np.sqrt(mean_squared_error(v_ref, v_test))
    psnr_val = psnr(v_ref, v_test, data_range=v_ref.max() - v_ref.min())
    ssim_val = ssim(v_ref, v_test, data_range=v_ref.max() - v_ref.min())

    print(f"\nMétricas para {label}:")
    print(f"RMSE: {rmse_val:.6f}")
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    """


# Ejecutar simulaciones
run_simulation("posit", "Simulación con Posit", "sim_posit.png", "csv_posit.csv")
run_simulation("float32", "Simulación con Float32", "sim_float32.png", "csv_float.csv")
run_simulation("float64", "Simulación con Float64", "sim_float64.png", "csv_float64.csv")

import matplotlib.pyplot as plt

# Diccionario para guardar las métricas
metricas_dict = {
    "Formato": [],
    "RMSE": [],
    "PSNR": [],
    "SSIM": []
}

# Función modificada para guardar resultados
def calcular_metricas_y_guardar(ref_csv, test_csv, label, nombre_formato):
    df_ref = pd.read_csv(ref_csv)
    df_test = pd.read_csv(test_csv)

    min_len = min(len(df_ref), len(df_test))
    v_ref = df_ref["Voltaje (mV)"][:min_len].values
    v_test = df_test["Voltaje (mV)"][:min_len].values

    rmse_val = np.sqrt(mean_squared_error(v_ref, v_test))
    psnr_val = psnr(v_ref, v_test, data_range=v_ref.max() - v_ref.min())
    ssim_val = ssim(v_ref, v_test, data_range=v_ref.max() - v_ref.min())

    print(f"\nMétricas para {label}:")
    print(f"RMSE: {rmse_val:.6f}")
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")

    # Guardar en el diccionario
    metricas_dict["Formato"].append(nombre_formato)
    metricas_dict["RMSE"].append(rmse_val)
    metricas_dict["PSNR"].append(psnr_val)
    metricas_dict["SSIM"].append(ssim_val)

# Calcular métricas y guardar
calcular_metricas_y_guardar("csv_float64.csv", "csv_posit.csv", "Posit vs Float64", "Posit")
calcular_metricas_y_guardar("csv_float64.csv", "csv_float.csv", "Float32 vs Float64", "Float32")

# Convertir a DataFrame para graficar
df_metricas = pd.DataFrame(metricas_dict)

# Gráfica de barras
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
metricas = ["RMSE", "PSNR", "SSIM"]

for i, metrica in enumerate(metricas):
    axs[i].bar(df_metricas["Formato"], df_metricas[metrica])
    axs[i].set_title(metrica)
    axs[i].set_ylabel(metrica)
    axs[i].set_xlabel("Formato")
    axs[i].grid(True)

plt.suptitle("Comparación de métricas respecto a Float64")
plt.tight_layout()
plt.savefig("metricas_comparacion.png")
print("Gráfico guardado como 'metricas_comparacion.png'")

