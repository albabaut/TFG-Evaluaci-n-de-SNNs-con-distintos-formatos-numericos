import numpy as np
import matplotlib.pyplot as plt
from posit_wrapper import convert16

# Rango de voltajes a analizar (en mV)
v_min = -100
v_max = 100
n_points = 10000
voltajes = np.linspace(v_min, v_max, n_points)

err_f16 = np.abs(np.float16(voltajes) - voltajes)
err_posit = np.abs(np.array([convert16(v) for v in voltajes]) - voltajes)

# Dónde posit16 es mejor
posit_mejor = err_posit < err_f16
igual = err_posit == err_f16
float16_mejor = err_f16 < err_posit

# Resumen de rangos
if np.any(posit_mejor):
    indices = np.where(posit_mejor)[0]
    print("Rangos donde posit16 es mejor que float16:")
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1] + 1:
            print(f"  {voltajes[start]:.4f} mV a {voltajes[indices[i-1]]:.4f} mV")
            start = indices[i]
    print(f"  {voltajes[start]:.4f} mV a {voltajes[indices[-1]]:.4f} mV")
else:
    print("No hay zonas donde posit16 sea mejor que float16 en este rango.")

# Gráfica de errores
plt.figure(figsize=(10,6))
plt.plot(voltajes, err_f16, label='Error float16', color='blue')
plt.plot(voltajes, err_posit, label='Error posit16', color='red')
plt.fill_between(voltajes, 0, np.maximum(err_f16, err_posit), where=posit_mejor, color='green', alpha=0.3, label='posit16 mejor')
plt.fill_between(voltajes, 0, np.maximum(err_f16, err_posit), where=float16_mejor, color='orange', alpha=0.2, label='float16 mejor')
plt.xlabel('Voltaje (mV)')
plt.ylabel('Error absoluto de cuantización (mV)')
plt.title('Resolución de cuantización: float16 vs posit16')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/analyze_utils/quantization_resolution_compare.png', dpi=150, bbox_inches='tight')
plt.show() 