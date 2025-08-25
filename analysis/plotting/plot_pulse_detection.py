import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los resultados del experimento de pulsos
df = pd.read_csv("metrics_compare/pulse_detection_compare.csv")

# Crear figura
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Respuesta al estímulo pulsado (50–55 ms)", fontsize=16)

# Gráfico 1: Boxplot de spikes durante el pulso
sns.boxplot(data=df, x='label', y='pulse_spikes', ax=axs[0], palette='Set2')
axs[0].set_title("Spikes durante el pulso (50–55 ms)")
axs[0].set_ylabel("Número de spikes")

# Gráfico 2: Histograma de diferencias
pivot = df.pivot(index='neuron', columns='label', values='pulse_spikes')
pivot['delta'] = pivot['float16'] - pivot['posit16']
sns.histplot(pivot['delta'], bins=20, kde=True, ax=axs[1], color='mediumslateblue')
axs[1].axvline(0, color='black', linestyle='--')
axs[1].set_title("Diferencia de spikes (float16 - posit16)")
axs[1].set_xlabel("delta_spikes durante el pulso")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("images_compare/pulse_detection_summary.png")
plt.show()
