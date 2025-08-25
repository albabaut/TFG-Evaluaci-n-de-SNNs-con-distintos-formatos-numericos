import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los resultados
df = pd.read_csv("metrics_compare/noise_individual_inputs.csv")

# Crear figura
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Entrada ruidosa independiente por neurona", fontsize=16)

# Gráfico 1: Histograma de total_spikes
sns.histplot(data=df, x='total_spikes', hue='label', bins=30, ax=axs[0], kde=True, element="step")
axs[0].set_title("Distribución del número total de spikes")
axs[0].set_xlabel("Total de spikes")

# Gráfico 2: Boxplot de entropía de voltaje
sns.boxplot(data=df, x='label', y='voltage_entropy', ax=axs[1], palette='pastel')
axs[1].set_title("Entropía del voltaje por formato")
axs[1].set_ylabel("Entropía")

# Gráfico 3: Scatter comparando posit16 vs float16 (total spikes)
pivot = df.pivot(index='neuron', columns='label', values='total_spikes')
sns.scatterplot(x=pivot['float16'], y=pivot['posit16'], ax=axs[2])
axs[2].plot([pivot.min().min(), pivot.max().max()],
            [pivot.min().min(), pivot.max().max()], 'k--')
axs[2].set_title("Spikes: float16 vs posit16 por neurona")
axs[2].set_xlabel("Spikes (float16)")
axs[2].set_ylabel("Spikes (posit16)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("images_compare/noise_individual_inputs_summary.png")
plt.show()
