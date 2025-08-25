import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los resultados desde CSV
df = pd.read_csv("metrics_compare/compare_to_float64.csv")

# Crear figura 2x2
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Comparación float16 vs posit16 respecto a float64", fontsize=16)

# Gráfico 1: Boxplot RMSE
sns.boxplot(data=df, x='label', y='rmse', ax=axs[0, 0], palette='pastel')
axs[0, 0].set_title("RMSE del voltaje respecto a float64")
axs[0, 0].set_ylabel("RMSE (mV)")

# Gráfico 2: Boxplot F1-score
sns.boxplot(data=df, x='label', y='f1_score', ax=axs[0, 1], palette='Set2')
axs[0, 1].set_title("F1-score respecto a float64")
axs[0, 1].set_ylabel("F1-score")

# Gráfico 3: Scatter RMSE vs F1-score
sns.scatterplot(data=df, x='rmse', y='f1_score', hue='label', ax=axs[1, 0], alpha=0.7)
axs[1, 0].set_title("Relación entre RMSE y F1-score")
axs[1, 0].set_xlabel("RMSE")
axs[1, 0].set_ylabel("F1-score")

# Gráfico 4: Histograma de delta_spikes
sns.histplot(data=df, x='delta_spikes', hue='label', kde=True, bins=30, ax=axs[1, 1], multiple='stack')
axs[1, 1].set_title("Distribución de diferencia de spikes frente a float64")
axs[1, 1].set_xlabel("delta_spikes")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("images_compare/compare_float64_summary.png")
plt.show()
