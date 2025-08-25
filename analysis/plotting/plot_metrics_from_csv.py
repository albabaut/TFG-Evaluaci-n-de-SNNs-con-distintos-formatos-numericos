import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar CSV
df = pd.read_csv("metrics_compare/neuron_metrics.csv")

# Crear figura con subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Análisis comparativo float16 vs posit16", fontsize=16)

# Histograma de delta_spikes
sns.histplot(df['delta_spikes'], kde=True, bins=30, ax=axs[0, 0], color='royalblue')
axs[0, 0].set_title("Distribución de diferencia de spikes")
axs[0, 0].set_xlabel("delta_spikes (float16 - posit16)")
axs[0, 0].axvline(0, color='black', linestyle='--')

# Boxplot de RMSE y error acumulado
sns.boxplot(data=df[['rmse', 'accumulated_error']], ax=axs[0, 1], palette="Set2")
axs[0, 1].set_title("Boxplot: RMSE y Error acumulado")
axs[0, 1].set_ylabel("Valor")

# Dispersión RMSE vs delta_spikes
axs[1, 0].scatter(df['rmse'], df['delta_spikes'], alpha=0.7, color='darkgreen')
axs[1, 0].set_xlabel("RMSE voltaje (mV)")
axs[1, 0].set_ylabel("delta_spikes")
axs[1, 0].set_title("Relación entre RMSE y diferencia de spikes")
axs[1, 0].axhline(0, color='gray', linestyle=':')

# Boxplot F1-score
sns.boxplot(y=df['f1_score'], ax=axs[1, 1], color='mediumpurple')
axs[1, 1].set_title("Distribución de F1-score")
axs[1, 1].set_ylabel("F1-score")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("images_compare/summary_metrics_overview.png")
plt.show()
