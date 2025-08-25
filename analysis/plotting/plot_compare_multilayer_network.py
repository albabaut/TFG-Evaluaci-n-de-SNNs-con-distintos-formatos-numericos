import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar resultados y asegurar tipo numérico
# Cargar CSV y asegurar tipos
df = pd.read_csv("metrics_compare/multilayer_compare_float64.csv")
df['output_spikes'] = pd.to_numeric(df['output_spikes'], errors='coerce')
df['neuron'] = pd.to_numeric(df['neuron'], errors='coerce')
df = df.dropna(subset=['neuron', 'output_spikes'])
df['neuron'] = df['neuron'].astype(int)


# Separar por formato
df_ref = df[df['label'] == 'float64'].set_index('neuron')
df_f16 = df[df['label'] == 'float16'].set_index('neuron')
df_p16 = df[df['label'] == 'posit16'].set_index('neuron')

# Calcular errores absolutos respecto a float64
df_compare = pd.DataFrame({
    'neuron': df_ref.index,
    'error_float16': (df_f16['output_spikes'] - df_ref['output_spikes']).abs(),
    'error_posit16': (df_p16['output_spikes'] - df_ref['output_spikes']).abs(),
})

# Preparar datos para boxplot
df_err = pd.melt(df_compare, id_vars='neuron',
                 value_vars=['error_float16', 'error_posit16'],
                 var_name='formato', value_name='error_abs')
df_err['formato'] = df_err['formato'].str.replace('error_', '')
df_err = df_err.dropna(subset=['error_abs'])  # eliminar NaNs

# Crear figura
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Comparación con float64 en Red Multicapa", fontsize=16)

# Gráfico 1: Boxplot de errores
sns.boxplot(data=df_err, x='formato', y='error_abs', ax=axs[0], palette='pastel')
axs[0].set_title("Error absoluto respecto a float64")
axs[0].set_ylabel("Diferencia absoluta de spikes")

# Gráfico 2: Scatter por neurona
sns.scatterplot(x=df_ref['output_spikes'], y=df_f16['output_spikes'], ax=axs[1], label='float16')
sns.scatterplot(x=df_ref['output_spikes'], y=df_p16['output_spikes'], ax=axs[1], label='posit16')
axs[1].plot([0, 60], [0, 60], 'k--')
axs[1].set_xlabel("Spikes en float64")
axs[1].set_ylabel("Spikes en formato reducido")
axs[1].set_title("Spikes por neurona vs float64")
axs[1].legend()

# Gráfico 3: Histograma de errores
sns.histplot(df_compare['error_float16'], color='skyblue', label='float16', kde=True, ax=axs[2])
sns.histplot(df_compare['error_posit16'], color='coral', label='posit16', kde=True, ax=axs[2])
axs[2].set_title("Distribución del error absoluto")
axs[2].set_xlabel("Error absoluto de spikes")
axs[2].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("images_compare/multilayer_vs_float64_summary.png")
plt.show()
